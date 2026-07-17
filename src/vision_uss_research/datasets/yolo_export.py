"""Merge label runs into a YOLO detection dataset with sequence-disjoint splits.

Split policy:
- Splits are by SEQUENCE - frames from one approach are near-duplicates and must
  never straddle splits.
- The test split prefers human-clicked sequences ("gold"): for classes that have
  clicked sequences, a fraction of those is reserved for test and never trained
  on. Classes labeled only by the automatic pipeline fall back to auto-labeled
  test sequences (marked gold=False in the provenance).
"""

from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd

from vision_uss_research.labeling.boxes_io import load_boxes_tolerant, resolve_frame_path
from vision_uss_research.runs import git_sha


def merge_label_sources(sources: list[dict]) -> pd.DataFrame:
    """sources: [{'name', 'boxes_csv', 'frames_root', 'gold': bool}, ...].
    Returns the concatenated canonical rows with 'source' and 'gold' columns.
    When the same sequence/camera/frame appears in multiple sources (e.g. a
    sequence re-labeled in a newer run), the source that sorts LAST by name
    wins - run ids are date-prefixed, so newer runs override older ones."""
    parts = []
    for source in sorted(sources, key=lambda s: str(s["name"])):
        df = load_boxes_tolerant(Path(source["boxes_csv"]))
        df["source"] = source["name"]
        df["gold"] = bool(source.get("gold", False))
        parts.append(df)
    merged = pd.concat(parts, ignore_index=True)
    merged = merged.drop_duplicates(subset=["sequence_id", "camera", "frame_idx"],
                                    keep="last")
    return merged[merged["x0"].notna()].reset_index(drop=True)


def apply_class_groups(df: pd.DataFrame, class_groups: dict[str, str] | None
                       ) -> pd.DataFrame:
    """Add a 'class_name' column: the training class for each row. Objects named
    in class_groups map to their bucket (e.g. cone -> high_other); everything
    else keeps its own name. Splitting still stratifies by target_object, so
    bucket diversity is preserved across splits."""
    out = df.copy()
    groups = class_groups or {}
    out["class_name"] = out["target_object"].map(lambda o: groups.get(o, o))
    return out


def force_test_for_objects(split_df: pd.DataFrame,
                           holdout_objects: list[str]) -> pd.DataFrame:
    """Open-set holdout: every sequence of these objects goes to the TEST split
    (never trained on), so notebook 07 measures generalization to object types
    the model has never seen."""
    if not holdout_objects:
        return split_df
    out = split_df.copy()
    mask = out["target_object"].isin(holdout_objects)
    out.loc[mask, "split"] = "test"
    return out


def split_by_sequence(df: pd.DataFrame, seed: int = 0,
                      gold_test_frac: float = 0.20,
                      auto_test_frac: float = 0.10,
                      val_frac: float = 0.15) -> pd.DataFrame:
    """Assign a 'split' column (train/val/test) per SEQUENCE, stratified by class.
    Test prefers gold (clicked) sequences; classes without gold fall back to
    auto-labeled test sequences."""
    rng = random.Random(seed)
    seq_info = (df.groupby("sequence_id")
                .agg(target_object=("target_object", "first"),
                     gold=("gold", "any")))

    split_of: dict[str, str] = {}
    for obj, grp in seq_info.groupby("target_object"):
        gold_seqs = sorted(grp[grp["gold"]].index)
        auto_seqs = sorted(grp[~grp["gold"]].index)

        if gold_seqs:
            n_test = max(1, round(gold_test_frac * len(gold_seqs)))
            test = set(rng.sample(gold_seqs, n_test))
        else:
            n_test = max(1, round(auto_test_frac * len(auto_seqs)))
            test = set(rng.sample(auto_seqs, n_test))

        remaining = sorted(set(gold_seqs + auto_seqs) - test)
        # never sacrifice train for val: a class needs >=2 remaining sequences
        # before any go to val (tiny classes get test + train only)
        if len(remaining) >= 2:
            n_val = max(1, round(val_frac * len(remaining)))
            val = set(rng.sample(remaining, min(n_val, len(remaining) - 1)))
        else:
            val = set()

        for seq in test:
            split_of[seq] = "test"
        for seq in val:
            split_of[seq] = "val"
        for seq in remaining:
            split_of.setdefault(seq, "train")

    out = df.copy()
    out["split"] = out["sequence_id"].map(split_of)
    return out


# Drive FUSE throws transient Errno 5 under bursts of small writes; retried with
# exponential backoff. Module-level so tests can zero the wait.
IO_RETRIES = 3
IO_WAIT_S = 2.0


def _with_io_retry(fn):
    """Retry transient Drive FUSE OSErrors instead of losing a long write run."""
    import time
    for attempt in range(IO_RETRIES + 1):
        try:
            return fn()
        except OSError:
            if attempt == IO_RETRIES:
                raise
            time.sleep(IO_WAIT_S * (2 ** attempt))


def write_yolo_dataset(df: pd.DataFrame, out_dir: Path, class_ids: dict[str, int],
                       frames_roots: dict[str, Path], seed: int = 0,
                       copy_images: bool = True, progress: bool = True) -> dict:
    """Write images/{split}/, labels/{split}/, dataset.yaml and provenance.json.
    class_ids: target_object -> contiguous YOLO class id. frames_roots: source
    name -> frames root for resolving stored frame paths. Returns a summary."""
    out_dir = Path(out_dir)
    # training class: bucketed 'class_name' when apply_class_groups ran, else raw object
    name_col = "class_name" if "class_name" in df.columns else "target_object"
    unknown = sorted(set(df[name_col]) - set(class_ids))
    if unknown:
        raise ValueError(f"no class id for objects: {unknown}")

    import cv2

    counts: dict = defaultdict(int)
    missing_frames: list[str] = []
    n_written = 0
    n_skipped_existing = 0
    # dimensions are constant within one camera video: decode once, reuse
    size_cache: dict[tuple, tuple[int, int]] = {}

    # group per frame so multiple boxes on one image land in one label file
    group_cols = ["split", "sequence_id", "camera", "frame_idx", "source"]
    groups = list(df.groupby(group_cols))
    if progress:
        from tqdm.auto import tqdm
        groups = tqdm(groups, desc="Writing dataset", unit="img")

    for (split, seq_id, camera, frame_idx, source), grp in groups:
        stem = f"{seq_id}_{camera}_{int(frame_idx):05d}"
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        dst_img = img_dir / f"{stem}.jpg"
        lbl_path = lbl_dir / f"{stem}.txt"

        # resume: both artifacts already written by a previous run -> skip cheaply
        try:
            already_done = _with_io_retry(
                lambda: dst_img.exists() and lbl_path.exists())
        except OSError:
            already_done = False
        if already_done:
            n_skipped_existing += 1
            counts[(split, grp.iloc[0][name_col])] += 1
            continue

        src_path = resolve_frame_path(grp.iloc[0], frames_roots[source])
        if src_path is None:
            missing_frames.append(str(grp.iloc[0]["frame_path"]))
            continue

        size_key = (source, seq_id, camera)
        if size_key not in size_cache:
            img = cv2.imread(str(src_path))
            if img is None:
                missing_frames.append(str(src_path))
                continue
            size_cache[size_key] = img.shape[:2]
        h, w = size_cache[size_key]

        lines = []
        for _, row in grp.iterrows():
            cls = class_ids[row[name_col]]
            cx = (row["x0"] + row["x1"]) / 2 / w
            cy = (row["y0"] + row["y1"]) / 2 / h
            bw = (row["x1"] - row["x0"]) / w
            bh = (row["y1"] - row["y0"]) / h
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        label_text = "\n".join(lines) + "\n"

        # every filesystem touch is retried; a frame that still fails is
        # recorded and skipped - one flaky path must never kill the run
        try:
            _with_io_retry(lambda: img_dir.mkdir(parents=True, exist_ok=True))
            _with_io_retry(lambda: lbl_dir.mkdir(parents=True, exist_ok=True))
            if not dst_img.exists():
                if copy_images:
                    _with_io_retry(lambda: shutil.copyfile(src_path, dst_img))
                else:
                    dst_img.symlink_to(src_path.resolve())
            _with_io_retry(lambda: lbl_path.write_text(label_text))
        except OSError:
            missing_frames.append(str(src_path))
            continue

        n_written += 1
        counts[(split, grp.iloc[0][name_col])] += 1

    names = [obj for obj, _ in sorted(class_ids.items(), key=lambda kv: kv[1])]
    dataset_yaml = (
        f"path: {out_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names: {json.dumps(names)}\n"
    )
    (out_dir / "dataset.yaml").write_text(dataset_yaml)

    split_seq = (df.groupby("split")["sequence_id"].nunique().to_dict())
    provenance = {
        "git_sha": git_sha(),
        "seed": seed,
        "class_ids": class_ids,
        "sources": sorted(df["source"].unique().tolist()),
        "sequences_per_split": split_seq,
        "frames_per_split_class": {f"{s}/{o}": n for (s, o), n in sorted(counts.items())},
        "gold_test_sequences": sorted(
            df[(df["split"] == "test") & df["gold"]]["sequence_id"].unique().tolist()),
        "images_written": n_written,
        "images_skipped_existing": n_skipped_existing,
        "missing_frames": missing_frames,
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=1))
    return provenance
