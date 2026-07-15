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
    Returns the concatenated canonical rows with 'source' and 'gold' columns."""
    parts = []
    for source in sources:
        df = load_boxes_tolerant(Path(source["boxes_csv"]))
        df["source"] = source["name"]
        df["gold"] = bool(source.get("gold", False))
        parts.append(df)
    merged = pd.concat(parts, ignore_index=True)
    return merged[merged["x0"].notna()].reset_index(drop=True)


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
        n_val = max(1, round(val_frac * len(remaining))) if remaining else 0
        val = set(rng.sample(remaining, min(n_val, len(remaining))))

        for seq in test:
            split_of[seq] = "test"
        for seq in val:
            split_of[seq] = "val"
        for seq in remaining:
            split_of.setdefault(seq, "train")

    out = df.copy()
    out["split"] = out["sequence_id"].map(split_of)
    return out


def write_yolo_dataset(df: pd.DataFrame, out_dir: Path, class_ids: dict[str, int],
                       frames_roots: dict[str, Path], seed: int = 0,
                       copy_images: bool = True) -> dict:
    """Write images/{split}/, labels/{split}/, dataset.yaml and provenance.json.
    class_ids: target_object -> contiguous YOLO class id. frames_roots: source
    name -> frames root for resolving stored frame paths. Returns a summary."""
    out_dir = Path(out_dir)
    unknown = sorted(set(df["target_object"]) - set(class_ids))
    if unknown:
        raise ValueError(f"no class id for objects: {unknown}")

    counts: dict = defaultdict(int)
    missing_frames: list[str] = []
    # group per frame so multiple boxes on one image land in one label file
    group_cols = ["split", "sequence_id", "camera", "frame_idx", "source"]
    for (split, seq_id, camera, frame_idx, source), grp in df.groupby(group_cols):
        src_path = resolve_frame_path(grp.iloc[0], frames_roots[source])
        if src_path is None:
            missing_frames.append(str(grp.iloc[0]["frame_path"]))
            continue
        import cv2
        img = cv2.imread(str(src_path))
        if img is None:
            missing_frames.append(str(src_path))
            continue
        h, w = img.shape[:2]

        stem = f"{seq_id}_{camera}_{int(frame_idx):05d}"
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        dst_img = img_dir / f"{stem}.jpg"
        if not dst_img.exists():
            if copy_images:
                shutil.copyfile(src_path, dst_img)
            else:
                dst_img.symlink_to(src_path.resolve())

        lines = []
        for _, row in grp.iterrows():
            cls = class_ids[row["target_object"]]
            cx = (row["x0"] + row["x1"]) / 2 / w
            cy = (row["y0"] + row["y1"]) / 2 / h
            bw = (row["x1"] - row["x0"]) / w
            bh = (row["y1"] - row["y0"]) / h
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        counts[(split, grp.iloc[0]["target_object"])] += 1

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
        "missing_frames": missing_frames,
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=1))
    return provenance
