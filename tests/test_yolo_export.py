import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from vision_uss_research.labeling.boxes_io import append_boxes
from vision_uss_research.datasets.yolo_export import (
    merge_label_sources, split_by_sequence, write_yolo_dataset,
)


@pytest.fixture
def label_world(tmp_path):
    """Two sources: auto (curbstone, 10 seqs) and clicks (woodenboard, 5 seqs),
    with real tiny frame images on disk."""
    world = {"sources": [], "frames_roots": {}}
    for name, obj, n_seqs, gold in [("auto", "curbstone", 10, False),
                                    ("clicks", "woodenboard", 5, True)]:
        frames_root = tmp_path / name / "frames"
        boxes_csv = tmp_path / name / "boxes.csv"
        rows = []
        for s in range(n_seqs):
            seq = f"{name}_seq{s}"
            for f in range(3):
                rel = f"{seq}/rear/{f:05d}.jpg"
                img_path = frames_root / rel
                img_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(img_path), np.zeros((64, 96, 3), np.uint8))
                rows.append(dict(sequence_id=seq, target_object=obj, camera="rear",
                                 frame_idx=f, frame_path=rel,
                                 x0=10, y0=20, x1=40, y1=50, mask_area_frac=0.02))
        append_boxes(boxes_csv, rows)
        world["sources"].append({"name": name, "boxes_csv": boxes_csv,
                                 "frames_root": frames_root, "gold": gold})
        world["frames_roots"][name] = frames_root
    return world


def test_merge_and_split_disjoint_with_gold_test(label_world):
    merged = merge_label_sources(label_world["sources"])
    assert set(merged["source"]) == {"auto", "clicks"}

    split = split_by_sequence(merged, seed=0)
    # every sequence in exactly one split
    per_seq = split.groupby("sequence_id")["split"].nunique()
    assert (per_seq == 1).all()
    assert set(split["split"]) == {"train", "val", "test"}
    # gold class: test sequences must be gold (clicked)
    wb_test = split[(split["target_object"] == "woodenboard") & (split["split"] == "test")]
    assert len(wb_test) and wb_test["gold"].all()
    # auto-only class still gets a test split (fallback)
    cs_test = split[(split["target_object"] == "curbstone") & (split["split"] == "test")]
    assert len(cs_test) and not cs_test["gold"].any()


def test_split_deterministic(label_world):
    merged = merge_label_sources(label_world["sources"])
    a = split_by_sequence(merged, seed=0)["split"].tolist()
    b = split_by_sequence(merged, seed=0)["split"].tolist()
    c = split_by_sequence(merged, seed=1)["split"].tolist()
    assert a == b
    assert a != c


def test_write_yolo_dataset(label_world, tmp_path):
    merged = merge_label_sources(label_world["sources"])
    split = split_by_sequence(merged, seed=0)
    out = tmp_path / "dataset"
    class_ids = {"curbstone": 0, "woodenboard": 1}

    provenance = write_yolo_dataset(split, out, class_ids,
                                    label_world["frames_roots"], seed=0)

    assert (out / "dataset.yaml").exists()
    assert not provenance["missing_frames"]
    # images and labels pair up per split
    for split_name in ["train", "val", "test"]:
        imgs = sorted(p.stem for p in (out / "images" / split_name).glob("*.jpg"))
        lbls = sorted(p.stem for p in (out / "labels" / split_name).glob("*.txt"))
        assert imgs and imgs == lbls
    # label format: class cx cy w h normalized to (96, 64)
    any_label = next((out / "labels" / "train").glob("*.txt")).read_text().split()
    assert any_label[0] in {"0", "1"}
    cx, cy, bw, bh = map(float, any_label[1:])
    assert abs(cx - 25 / 96) < 1e-4 and abs(cy - 35 / 64) < 1e-4
    assert abs(bw - 30 / 96) < 1e-4 and abs(bh - 30 / 64) < 1e-4
    # provenance records the gold holdout
    recorded = json.loads((out / "provenance.json").read_text())
    assert recorded["gold_test_sequences"]
    assert all(s.startswith("clicks_") for s in recorded["gold_test_sequences"])


def test_tiny_class_never_loses_train(tmp_path):
    """A class with only 2 sequences must end up test+train, never test+val."""
    frames_root = tmp_path / "clicks" / "frames"
    boxes_csv = tmp_path / "clicks" / "boxes.csv"
    rows = []
    for s in range(2):
        seq = f"tiny_seq{s}"
        rel = f"{seq}/rear/00000.jpg"
        img = frames_root / rel
        img.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(img), np.zeros((64, 96, 3), np.uint8))
        rows.append(dict(sequence_id=seq, target_object="bicyclestand",
                         camera="rear", frame_idx=0, frame_path=rel,
                         x0=10, y0=20, x1=40, y1=50, mask_area_frac=0.02))
    append_boxes(boxes_csv, rows)
    merged = merge_label_sources([{"name": "clicks", "boxes_csv": boxes_csv,
                                   "frames_root": frames_root, "gold": True}])
    split = split_by_sequence(merged, seed=0)
    by_split = split.groupby("split")["sequence_id"].nunique().to_dict()
    assert by_split.get("test") == 1
    assert by_split.get("train") == 1
    assert "val" not in by_split


def test_class_groups_and_open_set_holdout(label_world):
    from vision_uss_research.datasets.yolo_export import (apply_class_groups,
                                                          force_test_for_objects)
    merged = merge_label_sources(label_world["sources"])
    # bucket curbstone into low_other; woodenboard stays named
    grouped = apply_class_groups(merged, {"curbstone": "low_other"})
    assert set(grouped["class_name"]) == {"low_other", "woodenboard"}
    # no grouping -> identity
    identity = apply_class_groups(merged, None)
    assert (identity["class_name"] == identity["target_object"]).all()

    split = split_by_sequence(grouped, seed=0)
    held = force_test_for_objects(split, ["curbstone"])
    assert (held[held["target_object"] == "curbstone"]["split"] == "test").all()
    # untouched class keeps its original split assignment
    pd.testing.assert_series_equal(
        held[held["target_object"] == "woodenboard"]["split"],
        split[split["target_object"] == "woodenboard"]["split"])


def test_write_dataset_uses_grouped_class_ids(label_world, tmp_path):
    from vision_uss_research.datasets.yolo_export import apply_class_groups
    merged = apply_class_groups(merge_label_sources(label_world["sources"]),
                                {"curbstone": "low_other"})
    split = split_by_sequence(merged, seed=0)
    out = tmp_path / "ds"
    provenance = write_yolo_dataset(split, out, {"low_other": 0, "woodenboard": 1},
                                    label_world["frames_roots"], seed=0)
    assert not provenance["missing_frames"]
    labels = [p.read_text().split()[0]
              for p in (out / "labels").rglob("*.txt")]
    assert set(labels) == {"0", "1"}
    names = json.loads((out / "dataset.yaml").read_text().splitlines()[-1].split("names: ")[1])
    assert names == ["low_other", "woodenboard"]


def test_merge_dedupes_overlapping_sources_newest_wins(tmp_path):
    """The same frame labeled by two runs keeps only the newer run's row."""
    frames_root = tmp_path / "frames"
    rel = "seqX/rear/00000.jpg"
    img = frames_root / rel
    img.parent.mkdir(parents=True)
    cv2.imwrite(str(img), np.zeros((64, 96, 3), np.uint8))

    sources = []
    for run_name, x0 in [("20260716_labels_old", 10), ("20260717_labels_new", 50)]:
        csv_path = tmp_path / run_name / "boxes.csv"
        append_boxes(csv_path, [dict(sequence_id="seqX", target_object="curbstone",
                                     camera="rear", frame_idx=0, frame_path=rel,
                                     x0=x0, y0=20, x1=x0 + 30, y1=50,
                                     mask_area_frac=0.02)])
        sources.append({"name": run_name, "boxes_csv": csv_path,
                        "frames_root": frames_root, "gold": False})

    merged = merge_label_sources(sources)
    assert len(merged) == 1
    assert merged.iloc[0]["x0"] == 50
    assert merged.iloc[0]["source"] == "20260717_labels_new"


def test_write_dataset_resumes_without_rereading(label_world, tmp_path):
    """Second run over the same out_dir skips every already-written frame."""
    merged = merge_label_sources(label_world["sources"])
    split = split_by_sequence(merged, seed=0)
    out = tmp_path / "ds"
    class_ids = {"curbstone": 0, "woodenboard": 1}

    first = write_yolo_dataset(split, out, class_ids,
                               label_world["frames_roots"], progress=False)
    assert first["images_written"] > 0
    assert first["images_skipped_existing"] == 0

    second = write_yolo_dataset(split, out, class_ids,
                                label_world["frames_roots"], progress=False)
    assert second["images_written"] == 0
    assert second["images_skipped_existing"] == first["images_written"]
    # per-class counts identical across the resume
    assert second["frames_per_split_class"] == first["frames_per_split_class"]


def test_write_dataset_size_cache_per_sequence(tmp_path):
    """Label normalization uses each sequence's own frame dimensions."""
    frames_root = tmp_path / "frames"
    rows = []
    for seq, (w, h) in [("wide_seq", (96, 64)), ("tall_seq", (64, 96))]:
        rel = f"{seq}/rear/00000.jpg"
        img = frames_root / rel
        img.parent.mkdir(parents=True)
        cv2.imwrite(str(img), np.zeros((h, w, 3), np.uint8))
        rows.append(dict(sequence_id=seq, target_object="curbstone", camera="rear",
                         frame_idx=0, frame_path=rel, x0=0, y0=0, x1=32, y1=32,
                         mask_area_frac=0.1))
    csv_path = tmp_path / "boxes.csv"
    append_boxes(csv_path, rows)
    merged = merge_label_sources([{"name": "s", "boxes_csv": csv_path,
                                   "frames_root": frames_root, "gold": False}])
    merged["split"] = "train"
    merged["gold"] = False
    out = tmp_path / "ds"
    write_yolo_dataset(merged, out, {"curbstone": 0}, {"s": frames_root},
                       progress=False)
    wide = (out / "labels" / "train" / "wide_seq_rear_00000.txt").read_text().split()
    tall = (out / "labels" / "train" / "tall_seq_rear_00000.txt").read_text().split()
    assert abs(float(wide[3]) - 32 / 96) < 1e-4   # bw normalized by w=96
    assert abs(float(tall[3]) - 32 / 64) < 1e-4   # bw normalized by w=64


def test_write_dataset_survives_persistent_io_failure(label_world, tmp_path, monkeypatch):
    """A path that keeps failing lands in missing_frames; the run completes."""
    import vision_uss_research.datasets.yolo_export as ye
    monkeypatch.setattr(ye, "IO_WAIT_S", 0.0)

    merged = merge_label_sources(label_world["sources"])
    split = split_by_sequence(merged, seed=0)
    out = tmp_path / "ds"
    class_ids = {"curbstone": 0, "woodenboard": 1}

    real_copy = ye.shutil.copyfile
    fail_stem = sorted(split["sequence_id"].unique())[0]

    def flaky_copy(src, dst):
        if fail_stem in str(dst):
            raise OSError(5, "Input/output error")
        return real_copy(src, dst)

    monkeypatch.setattr(ye.shutil, "copyfile", flaky_copy)
    provenance = write_yolo_dataset(split, out, class_ids,
                                    label_world["frames_roots"], progress=False)
    assert provenance["missing_frames"], "failing frames should be recorded"
    assert provenance["images_written"] > 0, "other frames still written"
    # a later run with healthy I/O picks the failed frames back up
    monkeypatch.setattr(ye.shutil, "copyfile", real_copy)
    retry = write_yolo_dataset(split, out, class_ids,
                               label_world["frames_roots"], progress=False)
    assert not retry["missing_frames"]
    assert retry["images_written"] == len(provenance["missing_frames"])


def test_progress_heartbeat_written(label_world, tmp_path):
    merged = merge_label_sources(label_world["sources"])
    split = split_by_sequence(merged, seed=0)
    out = tmp_path / "ds"
    progress_file = out / "progress.json"
    out.mkdir()
    write_yolo_dataset(split, out, {"curbstone": 0, "woodenboard": 1},
                       label_world["frames_roots"], progress=False,
                       progress_file=progress_file, progress_every=5)
    prog = json.loads(progress_file.read_text())
    assert prog["status"] == "completed"
    assert prog["done"] == prog["total"] > 0
    assert prog["written"] == prog["total"]
    assert prog["missing"] == 0


def test_manifest_and_materialize_roundtrip(label_world, tmp_path):
    from vision_uss_research.datasets.yolo_export import (materialize_yolo_dataset,
                                                          write_dataset_manifest)
    merged = merge_label_sources(label_world["sources"])
    split = split_by_sequence(merged, seed=0)
    drive_dir = tmp_path / "drive_run"
    class_ids = {"curbstone": 0, "woodenboard": 1}

    prov = write_dataset_manifest(split, drive_dir, class_ids,
                                  label_world["frames_roots"], seed=0)
    # Drive side holds only small files - no images
    assert (drive_dir / "dataset_manifest.csv").exists()
    assert not list(drive_dir.rglob("*.jpg"))
    # target_object survives into the manifest (held-out slicing needs it)
    man = pd.read_csv(drive_dir / "dataset_manifest.csv")
    assert "target_object" in man.columns
    assert set(man["target_object"]) == {"curbstone", "woodenboard"}
    assert prov["class_names"] == ["curbstone", "woodenboard"]
    assert prov["gold_test_sequences"]

    local = tmp_path / "local_ds"
    heartbeat = drive_dir / "progress.json"
    result = materialize_yolo_dataset(
        drive_dir / "dataset_manifest.csv",
        {k: Path(v) for k, v in prov["frames_roots"].items()},
        local, class_names=prov["class_names"], progress=False,
        heartbeat_file=heartbeat, progress_every=5)
    assert result["images_written"] > 0 and not result["missing_frames"]
    # sharded per sequence, labels mirror images
    imgs = sorted(p.relative_to(local / "images") for p in local.rglob("*.jpg"))
    lbls = sorted(p.relative_to(local / "labels").with_suffix(".jpg")
                  for p in (local / "labels").rglob("*.txt"))
    assert imgs and imgs == lbls
    assert all(len(p.parts) == 3 for p in imgs)  # split/sequence_id/stem.jpg
    # labels normalized against real image size (96x64, box 10,20-40,50)
    any_lbl = next((local / "labels").rglob("*.txt")).read_text().split()
    assert abs(float(any_lbl[3]) - 30 / 96) < 1e-4
    # heartbeat completed
    prog = json.loads(heartbeat.read_text())
    assert prog["status"] == "completed" and prog["done"] == prog["total"]
    # dataset.yaml points at the local dir
    assert str(local.resolve()) in (local / "dataset.yaml").read_text()

    # resume: second materialization writes nothing
    again = materialize_yolo_dataset(
        drive_dir / "dataset_manifest.csv",
        {k: Path(v) for k, v in prov["frames_roots"].items()},
        local, class_names=prov["class_names"], progress=False)
    assert again["images_written"] == 0
    assert again["images_skipped_existing"] == result["images_written"]


def test_materialize_parallel_correctness(tmp_path):
    """Many sequences materialized with a thread pool: every image has its
    label, labels use each sequence's own dimensions, no cross-shard collisions."""
    from vision_uss_research.datasets.yolo_export import (materialize_yolo_dataset,
                                                          write_dataset_manifest)
    frames_root = tmp_path / "frames"
    boxes_csv = tmp_path / "boxes.csv"
    rows = []
    # 40 sequences x 2 dims x 3 frames -> exercises sharding + per-seq dims in parallel
    for i in range(40):
        obj = "curbstone" if i % 2 else "woodenboard"
        w, h = (96, 64) if i % 2 else (64, 96)
        seq = f"{obj}_seq{i}"
        for f in range(3):
            rel = f"{seq}/rear/{f:05d}.jpg"
            img = frames_root / rel
            img.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(img), np.zeros((h, w, 3), np.uint8))
            rows.append(dict(sequence_id=seq, target_object=obj, camera="rear",
                             frame_idx=f, frame_path=rel, x0=10, y0=20, x1=40, y1=50,
                             mask_area_frac=0.02))
    append_boxes(boxes_csv, rows)
    merged = merge_label_sources([{"name": "s", "boxes_csv": boxes_csv,
                                   "frames_root": frames_root, "gold": True}])
    split = split_by_sequence(merged, seed=0)
    drive_dir = tmp_path / "drive"
    prov = write_dataset_manifest(split, drive_dir, {"curbstone": 0, "woodenboard": 1},
                                  {"s": frames_root}, seed=0)

    local = tmp_path / "local"
    result = materialize_yolo_dataset(drive_dir / "dataset_manifest.csv",
                                      {"s": frames_root}, local,
                                      class_names=prov["class_names"],
                                      progress=False, num_workers=16)
    assert result["images_written"] == 120 and not result["missing_frames"]
    imgs = sorted(p.relative_to(local / "images") for p in local.rglob("*.jpg"))
    lbls = sorted(p.relative_to(local / "labels").with_suffix(".jpg")
                  for p in (local / "labels").rglob("*.txt"))
    assert imgs == lbls and len(imgs) == 120
    # a woodenboard (64x96) label: bw normalized by 64
    wb = next((local / "labels").rglob("woodenboard_*.txt")).read_text().split()
    assert abs(float(wb[3]) - 30 / 64) < 1e-4
    # a curbstone (96x64) label: bw normalized by 96
    cs = next((local / "labels").rglob("curbstone_*.txt")).read_text().split()
    assert abs(float(cs[3]) - 30 / 96) < 1e-4


def test_day_grouped_split_no_day_crosses_splits(tmp_path):
    """Leakage audit: with group='day', no recording day appears in more
    than one split, and every class keeps training data when it spans
    multiple days."""
    from vision_uss_research.datasets.yolo_export import split_by_day
    rows = []
    rng = np.random.default_rng(0)
    days = [f"2019{m:02d}{d:02d}" for m in (3, 4, 5) for d in (1, 8, 15, 22)]
    for i, day in enumerate(days * 3):                 # 36 sequences, 12 days
        seq = f"{110000+i}_{day}_LB_X_{i:03d}"
        obj = ["curbstone", "pole", "woodenboard"][i % 3]
        rows.append(dict(sequence_id=seq, target_object=obj, gold=False,
                         camera="rear", frame_idx=0, frame_path="x",
                         x0=0, y0=0, x1=10, y1=10, mask_area_frac=0.01,
                         source="s"))
    df = pd.DataFrame(rows)
    out = split_by_day(df, seed=0, test_frac=0.2, val_frac=0.15)
    day_of = out.sequence_id.str.split("_").str[1]
    crossings = out.groupby(day_of)["split"].nunique()
    assert (crossings == 1).all()
    train_classes = set(out[out.split == "train"].target_object)
    assert train_classes == {"curbstone", "pole", "woodenboard"}
    assert set(out.split.unique()) >= {"train", "test"}
