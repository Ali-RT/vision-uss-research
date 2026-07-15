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
