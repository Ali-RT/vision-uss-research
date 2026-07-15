from pathlib import Path

import pandas as pd

from vision_uss_research.labeling.boxes_io import (
    BOX_COLS, append_boxes, done_keys, load_boxes_tolerant,
    relativize_frame_path, resolve_frame_path, row_bbox,
)


def _mixed_csv(tmp_path: Path) -> Path:
    csv = tmp_path / "mixed.csv"
    lines = [
        "sequence_id,target_object,camera,frame_idx,frame_path,bbox_xyxy,mask_area_frac",
        'seqA,woodenboard,front,0,/old/f0.jpg,"(10, 20, 100, 200)",0.01',
        "seqA,woodenboard,front,1,/old/f1.jpg,,0.0",
        "seqB,pole,rear,0,/old/f0.jpg,1,2,3,4,0.03",  # new schema, no header
    ]
    csv.write_text("\n".join(lines) + "\n")
    return csv


def test_load_boxes_tolerant_mixed_schemas(tmp_path):
    df = load_boxes_tolerant(_mixed_csv(tmp_path))
    assert len(df) == 3
    assert df.iloc[0][["x0", "y0", "x1", "y1"]].tolist() == [10, 20, 100, 200]
    assert pd.isna(df.iloc[1]["x0"])
    assert df.iloc[2][["x0", "y0", "x1", "y1"]].tolist() == [1, 2, 3, 4]


def test_append_and_done_keys(tmp_path):
    csv = tmp_path / "boxes.csv"
    append_boxes(csv, [{"sequence_id": "s1", "target_object": "pole",
                        "camera": "rear", "frame_idx": 0, "frame_path": "s1/rear/0.jpg",
                        "x0": 1, "y0": 2, "x1": 3, "y1": 4, "mask_area_frac": 0.1}])
    append_boxes(csv, [{"sequence_id": "s2", "target_object": "pole",
                        "camera": "front", "frame_idx": 0, "frame_path": "s2/front/0.jpg",
                        "x0": 1, "y0": 2, "x1": 3, "y1": 4, "mask_area_frac": 0.1}])
    df = load_boxes_tolerant(csv)
    assert list(df.columns) == BOX_COLS
    assert len(df) == 2
    assert done_keys(csv) == {"s1::rear", "s2::front"}


def test_relativize_and_resolve_roundtrip(tmp_path):
    frames_root = tmp_path / "frames"
    frame = frames_root / "seq1" / "rear" / "00003.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"jpg")

    rel = relativize_frame_path(frame, frames_root)
    assert not Path(rel).is_absolute()
    row = {"frame_path": rel, "sequence_id": "seq1", "camera": "rear"}
    assert resolve_frame_path(row, frames_root) == frames_root / rel


def test_resolve_rebases_stale_absolute_path(tmp_path):
    frames_root = tmp_path / "frames"
    frame = frames_root / "seq1" / "rear" / "00003.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"jpg")

    row = {"frame_path": "/content/drive/OLD/frames/seq1/rear/00003.jpg",
           "sequence_id": "seq1", "camera": "rear"}
    assert resolve_frame_path(row, frames_root) == frame
    row_missing = {"frame_path": "/content/drive/OLD/frames/gone/rear/x.jpg",
                   "sequence_id": "gone", "camera": "rear"}
    assert resolve_frame_path(row_missing, frames_root) is None


def test_row_bbox_both_schemas():
    assert row_bbox(pd.Series({"x0": 1, "y0": 2, "x1": 3, "y1": 4})) == (1, 2, 3, 4)
    assert row_bbox(pd.Series({"x0": None, "bbox_xyxy": "(1, 2, 3, 4)"})) == (1, 2, 3, 4)
    assert row_bbox(pd.Series({"x0": None, "bbox_xyxy": ""})) is None
