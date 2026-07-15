"""The one true schema for propagated box labels, plus tolerant I/O.

Frame paths are stored RELATIVE to the run's frames root whenever possible, so
labels stay valid when the artifacts folder moves (the stale-absolute-path bug).
The tolerant loader still understands the two historical schemas.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

BOX_COLS = ["sequence_id", "target_object", "camera", "frame_idx", "frame_path",
            "x0", "y0", "x1", "y1", "mask_area_frac"]

_NUMERIC_COLS = ["frame_idx", "x0", "y0", "x1", "y1", "mask_area_frac"]


def relativize_frame_path(frame_path: Path, frames_root: Path) -> str:
    """Store paths relative to the frames root when inside it."""
    try:
        return str(Path(frame_path).resolve().relative_to(Path(frames_root).resolve()))
    except ValueError:
        return str(frame_path)


def resolve_frame_path(row, frames_root: Path) -> Path | None:
    """Resolve a stored frame path: relative to frames_root, an absolute path
    that still exists, or (legacy absolute) rebased onto frames_root by its
    <seq>/<camera>/<name> tail."""
    raw = Path(str(row["frame_path"]))
    if not raw.is_absolute():
        candidate = Path(frames_root) / raw
        return candidate if candidate.exists() else None
    if raw.exists():
        return raw
    rebased = (Path(frames_root) / str(row["sequence_id"])
               / str(row["camera"]) / raw.name)
    return rebased if rebased.exists() else None


def append_boxes(path: Path, rows: list[dict]) -> None:
    """Append rows in the canonical schema (header written once)."""
    path = Path(path)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BOX_COLS)
        if write_header:
            writer.writeheader()
        writer.writerows({k: r.get(k) for k in BOX_COLS} for r in rows)


def load_boxes_tolerant(path: Path) -> pd.DataFrame:
    """Read a propagated_boxes.csv even when legacy 7-column rows (bbox_xyxy
    tuple strings) and canonical 10-column rows were appended into one file."""
    rows = []
    with Path(path).open(newline="") as f:
        for rec in csv.reader(f):
            if not rec or rec[0] == "sequence_id":  # header lines, either schema
                continue
            if len(rec) == len(BOX_COLS):
                rows.append(dict(zip(BOX_COLS, rec)))
            elif len(rec) == 7:  # legacy: bbox_xyxy tuple string at index 5
                nums = re.findall(r"-?\d+", rec[5] or "")
                bbox = list(map(int, nums)) if len(nums) == 4 else [None] * 4
                rows.append({"sequence_id": rec[0], "target_object": rec[1],
                             "camera": rec[2], "frame_idx": rec[3],
                             "frame_path": rec[4],
                             "x0": bbox[0], "y0": bbox[1],
                             "x1": bbox[2], "y1": bbox[3],
                             "mask_area_frac": rec[6]})
            # anything else: malformed line, drop it
    df = pd.DataFrame(rows, columns=BOX_COLS)
    for col in _NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def row_bbox(row) -> tuple[int, int, int, int] | None:
    """Bbox from either schema: x0..y1 columns or a legacy bbox_xyxy string."""
    if pd.notna(row.get("x0")):
        return int(row["x0"]), int(row["y0"]), int(row["x1"]), int(row["y1"])
    nums = re.findall(r"-?\d+", str(row.get("bbox_xyxy", "") or ""))
    return tuple(map(int, nums)) if len(nums) == 4 else None


def done_keys(boxes_csv: Path) -> set[str]:
    """'<sequence_id>::<camera>' keys already present - the resume set."""
    if not Path(boxes_csv).exists():
        return set()
    df = load_boxes_tolerant(boxes_csv)
    return set(df["sequence_id"].astype(str) + "::" + df["camera"].astype(str))
