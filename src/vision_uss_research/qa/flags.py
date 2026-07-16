"""Mechanical label-QA flags. Heuristics that rank frames for human review,
not verdicts: big_box (tracker exploded onto background), tiny_box (collapsed
mask; per-class thresholds because pole/woodenboard are genuinely tiny),
extreme_aspect, jump (box teleported between consecutive frames = drift/re-lock).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class FlagThresholds:
    big_box_area: float = 0.45
    tiny_box_area_default: float = 0.0008
    # small/thin classes are GENUINELY tiny in-frame (full-run medians: cone
    # 0.0009, pole 0.0014, tree/bollard trunk-thin) - per-class floors
    tiny_box_area_by_class: dict = field(default_factory=lambda: {
        "pole": 0.0002, "squarepole": 0.0002,
        "woodenboard": 0.00015, "hose": 0.00015,
        "cone": 0.0002, "tree": 0.0002, "bollard": 0.0002,
        "car": 0.0003, "cubestandard": 0.0003,
    })
    # aspect is symmetric (max(w/h, h/w)); elongated classes exceed 12 by
    # normal geometry: a head-on curb is ~900x40 px, a pole is 1:15 vertical
    extreme_aspect_default: float = 12.0
    extreme_aspect_by_class: dict = field(default_factory=lambda: {
        "curbstone": 45.0, "curbstone_side": 45.0, "speedbump": 45.0,
        "hose": 45.0, "woodenboard": 30.0, "ubarrier": 30.0,
        "pole": 40.0, "squarepole": 40.0, "tree": 40.0, "bollard": 30.0,
        "bicyclestand": 30.0,
    })
    jump_iou: float = 0.10


def box_iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


def compute_flags(boxed: pd.DataFrame,
                  thresholds: FlagThresholds = FlagThresholds()) -> pd.DataFrame:
    """boxed: canonical-schema rows that HAVE a box (x0 notna), with an 'aspect'
    column optional (derived here when absent). Returns one row per flagged frame
    with a '|'-joined 'flags' column."""
    boxed = boxed.copy()
    if "aspect" not in boxed.columns:
        box_w = boxed["x1"] - boxed["x0"]
        box_h = (boxed["y1"] - boxed["y0"]).clip(lower=1)
        boxed["aspect"] = (box_w / box_h).clip(lower=1e-6)

    flags = []
    sort_cols = ["sequence_id", "camera", "frame_idx"]
    for (_, _), grp in boxed.sort_values(sort_cols).groupby(["sequence_id", "camera"]):
        prev_box = None
        for _, row in grp.iterrows():
            cur_box = (row["x0"], row["y0"], row["x1"], row["y1"])
            reasons = []
            if row["mask_area_frac"] > thresholds.big_box_area:
                reasons.append("big_box")
            tiny = thresholds.tiny_box_area_by_class.get(
                row["target_object"], thresholds.tiny_box_area_default)
            if row["mask_area_frac"] < tiny:
                reasons.append("tiny_box")
            aspect_limit = thresholds.extreme_aspect_by_class.get(
                row["target_object"], thresholds.extreme_aspect_default)
            if max(row["aspect"], 1 / row["aspect"]) > aspect_limit:
                reasons.append("extreme_aspect")
            if prev_box is not None and box_iou(prev_box, cur_box) < thresholds.jump_iou:
                reasons.append("jump")
            if reasons:
                record = row[["sequence_id", "target_object", "camera",
                              "frame_idx", "frame_path"]].to_dict()
                if "source" in row:
                    record["source"] = row["source"]
                record.update({"x0": row["x0"], "y0": row["y0"],
                               "x1": row["x1"], "y1": row["y1"],
                               "flags": "|".join(reasons)})
                flags.append(record)
            prev_box = cur_box
    return pd.DataFrame(flags)
