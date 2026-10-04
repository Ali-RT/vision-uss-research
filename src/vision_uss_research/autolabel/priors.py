"""Learned box priors - the v2 auto-labeler's size and placement step.

v1 draws a fixed 0.75 x 1.75 m box centred on the map object. v2 keeps the
same map-object face (with the lateral offset) but takes, from labelled
TRAINING sequences of the same group (object class, optionally x approach):
  - box length along / width across the ego heading (median), and
  - the offset from the face centre to the human box centre, along the ego
    heading (signed away from the car) and across it (median).
Groups with < `min_n` training sequences fall back to the global median.

Inputs are rows of a notebook-16 results table (face_cx, face_cy,
ego_yaw_rad, obj_side, gt_polygon, ...). Object class and approach come from
the campaign metadata, so a deployed auto-labeler has them at labelling time.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from vision_uss_research.autolabel.geometry import box_metrics, centroid, oriented_box
from vision_uss_research.autolabel.uss_box import polygon_from_str

PRIOR_COLS = ["off_along_m", "off_across_m", "gt_len_along_m", "gt_wid_across_m"]


def add_offsets(df: pd.DataFrame) -> pd.DataFrame:
    """Human box centre relative to the map-object face, in the ego frame:
    `off_along_m` > 0 = further from the car, `off_across_m` > 0 = to its left."""
    df = df.copy()
    gc = np.array([centroid(polygon_from_str(p)) for p in df["gt_polygon"]])
    dx, dy = gc[:, 0] - df["face_cx"].to_numpy(), gc[:, 1] - df["face_cy"].to_numpy()
    c, s = np.cos(df["ego_yaw_rad"].to_numpy()), np.sin(df["ego_yaw_rad"].to_numpy())
    df["off_along_m"] = (dx * c + dy * s) * df["obj_side"].to_numpy()
    df["off_across_m"] = -dx * s + dy * c
    return df


def fit_priors(train: pd.DataFrame, keys: list[str], use_offset: bool = True,
               min_n: int = 5) -> dict:
    """Group medians of the prior columns (+ global fallback)."""
    cols = PRIOR_COLS if use_offset else PRIOR_COLS[2:]
    return {"keys": keys, "use_offset": use_offset, "min_n": min_n,
            "group": train.groupby(keys)[cols].median(),
            "n": train.groupby(keys).size(),
            "global": train[cols].median()}


def prior_box(row, priors: dict) -> tuple[list, bool]:
    """(corners, used_group_prior) for one results row."""
    keys = priors["keys"]
    key = tuple(row[k] for k in keys) if len(keys) > 1 else row[keys[0]]
    use = key in priors["group"].index and priors["n"][key] >= priors["min_n"]
    st = priors["group"].loc[key] if use else priors["global"]
    a, b = (st["off_along_m"], st["off_across_m"]) if priors["use_offset"] else (0.0, 0.0)
    yaw, side = float(row["ego_yaw_rad"]), int(row["obj_side"])
    cx = row["face_cx"] + side * a * np.cos(yaw) - b * np.sin(yaw)
    cy = row["face_cy"] + side * a * np.sin(yaw) + b * np.cos(yaw)
    return oriented_box(cx, cy, yaw, float(st["gt_len_along_m"]),
                        float(st["gt_wid_across_m"])), bool(use)


def fold_of(sequence_id: str, k: int) -> int:
    """Stable fold assignment from the sequence id."""
    return int(hashlib.md5(str(sequence_id).encode()).hexdigest(), 16) % k


def cross_fit(df: pd.DataFrame, keys: list[str], use_offset: bool = True,
              k: int = 5, min_n: int = 5, prefix: str = "prior") -> pd.DataFrame:
    """Score every row with priors fit on the OTHER folds. Returns a frame
    indexed like `df` with `{prefix}_<box metrics>` and `{prefix}_group_prior`."""
    if "off_along_m" not in df:
        df = add_offsets(df)
    folds = df["sequence_id"].map(lambda s: fold_of(s, k))
    out = []
    for f in range(k):
        priors = fit_priors(df[folds != f], keys, use_offset, min_n)
        for i, row in df[folds == f].iterrows():
            corners, used = prior_box(row, priors)
            m = box_metrics(corners, polygon_from_str(row["gt_polygon"]))
            out.append({"_i": i, f"{prefix}_group_prior": int(used),
                        **{f"{prefix}_{name}": v for name, v in m.items()}})
    return pd.DataFrame(out).set_index("_i").reindex(df.index)
