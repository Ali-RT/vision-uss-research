"""Click-and-track ground truth (notebook 19).

One click on the object in one frame of a trace; the SAM3 video tracker
follows it forward and backward through every frame. Corrections are more
clicks on other frames (each frame then takes the track of the nearest click)
or "not visible after / before frame X" cut-offs. Pure functions here; the
tracker runs in the notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BOX = ["x0", "y0", "x1", "y1"]


def mask_to_box(mask: np.ndarray, min_area: int = 100):
    """xyxy box of a binary mask, or None when the mask is (nearly) empty."""
    if mask is None or int(mask.sum()) < min_area:
        return None
    ys, xs = np.nonzero(mask)
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def stop_after_loss(boxes: list, stop_after: int = 8) -> list:
    """Boxes in tracking order (away from the click). Once the object is lost
    for `stop_after` consecutive frames it has left the view: every later
    frame becomes None, so the tracker cannot re-lock onto a look-alike (the
    wheel stops across the lot once the target curb is under the bumper)."""
    out, missing, gone = [], 0, False
    for b in boxes:
        if gone:
            out.append(None)
            continue
        missing = missing + 1 if b is None else 0
        if missing >= stop_after:
            gone = True
            out = out[:len(out) - (missing - 1)] + [None] * (missing - 1)
            out.append(None)
            continue
        out.append(b)
    return out


def merge_by_nearest_click(frames: list[int], tracks: dict[int, dict[int, tuple | None]]
                           ) -> dict[int, tuple | None]:
    """tracks: {click_frame: {frame: box|None}}. Each frame takes the answer
    of the nearest click (in frame index) whose track covers it."""
    clicks = sorted(tracks)
    out = {}
    for f in frames:
        for c in sorted(clicks, key=lambda c: (abs(c - f), c)):
            if f in tracks[c]:
                out[f] = tracks[c][f]
                break
        else:
            out[f] = None
    return out


def apply_cutoffs(boxes: dict[int, tuple | None], not_visible_after: int | None = None,
                  not_visible_before: int | None = None) -> dict[int, tuple | None]:
    out = dict(boxes)
    for f in out:
        if (not_visible_after is not None and f > not_visible_after) or \
                (not_visible_before is not None and f < not_visible_before):
            out[f] = None
    return out


def to_gold_rows(side: str, trace: int, category: str,
                 boxes: dict[int, tuple | None], reviewer: str = "clicktrack") -> pd.DataFrame:
    """Gold-CSV rows (format of scripts/gold_review_template.html exports)."""
    rows = []
    for f, b in sorted(boxes.items()):
        rows.append({"item_id": f"{side}_{trace}_{int(f):05d}", "side": side,
                     "trace": int(trace), "frame_idx": int(f), "category": category,
                     "reviewer": reviewer, "status": "box" if b is not None else "not_visible",
                     **dict(zip(BOX, b if b is not None else [np.nan] * 4)),
                     "truncated": 0, "time": ""})
    return pd.DataFrame(rows)
