"""Silver ground truth from two independent models (notebook 18).

SAM3 (text prompt, every frame on its own - never seeded by the detector or
tracker under test) and Qwen2.5-VL (asked to locate the named object) label
the same frames. Agreement becomes silver ground truth in the gold-CSV format
that scripts/rate_autolabel.py reads; disagreement is 'disputed' and left out.
A small human audit of the silver labels measures how far to trust it.
"""

from __future__ import annotations

import json
import re

import numpy as np

from vision_uss_research.label_kpi import box_iou

# Appearance prompts for SAM3 (the category itself is known per video).
SAM3_PROMPTS = {
    "isopole": "vertical pole", "car": "car", "wall": "wall",
    "dummyadult": "mannequin", "dummychild": "child mannequin",
    "box30": "white box", "box40": "white box", "curbhigh": "curb",
    "woodencurb10": "wooden beam lying on the ground",
    "metalbar": "metal bar lying on the ground",
}

# Plain-language names for Qwen.
QWEN_NAMES = {
    "isopole": "the blue vertical test pole", "car": "the white car (or car-shaped dummy)",
    "wall": "the wall", "dummyadult": "the adult-size mannequin (pedestrian dummy)",
    "dummychild": "the child-size mannequin (pedestrian dummy)",
    "box30": "the white box on the ground", "box40": "the white box on the ground",
    "curbhigh": "the concrete curb (wheel stop) on the ground",
    "woodencurb10": "the wooden beam lying across the ground",
    "metalbar": "the thin metal bar lying across the ground",
}


def qwen_prompt(category: str) -> str:
    name = QWEN_NAMES.get(category, category)
    return (
        f"This image is from a camera mounted low on a vehicle bumper. Locate {name} "
        "that the vehicle is approaching. Ignore other objects in the background.\n"
        'Answer with JSON only: {"visible": true, "bbox_2d": [x1, y1, x2, y2]} '
        "for a tight box around the visible part, or "
        '{"visible": false} if it is not in the image.'
    )


def parse_qwen(text: str, resized_wh: tuple[int, int],
               orig_wh: tuple[int, int]) -> tuple[bool | None, np.ndarray | None]:
    """(visible, box in original pixels) from Qwen's reply; (None, None) when
    the reply cannot be read. Qwen2.5-VL answers in the pixel space of the
    image it was fed (resized to multiples of 28), so boxes are scaled back."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None, None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, None
    if isinstance(obj, list):
        obj = obj[0] if obj else {}
    vis = obj.get("visible")
    if vis is False:
        return False, None
    box = obj.get("bbox_2d")
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        return (None, None) if vis is None else (bool(vis), None)
    sx, sy = orig_wh[0] / resized_wh[0], orig_wh[1] / resized_wh[1]
    b = np.array([box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy], float)
    b = np.clip(b, 0, [orig_wh[0] - 1, orig_wh[1] - 1, orig_wh[0] - 1, orig_wh[1] - 1])
    if b[2] - b[0] < 2 or b[3] - b[1] < 2:
        return None, None
    return True, b


def combine(sam_box, qwen_visible, qwen_box, img_wh: tuple[int, int],
            agree_iou: float = 0.5, edge: float = 2.0) -> dict:
    """Silver label for one frame. Both boxes overlapping -> 'box' (SAM3's
    box: mask-tight); SAM3 nothing and Qwen 'not visible' -> 'not_visible';
    anything else -> 'disputed' (with the reason)."""
    out = {"status": "disputed", "x0": np.nan, "y0": np.nan, "x1": np.nan,
           "y1": np.nan, "truncated": 0, "reason": ""}
    has_sam = sam_box is not None and not np.any(np.isnan(np.asarray(sam_box, float)))
    if has_sam and qwen_box is not None:
        iou = box_iou(np.asarray(sam_box, float), np.asarray(qwen_box, float))
        out["iou"] = round(iou, 3)
        if iou >= agree_iou:
            x0, y0, x1, y1 = (float(v) for v in sam_box)
            w, h = img_wh
            cut = x0 <= edge or y0 <= edge or x1 >= w - 1 - edge or y1 >= h - 1 - edge
            out.update(status="box", x0=x0, y0=y0, x1=x1, y1=y1, truncated=int(cut))
        else:
            out["reason"] = "boxes differ"
    elif not has_sam and qwen_visible is False:
        out["status"] = "not_visible"
    elif has_sam:
        out["reason"] = "qwen unreadable" if qwen_visible is None else "only sam3 sees it"
    else:
        out["reason"] = "only qwen sees it" if qwen_visible else "qwen unreadable"
    return out


def select_test_items(frames, split_cfg: dict, step: int = 10):
    """The rating frames: every `step`-th frame of each held-out test trace's
    final approach (bundle frames.csv rows). Shared by the human review shards
    and the silver ground truth so both label exactly the same frames."""
    import pandas as pd

    from vision_uss_research.cuv_finetune import final_approach, split_traces

    meta = (frames.groupby(["side", "trace"])
            .agg(label_class=("label_class", "first"), speed=("speed", "first"))
            .reset_index())
    split = split_traces(meta, split_cfg["n_test_per_group"],
                         split_cfg["n_val_per_group"], split_cfg["seed"])
    test = frames[[split[(s, int(t))] == "test" for s, t in zip(frames["side"], frames["trace"])]]
    test = pd.concat([final_approach(g) for _, g in test.groupby(["side", "trace"])])
    test = test.sort_values(["side", "trace", "frame_idx"])
    items = test[test.groupby(["side", "trace"]).cumcount() % step == 0].reset_index(drop=True)
    items["item_id"] = (items["side"] + "_" + items["trace"].astype(str) + "_"
                        + items["frame_idx"].map(lambda f: f"{int(f):05d}"))
    return items
