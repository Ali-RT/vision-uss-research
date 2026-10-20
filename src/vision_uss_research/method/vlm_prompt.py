"""VLM-named prompts for SAM3: Qwen2.5-VL names (and boxes) the obstacle,
SAM3 segments it with that noun.

SAM3 responds to concrete nouns ("hedge", "bollard") but not to generic ones
("object", "obstacle" - notebook 22: no candidates on 86-98% of non-cone
frames). The VLM supplies the noun, so the pipeline stays class-agnostic.
Asked once on the USS-region crop and once on the full frame, the comparison
isolates what the USS region adds.
"""

from __future__ import annotations

import json
import re

import numpy as np

ARTICLES = {"a", "an", "the", "some", "this", "that"}


def name_prompt(view: str) -> str:
    """view: 'crop' (USS-region crop) or 'full' (whole frame, no USS)."""
    where = ("This crop is centred where the vehicle's ultrasonic sensors detect "
             "something. " if view == "crop" else "")
    return (
        "This image is from a camera mounted on a vehicle that is slowly approaching "
        f"an obstacle while parking. {where}Name the main physical obstacle the "
        "vehicle is approaching, as a short noun phrase of 1-3 words (for example "
        '"hedge", "traffic cone", "parked car", "curb", "pole"), and give a tight box '
        "around it. Ignore the road surface, markings, sky and distant background.\n"
        'Answer with JSON only: {"name": "...", "bbox_2d": [x1, y1, x2, y2]}, or '
        '{"name": null} if there is no obstacle.'
    )


def clean_name(name) -> str | None:
    """Lower-case noun phrase for a SAM3 text prompt; None if unusable."""
    if not isinstance(name, str):
        return None
    words = [w for w in re.sub(r"[^a-z0-9 \-]", " ", name.lower()).split()
             if w not in ARTICLES]
    if not words:
        return None
    return " ".join(words[:4])


def parse_name_box(text: str, resized_wh: tuple[int, int],
                   orig_wh: tuple[int, int]) -> tuple[str | None, np.ndarray | None]:
    """(name, box in original pixels) from a Qwen reply. Qwen2.5-VL answers in
    the pixel space of the image it was fed (resized to multiples of 28), so
    boxes are scaled back. Either part may be None."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None, None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, None
    if isinstance(obj, list):
        obj = obj[0] if obj else {}
    name = clean_name(obj.get("name"))
    box = obj.get("bbox_2d")
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        return name, None
    sx, sy = orig_wh[0] / resized_wh[0], orig_wh[1] / resized_wh[1]
    b = np.array([box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy], float)
    b = np.clip(b, 0, [orig_wh[0] - 1, orig_wh[1] - 1, orig_wh[0] - 1, orig_wh[1] - 1])
    if b[2] - b[0] < 2 or b[3] - b[1] < 2:
        return name, None
    return name, b


def padded_crop(region, img_w: int, img_h: int, pad: float = 0.2,
                min_px: int = 16) -> tuple[int, ...] | None:
    """Integer crop window: the region grown by `pad` of its size, clipped.
    None when the region falls (almost) outside the image - a USS object
    projected off-frame gives no usable view, which counts as a USS miss."""
    x0, y0, x1, y1 = region
    if x1 - x0 < min_px or y1 - y0 < min_px:
        return None
    dw, dh = (x1 - x0) * pad, (y1 - y0) * pad
    c = (int(max(0, x0 - dw)), int(max(0, y0 - dh)),
         int(min(img_w, x1 + dw)), int(min(img_h, y1 + dh)))
    return c if c[2] - c[0] >= min_px and c[3] - c[1] >= min_px else None


def decision_prompt(view: str) -> str:
    """name_prompt plus the drive-over judgment, in one call."""
    where = ("This crop is centred where the vehicle's ultrasonic sensors detect "
             "something. " if view == "crop" else "")
    return (
        "This image is from a camera mounted on a vehicle that is slowly approaching "
        f"an obstacle while parking. {where}Name the main physical obstacle the "
        "vehicle is approaching, as a short noun phrase of 1-3 words, give a tight box "
        "around it, and say whether an ordinary passenger car could drive over it "
        "(low obstacles under about 25 cm, such as a curb, speed bump or board lying "
        "on the ground) or must stop (anything taller). Ignore the road surface, "
        "markings, sky and distant background.\n"
        'Answer with JSON only: {"name": "...", "bbox_2d": [x1, y1, x2, y2], '
        '"drive_over": true or false}, or {"name": null} if there is no obstacle.'
    )


def parse_drive_over(text: str) -> bool | None:
    """The drive_over flag from a Qwen reply; None if missing or unreadable."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(obj, list):
        obj = obj[0] if obj else {}
    v = obj.get("drive_over")
    return v if isinstance(v, bool) else None


# Words and phrases a VLM uses for drivable-over (< ~25 cm) things. Matched as
# whole words (a trailing plural "s"/"es" allowed), never as substrings:
# "trailer" must not match "rail", "wooden fence" must not match "wood".
LOW_WORDS = ("curb", "kerb", "speed bump", "bump", "board", "plank",
             "hose", "step", "grass", "marking", "line", "manhole", "drain", "mat",
             "wheel stop", "stopper", "pothole")
# Words whose height depends on context (guard rail vs. track rail, ceiling
# beam vs. beam on the ground, wooden fence vs. wooden board). A name whose
# only lexicon hit is one of these gets no bin (None): an explicit abstention,
# not a guess.
AMBIGUOUS_WORDS = ("rail", "beam", "wood", "wooden", "log")


def _has_word(name: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}(?:s|es)?\b", name) is not None


def name_to_bin(name) -> str | None:
    """Height bin from the VLM's noun (lexicon): 'low' | 'high' | None.

    'low' if any LOW_WORDS phrase occurs as a whole word; None if no low phrase
    occurs but an AMBIGUOUS_WORDS word does; otherwise 'high' (any other
    object noun). None also for an unusable name."""
    n = clean_name(name)
    if n is None:
        return None
    if any(_has_word(n, w) for w in LOW_WORDS):
        return "low"
    if any(_has_word(n, w) for w in AMBIGUOUS_WORDS):
        return None
    return "high"
