"""Bird's-eye-view box geometry for the USS auto-labeler evaluation.

Pure numpy, no MF4 access - unit-testable. All coordinates are world metres
(the frame of `VHM_VehicleState_XPos/YPos` and of the Label V2 PolygonX/Y
columns after /1000). Polygons are lists of (x, y) vertices.

IoU uses Sutherland-Hodgman clipping, which needs a CONVEX clip polygon. Every
predicted box here is a rectangle, so `polygon_iou(gt, pred)` clips the (maybe
concave) human polygon by the convex prediction - exact for our use.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np


def signed_area(poly) -> float:
    p = np.asarray(poly, dtype=float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_area(poly) -> float:
    return abs(signed_area(poly)) if len(poly) >= 3 else 0.0


def centroid(poly) -> tuple[float, float]:
    """Area centroid; falls back to the vertex mean for degenerate polygons."""
    p = np.asarray(poly, dtype=float)
    a = signed_area(p)
    if abs(a) < 1e-12:
        return float(p[:, 0].mean()), float(p[:, 1].mean())
    x, y = p[:, 0], p[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    return (float(np.sum((x + xn) * cross) / (6 * a)),
            float(np.sum((y + yn) * cross) / (6 * a)))


def _ccw(poly) -> list:
    p = [tuple(map(float, v)) for v in poly]
    return p if signed_area(p) > 0 else p[::-1]


def clip_area(subject, convex_clip) -> float:
    """Area of subject ∩ convex_clip (Sutherland-Hodgman)."""
    out = _ccw(subject)
    clip = _ccw(convex_clip)

    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def intersect(p, q, a, b):
        x1, y1, x2, y2 = *p, *q
        x3, y3, x4, y4 = *a, *b
        d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    for i in range(len(clip)):
        a, b = clip[i - 1], clip[i]
        inp, out = out, []
        for j in range(len(inp)):
            p, q = inp[j - 1], inp[j]
            if inside(q, a, b):
                if not inside(p, a, b):
                    out.append(intersect(p, q, a, b))
                out.append(q)
            elif inside(p, a, b):
                out.append(intersect(p, q, a, b))
        if not out:
            return 0.0
    return polygon_area(out) if len(out) >= 3 else 0.0


def polygon_iou(subject, convex_clip) -> float:
    inter = clip_area(subject, convex_clip)
    union = polygon_area(subject) + polygon_area(convex_clip) - inter
    return float(inter / union) if union > 0 else 0.0


def oriented_box(cx: float, cy: float, heading_rad: float,
                 length: float, width: float) -> list[tuple[float, float]]:
    """Rectangle centred at (cx, cy); `length` runs along `heading_rad`,
    `width` across it. Corner order matches the original auto-labeler
    (bottom-left, bottom-right, top-right, top-left in the heading frame)."""
    c, s = np.cos(heading_rad), np.sin(heading_rad)
    h, w = length / 2, width / 2
    return [(cx - h * c + w * s, cy - h * s - w * c),
            (cx + h * c + w * s, cy + h * s - w * c),
            (cx + h * c - w * s, cy + h * s + w * c),
            (cx - h * c - w * s, cy - h * s + w * c)]


def vehicle_to_world(x_m: float, y_m: float, yaw_rad: float,
                     dx_m: float, dy_m: float) -> tuple[float, float]:
    """Vehicle-frame offset (dx forward, dy left) -> world, given the ego pose."""
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return x_m + dx_m * c - dy_m * s, y_m + dx_m * s + dy_m * c


def nn_corner_error(pred, gt) -> float:
    """The ORIGINAL metric (helpers_mf4.compare_coordinates): each predicted
    corner to its nearest GT corner, NOT one-to-one, averaged. Optimistic -
    several predicted corners may share one GT corner. Kept only so the as-run
    numbers reproduce."""
    g = np.asarray(gt, dtype=float)
    return float(np.mean([np.min(np.hypot(*(g - np.asarray(p)).T)) for p in pred]))


def matched_corner_error(pred, gt) -> float | None:
    """One-to-one corner error: mean distance under the best assignment of
    predicted to GT corners. Defined only for equal vertex counts <= 6."""
    if len(pred) != len(gt) or len(gt) > 6:
        return None
    p, g = np.asarray(pred, dtype=float), np.asarray(gt, dtype=float)
    return float(min(np.mean(np.hypot(*(p - g[list(perm)]).T))
                     for perm in permutations(range(len(g)))))


def centroid_error(pred, gt) -> float:
    """Size-independent localisation error: distance between area centroids."""
    (px, py), (gx, gy) = centroid(pred), centroid(gt)
    return float(np.hypot(px - gx, py - gy))


def box_metrics(pred, gt) -> dict:
    return {
        "iou": round(polygon_iou(gt, pred), 4),
        "corner_err_nn_m": round(nn_corner_error(pred, gt), 4),
        "corner_err_matched_m": (None if (m := matched_corner_error(pred, gt)) is None
                                 else round(m, 4)),
        "centroid_err_m": round(centroid_error(pred, gt), 4),
        "pred_area_m2": round(polygon_area(pred), 4),
    }
