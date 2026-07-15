"""Approach-corridor prior for picking the labeled target among distractors.

The staged target sits roughly ahead of the bumper: horizontally central and
below the horizon band. Tuned against the seed_log distributions (pole averages
~11 candidates per seed frame; speedbump ~1).
"""

from __future__ import annotations

import numpy as np


def corridor_weight(cx_norm: float, cy_norm: float,
                    sigma: float = 0.25, min_cy: float = 0.35) -> float:
    """Weight in [0, 1]: gaussian around image-center horizontally; hard penalty
    above the horizon band (cy_norm < min_cy)."""
    horizontal = float(np.exp(-((cx_norm - 0.5) ** 2) / (2 * sigma ** 2)))
    vertical = 1.0 if cy_norm >= min_cy else 0.25
    return horizontal * vertical


def pick_target_instance(result: dict, image_shape: tuple[int, int],
                         sigma: float = 0.25, min_cy: float = 0.35
                         ) -> tuple[int | None, dict]:
    """result: {'masks': (N,H,W) bool, 'boxes': (N,4) xyxy, 'scores': (N,)}.
    Returns (index of the best instance by score x corridor weight, diagnostics
    for seed_log)."""
    h, w = image_shape
    n = len(result["scores"])
    if n == 0:
        return None, {"n_candidates": 0}
    combined = []
    for i in range(n):
        x0, y0, x1, y1 = result["boxes"][i]
        cx_norm, cy_norm = ((x0 + x1) / 2) / w, ((y0 + y1) / 2) / h
        combined.append(float(result["scores"][i])
                        * corridor_weight(cx_norm, cy_norm, sigma, min_cy))
    best = int(np.argmax(combined))
    return best, {
        "n_candidates": n,
        "seed_score": round(float(result["scores"][best]), 3),
        "combined_score": round(combined[best], 3),
    }


def masks_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
