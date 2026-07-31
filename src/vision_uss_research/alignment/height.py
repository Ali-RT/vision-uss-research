"""Geometric object height from a bounding box + USS distance (Track B).

Pinhole model: an object of real height H at distance Z projects to a pixel
height h = f * H / Z, where f is the focal length in pixels. So:

    focal f  = median( h_px * Z / H )   over known-height calibration objects
    height H = h_px * Z / f             for any detection

Distance Z and pixel height h are per frame; f is one constant per camera.
Pure numpy - no MF4/asammdf dependency, so this is unit-testable in CI.
"""

from __future__ import annotations

import numpy as np

# real heights (mm) of the calibration classes - single rigid objects whose
# height is known and constant across the campaign
KNOWN_HEIGHT_MM = {
    "dummychild": 1250.0,   # NCAP child dummy, 125 cm
    "pole": 1080.0,         # ISO test pole, 108 cm
}

LOW_HIGH_THRESHOLD_MM = 250.0   # < 25 cm = low (drive-over-relevant)


def estimate_focal(box_px_heights, distances_mm, real_heights_mm) -> float:
    """Robust focal length (pixels) from calibration triples: the MEDIAN of
    h_px * Z / H, which shrugs off the outliers that box/distance noise
    produce. Inputs are array-likes of equal length; non-finite or
    non-positive rows are dropped. NaN if nothing usable remains."""
    h = np.asarray(box_px_heights, dtype=float)
    z = np.asarray(distances_mm, dtype=float)
    H = np.asarray(real_heights_mm, dtype=float)
    good = np.isfinite(h) & np.isfinite(z) & np.isfinite(H) & (h > 0) & (z > 0) & (H > 0)
    if good.sum() == 0:
        return float("nan")
    return float(np.median(h[good] * z[good] / H[good]))


def estimate_height_mm(box_px_height, distance_mm, focal_px) -> np.ndarray:
    """height = h_px * Z / f. Elementwise; NaN where inputs are missing."""
    h = np.asarray(box_px_height, dtype=float)
    z = np.asarray(distance_mm, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = h * z / float(focal_px)
    out = np.asarray(out, dtype=float)
    out[~np.isfinite(out)] = np.nan
    return out


def height_error(pred_mm, true_mm) -> dict:
    """MAE, RMSE and mean relative error (pred vs known height), NaNs dropped."""
    p = np.asarray(pred_mm, dtype=float)
    t = np.asarray(true_mm, dtype=float)
    good = np.isfinite(p) & np.isfinite(t) & (t > 0)
    if good.sum() == 0:
        return {"n": 0, "mae_mm": None, "rmse_mm": None, "mean_rel_err": None}
    p, t = p[good], t[good]
    return {
        "n": int(good.sum()),
        "mae_mm": float(np.mean(np.abs(p - t))),
        "rmse_mm": float(np.sqrt(np.mean((p - t) ** 2))),
        "mean_rel_err": float(np.mean(np.abs(p - t) / t)),
    }


def low_high_from_height(height_mm, threshold_mm: float = LOW_HIGH_THRESHOLD_MM):
    """'low' / 'high' from a metric height; None where height is NaN."""
    h = np.asarray(height_mm, dtype=float)
    out = np.where(h < threshold_mm, "low", "high").astype(object)
    out[~np.isfinite(h)] = None
    return out


def roc_auc(scores, is_positive) -> float:
    """AUC with the convention: a HIGHER score means more likely positive.
    Rank-based (Mann-Whitney U), tie-aware. NaN if a class is empty."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(is_positive, dtype=int)
    good = np.isfinite(s)
    s, y = s[good], y[good]
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks within tied score groups
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def calibrate_low_threshold(height_mm, is_low) -> dict:
    """Choose the height threshold that best separates low from high: predict
    'low' when height < threshold, maximizing balanced accuracy (mean of low
    and high recall). Returns the threshold, its balanced accuracy, and the
    threshold-independent AUC (separability of the score). Fitting this on
    labeled data lets the biased-but-monotonic geometric height act as a
    class-agnostic Low/High score - useful for objects with no class label."""
    h = np.asarray(height_mm, dtype=float)
    y = np.asarray(is_low, dtype=int)
    good = np.isfinite(h)
    h, y = h[good], y[good]
    n_low, n_high = int(y.sum()), int((y == 0).sum())
    if n_low == 0 or n_high == 0:
        return {"threshold_mm": None, "balanced_accuracy": None, "auc": None,
                "n_low": n_low, "n_high": n_high}
    best_t, best_bacc = None, -1.0
    for t in np.unique(h):
        pred_low = h < t
        rl = pred_low[y == 1].mean()
        rh = (~pred_low[y == 0]).mean()
        bacc = (rl + rh) / 2
        if bacc > best_bacc:
            best_bacc, best_t = bacc, float(t)
    # lower height -> more likely low, so feed -h as the "higher=positive" score
    return {"threshold_mm": best_t, "balanced_accuracy": float(best_bacc),
            "auc": roc_auc(-h, y), "n_low": n_low, "n_high": n_high}
