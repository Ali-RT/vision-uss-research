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
