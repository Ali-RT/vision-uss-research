import numpy as np

from vision_uss_research.alignment.height import (estimate_focal,
                                                  estimate_height_mm,
                                                  height_error,
                                                  low_high_from_height)


def test_estimate_focal_recovers_ground_truth():
    # synthesize a perfect pinhole: h = f*H/Z with f=800 px
    f_true = 800.0
    H = np.array([1250.0, 1250.0, 1080.0, 1080.0])   # dummy, pole heights (mm)
    Z = np.array([2000.0, 1500.0, 2500.0, 1000.0])   # distances (mm)
    h = f_true * H / Z
    assert abs(estimate_focal(h, Z, H) - f_true) < 1e-6


def test_estimate_focal_is_robust_to_outliers():
    f_true = 800.0
    H = np.full(11, 1250.0)
    Z = np.full(11, 2000.0)
    h = f_true * H / Z
    h[0] = h[0] * 5      # one gross outlier (bad box)
    # median absorbs it; a mean would not
    assert abs(estimate_focal(h, Z, H) - f_true) < 1.0


def test_estimate_focal_drops_bad_rows():
    assert np.isnan(estimate_focal([np.nan, 0, -3], [1, 2, 3], [1, 2, 3]))
    # one valid row among junk still yields it
    assert abs(estimate_focal([np.nan, 100.0], [np.nan, 2000.0],
                              [np.nan, 1250.0]) - 100.0 * 2000 / 1250) < 1e-6


def test_estimate_height_roundtrips_with_focal():
    f = 800.0
    # a 300 mm object at 1500 mm projects to h = 800*300/1500 = 160 px
    h_px = np.array([160.0])
    z = np.array([1500.0])
    est = estimate_height_mm(h_px, z, f)
    assert abs(est[0] - 300.0) < 1e-6
    # missing distance -> NaN, no crash
    assert np.isnan(estimate_height_mm([160.0], [np.nan], f)[0])


def test_height_error_metrics():
    err = height_error([100.0, 200.0, 300.0], [110.0, 180.0, 300.0])
    assert err["n"] == 3
    assert abs(err["mae_mm"] - (10 + 20 + 0) / 3) < 1e-6
    assert err["mean_rel_err"] > 0
    # all-NaN -> graceful
    empty = height_error([np.nan], [np.nan])
    assert empty["n"] == 0 and empty["mae_mm"] is None


def test_low_high_from_height():
    out = low_high_from_height([90.0, 250.0, 300.0, np.nan])
    assert list(out[:3]) == ["low", "high", "high"]   # <250 low, >=250 high
    assert out[3] is None
