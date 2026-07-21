import numpy as np

from vision_uss_research.alignment.mf4 import (distance_at, find_camera_channel,
                                               find_channel, mask_sentinel,
                                               pts_match_error, sorted_pairs,
                                               video_to_mf4)


def test_find_camera_channel_prefers_first_available():
    names = {"WebCam", "WebCam3", "Other"}
    assert find_camera_channel(names, "front") == "WebCam"
    assert find_camera_channel(names, "rear") == "WebCam3"
    assert find_camera_channel({"Other"}, "rear") is None
    # explicit candidate order is respected
    assert find_channel({"B", "A"}, ["A", "B"]) == "A"
    assert find_channel({"B"}, ["A", "B"]) == "B"


def test_video_to_mf4_maps_frames_onto_master_clock():
    # the sample's real pattern: video starts at 0, MF4 offset ~1.76s
    pts = np.array([0.0, 1.0, 2.0, 3.0])
    mf4 = np.array([1.76, 2.76, 3.76, 4.76])
    p, m = sorted_pairs(pts[::-1], mf4[::-1])   # unsorted input is handled
    assert np.allclose(p, pts) and np.allclose(m, mf4)
    assert np.isclose(video_to_mf4(0.0, p, m), 1.76)
    assert np.isclose(video_to_mf4(1.5, p, m), 3.26)      # interpolated
    assert np.allclose(video_to_mf4(np.array([0.0, 3.0]), p, m), [1.76, 4.76])


def test_mask_sentinel_pdc_and_objbuff():
    pdc = np.array([23.0, 4095.0, 100.0, 4095.0])
    masked = mask_sentinel(pdc, 4095.0, "ge")
    assert np.isnan(masked[1]) and np.isnan(masked[3])
    assert masked[0] == 23.0 and np.isfinite(masked).sum() == 2
    # ObjBuff uses 0 = no track (exact match, values above stay)
    ob = np.array([0.0, 1079.0, 0.0, 3600.0])
    m2 = mask_sentinel(ob, 0.0, "eq")
    assert np.isnan(m2[0]) and m2[3] == 3600.0
    # no sentinel -> untouched
    assert np.allclose(mask_sentinel(pdc, None), pdc)


def test_distance_at_skips_nan_readings():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    v = np.array([100.0, np.nan, 60.0, 40.0])
    # interpolates across the NaN gap using valid neighbours only
    assert np.isclose(distance_at(1.0, t, v), 80.0)
    assert np.isclose(distance_at(2.5, t, v), 50.0)
    # all-NaN signal -> NaN, not a crash
    assert np.isnan(distance_at(1.0, t, np.full(4, np.nan)))
    assert np.all(np.isnan(distance_at(np.array([0.0, 1.0]), t, np.full(4, np.nan))))


def test_pts_match_error_detects_frame_count_mismatch():
    pts = np.array([0.0, 0.396, 1.189])
    assert pts_match_error(pts, pts) == 0.0
    assert pts_match_error(pts + 1e-4, pts) < 1e-3
    # differing counts mean the pairing assumption is broken -> None
    assert pts_match_error(pts[:2], pts) is None
    assert pts_match_error(np.array([]), np.array([])) is None
