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


def test_distance_at_does_not_clamp_outside_readings():
    """Frames before the object enters USS range must be NaN, never a
    fabricated constant - a fake distance would corrupt height estimates."""
    t = np.array([0.0, 1.0, 2.0, 3.0])
    # valid readings only in the second half of the recording
    v = np.array([np.nan, np.nan, 120.0, 80.0])
    assert np.isnan(distance_at(0.0, t, v))     # before first reading
    assert np.isnan(distance_at(1.5, t, v))     # still before
    assert np.isclose(distance_at(2.5, t, v), 100.0)   # inside -> interpolated
    assert np.isnan(distance_at(9.0, t, v))     # after last reading


def test_pts_match_error_detects_frame_count_mismatch():
    pts = np.array([0.0, 0.396, 1.189])
    assert pts_match_error(pts, pts) == 0.0
    assert pts_match_error(pts + 1e-4, pts) < 1e-3
    # differing counts mean the pairing assumption is broken -> None
    assert pts_match_error(pts[:2], pts) is None
    assert pts_match_error(np.array([]), np.array([])) is None


def test_camera_for_direction():
    from vision_uss_research.alignment.mf4 import camera_for_direction
    assert camera_for_direction("forward") == "front"
    assert camera_for_direction("backward") == "rear"
    assert camera_for_direction("Backward") == "rear"      # case tolerant
    assert camera_for_direction("unknown") is None
    assert camera_for_direction("") is None


def test_approach_monotonicity():
    from vision_uss_research.alignment.mf4 import approach_monotonicity
    # clean approach: every step decreases
    assert approach_monotonicity(np.array([300.0, 250, 200, 150, 100])) == 1.0
    # receding
    assert approach_monotonicity(np.array([100.0, 150, 200])) == 0.0
    # half noise
    assert approach_monotonicity(np.array([200.0, 150, 180, 120])) == 2 / 3
    # NaNs are ignored, not counted as steps
    assert approach_monotonicity(np.array([300.0, np.nan, 200, 100])) == 1.0
    # too few readings / constant -> None
    assert approach_monotonicity(np.array([100.0, 100])) is None
    assert approach_monotonicity(np.array([5.0, 5, 5, 5])) is None


def test_nearest_obstacle_distance_takes_min_across_zones(monkeypatch):
    """min across direction zones = distance to nearest obstacle ahead."""
    import vision_uss_research.alignment.mf4 as m

    class FakeSig:
        def __init__(self, t, v):
            self.timestamps, self.samples = np.array(t, float), np.array(v, float)

    zones = {
        "sigPDC_Zone13_Distance": FakeSig([0, 1, 2], [200.0, 180.0, 4095.0]),
        "sigPDC_Zone14_Distance": FakeSig([0, 1, 2], [4095.0, 150.0, 90.0]),
    }

    class FakeMDF:
        groups = []
        def get(self, ch):
            return zones[ch]

    t, nearest, used = m.nearest_obstacle_distance(
        FakeMDF(), "rear", names=set(zones))
    assert used == ["sigPDC_Zone13_Distance", "sigPDC_Zone14_Distance"]
    # t=0: only zone13 valid -> 200; t=1: min(180,150)=150; t=2: only zone14 -> 90
    assert np.allclose(nearest, [200.0, 150.0, 90.0])
    # a direction with no populated zones returns empties, not a crash
    t2, n2, u2 = m.nearest_obstacle_distance(FakeMDF(), "front", names=set(zones))
    assert len(t2) == 0 and u2 == []
