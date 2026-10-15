import numpy as np
import pytest

from vision_uss_research.method.uss_prompt import (GroundCamera, box_hit,
                                                   corridor_slot, fit_ground_camera,
                                                   range_ahead_mm, slots_at,
                                                   window_video_seconds)


def test_window_matches_extraction_rule():
    s = window_video_seconds(10.0)
    assert len(s) == 20 and s[0] == pytest.approx(3.0) and s[-1] == pytest.approx(8.5)


def _slot(t, p1x, p1y=-200.0, p2x=None, p2y=200.0):
    p2x = p1x if p2x is None else p2x
    x1, x2 = np.asarray(p1x, float), np.asarray(p2x, float)
    live = (x1 != 0) | (x2 != 0)          # a dead sample is all-zero in the MF4
    return {"t": np.asarray(t, float), "p1x": x1, "p2x": x2,
            "p1y": np.where(live, p1y, 0.0), "p2y": np.where(live, p2y, 0.0)}


def test_slots_at_nearest_live_sample_within_tolerance():
    slots = {1: _slot([0.0, 0.1, 0.2], [5000.0, 0.0, 4800.0]),   # middle sample dead
             2: _slot([5.0], [6000.0])}                            # too far in time
    rows = slots_at(slots, 0.11, max_dt=0.15)
    assert [r["slot"] for r in rows] == [1]
    assert rows[0]["mid_x"] == pytest.approx(4800.0)   # dead t=0.1 skipped; t=0.2 nearest live
    assert slots_at({3: _slot([0.0], [0.0], p1y=0, p2y=0)}, 0.0) == []


def test_corridor_slot_picks_nearest_ahead_inside_corridor():
    rows = [{"slot": 1, "mid_x": 6000.0, "mid_y": 0.0},
            {"slot": 2, "mid_x": 5000.0, "mid_y": 2500.0},     # outside corridor
            {"slot": 3, "mid_x": 5200.0, "mid_y": 300.0},      # nearest valid
            {"slot": 4, "mid_x": 3000.0, "mid_y": 0.0}]        # behind front bumper
    assert corridor_slot(rows, "front")["slot"] == 3
    rear_rows = [{"slot": 7, "mid_x": -2500.0, "mid_y": 0.0},
                 {"slot": 8, "mid_x": 5000.0, "mid_y": 0.0}]
    assert corridor_slot(rear_rows, "rear")["slot"] == 7
    assert range_ahead_mm(-2500.0, "rear") == pytest.approx(1200.0)


@pytest.mark.parametrize("camera", ["front", "rear"])
def test_fit_recovers_synthetic_camera(camera):
    true = GroundCamera.initial(camera).with_vector(
        GroundCamera.initial(camera).vector())
    true.f, true.h, true.pitch, true.yaw = 620.0, 650.0, np.radians(14), np.radians(2)
    true.u0, true.v0 = 470.0, 330.0
    rng = np.random.default_rng(0)
    base = true.xc
    ahead = rng.uniform(500, 5000, 400)
    x = base + true.facing * ahead
    y = rng.uniform(-1200, 1200, 400)
    u, v, ok = true.project(x, y)
    assert ok.all()
    u = u + rng.normal(0, 1.0, u.size)
    v = v + rng.normal(0, 1.0, v.size)
    # 10% gross outliers (wrong slot/box pairing) must not wreck the fit
    bad = rng.random(u.size) < 0.10
    u[bad] += rng.uniform(-300, 300, bad.sum())
    cam, info = fit_ground_camera(x, y, u, v, camera)
    pu, pv, _ = cam.project(x[~bad], y[~bad])
    true_u, true_v, _ = true.project(x[~bad], y[~bad])
    assert np.median(np.hypot(pu - true_u, pv - true_v)) < 3.0
    assert info["success"]


def test_box_hit_padding():
    assert box_hit(105, 210, 100, 100, 200, 200)                 # just below bottom edge
    assert not box_hit(50, 150, 100, 100, 200, 200)
    assert not box_hit(150, 260, 100, 100, 200, 200)             # far below
