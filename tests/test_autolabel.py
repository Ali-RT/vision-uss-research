import numpy as np
import pytest

from vision_uss_research.autolabel.geometry import (box_metrics, centroid,
                                                    matched_corner_error,
                                                    nn_corner_error, oriented_box,
                                                    polygon_area, polygon_iou,
                                                    vehicle_to_world)
from vision_uss_research.autolabel.uss_box import (box_behind_face, box_v1,
                                                   build_box, extents_along,
                                                   oracle_size_iou,
                                                   parse_label_targets,
                                                   polygon_from_str, polygon_to_str,
                                                   select_slot_v1, select_slot_v2)

SQUARE = [(0, 0), (1, 0), (1, 1), (0, 1)]


def test_iou_basics():
    assert polygon_area(SQUARE) == pytest.approx(1.0)
    assert polygon_iou(SQUARE, SQUARE) == pytest.approx(1.0)
    assert polygon_iou(SQUARE, [(2, 2), (3, 2), (3, 3), (2, 3)]) == 0.0
    # half-shifted square: overlap 0.5, union 1.5
    shifted = [(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)]
    assert polygon_iou(SQUARE, shifted) == pytest.approx(1 / 3)
    # vertex order / orientation does not matter
    assert polygon_iou(SQUARE[::-1], shifted) == pytest.approx(1 / 3)


def test_iou_concave_subject_convex_clip():
    # L-shape (area 3) clipped by the unit square at its corner -> overlap 1
    ell = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
    assert polygon_area(ell) == pytest.approx(3.0)
    assert polygon_iou(ell, SQUARE) == pytest.approx(1 / 3)


def test_oriented_box_rotation_and_size():
    b = oriented_box(10.0, 5.0, np.pi / 2, length=0.75, width=1.75)
    assert polygon_area(b) == pytest.approx(0.75 * 1.75)
    assert centroid(b) == pytest.approx((10.0, 5.0))
    xs, ys = zip(*b)
    # heading +90deg: length now runs along y, width along x
    assert max(ys) - min(ys) == pytest.approx(0.75)
    assert max(xs) - min(xs) == pytest.approx(1.75)


def test_vehicle_to_world():
    assert vehicle_to_world(1.0, 2.0, 0.0, 3.0, 1.0) == pytest.approx((4.0, 3.0))
    assert vehicle_to_world(0.0, 0.0, np.pi / 2, 1.0, 0.0) == pytest.approx((0.0, 1.0))


def test_corner_errors_nn_is_optimistic():
    gt = SQUARE
    # two predicted corners collapse onto one GT corner: NN hides it
    pred = [(0, 0), (0.01, 0), (1, 1), (0, 1)]
    assert nn_corner_error(pred, gt) < matched_corner_error(pred, gt)
    assert matched_corner_error(SQUARE, SQUARE[1:] + SQUARE[:1]) == pytest.approx(0.0)
    assert matched_corner_error(SQUARE, SQUARE + [(0.5, 0.5)]) is None


def _slot(x_mm, y_mm=None, t=None):
    x = np.asarray(x_mm, float)
    y = np.zeros_like(x) if y_mm is None else np.asarray(y_mm, float)
    return {"t": np.arange(len(x), dtype=float) if t is None else np.asarray(t, float),
            "p1x": x, "p2x": x, "p1y": y, "p2y": y}


POSE = {"x": (np.arange(5.0), np.full(5, 100.0)),
        "y": (np.arange(5.0), np.full(5, 200.0)),
        "yaw": (np.arange(5.0), np.full(5, 90.0))}


def test_select_slot_v1_prefers_bumper_and_needs_approach():
    slots = {1: _slot([6000, 5000, 4300]),            # approaching, ends 100 mm from front bumper
             2: _slot([4200, 4200, 4200]),            # exactly at bumper but never approaches
             3: _slot([3000, 2000, 1000], [0, 0, 900])}
    assert select_slot_v1(slots)[0] == 1


def test_select_v1_scores_final_zero_sample_v2_skips_it():
    # slot 1 dropped out (final 0), slot 2 ends mid-range: v1 scores slot 1 on
    # the 0 -> |0 - (-1300)| = 1300, so slot 2 (|3000 - 4200| = 1200) wins;
    # v2 uses the last valid sample (4250) and picks slot 1.
    slots = {1: _slot([6000, 5000, 4250, 0]), 2: _slot([5000, 4000, 3500, 3000])}
    assert select_slot_v1(slots)[0] == 2
    assert select_slot_v2(slots)[0] == 1


def test_build_box_defaults_equal_prototype_v1():
    slot = _slot([6000, 5000, 4250, 0], [0, 0, 300, 0])
    assert np.allclose(build_box(POSE, slot)[0], box_v1(POSE, slot))
    # v1 ignores lateral offset; heading 90deg -> object 4.25 m along +y
    cx, cy = centroid(box_v1(POSE, slot))
    assert (cx, cy) == pytest.approx((100.0, 204.25))
    # lateral +1 moves the centre 0.3 m to the vehicle's left (-x at 90deg)
    cx2, _ = build_box(POSE, slot, lateral=+1)[1]
    assert cx2 == pytest.approx(99.7)


def test_seg_box_sized_by_segment():
    slot = {"t": np.arange(3.0), "p1x": np.array([5000., 4500, 4000]),
            "p2x": np.array([5000., 4500, 4000]), "p1y": np.array([-500., -500, -500]),
            "p2y": np.array([500., 500, 500])}
    corners, _ = build_box(POSE, slot, lateral=+1, size="seg")
    assert polygon_area(corners) == pytest.approx(1.0 * 0.30)


def test_oracle_size_iou_is_one_at_true_centre():
    gt = oriented_box(3.0, 4.0, 0.3, 1.2, 0.5)
    assert oracle_size_iou(gt, centroid(gt)) == pytest.approx(1.0)
    assert oracle_size_iou(gt, (3.0 + 10, 4.0)) == 0.0


def test_parse_label_targets_keeps_low_high_polygons(tmp_path):
    p = tmp_path / "x_PAS_m_label_V2.csv"
    p.write_text(
        '﻿Which object,"class (0=traversable, 1=low, 2=unknown, 3=high)",PolygonX,PolygonY\n'
        'bicyclestand,3,"1000,2000,2000,1000","0,0,1000,1000"\n'
        'empty,0,"0,1,1","0,0,1"\n'
        'curbstone,1,"",""\n'
        'curbstone,1,"0,1000,1000,0","0,0,500,500"\n', encoding="utf-8")
    t = parse_label_targets(p)
    assert [x["object"] for x in t] == ["bicyclestand", "curbstone"]
    assert t[0]["height_bin"] == "high" and t[1]["height_bin"] == "low"
    assert polygon_area(t[0]["polygon"]) == pytest.approx(1.0)       # mm -> m


def test_box_metrics_keys():
    m = box_metrics(SQUARE, SQUARE)
    assert m["iou"] == 1.0 and m["centroid_err_m"] == 0.0 and m["pred_area_m2"] == 1.0


def test_depth_shift_puts_near_edge_on_face():
    # heading +90deg, object ahead: box centre moves +length/2 along +y
    corners, (cx, cy) = box_behind_face((0.0, 10.0), np.pi / 2, +1, 2.0, 1.0)
    assert (cx, cy) == pytest.approx((0.0, 11.0))
    assert min(y for _, y in corners) == pytest.approx(10.0)     # near edge on the face
    # object behind the reference point: shift the other way
    _, (_, cy2) = box_behind_face((0.0, -10.0), np.pi / 2, -1, 2.0, 1.0)
    assert cy2 == pytest.approx(-11.0)
    slot = _slot([6000, 5000, 4250, 0])
    shifted = build_box(POSE, slot, lateral=+1, depth_shift=True)[1]
    assert shifted[1] == pytest.approx(204.25 + 0.75 / 2)


def test_extents_and_polygon_roundtrip():
    b = oriented_box(0.0, 0.0, 0.4, 3.0, 1.0)
    assert extents_along(b, 0.4) == pytest.approx((3.0, 1.0))
    assert extents_along(b, 0.4 + np.pi / 2) == pytest.approx((1.0, 3.0))
    assert polygon_from_str(polygon_to_str(SQUARE)) == [tuple(map(float, v)) for v in SQUARE]
