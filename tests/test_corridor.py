import numpy as np

from vision_uss_research.labeling.corridor import (
    corridor_weight, masks_to_bbox, pick_target_instance,
)


def test_central_low_candidate_beats_higher_scoring_edge_candidate():
    result = {
        "masks": np.zeros((2, 640, 960), bool),
        "boxes": np.array([[20.0, 400, 120, 500],     # far left, higher score
                           [430.0, 400, 530, 500]]),  # dead center
        "scores": np.array([0.9, 0.7]),
    }
    best, diag = pick_target_instance(result, (640, 960))
    assert best == 1
    assert diag["n_candidates"] == 2


def test_above_horizon_penalized():
    assert corridor_weight(0.5, 0.1) < corridor_weight(0.5, 0.6)


def test_empty_result():
    empty = {"masks": np.zeros((0, 1, 1), bool),
             "boxes": np.zeros((0, 4)), "scores": np.array([])}
    best, diag = pick_target_instance(empty, (640, 960))
    assert best is None and diag["n_candidates"] == 0


def test_masks_to_bbox():
    m = np.zeros((10, 10), bool)
    m[2:5, 3:8] = True
    assert masks_to_bbox(m) == (3, 2, 7, 4)
    assert masks_to_bbox(np.zeros((10, 10), bool)) is None
