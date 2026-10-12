import numpy as np

from vision_uss_research.online_labeler import (ExitGuard, TargetState, TrackView,
                                                 select_target)


def tv(i, box, score=0.8, cls="pole", matched=True):
    return TrackView(i, np.array(box, float), score, cls, matched)


def test_adopts_best_wanted_track_in_corridor():
    tracks = [tv(1, [0, 0, 60, 40], 0.9, "pole"),          # top-left corner
              tv(2, [300, 300, 340, 380], 0.6, "pole"),    # ahead of the bumper
              tv(3, [300, 300, 340, 380], 0.95, "car")]    # not the job's class
    assert select_target(tracks, TargetState(), {"pole"}, 640, 480).track_id == 2


def test_prediction_only_when_nothing_detected():
    state = TargetState(track_id=2, since_match=3)
    pred = tv(2, [300, 300, 340, 380], matched=False)
    assert select_target([pred], state, {"pole"}, 640, 480, max_predict=15).track_id == 2
    state.since_match = 15
    assert select_target([pred], state, {"pole"}, 640, 480, max_predict=15) is None


def test_detection_beats_prediction():
    state = TargetState(track_id=2, since_match=1)
    pred = tv(2, [250, 300, 290, 380], matched=False)          # drifting prediction
    fresh = tv(5, [310, 300, 350, 380], 0.8)                    # same pole, new id
    assert select_target([pred, fresh], state, {"pole"}, 640, 480).track_id == 5


def test_never_adopts_unmatched_or_unwanted():
    tracks = [tv(1, [300, 300, 340, 380], matched=False), tv(2, [300, 300, 340, 380], cls="car")]
    assert select_target(tracks, TargetState(), {"pole"}, 640, 480) is None


def test_exit_guard_stops_after_object_leaves_bottom():
    g = ExitGuard(img_h=480)
    assert g.keep([0, 300, 640, 360], True)          # fully in view
    assert g.keep([0, 440, 640, 480], True)          # sliding out, top moving down
    assert not g.keep([0, 470, 640, 480], False)     # no prediction once at the edge
    assert not g.keep([0, 430, 640, 480], True)      # top jumped up: asphalt strip
    assert not g.keep([0, 435, 640, 480], True)      # stays exited
    assert g.keep([0, 200, 640, 300], True)          # fully in view again resets


def test_exit_guard_allows_prediction_mid_image():
    g = ExitGuard(img_h=480)
    assert g.keep([100, 200, 200, 300], True)
    assert g.keep([102, 204, 202, 306], False)
