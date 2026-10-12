"""Online detect + track auto-labeler (the labeling service's core loop).

Per frame, in order: the detector proposes boxes, BoT-SORT (Ultralytics,
with camera-motion compensation) updates its tracks with them, and the
target track - the staged object of the job's known category - is chosen or
kept. When the target goes unmatched, the tracker's motion prediction carries
its box for up to `max_predict` frames. Tracking is part of the loop, not a
cleanup pass: nothing here looks at future frames.

Per frame the labeler emits the target box, its track id, the source
("det" = matched to a detection this frame, "pred" = motion prediction) and
the raw features a label-confidence model needs (detector score, track age,
hit streak, frames since last match, competing tracks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vision_uss_research.cuv_finetune import LOW_PROFILE
from vision_uss_research.labeling.corridor import corridor_weight

BOTSORT_DEFAULTS = {
    "tracker_type": "botsort", "track_high_thresh": 0.25, "track_low_thresh": 0.1,
    "new_track_thresh": 0.25, "track_buffer": 30, "match_thresh": 0.8,
    "fuse_score": True, "gmc_method": "sparseOptFlow", "proximity_thresh": 0.5,
    "appearance_thresh": 0.8, "with_reid": False, "model": "auto",
}


@dataclass
class TrackView:
    """One tracker track as the target selector sees it."""
    track_id: int
    box: np.ndarray          # xyxy
    score: float
    cls: str
    matched: bool            # matched to a detection this frame


@dataclass
class TargetState:
    track_id: int | None = None
    first_frame: int = 0
    hits: int = 0            # frames matched to a detection since adoption
    streak: int = 0          # consecutive matched frames
    since_match: int = 0     # frames since the last match
    history: list = field(default_factory=list)


def target_score(t: TrackView, w: int, h: int) -> float:
    cx, cy = (t.box[0] + t.box[2]) / 2 / w, (t.box[1] + t.box[3]) / 2 / h
    return t.score * corridor_weight(cx, cy)


def select_target(tracks: list[TrackView], state: TargetState, wanted: set[str],
                  w: int, h: int, max_predict: int = 15,
                  min_score: float = 0.05) -> TrackView | None:
    """Pick the frame's target track. A matched current target is kept. Else a
    detection of the job's category wins over any motion prediction (one
    staged object per job: a fresh detection beats a drifting prediction,
    e.g. when the object's track id changed). Only with no such detection is
    the current target carried by prediction, for at most max_predict frames."""
    by_id = {t.track_id: t for t in tracks}
    cur = by_id.get(state.track_id) if state.track_id is not None else None
    if cur is not None and cur.matched:
        return cur
    cands = [t for t in tracks if t.matched and t.cls in wanted
             and target_score(t, w, h) >= min_score]
    if cands:
        return max(cands, key=lambda t: target_score(t, w, h))
    if cur is not None and state.since_match < max_predict:
        return cur
    return None


class ExitGuard:
    """Online bottom-edge exit rule for low-profile objects (curbs, bars,
    boxes): their box top moves down while the vehicle approaches. Once a
    bottom-touching box's top jumps back up, the object has left the image and
    later bottom-touching boxes are false (the asphalt strip behind a curb),
    until a box is fully in view again. A motion prediction is never trusted
    for an object whose last matched box already touched the bottom edge."""

    def __init__(self, img_h: int, tol: float = 4.0, edge: float = 2.0):
        self.img_h, self.tol, self.edge = img_h, tol, edge
        self.last_y0, self.exited, self.last_match_at_bottom = None, False, False

    def keep(self, box, matched: bool) -> bool:
        y0, y1 = float(box[1]), float(box[3])
        at_bottom, at_top = y1 >= self.img_h - self.edge, y0 <= self.edge
        if not matched:
            return not self.last_match_at_bottom and not self.exited
        self.last_match_at_bottom = at_bottom and not at_top
        if not at_bottom or at_top:
            self.last_y0, self.exited = None, False
            return True
        if self.exited:
            return False
        if self.last_y0 is None or y0 >= self.last_y0 - self.tol:
            self.last_y0 = y0
            return True
        self.exited = True
        return False


class OnlineLabeler:
    """Detector + BoT-SORT + target selection for one video (one job)."""

    def __init__(self, model, wanted_classes: set[str], imgsz: int = 480,
                 det_conf: float = 0.05, device=None, max_predict: int = 15,
                 tracker_cfg: dict | None = None, category: str | None = None):
        from ultralytics.trackers.bot_sort import BOTSORT
        from ultralytics.utils import IterableSimpleNamespace

        self.model, self.wanted = model, set(wanted_classes)
        self.imgsz, self.det_conf, self.device = imgsz, det_conf, device
        self.max_predict = max_predict
        args = IterableSimpleNamespace(**{**BOTSORT_DEFAULTS, **(tracker_cfg or {})})
        self.tracker = BOTSORT(args)   # track_buffer is in frames (30 = 1 s)
        self.state = TargetState()
        self.frame_idx = -1
        self.low_profile = category in LOW_PROFILE
        self.guard = None

    def step(self, frame: np.ndarray, frame_idx: int | None = None) -> dict:
        self.frame_idx = self.frame_idx + 1 if frame_idx is None else frame_idx
        h, w = frame.shape[:2]
        res = self.model.predict(frame, imgsz=self.imgsz, conf=self.det_conf,
                                 device=self.device, verbose=False)[0]
        names = res.names
        out = self.tracker.update(res.boxes.cpu().numpy(), frame)
        tracks = [TrackView(int(r[4]), np.asarray(r[:4], float), float(r[5]),
                            names[int(r[6])], True) for r in out]
        matched_ids = {t.track_id for t in tracks}
        for s in self.tracker.lost_stracks:          # motion-predicted this frame
            if s.track_id not in matched_ids:
                tracks.append(TrackView(int(s.track_id), np.asarray(s.xyxy, float),
                                        float(s.score), names[int(s.cls)], False))

        st = self.state
        tgt = select_target(tracks, st, self.wanted, w, h, self.max_predict)
        if tgt is not None and self.low_profile:
            self.guard = self.guard or ExitGuard(h)
            if not self.guard.keep(tgt.box, tgt.matched):
                tgt = None
        if tgt is None:
            st.track_id, st.streak = None, 0
            st.since_match += 1
            return {"frame_idx": self.frame_idx, "track_id": None, "src": None,
                    "n_tracks": len(tracks), "since_match": st.since_match}
        if tgt.track_id != st.track_id:
            st.track_id, st.first_frame, st.hits, st.streak = tgt.track_id, self.frame_idx, 0, 0
        if tgt.matched:
            st.hits, st.streak, st.since_match = st.hits + 1, st.streak + 1, 0
        else:
            st.streak, st.since_match = 0, st.since_match + 1
        box = np.clip(tgt.box, [0, 0, 0, 0], [w - 1, h - 1, w - 1, h - 1])
        return {"frame_idx": self.frame_idx, "track_id": tgt.track_id,
                "src": "det" if tgt.matched else "pred",
                "x0": float(box[0]), "y0": float(box[1]),
                "x1": float(box[2]), "y1": float(box[3]),
                "det_cls": tgt.cls, "det_score": round(tgt.score, 4),
                "track_age": self.frame_idx - st.first_frame + 1,
                "hits": st.hits, "streak": st.streak, "since_match": st.since_match,
                "n_tracks": len(tracks),
                "n_wanted": sum(t.matched and t.cls in self.wanted for t in tracks)}


def run_video(labeler: OnlineLabeler, frame_paths: list[Path],
              frame_idxs: list[int]) -> list[dict]:
    import cv2
    rows = []
    for p, f in zip(frame_paths, frame_idxs):
        rows.append(labeler.step(cv2.imread(str(p)), int(f)))
    return rows
