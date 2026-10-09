"""CUV_ObjectCheck fine-tune helpers (notebook 17).

Pure functions behind the notebook's stages, kept here so they are testable:
single-object track cleanup (spikes, bottom-edge exit, gap interpolation),
YOLOE wall-box picking, SAM3 gap planning and acceptance, trace-level splits,
the training manifest, and the SOP spec evaluation. Model objects (YOLOE,
SAM3, YOLO) are passed in; nothing here imports torch or ultralytics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vision_uss_research.labeling.corridor import corridor_weight

BOX = ["x0", "y0", "x1", "y1"]

# Detector class -> height bin (dataset_v2 binning).
DET_BIN = {"curbstone": "low", "speedbump": "low", "woodenboard": "low",
           "low_other": "low", "pole": "high", "dummychild": "high",
           "bicyclestand": "high", "high_other": "high"}

# Objects whose top is below the camera: their box top moves DOWN while the
# vehicle approaches, which the bottom-edge exit rule relies on.
LOW_PROFILE = {"curbhigh", "woodencurb10", "metalbar", "box30", "box40"}

# Bosch proposal for SOP Midsize: (distance m, required detection rate).
SPEC = {"dummychild": (2.20, 0.90), "dummyadult": (2.20, 0.90),
        "isopole": (2.00, 0.95), "wall": (1.60, 0.90), "car": (1.50, 0.95),
        "box30": (1.20, 0.90), "box40": (1.40, 0.90),
        "curbhigh": (1.50, 0.90), "woodencurb10": (1.00, 0.90),
        "metalbar": (1.00, 0.90)}


# ------------------------------------------------------------ track cleanup

def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
             - inter)
    return inter / union if union > 0 else 0.0


def drop_spikes(frames: np.ndarray, boxes: np.ndarray, max_gap: int,
                spike_iou: float = 0.1) -> np.ndarray:
    """Keep-mask: an anchor overlapping neither neighbour while the neighbours
    overlap each other is a one-frame spike."""
    keep = np.ones(len(frames), bool)
    for k in range(1, len(frames) - 1):
        if frames[k + 1] - frames[k - 1] > 2 * (max_gap + 1):
            continue
        prev_ok = iou(boxes[k], boxes[k - 1]) >= spike_iou
        next_ok = iou(boxes[k], boxes[k + 1]) >= spike_iou
        if not prev_ok and not next_ok and \
                iou(boxes[k - 1], boxes[k + 1]) >= spike_iou:
            keep[k] = False
    return keep


def drop_after_exit(boxes: np.ndarray, img_h: int, tol: float = 4.0,
                    edge: float = 2.0) -> np.ndarray:
    """Keep-mask for a low-profile object leaving through the bottom edge: once
    a bottom-touching box's top jumps back up, the object has left and later
    bottom-touching boxes are false positives, until a box is fully in view."""
    keep = np.ones(len(boxes), bool)
    last_y0, exited = None, False
    for k, (_, y0, _, y1) in enumerate(boxes):
        at_bottom, at_top = y1 >= img_h - edge, y0 <= edge
        if not at_bottom or at_top:
            last_y0, exited = None, False
        elif exited:
            keep[k] = False
        elif last_y0 is None or y0 >= last_y0 - tol:
            last_y0 = y0
        else:
            exited, keep[k] = True, False
    return keep


def interpolate_gaps(frames: np.ndarray, boxes: np.ndarray, max_gap: int,
                     min_bridge_iou: float = 0.3) -> list[tuple[int, np.ndarray]]:
    """Linear boxes for gaps of <= max_gap frames between overlapping anchors."""
    out = []
    for (fa, ba), (fb, bb) in zip(zip(frames, boxes),
                                  zip(frames[1:], boxes[1:])):
        gap = fb - fa - 1
        if gap < 1 or gap > max_gap or iou(ba, bb) < min_bridge_iou:
            continue
        for f in range(fa + 1, fb):
            w = (f - fa) / (fb - fa)
            out.append((int(f), (1 - w) * ba + w * bb))
    return out


def pick_box(boxes: np.ndarray, confs: np.ndarray, w: int, h: int,
             min_score: float = 0.05) -> int | None:
    """Index of the best box under the approach-corridor prior, or None."""
    best, best_score = None, min_score
    for i, (b, c) in enumerate(zip(boxes, confs)):
        score = float(c) * corridor_weight((b[0] + b[2]) / 2 / w,
                                           (b[1] + b[3]) / 2 / h)
        if score >= best_score:
            best, best_score = i, score
    return best


def clean_track(frames: np.ndarray, boxes: np.ndarray, label_class: str,
                img_h: int, max_gap: int = 20) -> pd.DataFrame:
    """Anchors -> exit rule (low-profile only) -> spikes -> interpolated gaps.
    Returns frame_idx, x0..y1, src ('det' or 'interp')."""
    order = np.argsort(frames)
    frames, boxes = np.asarray(frames)[order], np.asarray(boxes, float)[order]
    if label_class in LOW_PROFILE and len(boxes):
        keep = drop_after_exit(boxes, img_h)
        frames, boxes = frames[keep], boxes[keep]
    keep = drop_spikes(frames, boxes, max_gap)
    frames, boxes = frames[keep], boxes[keep]
    rows = [(int(f), *b, "det") for f, b in zip(frames, boxes)]
    rows += [(f, *b, "interp") for f, b in interpolate_gaps(frames, boxes, max_gap)]
    return (pd.DataFrame(rows, columns=["frame_idx", *BOX, "src"])
            .sort_values("frame_idx").reset_index(drop=True))


# ------------------------------------------------------------ SAM3 gap fill

def plan_sam3_segments(all_frames, labeled: dict, min_gap: int = 5,
                       max_tail: int = 90, backfill: bool = False,
                       max_lead: int = 60) -> list[dict]:
    """Segments for SAM3 to fill, each seeded from a labeled anchor box.

    `all_frames`: sorted frame indices available for the trace; `labeled`:
    {frame_idx: box}. Interior gaps (>= min_gap unlabeled frames between two
    anchors) run forward from the left anchor and must meet the right anchor;
    the tail after the last anchor runs forward up to max_tail frames (the near
    range the detector loses); with `backfill`, the lead before the first anchor
    runs backward up to max_lead frames."""
    frames = [int(f) for f in all_frames]
    pos = {f: i for i, f in enumerate(frames)}
    anchors = sorted(f for f in labeled if f in pos)
    segs = []
    if not anchors:
        return segs
    for a, b in zip(anchors, anchors[1:]):
        gap = frames[pos[a] + 1:pos[b]]
        if len(gap) >= min_gap:
            segs.append({"direction": "forward", "seed_frame": a,
                         "seed_box": labeled[a], "frames": gap,
                         "end_frame": b, "end_box": labeled[b]})
    tail = frames[pos[anchors[-1]] + 1:pos[anchors[-1]] + 1 + max_tail]
    if len(tail) >= min_gap:
        segs.append({"direction": "forward", "seed_frame": anchors[-1],
                     "seed_box": labeled[anchors[-1]], "frames": tail,
                     "end_frame": None, "end_box": None})
    if backfill:
        lead = frames[max(0, pos[anchors[0]] - max_lead):pos[anchors[0]]]
        if len(lead) >= min_gap:
            segs.append({"direction": "backward", "seed_frame": anchors[0],
                         "seed_box": labeled[anchors[0]],
                         "frames": lead[::-1], "end_frame": None,
                         "end_box": None})
    return segs


def mask_box(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], float)


def accept_sam3_segment(seg: dict, masks: dict, label_class: str, img_h: int,
                        min_area_px: int = 150, end_iou: float = 0.3
                        ) -> list[tuple[int, np.ndarray]]:
    """Gap-frame boxes from SAM3 masks for one segment.

    Interior segment: rejected whole unless the propagated mask still matches
    the right anchor (IoU >= end_iou) - otherwise the tracker drifted. Open
    segment (tail/lead): accepted until the first empty or tiny mask; low-
    profile objects also go through the bottom-edge exit rule."""
    if seg["end_frame"] is not None:
        end = mask_box(masks.get(seg["end_frame"], np.zeros((1, 1), bool)))
        if end is None or iou(end, seg["end_box"]) < end_iou:
            return []
    out = []
    for f in seg["frames"]:
        m = masks.get(f)
        if m is None or int(m.sum()) < min_area_px:
            if seg["end_frame"] is None:
                break
            continue
        out.append((int(f), mask_box(m)))
    if seg["end_frame"] is None and label_class in LOW_PROFILE and out:
        boxes = np.array([np.asarray(seg["seed_box"], float)] + [b for _, b in out])
        keep = drop_after_exit(boxes, img_h)[1:]
        cut = len(keep) if keep.all() else int(np.argmin(keep))
        out = out[:cut]
    return out


# ------------------------------------------------------------ dataset

def split_traces(traces: pd.DataFrame, n_test: int = 2, n_val: int = 1,
                 seed: int = 0) -> dict[tuple[str, int], str]:
    """Trace-level split per (label_class, side): test takes one trace per
    speed first (both speeds held out), then val, rest train. Groups too small
    to keep >= 1 train trace go entirely to train."""
    rng = np.random.default_rng(seed)
    out = {}
    for _, g in traces.groupby(["label_class", "side"], sort=True):
        keys = [(r.side, int(r.trace), r.speed) for r in g.itertuples()]
        rng.shuffle(keys)
        if len(keys) < n_test + n_val + 1:
            out.update({(s, t): "train" for s, t, _ in keys})
            continue
        test, seen = [], set()
        for k in keys:
            if len(test) < n_test and k[2] not in seen:
                test.append(k)
                seen.add(k[2])
        for k in keys:
            if len(test) < n_test and k not in test:
                test.append(k)
        rest = [k for k in keys if k not in test]
        for s, t, _ in test:
            out[(s, t)] = "test"
        for i, (s, t, _) in enumerate(rest):
            out[(s, t)] = "val" if i < n_val else "train"
    return out


def cuv_manifest_rows(labels: pd.DataFrame, split_map: dict, class_map: dict,
                      class_ids: dict, stride: int = 3,
                      source: str = "cuv_bundle") -> pd.DataFrame:
    """Manifest rows (notebook 05 schema) for CUV labels: every `stride`-th
    labeled frame per trace, tracelist object mapped to a dataset_v2 class
    (None = excluded), frame paths relative to the bundle's frames/ dir."""
    df = labels.copy()
    df["class_name"] = df["label_class"].map(class_map)
    df = df[df["class_name"].notna()]
    df = df.sort_values(["side", "trace", "frame_idx"])
    df = df[df.groupby(["side", "trace"]).cumcount() % stride == 0]
    df["split"] = [split_map.get((s, int(t))) for s, t in zip(df["side"], df["trace"])]
    df = df[df["split"].notna()]
    return pd.DataFrame({
        "split": df["split"],
        "sequence_id": "cuv_" + df["side"] + "_" + df["trace"].astype(str),
        "target_object": df["label_class"],
        "camera": df["side"],
        "frame_idx": df["frame_idx"].astype(int),
        "source": source,
        "frame_path": (df["side"] + "_" + df["trace"].astype(str) + "/"
                       + df["frame_idx"].map(lambda f: f"{int(f):05d}.jpg")),
        "x0": df["x0"], "y0": df["y0"], "x1": df["x1"], "y1": df["y1"],
        "class_id": df["class_name"].map(class_ids).astype(int),
    }).reset_index(drop=True)


def replay_rows(old_manifest: pd.DataFrame, frac: float,
                seed: int = 0) -> pd.DataFrame:
    """Old-dataset replay: a sequence-level sample of train and val (test is
    never trained on or validated against here)."""
    rng = np.random.default_rng(seed)
    parts = []
    for split in ("train", "val"):
        g = old_manifest[old_manifest["split"] == split]
        seqs = g["sequence_id"].unique()
        n = max(1, int(round(len(seqs) * frac))) if len(seqs) else 0
        keep = set(rng.choice(seqs, size=n, replace=False)) if n else set()
        parts.append(g[g["sequence_id"].isin(keep)])
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------ spec eval

def final_approach(g: pd.DataFrame) -> pd.DataFrame:
    """Frames from the farthest point to the stop (drops any receding part)."""
    g = g.sort_values("frame_idx").reset_index(drop=True)
    return g.iloc[int(g["dist_m"].to_numpy().argmax()):]


def spec_eval(frames: pd.DataFrame, window: float = 0.25,
              spec: dict = SPEC) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-trace and per-class spec results. `frames` needs side, trace,
    frame_idx, label_class, height_bin, dist_m and det_bin (None = no box).
    A frame is a HIT when det_bin == height_bin; a trace passes when >= 50% of
    its frames in [spec - window, spec] are HITs."""
    df = frames.assign(hit=frames["det_bin"] == frames["height_bin"])
    per_trace = []
    for (side, trace), g in df.groupby(["side", "trace"]):
        label = g["label_class"].iloc[0]
        spec_d = spec[label][0]
        g = final_approach(g)
        win = g[(g["dist_m"] <= spec_d) & (g["dist_m"] >= spec_d - window)]
        boxed = g[g["det_bin"].notna()]
        per_trace.append({
            "side": side, "trace": int(trace), "label_class": label,
            "spec_m": spec_d, "window_frames": len(win),
            "window_hit_rate": round(float(win["hit"].mean()), 2) if len(win) else None,
            "passes": bool(len(win) and win["hit"].mean() >= 0.5),
            "wrong_bin_rate": round(float((boxed["det_bin"] != boxed["height_bin"]).mean()), 3) if len(boxed) else None,
        })
    pt = pd.DataFrame(per_trace)
    summary = (pt.groupby("label_class")
               .agg(traces=("trace", "size"), passing=("passes", "sum"),
                    wrong_bin_rate=("wrong_bin_rate", "mean"))
               .reset_index())
    summary["spec"] = summary["label_class"].map(
        lambda c: f"{spec[c][0]:.1f} m / {spec[c][1]:.0%}")
    summary["passing"] = summary["passing"].astype(int)
    return pt, summary
