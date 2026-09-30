"""USS-geometry auto-labeler: a BEV box for the staged target from the MF4.

Port of the Bosch prototype (M. K. Uludag, Aug 2024, `helpers_mf4.py`,
mdfreader) to asammdf, plus fixed variants, so both can be scored against the
human Label V2 polygons.

What the prototype does (v1) - read from its source, not from the report:
  1. pick one production USS map object slot `Map_Obj{n}_P1x/P2x/P1y/P2y`
     (vehicle frame, mm) whose mid-x decreases somewhere and whose FINAL
     sample sits nearest a bumper line (x = 4200 front / -1300 rear) with
     y near 0;
  2. take that slot's final mid-x (last non-zero if the final one is 0),
     IGNORE its lateral offset, and project it along the final ego yaw from
     the final `VHM_VehicleState_XPos/YPos` (m);
  3. draw a FIXED 0.75 x 1.75 m rectangle aligned with the ego heading.
The committed source hard-codes `obj_number = 8` inside the search loop and
has a live `breakpoint()`; the saved 25-trace evaluation
(comparison_results.xlsx) reproduces exactly only WITHOUT the hard-code
(verified: Box_40cm_802 -> 0.1739 m, Wall_smooth_930 -> 0.7170 m), so v1 here
is the search loop as its docstring describes it.

Ablations (`VARIANTS`) each change ONE thing relative to v1: lateral offset
(both signs - the Map_Obj y convention is undocumented), pose/object time
sync, track-based slot selection, and segment-sized boxes.
Diagnostic `*_oracle_size_iou`: the human polygon's own shape re-centred on a
variant's box centre - IoU with localisation error only, no size error. The
gap between it and the variant's IoU is the fixed-template size penalty.

`asammdf` is imported lazily (same convention as alignment/mf4.py).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from vision_uss_research.autolabel.geometry import (box_metrics, centroid,
                                                    oriented_box, polygon_area,
                                                    polygon_iou, vehicle_to_world)

POSE_CHANNELS = {"x": "VHM_VehicleState_XPos", "y": "VHM_VehicleState_YPos",
                 "yaw": "VHM_VehicleState_YawAngle"}          # m, m, degrees
MAP_OBJ = "Map_Obj{n}_{c}"                                    # vehicle frame, mm
N_SLOTS = 20
BUMPER_TARGETS_MM = (-1300.0, 4200.0)   # prototype's rear / front bumper lines
V1_BOX_LENGTH_M = 0.75                  # along ego heading
V1_BOX_WIDTH_M = 1.75                   # across ego heading
SEG_MIN_LENGTH_M = 0.30
SEG_DEPTH_M = 0.30
TARGET_CLASSES = {"1": "low", "3": "high"}   # Label V2 class codes kept


# ---------------------------------------------------------------- signals

def read_signals(mdf, n_slots: int = N_SLOTS) -> dict:
    """Pose + every Map_Obj slot present, as {name: (timestamps, samples)}."""
    names = set(mdf.channels_db)
    out = {"pose": {}, "slots": {}}
    for k, ch in POSE_CHANNELS.items():
        if ch not in names:
            raise KeyError(f"missing pose channel {ch}")
        s = mdf.get(ch)
        out["pose"][k] = (np.asarray(s.timestamps, float), np.asarray(s.samples, float))
    for n in range(1, n_slots + 1):
        chans = [MAP_OBJ.format(n=n, c=c) for c in ("P1x", "P2x", "P1y", "P2y")]
        if not all(c in names for c in chans):
            continue
        sigs = [mdf.get(c) for c in chans]
        out["slots"][n] = {"t": np.asarray(sigs[0].timestamps, float),
                           **{c: np.asarray(s.samples, float)
                              for c, s in zip(("p1x", "p2x", "p1y", "p2y"), sigs)}}
    return out


def _bumper_score(mid_x: float, mid_y: float) -> float:
    return min(abs(mid_x - t) for t in BUMPER_TARGETS_MM) + abs(mid_y)


# ---------------------------------------------------------------- v1 (as-run)

def select_slot_v1(slots: dict) -> tuple[int | None, float | None]:
    """Prototype selection, scored on each slot's FINAL sample (even if 0)."""
    best, best_score = None, float("inf")
    for n, s in slots.items():
        ax = (s["p1x"] + s["p2x"]) / 2
        ay = (s["p1y"] + s["p2y"]) / 2
        if len(ax) > 1 and np.any(np.diff(ax) < 0):
            score = _bumper_score(ax[-1], ay[-1])
            if score < best_score:
                best, best_score = n, score
    return best, (None if best is None else float(best_score))


def box_v1(pose: dict, slot: dict) -> list[tuple[float, float]]:
    x, y, yaw = (pose[k][1][-1] for k in ("x", "y", "yaw"))
    p1x, p2x = slot["p1x"][-1], slot["p2x"][-1]
    if p1x == 0 and p2x == 0:
        for v, v2 in zip(slot["p1x"][::-1], slot["p2x"][::-1]):
            if v != 0:
                p1x, p2x = v, v2
                break
    r = np.radians(yaw)
    off = (p1x + p2x) / 2000.0
    return oriented_box(x + off * np.cos(r), y + off * np.sin(r), r,
                        V1_BOX_LENGTH_M, V1_BOX_WIDTH_M)


# ---------------------------------------------------------------- ablations

def _last_valid(slot: dict) -> int | None:
    live = np.flatnonzero((slot["p1x"] != 0) | (slot["p2x"] != 0)
                          | (slot["p1y"] != 0) | (slot["p2y"] != 0))
    return int(live[-1]) if live.size else None


def select_slot_v2(slots: dict, min_valid: int = 3) -> tuple[int | None, float | None]:
    """Same bumper criterion, but on each slot's last VALID sample and only
    for slots with a real track (>= min_valid live samples, approaching)."""
    best, best_score = None, float("inf")
    for n, s in slots.items():
        live = (s["p1x"] != 0) | (s["p2x"] != 0)
        if live.sum() < min_valid:
            continue
        ax = ((s["p1x"] + s["p2x"]) / 2)[live]
        ay = ((s["p1y"] + s["p2y"]) / 2)[live]
        if not np.any(np.diff(ax) < 0):
            continue
        score = _bumper_score(ax[-1], ay[-1])
        if score < best_score:
            best, best_score = n, score
    return best, (None if best is None else float(best_score))


def pose_at(pose: dict, t: float) -> tuple[float, float, float]:
    """Ego (x, y, yaw_rad) interpolated at time t (yaw unwrapped first)."""
    x = np.interp(t, *pose["x"])
    y = np.interp(t, *pose["y"])
    ty, yaw = pose["yaw"]
    return float(x), float(y), float(np.interp(t, ty, np.unwrap(np.radians(yaw))))


def face_geometry(pose: dict, slot: dict, lateral: int = 0, timing: str = "final"
                  ) -> dict | None:
    """Where the map object's reflecting face is, in world metres.

    Returns {"p1", "p2", "centre", "yaw" (rad), "side" (+1 object ahead of the
    reference point, -1 behind)} or None. See `build_box` for the arguments."""
    if timing == "final":
        x, y, yaw = (pose[k][1][-1] for k in ("x", "y", "yaw"))
        yaw = float(np.radians(yaw))
        i = len(slot["p1x"]) - 1
        if slot["p1x"][i] == 0 and slot["p2x"][i] == 0:
            i = _last_valid(slot)
    else:
        i = _last_valid(slot)
        if i is None:
            return None
        x, y, yaw = pose_at(pose, slot["t"][i])
    if i is None:
        return None
    sgn = float(lateral)
    p1 = vehicle_to_world(x, y, yaw, slot["p1x"][i] / 1000, sgn * slot["p1y"][i] / 1000)
    p2 = vehicle_to_world(x, y, yaw, slot["p2x"][i] / 1000, sgn * slot["p2y"][i] / 1000)
    mid_x = (slot["p1x"][i] + slot["p2x"][i]) / 2
    return {"p1": p1, "p2": p2, "centre": ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2),
            "yaw": yaw, "side": 1 if mid_x >= 0 else -1}


def box_behind_face(face_centre, yaw: float, side: int, length: float, width: float,
                    depth_shift: bool = True) -> tuple[list, tuple[float, float]]:
    """Ego-aligned `length` x `width` box whose NEAR edge sits on the reflecting
    face (depth_shift) or centred on it (v1). The USS sees the object's near
    face, so a box centred there is half a length too close to the car."""
    cx, cy = face_centre
    if depth_shift:
        cx += side * length / 2 * np.cos(yaw)
        cy += side * length / 2 * np.sin(yaw)
    return oriented_box(cx, cy, yaw, length, width), (cx, cy)


def build_box(pose: dict, slot: dict, lateral: int = 0, size: str = "fixed",
              timing: str = "final", depth_shift: bool = False
              ) -> tuple[list, tuple[float, float]] | None:
    """One box, changing ONE thing at a time relative to v1.

    lateral: 0 = ignore the object's y offset (v1), +1 / -1 = add it with
             that sign. The prototype never documented the Map_Obj y axis;
             the al_eval_v1 smoke run (172 sequences) settled it: +1 (y left)
             halves the centroid error, -1 makes it worse.
    size:    "fixed" = 0.75 x 1.75 m along ego heading (v1); "seg" = the map
             object's P1-P2 segment (min length / fixed depth), which needs a
             lateral sign.
    timing:  "final" = final ego pose + final (or last non-zero) object
             sample (v1); "sync" = the slot's last valid sample with the ego
             pose interpolated to that timestamp.
    depth_shift: move the fixed box back so its near edge is on the face.
    Returns (corners, centre) or None."""
    f = face_geometry(pose, slot, lateral, timing)
    if f is None:
        return None
    if size == "fixed":
        return box_behind_face(f["centre"], f["yaw"], f["side"],
                               V1_BOX_LENGTH_M, V1_BOX_WIDTH_M, depth_shift)
    (x1, y1), (x2, y2) = f["p1"], f["p2"]
    seg = float(np.hypot(x2 - x1, y2 - y1))
    heading = np.arctan2(y2 - y1, x2 - x1) if seg > 0.05 else f["yaw"] + np.pi / 2
    return (oriented_box(*f["centre"], heading, max(seg, SEG_MIN_LENGTH_M), SEG_DEPTH_M),
            f["centre"])


# name -> (slot selector, build_box kwargs). "v1" must stay the as-run prototype.
VARIANTS = {
    "v1":      ("v1", {}),
    "lat_pos": ("v1", {"lateral": +1}),
    "lat_neg": ("v1", {"lateral": -1}),
    "sync":    ("v1", {"timing": "sync"}),
    "sel2":    ("v2", {}),
    "seg_pos": ("v1", {"lateral": +1, "size": "seg"}),
    "seg_neg": ("v1", {"lateral": -1, "size": "seg"}),
    "depth":   ("v1", {"lateral": +1, "depth_shift": True}),   # lat_pos + depth shift
}
# Face geometry stored per row (lat_pos convention) so size priors can be
# scored in the notebook without re-reading the MF4.
FACE_LATERAL = +1
FIXED_SIZE_VARIANTS = [k for k, (_, kw) in VARIANTS.items() if kw.get("size", "fixed") == "fixed"]


# ---------------------------------------------------------------- labels

def parse_label_targets(label_csv: Path) -> list[dict]:
    """Human target polygons from a Label V2 CSV: rows with class low/high and
    a polygon of >= 3 vertices. Polygon returned in metres."""
    targets = []
    with open(label_csv, newline="", encoding="utf-8-sig", errors="replace") as f:
        for row in csv.DictReader(f):
            cls_col = next((k for k in row if k and k.strip().lower().startswith("class")), None)
            code = str(row.get(cls_col, "")).strip().split(".")[0] if cls_col else ""
            obj = str(row.get("Which object", "")).strip()
            if code not in TARGET_CLASSES or obj.lower() == "empty":
                continue
            try:
                xs = [float(v) / 1000 for v in str(row.get("PolygonX", "")).split(",") if v.strip()]
                ys = [float(v) / 1000 for v in str(row.get("PolygonY", "")).split(",") if v.strip()]
            except ValueError:
                continue
            if len(xs) >= 3 and len(xs) == len(ys):
                targets.append({"object": obj, "height_bin": TARGET_CLASSES[code],
                                "polygon": list(zip(xs, ys))})
    return targets


# ---------------------------------------------------------------- one sequence

def convex_hull(points) -> list[tuple[float, float]]:
    pts = sorted(set(map(tuple, np.asarray(points, float).tolist())))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def oracle_size_iou(gt, centre) -> float:
    """IoU of the human polygon with its own shape moved to `centre`."""
    gx, gy = centroid(gt)
    moved = [(x - gx + centre[0], y - gy + centre[1]) for x, y in convex_hull(gt)]
    return polygon_iou(gt, moved)


def extents_along(poly, heading: float) -> tuple[float, float]:
    """Polygon extent along `heading` and across it (m)."""
    p = np.asarray(poly, dtype=float)
    u = np.array([np.cos(heading), np.sin(heading)])
    v = np.array([-u[1], u[0]])
    a, b = p @ u, p @ v
    return float(a.max() - a.min()), float(b.max() - b.min())


def polygon_to_str(poly) -> str:
    return ";".join(f"{x:.3f} {y:.3f}" for x, y in poly)


def polygon_from_str(s: str) -> list[tuple[float, float]]:
    return [tuple(map(float, v.split())) for v in str(s).split(";") if v.strip()]


def evaluate_sequence(mf4_path: Path, label_csv: Path) -> dict:
    """One results row: every method's box vs the (first) human target.
    Never raises - failures land in `error`."""
    from asammdf import MDF

    row: dict = {"error": ""}
    try:
        targets = parse_label_targets(label_csv)
    except Exception as e:
        row["error"] = f"label:{type(e).__name__}"
        return row
    row["n_targets"] = len(targets)
    if not targets:
        row["error"] = "no_target_polygon"
        return row
    gt = targets[0]
    row.update(object=gt["object"], height_bin=gt["height_bin"],
               gt_area_m2=round(polygon_area(gt["polygon"]), 4),
               gt_n_vertices=len(gt["polygon"]))
    mdf = None
    try:
        mdf = MDF(str(mf4_path))
        sig = read_signals(mdf)
        row["n_slots"] = len(sig["slots"])

        chosen = {}
        for sel, fn in (("v1", select_slot_v1), ("v2", select_slot_v2)):
            n, score = fn(sig["slots"])
            chosen[sel] = n
            row.update({f"slot_{sel}": n, f"slot_{sel}_score_mm": score})
        for name, (sel, kw) in VARIANTS.items():
            n = chosen[sel]
            if n is None:
                continue
            built = build_box(sig["pose"], sig["slots"][n], **kw)
            if built is None:
                continue
            corners, centre = built
            row.update({f"{name}_{k}": v for k, v in
                        box_metrics(corners, gt["polygon"]).items()})
            if name in FIXED_SIZE_VARIANTS:
                row[f"{name}_oracle_size_iou"] = round(
                    oracle_size_iou(gt["polygon"], centre), 4)
        if chosen["v1"] is not None:
            f = face_geometry(sig["pose"], sig["slots"][chosen["v1"]], FACE_LATERAL)
            if f is not None:
                length, width = extents_along(gt["polygon"], f["yaw"])
                row.update(face_cx=round(f["centre"][0], 4), face_cy=round(f["centre"][1], 4),
                           ego_yaw_rad=round(f["yaw"], 6), obj_side=f["side"],
                           gt_len_along_m=round(length, 4), gt_wid_across_m=round(width, 4))
        row["gt_polygon"] = polygon_to_str(gt["polygon"])
        if chosen["v1"] is None and chosen["v2"] is None:
            row["error"] = "no_object_selected"
    except Exception as e:
        row["error"] = f"mf4:{type(e).__name__}:{e}"
    finally:
        if mdf is not None:
            try:
                mdf.close()
            except Exception:
                pass
    return row
