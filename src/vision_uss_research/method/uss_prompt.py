"""USS-prompted traversability, step 0: project production USS map objects
into the bumper-camera image.

A map-object slot (`Map_Obj{n}_P1/P2`, vehicle frame, mm, x forward / y left;
the y sign was settled in the autolabel track) is the USS stack's estimate of
an obstacle's near face. If one fixed camera model per bumper maps that face
onto the obstacle in the image, the USS can tell the camera WHERE to look and
the camera only has to judge HOW HIGH - the split the corrected WACV numbers
point to (camera nearly always right when it decides, limited by coverage).

Frame timing reuses alignment/mf4.py: labelled frame i of the r30-85_n20
window sits at video second linspace(.30, .85, 20)[i] * duration, mapped to
MF4 time through the camera channel's (pts, mf4 time) pairing. Map objects are
already in the vehicle frame and the camera is rigid on the vehicle, so no ego
pose is involved.

Pure numpy; scipy only inside `fit_ground_camera`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

FRONT_BUMPER_MM = 4200.0     # same reference lines as autolabel/uss_box.py
REAR_BUMPER_MM = -1300.0
FACING = {"front": 1, "rear": -1}


MAP_OBJ = "Map_Obj{n}_{c}"
N_SLOTS = 20


def read_slots(mdf, n_slots: int = N_SLOTS) -> dict:
    """Every Map_Obj slot present in an open MDF, without requiring the ego
    pose channels (autolabel.uss_box.read_signals does): {n: {t, p1x, ...}}."""
    names = set(mdf.channels_db)
    slots = {}
    for n in range(1, n_slots + 1):
        chans = [MAP_OBJ.format(n=n, c=c) for c in ("P1x", "P2x", "P1y", "P2y")]
        if not all(c in names for c in chans):
            continue
        sigs = [mdf.get(c) for c in chans]
        slots[n] = {"t": np.asarray(sigs[0].timestamps, float),
                    **{k: np.asarray(s.samples, float)
                       for k, s in zip(("p1x", "p2x", "p1y", "p2y"), sigs)}}
    return slots


def window_video_seconds(duration_s: float, n_frames: int = 20,
                         ratio: tuple[float, float] = (0.30, 0.85)) -> np.ndarray:
    """Video seconds of the extraction window's frames (sequences.py rule)."""
    return np.linspace(ratio[0], ratio[1], n_frames) * float(duration_s)


def slots_at(slots: dict, t: float, max_dt: float = 0.15) -> list[dict]:
    """Every map-object slot with a live (non-zero) sample within `max_dt`
    seconds of MF4 time `t`, at its nearest live sample."""
    out = []
    for n, s in slots.items():
        live = np.flatnonzero((s["p1x"] != 0) | (s["p2x"] != 0)
                              | (s["p1y"] != 0) | (s["p2y"] != 0))
        if live.size == 0:
            continue
        j = live[np.argmin(np.abs(s["t"][live] - t))]
        dt = float(s["t"][j] - t)
        if abs(dt) > max_dt:
            continue
        p = {k: float(s[k][j]) for k in ("p1x", "p1y", "p2x", "p2y")}
        out.append({"slot": int(n), "dt": dt, **p,
                    "mid_x": (p["p1x"] + p["p2x"]) / 2,
                    "mid_y": (p["p1y"] + p["p2y"]) / 2})
    return out


def range_ahead_mm(mid_x: float, camera: str) -> float:
    """Distance of a vehicle-frame x in front of the camera's bumper line."""
    if camera == "front":
        return mid_x - FRONT_BUMPER_MM
    return REAR_BUMPER_MM - mid_x


def corridor_slot(slot_rows: list[dict], camera: str,
                  max_lateral_mm: float = 1200.0,
                  min_range_mm: float = -300.0) -> dict | None:
    """The inference-time choice, made WITHOUT the image: the nearest slot in
    front of the camera's bumper and inside the lateral driving corridor."""
    best, best_r = None, np.inf
    for r in slot_rows:
        rng = range_ahead_mm(r["mid_x"], camera)
        if rng < min_range_mm or abs(r["mid_y"]) > max_lateral_mm:
            continue
        if rng < best_r:
            best, best_r = r, rng
    return best


@dataclass
class GroundCamera:
    """Pinhole camera over a flat ground plane, in the vehicle frame (mm).

    Mounted at (xc, yc, h) facing +x (front) or -x (rear); `pitch` is the
    downward tilt and `yaw` the toe-in, both radians; `k1` one radial
    distortion term on normalised image coordinates."""
    facing: int
    f: float = 500.0
    u0: float = 480.0
    v0: float = 320.0
    xc: float = 4000.0
    yc: float = 0.0
    h: float = 700.0
    pitch: float = np.radians(10.0)
    yaw: float = 0.0
    k1: float = 0.0

    PARAMS = ("f", "u0", "v0", "xc", "yc", "h", "pitch", "yaw", "k1")

    @classmethod
    def initial(cls, camera: str) -> "GroundCamera":
        facing = FACING[camera]
        xc = FRONT_BUMPER_MM - 200 if facing > 0 else REAR_BUMPER_MM + 200
        return cls(facing=facing, xc=xc)

    def vector(self) -> np.ndarray:
        return np.array([getattr(self, p) for p in self.PARAMS], float)

    def with_vector(self, v) -> "GroundCamera":
        return GroundCamera(self.facing, *map(float, v))

    def project(self, x_mm, y_mm):
        """Ground points -> (u, v, in_front). `in_front` False where the point
        is behind the image plane (projection meaningless)."""
        x_mm, y_mm = np.asarray(x_mm, float), np.asarray(y_mm, float)
        fwd = self.facing * (x_mm - self.xc)
        left = self.facing * (y_mm - self.yc)
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        fwd, left = fwd * cy + left * sy, -fwd * sy + left * cy
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        zc = fwd * cp + self.h * sp            # optical axis
        yc = -fwd * sp + self.h * cp           # image down
        xc = -left                             # image right
        ok = zc > 1.0
        zs = np.where(ok, zc, 1.0)
        xn, yn = xc / zs, yc / zs
        d = 1.0 + self.k1 * (xn ** 2 + yn ** 2)
        return self.u0 + self.f * xn * d, self.v0 + self.f * yn * d, ok

    def as_dict(self) -> dict:
        return asdict(self)


def fit_ground_camera(x_mm, y_mm, u, v, camera: str,
                      init: GroundCamera | None = None,
                      f_scale_px: float = 15.0) -> tuple[GroundCamera, dict]:
    """Robust least squares of image points (u, v) against ground points.

    The corpus cameras are NOT bumper mounts (the front one sees the hood:
    windshield / rear-window rigs), so mount position and height get wide
    bounds and the fit is multi-started over mount x, height, focal length
    and pitch; the lowest robust cost wins. Weak priors keep the
    under-determined directions sane (yc ~ 0, yaw ~ 0, k1 ~ 0); the soft-L1
    loss tolerates wrong slot/box pairings."""
    from scipy.optimize import least_squares

    base = init or GroundCamera.initial(camera)
    x_mm, y_mm = np.asarray(x_mm, float), np.asarray(y_mm, float)
    u, v = np.asarray(u, float), np.asarray(v, float)
    n = max(len(u), 1)
    prior_w = np.sqrt(n) * 0.05

    def resid(p):
        cam = base.with_vector(p)
        pu, pv, ok = cam.project(x_mm, y_mm)
        r = np.concatenate([np.where(ok, pu - u, 500.0), np.where(ok, pv - v, 500.0)])
        prior = prior_w * np.array([cam.yc / 300.0, cam.yaw / np.radians(5.0),
                                    cam.k1 / 0.2])
        return np.concatenate([r, prior])

    lo = [150, 0, 0, -6000, -1500, 200, np.radians(-20), np.radians(-30), -0.6]
    hi = [3000, 1920, 1280, 6000, 1500, 2500, np.radians(70), np.radians(30), 0.6]
    bumper = FRONT_BUMPER_MM if base.facing > 0 else REAR_BUMPER_MM
    inner = 1800.0 if base.facing > 0 else -300.0      # windshield / rear window
    starts = []
    if init is not None:
        starts.append(init.vector())
    for xc in (bumper - base.facing * 200, inner):
        for h in (600.0, 1300.0):
            for f in (400.0, 900.0):
                for pitch in (np.radians(5), np.radians(20)):
                    s = base.vector()
                    s[[0, 3, 5, 6]] = (f, xc, h, pitch)
                    starts.append(s)
    best = None
    for s in starts:
        s = np.clip(s, lo, hi)
        sol = least_squares(resid, s, bounds=(lo, hi), loss="soft_l1", f_scale=f_scale_px)
        if best is None or sol.cost < best.cost:
            best = sol
    cam = base.with_vector(best.x)
    pu, pv, ok = cam.project(x_mm, y_mm)
    err = np.hypot(pu - u, pv - v)[ok]
    at_bound = [nm for nm, val, a, b in zip(GroundCamera.PARAMS, best.x, lo, hi)
                if np.isclose(val, a, rtol=1e-3) or np.isclose(val, b, rtol=1e-3)]
    return cam, {"n": int(len(u)), "success": bool(best.success), "at_bound": at_bound,
                 "median_px": float(np.median(err)) if err.size else float("nan"),
                 "p90_px": float(np.percentile(err, 90)) if err.size else float("nan")}


def box_hit(u, v, x0, y0, x1, y1, pad: float = 0.10, below: float = 0.15):
    """Does the projected point land on the object's image box? Pads the box
    by `pad` of its width/height, plus `below` of its height under the bottom
    edge (the near-face ground point sits at or just below the box bottom)."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    w = np.asarray(x1, float) - np.asarray(x0, float)
    hgt = np.asarray(y1, float) - np.asarray(y0, float)
    return ((u >= x0 - pad * w) & (u <= x1 + pad * w)
            & (v >= y0 - pad * hgt) & (v <= y1 + below * hgt))


def region_box(u: float, v: float, img_w: int, img_h: int, half_w: float = 160.0,
               below: float = 100.0, above: float = 250.0) -> tuple[float, ...]:
    """Search region around a projected near-face ground point (u, v). The
    obstacle stands ABOVE its ground contact, so the region extends `above`
    px up and `below` px down (projection error is ~symmetric, p80 ~100 px).
    Clipped to the image."""
    x0, x1 = max(0.0, u - half_w), min(float(img_w), u + half_w)
    y0, y1 = max(0.0, v - above), min(float(img_h), v + below)
    return (x0, y0, x1, y1)


def mask_box(mask: np.ndarray) -> tuple[float, ...] | None:
    """Tight (x0, y0, x1, y1) of a boolean mask; None if empty."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def track_target_slots(slots, min_frames: int = 4, min_approach_mm: float = 500.0,
                       max_lateral_mm: float = 1200.0, min_last_range_mm: float = -500.0):
    """One target slot per (sequence, camera), chosen by its whole TRACK, not
    per frame: seen in >= min_frames frames, range shrinking by
    >= min_approach_mm, median |lateral| inside the corridor, and ending
    nearest the bumper. Per-frame "nearest in corridor" picks ground/ghost
    clutter near the bumper on many frames; the track rule mirrors the
    autolabel track's validated slot selection.

    `slots`: DataFrame with sequence_id, camera, slot, frame_idx, range_mm,
    mid_y (notebook 20 output + range_mm). Returns sequence_id, camera, slot."""
    g = slots.sort_values("frame_idx").groupby(["sequence_id", "camera", "slot"])
    tr = g.agg(n=("frame_idx", "size"), r_first=("range_mm", "first"),
               r_last=("range_mm", "last"),
               y_med=("mid_y", lambda s: s.abs().median())).reset_index()
    tr = tr[(tr.n >= min_frames) & (tr.r_first - tr.r_last >= min_approach_mm)
            & (tr.y_med <= max_lateral_mm) & (tr.r_last >= min_last_range_mm)]
    return (tr.sort_values("r_last").drop_duplicates(["sequence_id", "camera"])
            [["sequence_id", "camera", "slot"]].reset_index(drop=True))
