"""Align video frames to MF4 master time, and sample USS distance at frames.

Key convention (verified, see docs/mf4_video_alignment.md): the MF4 carries a
camera channel per view (`WebCam` front, `WebCam3` rear) where, for every video
frame, the channel's SAMPLE VALUE is that frame's video PTS in seconds and the
channel's TIMESTAMP is the frame's MF4 master time. That pairing is an exact
clock mapping - no cycle-number guessing.

`asammdf` is imported lazily so the rest of the package (and CI) never needs it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# camera view -> candidate MF4 channel names, best first
CAMERA_CHANNELS: dict[str, list[str]] = {
    "front": ["WebCam", "WebCam1"],
    "rear": ["WebCam3", "WebCam2"],
}

# USS distance channels, best first. PDC zones are the production
# park-distance signal (0xFFF = no object); ObjBuff is the per-tracked-object
# last detected distance. Units differ - see `DISTANCE_SENTINELS`.
DISTANCE_CHANNELS: dict[str, list[str]] = {
    "rear": [f"sigPDC_Zone{z:02d}_Distance" for z in (13, 14, 15, 16, 17, 18)],
    "front": [f"sigPDC_Zone{z:02d}_Distance" for z in (1, 2, 3, 4, 5, 6)],
}
OBJBUFF_DIST = "MAP_ObjBuff_elm{i}_obj_OVFAttributes_LastDetDist"

DISTANCE_SENTINELS = {"pdc": 4095.0, "objbuff": 0.0}


def find_channel(names, candidates: list[str]) -> str | None:
    """First candidate present in `names` (a set/iterable of channel names)."""
    available = set(names)
    for c in candidates:
        if c in available:
            return c
    return None


def find_camera_channel(names, camera: str) -> str | None:
    return find_channel(names, CAMERA_CHANNELS.get(camera, []))


def sorted_pairs(pts: np.ndarray, mf4_time: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    """(video_pts, mf4_time) sorted by pts, ready for interpolation."""
    pts = np.asarray(pts, dtype=float)
    mf4_time = np.asarray(mf4_time, dtype=float)
    order = np.argsort(pts)
    return pts[order], mf4_time[order]


def video_to_mf4(video_seconds, pts: np.ndarray, mf4_time: np.ndarray):
    """Map video PTS seconds -> MF4 master seconds via the frame pairing."""
    return np.interp(video_seconds, pts, mf4_time)


def mask_sentinel(values: np.ndarray, sentinel: float | None,
                  mode: str = "ge") -> np.ndarray:
    """Replace no-reading sentinels with NaN. `mode='ge'` masks values >=
    sentinel (PDC 0xFFF), `'eq'` masks exact matches (ObjBuff 0 = no track)."""
    v = np.asarray(values, dtype=float).copy()
    if sentinel is None:
        return v
    if mode == "ge":
        v[v >= sentinel] = np.nan
    else:
        v[v == sentinel] = np.nan
    return v


def distance_at(mf4_seconds, times: np.ndarray, values: np.ndarray):
    """Interpolate a distance signal (NaNs = no reading) at MF4 times.

    Outside the span of valid readings the result is NaN, NOT the nearest
    value: np.interp clamps by default, which would invent a constant distance
    for frames recorded before the object entered USS range. A fabricated
    distance would silently corrupt any height estimate, so 'no reading' must
    stay visible as NaN."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    good = ~np.isnan(values)
    if good.sum() == 0:
        return np.full(np.shape(mf4_seconds), np.nan, dtype=float)
    return np.interp(mf4_seconds, times[good], values[good],
                     left=np.nan, right=np.nan)


def pts_match_error(channel_samples: np.ndarray,
                    video_pts: np.ndarray) -> float | None:
    """Max abs difference between the camera channel's sample values and the
    video's own frame PTS - the alignment correctness check. None if the
    lengths differ (then the pairing assumption does not hold)."""
    a = np.sort(np.asarray(channel_samples, dtype=float))
    b = np.sort(np.asarray(video_pts, dtype=float))
    if len(a) != len(b) or len(a) == 0:
        return None
    return float(np.max(np.abs(a - b)))


def nearest_obstacle_distance(mdf, camera: str, names=None
                              ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-timestamp distance to the NEAREST obstacle in the approach direction:
    the element-wise min across the direction-relevant PDC zones (sentinels
    masked). For staged single-target approaches that nearest obstacle IS the
    labeled target, which is why this is a defensible v1 rule.

    Returns (times, distances_cm, zone_channels_used). Distances are NaN where
    no zone reports a reading."""
    if names is None:
        names = {c.name for g in mdf.groups for c in g.channels}
    used, series, base_t = [], [], None
    for ch in DISTANCE_CHANNELS.get(camera, []):
        if ch not in names:
            continue
        sig = mdf.get(ch)
        t = np.asarray(sig.timestamps, dtype=float)
        v = mask_sentinel(np.asarray(sig.samples, dtype=float),
                          DISTANCE_SENTINELS["pdc"], "ge")
        if np.isfinite(v).sum() == 0:
            continue
        if base_t is None:
            base_t = t
            series.append(v)
        else:
            # zones share a clock but may differ in length; resample onto base_t
            good = np.isfinite(v)
            series.append(np.interp(base_t, t[good], v[good],
                                    left=np.nan, right=np.nan))
        used.append(ch)
    if base_t is None:
        return np.array([]), np.array([]), []
    # min across zones, NaN where no zone reports (via +inf sentinel, so numpy
    # never warns about all-NaN slices)
    stack = np.vstack(series)
    filled = np.where(np.isnan(stack), np.inf, stack)
    nearest = np.min(filled, axis=0)
    nearest[~np.isfinite(nearest)] = np.nan
    return base_t, nearest, used


def objbuff_distance(mdf, names=None, n_objbuff: int = 20
                     ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Nearest-object distance from the MAP object buffer (mm).

    Preferred over `nearest_obstacle_distance` (PDC zones) for frame-level work:
    measured on the sample sequence, the object buffer covered 20/20 frames of
    the 0.30-0.85 approach window (range ~370-3000 mm) while PDC zones covered
    only 3/20 - PDC is a short-range (~1.2 m) parking warning, whereas the
    object buffer holds MAP objects out to ~5 m.

    CAVEAT: this is the min across ALL live slots, so in multi-object scenes it
    can switch between objects (monotonicity < 1 is the tell). Selecting the
    slot that tracks the *labeled* target is the open problem - see
    notebooks/11."""
    if names is None:
        names = {c.name for g in mdf.groups for c in g.channels}
    series, base_t, used = [], None, []
    for i in range(1, n_objbuff + 1):
        ch = OBJBUFF_DIST.format(i=i)
        if ch not in names:
            continue
        sig = mdf.get(ch)
        t = np.asarray(sig.timestamps, dtype=float)
        v = mask_sentinel(np.asarray(sig.samples, dtype=float),
                          DISTANCE_SENTINELS["objbuff"], "eq")
        if np.isfinite(v).sum() == 0:
            continue
        if base_t is None:
            base_t = t
            series.append(v)
        else:
            good = np.isfinite(v)
            series.append(np.interp(base_t, t[good], v[good],
                                    left=np.nan, right=np.nan))
        used.append(ch)
    if base_t is None:
        return np.array([]), np.array([]), []
    stack = np.vstack(series)
    filled = np.where(np.isnan(stack), np.inf, stack)
    nearest = np.min(filled, axis=0)
    nearest[~np.isfinite(nearest)] = np.nan
    return base_t, nearest, used


def approach_monotonicity(distances: np.ndarray) -> float | None:
    """Fraction of consecutive valid readings where distance DECREASES - the
    signature of approaching a static object. ~1.0 = clean approach, ~0.5 =
    noise, ~0.0 = receding. None when there are too few readings."""
    v = np.asarray(distances, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return None
    d = np.diff(v)
    moving = d[d != 0]
    if len(moving) == 0:
        return None
    return float((moving < 0).sum() / len(moving))


def video_frame_count(video_path: Path) -> int:
    """Frame count from the container header (cheap - no decode)."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def frame_distances(mf4_path: Path, camera: str, frame_pts,
                    source: str = "objbuff") -> dict:
    """The whole Track B chain for one sequence, in one call.

    video PTS -> MF4 master time (camera-channel pairing) -> nearest-object
    USS distance in MILLIMETRES, sampled at each frame.

    `source`: "objbuff" (default - MAP object buffer, ~5 m range, near-full
    frame coverage) or "pdc" (direction-relevant PDC zones, cm, ~1.2 m range,
    so mostly NaN across a full approach window). Frames outside the span of
    valid readings are NaN, never a fabricated constant.

    Returns {"distance_mm" aligned to frame_pts, "camera_channel",
    "channels_used", "monotonicity", "coverage", "error"}. Never raises."""
    from asammdf import MDF

    out = {"distance_mm": None, "camera_channel": "", "channels_used": [],
           "monotonicity": None, "coverage": 0.0, "error": ""}
    try:
        mdf = MDF(str(mf4_path))
    except Exception as e:
        out["error"] = f"open:{type(e).__name__}"
        return out
    try:
        names = {c.name for g in mdf.groups for c in g.channels}
        cam_ch = find_camera_channel(names, camera)
        if cam_ch is None:
            out["error"] = f"no_camera_channel:{camera}"
            return out
        sig = mdf.get(cam_ch)
        pts, mf4t = sorted_pairs(sig.samples, sig.timestamps)
        out["camera_channel"] = cam_ch

        if source == "pdc":
            t, dist, chans = nearest_obstacle_distance(mdf, camera, names)
            unit_scale = 10.0        # PDC zones read in cm
        else:
            t, dist, chans = objbuff_distance(mdf, names)
            unit_scale = 1.0         # object buffer already in mm
        if len(t) == 0:
            out["error"] = f"no_distance_channel:{source}:{camera}"
            return out
        out["channels_used"] = chans
        out["monotonicity"] = approach_monotonicity(dist)

        mf4_at_frames = video_to_mf4(np.asarray(frame_pts, dtype=float), pts, mf4t)
        d = distance_at(mf4_at_frames, t, dist) * unit_scale
        out["distance_mm"] = d
        out["coverage"] = float(np.isfinite(d).mean()) if np.size(d) else 0.0
    except Exception as e:
        out["error"] = f"probe:{type(e).__name__}:{e}"
    finally:
        try:
            mdf.close()
        except Exception:
            pass
    return out


def camera_for_direction(direction: str) -> str | None:
    """Only ONE camera matters per sequence: the one facing the approach.
    forward -> front, backward -> rear (same rule as the labeling pipeline)."""
    return {"forward": "front", "backward": "rear"}.get(str(direction).lower())


def probe_sequence(mf4_path: Path, videos: dict[str, Path],
                   n_objbuff: int = 20, required_camera: str | None = None) -> dict:
    """One EDA row: does this sequence support the alignment recipe?

    `videos`: {"front": path, "rear": path} (either may be missing).
    `required_camera`: the direction-relevant camera ("front"/"rear"). When
    given, `alignable` is judged on THAT camera only - the other view never
    sees the approached object, so its alignment is irrelevant. When None,
    any camera counts (legacy behaviour).

    Reports per camera whether the MF4 camera channel exists, whether its
    sample count equals the video frame count, and the PTS ramp shape; plus
    which USS distance channels carry real readings. Never raises - failures
    are returned in the row."""
    from asammdf import MDF

    row: dict = {"mf4": str(mf4_path), "error": ""}
    try:
        mdf = MDF(str(mf4_path))
    except Exception as e:
        row["error"] = f"open:{type(e).__name__}:{e}"
        return row

    try:
        names = {c.name for g in mdf.groups for c in g.channels}
        row["n_channels"] = len(names)

        for camera, video_path in videos.items():
            ch = find_camera_channel(names, camera)
            row[f"{camera}_channel"] = ch or ""
            if ch is None or video_path is None:
                continue
            sig = mdf.get(ch)
            samples = np.asarray(sig.samples, dtype=float)
            times = np.asarray(sig.timestamps, dtype=float)
            row[f"{camera}_n_samples"] = int(len(samples))
            row[f"{camera}_n_frames"] = int(video_frame_count(video_path))
            row[f"{camera}_counts_match"] = int(
                row[f"{camera}_n_samples"] == row[f"{camera}_n_frames"])
            if len(times):
                row[f"{camera}_mf4_span_s"] = round(float(times[-1] - times[0]), 3)
                row[f"{camera}_pts_span_s"] = round(
                    float(np.max(samples) - np.min(samples)), 3)
            # PTS agreement uses the channel's own samples as the video clock;
            # a monotone 0..duration ramp is the signature we rely on.
            row[f"{camera}_pts_monotone"] = int(
                bool(np.all(np.diff(np.sort(samples)) >= -1e-6)))

        # distance channels: only the zones facing the approach matter
        zone_cameras = [required_camera] if required_camera else list(videos)
        usable = []
        for camera in zone_cameras:
            for ch in DISTANCE_CHANNELS.get(camera, []):
                if ch not in names:
                    continue
                v = mask_sentinel(np.asarray(mdf.get(ch).samples, float),
                                  DISTANCE_SENTINELS["pdc"], "ge")
                if np.isfinite(v).sum() > 0:
                    usable.append(f"{ch}:{int(np.isfinite(v).sum())}")
        n_objbuff_ok = 0
        for i in range(1, n_objbuff + 1):
            ch = OBJBUFF_DIST.format(i=i)
            if ch not in names:
                continue
            v = np.asarray(mdf.get(ch).samples, float)
            if np.count_nonzero(v) > 0:
                n_objbuff_ok += 1
        row["pdc_channels_with_data"] = "|".join(usable)
        row["n_objbuff_with_data"] = n_objbuff_ok

        # alignable: judged on the direction-relevant camera when known
        if required_camera:
            row["required_camera"] = required_camera
            cam_ok = int(row.get(f"{required_camera}_counts_match", 0))
        else:
            row["required_camera"] = ""
            cam_ok = int(any(row.get(f"{c}_counts_match", 0) for c in videos))
        row["required_camera_counts_match"] = cam_ok
        row["alignable"] = int(cam_ok and (bool(usable) or n_objbuff_ok > 0))
    except Exception as e:
        row["error"] = f"probe:{type(e).__name__}:{e}"
    finally:
        try:
            mdf.close()
        except Exception:
            pass
    return row
