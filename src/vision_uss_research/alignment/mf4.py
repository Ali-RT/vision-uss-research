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
    Returns NaN when the signal never has a valid reading."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    good = ~np.isnan(values)
    if good.sum() == 0:
        return np.full(np.shape(mf4_seconds), np.nan, dtype=float)
    return np.interp(mf4_seconds, times[good], values[good])


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


def video_frame_count(video_path: Path) -> int:
    """Frame count from the container header (cheap - no decode)."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


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
