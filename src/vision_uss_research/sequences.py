"""Sequence-folder access shared by all labeling notebooks and scripts.

Camera selection from driving direction, mid-approach frame extraction with a
run-independent cache, and retries for Colab's flaky Drive FUSE mount.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".webm", ".avi"}


def with_drive_retry(fn, *args, retries: int = 2, wait_s: float = 3.0):
    """Colab's Drive FUSE mount throws transient OSError (Errno 5) - retry, then give up."""
    for attempt in range(retries + 1):
        try:
            return fn(*args)
        except OSError:
            if attempt == retries:
                raise
            time.sleep(wait_s)


def sequence_files(seq_dir: Path) -> dict:
    """Key files at the top level of a raw sequence folder."""
    top = [p for p in seq_dir.iterdir() if p.is_file()]

    def pick(pred):
        matches = sorted(p for p in top if pred(p))
        return matches[0] if matches else None

    return {
        "metadata_json": pick(lambda p: p.suffix.lower() == ".json"),
        "label_v2_csv": pick(lambda p: "pas_m_label_v2" in p.name.lower()
                             and p.suffix.lower() == ".csv"),
        "front_video": pick(lambda p: "front" in p.name.lower()
                            and p.suffix.lower() in VIDEO_EXTENSIONS),
        "rear_video": pick(lambda p: "rear" in p.name.lower()
                           and p.suffix.lower() in VIDEO_EXTENSIONS),
        "topview_video": pick(lambda p: "topview" in p.name.lower()
                              and p.suffix.lower() in VIDEO_EXTENSIONS),
    }


def driving_direction(seq_dir: Path) -> str:
    """'forward' / 'backward' from the metadata JSON tags, else 'unknown'."""
    found = sequence_files(seq_dir)
    if not found["metadata_json"]:
        return "unknown"
    with found["metadata_json"].open() as f:
        metadata = json.load(f)
    for tag in metadata.get("tags", []):
        if tag.get("tagSubCategory", {}).get("name", "").lower() == "driving direction":
            return tag.get("name", "unknown")
    return "unknown"


def select_camera_videos(seq_dir: Path) -> tuple[list[tuple[str, Path]], str]:
    """Camera videos to evaluate: the direction-matching one, or BOTH when the
    driving direction is unknown (guessing wrong shows frames where the target
    never appears)."""
    found = sequence_files(seq_dir)
    direction = driving_direction(seq_dir)
    if direction == "backward":
        videos = [("rear", found["rear_video"])]
    elif direction == "forward":
        videos = [("front", found["front_video"])]
    else:
        videos = [("front", found["front_video"]), ("rear", found["rear_video"])]
    return [(c, p) for c, p in videos if p is not None], direction


def frames_window_id(n_frames: int, ratio_range: tuple[float, float]) -> str:
    """Cache key for an extraction window, e.g. r30-85_n20. Putting the window
    parameters in the path lets multiple runs share one frame cache safely."""
    lo, hi = (int(round(r * 100)) for r in ratio_range)
    return f"r{lo}-{hi}_n{n_frames}"


def extract_sequence_frames(frames_root: Path, seq_id: str, camera: str,
                            video_path: Path, n_frames: int,
                            ratio_range: tuple[float, float],
                            jpeg_quality: int = 92) -> Path | None:
    """Extract n_frames JPEGs evenly spaced inside ratio_range into
    frames_root/<window_id>/<seq_id>/<camera>/ (cached: skipped when already
    populated). Returns the frame directory, or None if nothing decoded."""
    out_dir = frames_root / frames_window_id(n_frames, ratio_range) / seq_id / camera
    if out_dir.exists() and len(list(out_dir.glob("*.jpg"))) >= n_frames - 2:
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    duration_ms = (1000.0 * cap.get(cv2.CAP_PROP_FRAME_COUNT)
                   / max(cap.get(cv2.CAP_PROP_FPS), 1e-6))
    ratios = np.linspace(ratio_range[0], ratio_range[1], n_frames)
    kept = 0
    for i, r in enumerate(ratios):
        cap.set(cv2.CAP_PROP_POS_MSEC, r * duration_ms)
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(str(out_dir / f"{i:05d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            kept += 1
    cap.release()
    return out_dir if kept else None


def nearest_frame(frames_dir: Path, preferred_idx: int) -> Path | None:
    """The frame at preferred_idx, or the nearest existing one (extraction can be
    partial on corrupt video segments). None when the folder holds no frames."""
    preferred = frames_dir / f"{preferred_idx:05d}.jpg"
    if preferred.exists():
        return preferred
    available = sorted(frames_dir.glob("*.jpg"))
    if not available:
        return None
    return min(available, key=lambda p: abs(int(p.stem) - preferred_idx))
