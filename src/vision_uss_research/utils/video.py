from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    opened: bool
    frame_count: int
    fps: float
    width: int
    height: int
    duration_sec: float


def get_video_info(video_path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return VideoInfo(
            path=str(video_path),
            opened=False,
            frame_count=0,
            fps=0.0,
            width=0,
            height=0,
            duration_sec=0.0,
        )

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_sec = frame_count / fps if fps > 0 and frame_count > 0 else 0.0

    cap.release()

    return VideoInfo(
        path=str(video_path),
        opened=True,
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
        duration_sec=duration_sec,
    )


def read_frame_at_ratio(video_path: Path, ratio: float) -> tuple[np.ndarray | None, int, VideoInfo]:
    info = get_video_info(video_path)
    if not info.opened or info.frame_count <= 0:
        return None, -1, info

    ratio = max(0.0, min(1.0, ratio))
    frame_idx = int(round((info.frame_count - 1) * ratio))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, -1, info

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return None, frame_idx, info

    return frame, frame_idx, info


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"Failed to write image to {path}")


def resize_with_padding(image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def draw_text_block(image: np.ndarray, lines: list[str], x: int = 16, y: int = 28) -> np.ndarray:
    out = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness = 2
    line_h = 22

    max_width = 0
    for line in lines:
        (w, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_width = max(max_width, w)

    box_w = max_width + 20
    box_h = line_h * len(lines) + 12

    overlay = out.copy()
    cv2.rectangle(overlay, (8, 8), (8 + box_w, 8 + box_h), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

    yy = y
    for line in lines:
        cv2.putText(out, line, (x, yy), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        yy += line_h

    return out


def make_side_by_side(left: np.ndarray, right: np.ndarray, gap: int = 20) -> np.ndarray:
    left_h, _ = left.shape[:2]
    right_h, _ = right.shape[:2]
    target_h = max(left_h, right_h)

    def resize_to_h(img: np.ndarray, h: int) -> np.ndarray:
        ih, iw = img.shape[:2]
        if ih == h:
            return img
        scale = h / ih
        new_w = max(1, int(round(iw * scale)))
        return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_AREA)

    left_r = resize_to_h(left, target_h)
    right_r = resize_to_h(right, target_h)
    spacer = np.ones((target_h, gap, 3), dtype=np.uint8) * 255
    return np.concatenate([left_r, spacer, right_r], axis=1)