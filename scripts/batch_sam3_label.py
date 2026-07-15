"""Batch auto-labeling with SAM3 for the text-promptable classes.

Per sequence:
  1. select the evidence camera from driving direction (both cameras when unknown),
  2. extract N frames in the mid-approach window (cached; also the training images),
  3. run SAM3 text-prompted segmentation ONCE on a seed frame,
  4. pick the target instance among distractors with an approach-corridor prior
     (staged target sits roughly ahead of the bumper: horizontally central, below
     the horizon),
  5. propagate that instance's box with the SAM3 tracker across all frames,
  6. append rows to propagated_boxes.csv (same schema as the click labeler, so
     notebooks/label_qa_review.ipynb and the dataset merge work unchanged).

Resumable: already-labeled sequence/cameras are skipped; rows append after each
sequence; failures land in failures.csv; seed diagnostics in seed_log.csv for
filter tuning.

Requires GPU + `pip install -U transformers accelerate` (run in Colab):
    !python scripts/batch_sam3_label.py --profile colab_drive --objects curbstone
Smoke test without GPU/model:
    python scripts/batch_sam3_label.py --profile local --dry-run --limit 2
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vision_uss_research.settings import load_paths

VIDEO_EXTENSIONS = {".mp4", ".webm", ".avi"}
SAM3_MODEL_ID = "facebook/sam3"

# Text prompts per object (appearance only - the class comes from sequence metadata).
OBJECT_PROMPTS = {
    "curbstone": "curb",
    "curbstone_side": "curb",
    "speedbump": "speed bump",
    "pole": "vertical pole",
    "squarepole": "vertical pole",
    "cone": "traffic cone",
    "bollard": "bollard",
    "car": "car",
    "tree": "tree trunk",
    "bush": "bush",
    "stonelarge": "large stone block",
    "stonemiddle": "stone block",
    "cubestandard": "cube-shaped test object",
    "ubarrier": "metal barrier",
    "fence": "fence",
}

BOX_COLS = ["sequence_id", "target_object", "camera", "frame_idx", "frame_path",
            "x0", "y0", "x1", "y1", "mask_area_frac"]


# ---------------------------------------------------------------- sequence I/O

def with_drive_retry(fn, *args, retries=2, wait_s=3):
    for attempt in range(retries + 1):
        try:
            return fn(*args)
        except OSError:
            if attempt == retries:
                raise
            time.sleep(wait_s)


def sequence_files(seq_dir: Path) -> dict:
    top = [p for p in seq_dir.iterdir() if p.is_file()]

    def pick(pred):
        matches = sorted(p for p in top if pred(p))
        return matches[0] if matches else None

    return {
        "metadata_json": pick(lambda p: p.suffix.lower() == ".json"),
        "front_video": pick(lambda p: "front" in p.name.lower()
                            and p.suffix.lower() in VIDEO_EXTENSIONS),
        "rear_video": pick(lambda p: "rear" in p.name.lower()
                           and p.suffix.lower() in VIDEO_EXTENSIONS),
    }


def select_camera_videos(seq_dir: Path) -> list[tuple[str, Path]]:
    found = sequence_files(seq_dir)
    direction = "unknown"
    if found["metadata_json"]:
        with found["metadata_json"].open() as f:
            metadata = json.load(f)
        for tag in metadata.get("tags", []):
            if tag.get("tagSubCategory", {}).get("name", "").lower() == "driving direction":
                direction = tag.get("name", "unknown")
    if direction == "backward":
        videos = [("rear", found["rear_video"])]
    elif direction == "forward":
        videos = [("front", found["front_video"])]
    else:
        videos = [("front", found["front_video"]), ("rear", found["rear_video"])]
    return [(c, p) for c, p in videos if p is not None]


def extract_sequence_frames(frames_root: Path, seq_id: str, camera: str,
                            video_path: Path, n_frames: int,
                            ratio_range: tuple[float, float]) -> Path | None:
    out_dir = frames_root / seq_id / camera
    if out_dir.exists() and len(list(out_dir.glob("*.jpg"))) >= n_frames - 2:
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    duration_ms = 1000.0 * cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1e-6)
    ratios = np.linspace(ratio_range[0], ratio_range[1], n_frames)
    kept = 0
    for i, r in enumerate(ratios):
        cap.set(cv2.CAP_PROP_POS_MSEC, r * duration_ms)
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(str(out_dir / f"{i:05d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            kept += 1
    cap.release()
    return out_dir if kept else None


# ------------------------------------------------------- target instance choice

def corridor_weight(cx_norm: float, cy_norm: float,
                    sigma: float = 0.25, min_cy: float = 0.35) -> float:
    """Prior for 'the staged target ahead of the bumper': horizontally central
    (gaussian around 0.5) and below the horizon band (cy_norm >= min_cy)."""
    horizontal = float(np.exp(-((cx_norm - 0.5) ** 2) / (2 * sigma ** 2)))
    vertical = 1.0 if cy_norm >= min_cy else 0.25
    return horizontal * vertical


def pick_target_instance(result: dict, image_shape: tuple[int, int]) -> tuple[int | None, dict]:
    """result: {'masks': (N,H,W) bool, 'boxes': (N,4) xyxy, 'scores': (N,)}.
    Returns (index of best instance or None, diagnostics)."""
    h, w = image_shape
    n = len(result["scores"])
    if n == 0:
        return None, {"n_candidates": 0}
    combined = []
    for i in range(n):
        x0, y0, x1, y1 = result["boxes"][i]
        cx_norm, cy_norm = ((x0 + x1) / 2) / w, ((y0 + y1) / 2) / h
        combined.append(float(result["scores"][i])
                        * corridor_weight(cx_norm, cy_norm))
    best = int(np.argmax(combined))
    return best, {
        "n_candidates": n,
        "seed_score": round(float(result["scores"][best]), 3),
        "combined_score": round(combined[best], 3),
    }


def masks_to_bbox(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


# ------------------------------------------------------------------ SAM3 layer

class Sam3Labeler:
    """Text-prompted seed segmentation + tracker propagation (GPU)."""

    def __init__(self, device: str, score_threshold: float):
        import torch
        from transformers import (Sam3Model, Sam3Processor,
                                  Sam3TrackerVideoModel, Sam3TrackerVideoProcessor)

        self.torch = torch
        self.device = device
        self.dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.score_threshold = score_threshold
        self.pcs_processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID)
        self.pcs_model = Sam3Model.from_pretrained(SAM3_MODEL_ID).to(device).eval()
        self.tracker = (Sam3TrackerVideoModel.from_pretrained(SAM3_MODEL_ID)
                        .to(device, dtype=self.dtype).eval())
        self.tracker_processor = Sam3TrackerVideoProcessor.from_pretrained(SAM3_MODEL_ID)

    def free(self):
        gc.collect()
        if self.device == "cuda":
            self.torch.cuda.empty_cache()

    def segment_text(self, image_rgb: np.ndarray, text_prompt: str) -> dict:
        from PIL import Image
        torch = self.torch
        pil = Image.fromarray(image_rgb)
        with torch.inference_mode():
            inputs = self.pcs_processor(images=pil, text=text_prompt,
                                        return_tensors="pt").to(self.device)  # API line
            outputs = self.pcs_model(**inputs)
            results = self.pcs_processor.post_process_instance_segmentation(  # API line
                outputs, threshold=self.score_threshold, mask_threshold=0.5,
                target_sizes=[(pil.height, pil.width)],
            )[0]
        masks = results["masks"].cpu().numpy().astype(bool)
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        return {"masks": masks, "boxes": boxes, "scores": scores}

    def track_forward(self, frames: list, seed_idx: int,
                      seed_box: tuple) -> dict[int, np.ndarray]:
        torch = self.torch
        h, w = frames[0].height, frames[0].width
        with torch.inference_mode():
            session = self.tracker_processor.init_video_session(          # API line
                video=frames, inference_device=self.device, dtype=self.dtype)
            self.tracker_processor.add_inputs_to_inference_session(       # API line
                session, frame_idx=seed_idx, obj_ids=1,
                input_boxes=[[list(map(float, seed_box))]])
            masks = {}
            for output in self.tracker.propagate_in_video_iterator(       # API line
                    session, start_frame_idx=seed_idx):
                video_masks = self.tracker_processor.post_process_masks(
                    [output.pred_masks], original_sizes=[[h, w]], binarize=True)[0]
                masks[output.frame_idx] = video_masks[0, 0].cpu().numpy().astype(bool)
        return masks

    def propagate(self, frames_dir: Path, seed_idx: int, seed_box: tuple) -> dict[int, np.ndarray]:
        from PIL import Image
        frames = [Image.open(p).convert("RGB")
                  for p in sorted(frames_dir.glob("*.jpg"))]
        results = self.track_forward(frames, seed_idx, seed_box)
        self.free()
        if seed_idx > 0:
            rev = self.track_forward(list(reversed(frames)),
                                     len(frames) - 1 - seed_idx, seed_box)
            for ridx, mask in rev.items():
                results.setdefault(len(frames) - 1 - ridx, mask)
        self.free()
        return results


# ------------------------------------------------------------------------ main

def append_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch SAM3 text-prompt labeling")
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--objects", type=str, default="curbstone,speedbump,pole",
                        help="comma-separated object names (must exist in OBJECT_PROMPTS)")
    parser.add_argument("--inventory", type=Path, default=None,
                        help="Default: outputs/profiles/object_inventory/inventory_folders.csv")
    parser.add_argument("--out-name", type=str, default="sam3_text_labels")
    parser.add_argument("--raw-root", type=Path, default=None,
                        help="Optional override for the raw data root")
    parser.add_argument("--limit", type=int, default=0,
                        help="max sequences per object (0 = all)")
    parser.add_argument("--n-frames", type=int, default=20)
    parser.add_argument("--ratio-range", type=float, nargs=2, default=(0.30, 0.85))
    parser.add_argument("--seed-ratio", type=float, default=0.60)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true",
                        help="no model: select sequences + extract frames only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_paths(profile=args.profile)

    objects = [o.strip() for o in args.objects.split(",") if o.strip()]
    unknown = [o for o in objects if o not in OBJECT_PROMPTS]
    if unknown:
        sys.exit(f"no prompt defined for: {unknown} - add to OBJECT_PROMPTS")

    raw_root = args.raw_root.resolve() if args.raw_root else paths.raw_data_root
    inventory_csv = (args.inventory or
                     paths.outputs_dir / "profiles" / "object_inventory" / "inventory_folders.csv")
    out_dir = paths.outputs_dir / "experiments" / args.out_name
    frames_root = out_dir / "frames"
    boxes_csv = out_dir / "propagated_boxes.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    inv = pd.read_csv(inventory_csv)
    usable = inv[(inv["status"] == "complete")
                 & ((inv["has_front_video"] == 1) | (inv["has_rear_video"] == 1))]

    rng = random.Random(args.seed)
    work: list[dict] = []
    for obj in objects:
        mask = usable["label_v2_objects"].fillna("").str.split("|").apply(
            lambda names: any(n.strip().lower() == obj for n in names))
        candidates = sorted(usable[mask]["sequence_id"].tolist())
        if args.limit:
            candidates = sorted(rng.sample(candidates, min(args.limit, len(candidates))))
        work += [{"sequence_id": s, "target_object": obj} for s in candidates]
        print(f"{obj:<15} {len(candidates)} sequences queued")

    done_keys: set[str] = set()
    if boxes_csv.exists():
        prev = pd.read_csv(boxes_csv)
        done_keys = set(prev["sequence_id"].astype(str) + "::" + prev["camera"].astype(str))
        print(f"resuming: {len(done_keys)} sequence/cameras already labeled")

    labeler = None
    if not args.dry_run:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            print("WARNING: no GPU - this will be extremely slow")
        labeler = Sam3Labeler(device, args.score_threshold)

    seed_idx = int(round((args.seed_ratio - args.ratio_range[0])
                         / (args.ratio_range[1] - args.ratio_range[0])
                         * (args.n_frames - 1)))
    stats = defaultdict(int)

    for item in tqdm(work, desc="Labeling sequences", unit="seq"):
        seq_id, obj = item["sequence_id"], item["target_object"]
        seq_dir = raw_root / seq_id
        prompt = OBJECT_PROMPTS[obj]
        try:
            videos = with_drive_retry(select_camera_videos, seq_dir)
        except Exception as e:
            append_csv(out_dir / "failures.csv",
                       [{"sequence_id": seq_id, "camera": "", "stage": "select_camera",
                         "error": f"{type(e).__name__}: {e}"}],
                       ["sequence_id", "camera", "stage", "error"])
            stats["failed"] += 1
            continue

        for camera, video_path in videos:
            key = f"{seq_id}::{camera}"
            if key in done_keys:
                stats["skipped_done"] += 1
                continue
            try:
                frames_dir = with_drive_retry(
                    extract_sequence_frames, frames_root, seq_id, camera,
                    video_path, args.n_frames, tuple(args.ratio_range))
                if frames_dir is None:
                    raise RuntimeError("no frames extracted")
                seed_path = frames_dir / f"{seed_idx:05d}.jpg"
                if not seed_path.exists():
                    available = sorted(frames_dir.glob("*.jpg"))
                    if not available:
                        raise RuntimeError("empty frames dir")
                    seed_path = min(available,
                                    key=lambda p: abs(int(p.stem) - seed_idx))
                actual_seed_idx = int(seed_path.stem)

                if args.dry_run:
                    stats["dry_ok"] += 1
                    continue

                image_rgb = cv2.cvtColor(cv2.imread(str(seed_path)), cv2.COLOR_BGR2RGB)
                result = labeler.segment_text(image_rgb, prompt)
                best, diag = pick_target_instance(result, image_rgb.shape[:2])
                append_csv(out_dir / "seed_log.csv",
                           [{"sequence_id": seq_id, "camera": camera,
                             "target_object": obj, "seed_frame_idx": actual_seed_idx,
                             **diag}],
                           ["sequence_id", "camera", "target_object",
                            "seed_frame_idx", "n_candidates", "seed_score",
                            "combined_score"])
                if best is None:
                    stats["no_seed_detection"] += 1
                    continue
                seed_box = tuple(map(float, result["boxes"][best]))

                masks = labeler.propagate(frames_dir, actual_seed_idx, seed_box)
                rows = []
                for frame_idx, mask in sorted(masks.items()):
                    bbox = masks_to_bbox(mask)
                    rows.append({
                        "sequence_id": seq_id, "target_object": obj, "camera": camera,
                        "frame_idx": frame_idx,
                        "frame_path": str(frames_dir / f"{frame_idx:05d}.jpg"),
                        "x0": bbox[0] if bbox else None,
                        "y0": bbox[1] if bbox else None,
                        "x1": bbox[2] if bbox else None,
                        "y1": bbox[3] if bbox else None,
                        "mask_area_frac": float(mask.mean()),
                    })
                append_csv(boxes_csv, rows, BOX_COLS)
                done_keys.add(key)
                stats["labeled"] += 1
            except Exception as e:
                if labeler is not None:
                    labeler.free()
                append_csv(out_dir / "failures.csv",
                           [{"sequence_id": seq_id, "camera": camera, "stage": "label",
                             "error": f"{type(e).__name__}: {e}"}],
                           ["sequence_id", "camera", "stage", "error"])
                stats["failed"] += 1

    print("\nsummary:")
    for k, v in sorted(stats.items()):
        print(f"  {k:<18} {v}")
    print(f"\nboxes    : {boxes_csv}")
    print(f"seed log : {out_dir / 'seed_log.csv'}")
    print(f"failures : {out_dir / 'failures.csv'}")
    print("\nNext: add this boxes CSV to LABEL_SOURCES in notebooks/label_qa_review.ipynb")


if __name__ == "__main__":
    main()
