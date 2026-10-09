"""Pre-label the CUV_ObjectCheck collection with the trained YOLO detector.

The collection (OneDrive, 2026-09-22) is one staged object per trace:
raw AVI (640x480) + XCP MF4, with `tracelist-objectdata.xlsx` as per-trace
ground truth for object type and approach speed. The detector only
LOCALIZES; the label class always comes from the tracelist (the model
cannot resolve e.g. box height bins from appearance).

Per trace:
  1. resolve object/speed from the tracelist (rear ids typed 2xxx are 3xxx),
  2. extract N frames in the mid-approach ratio window (cached),
  3. run the trained detector, keep the best box per frame under the
     approach-corridor prior (staged target: horizontally central, below
     the horizon) - same prior as scripts/batch_sam3_label.py,
  4. append rows to boxes.csv (schema compatible with the SAM3/click
     labelers so notebooks/04_label_qa_review.ipynb works unchanged, plus
     detector diagnostics), write a per-trace annotated gallery sheet.

Resumable: traces already in boxes.csv are skipped. OneDrive placeholder
files hydrate on first read (~400MB per AVI; full rear side ~23GB).

Examples:
    uv run --with "ultralytics>=8.3" --with openpyxl \
        python scripts/prelabel_cuv_objectcheck.py --traces 3170,3141
    ... --sides rear            # full rear side
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_ROOT = Path(
    "/Users/ral3ply/OneDrive - Bosch Group/Ali/Data_CUV_ObjectCheck")
DEFAULT_WEIGHTS = (PROJECT_ROOT / "outputs/models/20260719_train_yolo_v1"
                   / "train/weights/best.pt")

# Tracelist object -> (canonical label class, detector classes that count as
# agreement, height bin, camera_blind). Label class is per-trace ground truth;
# detector agreement is a QA diagnostic only. Wall and adult dummy were never
# trained classes; metal bar is near-invisible at 640x480 (expect misses).
OBJECT_MAP: dict[str, tuple[str, set[str], str, bool]] = {
    "ISO pole":             ("isopole",       {"pole"},                     "high", False),
    "ADAC Ballon CAR":      ("car",           {"high_other"},               "high", False),
    "Wall Structured":      ("wall",          {"high_other"},               "high", False),
    "Adult Dummy":          ("dummyadult",    {"dummychild", "high_other"}, "high", False),
    "Child Dummy":          ("dummychild",    {"dummychild"},               "high", False),
    "box 30cm":             ("box30",         {"high_other"},               "high", False),
    "box 40cm":             ("box40",         {"high_other"},               "high", False),
    "Curbstone high":       ("curbhigh",      {"curbstone"},                "low",  False),
    "Metal Bar Horizontal": ("metalbar",      {"low_other", "curbstone"},   "low",  True),
    "Wooden Curb 10cm":     ("woodencurb10",  {"curbstone", "woodenboard"}, "low",  False),
}

BOX_COLS = ["sequence_id", "target_object", "camera", "frame_idx", "frame_path",
            "x0", "y0", "x1", "y1", "det_class", "det_conf", "corridor_score",
            "det_agrees", "label_class", "height_bin", "speed", "trace_note"]


def corridor_weight(cx_norm: float, cy_norm: float,
                    sigma: float = 0.25, min_cy: float = 0.35) -> float:
    """Prior that the staged target sits in the approach corridor
    (gaussian around 0.5) and below the horizon band (cy_norm >= min_cy)."""
    horizontal = float(np.exp(-((cx_norm - 0.5) ** 2) / (2 * sigma ** 2)))
    vertical = 1.0 if cy_norm >= min_cy else 0.25
    return horizontal * vertical


@dataclass
class Trace:
    trace_id: int
    side: str          # "front" | "rear"
    object_name: str   # tracelist spelling
    speed: str
    note: str
    avi: Path | None


def fix_trace_id(raw) -> int | None:
    """Rear ids 2126-2174 in the sheet are a typo for 3126-3174."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    tid = int(raw)
    if 2100 <= tid < 3000:
        tid += 1000
    return tid


def load_tracelist(data_root: Path) -> list[Trace]:
    df = pd.read_excel(data_root / "tracelist-objectdata.xlsx")
    df.columns = ["rear", "front", "object", "speed", "extra_trace", "note"]

    avi_index: dict[tuple[str, int], Path] = {}
    for side, folder in (("front", "Front"), ("rear", "Rear")):
        for p in (data_root / folder).glob("*.avi"):
            m = re.search(r"_(\d{4})_", p.name)
            if m:
                avi_index[(side, int(m.group(1)))] = p

    traces: dict[tuple[str, int], Trace] = {}
    for _, row in df.iterrows():
        obj = str(row["object"]).strip()
        speed = str(row["speed"]).strip()
        note = "" if pd.isna(row["note"]) else str(row["note"]).strip()
        for side, col in (("rear", "rear"), ("front", "front")):
            tid = fix_trace_id(row[col])
            if tid is None:
                continue
            key = (side, tid)
            if key in traces:        # duplicate sheet row (e.g. 3120 rerun)
                traces[key].note = (traces[key].note + "; " + note).strip("; ")
                continue
            traces[key] = Trace(tid, side, obj, speed, note,
                                avi_index.get(key))
        # the extra_trace column holds a flagged sibling trace (dropped
        # object / logging issue) on the same side as the front column
        extra = fix_trace_id(row["extra_trace"])
        if extra is not None and ("front", extra) not in traces:
            traces[("front", extra)] = Trace(extra, "front", obj, speed,
                                             note or "flagged in tracelist",
                                             avi_index.get(("front", extra)))
    return sorted(traces.values(), key=lambda t: (t.side, t.trace_id))


def hydrate(path: Path) -> None:
    """Force OneDrive to download a cloud placeholder before cv2 opens it
    (cv2's ffmpeg backend times out at ~45s on a cold placeholder)."""
    st = path.stat()
    if st.st_blocks * 512 >= st.st_size:
        return
    print(f"    hydrating {path.name} ({st.st_size / 1e6:.0f}MB)...")
    with path.open("rb") as f:
        while f.read(1 << 24):
            pass


def extract_frames(avi: Path, out_dir: Path, n_frames: int,
                   ratio_range: tuple[float, float]) -> list[tuple[int, Path]]:
    """Sample n_frames in the approach window; cached on disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cached = sorted(out_dir.glob("*.jpg"))
    if len(cached) == n_frames:
        return [(int(p.stem), p) for p in cached]
    hydrate(avi)

    cap = cv2.VideoCapture(str(avi))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"no frames reported for {avi.name}")
    lo, hi = (int(total * ratio_range[0]), int(total * ratio_range[1]))
    wanted = sorted(set(np.linspace(lo, max(lo, hi - 1), n_frames).astype(int)))

    out: list[tuple[int, Path]] = []
    wanted_set = set(wanted)
    idx = 0
    while idx <= wanted[-1]:
        ok = cap.grab()
        if not ok:
            break
        if idx in wanted_set:
            ok, frame = cap.retrieve()
            if ok:
                p = out_dir / f"{idx:05d}.jpg"
                cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                out.append((idx, p))
        idx += 1
    cap.release()
    if not out:
        raise RuntimeError(f"extracted 0 frames from {avi.name}")
    return out


def pick_box(result, names: dict[int, str], min_score: float):
    """Best detection under the corridor prior; None if nothing passes."""
    h, w = result.orig_shape
    best = None
    for b in result.boxes:
        x0, y0, x1, y1 = b.xyxy[0].tolist()
        cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
        score = float(b.conf) * corridor_weight(cx, cy)
        if score < min_score:
            continue
        if best is None or score > best["corridor_score"]:
            best = {"x0": round(x0, 1), "y0": round(y0, 1),
                    "x1": round(x1, 1), "y1": round(y1, 1),
                    "det_class": names[int(b.cls)],
                    "det_conf": round(float(b.conf), 3),
                    "corridor_score": round(score, 3)}
    return best


def write_gallery(rows: list[dict], frames: list[tuple[int, Path]],
                  out_path: Path, n_tiles: int = 4) -> None:
    picks = np.linspace(0, len(frames) - 1, n_tiles).astype(int)
    tiles = []
    by_idx = {r["frame_idx"]: r for r in rows}
    for i in picks:
        frame_idx, p = frames[i]
        img = cv2.imread(str(p))
        r = by_idx.get(frame_idx)
        if r and r["x0"] is not None:
            pt1 = (int(r["x0"]), int(r["y0"]))
            pt2 = (int(r["x1"]), int(r["y1"]))
            cv2.rectangle(img, pt1, pt2, (0, 220, 0), 2)
            cv2.putText(img, f'{r["det_class"]} {r["det_conf"]:.2f}',
                        (pt1[0], max(14, pt1[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
        cv2.putText(img, f"f{frame_idx}", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        tiles.append(img)
    grid = cv2.vconcat([cv2.hconcat(tiles[:2]), cv2.hconcat(tiles[2:])])
    cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])


def append_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-label CUV_ObjectCheck with the trained detector")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out-name", type=str, default="prelabel_cuv_v1")
    parser.add_argument("--sides", type=str, default="front,rear")
    parser.add_argument("--traces", type=str, default="",
                        help="comma-separated trace ids (default: all)")
    parser.add_argument("--limit", type=int, default=0,
                        help="max traces (0 = all)")
    parser.add_argument("--n-frames", type=int, default=48)
    parser.add_argument("--ratio-range", type=float, nargs=2,
                        default=(0.25, 0.97))
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--min-score", type=float, default=0.05,
                        help="min conf * corridor weight to keep a box")
    parser.add_argument("--dry-run", action="store_true",
                        help="no model: parse tracelist + extract frames only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = PROJECT_ROOT / "outputs" / "prelabel" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    boxes_csv = out_dir / "boxes.csv"
    (out_dir / "gallery").mkdir(exist_ok=True)

    traces = load_tracelist(args.data_root)
    sides = {s.strip() for s in args.sides.split(",")}
    only = {int(t) for t in args.traces.split(",") if t.strip()}
    traces = [t for t in traces if t.side in sides
              and (not only or t.trace_id in only)]
    if args.limit:
        traces = traces[:args.limit]

    done: set[tuple[str, int]] = set()
    if boxes_csv.exists():
        prev = pd.read_csv(boxes_csv, usecols=["sequence_id", "camera"])
        done = {(c, int(s)) for s, c in
                zip(prev["sequence_id"], prev["camera"])}

    unknown = sorted({t.object_name for t in traces
                      if t.object_name not in OBJECT_MAP})
    if unknown:
        sys.exit(f"tracelist objects without OBJECT_MAP entry: {unknown}")

    model = None
    if not args.dry_run:
        from ultralytics import YOLO
        model = YOLO(str(args.weights))

    stats: dict[str, int] = {"labeled": 0, "skipped_done": 0, "no_avi": 0,
                             "failed": 0, "frames_no_box": 0}
    for t in traces:
        if (t.side, t.trace_id) in done:
            stats["skipped_done"] += 1
            continue
        if t.avi is None:
            stats["no_avi"] += 1
            print(f"  {t.side} {t.trace_id} {t.object_name}: no AVI")
            continue
        label_class, agree_set, height_bin, blind = OBJECT_MAP[t.object_name]
        try:
            frames_dir = out_dir / "frames" / f"{t.side}_{t.trace_id}"
            frames = extract_frames(t.avi, frames_dir, args.n_frames,
                                    tuple(args.ratio_range))
            rows = []
            if model is not None:
                results = model.predict([str(p) for _, p in frames],
                                        conf=args.conf, imgsz=args.imgsz,
                                        verbose=False)
                for (frame_idx, p), res in zip(frames, results):
                    box = pick_box(res, model.names, args.min_score)
                    if box is None:
                        stats["frames_no_box"] += 1
                        box = {"x0": None, "y0": None, "x1": None, "y1": None,
                               "det_class": None, "det_conf": None,
                               "corridor_score": None}
                    rows.append({
                        "sequence_id": t.trace_id,
                        "target_object": t.object_name,
                        "camera": t.side,
                        "frame_idx": frame_idx, "frame_path": str(p),
                        **box,
                        "det_agrees": (box["det_class"] in agree_set
                                       if box["det_class"] else False),
                        "label_class": label_class,
                        "height_bin": height_bin,
                        "speed": t.speed, "trace_note": t.note,
                    })
                append_csv(boxes_csv, rows, BOX_COLS)
                write_gallery(rows, frames,
                              out_dir / "gallery" / f"{t.side}_{t.trace_id}_{label_class}.jpg")
            hit = sum(1 for r in rows if r["x0"] is not None)
            agree = sum(1 for r in rows if r["det_agrees"])
            tag = " [camera-blind class]" if blind else ""
            print(f"  {t.side} {t.trace_id} {t.object_name}: "
                  f"{hit}/{len(frames)} boxes, {agree} agree{tag}")
            stats["labeled"] += 1
        except Exception as e:
            stats["failed"] += 1
            append_csv(out_dir / "failures.csv",
                       [{"sequence_id": t.trace_id, "camera": t.side,
                         "error": f"{type(e).__name__}: {e}"}],
                       ["sequence_id", "camera", "error"])
            print(f"  {t.side} {t.trace_id} FAILED: {e}")

    (out_dir / "run.json").write_text(json.dumps({
        "run_id": args.out_name,
        "weights": str(args.weights),
        "data_root": str(args.data_root),
        "n_frames": args.n_frames, "ratio_range": list(args.ratio_range),
        "conf": args.conf, "imgsz": args.imgsz, "min_score": args.min_score,
        "stats": stats}, indent=1))

    print("\nsummary:")
    for k, v in sorted(stats.items()):
        print(f"  {k:<16} {v}")
    print(f"\nboxes   : {boxes_csv}")
    print(f"gallery : {out_dir / 'gallery'}")


if __name__ == "__main__":
    main()
