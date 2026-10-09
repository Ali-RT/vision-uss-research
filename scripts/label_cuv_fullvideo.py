"""Run the trained detector on EVERY frame of the CUV_ObjectCheck videos.

One decode pass per trace produces three outputs under
outputs/prelabel/<out-name>/:

  per_trace/<side>_<trace>.csv   one row per video frame: picked box (same
                                 corridor rule as prelabel_cuv_objectcheck.py),
                                 detector class/conf, frame Low/High bin vs
                                 tracelist bin, accepted flag
  detections.csv                 every raw detection >= --conf (for eval)
  videos/<side>_<trace>_<label>.mp4
                                 annotated video: picked box green (detector
                                 class agrees with tracelist) / orange
                                 (disagrees), other detections grey, banner
                                 with the frame's Low/High call
  yolo/images, yolo/labels, yolo/data.yaml
                                 YOLO dataset of ACCEPTED frames only (picked
                                 box whose detector class agrees with the
                                 tracelist object). Frames without an accepted
                                 box are left out, not written as negatives:
                                 the object may be visible but missed.

The YOLO class is the tracelist object (fine-grained); frames.csv carries
the height bin and detector class so it can be remapped to the 8-class or
binary scheme. Resumable per trace.

    uv run --with "ultralytics>=8.3" --with openpyxl \
        python scripts/label_cuv_fullvideo.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from prelabel_cuv_objectcheck import (DEFAULT_DATA_ROOT, DEFAULT_WEIGHTS,
                                      OBJECT_MAP, PROJECT_ROOT,
                                      corridor_weight, hydrate,
                                      load_tracelist)

# Height bin of each trained detector class (dataset_v2 binning).
DET_BIN = {"curbstone": "low", "speedbump": "low", "woodenboard": "low",
           "low_other": "low", "pole": "high", "dummychild": "high",
           "bicyclestand": "high", "high_other": "high"}

YOLO_CLASSES = sorted({v[0] for v in OBJECT_MAP.values()})

GREEN, ORANGE, GREY, WHITE, RED = ((60, 200, 60), (0, 150, 255),
                                   (170, 170, 170), (255, 255, 255),
                                   (60, 60, 230))


def pick(dets: list[dict], w: int, h: int, min_score: float) -> dict | None:
    """Best detection under the approach-corridor prior, or None."""
    best, best_score = None, min_score
    for d in dets:
        cx = (d["x0"] + d["x1"]) / 2 / w
        cy = (d["y0"] + d["y1"]) / 2 / h
        score = d["conf"] * corridor_weight(cx, cy)
        if score >= best_score:
            best, best_score = d, score
    return best


def annotate(img: np.ndarray, dets: list[dict], best: dict | None,
             agrees: bool, banner: str, bin_ok: bool | None) -> np.ndarray:
    out = img.copy()
    for d in dets:
        if d is best:
            continue
        cv2.rectangle(out, (int(d["x0"]), int(d["y0"])),
                      (int(d["x1"]), int(d["y1"])), GREY, 1)
        cv2.putText(out, f'{d["cls"]} {d["conf"]:.2f}',
                    (int(d["x0"]), max(12, int(d["y0"]) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREY, 1)
    if best is not None:
        color = GREEN if agrees else ORANGE
        p1 = (int(best["x0"]), int(best["y0"]))
        cv2.rectangle(out, p1, (int(best["x1"]), int(best["y1"])), color, 2)
        cv2.putText(out, f'{best["cls"]} {best["conf"]:.2f}',
                    (p1[0], max(14, p1[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, banner, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                WHITE if bin_ok is None else GREEN if bin_ok else RED, 1)
    return out


def process_trace(t, model, args, out_dir: Path) -> pd.DataFrame:
    label_class, agree_set, height_bin, _ = OBJECT_MAP[t.object_name]
    key = f"{t.side}_{t.trace_id}"
    hydrate(t.avi)
    cap = cv2.VideoCapture(str(t.avi))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    vid_tmp = out_dir / "videos" / f"{key}_{label_class}.part.mp4"
    writer = None
    img_dir = out_dir / "yolo" / "images"
    lbl_dir = out_dir / "yolo" / "labels"
    cls_id = YOLO_CLASSES.index(label_class)

    rows, raw = [], []
    frame_idx = 0
    while True:
        batch = []
        for _ in range(args.batch):
            ok, frame = cap.read()
            if not ok:
                break
            batch.append(frame)
        if not batch:
            break
        results = model.predict(batch, conf=args.conf, imgsz=args.imgsz,
                                device=args.device, verbose=False)
        for frame, res in zip(batch, results):
            h, w = frame.shape[:2]
            if writer is None:
                writer = cv2.VideoWriter(str(vid_tmp),
                                         cv2.VideoWriter_fourcc(*"avc1"),
                                         fps, (w, h))
            dets = []
            for b in res.boxes:
                x0, y0, x1, y1 = b.xyxy[0].tolist()
                dets.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1,
                             "cls": model.names[int(b.cls)],
                             "conf": float(b.conf)})
            for d in dets:
                raw.append({"side": t.side, "trace": t.trace_id,
                            "frame_idx": frame_idx,
                            **{k: round(v, 1) if k[0] in "xy" else v
                               for k, v in d.items()}})
            best = pick(dets, w, h, args.min_score)
            agrees = best is not None and best["cls"] in agree_set
            det_bin = DET_BIN[best["cls"]] if best else None
            bin_ok = None if det_bin is None else det_bin == height_bin

            if agrees:
                stem = f"{key}_{frame_idx:05d}"
                cv2.imwrite(str(img_dir / f"{stem}.jpg"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                cx = (best["x0"] + best["x1"]) / 2 / w
                cy = (best["y0"] + best["y1"]) / 2 / h
                bw = (best["x1"] - best["x0"]) / w
                bh = (best["y1"] - best["y0"]) / h
                (lbl_dir / f"{stem}.txt").write_text(
                    f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            call = (f"{best['cls']} {best['conf']:.2f} -> {det_bin.upper()}"
                    f" ({'OK' if bin_ok else 'WRONG'})" if best else "no box")
            banner = (f"{t.side} {t.trace_id} | {t.object_name} ({height_bin}) "
                      f"| f{frame_idx}/{total} | {call}")
            writer.write(annotate(frame, dets, best, agrees, banner, bin_ok))

            rows.append({
                "side": t.side, "trace": t.trace_id, "object": t.object_name,
                "label_class": label_class, "height_bin": height_bin,
                "speed": t.speed, "frame_idx": frame_idx, "n_frames": total,
                "ratio": round(frame_idx / max(1, total - 1), 4),
                "n_dets": len(dets),
                "x0": round(best["x0"], 1) if best else None,
                "y0": round(best["y0"], 1) if best else None,
                "x1": round(best["x1"], 1) if best else None,
                "y1": round(best["y1"], 1) if best else None,
                "det_class": best["cls"] if best else None,
                "det_conf": round(best["conf"], 3) if best else None,
                "det_bin": det_bin, "bin_correct": bin_ok,
                "accepted": agrees,
            })
            frame_idx += 1
    cap.release()
    if writer is not None:
        writer.release()
        vid_tmp.rename(vid_tmp.with_name(vid_tmp.name.replace(".part", "")))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_trace" / f"{key}.csv", index=False)
    pd.DataFrame(raw, columns=["side", "trace", "frame_idx", "x0", "y0",
                               "x1", "y1", "cls", "conf"]).to_csv(
        out_dir / "per_trace" / f"{key}.dets.csv", index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out-name", default="fullvideo_cuv_v1")
    parser.add_argument("--traces", default="")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--min-score", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "outputs" / "prelabel" / args.out_name
    for sub in ("per_trace", "videos", "yolo/images", "yolo/labels"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    (out_dir / "yolo" / "data.yaml").write_text(
        "# CUV_ObjectCheck auto-labels; class = tracelist object.\n"
        "# Remap via frames.csv (height_bin, det_class) for 8-class/binary.\n"
        f"path: {out_dir / 'yolo'}\ntrain: images\nval: images\n"
        f"names:\n" + "".join(f"  {i}: {n}\n"
                              for i, n in enumerate(YOLO_CLASSES)))

    from ultralytics import YOLO
    model = YOLO(str(args.weights))

    only = {int(x) for x in args.traces.split(",") if x.strip()}
    traces = [t for t in load_tracelist(args.data_root)
              if t.avi is not None and (not only or t.trace_id in only)]
    for i, t in enumerate(traces, 1):
        key = f"{t.side}_{t.trace_id}"
        if (out_dir / "per_trace" / f"{key}.csv").exists():
            continue
        df = process_trace(t, model, args, out_dir)
        print(f"[{i}/{len(traces)}] {key} {t.object_name}: {len(df)} frames, "
              f"box {df.x0.notna().mean():.0%}, accepted {df.accepted.mean():.0%}, "
              f"bin correct {df.bin_correct.eq(True).sum()}/{df.x0.notna().sum()}",
              flush=True)

    frames = pd.concat([pd.read_csv(p) for p in
                        sorted((out_dir / "per_trace").glob("*_[0-9]*.csv"))
                        if not p.name.endswith(".dets.csv")])
    frames.to_csv(out_dir / "frames.csv", index=False)
    pd.concat([pd.read_csv(p) for p in
               sorted((out_dir / "per_trace").glob("*.dets.csv"))]).to_csv(
        out_dir / "detections.csv", index=False)
    (out_dir / "run.json").write_text(json.dumps({
        "run_id": args.out_name, "weights": str(args.weights),
        "conf": args.conf, "min_score": args.min_score, "imgsz": args.imgsz,
        "yolo_classes": YOLO_CLASSES, "n_frames": len(frames),
        "n_accepted": int(frames.accepted.sum())}, indent=1))
    print(f"\n{len(frames)} frames, {int(frames.accepted.sum())} accepted "
          f"-> {out_dir}")


if __name__ == "__main__":
    main()
