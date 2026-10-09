"""Inference-scale sweep on CUV_ObjectCheck (every Nth frame of every trace).

The full-video run (imgsz 960) misses large near objects that the same model
finds at smaller inference sizes: the 640x480 frames upscaled to 960 make
near objects larger than anything in training. This sweep reruns the picking
rule of label_cuv_fullvideo.py at several sizes plus a 480+960 union and
joins the distance from eval/frames_dist.csv, so configs compare on HIT rate
at the spec distance and on wrong-bin picks.

    uv run --with "ultralytics>=8.3" --with tabulate \
        python scripts/scale_sweep_cuv.py --step 4
"""

from __future__ import annotations

import argparse
import glob

import cv2
import numpy as np
import pandas as pd

from eval_cuv_objectcheck import SPEC, SPEC_WINDOW
from label_cuv_fullvideo import DET_BIN, pick
from prelabel_cuv_objectcheck import (DEFAULT_DATA_ROOT, DEFAULT_WEIGHTS,
                                      PROJECT_ROOT)

CONFIGS = {"960": [960], "640": [640], "480": [480], "480+960": [480, 960]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="fullvideo_cuv_v1")
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--min-score", type=float, default=0.05)
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(str(DEFAULT_WEIGHTS))
    run_dir = PROJECT_ROOT / "outputs" / "prelabel" / args.run
    fd = pd.read_csv(run_dir / "eval" / "frames_dist.csv")
    fd = fd[fd["moved"]]

    rows = []
    for (side, trace), g in fd.groupby(["side", "trace"]):
        folder = "Front" if side == "front" else "Rear"
        avi = glob.glob(f"{DEFAULT_DATA_ROOT}/{folder}/*_{trace}_*.avi")[0]
        want = set(g["frame_idx"][::args.step])
        cap = cv2.VideoCapture(avi)
        frames, idxs, i = [], [], 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if i in want:
                frames.append(cap.retrieve()[1])
                idxs.append(i)
            i += 1
        cap.release()
        dets = {}
        for sz in (480, 640, 960):
            out = []
            for b0 in range(0, len(frames), 32):
                for res in model.predict(frames[b0:b0 + 32], imgsz=sz,
                                         conf=args.conf, device="mps",
                                         verbose=False):
                    out.append([{"x0": x0, "y0": y0, "x1": x1, "y1": y1,
                                 "cls": model.names[int(c)], "conf": float(p)}
                                for (x0, y0, x1, y1), c, p in
                                zip(res.boxes.xyxy.tolist(),
                                    res.boxes.cls.tolist(),
                                    res.boxes.conf.tolist())])
            dets[sz] = out
        h, w = frames[0].shape[:2]
        meta = g.set_index("frame_idx")
        for k, fidx in enumerate(idxs):
            r = meta.loc[fidx]
            row = {"side": side, "trace": trace, "frame_idx": fidx,
                   "label_class": r["label_class"], "height_bin": r["height_bin"],
                   "dist_m": r["dist_m"]}
            for name, sizes in CONFIGS.items():
                best = pick([d for s in sizes for d in dets[s][k]], w, h,
                            args.min_score)
                row[f"bin_{name}"] = DET_BIN[best["cls"]] if best else None
            rows.append(row)
        print(f"{side}_{trace}: {len(idxs)} frames", flush=True)

    df = pd.DataFrame(rows)
    out_dir = run_dir / "eval"
    df.to_csv(out_dir / "scale_sweep_frames.csv", index=False)

    summary = []
    for label, g in df.groupby("label_class"):
        spec_d, spec_rate = SPEC[label]
        win = g[(g["dist_m"] <= spec_d) & (g["dist_m"] >= spec_d - SPEC_WINDOW)]
        row = {"label_class": label, "spec_m": spec_d, "spec_rate": spec_rate}
        for name in CONFIGS:
            col = f"bin_{name}"
            per_trace = win.groupby(["side", "trace"]).apply(
                lambda x: (x[col] == x["height_bin"]).mean() >= 0.5)
            boxed = g[g[col].notna()]
            row[f"spec_{name}"] = f"{int(per_trace.sum())}/{len(per_trace)}"
            row[f"wrongbin_{name}"] = round(float((boxed[col] != boxed["height_bin"]).mean()), 2) if len(boxed) else None
        summary.append(row)
    s = pd.DataFrame(summary)
    s.to_csv(out_dir / "scale_sweep_summary.csv", index=False)
    print(s.to_markdown(index=False))

    bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 20.0]
    df["dist_bin"] = pd.cut(df["dist_m"], bins)
    for name in CONFIGS:
        df[f"hit_{name}"] = df[f"bin_{name}"] == df["height_bin"]
    by = df.groupby(["label_class", "dist_bin"], observed=True)[
        [f"hit_{n}" for n in CONFIGS]].mean().round(2)
    by.to_csv(out_dir / "scale_sweep_by_distance.csv")


if __name__ == "__main__":
    main()
