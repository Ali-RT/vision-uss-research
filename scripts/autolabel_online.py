"""Run the online detect + track auto-labeler over the CUV bundle.

Every trace is one job with a known object category (tracelist). The category
maps to the detector classes the target may carry (config `class_map`;
categories outside the detector's classes fall back to all classes of their
height bin). Writes outputs/autolabel_online/<run>/: per-trace CSVs, labels.csv,
summary.csv (coverage by distance band vs the existing post-processed labels,
target track switches) and, with --render, annotated videos (green = matched
detection, yellow = motion prediction, track id in the label).

    uv run --with "ultralytics>=8.4" --with "lap>=0.5.12" \\
        python scripts/autolabel_online.py --render
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_uss_research.cuv_finetune import DET_BIN  # noqa: E402
from vision_uss_research.online_labeler import OnlineLabeler  # noqa: E402

BANDS = [0, 1, 2, 4, 8]


def wanted_classes(category: str, height_bin: str, class_map: dict) -> set[str]:
    mapped = class_map.get(category)
    if mapped:
        return {mapped}
    return {c for c, b in DET_BIN.items() if b == height_bin}


def render(rows: pd.DataFrame, frame_dir: Path, out_path: Path, title: str) -> None:
    writer = None
    for r in rows.itertuples():
        img = cv2.imread(str(frame_dir / f"{r.frame_idx:05d}.jpg"))
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"), 30, (w, h))
        if isinstance(r.src, str):
            color = (60, 200, 60) if r.src == "det" else (0, 220, 255)
            cv2.rectangle(img, (int(r.x0), int(r.y0)), (int(r.x1), int(r.y1)), color, 2)
            cv2.putText(img, f"id{int(r.track_id)} {r.src} {r.det_cls} {r.det_score:.2f}",
                        (int(r.x0), max(14, int(r.y0) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(img, f"{title} | f{r.frame_idx} | {r.dist_m:.2f} m | {r.src if isinstance(r.src, str) else 'no label'}",
                    (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(img)
    if writer is not None:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", type=Path,
                        default=PROJECT_ROOT / "outputs/cuv_objectcheck/cuv_bundle_v1")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT /
                        "outputs/models/train_cuv_v1_yolo11s_ft/train/weights/best.pt")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT /
                        "configs/experiments/cuv_finetune_v1.yaml")
    parser.add_argument("--out-name", default="online_v1")
    parser.add_argument("--traces", default="", help="comma-separated trace ids")
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--max-predict", type=int, default=15)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO
    cfg = yaml.safe_load(args.config.read_text())
    out_dir = PROJECT_ROOT / "outputs" / "autolabel_online" / args.out_name
    (out_dir / "per_trace").mkdir(parents=True, exist_ok=True)
    if args.render:
        (out_dir / "videos").mkdir(exist_ok=True)

    frames = pd.read_csv(args.bundle / "frames.csv")
    only = {int(t) for t in args.traces.split(",") if t.strip()}
    model = YOLO(str(args.weights))
    parts = []
    for (side, trace), g in frames.groupby(["side", "trace"]):
        if only and trace not in only:
            continue
        key = f"{side}_{trace}"
        csv = out_dir / "per_trace" / f"{key}.csv"
        g = g.sort_values("frame_idx")
        category, hbin = g["label_class"].iloc[0], g["height_bin"].iloc[0]
        if not csv.exists():
            labeler = OnlineLabeler(model, wanted_classes(category, hbin, cfg["class_map"]),
                                    imgsz=args.imgsz, device=args.device,
                                    max_predict=args.max_predict, category=category)
            fdir = args.bundle / "frames" / key
            rows = [labeler.step(cv2.imread(str(fdir / f"{f:05d}.jpg")), int(f))
                    for f in g["frame_idx"]]
            res = pd.DataFrame(rows).merge(
                g[["frame_idx", "dist_m", "label_src", "x0", "y0", "x1", "y1"]]
                .rename(columns={"label_src": "old_src", "x0": "old_x0", "y0": "old_y0",
                                 "x1": "old_x1", "y1": "old_y1"}), on="frame_idx")
            res.insert(0, "side", side)
            res.insert(1, "trace", trace)
            res.insert(2, "category", category)
            res.to_csv(csv, index=False)
            if args.render:
                render(res, fdir, out_dir / "videos" / f"{key}_{category}.mp4",
                       f"{side} {trace} | {category}")
            switches = res["track_id"].dropna().diff().fillna(0).ne(0).sum()
            print(f"{key} {category}: {res['src'].notna().mean():.0%} labeled "
                  f"({(res['src'] == 'pred').sum()} predicted), {switches} target switches",
                  flush=True)
        parts.append(pd.read_csv(csv))

    lab = pd.concat(parts, ignore_index=True)
    lab.to_csv(out_dir / "labels.csv", index=False)
    lab["band"] = pd.cut(lab["dist_m"], BANDS)
    cov = (lab.groupby(["category", "band"], observed=True)
           .agg(frames=("frame_idx", "size"),
                online=("src", lambda s: s.notna().mean()),
                online_pred=("src", lambda s: (s == "pred").mean()),
                old_pipeline=("old_src", lambda s: s.notna().mean()))
           .round(2))
    sw = (lab.dropna(subset=["track_id"]).groupby(["side", "trace", "category"])["track_id"]
          .agg(lambda s: int(s.diff().fillna(0).ne(0).sum())).rename("target_switches"))
    cov.to_csv(out_dir / "coverage.csv")
    sw.to_csv(out_dir / "switches.csv")
    print("\ncoverage by distance band (share of frames with a label):")
    print(cov.unstack("band").to_string())
    print(f"\ntraces with target switches: {(sw > 0).sum()}/{len(sw)}; total {int(sw.sum())}")


if __name__ == "__main__":
    main()
