"""Offline single-object tracking + gap filling for CUV_ObjectCheck labels.

The detector labels every frame independently (label_cuv_fullvideo.py), so
boxes flicker. Each trace has exactly one staged object, so the per-frame
boxes are linked into ONE track:

  1. anchors   = frames whose picked box class agrees with the tracelist
                 (`accepted` in per_trace/<key>.csv);
  2. exit      = an object leaving through the bottom edge (box touches the
                 bottom but not the top) has a top edge that keeps moving
                 down; once a bottom-touching box jumps back up, the object
                 has left and later bottom-touching boxes are false
                 positives (e.g. the asphalt strip after a curb exits), until
                 a box is fully in view again. Only for LOW_PROFILE objects
                 (top below camera height); for taller objects the top edge
                 rises while approaching, so the rule does not apply;
     spikes    = an anchor that overlaps neither neighbouring anchor while the
                 neighbours overlap each other is dropped;
  3. gap fill  = between two consecutive anchors at most --max-gap frames
                 apart whose boxes overlap (IoU >= --min-bridge-iou), each
                 gap frame takes a raw detection of any class matching the
                 interpolated box (IoU >= --rescue-iou, "rescued"), else the
                 linearly interpolated box ("interp").

Offline only: gap filling uses future frames, so it is a labeling tool, not
a model of what the camera knows at run time.

Writes <run>/tracked/: per-trace and combined frames_tracked.csv,
stats.csv, yolo/ (images + labels for every tracked frame), and videos/
(green = detector, cyan = rescued detection, yellow = interpolated).

    uv run --with openpyxl python scripts/track_cuv_labels.py \
        --run fullvideo_cuv_v2_480
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from label_cuv_fullvideo import YOLO_CLASSES
from prelabel_cuv_objectcheck import DEFAULT_DATA_ROOT, PROJECT_ROOT, hydrate

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from vision_uss_research.cuv_finetune import (BOX, LOW_PROFILE,  # noqa: E402
                                              drop_after_exit, drop_spikes, iou)

COLORS = {"det": (60, 200, 60), "rescued": (230, 200, 0),
          "interp": (0, 220, 255)}


def track_trace(fr: pd.DataFrame, dets: pd.DataFrame,
                args) -> pd.DataFrame:
    fr = fr.sort_values("frame_idx").reset_index(drop=True)
    fr["track_src"] = None
    for c in BOX:
        fr[f"t_{c}"] = np.nan

    anchors = fr[fr["accepted"]]
    a_frames = anchors["frame_idx"].to_numpy()
    a_boxes = anchors[BOX].to_numpy(float)
    keep = (drop_after_exit(a_boxes, args.img_h, args.exit_tol)
            if fr["label_class"].iloc[0] in LOW_PROFILE
            else np.ones(len(a_boxes), bool))
    n_exit = int((~keep).sum())
    a_frames, a_boxes = a_frames[keep], a_boxes[keep]
    keep = drop_spikes(a_frames, a_boxes, args.max_gap, args.spike_iou)
    a_frames, a_boxes = a_frames[keep], a_boxes[keep]
    n_spikes = int((~keep).sum())

    pos = {f: i for i, f in enumerate(fr["frame_idx"])}
    for f, b in zip(a_frames, a_boxes):
        fr.loc[pos[f], [f"t_{c}" for c in BOX]] = b
        fr.loc[pos[f], "track_src"] = "det"

    by_frame = {f: g[BOX].to_numpy(float) for f, g in dets.groupby("frame_idx")}
    for (fa, ba), (fb, bb) in zip(zip(a_frames, a_boxes),
                                  zip(a_frames[1:], a_boxes[1:])):
        gap = fb - fa - 1
        if gap < 1 or gap > args.max_gap or iou(ba, bb) < args.min_bridge_iou:
            continue
        for f in range(fa + 1, fb):
            w = (f - fa) / (fb - fa)
            interp = (1 - w) * ba + w * bb
            src, box = "interp", interp
            cands = by_frame.get(f)
            if cands is not None:
                ious = [iou(interp, c) for c in cands]
                k = int(np.argmax(ious))
                if ious[k] >= args.rescue_iou:
                    src, box = "rescued", cands[k]
            fr.loc[pos[f], [f"t_{c}" for c in BOX]] = box
            fr.loc[pos[f], "track_src"] = src
    fr.attrs["n_spikes"], fr.attrs["n_exit"] = n_spikes, n_exit
    return fr


def render(fr: pd.DataFrame, avi: Path, out_dir: Path, key: str,
           label_class: str, title: str) -> None:
    hydrate(avi)
    cap = cv2.VideoCapture(str(avi))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cls_id = YOLO_CLASSES.index(label_class)
    rows = fr.set_index("frame_idx")
    writer, idx = None, 0
    tmp = out_dir / "videos" / f"{key}_{label_class}.part.mp4"
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if writer is None:
            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"avc1"),
                                     fps, (w, h))
        r = rows.loc[idx] if idx in rows.index else None
        src = r["track_src"] if r is not None else None
        vis = frame.copy()
        if src:
            x0, y0, x1, y1 = (float(r[f"t_{c}"]) for c in BOX)
            stem = f"{key}_{idx:05d}"
            cv2.imwrite(str(out_dir / "yolo" / "images" / f"{stem}.jpg"),
                        frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (out_dir / "yolo" / "labels" / f"{stem}.txt").write_text(
                f"{cls_id} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
                f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}\n")
            cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)),
                          COLORS[src], 2)
            cv2.putText(vis, src, (int(x0), max(14, int(y0) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[src], 2)
        cv2.rectangle(vis, (0, 0), (w, 24), (0, 0, 0), -1)
        cv2.putText(vis, f"{title} | f{idx} | {src or 'no box'}", (6, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(vis)
        idx += 1
    cap.release()
    if writer is not None:
        writer.release()
        tmp.rename(tmp.with_name(tmp.name.replace(".part", "")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="fullvideo_cuv_v2_480")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--max-gap", type=int, default=20,
                        help="longest gap (frames) to fill; 20 = 0.67 s")
    parser.add_argument("--min-bridge-iou", type=float, default=0.3)
    parser.add_argument("--rescue-iou", type=float, default=0.5)
    parser.add_argument("--spike-iou", type=float, default=0.1)
    parser.add_argument("--img-h", type=int, default=480)
    parser.add_argument("--exit-tol", type=float, default=4.0,
                        help="px a bottom-edge box top may move up before "
                             "the object counts as exited")
    parser.add_argument("--no-render", action="store_true",
                        help="stats only, skip yolo/ and videos/")
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / "outputs" / "prelabel" / args.run
    out_dir = run_dir / "tracked"
    for sub in ("per_trace", "videos", "yolo/images", "yolo/labels"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    (out_dir / "yolo" / "data.yaml").write_text(
        "# CUV_ObjectCheck tracked labels; class = tracelist object.\n"
        f"path: {out_dir / 'yolo'}\ntrain: images\nval: images\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(YOLO_CLASSES)))

    stats, all_fr = [], []
    files = [p for p in sorted(glob.glob(str(run_dir / "per_trace" / "*.csv")))
             if not p.endswith(".dets.csv")]
    for p in files:
        fr = pd.read_csv(p)
        side, trace = fr["side"].iloc[0], int(fr["trace"].iloc[0])
        key = f"{side}_{trace}"
        dets = pd.read_csv(p.replace(".csv", ".dets.csv"))
        tr = track_trace(fr, dets, args)
        tr.to_csv(out_dir / "per_trace" / f"{key}.csv", index=False)
        all_fr.append(tr)

        src = tr["track_src"].value_counts()
        stats.append({"key": key, "label_class": tr["label_class"].iloc[0],
                      "frames": len(tr), "accepted": int(tr["accepted"].sum()),
                      "spikes_dropped": tr.attrs["n_spikes"],
                      "exit_dropped": tr.attrs["n_exit"],
                      "det": int(src.get("det", 0)),
                      "rescued": int(src.get("rescued", 0)),
                      "interp": int(src.get("interp", 0))})
        if not args.no_render:
            folder = "Front" if side == "front" else "Rear"
            avi = next((args.data_root / folder).glob(f"*_{trace}_*.avi"))
            render(tr, avi, out_dir, key, tr["label_class"].iloc[0],
                   f"{side} {trace} | {tr['object'].iloc[0]}")
        print(f"{key}: det {stats[-1]['det']}, rescued {stats[-1]['rescued']}, "
              f"interp {stats[-1]['interp']}, spikes {stats[-1]['spikes_dropped']}",
              flush=True)

    pd.concat(all_fr).to_csv(out_dir / "frames_tracked.csv", index=False)
    s = pd.DataFrame(stats)
    s.to_csv(out_dir / "stats.csv", index=False)
    g = s.groupby("label_class")[["accepted", "exit_dropped",
                                  "spikes_dropped", "det",
                                  "rescued", "interp"]].sum()
    g["labels_total"] = g["det"] + g["rescued"] + g["interp"]
    g["gain_pct"] = (100 * (g["labels_total"] - g["accepted"])
                     / g["accepted"].clip(lower=1)).round(1)
    print(g.to_string())


if __name__ == "__main__":
    main()
