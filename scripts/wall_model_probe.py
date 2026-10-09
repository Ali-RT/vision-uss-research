"""Probe newer Ultralytics models on CUV_ObjectCheck wall traces.

Our detector never boxes the staged wall (wall was never a training class).
This checks whether off-the-shelf YOLO26-family models do: the stock COCO
detector, the open-vocabulary YOLOE-26 with text prompts, and its
prompt-free variant. Frames are taken at fixed distances (from the eval's
frames_dist.csv) on the wall traces, plus non-wall traces at 1.6 m to count
false "wall" calls.

    uv run --with "ultralytics>=8.4" python scripts/wall_model_probe.py
"""

from __future__ import annotations

import glob

import cv2
import pandas as pd

from prelabel_cuv_objectcheck import DEFAULT_DATA_ROOT, PROJECT_ROOT

RUN = PROJECT_ROOT / "outputs" / "prelabel" / "fullvideo_cuv_v2_480"
OUT = RUN / "wall_probe"
WALL_DISTS = (4.0, 2.5, 1.6, 1.0, 0.5)
PROMPTS = ["wall", "car", "person", "box", "pole", "curb"]


def grab(side: str, trace: int, frame_idx: int):
    folder = "Front" if side == "front" else "Rear"
    avi = glob.glob(f"{DEFAULT_DATA_ROOT}/{folder}/*_{trace}_*.avi")[0]
    cap = cv2.VideoCapture(avi)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def sample_frames() -> list[dict]:
    fd = pd.read_csv(RUN / "eval" / "frames_dist.csv")
    fd = fd[fd["moved"]]
    out = []
    for (side, trace), g in fd.groupby(["side", "trace"]):
        g = g.sort_values("frame_idx").reset_index(drop=True)
        g = g.iloc[int(g["dist_m"].to_numpy().argmax()):]
        label = g["label_class"].iloc[0]
        dists = WALL_DISTS if label == "wall" else (1.6,)
        for d in dists:
            r = g.iloc[(g["dist_m"] - d).abs().argmin()]
            out.append({"side": side, "trace": trace, "label": label,
                        "target_m": d, "dist_m": round(float(r["dist_m"]), 2),
                        "frame_idx": int(r["frame_idx"])})
    return out


def main() -> None:
    from ultralytics import YOLO, YOLOE
    OUT.mkdir(exist_ok=True)
    frames = sample_frames()
    for f in frames:
        f["img"] = grab(f["side"], f["trace"], f["frame_idx"])

    models = {}
    models["ours"] = YOLO(str(PROJECT_ROOT / "outputs/models/20260719_train_yolo_v1/train/weights/best.pt"))
    models["yolo26s_coco"] = YOLO("yolo26s.pt")
    yoloe = YOLOE("yoloe-26s-seg.pt")
    yoloe.set_classes(PROMPTS)
    models["yoloe26s_text"] = yoloe
    models["yoloe26s_pf"] = YOLOE("yoloe-26s-seg-pf.pt")

    rows = []
    for name, m in models.items():
        for f in frames:
            res = m.predict(f["img"], imgsz=480 if name == "ours" else 640,
                            conf=0.15, device="mps", verbose=False)[0]
            h, w = f["img"].shape[:2]
            dets = []
            for b, c, p in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(),
                               res.boxes.conf.tolist()):
                area = (b[2] - b[0]) * (b[3] - b[1]) / (w * h)
                dets.append((res.names[int(c)], round(p, 2), round(area, 2)))
            dets.sort(key=lambda d: -d[1])
            wall = [d for d in dets if d[0] == "wall"]
            rows.append({"model": name, "label": f["label"], "trace": f["trace"],
                         "target_m": f["target_m"], "dist_m": f["dist_m"],
                         "wall_conf": wall[0][1] if wall else None,
                         "top3": "; ".join(f"{n} {p}" for n, p, _ in dets[:3])})
            if f["label"] == "wall" and name in ("yoloe26s_text", "yoloe26s_pf") \
                    and f["target_m"] in (2.5, 1.6, 0.5) and f["trace"] in (3127, 3130):
                cv2.imwrite(str(OUT / f"{name}_{f['trace']}_{f['target_m']}m.jpg"),
                            res.plot())
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "probe.csv", index=False)

    wall = df[df["label"] == "wall"]
    print("WALL traces: wall detected / frames, by distance")
    print(wall.assign(hit=wall["wall_conf"].notna())
          .pivot_table(index="model", columns="target_m", values="hit",
                       aggfunc=lambda s: f"{int(s.sum())}/{len(s)}"))
    other = df[df["label"] != "wall"]
    print("\nNON-wall traces at 1.6 m: false 'wall' calls / frames")
    print(other.assign(fp=other["wall_conf"].notna()).groupby("model")["fp"]
          .agg(lambda s: f"{int(s.sum())}/{len(s)}"))
    print("\nWhat each model sees on wall frames (trace 3127):")
    print(wall[wall["trace"] == 3127][["model", "target_m", "top3"]].to_string(index=False))


if __name__ == "__main__":
    main()
