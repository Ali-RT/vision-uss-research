"""Evaluate the trained detector on CUV_ObjectCheck against the SOP distance spec.

Consumes outputs/prelabel/<run>/per_trace/*.csv from label_cuv_fullvideo.py
and the trace MF4s. Per trace:

  1. sync: AVI and MF4 start on a common trigger (all 73 durations agree
     within 0.1 s), so frames are start-aligned (--sync start, default).
     --sync corr instead cross-correlates video motion with MF4 speed; it
     is noisy (textureless close-ups, e.g. walls) and kept only as a
     sensitivity check.
  2. distance: bumper-to-object distance per video frame from
     VHM_VehicleState_DrivenDist (straight-line approach) plus the final USS
     gap (nearest Map_ObjN_Dist over the last 0.5 s; median fallback when the
     USS sees nothing, flagged). Traces that never move are excluded.
  3. metrics: a frame is a camera HIT when it has a picked box whose
     detector height bin matches the tracelist bin. Per trace: spec-window
     hit (>= 50% HIT frames within [spec - 0.25 m, spec]), first stable
     detection distance (10 consecutive HIT frames), Low/High accuracy of
     boxed frames.

Outputs eval/ under the run dir: frames_dist.csv, sync.csv, per_trace.csv,
by_distance.csv, summary.csv, report.md.

    uv run python scripts/eval_cuv_objectcheck.py --run fullvideo_cuv_v1
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from prelabel_cuv_objectcheck import DEFAULT_DATA_ROOT, PROJECT_ROOT

# Bosch proposal for SOP Midsize: (distance m, required detection rate).
SPEC = {"dummychild": (2.20, 0.90), "dummyadult": (2.20, 0.90),
        "isopole": (2.00, 0.95), "wall": (1.60, 0.90), "car": (1.50, 0.95),
        "box30": (1.20, 0.90), "box40": (1.40, 0.90),
        "curbhigh": (1.50, 0.90), "woodencurb10": (1.00, 0.90),
        "metalbar": (1.00, 0.90)}
SPEC_WINDOW = 0.25
STABLE_RUN = 10
DIST_BINS = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
MIN_SYNC_CORR = 0.5
MAX_LAG = 45


def find_files(data_root: Path, side: str, trace: int) -> tuple[Path, Path]:
    folder = "Front" if side == "front" else "Rear"
    avi = next((data_root / folder).glob(f"*_{trace}_*.avi"))
    return avi, avi.with_suffix(".mf4")


def video_motion(avi: Path) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(avi))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    prev, diffs = None, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(frame, (160, 120)),
                         cv2.COLOR_BGR2GRAY).astype(np.float32)
        diffs.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
        prev = g
    cap.release()
    return np.convolve(diffs, np.ones(5) / 5, "same"), fps


def load_mf4(mf4: Path) -> dict:
    from asammdf import MDF
    chans = ["VHM_VehicleState_DrivenDist"] + [f"Map_Obj{i}_Dist"
                                               for i in range(1, 21)]
    m = MDF(mf4, channels=chans)
    s = m.get("VHM_VehicleState_DrivenDist")
    t, d = s.timestamps, np.asarray(s.samples, float)
    gaps = []
    for i in range(1, 21):
        c = m.get(f"Map_Obj{i}_Dist")
        v, tc = np.asarray(c.samples, float), c.timestamps
        sel = (tc > t[-1] - 0.5) & (v > 0) & (v < 1000)
        if sel.any():
            gaps.append(float(np.median(v[sel])) / 1000)
    return {"t": t, "d": d, "gap": min(gaps) if gaps else None}


def sync_lag(motion: np.ndarray, fps: float, t: np.ndarray,
             d: np.ndarray) -> tuple[int, float]:
    speed = np.abs(np.gradient(d, t))
    tv = t[0] + np.arange(len(motion)) / fps
    sv = np.interp(tv, t, speed)
    core = slice(MAX_LAG, len(motion) - MAX_LAG)
    if np.std(sv[core]) < 1e-6:
        return 0, 0.0
    corrs = {k: np.corrcoef(np.roll(motion, k)[core], sv[core])[0, 1]
             for k in range(-MAX_LAG, MAX_LAG + 1)}
    best = max(corrs, key=lambda k: corrs[k])
    return best, float(corrs[best])


def first_stable_distance(g: pd.DataFrame) -> float | None:
    """Largest distance from which STABLE_RUN consecutive frames are HITs."""
    hits = g["hit"].to_numpy()
    dist = g["dist_m"].to_numpy()
    run = 0
    for i, h in enumerate(hits):
        run = run + 1 if h else 0
        if run == STABLE_RUN:
            return float(dist[i - STABLE_RUN + 1])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="fullvideo_cuv_v1")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--sync", choices=["start", "corr"], default="start")
    parser.add_argument("--eval-name", default="eval")
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / "outputs" / "prelabel" / args.run
    eval_dir = run_dir / args.eval_name
    eval_dir.mkdir(exist_ok=True)
    sync_csv = eval_dir / "sync.csv"
    sync_cache = (pd.read_csv(sync_csv).set_index("key").to_dict("index")
                  if sync_csv.exists() else {})

    files = [p for p in sorted(glob.glob(str(run_dir / "per_trace" / "*.csv")))
             if not p.endswith(".dets.csv")]
    all_frames, sync_rows = [], []
    for p in files:
        g = pd.read_csv(p)
        side, trace = g["side"].iloc[0], int(g["trace"].iloc[0])
        key = f"{side}_{trace}"
        avi, mf4 = find_files(args.data_root, side, trace)
        mf = load_mf4(mf4)
        if key in sync_cache:
            lag, corr = int(sync_cache[key]["lag"]), sync_cache[key]["corr"]
            fps = sync_cache[key]["fps"]
        elif args.sync == "corr":
            motion, fps = video_motion(avi)
            lag, corr = sync_lag(motion, fps, mf["t"], mf["d"])
        else:
            cap = cv2.VideoCapture(str(avi))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
            lag, corr = 0, 1.0
        synced = corr >= MIN_SYNC_CORR and abs(lag) < MAX_LAG
        used_lag = lag if synced else 0

        moved = np.ptp(mf["d"]) >= 1.0
        # approach direction from the last point >= 0.5 m before the stop
        # (the vehicle may stand still for several seconds at the end)
        away = np.nonzero(np.abs(mf["d"] - mf["d"][-1]) >= 0.5)[0]
        direction = (float(np.sign(mf["d"][-1] - mf["d"][away[-1]]))
                     if len(away) else 1.0)
        gap_ok = mf["gap"] is not None
        sync_rows.append({"key": key, "lag": lag, "corr": round(corr, 3),
                          "fps": fps, "synced": synced, "moved": moved,
                          "gap_m": mf["gap"], "gap_from_uss": gap_ok,
                          "direction": direction})

        t_frame = mf["t"][0] + (g["frame_idx"].to_numpy() + used_lag) / fps
        g["drive_pos"] = np.interp(t_frame, mf["t"], mf["d"])
        g["_d_end"], g["_dir"] = mf["d"][-1], direction
        g["gap_m"] = mf["gap"]
        g["moved"], g["synced"] = moved, synced
        all_frames.append(g)

    sync = pd.DataFrame(sync_rows)
    sync.to_csv(sync_csv, index=False)
    median_gap = float(sync.loc[sync.gap_from_uss, "gap_m"].median())

    df = pd.concat(all_frames, ignore_index=True)
    df["gap_m"] = df["gap_m"].fillna(median_gap)
    obj_pos = df["_d_end"] + df["_dir"] * df["gap_m"]
    df["dist_m"] = (obj_pos - df["drive_pos"]).abs().round(3)
    df = df.drop(columns=["_d_end", "_dir"])
    df["boxed"] = df["x0"].notna()
    df["hit"] = df["boxed"] & (df["det_bin"] == df["height_bin"])
    df.to_csv(eval_dir / "frames_dist.csv", index=False)

    ev = df[df["moved"]]

    # detection rate by distance bin
    ev = ev.assign(dist_bin=pd.cut(ev["dist_m"], DIST_BINS))
    by_dist = (ev.groupby(["label_class", "dist_bin"], observed=True)["hit"]
               .mean().unstack().round(2))
    by_dist.to_csv(eval_dir / "by_distance.csv")

    per_trace = []
    for (side, trace), g in ev.groupby(["side", "trace"]):
        label = g["label_class"].iloc[0]
        spec_d, _ = SPEC[label]
        g = g.sort_values("frame_idx").reset_index(drop=True)
        # final approach only: from the farthest point to the stop
        g = g.iloc[int(g["dist_m"].to_numpy().argmax()):]
        win = g[(g["dist_m"] <= spec_d) & (g["dist_m"] >= spec_d - SPEC_WINDOW)]
        boxed = g[g["boxed"]]
        near = boxed[boxed["dist_m"] <= 3.0]
        trace_call = (near["det_bin"].mode().iloc[0] if len(near) else None)
        per_trace.append({
            "side": side, "trace": trace, "label_class": label,
            "speed": g["speed"].iloc[0], "spec_m": spec_d,
            "start_dist_m": round(float(g["dist_m"].max()), 2),
            "spec_window_frames": len(win),
            "spec_window_hit_rate": round(float(win["hit"].mean()), 2) if len(win) else None,
            "spec_hit": bool(len(win) and win["hit"].mean() >= 0.5),
            "first_stable_m": first_stable_distance(g),
            "last_hit_m": (round(float(g.loc[g["hit"], "dist_m"].min()), 2)
                           if g["hit"].any() else None),
            "boxed_frames": len(boxed),
            "bin_acc_boxed": round(float((boxed["det_bin"] == boxed["height_bin"]).mean()), 3) if len(boxed) else None,
            "trace_call_3m": trace_call,
            "trace_call_ok": trace_call == g["height_bin"].iloc[0] if trace_call else None,
            "synced": bool(g["synced"].iloc[0]),
        })
    pt = pd.DataFrame(per_trace)
    pt.to_csv(eval_dir / "per_trace.csv", index=False)

    rows = []
    for label, g in pt.groupby("label_class"):
        spec_d, spec_rate = SPEC[label]
        f = ev[ev["label_class"] == label]
        boxed = f[f["boxed"]]
        rows.append({
            "label_class": label, "bin": f["height_bin"].iloc[0],
            "traces": len(g), "spec_m": spec_d, "spec_rate": spec_rate,
            "camera_rate_at_spec": round(float(g["spec_hit"].mean()), 2),
            "meets_spec": bool(g["spec_hit"].mean() >= spec_rate),
            "median_first_stable_m": (round(float(g["first_stable_m"].median()), 2)
                                      if g["first_stable_m"].notna().any() else None),
            "traces_never_stable": int(g["first_stable_m"].isna().sum()),
            "frame_bin_acc": round(float((boxed["det_bin"] == boxed["height_bin"]).mean()), 3) if len(boxed) else None,
            "trace_call_acc": round(float(g["trace_call_ok"].dropna().mean()), 2) if g["trace_call_ok"].notna().any() else None,
        })
    summary = pd.DataFrame(rows).sort_values("camera_rate_at_spec", ascending=False)
    summary.to_csv(eval_dir / "summary.csv", index=False)

    excluded = sync[~sync["moved"]]["key"].tolist()
    unsynced = sync[~sync["synced"] & sync["moved"]]["key"].tolist()
    no_gap = sync[~sync["gap_from_uss"]]["key"].tolist()
    report = [
        f"# CUV_ObjectCheck camera eval ({args.run})", "",
        f"Traces evaluated: {pt.shape[0]}. Excluded (no vehicle motion): {excluded or 'none'}.",
        f"Sync: {args.sync}" + (f"; fallback to lag 0 (corr < {MIN_SYNC_CORR}): {unsynced or 'none'}" if args.sync == "corr" else " (lag 0)") + ".",
        f"USS end gap missing, median {median_gap:.2f} m used: {no_gap or 'none'}.", "",
        "HIT = picked box whose detector Low/High bin matches the tracelist bin.",
        f"Spec hit = >= 50% HIT frames within [spec - {SPEC_WINDOW} m, spec].", "",
        "## Summary vs SOP spec", "", summary.to_markdown(index=False), "",
        "## HIT rate by distance bin (m)", "", by_dist.to_markdown(), "",
    ]
    (eval_dir / "report.md").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
