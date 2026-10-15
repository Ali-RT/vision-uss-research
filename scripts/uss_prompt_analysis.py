"""Score notebook 20: do production USS map objects project onto the obstacle?

Fits one ground-plane pinhole camera per bumper (method/uss_prompt.py) with
day-grouped cross-validation, then reports, on held-out recording days:
  coverage   - labelled frames with a map object in the driving corridor
  hit        - corridor object (chosen WITHOUT the image) projects onto the
               obstacle's image box
  oracle_hit - the best of all live objects projects onto the box (upper
               bound: association solved)
overall, by range, by camera, by height bin, and on the sequences the
detector misses entirely (the coverage the method is meant to add).

Usage:
  python scripts/uss_prompt_analysis.py --run-dir <prompt_uss_proj_v1 dir> \
      --manifest paper/results/manifests/20260718_dataset_v2 \
      [--frame-preds paper/results/eval_causal/frame_predictions.csv] --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vision_uss_research.method.uss_prompt import (GroundCamera, box_hit,  # noqa: E402
                                                   fit_ground_camera, range_ahead_mm)

LOW = {"curbstone", "speedbump", "woodenboard", "low_other"}
RANGE_BINS = [-1000, 1000, 2000, 3000, 6000]
MAX_LATERAL_MM, MIN_RANGE_MM = 1200.0, -300.0


def load(run_dir: Path, manifest_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    slots = pd.read_csv(run_dir / "slots_at_frames.csv")
    man = pd.read_csv(manifest_dir / "dataset_manifest.csv")
    names = json.loads((manifest_dir / "provenance.json").read_text())["class_names"]
    man["class_name"] = man.class_id.map(lambda i: names[int(i)])
    man["area"] = (man.x1 - man.x0) * (man.y1 - man.y0)
    boxes = (man.sort_values("area", ascending=False)
             .drop_duplicates(["sequence_id", "camera", "frame_idx"]))
    boxes["bin"] = np.where(boxes.class_name.isin(LOW), "low", "high")
    slots["range_mm"] = [range_ahead_mm(x, c) for x, c in zip(slots.mid_x, slots.camera)]
    return slots, boxes


def corridor(slots: pd.DataFrame) -> pd.DataFrame:
    c = slots[(slots.range_mm >= MIN_RANGE_MM) & (slots.mid_y.abs() <= MAX_LATERAL_MM)]
    return c.sort_values("range_mm").drop_duplicates(["sequence_id", "camera", "frame_idx"])


def fold_of(seq_ids: pd.Series, k: int) -> pd.Series:
    days = seq_ids.str.split("_").str[1]
    order = {d: i % k for i, d in enumerate(sorted(days.unique()))}
    return days.map(order)


def fit_camera(train: pd.DataFrame, cam_name: str) -> GroundCamera:
    ub, vb = (train.x0 + train.x1) / 2, train.y1
    cam, _ = fit_ground_camera(train.mid_x, train.mid_y, ub, vb, cam_name)
    return cam


def run(slots, boxes, k=5, frame_preds=None) -> tuple[dict, pd.DataFrame]:
    keys = ["sequence_id", "camera", "frame_idx"]
    corr = corridor(slots).merge(boxes, on=keys)
    allslots = slots.merge(boxes, on=keys)
    frames = boxes[boxes.sequence_id.isin(slots.sequence_id.unique())].copy()
    frames["covered"] = frames.set_index(keys).index.isin(corr.set_index(keys).index)
    for df in (corr, allslots, frames):
        df["fold"] = fold_of(df.sequence_id, k)
    scored = []
    for cam_name in ("front", "rear"):
        for f in range(k):
            tr = corr[(corr.camera == cam_name) & (corr.fold != f)]
            if len(tr) < 30:
                continue
            cam = fit_camera(tr, cam_name)
            # pass 2: re-associate training frames to the slot nearest the box
            a = allslots[(allslots.camera == cam_name) & (allslots.fold != f)].copy()
            pu, pv, ok = cam.project(a.mid_x, a.mid_y)
            a["err"] = np.where(ok, np.hypot(pu - (a.x0 + a.x1) / 2, pv - a.y1), np.inf)
            best = a.sort_values("err").drop_duplicates(keys)
            best = best[best.err < 150]
            if len(best) >= 30:
                cam = fit_camera(best, cam_name)
            te = corr[(corr.camera == cam_name) & (corr.fold == f)].copy()
            pu, pv, ok = cam.project(te.mid_x, te.mid_y)
            te["hit"] = ok & box_hit(pu, pv, te.x0, te.y0, te.x1, te.y1)
            te["err_u_rel"] = np.abs(pu - (te.x0 + te.x1) / 2) / (te.x1 - te.x0)
            te["err_v_px"] = np.abs(pv - te.y1)
            ta = allslots[(allslots.camera == cam_name) & (allslots.fold == f)].copy()
            pu, pv, ok = cam.project(ta.mid_x, ta.mid_y)
            ta["h"] = ok & box_hit(pu, pv, ta.x0, ta.y0, ta.x1, ta.y1)
            oracle = ta.groupby(keys).h.any().rename("oracle_hit").reset_index()
            scored.append(te.merge(oracle, on=keys, how="left"))
    sc = pd.concat(scored, ignore_index=True) if scored else pd.DataFrame()
    sc["range_bin"] = pd.cut(sc.range_mm, RANGE_BINS)

    def block(d, fr):
        return {"frames": int(len(fr)), "coverage": round(float(fr.covered.mean()), 4),
                "hit_given_covered": round(float(d.hit.mean()), 4) if len(d) else None,
                "hit_overall": round(float(d.hit.sum() / max(len(fr), 1)), 4),
                "oracle_hit_given_covered": round(float(d.oracle_hit.mean()), 4) if len(d) else None,
                "median_err_u_rel": round(float(d.err_u_rel.median()), 3) if len(d) else None,
                "median_err_v_px": round(float(d.err_v_px.median()), 1) if len(d) else None}

    out = {"overall": block(sc, frames)}
    for col in ("camera", "bin"):
        out[f"by_{col}"] = {g: block(sc[sc[col] == g], frames[frames[col] == g])
                            for g in sorted(frames[col].unique())}
    out["by_range_mm"] = {str(b): {"n": int(len(d)), "hit": round(float(d.hit.mean()), 4)}
                          for b, d in sc.groupby("range_bin", observed=True)}
    if frame_preds is not None:
        # frame_preds holds every test-split detection; a test sequence absent
        # from it is one the detector never fired on
        detected = set(frame_preds.sequence_id.unique())
        missed = frames[(frames.split == "test") & ~frames.sequence_id.isin(detected)]
        m_sc = sc[sc.sequence_id.isin(missed.sequence_id)]
        out["detector_missed_sequences"] = {
            "sequences": int(missed.sequence_id.nunique()), **block(m_sc, missed),
            "sequences_with_any_hit": int(m_sc.groupby("sequence_id").hit.any().sum())}
    return out, sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--frame-preds", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    slots, boxes = load(a.run_dir, a.manifest)
    fp = pd.read_csv(a.frame_preds) if a.frame_preds else None
    summary, scored = run(slots, boxes, a.folds, fp)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    scored.to_csv(a.out / "scored_frames.csv", index=False)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
