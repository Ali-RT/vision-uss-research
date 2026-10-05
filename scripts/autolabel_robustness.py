"""Robustness checks for the auto-labeler paper (offline, no MF4 access).

1. Leakage: priors cross-fitted by RECORDING DAY instead of by sequence, and
   day-cluster bootstrap CIs (sequences of one session are near-duplicates).
2. Height effect as a test: OLS of IoU on a low-object indicator, controlling
   for human-box size and approach geometry; day-cluster bootstrap CIs.
3. Threshold sensitivity: share of boxes with IoU >= t for t = 0.1 .. 0.9.
4. Selection: the ladder on ALL scored sequences (multi-target included,
   first low/high target), not only single-target ones.

    PYTHONPATH=src python scripts/autolabel_robustness.py outputs/autolabel/al_eval_v2
Writes <run_dir>/analysis/robustness_*.csv and robustness.json.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from vision_uss_research.autolabel.priors import add_offsets, cross_fit, recording_day

REPO = Path(__file__).resolve().parents[1]
N_BOOT, SEED, K = 1000, 0, 5
PRIORS = (("cls", ["obj"], False), ("cls_off", ["obj"], True),
          ("cls_appr", ["obj", "approach"], True))


def canonical(series: pd.Series) -> pd.Series:
    m = yaml.safe_load((REPO / "configs/datasets/label_v2_class_mapping.yaml").read_text())
    alias = {}
    for canon, variants in (m.get("canonical_objects") or {}).items():
        for v in [canon, *variants]:
            alias[re.sub(r"[^a-z0-9]+", "", str(v).lower())] = canon
    return series.map(lambda o: alias.get(re.sub(r"[^a-z0-9]+", "", str(o).lower()),
                                          str(o).strip().lower()))


def day_boot(df: pd.DataFrame, stat, n: int = N_BOOT, seed: int = SEED):
    """Percentile CI of stat(df) under resampling of whole recording days."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby("day")]
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        vals.append(stat(pd.concat([groups[i] for i in pick])))
    return tuple(float(v) for v in np.round(np.nanpercentile(vals, [2.5, 97.5]), 3))


def add_priors(df: pd.DataFrame, fold_by: str, tag: str) -> pd.DataFrame:
    df = df.copy()
    for prefix, keys, off in PRIORS:
        r = cross_fit(df, keys, use_offset=off, k=K, prefix=f"{prefix}{tag}", fold_by=fold_by)
        df = df.join(r)
    return df


def ols_low_effect(df: pd.DataFrame, y: str, controls: str) -> float:
    """Coefficient of the low-object indicator in an OLS of IoU."""
    X = [np.ones(len(df)), (df["height_bin"] == "low").astype(float).to_numpy()]
    if controls in ("size", "size+approach"):
        la = np.log(df["gt_area_m2"].to_numpy())
        X += [la, la ** 2]
    if controls in ("size+approach", "size+approach+shape"):
        for a in ("45deg", "displaced", "untagged"):
            X.append((df["approach"] == a).astype(float).to_numpy())
    if controls == "size+approach+shape":
        le = np.log(df["gt_elongation"].to_numpy(float))
        X += [le, np.sin(np.radians(df["gt_axis_vs_ego_deg"].to_numpy(float))) * le]
    X = np.column_stack(X)
    beta, *_ = np.linalg.lstsq(X, df[y].to_numpy(float), rcond=None)
    return float(beta[1])


def main(run_dir: Path):
    out = run_dir / "analysis"
    raw = pd.read_csv(run_dir / "autolabel_eval.csv", low_memory=False)
    raw["error"] = raw["error"].fillna("")
    raw["obj"] = canonical(raw["object"])
    raw["approach"] = raw["approach"].fillna("untagged")
    raw["day"] = raw["sequence_id"].map(recording_day)
    scored = raw[(raw["error"] == "") & raw["v1_iou"].notna() & raw["face_cx"].notna()].copy()
    scored = add_offsets(scored.reset_index(drop=True))
    single = scored[scored["n_targets"] == 1].copy()
    res: dict = {"n_single": int(len(single)), "n_scored": int(len(scored)),
                 "n_days": int(single["day"].nunique())}

    # ---- 1. sequence- vs day-grouped folds, day-cluster CIs
    single = add_priors(single, "sequence", "")
    single = add_priors(single, "day", "_day")
    rows = []
    for v in ["v1", "lat_pos", "cls", "cls_day", "cls_off", "cls_off_day",
              "cls_appr", "cls_appr_day"]:
        col = f"{v}_iou"
        for hb in ("all", "high", "low"):
            g = single if hb == "all" else single[single["height_bin"] == hb]
            rows.append({"variant": v, "subset": hb, "n": len(g),
                         "iou_med": round(g[col].median(), 3),
                         "iou_med_day_ci": day_boot(g, lambda d, c=col: d[c].median()),
                         "iou>=0.5": round((g[col] >= 0.5).mean(), 3)})
    folds = pd.DataFrame(rows)
    folds.to_csv(out / "robustness_folds.csv", index=False)

    # ---- 2. height effect, controlled (shape = elongation and elongation x
    # obliqueness to the ego heading, from the analysis script's enriched table)
    shape = pd.read_csv(out / "eval_enriched.csv", low_memory=False,
                        usecols=["sequence_id", "gt_elongation", "gt_axis_vs_ego_deg"])
    single = single.merge(shape, on="sequence_id", how="left")
    reg = []
    for y in ("v1_iou", "cls_appr_day_iou"):
        for ctrl in ("none", "size", "size+approach", "size+approach+shape"):
            b = ols_low_effect(single, y, ctrl)
            ci = day_boot(single, lambda d, y=y, c=ctrl: ols_low_effect(d, y, c), n=500)
            reg.append({"iou": y.replace("_iou", ""), "controls": ctrl,
                        "low_minus_high": round(b, 3), "day_ci": ci})
    reg = pd.DataFrame(reg)
    reg.to_csv(out / "robustness_height_regression.csv", index=False)

    # ---- 3. threshold sensitivity
    ts = np.round(np.arange(0.1, 0.91, 0.1), 1)
    thr = pd.DataFrame({"t": ts, **{v: [round((single[f"{v}_iou"] >= t).mean(), 4) for t in ts]
                                    for v in ("v1", "lat_pos", "cls_day", "cls_appr_day")}})
    thr.to_csv(out / "robustness_thresholds.csv", index=False)

    # ---- 4. all scored sequences (multi-target included)
    allsc = add_priors(scored, "day", "_day")
    sel = []
    for name, g in (("single-target", single), ("all scored", allsc)):
        for v in ("v1", "lat_pos", "cls_appr_day"):
            sel.append({"set": name, "n": len(g), "variant": v,
                        "iou_med": round(g[f"{v}_iou"].median(), 3),
                        "iou>=0.5": round((g[f"{v}_iou"] >= 0.5).mean(), 3)})
    sel = pd.DataFrame(sel)
    sel.to_csv(out / "robustness_selection.csv", index=False)

    single.drop(columns=[c for c in single.columns if c.startswith("_")]).to_csv(
        out / "eval_robust.csv", index=False)
    (out / "robustness.json").write_text(json.dumps(res, indent=1))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(json.dumps(res))
        print("\nFOLDS\n", folds.to_string(index=False))
        print("\nHEIGHT REGRESSION (coef of low vs high)\n", reg.to_string(index=False))
        print("\nTHRESHOLDS\n", thr.to_string(index=False))
        print("\nSELECTION\n", sel.to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
