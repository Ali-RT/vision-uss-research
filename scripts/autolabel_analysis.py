"""Offline analysis of an `al_eval_*` run (notebook 16 output) - no MF4 access.

Reads autolabel_eval.csv, re-derives the cross-fitted size priors exactly as
notebook 16 section 5 does, adds orientation diagnostics, and writes tables +
paper figures to <run_dir>/analysis/.

    PYTHONPATH=src python scripts/autolabel_analysis.py outputs/autolabel/al_eval_v2
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from vision_uss_research.autolabel.geometry import box_metrics, centroid, oriented_box
from vision_uss_research.autolabel.priors import add_offsets, cross_fit, fold_of, recording_day
from vision_uss_research.autolabel.uss_box import convex_hull, polygon_from_str

REPO = Path(__file__).resolve().parents[1]
K_FOLDS, MIN_CLASS_TRAIN, N_BOOT, SEED, MIN_OBJ_N = 5, 5, 2000, 0, 30
rng = np.random.default_rng(SEED)


def canonical_map() -> dict:
    m = yaml.safe_load((REPO / "configs/datasets/label_v2_class_mapping.yaml").read_text())
    alias = {}
    for canon, variants in (m.get("canonical_objects") or {}).items():
        for v in [canon, *variants]:
            alias[re.sub(r"[^a-z0-9]+", "", str(v).lower())] = canon
    return alias


def boot_ci(x, fn=np.median, n=N_BOOT):
    """Percentile CI under a DAY-CLUSTER bootstrap when `x` is a Series indexed
    like `ev` (days resampled whole; sequences of one session are
    near-duplicates), else a plain bootstrap."""
    s = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    if len(s) < 5:
        return (np.nan, np.nan)
    if DAYS is not None and s.index.isin(DAYS.index).all():
        groups = [g.to_numpy(float) for _, g in s.groupby(DAYS.loc[s.index])]
        b = [fn(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]))
             for _ in range(n)]
    else:
        v = s.to_numpy(float)
        b = [fn(v[rng.integers(0, len(v), len(v))]) for _ in range(n)]
    return tuple(float(v) for v in np.round(np.percentile(b, [2.5, 97.5]), 3))


DAYS: pd.Series | None = None   # set in main(): recording day per ev row


def min_area_rect(poly):
    """(centre, angle_rad of the LONG side, length, width) of the min-area
    bounding rectangle - rotating calipers over convex-hull edges."""
    hull = np.asarray(convex_hull(poly), float)
    if len(hull) < 3:
        return None
    best = None
    for i in range(len(hull)):
        e = hull[(i + 1) % len(hull)] - hull[i]
        if not np.any(e):
            continue
        a = np.arctan2(e[1], e[0])
        u, v = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
        pu, pv = hull @ u, hull @ v
        area = (pu.max() - pu.min()) * (pv.max() - pv.min())
        if best is None or area < best[0]:
            best = (area, a, pu, pv, u, v)
    _, a, pu, pv, u, v = best
    length, width = pu.max() - pu.min(), pv.max() - pv.min()
    c = u * (pu.max() + pu.min()) / 2 + v * (pv.max() + pv.min()) / 2
    if width > length:
        a, length, width = a + np.pi / 2, width, length
    return (float(c[0]), float(c[1])), float(a), float(length), float(width)


def angle_diff_deg(a, b):
    """Smallest angle between two undirected axes, in [0, 90]."""
    d = abs((np.degrees(a - b) + 90) % 180 - 90)
    return float(d)


def main(run_dir: Path):
    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)
    d = pd.read_csv(run_dir / "autolabel_eval.csv", low_memory=False)
    d["error"] = d["error"].fillna("")
    alias = canonical_map()
    d["obj"] = d["object"].map(
        lambda o: alias.get(re.sub(r"[^a-z0-9]+", "", str(o).lower()), str(o).strip().lower()))
    d["approach"] = d["approach"].fillna("untagged")
    ev = d[(d["error"] == "") & d["v1_iou"].notna() & (d["n_targets"] == 1)
           & d["face_cx"].notna()].copy().reset_index(drop=True)

    # ---- v2 priors (library code, 5-fold cross-fitted BY RECORDING DAY) - paper variants
    global DAYS
    ev = add_offsets(ev)
    ev["day"] = ev["sequence_id"].map(recording_day)
    DAYS = ev["day"]
    ev["fold"] = ev["day"].map(lambda d: fold_of(d, K_FOLDS))
    ev["_all"] = "all"
    for prefix, keys, off in (("glob", ["_all"], False), ("cls", ["obj"], False),
                              ("cls_off", ["obj"], True),
                              ("cls_appr", ["obj", "approach"], True)):
        ev = ev.join(cross_fit(ev, keys, use_offset=off, k=K_FOLDS,
                               min_n=MIN_CLASS_TRAIN, prefix=prefix, fold_by="day"))
    ev["cls_prior_used"] = ev["cls_group_prior"]

    # ---- diagnostics: human box orientation, oracle orientation / centre
    dims = ["gt_len_along_m", "gt_wid_across_m"]
    rows = []
    for k in range(K_FOLDS):
        train, test = ev[ev["fold"] != k], ev[ev["fold"] == k]
        cls_med, cls_n, glob = (train.groupby("obj")[dims].median(),
                                train.groupby("obj").size(), train[dims].median())
        for i, r in test.iterrows():
            use = r["obj"] in cls_med.index and cls_n[r["obj"]] >= MIN_CLASS_TRAIN
            L, W = (cls_med.loc[r["obj"]] if use else glob).to_numpy(float)
            gt = polygon_from_str(r["gt_polygon"])
            res = {"idx": i}
            rect = min_area_rect(gt)
            if rect is not None:
                _, ang, rl, rw = rect
                res["gt_axis_vs_ego_deg"] = angle_diff_deg(ang, r["ego_yaw_rad"])
                res["gt_elongation"] = rl / max(rw, 1e-6)
                res["_rect_len"], res["_rect_wid"] = rl, rw
                res["_gt_axis"] = ang
            # oracle centre: class size, ego heading, at the human box centroid
            gc = centroid(gt)
            c = oriented_box(gc[0], gc[1], r["ego_yaw_rad"], L, W)
            res["cls_oraclecentre_iou"] = box_metrics(c, gt)["iou"]
            rows.append(res)
    pri = pd.DataFrame(rows).set_index("idx")
    ev = ev.join(pri)
    rect_rows = []
    for k in range(K_FOLDS):
        tr, te = ev[ev["fold"] != k], ev[ev["fold"] == k]
        med = tr.groupby("obj")[["_rect_len", "_rect_wid"]].median()
        n = tr.groupby("obj").size()
        g = tr[["_rect_len", "_rect_wid"]].median()
        for i, r in te.iterrows():
            if pd.isna(r.get("_gt_axis")):
                continue
            use = r["obj"] in med.index and n[r["obj"]] >= MIN_CLASS_TRAIN
            L, W = (med.loc[r["obj"]] if use else g).to_numpy(float)
            gt = polygon_from_str(r["gt_polygon"])
            c = oriented_box(r["face_cx"], r["face_cy"], r["_gt_axis"], L, W)
            rect_rows.append({"idx": i, "cls_gtaxis_iou": box_metrics(c, gt)["iou"]})
    ev = ev.join(pd.DataFrame(rect_rows).set_index("idx"))

    # ---- 1. ladder with CIs
    ladder_vars = ["v1", "lat_pos", "depth", "glob", "cls", "cls_off", "cls_appr"]
    diag_vars = ["cls_gtaxis", "cls_oraclecentre", "lat_pos_oracle_size"]
    lad = []
    for v in ladder_vars + diag_vars:
        col = f"{v}_iou"
        for hb in ("all", "low", "high"):
            g = ev if hb == "all" else ev[ev["height_bin"] == hb]
            x = g[col].dropna()
            lad.append({"variant": v, "subset": hb, "n": len(x),
                        "iou_med": round(x.median(), 3), "iou_med_ci": boot_ci(x),
                        "iou_mean": round(x.mean(), 3),
                        "iou>=0.5": round((x >= 0.5).mean(), 3),
                        "centroid_med_m": (round(g[f"{v}_centroid_err_m"].median(), 3)
                                           if f"{v}_centroid_err_m" in g else np.nan)})
    lad = pd.DataFrame(lad)
    lad.to_csv(out / "ladder.csv", index=False)

    # ---- 2. per object
    objs = []
    for (o, hb), g in ev.groupby(["obj", "height_bin"]):
        if len(g) < MIN_OBJ_N:
            continue
        objs.append({"obj": o, "height_bin": hb, "n": len(g),
                     "gt_area_med_m2": round(g["gt_area_m2"].median(), 2),
                     "elongation_med": round(g["gt_elongation"].median(), 2),
                     "axis_vs_ego_med_deg": round(g["gt_axis_vs_ego_deg"].median(), 1),
                     **{f"{v}_iou": round(g[f"{v}_iou"].median(), 3)
                        for v in ["v1", "lat_pos", "cls", "cls_off", "cls_appr", "cls_gtaxis",
                                  "cls_oraclecentre"]},
                     "cls_appr_iou_ci": boot_ci(g["cls_appr_iou"]),
                     "lat_centroid_m": round(g["lat_pos_centroid_err_m"].median(), 3)})
    objs = pd.DataFrame(objs).sort_values(["height_bin", "n"], ascending=[True, False])
    objs.to_csv(out / "by_object.csv", index=False)

    # ---- 3. approach
    appr = (ev.groupby(["approach", "height_bin"])
              .agg(n=("v1_iou", "size"), v1=("v1_iou", "median"),
                   lat_pos=("lat_pos_iou", "median"), cls=("cls_iou", "median"),
                   cls_appr=("cls_appr_iou", "median"),
                   cls_gtaxis=("cls_gtaxis_iou", "median"),
                   v1_centroid=("v1_centroid_err_m", "median"),
                   lat_centroid=("lat_pos_centroid_err_m", "median"),
                   axis_vs_ego=("gt_axis_vs_ego_deg", "median"))
              .round(3).reset_index())
    appr.to_csv(out / "by_approach.csv", index=False)

    # ---- 4. height vs size: within GT-area strata, low vs high
    ev["area_q"] = pd.qcut(ev["gt_area_m2"], 5, duplicates="drop")
    strat = (ev.groupby(["area_q", "height_bin"], observed=True)
               .agg(n=("v1_iou", "size"), v1=("v1_iou", "median"), cls=("cls_iou", "median"),
                    cls_appr=("cls_appr_iou", "median"),
                    lat_oracle=("lat_pos_oracle_size_iou", "median"),
                    lat_centroid=("lat_pos_centroid_err_m", "median"))
               .round(3).reset_index())
    strat["area_q"] = strat["area_q"].astype(str)
    strat.to_csv(out / "height_vs_size.csv", index=False)

    # ---- 5. review policy: auto-accept a (class, approach) group when its
    # cross-fitted training share of v2 boxes with IoU >= 0.5 reaches t
    ev["_good"] = ev["cls_appr_iou"] >= 0.5
    ev["_grp"] = ev["obj"] + "|" + ev["approach"]
    pol = []
    for t in (0.5, 0.6, 0.7):
        acc = pd.Series(False, index=ev.index)
        for k in range(K_FOLDS):
            tr, te = ev[ev["fold"] != k], ev[ev["fold"] == k]
            rate, n = tr.groupby("_grp")["_good"].mean(), tr.groupby("_grp").size()
            ok = set(rate[(rate >= t) & (n >= 20)].index)
            acc[te.index] = te["_grp"].isin(ok)
        a = ev[acc]
        pol.append({"threshold": t, "accepted_share": round(acc.mean(), 4),
                    "accepted_iou>=0.5": round(a["_good"].mean(), 4),
                    "accepted_iou_med": round(a["cls_appr_iou"].median(), 3),
                    "rest_iou>=0.5": round(ev.loc[~acc, "_good"].mean(), 4)})
    pol = pd.DataFrame(pol)
    pol.to_csv(out / "review_policy.csv", index=False)
    grp = (ev.groupby(["obj", "approach"]).agg(n=("_good", "size"), good=("_good", "mean"),
                                              iou_med=("cls_appr_iou", "median"))
             .round(3).reset_index().sort_values("good", ascending=False))
    grp.to_csv(out / "by_object_approach.csv", index=False)

    ev.drop(columns=[c for c in ev.columns if c.startswith("_") and c != "_gt_axis"]).to_csv(
        out / "eval_enriched.csv", index=False)
    summary = {"n_single_target": int(len(ev)),
               "by_height": ev["height_bin"].value_counts().to_dict(),
               "cls_prior_used": round(float(ev["cls_prior_used"].mean()), 4)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(json.dumps(summary))
        print("\nLADDER\n", lad.to_string(index=False))
        print("\nBY OBJECT\n", objs.to_string(index=False))
        print("\nBY APPROACH\n", appr.to_string(index=False))
        print("\nHEIGHT vs SIZE\n", strat.to_string(index=False))
        print("\nREVIEW POLICY\n", pol.to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
