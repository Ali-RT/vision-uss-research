"""Offline analysis of an `al_eval_*` run (notebook 16 output) - no MF4 access.

Reads autolabel_eval.csv, re-derives the cross-fitted size priors exactly as
notebook 16 section 5 does, adds orientation diagnostics, and writes tables +
paper figures to <run_dir>/analysis/.

    PYTHONPATH=src python scripts/autolabel_analysis.py outputs/autolabel/al_eval_v2
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from vision_uss_research.autolabel.geometry import box_metrics, centroid, oriented_box
from vision_uss_research.autolabel.uss_box import (box_behind_face, convex_hull,
                                                   polygon_from_str)

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
    x = pd.to_numeric(pd.Series(np.asarray(x).ravel()), errors="coerce").dropna().to_numpy(float)
    if len(x) < 5:
        return (np.nan, np.nan)
    b = [fn(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return tuple(float(v) for v in np.round(np.percentile(b, [2.5, 97.5]), 3))


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

    # ---- cross-fitted size priors (same recipe as notebook 16 section 5) + diagnostics
    ev["fold"] = ev["sequence_id"].map(
        lambda s: int(hashlib.md5(s.encode()).hexdigest(), 16) % K_FOLDS)
    dims = ["gt_len_along_m", "gt_wid_across_m"]
    rows = []
    for k in range(K_FOLDS):
        train, test = ev[ev["fold"] != k], ev[ev["fold"] == k]
        cls_med, cls_n, glob = (train.groupby("obj")[dims].median(),
                                train.groupby("obj").size(), train[dims].median())
        for i, r in test.iterrows():
            use = r["obj"] in cls_med.index and cls_n[r["obj"]] >= MIN_CLASS_TRAIN
            L, W = (cls_med.loc[r["obj"]] if use else glob).to_numpy(float)
            gL, gW = glob.to_numpy(float)
            gt = polygon_from_str(r["gt_polygon"])
            face = (r["face_cx"], r["face_cy"])
            res = {"idx": i, "cls_prior_used": int(use)}
            for name, (bl, bw) in (("glob", (gL, gW)), ("cls", (L, W))):
                c, _ = box_behind_face(face, r["ego_yaw_rad"], int(r["obj_side"]), bl, bw,
                                       depth_shift=False)
                res.update({f"{name}_{m}": v for m, v in box_metrics(c, gt).items()})
            # orientation diagnostics: the human box's own axis vs ego heading
            rect = min_area_rect(gt)
            if rect is not None:
                _, ang, rl, rw = rect
                res["gt_axis_vs_ego_deg"] = angle_diff_deg(ang, r["ego_yaw_rad"])
                res["gt_elongation"] = rl / max(rw, 1e-6)
                # oracle orientation: class size along/across the GT axes, at the
                # predicted centre. Class size is measured along ego heading, so
                # re-measure it along the GT axis from training rect dims instead.
                res["_rect_len"], res["_rect_wid"] = rl, rw
                res["_gt_axis"] = ang
            # oracle centre: class size, ego heading, at the GT centroid
            gc = centroid(gt)
            c = oriented_box(gc[0], gc[1], r["ego_yaw_rad"], L, W)
            res["cls_oraclecentre_iou"] = box_metrics(c, gt)["iou"]
            rows.append(res)
        # object-aligned class box: per-class median of min-area-rect dims,
        # oriented along the GT axis (orientation oracle) at the predicted centre
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
    ladder_vars = ["v1", "lat_pos", "depth", "glob", "cls"]
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
                        for v in ["v1", "lat_pos", "cls", "cls_gtaxis", "cls_oraclecentre"]},
                     "cls_iou_ci": boot_ci(g["cls_iou"]),
                     "lat_centroid_m": round(g["lat_pos_centroid_err_m"].median(), 3)})
    objs = pd.DataFrame(objs).sort_values(["height_bin", "n"], ascending=[True, False])
    objs.to_csv(out / "by_object.csv", index=False)

    # ---- 3. approach
    appr = (ev.groupby(["approach", "height_bin"])
              .agg(n=("v1_iou", "size"), v1=("v1_iou", "median"),
                   lat_pos=("lat_pos_iou", "median"), cls=("cls_iou", "median"),
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
                    lat_oracle=("lat_pos_oracle_size_iou", "median"),
                    lat_centroid=("lat_pos_centroid_err_m", "median"))
               .round(3).reset_index())
    strat["area_q"] = strat["area_q"].astype(str)
    strat.to_csv(out / "height_vs_size.csv", index=False)

    ev.drop(columns=[c for c in ev.columns if c.startswith("_")]).to_csv(
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


if __name__ == "__main__":
    main(Path(sys.argv[1]))
