"""Rate auto-labeling pipelines against the reviewers' gold CSVs.

Reads every CSV exported from the gold review shards, resolves double-reviewed
items, and scores each pipeline per category x distance band and overall
(recall, precision, mean IoU, labels where nothing is visible). For the online
detect + track pipeline it also fits the label-confidence model (per-category
logistic regression on tracker/detector features, out-of-fold by trace) and
reports calibration and the auto-accept share at target precisions. Writes
outputs/autolabel_rating/<name>/ (report.md + CSVs + confidence_model.json).

    uv run python scripts/rate_autolabel.py --gold-dir <folder with gold_*.csv>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_uss_research.label_kpi import (BANDS, BOX, KEY, audit_silver,  # noqa: E402
                                           auto_accept,
                                           calibration_table, design,
                                           false_events_per_hour, fit_logistic,
                                           kpi_table, object_detection_rate,
                                           out_of_fold_confidence, resolve_gold,
                                           score_labels, tpr_per_sequence)

FEATURES = ["det_score", "is_pred", "log_age", "streak_c", "since_match",
            "n_wanted", "rel_area", "at_bottom", "at_side"]


def online_features(df: pd.DataFrame, w: int = 640, h: int = 480) -> pd.DataFrame:
    df = df.copy()
    df["is_pred"] = (df["src"] == "pred").astype(float)
    df["log_age"] = np.log1p(df["track_age"].fillna(0))
    df["streak_c"] = df["streak"].fillna(0).clip(upper=30)
    df["rel_area"] = ((df["px1"] - df["px0"]) * (df["py1"] - df["py0"]) / (w * h)).fillna(0)
    df["at_bottom"] = (df["py1"] >= h - 2).astype(float)
    df["at_side"] = ((df["px0"] <= 2) | (df["px1"] >= w - 2)).astype(float)
    return df


def load_pipelines(args) -> dict[str, pd.DataFrame]:
    pipes = {}
    online = pd.read_csv(args.online)
    pipes["online_detect_track"] = online[KEY + BOX + [c for c in online.columns
                                                        if c not in KEY + BOX]]
    old = pd.read_csv(args.bundle / "frames.csv")
    pipes["post_processed_tracker"] = old[KEY + BOX + ["label_src"]]
    for spec in args.extra:
        name, path = spec.split("=", 1)
        pipes[name] = pd.read_csv(path)
    return pipes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gold-dir", type=Path, required=True)
    parser.add_argument("--name", default="rating_v1")
    parser.add_argument("--bundle", type=Path,
                        default=PROJECT_ROOT / "outputs/cuv_objectcheck/cuv_bundle_v1")
    parser.add_argument("--online", type=Path, default=PROJECT_ROOT /
                        "outputs/autolabel_online/online_v1/labels.csv")
    parser.add_argument("--extra", nargs="*", default=[],
                        help="more pipelines as name=labels.csv (side,trace,frame_idx,x0..y1)")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--bands", type=float, nargs="+", default=BANDS,
                        help="distance band edges in m (the deck's ROI rings)")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--audit-dir", type=Path, default=None,
                        help="human audit CSVs of silver ground truth (notebook 18)")
    args = parser.parse_args()

    out = PROJECT_ROOT / "outputs" / "autolabel_rating" / args.name
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(args.gold_dir.glob("*.csv"))
    if not files:
        sys.exit(f"no gold CSVs in {args.gold_dir} (reviewers' exported gold_*.csv files)")
    reviews = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    gold, agree = resolve_gold(reviews)
    dist = pd.read_csv(args.bundle / "frames.csv", usecols=KEY + ["dist_m"])
    gold = gold.merge(dist, on=KEY, how="left")
    gold["band"] = pd.cut(gold["dist_m"], args.bands)
    # video duration per trace (all bundle frames), for false events per hour
    durations_h = (pd.read_csv(args.bundle / "frames.csv", usecols=["side", "trace"])
                   .groupby(["side", "trace"]).size() / args.fps / 3600)
    gold.to_csv(out / "gold.csv", index=False)

    report = [f"# Auto-labeling rating ({args.name})", "",
              f"Gold: {len(reviews)} reviews -> {len(gold)} frames; "
              + ", ".join(f"{k} {v}" for k, v in gold["status"].value_counts().items()) + ".",
              f"A label is correct when the object is visible and IoU >= {args.iou}."]
    if args.audit_dir is not None:
        audit = pd.concat([pd.read_csv(p) for p in sorted(args.audit_dir.glob("*.csv"))],
                          ignore_index=True)
        acc = audit_silver(gold, audit, args.iou)
        acc.to_csv(out / "silver_audit.csv")
        report += ["", "## Ground-truth audit", "",
                   "Share of audited ground-truth labels a human confirmed (same status; "
                   f"boxes IoU >= {args.iou}). Every KPI below is bounded by this.", "",
                   acc.to_markdown()]
    if len(agree):
        both = agree[agree["status_a"].eq("box") & agree["status_b"].eq("box")]
        report += ["", "## Reviewer agreement", "",
                   f"{len(agree)} frames reviewed twice: {agree['agree'].mean():.0%} agree; "
                   f"mean IoU between reviewers' boxes {both['iou'].mean():.2f} "
                   f"({len(both)} frames where both drew a box). Disputed frames are left out."]
        agree.to_csv(out / "reviewer_agreement.csv", index=False)

    overall = []
    for name, labels in load_pipelines(args).items():
        s = score_labels(gold, labels, args.iou)
        s.to_csv(out / f"scored_{name}.csv", index=False)
        by_band = kpi_table(s, ["category", "band"])
        by_cat = kpi_table(s, ["category"])
        by_band.to_csv(out / f"kpi_{name}_category_band.csv")
        by_cat.to_csv(out / f"kpi_{name}_category.csv")
        overall.append(kpi_table(s.assign(all="all"), ["all"]).assign(pipeline=name)
                       .set_index("pipeline"))
        band24 = kpi_table(s[s["dist_m"].between(2, 4)].assign(all="2-4 m"), ["all"])
        tpr_band = tpr_per_sequence(s, ["band"])
        tpr_cat = tpr_per_sequence(s, ["category", "band"])
        det_band = object_detection_rate(s, ["band"])
        fph = false_events_per_hour(s, durations_h, ["category"])
        fph_all = false_events_per_hour(s.assign(all="all"), durations_h, ["all"])
        tpr_cat.to_csv(out / f"tpr_seq_{name}_category_band.csv")
        fph.to_csv(out / f"false_per_hour_{name}.csv")
        report += ["", f"## {name}", "", "Per category:", "", by_cat.to_markdown(), "",
                   "2-4 m band, all categories:", "", band24.to_markdown(), "",
                   "Deck-style KPIs (KPI Calculations, CC-DA/ESV4 2018):", "",
                   "TPR per sequence by band - frames with the object completely in "
                   "view, TPR per trace, then the mean over traces:", "",
                   tpr_band.to_markdown(), "",
                   "Object detected at least once in the band (share of traces):", "",
                   det_band.to_markdown(), "",
                   "False labels per hour of video (runs of consecutive wrong gold "
                   "frames; distinct wrong track ids):", "",
                   fph_all.to_markdown(), "", fph.to_markdown()]

        if name == "online_detect_track":
            lab = online_features(s[s["labeled"]])
            lab["confidence"] = out_of_fold_confidence(lab, FEATURES)
            y = lab["correct"].to_numpy(float)
            calib, ece = calibration_table(lab["confidence"].to_numpy(), y)
            acc = {f"{t:.0%}": auto_accept(lab["confidence"].to_numpy(), y, t)
                   for t in (0.95, 0.98, 0.99)}
            per_cat = {c: auto_accept(g["confidence"].to_numpy(), g["correct"].to_numpy(float), 0.98)
                       for c, g in lab.groupby("category")}
            lab.to_csv(out / "online_labels_with_confidence.csv", index=False)
            cats = sorted(lab["category"].unique())
            X = design(lab, FEATURES, cats)
            mu, sd = X.mean(0), X.std(0) + 1e-9
            w = fit_logistic((X - mu) / sd, y) if len(np.unique(y)) == 2 else None
            (out / "confidence_model.json").write_text(json.dumps({
                "features": FEATURES, "categories": cats, "mean": mu.tolist(),
                "std": sd.tolist(), "weights": None if w is None else w.tolist(),
                "note": "design(): features + per-category intercepts + per-category det_score slopes"},
                indent=1))
            report += ["", "### Label confidence (out-of-fold by trace)", "",
                       f"Expected calibration error: {ece:.3f}.", "", calib.to_markdown(index=False), "",
                       "Auto-accept (labels kept without review at a target precision):", "",
                       pd.DataFrame(acc).T.to_markdown(), "",
                       "Per category at 98% precision:", "",
                       pd.DataFrame(per_cat).T.to_markdown()]
            sw = (labels.dropna(subset=["track_id"])
                  .merge(gold[["side", "trace"]].drop_duplicates(), on=["side", "trace"])
                  .groupby(["side", "trace"])["track_id"]
                  .agg(lambda t: int(t.diff().fillna(0).ne(0).sum())))
            report += ["", "### Tracking on the gold traces", "",
                       f"Target track changes per trace: median {sw.median():.0f}, max {sw.max()} "
                       f"({(sw > 0).mean():.0%} of traces have at least one)."]

    comp = pd.concat(overall)
    comp.to_csv(out / "kpi_overall.csv")
    report[4:4] = ["", "## Overall", "", comp.to_markdown()]
    (out / "report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
