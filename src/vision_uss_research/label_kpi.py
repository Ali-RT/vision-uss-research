"""Rate an auto-labeling pipeline against a human gold set.

Gold: reviewer CSVs from scripts/gold_review_template.html (status box /
not_visible / unsure, box in image pixels). Items reviewed twice give the
reviewer agreement and are resolved before scoring: both boxes with IoU >=
0.5 -> averaged box; same non-box status -> that status; anything else ->
disputed (left out of the KPIs, counted).

A pipeline's labels: one row per frame with x0..y1 (NaN = no label) plus any
feature columns. Per gold frame a label is CORRECT when the object is visible
and IoU >= iou_thr, WRONG when it misses the object or the object is not
visible, and a MISS when a visible object got no label.

The label-confidence model is a per-category logistic regression on label
features (numpy only), scored out-of-fold with folds grouped by trace, so
every reported probability comes from a model that never saw that trace.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BOX = ["x0", "y0", "x1", "y1"]
KEY = ["side", "trace", "frame_idx"]
BANDS = [0, 1, 2, 4, 8]


def box_iou(a, b) -> float:
    if np.any(np.isnan(a)) or np.any(np.isnan(b)):
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ------------------------------------------------------------ gold

def resolve_gold(reviews: pd.DataFrame, agree_iou: float = 0.5
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One gold row per item, plus the per-item agreement table for items
    reviewed twice. Status 'unsure' never becomes gold."""
    gold, agree = [], []
    for item, g in reviews.groupby("item_id"):
        first = g.iloc[0]
        base = {"item_id": item, "side": first["side"], "trace": int(first["trace"]),
                "frame_idx": int(first["frame_idx"]), "category": first["category"]}
        if len(g) == 1:
            r = first
            status = r["status"] if r["status"] != "unsure" else "unsure"
            gold.append({**base, "status": status, "truncated": int(r.get("truncated", 0) or 0),
                         **{c: r[c] if status == "box" else np.nan for c in BOX}})
            continue
        a, b = g.iloc[0], g.iloc[1]
        iou = (box_iou(a[BOX].to_numpy(float), b[BOX].to_numpy(float))
               if a["status"] == b["status"] == "box" else np.nan)
        same = a["status"] == b["status"] and (a["status"] != "box" or iou >= agree_iou)
        agree.append({**base, "status_a": a["status"], "status_b": b["status"],
                      "iou": iou, "agree": bool(same)})
        if not same or a["status"] == "unsure":
            gold.append({**base, "status": "disputed", "truncated": 0,
                         **{c: np.nan for c in BOX}})
        elif a["status"] == "box":
            gold.append({**base, "status": "box",
                         "truncated": int(max(a.get("truncated", 0) or 0, b.get("truncated", 0) or 0)),
                         **{c: (float(a[c]) + float(b[c])) / 2 for c in BOX}})
        else:
            gold.append({**base, "status": a["status"], "truncated": 0,
                         **{c: np.nan for c in BOX}})
    return pd.DataFrame(gold), pd.DataFrame(agree)


# ------------------------------------------------------------ scoring

def score_labels(gold: pd.DataFrame, labels: pd.DataFrame,
                 iou_thr: float = 0.5) -> pd.DataFrame:
    """Gold frames (box / not_visible only) joined with a pipeline's labels;
    adds labeled, iou, correct, miss and wrong per frame."""
    g = gold[gold["status"].isin(["box", "not_visible"])]
    lab = labels[KEY + BOX + [c for c in labels.columns if c not in KEY + BOX]]
    df = g.merge(lab.rename(columns={c: "p" + c for c in BOX}), on=KEY, how="left",
                 suffixes=("", "_lab"))
    df["visible"] = df["status"] == "box"
    df["labeled"] = df["px0"].notna()
    df["iou"] = [box_iou(np.array(r[BOX], float), np.array(r[["p" + c for c in BOX]], float))
                 if r["visible"] and r["labeled"] else 0.0 for _, r in df.iterrows()]
    df["correct"] = df["visible"] & df["labeled"] & (df["iou"] >= iou_thr)
    df["miss"] = df["visible"] & ~df["labeled"]
    df["wrong"] = df["labeled"] & ~df["correct"]
    return df


def kpi_table(scored: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Recall, precision, mean IoU of correct labels, labels on frames where
    the object is not visible, per group."""
    def agg(g):
        vis, lab = g["visible"], g["labeled"]
        return pd.Series({
            "frames": len(g), "visible": int(vis.sum()), "labeled": int(lab.sum()),
            "recall": g["correct"].sum() / vis.sum() if vis.sum() else np.nan,
            "precision": g["correct"].sum() / lab.sum() if lab.sum() else np.nan,
            "mean_iou": g.loc[g["correct"], "iou"].mean() if g["correct"].any() else np.nan,
            "false_on_not_visible": int((lab & ~vis).sum()),
        })
    return scored.groupby(by, observed=True).apply(agg, include_groups=False).round(3)


# ------------------------------------------------------------ confidence

def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0,
                 iters: int = 50) -> np.ndarray:
    """L2 logistic regression (bias unpenalized) by Newton steps."""
    Xb = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(Xb.shape[1])
    reg = np.full(Xb.shape[1], l2)
    reg[0] = 0.0
    for _ in range(iters):
        p = _sigmoid(Xb @ w)
        grad = Xb.T @ (p - y) + reg * w
        hess = Xb.T @ (Xb * (p * (1 - p))[:, None]) + np.diag(reg + 1e-9)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return w


def predict_logistic(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return _sigmoid(np.hstack([np.ones((len(X), 1)), X]) @ w)


def design(df: pd.DataFrame, features: list[str], categories: list[str]) -> np.ndarray:
    """Features plus per-category intercepts and per-category score slopes:
    categories differ in how raw detector scores map to correctness."""
    X = df[features].astype(float).fillna(0).to_numpy()
    cats = np.stack([(df["category"] == c).to_numpy(float) for c in categories[1:]], axis=1) \
        if len(categories) > 1 else np.zeros((len(df), 0))
    score = df["det_score"].astype(float).fillna(0).to_numpy()[:, None]
    return np.hstack([X, cats, cats * score])


def out_of_fold_confidence(df: pd.DataFrame, features: list[str], n_folds: int = 5,
                           l2: float = 1.0, seed: int = 0) -> np.ndarray:
    """P(label correct) for every labeled row, each from a model fitted
    without that row's trace (folds grouped by trace)."""
    cats = sorted(df["category"].unique())
    traces = df[["side", "trace"]].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(traces)
    fold_of = {(s, int(t)): i % n_folds for i, (s, t) in enumerate(traces)}
    fold = np.array([fold_of[(s, int(t))] for s, t in zip(df["side"], df["trace"])])
    X, y = design(df, features, cats), df["correct"].to_numpy(float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    X = (X - mu) / sd
    out = np.full(len(df), np.nan)
    for k in range(n_folds):
        tr, te = fold != k, fold == k
        if te.any() and len(np.unique(y[tr])) == 2:
            out[te] = predict_logistic(fit_logistic(X[tr], y[tr], l2), X[te])
        elif te.any():
            out[te] = y[tr].mean() if tr.any() else 0.5
    return out


def calibration_table(p: np.ndarray, y: np.ndarray, bins: int = 10
                      ) -> tuple[pd.DataFrame, float]:
    """Reliability bins and expected calibration error (ECE)."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "labels": int(m.sum()),
                         "mean_confidence": round(float(p[m].mean()), 3),
                         "share_correct": round(float(y[m].mean()), 3)})
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return pd.DataFrame(rows), round(float(ece), 4)


def auto_accept(p: np.ndarray, y: np.ndarray, target_precision: float = 0.98
                ) -> dict:
    """Lowest confidence threshold whose accepted labels reach the target
    precision, and the share of labels accepted at it."""
    order = np.argsort(-p)
    prec = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
    ok = np.nonzero(prec >= target_precision)[0]
    if not len(ok):
        return {"threshold": None, "accepted_share": 0.0, "precision": None}
    k = ok[-1]
    return {"threshold": round(float(p[order][k]), 3),
            "accepted_share": round((k + 1) / len(y), 3),
            "precision": round(float(prec[k]), 3)}


# ------------------------------------------------------------ deck-style KPIs
# Following "KPI Calculations" (CC-DA/ESV4, 2018): a label counts for TPR only
# when it lies completely inside the region (here: not cut by the image edge);
# TPR is computed per frame, averaged within each sequence, then over
# sequences; FPR counts each false object once and normalizes by time.

def edge_cut(df: pd.DataFrame, w: int = 640, h: int = 480, edge: float = 2.0) -> pd.Series:
    """Object cut off at the bottom or a side of the image (flagged and the
    gold box really touches that edge). Running past the top edge is normal
    for tall objects at bumper height and does not count."""
    flagged = df["truncated"].fillna(0).astype(float) > 0
    if not all(c in df.columns for c in BOX):
        return flagged
    touches = (df["x0"] <= edge) | (df["x1"] >= w - 1 - edge) | (df["y1"] >= h - 1 - edge)
    return flagged & touches.fillna(False)


def tpr_per_sequence(scored: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Per trace: correct / visible over frames whose object is completely in
    view (gold 'cut by image edge' left out); then the mean over traces per
    group, with the trace count and the worst trace."""
    full = scored[scored["visible"] & ~edge_cut(scored)]
    per = (full.groupby(by + ["side", "trace"], observed=True)["correct"]
           .mean().rename("tpr").reset_index())
    return (per.groupby(by, observed=True)["tpr"]
            .agg(tpr_mean="mean", worst_trace="min", traces="size").round(3))


def object_detection_rate(scored: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Share of traces whose object got at least one correct label in the
    group (deck: 'detected at some point in time during the maneuver')."""
    vis = scored[scored["visible"]]
    per = vis.groupby(by + ["side", "trace"], observed=True)["correct"].any()
    return (per.groupby(by, observed=True)
            .agg(detected="mean", traces="size").round(3))


def false_events_per_hour(scored: pd.DataFrame, durations_h: pd.Series,
                          by: list[str]) -> pd.DataFrame:
    """False labels as events per hour of video. An event is a run of
    consecutive gold frames (in gold sampling order) with a wrong label; with a
    `track_id` column, distinct track ids carrying a wrong label are counted
    too. `durations_h`: hours of video per (side, trace)."""
    rows = []
    for keys, g in scored.groupby(by, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        runs = ids = 0
        hours = 0.0
        for (side, trace), t in g.groupby(["side", "trace"]):
            w = t.sort_values("frame_idx")["wrong"].to_numpy()
            runs += int(np.sum(w & ~np.r_[False, w[:-1]]))
            if "track_id" in t.columns:
                ids += t.loc[t["wrong"], "track_id"].dropna().nunique()
            hours += float(durations_h.get((side, trace), 0.0))
        row = dict(zip(by, keys))
        row.update(hours=round(hours, 4), false_runs=runs,
                   false_runs_per_h=round(runs / hours, 1) if hours else np.nan)
        if "track_id" in g.columns:
            row.update(false_track_ids=ids,
                       false_ids_per_h=round(ids / hours, 1) if hours else np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index(by)


def audit_silver(silver: pd.DataFrame, audit: pd.DataFrame,
                 iou_thr: float = 0.5) -> pd.DataFrame:
    """How often human auditors confirm the silver labels, per category:
    same status, and for boxes IoU >= iou_thr with the auditor's box."""
    a = audit[audit["status"] != "unsure"]
    m = silver.merge(a[["item_id", "status"] + BOX], on="item_id", suffixes=("", "_a"))
    m["iou"] = [box_iou(np.array(r[BOX], float), np.array(r[[c + "_a" for c in BOX]], float))
                if r["status"] == r["status_a"] == "box" else np.nan for _, r in m.iterrows()]
    m["confirmed"] = (m["status"] == m["status_a"]) & (
        (m["status"] != "box") | (m["iou"] >= iou_thr))
    out = m.groupby("category").agg(audited=("item_id", "size"),
                                    confirmed=("confirmed", "mean"),
                                    mean_iou=("iou", "mean"))
    out.loc["all"] = [len(m), m["confirmed"].mean(), m["iou"].mean()]
    return out.round(3)
