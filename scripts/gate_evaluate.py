"""Step 4 (M3): evaluate the gated cascade and every comparator; decide GO.

Implements STEP4_PROPOSAL.md revision 2, §3.4-§3.6 (routing), §4 (arms) and
§6 (endpoints and statistics). Run ONCE, after M2 sign-off, on frozen inputs.

Per rotation r (tier D: woodenboard, speedbump, bush, bollard; tier C:
treecubehose):
  gate/gate_<r>/{calibration.json, head.pt, prototypes.npz}   (gate_train.py)
  gate/det_<r>_test/{crops.csv, features.npy}                 (gate_features.py det)
  reports/eval_<tag>/{raw_predictions.csv, inference_status.csv?}
  datasets/<dataset>/dataset_manifest.csv                     (test split = population)
VLM outputs: scored.csv of the USS/full run and of the fixed-crop run for the
tier's frames (tier D: the step-3 union; tier C: the tier C runs).

Arms per sequence, on one common population per (rotation, population):
  detector, detector_high          step-3 rule (unique mode over frames)
  vlm_uss, vlm_full, vlm_fixed     VLM alone (USS crop with per-frame fallback)
  plain_mode, plain_strict         plain cascade, unique-mode and strict-majority
  gate_trained, gate_free, gate_conf   gated cascades (§3.4-3.6), strict majority
Routing for every cascade: a named class decides; anything else -> vlm_uss.

Statistics (§6): day-cluster bootstrap, 2000 resamples, seed 0; days are
resampled jointly over every record of the endpoint's pooled population.
One-sided 95% bounds (5th / 95th percentile); unrounded; equality passes;
>5% undefined resamples -> not estimable -> fail; >2% of requested sequences
excluded by execution failures -> fail; inconclusive never passes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cascade_gtfree as cg  # noqa: E402
from vision_uss_research.method import gate as g  # noqa: E402

B, SEED = 2000, 0
UNDEFINED_MAX, EXCLUDED_MAX = 0.05, 0.02
BUCKETS = {"low_other", "high_other"}
KEYS = ["sequence_id", "camera", "frame_idx"]
GATES = ("gate_trained", "gate_free", "gate_conf")
ARMS = ("detector", "detector_high", "vlm_uss", "vlm_full", "vlm_fixed",
        "plain_mode", "plain_strict") + GATES


# ---- routing -------------------------------------------------------------------
def unique_mode(x: pd.Series) -> str:
    m = x.mode()
    return "none" if len(m) != 1 else m.iat[0]


def frame_labels(det: pd.DataFrame, feats: np.ndarray, gate_dir: Path, class_ids: dict) -> pd.DataFrame:
    """Per-frame labels for the three gates (and the raw detector class)."""
    import torch
    cal = json.loads((gate_dir / "calibration.json").read_text())
    pz = np.load(gate_dir / "prototypes.npz")
    protos = {k: {} for k in ("trained", "training_free")}
    for key in pz.files:
        kind, c = key.split("__", 1)
        protos[kind][c] = pz[key]
    head = g.make_head()
    head.load_state_dict(torch.load(gate_dir / "head.pt", map_location="cpu"))
    head.eval()
    valid = (det.crop_status == "valid").to_numpy()
    z_t = np.full((len(det), g.EMB_DIM), np.nan)
    z_f = np.full((len(det), g.FEAT_DIM), np.nan)
    if valid.any():
        rows = det.feature_row.to_numpy()[valid]
        z_t[valid] = g.embed(head, feats[rows])
        z_f[valid] = g.embed(None, feats[rows])
    out = []
    for i, r in enumerate(det.itertuples(index=False)):
        k = None if r.crop_status == "no_detection" else r.class_name
        zt = z_t[i] if valid[i] else None
        zf = z_f[i] if valid[i] else None
        out.append({
            "det_class": k if k is not None else "none",
            "gate_trained": g.gate_frame(k, zt, protos["trained"], cal["trained"]["tau"], BUCKETS, class_ids),
            "gate_free": g.gate_frame(k, zf, protos["training_free"], cal["training_free"]["tau"],
                                      BUCKETS, class_ids),
            "gate_conf": g.confidence_frame(k, r.conf, k in protos["trained"],
                                            cal["confidence"]["tau"], BUCKETS)})
    lab = pd.DataFrame(out, index=det.index)
    for gname in GATES:                         # PASS frames carry their class
        lab[gname] = np.where(lab[gname] == g.PASS, lab.det_class, lab[gname])
    unavailable = {gname: cal[k]["tau"] is None for gname, k in
                   (("gate_trained", "trained"), ("gate_free", "training_free"), ("gate_conf", "confidence"))}
    return pd.concat([det[KEYS + ["crop_status"]], lab], axis=1), unavailable


def sequence_decisions(fl: pd.DataFrame, v: pd.DataFrame, vfix: pd.DataFrame,
                       class_bins: dict) -> pd.DataFrame:
    """One row per sequence: label per gate, routed decision per arm."""
    ev = fl[fl.crop_status != "error"]               # execution failures do not vote
    agg = ev.groupby("sequence_id").agg(
        det_mode=("det_class", unique_mode),
        det_strict=("det_class", lambda x: g.strict_majority(list(x))),
        **{f"{k}_lab": (k, lambda x: g.strict_majority(list(x))) for k in GATES})
    d = v.loc[v.index.intersection(agg.index)].join(agg, how="inner")
    d["fixed"] = vfix.uss.reindex(d.index)
    out = pd.DataFrame(index=d.index)
    out["truth"], out["type"], out["frames"] = d.truth, d.target_object, d.vlm_frames

    def routed(label: pd.Series) -> tuple[pd.Series, pd.Series]:
        named = label.map(lambda c: c in class_bins and c not in BUCKETS)
        return label.map(class_bins).where(named, d.uss), ~named

    out["detector"] = d.det_mode.map(class_bins)
    named_mode = d.det_mode.map(lambda c: c in class_bins and c not in BUCKETS)
    out["detector_high"] = out.detector.where(named_mode, "high")
    out["vlm_uss"], out["vlm_full"], out["vlm_fixed"] = d.uss, d.full, d.fixed
    out["plain_mode"], out["deferred_plain_mode"] = routed(d.det_mode)
    out["plain_strict"], out["deferred_plain_strict"] = routed(d.det_strict)
    for k in GATES:
        out[k], out[f"deferred_{k}"] = routed(d[f"{k}_lab"])
    out["day"] = out.index.str.split("_").str[1]
    return out


# ---- statistics ------------------------------------------------------------------
def day_resamples(days: np.ndarray) -> np.ndarray:
    u = np.array(sorted(set(days)))
    return u, np.random.default_rng(SEED).integers(0, len(u), size=(B, len(u)))


def ratio_boot(x: pd.DataFrame, num: pd.Series, den: pd.Series) -> tuple[float, np.ndarray]:
    """Point estimate and bootstrap draws of sum(num)/sum(den), days resampled."""
    days, idx = day_resamples(x.day.to_numpy())
    pos = {d: i for i, d in enumerate(days)}
    di = x.day.map(pos).to_numpy()
    n = np.bincount(di, weights=num.to_numpy(float), minlength=len(days))
    m = np.bincount(di, weights=den.to_numpy(float), minlength=len(days))
    with np.errstate(invalid="ignore", divide="ignore"):
        point = n.sum() / m.sum() if m.sum() else np.nan
        draws = n[idx].sum(1) / m[idx].sum(1)
    return point, draws


def diff_boot(x, num_a, num_b, den):
    """rate(a) - rate(b) on the same denominator mask."""
    pa, da = ratio_boot(x, num_a, den)
    pb, db = ratio_boot(x, num_b, den)
    return pa - pb, da - db


def bound(point, draws, side: str) -> dict:
    undefined = float(np.isnan(draws).mean())
    if np.isnan(point) or undefined > UNDEFINED_MAX:
        return {"point": point, "bound": np.nan, "undefined_frac": undefined, "estimable": False}
    q = 5 if side == "lower" else 95
    return {"point": float(point), "bound": float(np.nanpercentile(draws, q)),
            "undefined_frac": undefined, "estimable": True}


def endpoint(name, x, a, b, metric, side, margin, strict=False, excluded_frac=0.0) -> dict:
    """metric in {'accuracy','low_recall','false_low'}; a - b vs margin."""
    low, high = x.truth == "low", x.truth == "high"
    if metric == "accuracy":
        num_a, num_b, den = (x[a] == x.truth), (x[b] == x.truth), pd.Series(1, index=x.index)
    elif metric == "low_recall":
        num_a, num_b, den = (x[a] == "low") & low, (x[b] == "low") & low, low
    else:  # false_low over High truth
        num_a, num_b, den = (x[a] == "low") & high, (x[b] == "low") & high, high
    point, draws = diff_boot(x, num_a.astype(float), num_b.astype(float), den.astype(float))
    r = bound(point, draws, side)
    if not r["estimable"] or excluded_frac > EXCLUDED_MAX:
        ok = False
    elif side == "upper":
        ok = r["bound"] <= margin
    else:
        ok = r["bound"] > margin if strict else r["bound"] >= margin
    return {"endpoint": name, "arm_a": a, "arm_b": b, "metric": metric, "side": side,
            "margin": margin, "strict": strict, "n": len(x), "days": x.day.nunique(),
            "excluded_frac": excluded_frac, **r, "pass": bool(ok)}


def p3(x: pd.DataFrame, excluded_frac: float) -> dict:
    """VLM frames per sequence, gated / VLM-alone (counterfactual), upper <= 0.60."""
    gated = x.frames * x.deferred_gate_trained.astype(float)
    point, draws = ratio_boot(x, gated, x.frames.astype(float))
    r = bound(point, draws, "upper")
    ok = r["estimable"] and excluded_frac <= EXCLUDED_MAX and r["bound"] <= 0.60
    return {"endpoint": "P3", "arm_a": "gate_trained", "arm_b": "vlm_uss", "metric": "vlm_frame_ratio",
            "side": "upper", "margin": 0.60, "n": len(x), "days": x.day.nunique(),
            "excluded_frac": excluded_frac, **r, "pass": bool(ok)}


def bypass_claim(held: pd.DataFrame, comparator: str) -> dict:
    """bypass rate(comparator) - bypass rate(trained) > 0 (one-sided lower)."""
    def byp(arm):
        return ((~held[f"deferred_{arm}"]) & (held[arm] != held.truth)).astype(float)
    point, draws = diff_boot(held, byp(comparator), byp("gate_trained"), pd.Series(1.0, index=held.index))
    r = bound(point, draws, "lower")
    return {"endpoint": f"claim_bypass_vs_{comparator}", "arm_a": comparator, "arm_b": "gate_trained",
            "metric": "bypass_rate", "side": "lower", "margin": 0.0, "strict": True,
            "n": len(held), "days": held.day.nunique(), **r,
            "pass": bool(r["estimable"] and r["bound"] > 0)}


# ---- driver --------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True, help="JSON listing every input path")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text())
    seqs, accounting = [], []
    for rot, rc in cfg["rotations"].items():
        names = json.loads((Path(rc["dataset"]) / "provenance.json").read_text())["class_names"]
        class_ids = {n: i for i, n in enumerate(names)}
        class_bins = {n: cg.BINS[n] for n in names if n in cg.BINS}
        held = set(rc["heldout"])
        scored = pd.concat([pd.read_csv(p) for p in rc["vlm_scored"]], ignore_index=True)
        scored["uss_track"] = scored["uss_track"].fillna(True) if "uss_track" in scored else True
        scored = scored.drop_duplicates(KEYS)
        if rc.get("low_extra"):
            scored["truth"] = np.where(scored.target_object.isin(rc["low_extra"]), "low", scored.truth)
        fx = pd.read_csv(rc["vlm_fixed"]).drop_duplicates(KEYS)
        fx["uss_track"] = True
        v, vfix = cg.vlm_sequences(scored), cg.vlm_sequences(fx)
        man = pd.read_csv(Path(rc["dataset"]) / "dataset_manifest.csv", usecols=["split", "sequence_id"])
        test_ids = set(man.loc[man.split == "test", "sequence_id"])
        det = pd.read_csv(Path(rc["det_test"]) / "crops.csv")
        feats = np.load(Path(rc["det_test"]) / "features.npy")
        fl, unavailable = frame_labels(det, feats, Path(rc["gate"]), class_ids)
        sd = sequence_decisions(fl, v[v.index.isin(test_ids)], vfix, class_bins)
        sd["rotation"], sd["tier"] = rot, rc["tier"]
        sd["population"] = np.where(sd.type.isin(held), "heldout", "known")
        requested = set(scored.loc[scored.sequence_id.isin(test_ids), "sequence_id"])
        vf = set(v.index[v.vlm_status != "ok"]) | set(vfix.index[vfix.vlm_status != "ok"])
        sd = sd[~sd.index.isin(vf)]
        accounting.append({"rotation": rot, "requested": len(requested), "evaluated": len(sd),
                           "excluded": len(requested) - len(sd),
                           "excluded_frac": (len(requested) - len(sd)) / max(len(requested), 1),
                           "gate_unavailable": unavailable})
        seqs.append(sd)
    allseq = pd.concat(seqs)
    allseq.index.name = "sequence_id"
    acct = pd.DataFrame(accounting).set_index("rotation")
    a.out.mkdir(parents=True, exist_ok=True)
    allseq.to_csv(a.out / "sequence_decisions.csv")
    acct.to_csv(a.out / "accounting.csv")

    def excl(rots):
        r = acct.loc[list(rots)]
        return float(r.excluded.sum() / max(r.requested.sum(), 1))

    def unavailable(rots, gate="gate_trained"):
        return any(acct.loc[r, "gate_unavailable"][gate] for r in rots)

    C = allseq[allseq.tier == "C"]
    D = allseq[allseq.tier == "D"]
    c_rots, d_rots = sorted(set(C.rotation)), sorted(set(D.rotation))
    tall = C[(C.population == "heldout") & (C.truth == "high")]
    known = C[C.population == "known"]
    lowd = D[(D.population == "heldout") & D.type.isin(cfg.get("dev_low_types", ["woodenboard", "speedbump"]))]
    ec, ed = excl(c_rots), excl(d_rots)
    eps = [endpoint("P1", tall, "gate_trained", "detector_high", "false_low", "upper", 0.02, excluded_frac=ec),
           endpoint("P2a", known, "gate_trained", "detector_high", "accuracy", "lower", -0.01, excluded_frac=ec),
           endpoint("P2b", known, "gate_trained", "detector_high", "low_recall", "lower", -0.02, excluded_frac=ec),
           endpoint("P2c", known, "gate_trained", "detector_high", "false_low", "upper", 0.01, excluded_frac=ec),
           p3(C, ec),
           endpoint("S1a", lowd, "gate_trained", "vlm_uss", "low_recall", "lower", -0.05, excluded_frac=ed),
           endpoint("S1b", lowd, "gate_trained", "detector_high", "low_recall", "lower", 0.0, strict=True,
                    excluded_frac=ed)]
    if unavailable(c_rots) or unavailable(d_rots):
        for e in eps:
            e["pass"], e["note"] = False, "trained gate calibration unavailable"
    claims = []
    held_all = allseq[allseq.population == "heldout"]
    for comp in ("gate_free", "gate_conf"):
        claims.append(bypass_claim(held_all, comp))
        claims += [endpoint(f"claim_NI_P1_vs_{comp}", tall, "gate_trained", comp, "false_low", "upper", 0.02),
                   endpoint(f"claim_NI_P2a_vs_{comp}", known, "gate_trained", comp, "accuracy", "lower", -0.01),
                   endpoint(f"claim_NI_P2b_vs_{comp}", known, "gate_trained", comp, "low_recall", "lower", -0.02),
                   endpoint(f"claim_NI_P2c_vs_{comp}", known, "gate_trained", comp, "false_low", "upper", 0.01)]
    res = pd.DataFrame(eps + claims)
    res.to_csv(a.out / "endpoints.csv", index=False)
    go = bool(res[res.endpoint.isin(["P1", "P2a", "P2b", "P2c", "P3", "S1a", "S1b"])]["pass"].all())

    # descriptive tables: every arm per (tier, rotation, population, type), and per day
    rows = []
    for keys, x in allseq.groupby(["tier", "rotation", "population", "type"]):
        for arm in ARMS:
            low, high = x.truth == "low", x.truth == "high"
            rows.append({**dict(zip(["tier", "rotation", "population", "type"], keys)), "arm": arm,
                         "n": len(x), "correct": int((x[arm] == x.truth).sum()),
                         "low_n": int(low.sum()), "low_as_low": int((x[arm][low] == "low").sum()),
                         "high_n": int(high.sum()), "false_low": int((x[arm][high] == "low").sum()),
                         "days": x.day.nunique()})
    pd.DataFrame(rows).to_csv(a.out / "arms_by_type.csv", index=False)
    (allseq.groupby(["tier", "rotation", "population", "day"])
     .apply(lambda x: pd.Series({arm: (x[arm] == x.truth).mean() for arm in ARMS} | {"n": len(x)}),
            include_groups=False).to_csv(a.out / "arms_by_day.csv"))
    summary = {"GO": go, "note": "internal research/resource decision (§6); not proof of generalization",
               "endpoints": res[["endpoint", "point", "bound", "margin", "pass"]].to_dict("records")}
    (a.out / "decision.json").write_text(json.dumps(summary, indent=1, default=float))
    pd.set_option("display.width", 220)
    print(res[["endpoint", "n", "days", "point", "bound", "margin", "excluded_frac", "pass"]].to_string(index=False))
    print(f"\nGO = {go}")


if __name__ == "__main__":
    main()
