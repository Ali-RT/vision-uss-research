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


FAILED_DET = ("det_failed", "det_missing")
LOW_TYPES = {"curbstone", "speedbump", "woodenboard", "hose", "step"}


def det_failed(status: pd.Series) -> pd.Series:
    """Detector-side execution failures: never vote, always counted."""
    s = status.fillna("").astype(str)
    return s.str.startswith("error") | s.isin(FAILED_DET)


def frame_labels(det: pd.DataFrame, feats: np.ndarray, gate_dir: Path, class_ids: dict
                 ) -> tuple[pd.DataFrame, dict]:
    """Per-frame labels for the three gates and the raw detector class; failed
    frames are labelled FAILED and must be dropped before voting."""
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
    failed = det_failed(det.crop_status).to_numpy()
    valid = (det.crop_status == "valid").to_numpy()
    z_t = np.full((len(det), g.EMB_DIM), np.nan)
    z_f = np.full((len(det), g.FEAT_DIM), np.nan)
    if valid.any():
        rows = det.feature_row.to_numpy()[valid]
        z_t[valid] = g.embed(head, feats[rows])
        z_f[valid] = g.embed(None, feats[rows])
    out = []
    for i, r in enumerate(det.itertuples(index=False)):
        if failed[i]:
            out.append({"det_class": "FAILED", **{k: "FAILED" for k in GATES}})
            continue
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
    lab["det_failed"] = failed
    return pd.concat([det[KEYS + ["crop_status"]], lab], axis=1), unavailable


def check_vlm_rows(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate frame rows (counted); conflicting duplicates raise."""
    cols = [c for c in ("crop_name", "full_name", "truth", "error") if c in df]
    dup = df.duplicated(KEYS, keep=False)
    if dup.any():
        d = df[dup]
        conflict = d.groupby(KEYS)[cols].nunique(dropna=False).max(axis=1) > 1
        if conflict.any():
            raise ValueError(f"{label}: {int(conflict.sum())} frame keys with conflicting duplicate rows")
    n_dup = int(df.duplicated(KEYS).sum())
    return df.drop_duplicates(KEYS), n_dup


def frame_availability(requested: pd.DataFrame, df: pd.DataFrame, failed: pd.Series) -> pd.Series:
    """Per requested frame: ok | failed | missing."""
    key = lambda x: list(map(tuple, x[KEYS].astype(str).values))  # noqa: E731
    have = dict(zip(key(df), np.where(failed.to_numpy(), "failed", "ok")))
    return pd.Series([have.get(k, "missing") for k in key(requested)], index=requested.index)


def sequence_decisions(fl: pd.DataFrame, v: pd.DataFrame, vfix: pd.DataFrame,
                       class_bins: dict, eligible: list) -> pd.DataFrame:
    """One row per ELIGIBLE sequence: label per gate, routed decision per arm.
    Failed detector frames never vote."""
    ev = fl[~fl.det_failed]
    agg = ev.groupby("sequence_id").agg(
        det_mode=("det_class", unique_mode),
        det_strict=("det_class", lambda x: g.strict_majority(list(x))),
        **{f"{k}_lab": (k, lambda x: g.strict_majority(list(x))) for k in GATES})
    d = v.loc[eligible].join(agg.loc[eligible])
    d["fixed"] = vfix.uss.loc[eligible]
    assert d[["uss", "full", "fixed"]].notna().all().all(), "an eligible sequence lacks a VLM decision"
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
    """Point estimate and bootstrap draws of sum(num)/sum(den), days resampled.
    An empty population is undefined (NaN), never an error."""
    if len(x) == 0:
        return np.nan, np.full(B, np.nan)
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
def load_rotation(rot: str, rc: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(membership of the frozen requested population, decisions of eligible
    sequences, accounting). Missing inputs make every sequence excluded."""
    frames = pd.read_csv(rc["frames"])
    if frames.duplicated(KEYS).any():
        raise ValueError(f"{rot}: duplicate keys in the frozen frame list")
    low = LOW_TYPES | set(rc.get("low_extra", []))
    held = set(rc["heldout"])
    member = frames.drop_duplicates("sequence_id").set_index("sequence_id")[["target_object"]]
    member["truth"] = np.where(member.target_object.isin(low), "low", "high")
    member["population"] = np.where(member.target_object.isin(held), "heldout", "known")
    member["rotation"], member["tier"] = rot, rc["tier"]
    member["day"] = member.index.str.split("_").str[1]
    acct = {"rotation": rot, "requested_seqs": len(member), "requested_frames": len(frames)}
    need = [Path(rc["dataset"]) / "provenance.json", Path(rc["det_test"]) / "crops.csv",
            Path(rc["det_test"]) / "features.npy", Path(rc["gate"]) / "calibration.json",
            Path(rc["vlm_fixed"]), *map(Path, rc["vlm_scored"])]
    missing = [str(q) for q in need if not q.exists()]
    if missing:
        acct.update(evaluated_seqs=0, inputs_missing=missing,
                    gate_unavailable={k: True for k in GATES})
        return member, pd.DataFrame(), acct
    names = json.loads((Path(rc["dataset"]) / "provenance.json").read_text())["class_names"]
    class_ids = {n: i for i, n in enumerate(names)}
    class_bins = {n: cg.BINS[n] for n in names if n in cg.BINS}
    key = frames[KEYS].astype(str).apply(tuple, axis=1)
    keyset = set(key)

    def restrict(df):
        return df[df[KEYS].astype(str).apply(tuple, axis=1).isin(keyset)]

    scored, dup_s = check_vlm_rows(restrict(pd.concat([pd.read_csv(q) for q in rc["vlm_scored"]],
                                                      ignore_index=True)), f"{rot} scored")
    fx, dup_f = check_vlm_rows(restrict(pd.read_csv(rc["vlm_fixed"])), f"{rot} fixed")
    # truth: the frozen membership is authoritative; the primary VLM run must agree
    t = scored.groupby("sequence_id").truth.nunique()
    assert (t <= 1).all(), f"{rot} scored: inconsistent truth within a sequence"
    first = scored.drop_duplicates("sequence_id").set_index("sequence_id").truth
    assert (first == member.truth.reindex(first.index)).all(), \
        f"{rot} scored: truth differs from the frozen membership"
    fx_truth_mismatch = int((fx.truth != fx.sequence_id.map(member.truth)).sum())
    scored["truth"] = scored.sequence_id.map(member.truth)
    fx["truth"] = fx.sequence_id.map(member.truth)
    scored["uss_track"] = scored["uss_track"].fillna(True) if "uss_track" in scored else True
    fx["uss_track"] = True
    det = restrict(pd.read_csv(Path(rc["det_test"]) / "crops.csv"))
    det, dup_d = det.drop_duplicates(KEYS), int(det.duplicated(KEYS).sum())
    feats = np.load(Path(rc["det_test"]) / "features.npy")
    fl, unavailable = frame_labels(det, feats, Path(rc["gate"]), class_ids)
    av = pd.DataFrame({
        "detector": frame_availability(frames, det, det_failed(det.crop_status)),
        "vlm": frame_availability(frames, scored, scored.error.fillna("") != ""),
        "fixed": frame_availability(frames, fx, fx.error.fillna("") != "")}, index=frames.index)
    av["sequence_id"] = frames.sequence_id.values
    per = av.groupby("sequence_id").agg(**{f"{a}_{st}": (a, lambda x, st=st: int((x == st).sum()))
                                           for a in ("detector", "vlm", "fixed")
                                           for st in ("ok", "failed", "missing")})
    ok_all = (per.detector_ok > 0) & (per.vlm_ok > 0) & (per.fixed_ok > 0)
    eligible = list(per.index[ok_all])
    v, vfix = cg.vlm_sequences(scored), cg.vlm_sequences(fx)
    sd = sequence_decisions(fl, v, vfix, class_bins, eligible)
    sd = sd.join(member[["population", "rotation", "tier"]])
    sd = sd.join(per)
    tot = per[[c for c in per if c.startswith("detector_")]].sum(axis=1)   # requested frames
    acct.update(evaluated_seqs=len(sd), excluded_seqs=len(member) - len(sd),
                duplicate_rows_dropped={"scored": dup_s, "fixed": dup_f, "detector": dup_d},
                fixed_truth_overwritten=fx_truth_mismatch,
                frames={f"{c}_{st}": int((av[c] == st).sum()) for c in ("detector", "vlm", "fixed")
                        for st in ("ok", "failed", "missing")},
                partial_seqs={c: int(((per[f"{c}_ok"] > 0) & (per[f"{c}_ok"] < tot)).sum())
                              for c in ("detector", "vlm", "fixed")},
                gate_unavailable=unavailable, inputs_missing=[])
    return member, sd, acct


def excluded_frac(member: pd.DataFrame, evaluated: pd.Index, truth: str | None = None) -> float:
    m = member if truth is None else member[member.truth == truth]
    return float((~m.index.isin(evaluated)).sum() / len(m)) if len(m) else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True, help="JSON listing every input path")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text())
    members, seqs, accounting = [], [], []
    for rot, rc in cfg["rotations"].items():
        m, sd, acct = load_rotation(rot, rc)
        members.append(m)
        accounting.append(acct)
        if len(sd):
            seqs.append(sd)
    member = pd.concat(members)
    allseq = pd.concat(seqs) if seqs else pd.DataFrame(columns=["tier"])
    allseq.index.name = "sequence_id"
    acct = pd.DataFrame(accounting).set_index("rotation")
    a.out.mkdir(parents=True, exist_ok=True)
    allseq.to_csv(a.out / "sequence_decisions.csv")
    acct.to_json(a.out / "accounting.json", orient="index", indent=1)

    def unavailable(rots, gate):
        return any(acct.loc[r, "gate_unavailable"][gate] for r in rots)

    tiers = {t: sorted(r for r, rc in cfg["rotations"].items() if rc["tier"] == t) for t in ("C", "D")}
    C, D = allseq[allseq.tier == "C"], allseq[allseq.tier == "D"]
    mC, mD = member[member.tier == "C"], member[member.tier == "D"]
    dev_low = cfg.get("dev_low_types", ["woodenboard", "speedbump"])
    pops = {"P1": (mC[(mC.population == "heldout") & (mC.truth == "high")], C),
            "P2": (mC[mC.population == "known"], C),
            "P3": (mC, C),
            "S1": (mD[(mD.population == "heldout") & mD.target_object.isin(dev_low)], D)}

    def pop(name):
        m, frame = pops[name]
        return m, frame[frame.index.isin(m.index)]

    def ep(eid, popname, arm_b, metric, side, margin, strict=False, arm_a="gate_trained", tier="C"):
        m, x = pop(popname)
        truth = {"low_recall": "low", "false_low": "high"}.get(metric)
        ef = excluded_frac(m, x.index, truth)
        r = endpoint(eid, x, arm_a, arm_b, metric, side, margin, strict=strict, excluded_frac=ef)
        r["requested"] = int(len(m) if truth is None else (m.truth == truth).sum())
        gates = [g_ for g_ in (arm_a, arm_b) if g_ in GATES]
        if any(unavailable(tiers[tier], g_) for g_ in gates):
            r["pass"], r["note"] = False, "gate calibration unavailable"
        return r

    eps = [ep("P1", "P1", "detector_high", "false_low", "upper", 0.02),
           ep("P2a", "P2", "detector_high", "accuracy", "lower", -0.01),
           ep("P2b", "P2", "detector_high", "low_recall", "lower", -0.02),
           ep("P2c", "P2", "detector_high", "false_low", "upper", 0.01)]
    mP3, xP3 = pop("P3")
    r3 = p3(xP3, excluded_frac(mP3, xP3.index))
    r3["requested"] = len(mP3)
    if unavailable(tiers["C"], "gate_trained"):
        r3["pass"], r3["note"] = False, "gate calibration unavailable"
    eps.append(r3)
    eps += [ep("S1a", "S1", "vlm_uss", "low_recall", "lower", -0.05, tier="D"),
            ep("S1b", "S1", "detector_high", "low_recall", "lower", 0.0, strict=True, tier="D")]

    claims = []
    mheld = member[member.population == "heldout"]
    held = allseq[allseq.index.isin(mheld.index)]
    for comp in ("gate_free", "gate_conf"):
        rb = bypass_claim(held, comp)
        rb["requested"], rb["excluded_frac"] = len(mheld), excluded_frac(mheld, held.index)
        if rb["excluded_frac"] > EXCLUDED_MAX:
            rb["pass"] = False
        if any(unavailable(tiers["C"] + tiers["D"], g_) for g_ in ("gate_trained", comp)):
            rb["pass"], rb["note"] = False, "gate calibration unavailable"
        ni = [ep(f"claim_NI_P1_vs_{comp}", "P1", comp, "false_low", "upper", 0.02),
              ep(f"claim_NI_P2a_vs_{comp}", "P2", comp, "accuracy", "lower", -0.01),
              ep(f"claim_NI_P2b_vs_{comp}", "P2", comp, "low_recall", "lower", -0.02),
              ep(f"claim_NI_P2c_vs_{comp}", "P2", comp, "false_low", "upper", 0.01)]
        claims += [rb] + ni
        claims.append({"endpoint": f"claim_vs_{comp}", "metric": "conjunction: bypass + 4 NI",
                       "pass": bool(rb["pass"] and all(r["pass"] for r in ni))})
    res = pd.DataFrame(eps + claims)
    res.to_csv(a.out / "endpoints.csv", index=False)
    go = bool(res[res.endpoint.isin(["P1", "P2a", "P2b", "P2c", "P3", "S1a", "S1b"])]["pass"].all())

    def arm_counts(x, arm):
        low, high = x.truth == "low", x.truth == "high"
        return {"n": len(x), "correct": int((x[arm] == x.truth).sum()),
                "low_n": int(low.sum()), "low_as_low": int((x[arm][low] == "low").sum()),
                "false_high": int((x[arm][low] == "high").sum()),
                "high_n": int(high.sum()), "false_low": int((x[arm][high] == "low").sum())}

    rows, drows = [], []
    for keys, x in allseq.groupby(["tier", "rotation", "population", "type"]):
        for arm in ARMS:
            rows.append({**dict(zip(["tier", "rotation", "population", "type"], keys)), "arm": arm,
                         "days": x.day.nunique(), **arm_counts(x, arm)})
    for keys, x in allseq.groupby(["tier", "rotation", "population", "day"]):
        for arm in ARMS:
            drows.append({**dict(zip(["tier", "rotation", "population", "day"], keys)), "arm": arm,
                          **arm_counts(x, arm)})
    pd.DataFrame(rows).to_csv(a.out / "arms_by_type.csv", index=False)
    pd.DataFrame(drows).to_csv(a.out / "arms_by_day.csv", index=False)
    summary = {"GO": go, "note": "internal research/resource decision (§6); not proof of generalization",
               "endpoints": res[["endpoint", "point", "bound", "margin", "pass"]].to_dict("records")
               if "point" in res else []}
    (a.out / "decision.json").write_text(json.dumps(summary, indent=1, default=float))
    pd.set_option("display.width", 220)
    cols = [c for c in ("endpoint", "requested", "n", "days", "point", "bound", "margin", "excluded_frac",
                        "pass", "note") if c in res]
    print(res[cols].to_string(index=False))
    print(f"\nGO = {go}")


if __name__ == "__main__":
    main()
