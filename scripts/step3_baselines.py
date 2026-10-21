"""Step 3: all baseline arms side by side, plus routing metrics.

Arms, evaluated per population on ONE common set of sequences (detector
evaluated, VLM ok in every VLM arm present):
  detector        held-out (or baseline) detector alone; "none" = abstain
  detector_high   detector, with bucket / none replaced by constant High
  vlm_full        VLM on the full frame
  vlm_fixed       VLM on a fixed camera-only crop (same size and prompt as the
                  USS crop; needs notebook 24 CROP_MODE="fixed"), if supplied
  vlm_uss         VLM on the USS crop, full frame when the USS has no track
  cascade_uss     plain cascade (named class decides, else vlm_uss)
  cascade_full    plain cascade with vlm_full

Populations, per held-out detector T (woodenboard, speedbump, bush, bollard):
  heldout  - sequences of type T
  known    - sequences of every other type that are in THAT model's test split,
             read from its own dataset manifest (--manifest T=path). The
             baseline v2 split must not be used: e.g. cone is test-only in v2
             but mostly train in the held-out rotations. The manifest defines
             which frames are ELIGIBLE (in that model's test split); it is not
             evidence that inference ran. Without notebook 07's
             inference_status.csv, coverage is "unverified" and the stated
             assumption is that every eligible test frame was processed.
Metrics: n, correct, Low recall, false Low (High truth called Low, over High
truth; undefined without High truth), false High (Low truth called High,
over Low truth), semantic abstention BEFORE the default-High fallback
(VLM gave no usable name / detector gave no class), final no-decision after
fallback, and vlm_frames_per_seq_counterfactual: frames the VLM would see per
sequence under the routing (VLM outputs were precomputed; this is not a
measured call count or latency). Accounting per population: requested,
eligible and excluded sequences; requested / ineligible / failed / missing
frames; partial sequences.
Routing (detector side): unknown-routing recall (held-out sequences deferred),
wrong deferral on known types - split into NAMED known types (the detector has
their class) and BUCKET-member known types (their correct detector answer is
the bucket, so deferral is by design) - and bypass errors (held-out sequences
NOT deferred and wrong). Uncertainty: 95% percentile bootstrap over recording
days (sequence id field 2), 2000 resamples, seed 0, for accuracy, Low recall,
false-Low and false-High rates and for paired differences. The day is the
inferential unit; with fewer than 5 days an interval is flagged "weak".
Exact McNemar on sequences is reported only as an independence-assuming
descriptive sensitivity check. Day-wise effects are listed for the low types.

usage: python scripts/step3_baselines.py --results paper/results \
           --manifest woodenboard=.../dataset_v2holdwoodenboard/dataset_manifest.csv \
                      speedbump=... bush=... bollard=... \
           [--fixed method/prompt_decision_fixedcrop_v1/scored.csv] --out step3.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from math import comb

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cascade_gtfree as cg  # noqa: E402

HELDOUT = ("woodenboard", "speedbump", "bush", "bollard")
ARMS = ("detector", "detector_high", "vlm_full", "vlm_fixed", "vlm_uss",
        "cascade_uss", "cascade_full")
PAIRS = (("cascade_uss", "detector_high"), ("cascade_uss", "vlm_full"),
         ("vlm_uss", "vlm_full"), ("vlm_uss", "vlm_fixed"), ("vlm_fixed", "vlm_full"))
B, SEED = 2000, 0


RAW = {"vlm_full": "full_raw", "vlm_uss": "uss_raw", "vlm_fixed": "fixed_raw"}
WEAK_DAYS = 5


def decisions(v: pd.DataFrame, det: pd.DataFrame, vfix: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    """(one row per sequence of the common population, accounting)."""
    req = det.index
    d_ok = det.cls != cg.NOT_EVALUATED
    v_ok = (v.vlm_status.reindex(req) == "ok").fillna(False)
    f_ok = (vfix.vlm_status.reindex(req) == "ok").fillna(False) if vfix is not None else v_ok | True
    ok = d_ok & v_ok & f_ok
    acct = {"requested_seqs": len(req), "eligible_seqs": int(ok.sum()),
            "excl_not_evaluated": int((~d_ok).sum()),
            "excl_vlm_failed": int((d_ok & ~v_ok).sum()),
            "excl_fixed_failed": int((d_ok & v_ok & ~f_ok).sum()),
            "frames_requested": int(det.requested.sum()), "frames_ok": int(det.ok.sum()),
            "frames_failed": int(det.failed.sum()), "frames_missing": int(det.missing.sum()),
            "partial_seqs": int(det.partial[ok].sum())}
    ids = req[ok]
    d, cls = v.loc[ids], det.cls.loc[ids]
    named = ~cls.isin(cg.BUCKET | {"none"})
    detb = cls.map(cg.BINS)
    out = pd.DataFrame({
        "type": d.target_object, "truth": d.truth, "day": ids.str.split("_").str[1],
        "deferred": ~named, "detector_class": cls,
        "detector": detb, "detector_high": detb.where(named, "high"),
        "vlm_full": d.full, "vlm_uss": d.uss, "full_raw": d.full_raw, "uss_raw": d.uss_raw,
        "cascade_uss": cg.cascade(d, cls, "uss", "plain"),
        "cascade_full": cg.cascade(d, cls, "full", "plain"),
        "frames": d.vlm_frames}, index=ids)
    if vfix is not None:
        out["vlm_fixed"], out["fixed_raw"] = vfix.uss.loc[ids], vfix.uss_raw.loc[ids]
    return out, acct


def semantic_abstain(x: pd.DataFrame, arm: str) -> int:
    """Abstentions BEFORE the default-High fallback."""
    if arm in RAW:
        return int(x[RAW[arm]].isna().sum())
    if arm == "detector":
        return int((x.detector_class == "none").sum())
    if arm.startswith("cascade_"):
        raw = "uss_raw" if arm == "cascade_uss" else "full_raw"
        return int((x.deferred & x[raw].isna()).sum())
    return 0                                   # detector_high never abstains by construction


def rates(y: pd.DataFrame, arm: str) -> dict:
    dec, low, high = y[arm], y.truth == "low", y.truth == "high"
    return {"accuracy": (dec == y.truth).mean() if len(y) else np.nan,
            "low_recall": (dec[low] == "low").mean() if low.any() else np.nan,
            "false_low_rate": (dec[high] == "low").mean() if high.any() else np.nan,
            "false_high_rate": (dec[low] == "high").mean() if low.any() else np.nan}


def arm_metrics(x: pd.DataFrame, arm: str) -> dict:
    dec, low, high = x[arm], x.truth == "low", x.truth == "high"
    frames = {"cascade_uss": (x.frames * x.deferred).sum(), "cascade_full": (x.frames * x.deferred).sum(),
              "detector": 0, "detector_high": 0}.get(arm, x.frames.sum())
    r = {k: (None if pd.isna(v) else round(float(v), 3)) for k, v in rates(x, arm).items()}
    return {"n": len(x), "correct": int((dec == x.truth).sum()),
            "low_n": int(low.sum()), "high_n": int(high.sum()),
            "false_low": int((dec[high] == "low").sum()), "false_high": int((dec[low] == "high").sum()),
            "semantic_abstain": semantic_abstain(x, arm),
            "final_no_decision": int(dec.isna().sum()), **r,
            "vlm_frames_per_seq_counterfactual": round(float(frames) / len(x), 2) if len(x) else None}


def day_counts(x: pd.DataFrame, arm: str, days: np.ndarray) -> np.ndarray:
    """Per-day sufficient counts [n, correct, low_n, low_as_low, high_n, high_as_low, low_as_high]."""
    dec, low, high = x[arm], x.truth == "low", x.truth == "high"
    f = pd.DataFrame({"day": x.day, "n": 1, "correct": (dec == x.truth).astype(int),
                      "low_n": low.astype(int), "low_low": (low & (dec == "low")).astype(int),
                      "high_n": high.astype(int), "high_low": (high & (dec == "low")).astype(int),
                      "low_high": (low & (dec == "high")).astype(int)})
    return f.groupby("day").sum().reindex(days).fillna(0).to_numpy(float)


def rates_from_counts(c: np.ndarray) -> dict:
    """c: (..., 7) summed counts -> rate arrays (NaN where the denominator is 0)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"accuracy": c[..., 1] / c[..., 0], "low_recall": c[..., 3] / c[..., 2],
                "false_low_rate": c[..., 5] / c[..., 4], "false_high_rate": c[..., 6] / c[..., 2]}


def resample_index(x: pd.DataFrame, rng) -> tuple[np.ndarray, np.ndarray]:
    """(days, B x ndays matrix of resampled day indices); one draw shared by all
    arms and pairs of a table, so paired differences use the same resamples."""
    days = np.array(sorted(x.day.unique()))
    return days, rng.integers(0, len(days), size=(B, len(days)))


def boot_ci(vals: np.ndarray) -> tuple[float, float]:
    if np.isnan(vals).all():
        return (np.nan, np.nan)
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return round(float(lo), 3), round(float(hi), 3)


def ci(lo, hi) -> str:
    return "" if pd.isna(lo) else f"{lo}-{hi}"


def mcnemar(w: int, l: int) -> float:
    n = w + l
    return min(1.0, 2 * sum(comb(n, k) for k in range(max(w, l), n + 1)) / 2 ** n) if n else 1.0


def table(x: pd.DataFrame, population: str, model: str) -> list[dict]:
    rng = np.random.default_rng(SEED)
    ndays = x.day.nunique()
    weak = ndays < WEAK_DAYS
    days, idx = resample_index(x, rng)
    boot = {}
    for arm in ARMS:
        if arm in x:
            c = day_counts(x, arm, days)
            boot[arm] = rates_from_counts(c[idx].sum(axis=1)) if ndays >= 2 else None
    rows = []
    for arm in ARMS:
        if arm not in x:
            continue
        row = {"model": model, "population": population, "kind": "arm", "arm": arm,
               "days": ndays, "weak_day_uncertainty": weak, **arm_metrics(x, arm)}
        for k in ("accuracy", "low_recall", "false_low_rate", "false_high_rate"):
            row[f"{k}_ci95_day"] = ci(*boot_ci(boot[arm][k])) if boot[arm] else ""
        rows.append(row)
    for a, b in PAIRS:
        if a not in x or b not in x:
            continue
        row = {"model": model, "population": population, "kind": "paired", "arm": f"{a} - {b}",
               "n": len(x), "days": ndays, "weak_day_uncertainty": weak}
        for k in ("accuracy", "low_recall", "false_low_rate"):
            point = rates(x, a)[k] - rates(x, b)[k]
            lo, hi = boot_ci(boot[a][k] - boot[b][k]) if boot[a] else (np.nan, np.nan)
            row[f"{k}_diff"] = None if pd.isna(point) else round(float(point), 3)
            row[f"{k}_diff_ci95_day"] = ci(lo, hi)
            if k == "accuracy":
                row["across_days"] = ("not estimable" if pd.isna(lo) else
                                      ("interval excludes 0" if (lo > 0 or hi < 0) else "inconclusive"))
                if weak and not pd.isna(lo):
                    row["across_days"] += f" (weak: {ndays} days)"
        w = int(((x[a] == x.truth) & (x[b] != x.truth)).sum())
        l = int(((x[a] != x.truth) & (x[b] == x.truth)).sum())
        row.update({"wins": w, "losses": l, "mcnemar_p_seq_independence_assumed": round(mcnemar(w, l), 6)})
        rows.append(row)
    return rows


def daywise(x: pd.DataFrame, model: str) -> list[dict]:
    rows = []
    for day, g in x.groupby("day"):
        r = {"model": model, "population": "heldout", "kind": "daywise", "arm": day, "n": len(g)}
        for arm in ARMS:
            if arm in g:
                r[f"acc_{arm}"] = round(float((g[arm] == g.truth).mean()), 3)
        rows.append(r)
    return rows


NAMED = {"curbstone", "speedbump", "woodenboard", "pole", "dummychild", "bicyclestand"}


def routing(held: pd.DataFrame, known: pd.DataFrame, model: str) -> dict:
    bypass = (~held.deferred) & (held.detector != held.truth)
    kn, kb = known[known.type.isin(NAMED)], known[~known.type.isin(NAMED)]
    return {"model": model, "population": "routing", "kind": "routing",
            "heldout_n": len(held), "known_named_n": len(kn), "known_bucket_n": len(kb),
            "unknown_routing_recall": cg.rate(int(held.deferred.sum()), len(held)),
            "wrong_deferral_known_named": cg.rate(int(kn.deferred.sum()), len(kn)),
            "deferral_known_bucket_members": cg.rate(int(kb.deferred.sum()), len(kb)),
            "bypass_errors": int(bypass.sum()),
            "bypass_error_rate": cg.rate(int(bypass.sum()), len(held))}


def manifest_eligible(manifest: Path) -> tuple[set[str], set[str]]:
    """(test sequence ids, test image keys) of a held-out dataset manifest:
    ELIGIBILITY only, not evidence that inference ran."""
    m = pd.read_csv(manifest, usecols=["split", "sequence_id", "camera", "frame_idx"])
    m = m[m.split == "test"].drop_duplicates(["sequence_id", "camera", "frame_idx"])
    img = m.sequence_id + "_" + m.camera + "_" + m.frame_idx.map(lambda i: f"{int(i):05d}")
    return set(m.sequence_id), set(img)


def load_vlm(results: Path, fixed: Path | None):
    runs = ("prompt_decision_v1", "prompt_decision_low_v1", "prompt_decision_high_v1")
    scored = pd.concat([pd.read_csv(results / r / "scored.csv") for r in runs], ignore_index=True)
    scored["uss_track"] = scored["uss_track"].fillna(True)
    scored = scored.drop_duplicates(cg.KEYS)
    vfix = None
    if fixed is not None:
        fx = pd.read_csv(fixed).drop_duplicates(cg.KEYS)
        a = set(map(tuple, scored[cg.KEYS].astype(str).values))
        b = set(map(tuple, fx[cg.KEYS].astype(str).values))
        assert a == b, f"frame keys differ: {len(a - b)} only in VLM runs, {len(b - a)} only in fixed run"
        fx["uss_track"] = True                    # its crop column is the fixed crop
        vfix = cg.vlm_sequences(fx)
    return scored, cg.vlm_sequences(scored), vfix


def population_rows(frames: pd.DataFrame, raw: pd.DataFrame, eligible_img: set[str],
                    status: pd.DataFrame | None, v, vfix, name: str, model: str):
    img = cg.image_key(frames)
    inel = int((~img.isin(eligible_img)).sum())
    det = cg.detector_sequences(frames[img.isin(eligible_img)], raw, status)
    x, acct = decisions(v, det, vfix)
    acct["frames_ineligible"] = inel
    coverage = det.coverage.iat[0] if len(det) else ""
    if status is None:
        coverage = "unverified (assumes every eligible test frame was processed)"
    rows = [{**r, **acct, "coverage": coverage} for r in table(x, name, model)]
    return x, rows


def build(results: Path, fixed: Path | None, manifests: dict[str, Path]) -> pd.DataFrame:
    scored, v, vfix = load_vlm(results, fixed)
    rows = []
    for t in HELDOUT:
        rep = results / f"eval_hold{t}"
        st = rep / "inference_status.csv"
        status = pd.read_csv(st) if st.exists() else None
        raw = pd.read_csv(rep / "raw_predictions.csv")
        test_ids, eligible_img = manifest_eligible(manifests[t])
        assert set(raw.image) <= eligible_img, f"{t}: raw detections outside the test manifest"
        pop = {"heldout": scored[scored.target_object == t],
               "known": scored[(scored.target_object != t) & scored.sequence_id.isin(test_ids)]}
        assert set(pop["heldout"].sequence_id) <= test_ids, f"{t}: held-out sequences not all test"
        x = {}
        for name, frames in pop.items():
            x[name], r = population_rows(frames, raw, eligible_img, status, v, vfix, name, f"hold{t}")
            rows += r
        if t in ("woodenboard", "speedbump"):
            rows += daywise(x["heldout"], f"hold{t}")
        rows.append(routing(x["heldout"], x["known"], f"hold{t}"))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("paper/results"))
    ap.add_argument("--fixed", type=Path, default=None, help="fixed-crop scored.csv")
    ap.add_argument("--manifest", nargs=4, required=True,
                    help="type=path to each held-out dataset_manifest.csv (all four types)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    manifests = {k: Path(v) for k, v in (m.split("=", 1) for m in a.manifest)}
    assert set(manifests) == set(HELDOUT), f"need manifests for {HELDOUT}"
    t = build(a.results, a.fixed, manifests)
    t.to_csv(a.out, index=False)
    print(f"{len(t)} rows -> {a.out}")


if __name__ == "__main__":
    main()
