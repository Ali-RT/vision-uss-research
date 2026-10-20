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
             but mostly train in the held-out rotations. The manifest's test
             frames also define which frames the detector was run on
             (coverage "status_only": membership known, counts not).
Metrics: n, correct, Low recall, false Low (High truth called Low, over High
truth), false High (Low truth called High, over Low truth), detector
abstention, VLM calls per sequence (frames sent to the VLM / sequences).
Routing (detector side): unknown-routing recall (held-out sequences deferred),
wrong deferral on known types - split into NAMED known types (the detector has
their class) and BUCKET-member known types (their correct detector answer is
the bucket, so deferral is by design) - and bypass errors (held-out sequences
NOT deferred and wrong). Intervals: 95% bootstrap over recording days
(sequence id field 2), 2000 resamples, seed 0. Paired differences use the
same day resamples.

usage: python scripts/step3_baselines.py --results paper/results \
           --manifest woodenboard=.../dataset_v2holdwoodenboard/dataset_manifest.csv \
                      speedbump=... bush=... bollard=... \
           [--fixed method/prompt_decision_fixedcrop_v1/scored.csv] --out step3.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def decisions(v: pd.DataFrame, det: pd.DataFrame, vfix: pd.DataFrame | None) -> pd.DataFrame:
    """One row per sequence of the common population, one column per arm."""
    ok = (det.cls != cg.NOT_EVALUATED).reindex(v.index).fillna(False) & (v.vlm_status == "ok")
    if vfix is not None:
        ok &= (vfix.vlm_status.reindex(v.index) == "ok").fillna(False)
    ids = v.index[ok]
    d, cls = v.loc[ids], det.cls.loc[ids]
    named = ~cls.isin(cg.BUCKET | {"none"})
    detb = cls.map(cg.BINS)
    out = pd.DataFrame({
        "type": d.target_object, "truth": d.truth, "day": ids.str.split("_").str[1],
        "deferred": ~named, "detector_class": cls,
        "detector": detb, "detector_high": detb.where(named, "high"),
        "vlm_full": d.full, "vlm_uss": d.uss,
        "cascade_uss": cg.cascade(d, cls, "uss", "plain"),
        "cascade_full": cg.cascade(d, cls, "full", "plain"),
        "frames": d.vlm_frames}, index=ids)
    if vfix is not None:
        out["vlm_fixed"] = vfix.uss.loc[ids]
    return out


def arm_metrics(x: pd.DataFrame, arm: str) -> dict:
    dec, low, high = x[arm], x.truth == "low", x.truth == "high"
    calls = {"cascade_uss": (x.frames * x.deferred).sum(), "cascade_full": (x.frames * x.deferred).sum(),
             "detector": 0, "detector_high": 0}.get(arm, x.frames.sum())
    return {"n": len(x), "correct": int((dec == x.truth).sum()),
            "low_n": int(low.sum()), "low_recall": cg.rate(int((dec[low] == "low").sum()), int(low.sum())),
            "high_n": int(high.sum()),
            "false_low": int((dec[high] == "low").sum()),
            "false_low_rate": cg.rate(int((dec[high] == "low").sum()), int(high.sum())),
            "false_high": int((dec[low] == "high").sum()),
            "false_high_rate": cg.rate(int((dec[low] == "high").sum()), int(low.sum())),
            "abstain": int(dec.isna().sum()),
            "vlm_calls_per_seq": round(float(calls) / len(x), 2) if len(x) else None}


def day_bootstrap(x: pd.DataFrame, stat, rng) -> tuple[float, float]:
    days = x.day.unique()
    if len(days) < 2:
        return (np.nan, np.nan)
    groups = {d: g for d, g in x.groupby("day")}
    vals = []
    for _ in range(B):
        pick = rng.choice(days, len(days), replace=True)
        vals.append(stat(pd.concat([groups[d] for d in pick])))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return round(float(lo), 3), round(float(hi), 3)


def table(x: pd.DataFrame, population: str, model: str) -> list[dict]:
    rng = np.random.default_rng(SEED)
    rows = []
    for arm in ARMS:
        if arm not in x:
            continue
        m = arm_metrics(x, arm)
        lo, hi = day_bootstrap(x, lambda y, a=arm: (y[a] == y.truth).mean(), rng)
        rows.append({"model": model, "population": population, "kind": "arm", "arm": arm,
                     "days": x.day.nunique(), **m,
                     "accuracy": cg.rate(m["correct"], m["n"]), "acc_ci95_day": f"{lo}-{hi}"})
    for a, b in PAIRS:
        if a not in x or b not in x:
            continue
        diff = lambda y, a=a, b=b: (y[a] == y.truth).mean() - (y[b] == y.truth).mean()  # noqa: E731
        lo, hi = day_bootstrap(x, diff, rng)
        wins = int(((x[a] == x.truth) & (x[b] != x.truth)).sum())
        losses = int(((x[a] != x.truth) & (x[b] == x.truth)).sum())
        rows.append({"model": model, "population": population, "kind": "paired",
                     "arm": f"{a} - {b}", "n": len(x), "days": x.day.nunique(),
                     "accuracy": round(float(diff(x)), 3) if len(x) else None,
                     "acc_ci95_day": f"{lo}-{hi}", "wins": wins, "losses": losses,
                     "verdict": ("difference" if (lo > 0 or hi < 0) else "trend / no difference")})
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


def manifest_status(manifest: Path) -> tuple[set[str], pd.DataFrame]:
    """(test sequence ids, status-like frame table) from a held-out dataset manifest."""
    m = pd.read_csv(manifest, usecols=["split", "sequence_id", "camera", "frame_idx"])
    m = m[m.split == "test"].drop_duplicates(["sequence_id", "camera", "frame_idx"])
    img = m.sequence_id + "_" + m.camera + "_" + m.frame_idx.map(lambda i: f"{int(i):05d}")
    return set(m.sequence_id), pd.DataFrame({"image": img, "status": "ok"})


def build(results: Path, fixed: Path | None, manifests: dict[str, Path]) -> pd.DataFrame:
    runs = ("prompt_decision_v1", "prompt_decision_low_v1", "prompt_decision_high_v1")
    scored = pd.concat([pd.read_csv(results / r / "scored.csv") for r in runs], ignore_index=True)
    scored["uss_track"] = scored["uss_track"].fillna(True)
    scored = scored.drop_duplicates(cg.KEYS)
    v = cg.vlm_sequences(scored)
    vfix = None
    if fixed is not None:
        fx = pd.read_csv(fixed).drop_duplicates(cg.KEYS)
        fx["uss_track"] = True                    # its crop column is the fixed crop
        vfix = cg.vlm_sequences(fx)
    rows = []
    for t in HELDOUT:
        rep = results / f"eval_hold{t}"
        st = rep / "inference_status.csv"
        raw = pd.read_csv(rep / "raw_predictions.csv")
        test_ids, mstatus = manifest_status(manifests[t])
        status = pd.read_csv(st) if st.exists() else mstatus
        pop = {"heldout": scored[scored.target_object == t],
               "known": scored[(scored.target_object != t) & scored.sequence_id.isin(test_ids)]}
        assert set(pop["heldout"].sequence_id) <= test_ids, f"{t}: held-out sequences not all test"
        dec = {}
        for name, frames in pop.items():
            det = cg.detector_sequences(frames, raw, status)
            dec[name] = decisions(v.loc[v.index.intersection(det.index)], det, vfix)
            rows += [{**r, "coverage": det.coverage.iat[0]}
                     for r in table(dec[name], name, f"hold{t}")]
        rows.append(routing(dec["heldout"], dec["known"], f"hold{t}"))
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
    pd.set_option("display.width", 250)
    cols = ["model", "population", "arm", "n", "days", "accuracy", "acc_ci95_day", "low_recall",
            "false_low", "false_high", "abstain", "vlm_calls_per_seq", "wins", "losses", "verdict"]
    print(t[t.kind != "routing"].reindex(columns=cols).to_string(index=False))
    print(t[t.kind == "routing"].dropna(axis=1, how="all").to_string(index=False))


if __name__ == "__main__":
    main()
