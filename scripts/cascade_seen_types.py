"""Cascade on the BASELINE model's test split (most types seen in training).

Same VLM side and decision rules as scripts/cascade_gtfree.py; the detector
side is the baseline's per-frame top prediction (reports/eval_causal/
frame_predictions.csv) on the frames notebook 24 scored. Shows what a rule
does to types the detector already handles - the check that exposed the
"both" rule overruling correct curbstone calls.

usage: python scripts/cascade_seen_types.py --scored a/scored.csv ... \
           --frame-predictions paper/results/eval_causal/frame_predictions.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cascade_gtfree as cg  # noqa: E402

KEYS = ["sequence_id", "camera", "frame_idx"]


def detector_from_frame_predictions(scored: pd.DataFrame, fp: pd.DataFrame) -> pd.DataFrame:
    """Same columns as cascade_gtfree.detector_sequences, from the legacy
    per-frame top prediction file. It has no status artifact: coverage is
    "unverified" and a frame with no row counts as no detection."""
    frames = scored.drop_duplicates(KEYS)
    top = fp.sort_values("conf").drop_duplicates(KEYS, keep="last")
    m = frames.merge(top[KEYS + ["pred_class"]], on=KEYS, how="left")
    m["cls"] = m.pred_class.fillna("none")
    cls = m.groupby("sequence_id").cls.agg(
        lambda x: "none" if len(x.mode()) > 1 else x.mode().iat[0])
    out = m.groupby("sequence_id").size().rename("requested").to_frame()
    out["ok"], out["failed"], out["missing"] = out.requested, 0, 0
    out["cls"], out["partial"], out["coverage"] = cls, False, "unverified"
    return out


def test_split(scored: pd.DataFrame) -> pd.DataFrame:
    sc = scored.copy()
    sc["uss_track"] = sc["uss_track"].fillna(True) if "uss_track" in sc else True
    sc = sc.drop_duplicates(KEYS)
    return sc[sc.split == "test"]


def seen_type_decisions(scored: pd.DataFrame, frame_predictions: pd.DataFrame,
                        rule: str, arm: str) -> pd.DataFrame:
    """Per-sequence decisions on the TEST split only, on the common population
    (detector row present and VLM ok). One row per sequence."""
    sc = test_split(scored)
    v = cg.vlm_sequences(sc)
    det = detector_from_frame_predictions(sc, frame_predictions)
    pop = det.index[(det.cls != cg.NOT_EVALUATED)].intersection(v.index[v.vlm_status == "ok"])
    d, cls = v.loc[pop], det.cls.loc[pop]
    dec = cg.cascade(d, cls, arm, rule)
    detb = cls.map(cg.BINS)
    return pd.DataFrame({"type": d.target_object, "truth": d.truth, "detector_class": cls,
                         "detector": detb == d.truth, "vlm": d[arm] == d.truth,
                         "cascade": dec == d.truth,
                         "deferred": cls.isin(cg.BUCKET | {"none"}),
                         "cascade_false_low": (dec == "low") & (d.truth == "high")})


def seen_type_summary(scored: pd.DataFrame, frame_predictions: pd.DataFrame,
                      rule: str, arm: str) -> pd.DataFrame:
    """Per type plus 'all': counts via cascade_gtfree.evaluate (common population)."""
    sc = test_split(scored)
    v = cg.vlm_sequences(sc)
    det = detector_from_frame_predictions(sc, frame_predictions)
    rows = [{"type": "all", **cg.evaluate(v, det, arm, rule)}]
    for t in sorted(v.target_object.unique()):
        ids = v.index[v.target_object == t]
        rows.append({"type": t, **cg.evaluate(v.loc[ids], det.loc[det.index.intersection(ids)],
                                              arm, rule)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", nargs="+", required=True)
    ap.add_argument("--frame-predictions", required=True)
    ap.add_argument("--rule", choices=["plain", "both"], default="both")
    a = ap.parse_args()
    sc = pd.concat([pd.read_csv(p) for p in a.scored], ignore_index=True)
    fp = pd.read_csv(a.frame_predictions)
    pd.set_option("display.width", 250)
    for arm in ("uss", "full"):
        print(f"\n== rule {a.rule}, VLM arm {arm} (coverage unverified: legacy frame predictions)")
        print(seen_type_summary(sc, fp, a.rule, arm).to_string(index=False))


if __name__ == "__main__":
    main()
