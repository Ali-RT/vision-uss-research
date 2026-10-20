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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", nargs="+", required=True)
    ap.add_argument("--frame-predictions", required=True)
    ap.add_argument("--rule", choices=["plain", "both"], default="both")
    a = ap.parse_args()
    sc = pd.concat([pd.read_csv(p) for p in a.scored], ignore_index=True)
    sc["uss_track"] = sc["uss_track"].fillna(True) if "uss_track" in sc else True
    sc = sc.drop_duplicates(KEYS)
    sc = sc[sc.split == "test"]
    v = cg.vlm_sequences(sc)
    fp = (pd.read_csv(a.frame_predictions).sort_values("conf")
          .drop_duplicates(KEYS, keep="last"))
    m = sc.merge(fp[KEYS + ["pred_class"]], on=KEYS, how="left")
    cls = (m.assign(c=m.pred_class.fillna("none")).groupby("sequence_id").c
           .agg(lambda x: "none" if len(x.mode()) > 1 else x.mode().iat[0]))
    d = v.loc[cls.index]
    det = cls.map(cg.BINS)
    for arm in ("uss", "full"):
        dec = cg.cascade(d, cls, arm, a.rule)
        r = pd.DataFrame({"type": d.target_object, "detector": det == d.truth,
                          "vlm": d[arm] == d.truth, "cascade": dec == d.truth,
                          "deferred": cls.isin(cg.BUCKET | {"none"}),
                          "cascade_false_low": (dec == "low") & (d.truth == "high")})
        out = r.groupby("type").mean(numeric_only=True).round(3)
        out.insert(0, "seqs", r.groupby("type").size())
        print(f"\n== rule {a.rule}, VLM arm {arm}")
        print(out.to_string())
        print("all:", r.mean(numeric_only=True).round(3).to_dict(), "seqs", len(r))


if __name__ == "__main__":
    main()
