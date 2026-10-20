"""Regenerate the numbers behind EVALUATION_REPORT tables 3.3-3.6 as one CSV.

Sections (one row per cell; numerators, denominators, false-Low counts,
rates and Wilson intervals; both rules and both image arms):
  heldout   - held-out detectors on their held-out type (3.3 detector alone,
              3.5 cascade, VLM on the same population), label-free trigger
  vlm_alone - VLM alone per type and over all tall types (3.4)
  seen      - baseline detector test split (3.6), legacy coverage

usage: python scripts/report_tables.py --results paper/results --out tables.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cascade_gtfree as cg  # noqa: E402
import cascade_seen_types as cst  # noqa: E402

HELDOUT = ("woodenboard", "speedbump", "bush", "bollard")


def vlm_alone_rows(v: pd.DataFrame) -> list[dict]:
    rows = []
    groups = [(t, v[v.target_object == t]) for t in sorted(v.target_object.unique())]
    groups.append(("all_tall", v[v.truth == "high"]))
    for t, g in groups:
        g = g[g.vlm_status == "ok"]
        for arm in ("uss", "full"):
            k = int((g[arm] == g.truth).sum())
            fl = int(((g[arm] == "low") & (g.truth == "high")).sum())
            rows.append({"section": "vlm_alone", "type": t, "rule": "", "arm": arm, "n": len(g),
                         "vlm_correct": k, "vlm_rate": cg.rate(k, len(g)),
                         "vlm_ci95": cg.wilson(k, len(g)), "vlm_false_low": fl,
                         "vlm_abstain": int(g[f"{arm}_raw"].isna().sum())})
    return rows


def build(results: Path) -> pd.DataFrame:
    scored = pd.concat([pd.read_csv(results / f"prompt_decision_{s}_v1" / "scored.csv")
                        for s in ("low", "high")], ignore_index=True).drop_duplicates(cg.KEYS)
    v = cg.vlm_sequences(scored)
    rows = []
    for t in HELDOUT:
        rep = results / f"eval_hold{t}"
        st = rep / "inference_status.csv"
        det = cg.detector_sequences(scored[scored.target_object == t],
                                    pd.read_csv(rep / "raw_predictions.csv"),
                                    pd.read_csv(st) if st.exists() else None)
        for rule in ("plain", "both"):
            for arm in ("uss", "full"):
                rows.append({"section": "heldout", "type": t, "rule": rule, "arm": arm,
                             **cg.evaluate(v[v.target_object == t], det, arm, rule)})
    rows += vlm_alone_rows(v)
    seen = pd.concat([pd.read_csv(results / f"prompt_decision_{s}" / "scored.csv")
                      for s in ("v1", "low_v1", "high_v1")], ignore_index=True)
    fp = pd.read_csv(results / "eval_causal" / "frame_predictions.csv")
    for rule in ("plain", "both"):
        for arm in ("uss", "full"):
            for r in cst.seen_type_summary(seen, fp, rule, arm).to_dict("records"):
                rows.append({"section": "seen", "rule": rule, "arm": arm, **r})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("paper/results"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    t = build(a.results)
    t.to_csv(a.out, index=False)
    print(f"{len(t)} rows -> {a.out}")


if __name__ == "__main__":
    main()
