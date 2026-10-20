"""Step 2 before/after: every cell of scripts/report_tables.py computed with
the pre-fix code (8adfe58) vs. the current golden table.

usage: python scripts/compare_step2_before_after.py --results paper/results \
           --out paper/results/step2_before_after.csv
It extracts the old vlm_prompt.py and cascade_gtfree.py with `git show` into
a temporary directory; no checkout is changed.
"""
import argparse
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OLD_REV = "8adfe58"
KEYS = ["sequence_id", "camera", "frame_idx"]
CELLS = ["n", "detector_correct", "detector_false_low", "vlm_correct", "vlm_false_low",
         "cascade_correct", "cascade_false_low"]


def load_old(tmp: Path):
    for rel in ("src/vision_uss_research/method/vlm_prompt.py", "scripts/cascade_gtfree.py"):
        dst = tmp / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(subprocess.run(["git", "-C", str(ROOT), "show", f"{OLD_REV}:{rel}"],
                                      capture_output=True, text=True, check=True).stdout)
    sys.path.insert(0, str(ROOT / "src"))
    import vision_uss_research.method  # noqa: F401
    sp = importlib.util.spec_from_file_location("vision_uss_research.method.vlm_prompt",
                                                tmp / "src/vision_uss_research/method/vlm_prompt.py")
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    sys.modules["vision_uss_research.method.vlm_prompt"] = m
    sp = importlib.util.spec_from_file_location("cg_old", tmp / "scripts/cascade_gtfree.py")
    cg = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(cg)
    return cg


def old_table(cg, R: Path) -> pd.DataFrame:
    scored = pd.concat([pd.read_csv(R / f"prompt_decision_{s}_v1/scored.csv") for s in ("low", "high")], ignore_index=True).drop_duplicates(KEYS)
    v = cg.vlm_sequences(scored)
    rows = []
    def cells(section, t, rule, arm, d, cls):
        detb = cls.map(cg.BINS); dec = cg.cascade(d, cls, arm, rule); hi = d.truth == "high"
        r = {"section": section, "type": t, "rule": rule, "arm": arm, "n": len(d)}
        for name, dd in (("detector", detb), ("vlm", d[arm]), ("cascade", dec)):
            r[f"{name}_correct"] = int((dd == d.truth).sum()); r[f"{name}_false_low"] = int(((dd == "low") & hi).sum())
        return r
    for t in ("woodenboard", "speedbump", "bush", "bollard"):
        cls = cg.detector_sequences(scored[scored.target_object == t], pd.read_csv(R / f"eval_hold{t}/raw_predictions.csv"))
        for rule in ("plain", "both"):
            for arm in ("uss", "full"):
                rows.append(cells("heldout", t, rule, arm, v.loc[cls.index], cls))
    groups = [(t, v[v.target_object == t]) for t in sorted(v.target_object.unique())] + [("all_tall", v[v.truth == "high"])]
    for t, g in groups:
        for arm in ("uss", "full"):
            rows.append({"section": "vlm_alone", "type": t, "rule": "", "arm": arm, "n": len(g),
                         "vlm_correct": int((g[arm] == g.truth).sum()), "vlm_false_low": int(((g[arm] == "low") & (g.truth == "high")).sum())})
    seen = pd.concat([pd.read_csv(R / f"prompt_decision_{s}/scored.csv") for s in ("v1", "low_v1", "high_v1")], ignore_index=True)
    seen["uss_track"] = seen["uss_track"].fillna(True); seen = seen.drop_duplicates(KEYS); seen = seen[seen.split == "test"]
    vs = cg.vlm_sequences(seen)
    fp = pd.read_csv(R / "eval_causal/frame_predictions.csv").sort_values("conf").drop_duplicates(KEYS, keep="last")
    mm = seen.merge(fp[KEYS + ["pred_class"]], on=KEYS, how="left")
    cls = mm.assign(c=mm.pred_class.fillna("none")).groupby("sequence_id").c.agg(lambda x: "none" if len(x.mode()) > 1 else x.mode().iat[0])
    d = vs.loc[cls.index]
    for rule in ("plain", "both"):
        for arm in ("uss", "full"):
            rows.append(cells("seen", "all", rule, arm, d, cls))
            for t in sorted(d.target_object.unique()):
                ids = d.index[d.target_object == t]; rows.append(cells("seen", t, rule, arm, d.loc[ids], cls.loc[ids]))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=ROOT / "paper/results")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        old = old_table(load_old(Path(tmp)), a.results)
    new = pd.read_csv(ROOT / "tests/golden/report_tables.csv")
    key = ["section", "type", "rule", "arm"]
    for t in (old, new):
        t["rule"] = t.rule.fillna("")
    m = old.merge(new[key + CELLS], on=key, how="outer", suffixes=("_before", "_after"))
    rows = []
    for _, r in m.iterrows():
        for c in CELLS:
            b, x = r.get(c + "_before"), r.get(c + "_after")
            if pd.isna(b) and pd.isna(x):
                continue
            rows.append({**{k: r[k] for k in key}, "metric": c, "before": b, "after": x,
                         "changed": not (pd.notna(b) and pd.notna(x) and float(b) == float(x))})
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print(f"{len(out)} cells, {int(out.changed.sum())} changed -> {a.out}")
    print(out[out.changed].to_string(index=False))


if __name__ == "__main__":
    main()
