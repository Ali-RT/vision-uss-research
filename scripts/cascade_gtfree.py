"""GT-free cascade evaluation. STATUS 2026-10-20: candidate rules, NOT frozen.
The "both" rule below was designed after seeing the bush rotation and, on
the baseline model's seen types, overrules correct curbstone/woodenboard
calls (the VLM names nearby grass/bushes). Fix the rule before looking at
the tree/cubestandard rotation's results.

Detector side, no ground truth: per frame the NEAREST detection (largest y1,
i.e. closest on the ground plane) of a held-out model's raw_predictions.csv;
per sequence the majority over the frames that notebook 24 scored ("none"
when a frame has no detection; a tied majority also counts as "none").
VLM side: Qwen names mapped by name_to_bin, except ground-level words (curb,
grass, road, ...) which vote "no object"; the USS-crop arm falls back to the
full frame when the USS formed no track; all-abstain defaults to High (stop).
Decision (--rule plain): named class -> its bin; bucket / none -> the VLM.
Decision (--rule both):
  named class   -> its bin, but a named LOW is overruled to High when the VLM
                   positively names a High object (a VLM abstention does not
                   overrule - otherwise every curb the detector names
                   correctly would be overruled, since the VLM's "curb" is a
                   ground word);
  bucket / none -> the VLM.

usage: python scripts/cascade_gtfree.py --scored a/scored.csv b/scored.csv \
           --raw woodenboard=reports/eval_holdwoodenboard/raw_predictions.csv ...
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vision_uss_research.method.vlm_prompt import name_to_bin  # noqa: E402

BINS = {"curbstone": "low", "speedbump": "low", "woodenboard": "low", "low_other": "low",
        "pole": "high", "dummychild": "high", "bicyclestand": "high", "high_other": "high"}
BUCKET = {"low_other", "high_other"}
GROUND = re.compile(r"\b(curb\w*|kerb\w*|grass\w*|ground|road|pavement|asphalt|"
                    r"parking (space|spot|lot)|shadow|line|lines|sidewalk|street)\b")


def vlm_bin(name) -> str | None:
    if not isinstance(name, str) or GROUND.search(name.lower()):
        return None
    return name_to_bin(name)


def wilson(k: int, n: int, z: float = 1.96) -> str:
    """95% Wilson interval for k/n, as 'lo-hi' in percent."""
    if n == 0:
        return ""
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return f"{100 * max(c - h, 0):.1f}-{100 * min(c + h, 1):.1f}"


def vote(s: pd.Series):
    """Majority over a sequence's frames; a tie or no votes is an abstention (None)."""
    m = s.dropna().mode()
    return m.iat[0] if len(m) == 1 else None


def vlm_sequences(scored: pd.DataFrame) -> pd.DataFrame:
    sc = scored[scored.error.fillna("") == ""].copy()
    sc["cb"] = [vlm_bin(n) for n in sc.crop_name]
    sc["fb"] = [vlm_bin(n) for n in sc.full_name]
    v = sc.groupby("sequence_id").agg(target_object=("target_object", "first"),
                                      truth=("truth", "first"), uss_track=("uss_track", "first"),
                                      u=("cb", vote), n=("fb", vote))
    v["uss_raw"] = v.u.where(v.uss_track.astype(bool), v.n)   # None = abstained
    v["full_raw"] = v.n
    v["uss"] = v.uss_raw.fillna("high")                        # abstain -> stop
    v["full"] = v.full_raw.fillna("high")
    return v.drop(columns=["u", "n"])


NOT_EVALUATED = "not_evaluated"


def detector_sequences(scored: pd.DataFrame, raw: pd.DataFrame,
                       evaluated: set[str] | None = None) -> pd.Series:
    """Per-sequence detector answer from raw detections.

    `evaluated` is the set of image stems the detector actually ran on
    (notebook 07's inference_status.csv). With it, a frame missing from raw
    detections is "none" only if it was evaluated; frames never evaluated are
    dropped, and a sequence with no evaluated frame is NOT_EVALUATED. Without
    it (runs before 2026-10-21), every missing frame counts as "none" - the
    caller must report that evaluation coverage is unverified."""
    nearest = raw.sort_values("y1").drop_duplicates("image", keep="last").set_index("image").pred
    img = (scored.sequence_id + "_" + scored.camera + "_"
           + scored.frame_idx.map(lambda i: f"{int(i):05d}"))
    frames = scored.assign(cls=img.map(nearest).fillna("none"))
    if evaluated is not None:
        frames = frames[img.isin(evaluated) | img.isin(set(nearest.index))]
    out = frames.groupby("sequence_id").cls.agg(
        lambda x: "none" if len(x.mode()) > 1 else x.mode().iat[0])
    return out.reindex(scored.sequence_id.unique()).fillna(NOT_EVALUATED)


def cascade(v: pd.DataFrame, cls: pd.Series, arm: str, rule: str = "both") -> pd.Series:
    named = ~cls.isin(BUCKET | {"none"})
    det = cls.map(BINS)
    dec = det.where(named, v[arm])
    if rule == "plain":
        return dec
    return dec.mask(named & (det == "low") & (v[f"{arm}_raw"] == "high"), "high")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", nargs="+", required=True)
    ap.add_argument("--raw", nargs="+", required=True, help="type=path/raw_predictions.csv")
    ap.add_argument("--rule", choices=["plain", "both"], default="both")
    ap.add_argument("--status", nargs="*", default=[],
                    help="type=path/inference_status.csv (notebook 07, from 2026-10-21)")
    a = ap.parse_args()
    scored = pd.concat([pd.read_csv(p) for p in a.scored], ignore_index=True)
    scored = scored.drop_duplicates(["sequence_id", "camera", "frame_idx"])
    v = vlm_sequences(scored)
    status = {}
    for spec in a.status:
        t, path = spec.split("=", 1)
        st = pd.read_csv(path)
        status[t] = set(st.loc[st.status == "ok", "image"])
    rows = []
    for spec in a.raw:
        t, path = spec.split("=", 1)
        sub = scored[scored.target_object == t]
        cls = detector_sequences(sub, pd.read_csv(path), status.get(t))
        n_not_eval = int((cls == NOT_EVALUATED).sum())
        cls = cls[cls != NOT_EVALUATED]
        d = v.loc[cls.index]
        det = cls.map(BINS)
        for arm in ("uss", "full"):
            dec = cascade(d, cls, arm, a.rule)
            ok = int((dec == d.truth).sum())
            fl = int(((dec == "low") & (d.truth == "high")).sum())
            rows.append({"type": t, "arm": arm, "seqs": len(d),
                         "coverage": "verified" if t in status else "unverified",
                         "not_evaluated": n_not_eval,
                         "det_named": round((~cls.isin(BUCKET | {"none"})).mean(), 2),
                         "det_alone": round((det == d.truth).mean(), 3),
                         "cascade": round(ok / len(d), 3), "cascade_ci95": wilson(ok, len(d)),
                         "false_low": f"{fl}/{len(d)}", "false_low_ci95": wilson(fl, len(d))})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
