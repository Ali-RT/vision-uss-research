"""Frozen cascade evaluation (pre-registered 2026-10-20, before the
tree/cubestandard rotation was trained).

Detector side, no ground truth: per frame the NEAREST detection (largest y1,
i.e. closest on the ground plane) of a held-out model's raw_predictions.csv;
per sequence the majority over the frames that notebook 24 scored ("none"
when a frame has no detection).
VLM side: Qwen names mapped by name_to_bin, except ground-level words (curb,
grass, road, ...) which vote "no object"; the USS-crop arm falls back to the
full frame when the USS formed no track; all-abstain defaults to High (stop).
Decision:
  named class   -> its bin, but a named LOW is overruled to High when the VLM
                   says High (Low only when both agree);
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


def vote(s: pd.Series):
    s = s.dropna()
    return s.mode().iat[0] if len(s) else None


def vlm_sequences(scored: pd.DataFrame) -> pd.DataFrame:
    sc = scored[scored.error.fillna("") == ""].copy()
    sc["cb"] = [vlm_bin(n) for n in sc.crop_name]
    sc["fb"] = [vlm_bin(n) for n in sc.full_name]
    v = sc.groupby("sequence_id").agg(target_object=("target_object", "first"),
                                      truth=("truth", "first"), uss_track=("uss_track", "first"),
                                      u=("cb", vote), n=("fb", vote))
    v["uss"] = v.u.where(v.uss_track.astype(bool), v.n).fillna("high")
    v["full"] = v.n.fillna("high")
    return v.drop(columns=["u", "n"])


def detector_sequences(scored: pd.DataFrame, raw: pd.DataFrame) -> pd.Series:
    nearest = raw.sort_values("y1").drop_duplicates("image", keep="last").set_index("image").pred
    img = (scored.sequence_id + "_" + scored.camera + "_"
           + scored.frame_idx.map(lambda i: f"{int(i):05d}"))
    frames = scored.assign(cls=img.map(nearest).fillna("none"))
    return frames.groupby("sequence_id").cls.agg(lambda x: x.mode().iat[0])


def cascade(v: pd.DataFrame, cls: pd.Series, arm: str) -> pd.Series:
    named = ~cls.isin(BUCKET | {"none"})
    det = cls.map(BINS)
    dec = det.where(named, v[arm])
    return dec.mask(named & (det == "low") & (v[arm] == "high"), "high")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", nargs="+", required=True)
    ap.add_argument("--raw", nargs="+", required=True, help="type=path/raw_predictions.csv")
    a = ap.parse_args()
    scored = pd.concat([pd.read_csv(p) for p in a.scored], ignore_index=True)
    scored = scored.drop_duplicates(["sequence_id", "camera", "frame_idx"])
    v = vlm_sequences(scored)
    rows = []
    for spec in a.raw:
        t, path = spec.split("=", 1)
        sub = scored[scored.target_object == t]
        cls = detector_sequences(sub, pd.read_csv(path))
        d = v.loc[cls.index]
        det = cls.map(BINS)
        for arm in ("uss", "full"):
            dec = cascade(d, cls, arm)
            rows.append({"type": t, "arm": arm, "seqs": len(d),
                         "det_named": round((~cls.isin(BUCKET | {"none"})).mean(), 2),
                         "det_alone": round((det == d.truth).mean(), 3),
                         "cascade": round((dec == d.truth).mean(), 3),
                         "false_low": round(((dec == "low") & (d.truth == "high")).mean(), 3)})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
