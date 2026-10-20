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

Coverage and populations (step 2 revision, 2026-10-21):
  - Detector: with notebook 07's inference_status.csv the status is
    authoritative. Each requested frame is ok / failed / missing; only ok
    frames vote; a raw detection on an image without an ok status is a
    contradiction and raises. A sequence with no ok frame is not_evaluated; a
    sequence with some but not all frames ok is decided from its ok frames
    and counted as partial. Without a status file coverage is "unverified"
    and every requested frame is treated as evaluated (legacy runs).
  - VLM: frames with an error are dropped; a sequence whose frames all
    errored is vlm_failed (no decision - never a default High).
  - Every comparison uses one common population: sequences that are both
    detector-evaluated and VLM-ok. Exclusions are counted, never silent.
    Rates on an empty population are None.

usage: python scripts/cascade_gtfree.py --scored a/scored.csv b/scored.csv \
           --raw woodenboard=reports/eval_holdwoodenboard/raw_predictions.csv ... \
           [--status woodenboard=reports/eval_holdwoodenboard/inference_status.csv ...]
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


KEYS = ["sequence_id", "camera", "frame_idx"]
NOT_EVALUATED = "not_evaluated"


def vlm_bin(name) -> str | None:
    if not isinstance(name, str) or GROUND.search(name.lower()):
        return None
    return name_to_bin(name)


def wilson(k: int, n: int, z: float = 1.96) -> str:
    """95% Wilson interval for k/n, as 'lo-hi' in percent ('' when n == 0)."""
    if n == 0:
        return ""
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return f"{100 * max(c - h, 0):.1f}-{100 * min(c + h, 1):.1f}"


def rate(k: int, n: int) -> float | None:
    return round(k / n, 3) if n else None


def vote(s: pd.Series):
    """Majority over a sequence's frames; a tie or no votes is an abstention (None)."""
    m = s.dropna().mode()
    return m.iat[0] if len(m) == 1 else None


def image_key(frames: pd.DataFrame) -> pd.Series:
    return (frames.sequence_id + "_" + frames.camera + "_"
            + frames.frame_idx.map(lambda i: f"{int(i):05d}"))


def vlm_sequences(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per sequence in `scored` (including sequences whose VLM frames
    all errored: vlm_status 'vlm_failed', no decision)."""
    sc = scored.copy()
    sc["err"] = sc.error.fillna("") != ""
    ok = sc[~sc.err].copy()
    ok["cb"] = [vlm_bin(n) for n in ok.crop_name]
    ok["fb"] = [vlm_bin(n) for n in ok.full_name]
    base = sc.groupby("sequence_id").agg(target_object=("target_object", "first"),
                                         truth=("truth", "first"),
                                         uss_track=("uss_track", "first"),
                                         vlm_frames=("err", "size"),
                                         vlm_error_frames=("err", "sum"))
    votes = ok.groupby("sequence_id").agg(u=("cb", vote), n=("fb", vote))
    v = base.join(votes)
    v["vlm_status"] = (v.vlm_error_frames < v.vlm_frames).map({True: "ok", False: "vlm_failed"})
    v["uss_raw"] = v.u.where(v.uss_track.astype(bool), v.n)     # None = abstained
    v["full_raw"] = v.n
    good = v.vlm_status == "ok"
    v["uss"] = v.uss_raw.where(~good | v.uss_raw.notna(), "high")   # abstain -> stop
    v["full"] = v.full_raw.where(~good | v.full_raw.notna(), "high")
    v.loc[~good, ["uss", "full"]] = None
    return v.drop(columns=["u", "n"])


def _validate_status(status: pd.DataFrame, raw: pd.DataFrame) -> pd.Series:
    if status.image.duplicated().any():
        raise ValueError(f"duplicate status rows: {status.image[status.image.duplicated()].head(3).tolist()}")
    bad = set(status.status) - {"ok", "failed"}
    if bad:
        raise ValueError(f"unknown status values {bad}")
    st = status.set_index("image").status
    contra = set(raw.image) - set(st[st == "ok"].index)
    if contra:
        raise ValueError(f"{len(contra)} images have raw detections but no ok status, "
                         f"e.g. {sorted(contra)[:3]}")
    return st


def detector_sequences(scored: pd.DataFrame, raw: pd.DataFrame,
                       status: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-sequence detector answer from raw detections, with coverage counts.

    Columns: cls (class name, "none", or NOT_EVALUATED), requested, ok, failed,
    missing (frames), partial (some but not all requested frames ok), coverage
    ("verified" with a status file, else "unverified")."""
    nearest = raw.sort_values("y1").drop_duplicates("image", keep="last").set_index("image").pred
    frames = scored.drop_duplicates(KEYS).copy()
    frames["image"] = image_key(frames)
    if status is not None:
        st = _validate_status(status, raw)
        frames["fstatus"] = frames.image.map(st).fillna("missing")
        coverage = "verified"
    else:
        frames["fstatus"] = "ok"
        coverage = "unverified"
    frames["cls"] = frames.image.map(nearest).fillna("none")
    counts = frames.groupby("sequence_id").fstatus.agg(
        requested="size", ok=lambda x: (x == "ok").sum(),
        failed=lambda x: (x == "failed").sum(), missing=lambda x: (x == "missing").sum())
    good = frames[frames.fstatus == "ok"]
    cls = good.groupby("sequence_id").cls.agg(
        lambda x: "none" if len(x.mode()) > 1 else x.mode().iat[0])
    out = counts.join(cls.rename("cls"))
    out["cls"] = out.cls.fillna(NOT_EVALUATED)
    out["partial"] = (out.ok > 0) & (out.ok < out.requested)
    out["coverage"] = coverage
    return out


def cascade(v: pd.DataFrame, cls: pd.Series, arm: str, rule: str = "both") -> pd.Series:
    named = ~cls.isin(BUCKET | {"none"})
    det = cls.map(BINS)
    dec = det.where(named, v[arm])
    if rule == "plain":
        return dec
    return dec.mask(named & (det == "low") & (v[f"{arm}_raw"] == "high"), "high")


def evaluate(v: pd.DataFrame, det: pd.DataFrame, arm: str, rule: str) -> dict:
    """Counts on the common population (detector-evaluated and VLM-ok)."""
    req = det.index.union(v.index)
    d_ok = det.cls.reindex(req).fillna(NOT_EVALUATED) != NOT_EVALUATED
    v_ok = v.vlm_status.reindex(req).fillna("vlm_failed") == "ok"
    pop = req[d_ok & v_ok]
    d, cls = v.loc[pop], det.cls.loc[pop]
    dec = cascade(d, cls, arm, rule)
    detb = cls.map(BINS)
    high = d.truth == "high"
    n = len(pop)
    out = {"requested_seqs": len(req), "not_evaluated": int((~d_ok).sum()),
           "vlm_failed": int((d_ok & ~v_ok).sum()),
           "partial_seqs": int(det.partial.reindex(pop).sum()),
           "frames_requested": int(det.requested.sum()), "frames_ok": int(det.ok.sum()),
           "frames_failed": int(det.failed.sum()), "frames_missing": int(det.missing.sum()),
           "coverage": det.coverage.iat[0] if len(det) else "", "n": n,
           "det_named": int((~cls.isin(BUCKET | {"none"})).sum())}
    for name, dd in (("detector", detb), ("vlm", d[arm]), ("cascade", dec)):
        k = int((dd == d.truth).sum())
        fl = int(((dd == "low") & high).sum())
        out.update({f"{name}_correct": k, f"{name}_rate": rate(k, n),
                    f"{name}_ci95": wilson(k, n), f"{name}_false_low": fl})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", nargs="+", required=True)
    ap.add_argument("--raw", nargs="+", required=True, help="type=path/raw_predictions.csv")
    ap.add_argument("--rule", choices=["plain", "both"], default="both")
    ap.add_argument("--status", nargs="*", default=[],
                    help="type=path/inference_status.csv (notebook 07, from 2026-10-21)")
    a = ap.parse_args()
    scored = pd.concat([pd.read_csv(p) for p in a.scored], ignore_index=True)
    scored = scored.drop_duplicates(KEYS)
    v = vlm_sequences(scored)
    status = dict(spec.split("=", 1) for spec in a.status)
    rows = []
    for spec in a.raw:
        t, path = spec.split("=", 1)
        st = pd.read_csv(status[t]) if t in status else None
        det = detector_sequences(scored[scored.target_object == t], pd.read_csv(path), st)
        vt = v[v.target_object == t]
        for arm in ("uss", "full"):
            rows.append({"type": t, "arm": arm, "rule": a.rule, **evaluate(vt, det, arm, a.rule)})
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
