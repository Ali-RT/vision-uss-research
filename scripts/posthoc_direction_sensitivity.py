"""POST HOC sensitivity analysis (not part of the pre-registered Step 4 result).

The frozen frame lists took labelled frames from both cameras. A parking
manoeuvre is decided by the camera facing the driving direction
(forward -> front, backward -> rear), so this re-runs the unchanged
evaluation code on the SAME stored outputs with each sequence restricted to
its driving-direction camera:
  - direction known: keep only that camera's frames; a sequence left with no
    frame is dropped and counted;
  - direction unknown (no tag in the metadata JSON): keep all its frames.
No model is rerun; the GO decision of the protocol is unchanged.

usage (Colab, after the evaluate stage):
  python scripts/posthoc_direction_sensitivity.py --artifacts OUT --raw-root RAW --out OUT/gate/gate_v1/posthoc_dircam
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def directions(raw_root: Path, seqs: set[str], workers: int = 32) -> pd.Series:
    """Driving direction straight from each sequence's metadata JSON (the
    inventory copy on Drive has blank fields for most sequences)."""
    from concurrent.futures import ThreadPoolExecutor
    sys.path.insert(0, str(REPO / "src"))
    from vision_uss_research.sequences import driving_direction

    def one(s):
        try:
            return s, driving_direction(raw_root / s)
        except Exception:  # noqa: BLE001 - missing folder -> unknown
            return s, "unknown"
    with ThreadPoolExecutor(workers) as pool:
        out = dict(pool.map(one, sorted(seqs)))
    return pd.Series(out).str.lower().where(lambda x: x.isin(["forward", "backward"]))


def restrict(frames: pd.DataFrame, direction: pd.Series) -> tuple[pd.DataFrame, dict]:
    cam = frames.sequence_id.map(direction.map({"forward": "front", "backward": "rear"}))
    keep = cam.isna() | (frames.camera == cam)
    out = frames[keep]
    lost = set(frames.sequence_id) - set(out.sequence_id)
    return out, {"frames_before": len(frames), "frames_after": len(out),
                 "sequences_before": frames.sequence_id.nunique(), "sequences_after": out.sequence_id.nunique(),
                 "sequences_dropped_no_direction_frame": len(lost),
                 "sequences_unknown_direction": int(frames[cam.isna()].sequence_id.nunique())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--gate-run", default="gate_v1")
    ap.add_argument("--raw-root", type=Path, required=True, help="raw sequence folders (metadata JSON)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    gd = a.artifacts / "gate" / a.gate_run
    cfg = json.loads((gd / "evaluate_config.json").read_text())
    seqs = set()
    for rc in cfg["rotations"].values():
        seqs |= set(pd.read_csv(rc["frames"]).sequence_id)
    inv = directions(a.raw_root, seqs)
    known = inv.notna().mean()
    print(f"driving direction known for {inv.notna().sum()} / {len(inv)} sequences ({known:.0%})")
    if known < 0.8:
        raise SystemExit("direction unknown for > 20% of sequences - check --raw-root; nothing evaluated")
    a.out.mkdir(parents=True, exist_ok=True)
    inv.rename("driving_direction").to_csv(a.out / "driving_direction.csv")
    stats = {}
    for rot, rc in cfg["rotations"].items():
        f, stats[rot] = restrict(pd.read_csv(rc["frames"]), inv)
        p = a.out / f"frames_{rot}.csv"
        f.to_csv(p, index=False)
        rc["frames"] = str(p)
    (a.out / "evaluate_config.json").write_text(json.dumps(cfg, indent=1))
    (a.out / "restriction_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "gate_evaluate.py"), "--config",
                        str(a.out / "evaluate_config.json"), "--out", str(a.out / "evaluation")],
                       capture_output=True, text=True)
    print(r.stdout[-4000:], r.stderr[-3000:])


if __name__ == "__main__":
    main()
