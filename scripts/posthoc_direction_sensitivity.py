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
  python scripts/posthoc_direction_sensitivity.py --artifacts OUT --out OUT/gate/gate_v1/posthoc_dircam
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def find_inventory(artifacts: Path) -> Path:
    found = sorted(list(artifacts.glob("inventory/*/inventory_metadata.csv"))
                   + list(artifacts.glob("profiles/*/inventory_metadata.csv")))
    if not found:
        raise SystemExit(f"inventory_metadata.csv not found under {artifacts}")
    return found[-1]


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
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    gd = a.artifacts / "gate" / a.gate_run
    cfg = json.loads((gd / "evaluate_config.json").read_text())
    inv = pd.read_csv(find_inventory(a.artifacts)).set_index("sequence_id").driving_direction
    a.out.mkdir(parents=True, exist_ok=True)
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
