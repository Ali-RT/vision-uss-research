"""Step 4 (M2): the fixed tier C frame manifest (STEP4_PROPOSAL §5 tier C).

Every TEST sequence of the tier C rotation (held-out tree/cubestandard/hose
and its known types): up to 5 labelled frames (unique sequence/camera/frame
keys of the manifest), drawn with seed 0. No box-size filter, no prediction
of any model is read. The output is hashed at M2 and is the only frame list
used for tier C in M3.

usage: python scripts/tierc_frames.py \\
           --manifest .../dataset_v2holdtreecubehose/dataset_manifest.csv \\
           --heldout tree,cubestandard,hose --out paper/results/tierc_frames_v1.csv
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

KEYS = ["sequence_id", "camera", "frame_idx"]
FRAMES_PER_SEQ, SEED = 5, 0


def tier_c_frames(manifest: pd.DataFrame, heldout: set[str]) -> pd.DataFrame:
    m = manifest[manifest.split == "test"]
    targets = m.drop_duplicates("sequence_id").set_index("sequence_id").target_object
    frames = m.drop_duplicates(KEYS)[KEYS].sort_values(KEYS).reset_index(drop=True)
    picked = (frames.sample(frac=1, random_state=SEED)
              .groupby("sequence_id").head(FRAMES_PER_SEQ)
              .sort_values(KEYS).reset_index(drop=True))
    picked["target_object"] = picked.sequence_id.map(targets)
    picked["population"] = picked.target_object.map(lambda t: "heldout" if t in heldout else "known")
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    held = set(a.heldout.split(","))
    out = tier_c_frames(pd.read_csv(a.manifest), held)
    out.to_csv(a.out, index=False)
    h = hashlib.sha256(a.out.read_bytes()).hexdigest()
    s = out.drop_duplicates("sequence_id").groupby(["population", "target_object"]).size()
    print(s.to_string())
    print(f"{out.sequence_id.nunique()} sequences, {len(out)} frames -> {a.out}\nsha256 {h}")


if __name__ == "__main__":
    main()
