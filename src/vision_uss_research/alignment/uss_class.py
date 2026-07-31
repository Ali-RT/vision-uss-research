"""The USS stack's own Low/High decision, from the CusReplay classifier monitor.

Each sequence has a CusReplay resim log
(`*_Classifier_Monitor_Normed.csv.xz`) with, per processing cycle and per
tracked object, a `ClassProbHigh` and `HeightProb` in [0, 255] (a normalized
probability). This is the production USS height classification - the baseline
the camera detector is compared against in the fusion experiment.

Pure pandas + lzma; no MF4/asammdf. Unit-testable.
"""

from __future__ import annotations

import lzma
from pathlib import Path

import pandas as pd

PROB_SCALE = 255.0          # ClassProbHigh / HeightProb are 0..255
HIGH_THRESHOLD = 0.5        # >= 0.5 normalized -> the USS calls it "high"

_COLS = {
    "obj_id": "MAP_ClassifierMonitorNormed_ObjID",
    "obj_dist": "MAP_ClassifierMonitorNormed_ObjDist",
    "exist_prob": "MAP_ClassifierMonitorNormed_ExistProb",
    "class_prob_high": "MAP_ClassifierMonitorNormed_ClassProbHigh",
    "height_prob": "MAP_ClassifierMonitorNormed_HeightProb",
}


def read_classifier_monitor(path: Path) -> pd.DataFrame:
    """Read a `*_Classifier_Monitor_Normed.csv.xz` into a tidy frame with
    columns obj_id, obj_dist, exist_prob, class_prob_high, height_prob (raw
    0..255), keeping only rows with a real object reading (obj_dist > 0).
    Returns an empty frame on any read error (never raises)."""
    try:
        with lzma.open(path, "rt", errors="replace") as f:
            df = pd.read_csv(f, sep=";", low_memory=False)
    except Exception:
        return pd.DataFrame(columns=list(_COLS))
    out = pd.DataFrame()
    for short, full in _COLS.items():
        out[short] = pd.to_numeric(df.get(full), errors="coerce") if full in df else pd.NA
    out = out.dropna(subset=["obj_dist"])
    return out[out["obj_dist"] > 0].reset_index(drop=True)


def uss_sequence_decision(df: pd.DataFrame,
                          high_threshold: float = HIGH_THRESHOLD) -> dict:
    """Per-sequence USS Low/High call from the classifier monitor.

    The staged target is the DOMINANT tracked object (most cycles). Its median
    normalized ClassProbHigh, thresholded, is the USS decision. Returns
    class_prob_high / height_prob in [0, 1], the chosen obj_id, and reading
    count; all None when there are no readings."""
    if df is None or len(df) == 0:
        return {"obj_id": None, "n_readings": 0, "class_prob_high": None,
                "height_prob": None, "uss_bin": None}
    counts = df["obj_id"].value_counts()
    target = counts.index[0]                       # most-tracked object
    g = df[df["obj_id"] == target]
    cph = float(g["class_prob_high"].median()) / PROB_SCALE
    hp = (float(g["height_prob"].median()) / PROB_SCALE
          if g["height_prob"].notna().any() else None)
    return {
        "obj_id": (int(target) if pd.notna(target) else None),
        "n_readings": int(len(g)),
        "class_prob_high": round(cph, 4),
        "height_prob": round(hp, 4) if hp is not None else None,
        "uss_bin": "high" if cph >= high_threshold else "low",
    }


def find_cusreplay_csv(sequence_dir: Path) -> Path | None:
    """Locate the classifier-monitor log inside a sequence's CusReplay tree."""
    matches = sorted(Path(sequence_dir).rglob("*Classifier_Monitor_Normed.csv.xz"))
    return matches[0] if matches else None
