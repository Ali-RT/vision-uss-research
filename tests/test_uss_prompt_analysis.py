import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vision_uss_research.method.uss_prompt import GroundCamera

spec = importlib.util.spec_from_file_location(
    "upa", Path(__file__).resolve().parents[1] / "scripts" / "uss_prompt_analysis.py")
upa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upa)


def _world(tmp_path, n_seq=60, seed=0):
    """Synthetic corpus: a known rear camera, one target slot per frame plus a
    distractor slot outside the corridor, boxes drawn around the projected
    target. The analysis must recover near-perfect hits."""
    rng = np.random.default_rng(seed)
    cam = GroundCamera.initial("rear")
    cam.f, cam.h, cam.pitch = 560.0, 700.0, np.radians(12)
    slots, man = [], []
    for s in range(n_seq):
        seq = f"{100000 + s}_{20190301 + s % 12:08d}_LB_XO2617_{s:03d}"
        y0 = rng.uniform(-800, 800)
        for i in range(20):
            rng_mm = 4500 - i * 200
            x = cam.xc - 200 - rng_mm
            u, v, _ = cam.project(x, y0)
            man.append({"split": "test" if s % 5 == 0 else "train", "sequence_id": seq,
                        "camera": "rear", "frame_idx": i, "class_id": 0,
                        "x0": u - 30, "x1": u + 30, "y0": v - 80, "y1": v + 2})
            slots.append({"sequence_id": seq, "camera": "rear", "frame_idx": i,
                          "mid_x": x, "mid_y": y0})
            slots.append({"sequence_id": seq, "camera": "rear", "frame_idx": i,
                          "mid_x": x - 800, "mid_y": 2500.0})          # outside corridor
    md = tmp_path / "man"; md.mkdir()
    pd.DataFrame(man).to_csv(md / "dataset_manifest.csv", index=False)
    (md / "provenance.json").write_text(json.dumps({"class_names": ["pole"]}))
    rd = tmp_path / "run"; rd.mkdir()
    pd.DataFrame(slots).to_csv(rd / "slots_at_frames.csv", index=False)
    return rd, md


def test_analysis_recovers_synthetic_camera(tmp_path):
    rd, md = _world(tmp_path)
    slots, boxes = upa.load(rd, md)
    summary, scored = upa.run(slots, boxes, k=4)
    o = summary["overall"]
    assert o["coverage"] > 0.99
    assert o["hit_given_covered"] > 0.95
    assert o["oracle_hit_given_covered"] >= o["hit_given_covered"]
    assert o["median_err_v_px"] < 5


def test_detector_missed_block(tmp_path):
    rd, md = _world(tmp_path)
    slots, boxes = upa.load(rd, md)
    test_seqs = sorted(boxes[boxes.split == "test"].sequence_id.unique())
    fp = pd.DataFrame({"sequence_id": test_seqs[:-3]})        # 3 test seqs never detected
    summary, _ = upa.run(slots, boxes, k=4, frame_preds=fp)
    assert summary["detector_missed_sequences"]["sequences"] == 3
