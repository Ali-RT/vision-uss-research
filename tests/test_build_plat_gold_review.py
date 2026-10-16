import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "bpg", Path(__file__).resolve().parents[1] / "scripts" / "build_plat_gold_review.py")
bpg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bpg)


def test_items_carry_only_the_pseudo_label_proposal():
    df = pd.DataFrame([{"sequence_id": "110613_20190308_LB", "camera": "rear", "frame_idx": 3,
                        "target_object": "bush", "range_mm": 1500.0,
                        "x0": 10, "y0": 20, "x1": 110, "y1": 220}])
    rec = bpg.build_items(df)[0]
    assert rec["id"] == "110613_20190308_LB__rear__03"
    assert rec["pre"]["x1"] == 110 and rec["preName"] == "pseudo-label"
    assert "pre2" not in rec                        # never a Qwen/USS proposal


def test_cli_builds_shards_and_excludes(tmp_path, monkeypatch):
    root = tmp_path / "frames"
    rows = []
    for s in range(3):
        seq = f"{110000 + s}_20190308_LB"
        for f in range(4):
            p = root / seq / "front" / f"{f:05d}.jpg"
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), np.zeros((64, 96, 3), np.uint8))
            rows.append({"sequence_id": seq, "camera": "front", "frame_idx": f,
                         "target_object": "cone", "range_mm": 1000.0,
                         "x0": 1, "y0": 1, "x1": 20, "y1": 20})
    csv = tmp_path / "scored.csv"; pd.DataFrame(rows).to_csv(csv, index=False)
    excl = tmp_path / "done.csv"
    pd.DataFrame({"item_id": [bpg.item_id(rows[0]["sequence_id"], "front", 0)]}).to_csv(excl, index=False)
    monkeypatch.setattr("sys.argv", ["x", "--frames-csv", str(csv), "--frames-root", str(root),
                                     "--out", str(tmp_path / "out"), "--per-shard", "5",
                                     "--exclude", str(excl)])
    bpg.main()
    alloc = pd.read_csv(tmp_path / "out" / "plat_gold_v1" / "allocation.csv")
    assert len(alloc) == 11                          # 12 frames minus 1 excluded
    assert alloc.groupby("sequence_id").shard.nunique().max() == 1   # sequences never split
    assert (tmp_path / "out" / "plat_gold_v1" / "plat_gold_v1_shard_1.zip").exists()
