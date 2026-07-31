import lzma

import pandas as pd

from vision_uss_research.alignment.uss_class import (find_cusreplay_csv,
                                                     read_classifier_monitor,
                                                     uss_sequence_decision)


def _write_monitor(path, rows):
    """rows: list of (obj_id, obj_dist, exist, class_prob_high, height_prob)."""
    header = ("CycleNr;MAP_ClassifierMonitorNormed_ObjID;"
              "MAP_ClassifierMonitorNormed_ObjDist;"
              "MAP_ClassifierMonitorNormed_ExistProb;"
              "MAP_ClassifierMonitorNormed_ClassProbHigh;"
              "MAP_ClassifierMonitorNormed_HeightProb")
    lines = [header]
    for i, (oid, dist, ex, cph, hp) in enumerate(rows):
        lines.append(f"{i};{oid};{dist};{ex};{cph};{hp}")
    with lzma.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")


def test_read_filters_to_real_readings(tmp_path):
    p = tmp_path / "seq_Classifier_Monitor_Normed.csv.xz"
    _write_monitor(p, [(0, 0, 0, 0, 0),        # no object -> dropped
                       (1, 2000, 255, 200, 150),
                       (1, 1900, 255, 210, 160)])
    df = read_classifier_monitor(p)
    assert len(df) == 2
    assert set(df["obj_id"]) == {1}
    assert df["class_prob_high"].tolist() == [200, 210]


def test_uss_decision_high_and_low():
    # dominant object (id 1) reads high (median 200/255 = 0.78 >= 0.5)
    df_high = pd.DataFrame({
        "obj_id": [1, 1, 1, 2],
        "obj_dist": [2000, 1900, 1800, 1200],
        "exist_prob": [255, 255, 255, 255],
        "class_prob_high": [190, 200, 210, 20],
        "height_prob": [150, 160, 170, 40],
    })
    d = uss_sequence_decision(df_high)
    assert d["obj_id"] == 1 and d["n_readings"] == 3
    assert d["uss_bin"] == "high"
    assert abs(d["class_prob_high"] - 200 / 255) < 1e-3

    # dominant object reads low
    df_low = pd.DataFrame({
        "obj_id": [3, 3, 3], "obj_dist": [1500, 1400, 1300],
        "exist_prob": [255, 255, 255],
        "class_prob_high": [10, 15, 12], "height_prob": [50, 60, 40]})
    assert uss_sequence_decision(df_low)["uss_bin"] == "low"


def test_uss_decision_empty():
    d = uss_sequence_decision(pd.DataFrame())
    assert d["uss_bin"] is None and d["n_readings"] == 0


def test_find_cusreplay(tmp_path):
    nested = tmp_path / "CusReplay" / "108736"
    nested.mkdir(parents=True)
    target = nested / "x_Classifier_Monitor_Normed.csv.xz"
    target.write_bytes(b"")
    assert find_cusreplay_csv(tmp_path) == target
    assert find_cusreplay_csv(tmp_path / "empty") is None


def test_read_on_real_sample():
    """Sanity on the checked-in sample sequence (bicyclestand, class high)."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "data" / "samples"
    csvs = list(root.rglob("*Classifier_Monitor_Normed.csv.xz"))
    if not csvs:
        return  # sample not present
    df = read_classifier_monitor(csvs[0])
    assert len(df) > 100
    d = uss_sequence_decision(df)
    assert d["uss_bin"] in {"low", "high"}
    assert 0.0 <= d["class_prob_high"] <= 1.0
