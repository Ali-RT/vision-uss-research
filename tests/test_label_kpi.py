import numpy as np
import pandas as pd

from vision_uss_research.label_kpi import (auto_accept, calibration_table, fit_logistic,
                                           kpi_table, predict_logistic, resolve_gold,
                                           score_labels)


def _rev(item, frame, status, box=(np.nan,) * 4, reviewer="a"):
    return {"item_id": item, "side": "rear", "trace": 1, "frame_idx": frame,
            "category": "box30", "reviewer": reviewer, "status": status,
            "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3], "truncated": 0}


def test_resolve_gold_single_agree_and_disputed():
    reviews = pd.DataFrame([
        _rev("i1", 1, "box", (0, 0, 100, 100)),
        _rev("i2", 2, "box", (0, 0, 100, 100)), _rev("i2", 2, "box", (4, 0, 104, 100), "b"),
        _rev("i3", 3, "box", (0, 0, 100, 100)), _rev("i3", 3, "not_visible", reviewer="b"),
        _rev("i4", 4, "unsure"),
    ])
    gold, agree = resolve_gold(reviews)
    g = gold.set_index("item_id")
    assert g.loc["i1", "status"] == "box" and g.loc["i2", "x0"] == 2.0
    assert g.loc["i3", "status"] == "disputed" and g.loc["i4", "status"] == "unsure"
    assert agree.set_index("item_id")["agree"].to_dict() == {"i2": True, "i3": False}


def test_score_and_kpis():
    gold = pd.DataFrame([
        {"side": "rear", "trace": 1, "frame_idx": f, "category": "box30", "status": s,
         "x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3]}
        for f, s, b in [(1, "box", (0, 0, 100, 100)), (2, "box", (0, 0, 100, 100)),
                        (3, "box", (0, 0, 100, 100)), (4, "not_visible", (np.nan,) * 4)]])
    labels = pd.DataFrame([
        {"side": "rear", "trace": 1, "frame_idx": 1, "x0": 0, "y0": 0, "x1": 100, "y1": 100},
        {"side": "rear", "trace": 1, "frame_idx": 2, "x0": 300, "y0": 300, "x1": 400, "y1": 400},
        {"side": "rear", "trace": 1, "frame_idx": 4, "x0": 0, "y0": 0, "x1": 10, "y1": 10}])
    s = score_labels(gold, labels)
    k = kpi_table(s, ["category"]).iloc[0]
    assert k["visible"] == 3 and k["labeled"] == 3
    assert abs(k["recall"] - 1 / 3) < 1e-3 and abs(k["precision"] - 1 / 3) < 1e-3
    assert k["false_on_not_visible"] == 1


def test_logistic_and_calibration_and_accept():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 1))
    y = (rng.random(2000) < 1 / (1 + np.exp(-2 * x[:, 0]))).astype(float)
    p = predict_logistic(fit_logistic(x, y, l2=0.1), x)
    _, ece = calibration_table(p, y)
    assert ece < 0.05
    acc = auto_accept(np.array([0.99, 0.98, 0.9, 0.2]), np.array([1, 1, 1, 0.0]), 0.98)
    assert acc["accepted_share"] == 0.75 and acc["threshold"] == 0.9


def _scored():
    rows = []
    # trace 1: 4 visible frames, 3 correct, one cut by the edge (wrong)
    for f, c, tr in [(10, True, 0), (20, True, 0), (30, True, 0), (40, False, 1)]:
        rows.append({"side": "rear", "trace": 1, "frame_idx": f, "category": "box30",
                     "visible": True, "correct": c, "wrong": not c, "truncated": tr, "track_id": 1})
    # trace 2: 2 visible, none correct; wrong labels in two separate runs
    for f, w in [(10, True), (20, False), (30, True)]:
        rows.append({"side": "rear", "trace": 2, "frame_idx": f, "category": "box30",
                     "visible": f != 20, "correct": False, "wrong": w, "truncated": 0,
                     "track_id": 7 if f == 10 else 8})
    return pd.DataFrame(rows)


def test_tpr_per_sequence_excludes_cut_objects_and_averages_traces():
    from vision_uss_research.label_kpi import tpr_per_sequence
    t = tpr_per_sequence(_scored(), ["category"]).iloc[0]
    assert t["tpr_mean"] == 0.5 and t["worst_trace"] == 0.0 and t["traces"] == 2


def test_object_detection_rate():
    from vision_uss_research.label_kpi import object_detection_rate
    assert object_detection_rate(_scored(), ["category"]).iloc[0]["detected"] == 0.5


def test_false_events_per_hour():
    from vision_uss_research.label_kpi import false_events_per_hour
    dur = pd.Series({("rear", 1): 0.5, ("rear", 2): 0.5})
    r = false_events_per_hour(_scored(), dur, ["category"]).iloc[0]
    assert r["false_runs"] == 3 and r["false_runs_per_h"] == 3.0
    assert r["false_track_ids"] == 3


def test_audit_silver():
    from vision_uss_research.label_kpi import audit_silver
    silver = pd.DataFrame([
        {"item_id": "a", "category": "box30", "status": "box", "x0": 0, "y0": 0, "x1": 100, "y1": 100},
        {"item_id": "b", "category": "box30", "status": "box", "x0": 0, "y0": 0, "x1": 100, "y1": 100},
        {"item_id": "c", "category": "box30", "status": "not_visible", "x0": np.nan, "y0": np.nan, "x1": np.nan, "y1": np.nan}])
    audit = pd.DataFrame([
        {"item_id": "a", "status": "box", "x0": 2, "y0": 0, "x1": 102, "y1": 100},
        {"item_id": "b", "status": "box", "x0": 300, "y0": 300, "x1": 400, "y1": 400},
        {"item_id": "c", "status": "not_visible", "x0": np.nan, "y0": np.nan, "x1": np.nan, "y1": np.nan}])
    r = audit_silver(silver, audit)
    assert r.loc["all", "audited"] == 3 and abs(r.loc["all", "confirmed"] - 2 / 3) < 1e-3
