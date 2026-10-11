import numpy as np
import pandas as pd

from vision_uss_research.cuv_finetune import (accept_sam3_segment, clean_track,
                                              cuv_manifest_rows,
                                              drop_after_exit, drop_spikes,
                                              interpolate_gaps, pick_box,
                                              plan_sam3_segments, replay_rows,
                                              spec_eval, split_traces)


def test_drop_after_exit_stops_after_top_jumps_up():
    # curb sliding out (top moving down), then asphalt false positive higher up
    boxes = np.array([[0, 300, 640, 360], [0, 440, 640, 480],
                      [0, 446, 640, 480], [0, 434, 640, 480],
                      [0, 436, 640, 480]], float)
    keep = drop_after_exit(boxes, img_h=480)
    assert keep.tolist() == [True, True, True, False, False]


def test_drop_after_exit_resets_when_object_fully_in_view():
    boxes = np.array([[0, 446, 640, 480], [0, 430, 640, 480],
                      [0, 200, 640, 300], [0, 440, 640, 480]], float)
    assert drop_after_exit(boxes, img_h=480).tolist() == [True, False, True, True]


def test_drop_spikes_removes_isolated_outlier():
    frames = np.array([10, 11, 12])
    boxes = np.array([[100, 100, 200, 200], [400, 10, 450, 40],
                      [102, 101, 202, 201]], float)
    assert drop_spikes(frames, boxes, max_gap=20).tolist() == [True, False, True]


def test_interpolate_gaps_respects_max_gap_and_overlap():
    frames = np.array([0, 4, 40])
    boxes = np.array([[0, 0, 10, 10], [4, 0, 14, 10], [4, 0, 14, 10]], float)
    filled = interpolate_gaps(frames, boxes, max_gap=10)
    assert [f for f, _ in filled] == [1, 2, 3]
    assert np.allclose(filled[1][1], [2, 0, 12, 10])


def test_clean_track_marks_sources():
    tr = clean_track(np.array([0, 3]), np.array([[0, 0, 10, 10], [3, 0, 13, 10]]),
                     "isopole", img_h=480)
    assert tr["src"].tolist() == ["det", "interp", "interp", "det"]


def test_pick_box_prefers_corridor():
    boxes = np.array([[0, 0, 60, 40], [300, 300, 340, 380]], float)
    confs = np.array([0.9, 0.5])
    assert pick_box(boxes, confs, 640, 480) == 1


def test_plan_segments_interior_tail_and_lead():
    frames = list(range(100))
    labeled = {20: np.array([1, 1, 5, 5.0]), 40: np.array([1, 1, 6, 6.0]),
               43: np.array([1, 1, 6, 6.0])}
    segs = plan_sam3_segments(frames, labeled, min_gap=5, max_tail=30,
                              backfill=True, max_lead=10)
    kinds = [(s["direction"], s["seed_frame"], s["end_frame"], len(s["frames"]))
             for s in segs]
    assert ("forward", 20, 40, 19) in kinds          # interior gap
    assert ("forward", 43, None, 30) in kinds        # tail, capped
    assert ("backward", 20, None, 10) in kinds       # lead, capped
    assert not any(k[1] == 40 and k[2] == 43 for k in kinds)  # 2-frame gap skipped


def _mask(box, h=480, w=640):
    m = np.zeros((h, w), bool)
    x0, y0, x1, y1 = box
    m[y0:y1 + 1, x0:x1 + 1] = True
    return m


def test_accept_interior_segment_needs_end_match():
    seg = {"direction": "forward", "seed_frame": 0, "seed_box": [100, 100, 200, 200],
           "frames": [1, 2], "end_frame": 3, "end_box": np.array([100, 100, 200, 200.0])}
    good = {1: _mask((100, 100, 200, 200)), 2: _mask((100, 100, 200, 200)),
            3: _mask((100, 100, 200, 200))}
    drift = {**good, 3: _mask((400, 10, 450, 40))}
    assert [f for f, _ in accept_sam3_segment(seg, good, "car", 480)] == [1, 2]
    assert accept_sam3_segment(seg, drift, "car", 480) == []


def test_accept_tail_stops_at_empty_mask_and_exit():
    seg = {"direction": "forward", "seed_frame": 0, "seed_box": [0, 300, 640, 360],
           "frames": [1, 2, 3, 4], "end_frame": None, "end_box": None}
    masks = {1: _mask((0, 440, 639, 479)), 2: _mask((0, 420, 639, 479)),
             3: _mask((0, 400, 639, 479))}
    out = accept_sam3_segment(seg, masks, "curbhigh", 480)
    assert [f for f, _ in out] == [1]                # top jumped up at frame 2
    out_tall = accept_sam3_segment(seg, masks, "car", 480)
    assert [f for f, _ in out_tall] == [1, 2, 3]     # stops at missing frame 4


def _traces():
    rows = []
    for obj, side, ids in [("box30", "rear", range(3145, 3151)),
                           ("wall", "front", [3188])]:
        for i, t in enumerate(ids):
            rows.append({"side": side, "trace": t, "label_class": obj,
                         "speed": "3kph" if i < 3 else "5kph"})
    return pd.DataFrame(rows)


def test_split_traces_holds_out_both_speeds():
    split = split_traces(_traces(), n_test=2, n_val=1, seed=0)
    box = {k: v for k, v in split.items() if k[1] != 3188}
    assert sorted(box.values()).count("test") == 2
    assert sorted(box.values()).count("val") == 1
    test_speeds = {("3kph" if t < 3148 else "5kph") for (_, t), v in box.items()
                   if v == "test"}
    assert test_speeds == {"3kph", "5kph"}
    assert split[("front", 3188)] == "train"         # too small to split


def test_cuv_manifest_rows_maps_strides_and_excludes():
    labels = pd.DataFrame({
        "side": "rear", "trace": [3145] * 6 + [3165] * 2,
        "frame_idx": list(range(6)) + [0, 1],
        "label_class": ["box30"] * 6 + ["metalbar"] * 2,
        "x0": 1.0, "y0": 2.0, "x1": 3.0, "y1": 4.0})
    rows = cuv_manifest_rows(labels, {("rear", 3145): "train", ("rear", 3165): "train"},
                             {"box30": "high_other", "metalbar": None},
                             {"high_other": 7}, stride=3)
    assert rows["frame_idx"].tolist() == [0, 3]
    assert rows["class_id"].tolist() == [7, 7]
    assert rows["frame_path"].iloc[1] == "rear_3145/00003.jpg"
    assert rows["sequence_id"].iloc[0] == "cuv_rear_3145"


def test_replay_rows_drops_test_and_samples_sequences():
    old = pd.DataFrame({"split": ["train"] * 4 + ["val"] * 2 + ["test"],
                        "sequence_id": ["a", "a", "b", "c", "d", "e", "f"]})
    out = replay_rows(old, frac=0.5, seed=0)
    assert "test" not in set(out["split"])
    assert out[out.split == "train"]["sequence_id"].nunique() == 2
    assert out[out.split == "val"]["sequence_id"].nunique() == 1


def test_spec_eval_window_and_final_approach():
    dist = np.linspace(4.0, 0.2, 39)
    df = pd.DataFrame({"side": "rear", "trace": 1, "frame_idx": range(39),
                       "label_class": "box30", "height_bin": "high",
                       "dist_m": dist,
                       "det_bin": ["high" if d >= 1.0 else None for d in dist]})
    pt, summary = spec_eval(df)
    assert bool(pt["passes"].iloc[0])                # window 0.95-1.2 m mostly hit
    assert summary["passing"].iloc[0] == 1


def _cfg():
    return {"tag": "cuv_v1", "base_dataset": "datasets/x", "bundle": "b.zip",
            "class_map": {"wall": "high_other"}, "class_ids": {"high_other": 7},
            "split": {"seed": 0}, "frame_stride": 3,
            "train_defaults": {"imgsz": 640, "optimizer": "auto", "lr0": 0.01,
                               "augment": {"mosaic": 1.0, "scale": 0.5}},
            "experiments": {"ft": {"init": "base_model", "epochs": 10,
                                   "replay_frac": 0.35, "optimizer": "SGD",
                                   "lr0": 0.001, "augment": {"scale": 0.3}}}}


def test_experiment_config_overlays_defaults():
    from vision_uss_research.cuv_finetune import experiment_config
    ec = experiment_config(_cfg(), "ft")
    assert ec["optimizer"] == "SGD" and ec["lr0"] == 0.001 and ec["imgsz"] == 640
    assert ec["augment"] == {"mosaic": 1.0, "scale": 0.3}
    assert ec["name"] == "ft"


def test_dataset_id_separates_replay_and_data_keys():
    from vision_uss_research.cuv_finetune import dataset_id
    cfg = _cfg()
    a, b = dataset_id(cfg, 0.35), dataset_id(cfg, 1.0)
    assert a.startswith("dataset_cuv_v1_r035_") and b.startswith("dataset_cuv_v1_r100_")
    assert a == dataset_id(_cfg(), 0.35)                  # stable
    cfg["frame_stride"] = 5
    assert dataset_id(cfg, 0.35) != a                     # any data key changes it


def test_claim_run_dir_new_same_conflict(tmp_path):
    import json
    import pytest
    from vision_uss_research.cuv_finetune import claim_run_dir
    d = tmp_path / "run"
    assert claim_run_dir(d, {"a": 1}) == "new"
    d.mkdir()
    (d / "run.json").write_text(json.dumps({"config": {"a": 1}}))
    assert claim_run_dir(d, {"a": 1}) == "same"
    with pytest.raises(ValueError):
        claim_run_dir(d, {"a": 2})


def test_training_state(tmp_path):
    from vision_uss_research.cuv_finetune import training_state
    w = tmp_path / "weights"
    assert training_state(tmp_path) == "fresh"
    w.mkdir()
    (w / "last.pt").write_text("x")
    assert training_state(tmp_path) == "resume"
    (w / "best.pt").write_text("x")
    (tmp_path / "train_done.json").write_text("{}")
    assert training_state(tmp_path) == "done"


def test_best_epoch_uses_fitness(tmp_path):
    from vision_uss_research.cuv_finetune import best_epoch
    p = tmp_path / "results.csv"
    pd.DataFrame({"epoch": [1, 2, 3], "metrics/mAP50(B)": [0.9, 0.95, 0.8],
                  "metrics/mAP50-95(B)": [0.7, 0.6, 0.72]}).to_csv(p, index=False)
    assert best_epoch(p) == 3
    assert best_epoch(tmp_path / "missing.csv") is None


def test_spec_eval_localized_rejects_boxes_off_the_object():
    dist = np.linspace(2.0, 1.0, 11)
    base = {"side": "rear", "trace": 1, "frame_idx": range(11),
            "label_class": "box30", "height_bin": "high", "dist_m": dist,
            "det_bin": "high", "lx0": 100.0, "ly0": 100.0, "lx1": 200.0, "ly1": 200.0}
    on = pd.DataFrame({**base, "px0": 105.0, "py0": 100.0, "px1": 205.0, "py1": 200.0})
    off = pd.DataFrame({**base, "px0": 400.0, "py0": 10.0, "px1": 500.0, "py1": 60.0})
    pt_on, _ = spec_eval(on)
    pt_off, _ = spec_eval(off)
    assert bool(pt_on["passes_loc"].iloc[0]) and bool(pt_off["passes"].iloc[0])
    assert not bool(pt_off["passes_loc"].iloc[0])     # right bin, wrong place
    assert pt_off["window_loc_rate"].iloc[0] == 0.0


def test_spec_eval_localized_is_unknown_without_labels():
    dist = np.linspace(2.0, 1.0, 11)
    df = pd.DataFrame({"side": "rear", "trace": 1, "frame_idx": range(11),
                       "label_class": "box30", "height_bin": "high", "dist_m": dist,
                       "det_bin": "high", "lx0": np.nan, "ly0": np.nan,
                       "lx1": np.nan, "ly1": np.nan,
                       "px0": 1.0, "py0": 1.0, "px1": 9.0, "py1": 9.0})
    pt, _ = spec_eval(df)
    assert pt["passes_loc"].iloc[0] is None and bool(pt["passes"].iloc[0])
