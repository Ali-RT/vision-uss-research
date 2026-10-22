"""Rules of the Step 4 open-set gate (STEP4_PROPOSAL.md revision 2)."""
import numpy as np
import pandas as pd
import pytest

from vision_uss_research.method import gate as g

CIDS = {"curbstone": 0, "speedbump": 1, "woodenboard": 2, "pole": 3, "low_other": 6, "high_other": 7}
BUCKETS = {"low_other", "high_other"}


def test_select_detection_ties():
    d = pd.DataFrame({"pred": ["pole", "curbstone", "speedbump"], "conf": [0.5, 0.5, 0.9],
                      "y1": [300.0, 300.0, 100.0]})
    assert g.select_detection(d, CIDS).pred == "curbstone"     # y1 tie, conf tie -> lower id
    d.loc[0, "conf"] = 0.6
    assert g.select_detection(d, CIDS).pred == "pole"          # y1 tie -> higher conf
    assert g.select_detection(d.iloc[:0], CIDS) is None


def test_pad_clip_and_invalid():
    assert g.pad_clip_box(100, 100, 200, 150, 960, 640) == (90, 95, 210, 155)
    assert g.pad_clip_box(0, 0, 50, 50, 40, 40) == (0, 0, 40, 40)       # clipped
    assert g.pad_clip_box(10, 10, 15, 30, 960, 640) is None              # width 6 < 8


def test_crop_tensor_shape_and_normalization():
    from PIL import Image
    img = Image.new("RGB", (100, 80), (255, 255, 255))
    a = g.crop_tensor(img, (10, 10, 60, 50))
    assert a.shape == (3, 224, 224) and a.dtype == np.float32
    assert np.allclose(a[0], (1 - 0.485) / 0.229, atol=1e-5)


def test_calibrate_tau_exact_order_statistic():
    # evaluator example: N=200, 10 disagreements, 190 agreements at 0.5 -> tau 0.5
    agree = np.r_[np.zeros(10, bool), np.ones(190, bool)]
    score = np.r_[np.full(10, 0.9), np.full(190, 0.5)]
    c = g.calibrate_tau(agree, score)
    assert (c.q, c.tau, c.feasible) == (190, 0.5, True) and c.retention == 0.95
    # largest feasible threshold
    c = g.calibrate_tau(np.ones(20, bool), np.arange(20) / 20)
    assert c.q == 19 and c.tau == 0.05 and c.retention == 0.95
    # infeasible: 11 disagreements of 200
    c = g.calibrate_tau(np.r_[np.zeros(11, bool), np.ones(189, bool)], np.ones(200))
    assert c.tau == -1.0 and not c.feasible and c.retention == 189 / 200
    # empty
    c = g.calibrate_tau(np.array([], bool), np.array([]))
    assert c.tau is None and c.n == 0


def test_calibrate_conf():
    c = g.calibrate_conf(np.linspace(0.3, 0.9, 20))
    assert c.q == 19 and np.isclose(c.tau, np.linspace(0.3, 0.9, 20)[1])
    assert g.calibrate_conf(np.array([])).tau is None


def test_gate_frame_rules():
    protos = {"curbstone": np.array([1.0, 0.0]), "pole": np.array([0.0, 1.0])}
    z = np.array([0.8, 0.6])
    assert g.gate_frame(None, z, protos, 0.5, BUCKETS, CIDS) == g.NONE
    assert g.gate_frame("high_other", z, protos, 0.5, BUCKETS, CIDS) == g.BUCKET
    assert g.gate_frame("speedbump", z, protos, 0.5, BUCKETS, CIDS) == g.UNKNOWN   # no prototype
    assert g.gate_frame("curbstone", None, protos, 0.5, BUCKETS, CIDS) == g.UNKNOWN  # invalid crop
    assert g.gate_frame("curbstone", z, protos, 0.8, BUCKETS, CIDS) == g.PASS       # inclusive
    assert g.gate_frame("curbstone", z, protos, 0.81, BUCKETS, CIDS) == g.UNKNOWN
    assert g.gate_frame("pole", z, protos, 0.0, BUCKETS, CIDS) == g.UNKNOWN         # not nearest
    tie = np.array([1.0, 1.0]) / np.sqrt(2)
    assert g.gate_frame("curbstone", tie, protos, 0.0, BUCKETS, CIDS) == g.PASS     # tie -> lower id
    assert g.gate_frame("pole", tie, protos, 0.0, BUCKETS, CIDS) == g.UNKNOWN
    assert g.gate_frame("curbstone", z, protos, None, BUCKETS, CIDS) == g.UNKNOWN   # unavailable


def test_confidence_frame_shares_eligibility():
    assert g.confidence_frame("speedbump", 0.99, False, 0.3, BUCKETS) == g.UNKNOWN
    assert g.confidence_frame("pole", 0.3, True, 0.3, BUCKETS) == g.PASS
    assert g.confidence_frame("pole", 0.29, True, 0.3, BUCKETS) == g.UNKNOWN
    assert g.confidence_frame("low_other", 0.9, False, 0.3, BUCKETS) == g.BUCKET


def test_strict_majority():
    assert g.strict_majority(["pole", "pole", "none"]) == "pole"
    assert g.strict_majority(["pole", "pole", "none", "unknown"]) == g.UNKNOWN   # 50% is not >50%
    assert g.strict_majority(["pole", "none", "unknown"]) == g.UNKNOWN          # unique mode, no majority
    assert g.strict_majority([]) == g.UNKNOWN


def test_active_classes_and_heldout_assertion():
    crops = pd.DataFrame({"class_name": ["curbstone"] * 25 + ["pole"] * 25 + ["high_other"] * 10,
                          "sequence_id": [f"c{i % 6}" for i in range(25)] + [f"p{i % 3}" for i in range(25)]
                          + [f"h{i}" for i in range(10)],
                          "target_object": ["curbstone"] * 25 + ["pole"] * 25 + ["cone"] * 10})
    assert g.active_classes(crops, CIDS, {"woodenboard"}) == ["curbstone"]   # pole: 3 seqs; bucket: 10 crops
    with pytest.raises(AssertionError, match="held-out"):
        g.active_classes(crops, CIDS, {"cone"})


def test_sampler_balanced_with_replacement():
    index = {"a": {"s1": np.array([0, 1]), "s2": np.array([2])}, "b": {"s3": np.array([3])}}
    rows, labels = g.sample_batch(index, ["a", "b"], np.random.default_rng(0))
    assert len(rows) == 32 and (labels == 0).sum() == 16 and (labels == 1).sum() == 16
    assert set(rows[labels == 1]) == {3}


def test_prototypes_sequence_weighted():
    z = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    protos = g.prototypes(z, pd.Series(["a"] * 4), pd.Series(["s1", "s1", "s1", "s2"]), ["a"])
    assert np.allclose(protos["a"], np.array([1, 1]) / np.sqrt(2))   # s1 counts once


def test_supcon_loss_reference():
    torch = pytest.importorskip("torch")
    z = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]), dim=1)
    labels = torch.tensor([0, 0, 1, 2])          # label 2 has no positive -> dropped
    got = float(g.supcon_loss(z, labels, 0.1))
    zn = z.numpy()
    sim = zn @ zn.T / 0.1
    vals = []
    for i in range(3):
        pos = [j for j in range(4) if j != i and labels[j] == labels[i]]
        if not pos:
            continue
        others = [j for j in range(4) if j != i]
        lse = np.log(np.exp(sim[i, others]).sum())
        vals.append(-np.mean([sim[i, j] - lse for j in pos]))
    assert np.isclose(got, np.mean(vals), atol=1e-5)


def _write_rotation(tmp, name, rows):
    d = tmp / name
    d.mkdir()
    import json
    pd.DataFrame(rows, columns=["split", "sequence_id", "target_object", "camera", "frame_idx",
                                "source", "frame_path", "x0", "y0", "x1", "y1", "class_id"]
                 ).to_csv(d / "dataset_manifest.csv", index=False)
    (d / "provenance.json").write_text(json.dumps({"class_names": list(CIDS)}))
    return d / "dataset_manifest.csv"


def test_gt_rows_union_membership_and_heldout_guard(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import gate_features as gf
    a = _write_rotation(tmp_path, "a", [
        ("train", "s1", "curbstone", "rear", 0, "x", "", 1, 2, 30, 40, 0),
        ("val", "s2", "pole", "rear", 0, "x", "", 1, 2, 30, 40, 3),
        ("test", "s3", "woodenboard", "rear", 0, "x", "", 1, 2, 30, 40, 2)])
    b = _write_rotation(tmp_path, "b", [
        ("val", "s1", "curbstone", "rear", 0, "x", "", 1, 2, 30, 40, 0),
        ("train", "s3", "woodenboard", "rear", 0, "x", "", 1, 2, 30, 40, 2)])
    out = gf.gt_rows({"a": a, "b": b}, {"a": {"woodenboard"}, "b": {"pole"}}, set())
    r = out.set_index("sequence_id")
    assert r.loc["s1", "split_a"] == "train" and r.loc["s1", "split_b"] == "val"
    assert r.loc["s3", "split_a"] == "" and r.loc["s3", "split_b"] == "train"   # test never exported
    with pytest.raises(AssertionError, match="held-out"):
        gf.gt_rows({"a": a}, {"a": {"pole"}}, set())                           # pole in a's val
    with pytest.raises(AssertionError, match="reserved"):
        gf.gt_rows({"a": a}, {"a": {"woodenboard"}}, {"s1"})


def test_det_rows_selects_and_keeps_empty_frames():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import gate_features as gf
    raw = pd.DataFrame({"image": ["s1_rear_00000", "s1_rear_00000"], "pred": ["pole", "curbstone"],
                        "conf": [0.9, 0.4], "x0": [0, 0], "y0": [0, 0], "x1": [10, 10], "y1": [50, 80]})
    d = gf.det_rows(raw, pd.Series(["s1_rear_00000", "s2_front_00003"]), CIDS)
    assert d.loc[0, "class_name"] == "curbstone" and not d.loc[0, "no_detection"]
    assert bool(d.loc[1, "no_detection"]) and d.loc[1, "sequence_id"] == "s2"


def test_gate_train_end_to_end_synthetic(tmp_path, monkeypatch):
    import json
    import sys
    from pathlib import Path
    pytest.importorskip("torch")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import gate_train as gt
    monkeypatch.setattr(g, "STEPS_PER_EPOCH", 3)
    monkeypatch.setattr(g, "EPOCHS", 2)
    rng = np.random.default_rng(0)
    rows, feats = [], []
    for c, centre in (("curbstone", 0), ("pole", 1)):
        for s in range(6):
            for k in range(5):
                v = np.zeros(g.FEAT_DIM, np.float32); v[centre] = 1; v += rng.normal(0, .05, g.FEAT_DIM)
                rows.append({"sequence_id": f"{c}{s}", "camera": "rear", "frame_idx": k, "x0": 0, "y0": 0,
                             "x1": 40, "y1": 40, "class_name": c, "target_object": c,
                             "split_r": "train" if s < 5 else "val", "crop_status": "valid"})
                feats.append(v)
    gd = tmp_path / "gt"; gd.mkdir()
    pd.DataFrame(rows).assign(feature_row=range(len(rows))).to_csv(gd / "crops.csv", index=False)
    np.save(gd / "features.npy", np.stack(feats))
    val = [r for r in rows if r["split_r"] == "val"]
    dd = tmp_path / "det"; dd.mkdir()
    det = pd.DataFrame([{**{k: r[k] for k in ("sequence_id", "camera", "frame_idx", "x0", "y0", "x1", "y1",
                                                "class_name")}, "conf": 0.5 + 0.01 * i,
                         "crop_status": "valid", "no_detection": False} for i, r in enumerate(val)])
    det.assign(feature_row=range(len(det))).to_csv(dd / "crops.csv", index=False)
    np.save(dd / "features.npy", np.stack([feats[i] for i, r in enumerate(rows) if r["split_r"] == "val"]))
    md = tmp_path / "rot"; md.mkdir()
    pd.DataFrame([{"split": "val", "sequence_id": r["sequence_id"], "target_object": r["class_name"],
                   "camera": "rear", "frame_idx": r["frame_idx"], "x0": 0, "y0": 0, "x1": 40, "y1": 40,
                   "class_id": CIDS[r["class_name"]]} for r in val]).to_csv(md / "dataset_manifest.csv", index=False)
    (md / "provenance.json").write_text(json.dumps({"class_names": ["curbstone", "speedbump", "woodenboard",
                                                                    "pole", "dummychild", "bicyclestand",
                                                                    "low_other", "high_other"]}))
    out = tmp_path / "gate"
    monkeypatch.setattr(sys, "argv", ["x", "--rotation", "r", "--gt", str(gd), "--det", str(dd),
                                      "--manifest", str(md / "dataset_manifest.csv"),
                                      "--heldout", "woodenboard", "--out", str(out)])
    gt.main()
    cal = json.loads((out / "calibration.json").read_text())
    assert cal["active_classes"] == ["curbstone", "pole"] and cal["calibration_detections"] == 10
    for k in ("trained", "training_free", "confidence"):
        assert cal[k]["available"] and cal[k]["q"] == 10
        assert cal[k]["measured_retention"] >= 0.95 or not cal[k]["feasible"]
    assert cal["trained"]["unreliable_calibration"]          # 10 < 200 detections


def test_gt_rows_fills_missing_target_column(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import gate_features as gf
    a = _write_rotation(tmp_path, "a", [
        ("train", "s1", "curbstone", "rear", 0, "x", "", 1, 2, 30, 40, 0),
        ("test", "s3", "pole", "rear", 0, "x", "", 1, 2, 30, 40, 3)])
    b = _write_rotation(tmp_path, "b", [
        ("val", "s1", "curbstone", "rear", 0, "x", "", 1, 2, 30, 40, 0),
        ("train", "s3", "pole", "rear", 0, "x", "", 1, 2, 30, 40, 3)])
    pd.read_csv(b).drop(columns="target_object").to_csv(b, index=False)   # old-style manifest
    out = gf.gt_rows({"a": a, "b": b}, {"a": {"pole"}, "b": {"woodenboard"}}, set())
    assert out.set_index("sequence_id").loc["s3", "target_object"] == "pole"
    with pytest.raises(AssertionError, match="held-out"):
        gf.gt_rows({"a": a, "b": b}, {"a": {"pole"}, "b": {"pole"}}, set())   # guard still works


def test_tier_c_frames_fixed_and_test_only():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import tierc_frames as tc
    rows = [("test", "t1", "tree", "rear", i) for i in range(8)] + \
           [("test", "k1", "pole", "front", i) for i in range(3)] + \
           [("train", "x1", "pole", "front", 0)] + [("test", "t1", "tree", "rear", 0)]   # duplicate box
    m = pd.DataFrame(rows, columns=["split", "sequence_id", "target_object", "camera", "frame_idx"])
    a = tc.tier_c_frames(m, {"tree"})
    b = tc.tier_c_frames(m.sample(frac=1, random_state=3), {"tree"})
    pd.testing.assert_frame_equal(a, b)                       # independent of row order
    assert (a.groupby("sequence_id").size() == pd.Series({"k1": 3, "t1": 5})).all()
    assert "x1" not in set(a.sequence_id)
    assert set(a.loc[a.sequence_id == "t1", "population"]) == {"heldout"}


def test_det_rows_status_authoritative():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import gate_features as gf
    raw = pd.DataFrame({"image": ["s1_rear_00000"], "pred": ["pole"], "conf": [0.9],
                        "x0": [0], "y0": [0], "x1": [10], "y1": [50]})
    frames = pd.Series(["s1_rear_00000", "s1_rear_00001", "s1_rear_00002", "s1_rear_00003"])
    st = pd.DataFrame({"image": ["s1_rear_00000", "s1_rear_00001", "s1_rear_00002"],
                       "status": ["ok", "ok", "failed"], "n_detections": [1, 0, 0]})
    d = gf.det_rows(raw, frames, CIDS, st).set_index("image")
    assert d.loc["s1_rear_00000", "class_name"] == "pole" and d.loc["s1_rear_00000", "det_coverage"] == "verified"
    assert bool(d.loc["s1_rear_00001", "no_detection"])                  # evaluated, empty
    assert d.loc["s1_rear_00002", "det_status"] == gf.DET_FAILED          # never "no detection"
    assert d.loc["s1_rear_00003", "det_status"] == gf.DET_MISSING
    with pytest.raises(ValueError, match="n_detections"):                # status / raw disagreement
        gf.det_rows(raw, frames, CIDS, st.assign(n_detections=[0, 0, 0]))
    legacy = gf.det_rows(raw, frames, CIDS, None)
    assert (legacy.det_coverage == "unverified").all() and (legacy.det_status == "ok").all()


def test_freeze_check_detects_hash_and_runtime_mismatch(tmp_path, monkeypatch, capsys):
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import freeze_inventory as fi
    monkeypatch.setattr(fi, "git", lambda *a: "")           # code unchanged, clean tree
    (tmp_path / "a.csv").write_text("x\n1\n")
    inv = {"code_sha": "abc", "code_paths": ["src"], "runtime": {"python": fi.runtime()["python"]},
           "files": {"a.csv": {"sha256": fi.sha256(tmp_path / "a.csv"), "bytes": 4}}}
    p = tmp_path / "inv.json"
    p.write_text(json.dumps(inv))
    fi.check(tmp_path, p)                                    # passes
    (tmp_path / "a.csv").write_text("x\n2\n")
    with pytest.raises(SystemExit):
        fi.check(tmp_path, p)
    assert "hash mismatch a.csv" in capsys.readouterr().out
    (tmp_path / "a.csv").write_text("x\n1\n")
    inv["runtime"]["python"] = "0.0.0"
    p.write_text(json.dumps(inv))
    with pytest.raises(SystemExit):
        fi.check(tmp_path, p)


def test_freeze_declared_absent_and_m3_log(tmp_path, monkeypatch, capsys):
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import freeze_inventory as fi
    monkeypatch.setattr(fi, "git", lambda *a: "")
    inv = {"code_sha": "abc", "code_paths": ["src"], "runtime": {}, "files": {},
           "declared_absent": ["reports/eval_holdbush/inference_status.csv"],
           "m3_outputs": ["gate/gate_v1/gate_*"]}
    p = tmp_path / "inv.json"
    p.write_text(json.dumps(inv))
    fi.check(tmp_path, p)
    (tmp_path / "reports/eval_holdbush").mkdir(parents=True)
    (tmp_path / "reports/eval_holdbush/inference_status.csv").write_text("x")
    with pytest.raises(SystemExit):
        fi.check(tmp_path, p)
    assert "declared-absent" in capsys.readouterr().out
    out = tmp_path / "log.json"
    (tmp_path / "gate/gate_v1/gate_bush").mkdir(parents=True)
    (tmp_path / "gate/gate_v1/gate_bush/calibration.json").write_text("{}")
    fi.log(tmp_path, p, out)
    assert "gate/gate_v1/gate_bush/calibration.json" in json.loads(out.read_text())
    (tmp_path / "gate/gate_v1/gate_bush/calibration.json").write_text('{"tau": 1}')
    with pytest.raises(SystemExit, match="changed after it was logged"):
        fi.log(tmp_path, p, out)
