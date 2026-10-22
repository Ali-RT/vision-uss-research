"""§6 decision rules of scripts/gate_evaluate.py on synthetic data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gate_evaluate as ge  # noqa: E402


def _pop(n_days=10, per_day=10, truth="high", a_low=0, b_low=0):
    rows = []
    for d in range(n_days):
        for i in range(per_day):
            k = d * per_day + i
            rows.append({"day": f"d{d}", "truth": truth,
                         "a": "low" if k < a_low else ("high" if truth == "high" else "low"),
                         "b": "low" if k < b_low else ("high" if truth == "high" else "low")})
    return pd.DataFrame(rows)


def test_equality_passes_and_unrounded():
    x = _pop()                                   # identical arms -> diff 0 in every resample
    r = ge.endpoint("P", x, "a", "b", "false_low", "upper", 0.0)
    assert r["point"] == 0 and r["bound"] == 0 and r["pass"]   # bound == margin passes
    r = ge.endpoint("P", x, "a", "b", "false_low", "lower", 0.0, strict=True)
    assert not r["pass"]                         # strict superiority needs bound > 0


def test_false_low_margin_and_side():
    x = _pop(a_low=5)                            # a has 5% false Low, b none
    r = ge.endpoint("P1", x, "a", "b", "false_low", "upper", 0.02)
    assert np.isclose(r["point"], 0.05) and not r["pass"]


def test_undefined_resamples_not_estimable():
    x = _pop(truth="high")                       # no Low truth -> low_recall undefined everywhere
    r = ge.endpoint("S", x, "a", "b", "low_recall", "lower", -0.05)
    assert not r["estimable"] and not r["pass"]


def test_execution_failure_threshold_fails():
    x = _pop()
    assert ge.endpoint("P", x, "a", "b", "accuracy", "lower", -0.01, excluded_frac=0.02)["pass"]
    assert not ge.endpoint("P", x, "a", "b", "accuracy", "lower", -0.01, excluded_frac=0.021)["pass"]


def test_p3_ratio():
    x = pd.DataFrame({"day": [f"d{i % 5}" for i in range(20)], "frames": 5,
                      "deferred_gate_trained": [i < 8 for i in range(20)]})
    r = ge.p3(x, 0.0)
    assert np.isclose(r["point"], 0.4)


def test_bypass_claim_direction():
    held = pd.DataFrame({"day": [f"d{i % 10}" for i in range(100)], "truth": "low",
                         "gate_trained": "low", "deferred_gate_trained": True,
                         "gate_free": ["high" if i < 30 else "low" for i in range(100)],
                         "deferred_gate_free": [i >= 30 for i in range(100)]})
    r = ge.bypass_claim(held, "gate_free")
    assert np.isclose(r["point"], 0.30) and r["pass"]      # comparator - trained > 0


def test_unique_mode_and_routing_labels():
    assert ge.unique_mode(pd.Series(["pole", "pole", "none"])) == "pole"
    assert ge.unique_mode(pd.Series(["pole", "none"])) == "none"


def test_main_end_to_end_synthetic(tmp_path, monkeypatch):
    """Two rotations (tiers C and D) through gate_train -> gate_evaluate."""
    import json
    import pytest
    pytest.importorskip("torch")
    import gate_train as gt
    from vision_uss_research.method import gate as g
    monkeypatch.setattr(g, "STEPS_PER_EPOCH", 2)
    monkeypatch.setattr(g, "EPOCHS", 1)
    names = ["curbstone", "speedbump", "woodenboard", "pole", "dummychild", "bicyclestand",
             "low_other", "high_other"]
    rng = np.random.default_rng(0)

    def vec(c):
        v = np.zeros(g.FEAT_DIM, np.float32); v[names.index(c)] = 1
        return v + rng.normal(0, .05, g.FEAT_DIM).astype(np.float32)

    gt_rows, gt_feats = [], []
    for c in ("curbstone", "pole"):
        for s in range(6):
            for k in range(5):
                gt_rows.append({"sequence_id": f"tr{c}{s}_2019010{s}_x", "camera": "rear", "frame_idx": k,
                                "x0": 0, "y0": 0, "x1": 40, "y1": 40, "class_name": c, "target_object": c,
                                "split_C": "train" if s < 5 else "val", "split_D": "train" if s < 5 else "val",
                                "crop_status": "valid"})
                gt_feats.append(vec(c))
    gd = tmp_path / "gt"; gd.mkdir()
    pd.DataFrame(gt_rows).assign(feature_row=range(len(gt_rows))).to_csv(gd / "crops.csv", index=False)
    np.save(gd / "features.npy", np.stack(gt_feats))
    cfg = {"rotations": {}, "dev_low_types": ["woodenboard"]}
    for rot, tier, held_t, held_bin in (("C", "C", "tree", "high"), ("D", "D", "woodenboard", "low")):
        ds = tmp_path / f"ds{rot}"; ds.mkdir()
        (ds / "provenance.json").write_text(json.dumps({"class_names": names}))
        val = [r for r in gt_rows if r[f"split_{rot}"] == "val"]
        test_seqs = [(f"h{rot}{i}_2019020{i % 4}_x", held_t, held_bin) for i in range(8)] + \
                    [(f"k{rot}{i}_2019020{i % 4}_x", "pole", "high") for i in range(8)] + \
                    [(f"c{rot}{i}_2019020{i % 4}_x", "curbstone", "low") for i in range(8)]
        man = [{"split": "val", "sequence_id": r["sequence_id"], "target_object": r["class_name"],
                "camera": "rear", "frame_idx": r["frame_idx"], "x0": 0, "y0": 0, "x1": 40, "y1": 40,
                "class_id": names.index(r["class_name"])} for r in val]
        man += [{"split": "test", "sequence_id": s, "target_object": t, "camera": "rear", "frame_idx": f,
                 "x0": 0, "y0": 0, "x1": 40, "y1": 40, "class_id": 0} for s, t, _ in test_seqs for f in range(3)]
        pd.DataFrame(man).to_csv(ds / "dataset_manifest.csv", index=False)
        dv = tmp_path / f"detval{rot}"; dv.mkdir()
        pd.DataFrame([{**{k: r[k] for k in ("sequence_id", "camera", "frame_idx", "x0", "y0", "x1", "y1",
                                             "class_name")}, "conf": .9, "crop_status": "valid",
                       "no_detection": False, "feature_row": i} for i, r in enumerate(val)]
                     ).to_csv(dv / "crops.csv", index=False)
        np.save(dv / "features.npy", np.stack([vec(r["class_name"]) for r in val]))
        gd_out = tmp_path / f"gate{rot}"
        monkeypatch.setattr(sys, "argv", ["x", "--rotation", rot, "--gt", str(gd), "--det", str(dv),
                                          "--manifest", str(ds / "dataset_manifest.csv"),
                                          "--heldout", held_t, "--out", str(gd_out)])
        gt.main()
        # test detections: held-out sequences get a confident WRONG named class "pole"
        det, feats, scored = [], [], []
        for s, t, truth in test_seqs:
            for f in range(3):
                cls = "pole" if t in (held_t, "pole") else "curbstone"
                det.append({"sequence_id": s, "camera": "rear", "frame_idx": f, "class_name": cls,
                            "conf": .8, "x0": 0, "y0": 0, "x1": 40, "y1": 40, "crop_status": "valid",
                            "feature_row": len(feats)})
                feats.append(vec("woodenboard") if t == held_t else vec(cls))
                scored.append({"sequence_id": s, "camera": "rear", "frame_idx": f, "target_object": t,
                               "truth": truth, "uss_track": True, "split": "test", "error": "",
                               "crop": "1,1,2,2", "crop_name": {"tree": "tree", "woodenboard": "wooden board",
                                                               "pole": "pole", "curbstone": "curb"}[t],
                               "full_name": "pole" if t == "pole" else "board" if t == "woodenboard" else "tree"})
        dt = tmp_path / f"dettest{rot}"; dt.mkdir()
        pd.DataFrame(det).to_csv(dt / "crops.csv", index=False)
        np.save(dt / "features.npy", np.stack(feats))
        pd.DataFrame(scored).to_csv(tmp_path / f"scored{rot}.csv", index=False)
        pd.DataFrame(scored).to_csv(tmp_path / f"fixed{rot}.csv", index=False)
        cfg["rotations"][rot] = {"tier": tier, "heldout": [held_t], "dataset": str(ds), "gate": str(gd_out),
                                 "det_test": str(dt), "vlm_scored": [str(tmp_path / f"scored{rot}.csv")],
                                 "vlm_fixed": str(tmp_path / f"fixed{rot}.csv")}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    out = tmp_path / "eval"
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(tmp_path / "cfg.json"), "--out", str(out)])
    ge.main()
    sd = pd.read_csv(out / "sequence_decisions.csv", index_col=0)
    held = sd[sd.population == "heldout"]
    assert (held.plain_mode == "high").all()                              # wrong "pole" bypasses
    assert set(held.loc[held.rotation == "D", "gate_trained"]) == {"low"}   # gate defers -> VLM board
    assert json.loads((out / "decision.json").read_text())["GO"] in (True, False)
    assert (pd.read_csv(out / "accounting.csv").excluded == 0).all()
