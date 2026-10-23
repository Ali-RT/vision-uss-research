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


def _build(tmp_path, monkeypatch, n_held=8):
    """Two rotations (tiers C and D) through gate_train; returns (cfg dict, paths)."""
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
        for s_ in range(6):
            for k in range(5):
                gt_rows.append({"sequence_id": f"tr{c}{s_}_2019010{s_}_x", "camera": "rear", "frame_idx": k,
                                "x0": 0, "y0": 0, "x1": 40, "y1": 40, "class_name": c, "target_object": c,
                                "split_C": "train" if s_ < 5 else "val", "split_D": "train" if s_ < 5 else "val",
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
        test_seqs = [(f"h{rot}{i}_2019020{i % 4}_x", held_t, held_bin) for i in range(n_held)] + \
                    [(f"k{rot}{i}_2019020{i % 4}_x", "pole", "high") for i in range(8)] + \
                    [(f"c{rot}{i}_2019020{i % 4}_x", "curbstone", "low") for i in range(8)]
        man = [{"split": "val", "sequence_id": r["sequence_id"], "target_object": r["class_name"],
                "camera": "rear", "frame_idx": r["frame_idx"], "x0": 0, "y0": 0, "x1": 40, "y1": 40,
                "class_id": names.index(r["class_name"])} for r in val]
        man += [{"split": "test", "sequence_id": s_, "target_object": t, "camera": "rear", "frame_idx": f,
                 "x0": 0, "y0": 0, "x1": 40, "y1": 40, "class_id": 0} for s_, t, _ in test_seqs for f in range(3)]
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
        det, feats, scored, frames = [], [], [], []
        for s_, t, truth in test_seqs:
            for f in range(3):
                cls = "pole" if t in (held_t, "pole") else "curbstone"
                det.append({"sequence_id": s_, "camera": "rear", "frame_idx": f, "class_name": cls,
                            "conf": .8, "x0": 0, "y0": 0, "x1": 40, "y1": 40, "crop_status": "valid",
                            "feature_row": len(feats)})
                feats.append(vec("woodenboard") if t == held_t else vec(cls))
                scored.append({"sequence_id": s_, "camera": "rear", "frame_idx": f, "target_object": t,
                               "truth": truth, "uss_track": True, "split": "test", "error": "",
                               "crop": "1,1,2,2", "crop_name": {"tree": "tree", "woodenboard": "wooden board",
                                                               "pole": "pole", "curbstone": "curb"}[t],
                               "full_name": "pole" if t == "pole" else "board" if t == "woodenboard" else "tree"})
                frames.append({"sequence_id": s_, "camera": "rear", "frame_idx": f, "target_object": t})
        dt = tmp_path / f"dettest{rot}"; dt.mkdir()
        pd.DataFrame(det).to_csv(dt / "crops.csv", index=False)
        np.save(dt / "features.npy", np.stack(feats))
        pd.DataFrame(scored).to_csv(tmp_path / f"scored{rot}.csv", index=False)
        pd.DataFrame(scored).to_csv(tmp_path / f"fixed{rot}.csv", index=False)
        pd.DataFrame(frames).to_csv(tmp_path / f"frames{rot}.csv", index=False)
        cfg["rotations"][rot] = {"tier": tier, "heldout": [held_t], "dataset": str(ds), "gate": str(gd_out),
                                 "det_test": str(dt), "vlm_scored": [str(tmp_path / f"scored{rot}.csv")],
                                 "vlm_fixed": str(tmp_path / f"fixed{rot}.csv"),
                                 "frames": str(tmp_path / f"frames{rot}.csv")}
    return cfg


def _run(tmp_path, monkeypatch, cfg):
    import json
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    out = tmp_path / "eval"
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(tmp_path / "cfg.json"), "--out", str(out)])
    ge.main()
    return (pd.read_csv(out / "sequence_decisions.csv", index_col=0),
            pd.read_csv(out / "endpoints.csv").set_index("endpoint"),
            json.loads((out / "accounting.json").read_text()))


def test_main_end_to_end_synthetic(tmp_path, monkeypatch):
    sd, ep, acct = _run(tmp_path, monkeypatch, _build(tmp_path, monkeypatch))
    held = sd[sd.population == "heldout"]
    assert (held.plain_mode == "high").all()                              # wrong "pole" bypasses
    assert set(held.loc[held.rotation == "D", "gate_trained"]) == {"low"}   # gate defers -> VLM board
    assert all(a["excluded_seqs"] == 0 for a in acct.values())
    assert "claim_vs_gate_free" in ep.index


def _edit(path, fn):
    df = pd.read_csv(path)
    fn(df).to_csv(path, index=False)


def test_failed_detector_frames_do_not_vote(tmp_path, monkeypatch):
    cfg = _build(tmp_path, monkeypatch)
    crops = Path(cfg["rotations"]["C"]["det_test"]) / "crops.csv"
    # known pole sequence kC0: frames 1,2 failed (prefixed error) -> decide from frame 0 alone
    _edit(crops, lambda d: d.assign(crop_status=np.where((d.sequence_id == "kC0_20190200_x") & (d.frame_idx > 0),
                                                         "error:FileNotFoundError", d.crop_status)))
    # kC1: every frame failed -> no decision, counted
    _edit(crops, lambda d: d.assign(crop_status=np.where(d.sequence_id == "kC1_20190201_x", "det_failed",
                                                         d.crop_status)))
    sd, ep, acct = _run(tmp_path, monkeypatch, cfg)
    assert sd.loc["C|kC0_20190200_x", "plain_mode"] == "high" and sd.loc["C|kC0_20190200_x", "detector"] == "high"
    assert "C|kC1_20190201_x" not in sd.index and acct["C"]["excluded_seqs"] == 1
    assert acct["C"]["frames"]["detector_failed"] == 5


def test_missing_outputs_are_excluded_not_dropped(tmp_path, monkeypatch):
    cfg = _build(tmp_path, monkeypatch)
    _edit(Path(cfg["rotations"]["C"]["vlm_scored"][0]), lambda d: d[d.sequence_id != "cC0_20190200_x"])
    _edit(Path(cfg["rotations"]["C"]["vlm_fixed"]), lambda d: d[d.sequence_id != "cC1_20190201_x"])
    _edit(Path(cfg["rotations"]["C"]["vlm_scored"][0]),          # partial: one frame missing
          lambda d: d[~((d.sequence_id == "cC2_20190202_x") & (d.frame_idx == 2))])
    sd, ep, acct = _run(tmp_path, monkeypatch, cfg)
    assert {"C|cC0_20190200_x", "C|cC1_20190201_x"}.isdisjoint(sd.index)
    assert acct["C"]["excluded_seqs"] == 2 and acct["C"]["requested_seqs"] == 24
    assert acct["C"]["frames"]["vlm_missing"] == 4 and acct["C"]["partial_seqs"]["vlm"] == 1
    assert ep.loc["P2b", "requested"] == 8 and np.isclose(ep.loc["P2b", "excluded_frac"], 2 / 8)
    assert not ep.loc["P2b", "pass"]                             # 25% > 2%


def test_conflicting_duplicates_raise(tmp_path, monkeypatch):
    import pytest
    cfg = _build(tmp_path, monkeypatch)
    path = Path(cfg["rotations"]["C"]["vlm_scored"][0])
    d = pd.read_csv(path)
    dup = d.iloc[[0]].assign(crop_name="car")
    pd.concat([d, dup]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="conflicting duplicate"):
        _run(tmp_path, monkeypatch, cfg)


def test_concentrated_exclusion_fails_p1(tmp_path, monkeypatch):
    cfg = _build(tmp_path, monkeypatch, n_held=40)
    crops = Path(cfg["rotations"]["C"]["det_test"]) / "crops.csv"
    # 2 of 40 held-out tall sequences lost = 5% of P1, but only 2 / 56 = 3.6% of tier C
    lost = {"hC0_20190200_x", "hC1_20190201_x"}
    _edit(crops, lambda d: d.assign(crop_status=np.where(d.sequence_id.isin(lost), "det_missing", d.crop_status)))
    sd, ep, acct = _run(tmp_path, monkeypatch, cfg)
    assert np.isclose(ep.loc["P1", "excluded_frac"], 2 / 40) and not ep.loc["P1", "pass"]


def test_missing_rotation_and_unavailable_comparator(tmp_path, monkeypatch):
    import json
    cfg = _build(tmp_path, monkeypatch)
    calp = Path(cfg["rotations"]["C"]["gate"]) / "calibration.json"
    cal = json.loads(calp.read_text()); cal["confidence"]["tau"] = None
    calp.write_text(json.dumps(cal))
    cfg["rotations"]["D"]["det_test"] = str(tmp_path / "nonexistent")
    sd, ep, acct = _run(tmp_path, monkeypatch, cfg)
    assert acct["D"]["evaluated_seqs"] == 0 and acct["D"]["inputs_missing"]
    assert not ep.loc["S1a", "pass"] and ep.loc["S1a", "requested"] == 8
    assert not ep.loc["claim_vs_gate_conf", "pass"]
    assert ep.loc["claim_NI_P1_vs_gate_conf", "note"] == "gate calibration unavailable"


def test_records_keep_rotation_identity(tmp_path, monkeypatch):
    """The same sequence is held out in C and known in D: it must count once per
    rotation and population, and losing the held-out record must not be hidden
    by the surviving known record."""
    cfg = _build(tmp_path, monkeypatch)
    rd = cfg["rotations"]["D"]
    shared = "hC0_20190200_x"                                  # tier C held-out tree sequence
    src = pd.read_csv(tmp_path / "scoredC.csv")
    srow = src[src.sequence_id == shared].assign(target_object="pole", truth="high",
                                                  crop_name="pole", full_name="pole")
    for f in ("scoredD.csv", "fixedD.csv"):
        pd.concat([pd.read_csv(tmp_path / f), srow]).to_csv(tmp_path / f, index=False)
    fr = pd.read_csv(rd["frames"])
    pd.concat([fr, srow[["sequence_id", "camera", "frame_idx", "target_object"]]]).to_csv(rd["frames"], index=False)
    crops = Path(rd["det_test"]) / "crops.csv"
    dcr = pd.read_csv(crops)
    extra = dcr[dcr.sequence_id == "kD0_20190200_x"].assign(sequence_id=shared)
    pd.concat([dcr, extra]).to_csv(crops, index=False)
    # lose the HELD-OUT (tier C) record entirely; the known (tier D) record survives
    cc = Path(cfg["rotations"]["C"]["det_test"]) / "crops.csv"
    _edit(cc, lambda d: d.assign(crop_status=np.where(d.sequence_id == shared, "det_failed", d.crop_status)))
    sd, ep, acct = _run(tmp_path, monkeypatch, cfg)
    assert "D|" + shared in sd.index and "C|" + shared not in sd.index
    assert sd.loc["D|" + shared, "population"] == "known"
    held = sd[sd.population == "heldout"]
    assert len(held) == 15 and (held.rotation + "|" + held.sequence_id).is_unique   # 8 + 8 - 1 lost
    assert ep.loc["P1", "requested"] == 8 and np.isclose(ep.loc["P1", "excluded_frac"], 1 / 8)
    assert np.isclose(ep.loc["claim_bypass_vs_gate_free", "excluded_frac"], 1 / 16)


def test_conflicting_crop_field_duplicate_raises(tmp_path, monkeypatch):
    import pytest
    cfg = _build(tmp_path, monkeypatch)
    path = Path(cfg["rotations"]["C"]["vlm_scored"][0])
    d = pd.read_csv(path)
    pd.concat([d, d.iloc[[0]].assign(crop="")]).to_csv(path, index=False)   # same names, crop differs
    with pytest.raises(ValueError, match="conflicting duplicate"):
        _run(tmp_path, monkeypatch, cfg)


def test_conflicting_detector_duplicate_raises(tmp_path, monkeypatch):
    import pytest
    cfg = _build(tmp_path, monkeypatch)
    crops = Path(cfg["rotations"]["C"]["det_test"]) / "crops.csv"
    d = pd.read_csv(crops)
    pd.concat([d, d.iloc[[0]].assign(class_name="curbstone")]).to_csv(crops, index=False)
    with pytest.raises(ValueError, match="conflicting duplicate"):
        _run(tmp_path, monkeypatch, cfg)


def test_all_rotations_missing_is_counted_no_go(tmp_path, monkeypatch):
    import json
    cfg = _build(tmp_path, monkeypatch)
    for rc in cfg["rotations"].values():
        rc["det_test"] = str(tmp_path / "nonexistent")
    sd, ep, acct = _run(tmp_path, monkeypatch, cfg)
    assert len(sd) == 0 and not ep["pass"].any()
    assert json.loads((tmp_path / "eval" / "decision.json").read_text())["GO"] is False


def test_real_frozen_population_counts():
    """Counts from the committed frozen frame lists (no predictions involved)."""
    root = Path(__file__).resolve().parents[1] / "configs" / "step4"
    lists = {r: pd.read_csv(root / f"tierd_frames_{r}.csv") for r in ("woodenboard", "speedbump", "bush", "bollard")}
    lists["treecubehose"] = pd.read_csv(root / "tierc_frames_treecubehose.csv")
    held = {r: f[f.population == "heldout"].sequence_id.nunique() for r, f in lists.items()}
    assert held["woodenboard"] + held["speedbump"] == 391                 # S1
    assert sum(held.values()) == 1204                                     # bypass claim population


def test_combine_runs_earlier_run_precedence_and_within_run_conflict():
    import pytest
    base = {"sequence_id": "s", "camera": "rear", "truth": "low", "error": "", "full_name": "pole"}
    r1 = pd.DataFrame([{**base, "frame_idx": 0, "crop_name": "board", "crop": "1,1,2,2"}])
    r2 = pd.DataFrame([{**base, "frame_idx": 0, "crop_name": "pole", "crop": "3,3,4,4"},
                       {**base, "frame_idx": 1, "crop_name": "board", "crop": "1,1,2,2"}])
    out, n_dup, sup = ge.combine_runs([r1, r2], "x")
    assert out.set_index("frame_idx").loc[0, "crop_name"] == "board" and sup == [0, 1] and len(out) == 2
    bad = pd.concat([r2, r2.iloc[[0]].assign(crop="")])           # conflict inside one run still raises
    with pytest.raises(ValueError, match="conflicting duplicate"):
        ge.combine_runs([r1, bad], "x")


def test_tier_d_combination_equals_step3_union():
    """On the saved development outputs, combine_runs reproduces step 3's union."""
    import pytest
    root = Path(__file__).resolve().parents[1] / "paper" / "results"
    if not (root / "prompt_decision_v1").exists():
        pytest.skip("private results not present")
    runs = [pd.read_csv(root / r / "scored.csv") for r in ("prompt_decision_v1", "prompt_decision_low_v1",
                                                          "prompt_decision_high_v1")]
    out, _, sup = ge.combine_runs(runs, "tierD")
    union = pd.concat(runs, ignore_index=True).drop_duplicates(ge.KEYS)
    key = ge.KEYS + ["crop_name", "full_name", "crop"]
    pd.testing.assert_frame_equal(out[key].astype(str).sort_values(ge.KEYS).reset_index(drop=True),
                                  union[key].astype(str).sort_values(ge.KEYS).reset_index(drop=True))
    assert sup == [0, 119, 380]
