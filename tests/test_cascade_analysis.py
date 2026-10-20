"""Logic tests for the cascade analysis scripts (synthetic data; no results needed)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import cascade_gtfree as cg  # noqa: E402
import cascade_seen_types as cst  # noqa: E402


def _scored(rows):
    cols = ["sequence_id", "camera", "frame_idx", "target_object", "truth", "uss_track",
            "crop_name", "full_name", "error", "split"]
    return pd.DataFrame(rows, columns=cols)


def test_vote_tie_and_empty_abstain():
    assert cg.vote(pd.Series(["low", "low", "high"])) == "low"
    assert cg.vote(pd.Series(["low", "high"])) is None          # tie
    assert cg.vote(pd.Series([None, None])) is None


def test_vlm_ground_words_fallback_and_default_high():
    sc = _scored([
        # tracked: crop says board (low), full says car (high)
        ("a", "rear", 0, "woodenboard", "low", True, "wooden board", "car", "", "test"),
        ("a", "rear", 1, "woodenboard", "low", True, "board", "car", "", "test"),
        # untracked: USS arm must fall back to the full frame
        ("b", "rear", 0, "woodenboard", "low", False, None, "speed bump", "", "test"),
        # only ground words -> abstain -> High by default, raw stays None
        ("c", "rear", 0, "bush", "high", True, "grass", "curb", "", "test"),
        # an errored frame never counts
        ("c", "rear", 1, "bush", "high", True, "board", "board", "OOM", "test"),
    ])
    v = cg.vlm_sequences(sc)
    assert v.loc["a", "uss"] == "low" and v.loc["a", "full"] == "high"
    assert v.loc["b", "uss"] == "low"                           # fallback to full
    assert pd.isna(v.loc["c", "uss_raw"]) and v.loc["c", "uss"] == "high"
    assert len(v) == 3


def _raw(rows):
    return pd.DataFrame(rows, columns=["image", "pred", "y1"])


def _status(rows):
    return pd.DataFrame(rows, columns=["image", "status"])


def _bush_frames(*seq_frames):
    return _scored([(s, "rear", f, "bush", "high", True, "bush", "bush", "", "test")
                    for s, f in seq_frames])


def test_detector_nearest_and_tie_without_status():
    sc = _bush_frames(("s1", 0), ("s1", 1), ("s2", 0))
    raw = _raw([("s1_rear_00000", "pole", 100.0), ("s1_rear_00000", "curbstone", 300.0)])
    det = cg.detector_sequences(sc, raw)
    # nearest (largest y1) = curbstone; frame 1 has no detection -> tie -> none
    assert det.cls["s1"] == "none" and det.cls["s2"] == "none"
    assert (det.coverage == "unverified").all()


def test_status_is_authoritative():
    sc = _bush_frames(("s1", 0), ("s1", 1), ("s2", 0), ("s3", 0))
    raw = _raw([("s1_rear_00000", "pole", 100.0)])
    st = _status([("s1_rear_00000", "ok"), ("s2_rear_00000", "ok"),
                  ("s3_rear_00000", "failed")])
    det = cg.detector_sequences(sc, raw, st)
    assert det.cls["s1"] == "pole"                     # only its ok frame votes
    assert bool(det.partial["s1"]) and det.missing["s1"] == 1
    assert det.cls["s2"] == "none"                     # evaluated, nothing found
    assert det.cls["s3"] == cg.NOT_EVALUATED and det.failed["s3"] == 1
    assert (det.coverage == "verified").all()


def test_status_contradictions_raise():
    sc = _bush_frames(("s1", 0))
    raw = _raw([("s1_rear_00000", "pole", 100.0)])
    with pytest.raises(ValueError, match="no ok status"):
        cg.detector_sequences(sc, raw, _status([]))          # raw but no status
    with pytest.raises(ValueError, match="no ok status"):
        cg.detector_sequences(sc, raw, _status([("s1_rear_00000", "failed")]))
    with pytest.raises(ValueError, match="duplicate"):
        cg.detector_sequences(sc, raw, _status([("s1_rear_00000", "ok")] * 2))


def test_zero_coverage_does_not_crash_and_rates_are_none(tmp_path):
    import subprocess
    sc = _bush_frames(("s1", 0))
    sc.to_csv(tmp_path / "scored.csv", index=False)
    _raw([]).to_csv(tmp_path / "raw.csv", index=False)
    _status([("s1_rear_00000", "failed")]).to_csv(tmp_path / "status.csv", index=False)
    v = cg.vlm_sequences(sc)
    det = cg.detector_sequences(sc, _raw([]), _status([("s1_rear_00000", "failed")]))
    r = cg.evaluate(v, det, "uss", "plain")
    assert r["n"] == 0 and r["not_evaluated"] == 1 and r["cascade_rate"] is None
    out = subprocess.run([sys.executable, str(Path(cg.__file__)), "--scored",
                          str(tmp_path / "scored.csv"), "--raw", f"bush={tmp_path / 'raw.csv'}",
                          "--status", f"bush={tmp_path / 'status.csv'}"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_all_vlm_error_sequence_is_counted_not_crashing():
    sc = _scored([
        ("a", "rear", 0, "pole", "high", True, "pole", "pole", "", "test"),
        ("b", "rear", 0, "pole", "high", True, "pole", "pole", "OOM", "test"),
        ("b", "rear", 1, "pole", "high", True, "pole", "pole", "OOM", "test"),
    ])
    v = cg.vlm_sequences(sc)
    assert v.vlm_status["b"] == "vlm_failed" and pd.isna(v.uss["b"])   # no default High
    det = cg.detector_sequences(sc, _raw([]))
    r = cg.evaluate(v, det, "uss", "plain")
    assert r["n"] == 1 and r["vlm_failed"] == 1
    fp = pd.DataFrame({"sequence_id": ["a", "b"], "camera": ["rear", "rear"],
                       "frame_idx": [0, 0], "pred_class": ["pole", "pole"], "conf": [0.9, 0.9]})
    d = cst.seen_type_decisions(sc, fp, "plain", "uss")
    assert list(d.index) == ["a"]


def test_cascade_rules():
    v = pd.DataFrame({"uss": ["high", "high", "low"], "uss_raw": ["high", None, "low"]},
                     index=["x", "y", "z"])
    cls = pd.Series({"x": "curbstone", "y": "curbstone", "z": "high_other"})
    plain = cg.cascade(v, cls, "uss", "plain")
    assert list(plain) == ["low", "low", "low"]          # named decides; bucket defers
    both = cg.cascade(v, cls, "uss", "both")
    # only a POSITIVE VLM High overrules a named Low; an abstention does not
    assert list(both) == ["high", "low", "low"]


def test_seen_types_uses_test_split_only():
    sc = _scored([
        ("t1", "rear", 0, "pole", "high", True, "pole", "pole", "", "test"),
        ("t2", "rear", 0, "pole", "high", True, "pole", "pole", "", "train"),
    ])
    fp = pd.DataFrame({"sequence_id": ["t1", "t2"], "camera": ["rear", "rear"],
                       "frame_idx": [0, 0], "pred_class": ["pole", "pole"], "conf": [0.9, 0.9]})
    r = cst.seen_type_decisions(sc, fp, "plain", "uss")
    assert list(r.index) == ["t1"]
    assert bool(r.loc["t1", "cascade"])


RESULTS = Path(__file__).resolve().parents[1] / "paper" / "results"


@pytest.mark.skipif(not (RESULTS / "prompt_decision_low_v1").exists(),
                    reason="private results not present")
def test_report_tables_regenerate_exactly():
    """Every cell behind report tables 3.3-3.6 (numerators, denominators,
    false-Low counts, rates, intervals; both rules, both arms) matches the
    committed golden file regenerated at the step-2 revision."""
    import report_tables
    got = report_tables.build(RESULTS)
    want = pd.read_csv(Path(__file__).parent / "golden" / "report_tables.csv")
    key = ["section", "type", "rule", "arm"]
    norm = lambda t: (t.astype(str).replace({"nan": "", "None": ""})  # noqa: E731
                      .sort_values(key).reset_index(drop=True))
    got = got.reindex(columns=want.columns)
    pd.testing.assert_frame_equal(norm(got.fillna("")), norm(want.fillna("")), check_dtype=False)
