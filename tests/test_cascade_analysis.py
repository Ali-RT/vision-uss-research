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


def test_detector_nearest_tie_and_evaluation_status():
    sc = _scored([
        ("s1", "rear", 0, "bush", "high", True, "", "", "", "test"),
        ("s1", "rear", 1, "bush", "high", True, "", "", "", "test"),
        ("s2", "rear", 0, "bush", "high", True, "", "", "", "test"),
        ("s3", "rear", 0, "bush", "high", True, "", "", "", "test"),
    ])
    raw = pd.DataFrame({"image": ["s1_rear_00000", "s1_rear_00000"],
                        "pred": ["pole", "curbstone"], "y1": [100.0, 300.0]})
    cls = cg.detector_sequences(sc, raw)
    # nearest (largest y1) = curbstone; frame 1 has no detection -> tie -> none
    assert cls["s1"] == "none"
    assert cls["s2"] == "none" and cls["s3"] == "none"
    # with a status file: s3 never evaluated -> excluded, not "none"
    evaluated = {"s1_rear_00000", "s1_rear_00001", "s2_rear_00000"}
    cls = cg.detector_sequences(sc, raw, evaluated)
    assert cls["s2"] == "none"
    assert cls["s3"] == cg.NOT_EVALUATED


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
def test_report_denominators_on_saved_results():
    """Denominators of report tables 3.4-3.6 regenerate from the saved outputs."""
    scored = pd.concat([pd.read_csv(RESULTS / f"prompt_decision_{s}_v1" / "scored.csv")
                        for s in ("low", "high")], ignore_index=True)
    v = cg.vlm_sequences(scored)
    n = v.target_object.value_counts().to_dict()
    assert n == {"speedbump": 295, "bush": 200, "cone": 200, "car": 178,
                 "bollard": 129, "woodenboard": 96}
    for t in ("woodenboard", "speedbump", "bush", "bollard"):
        raw = pd.read_csv(RESULTS / f"eval_hold{t}" / "raw_predictions.csv")
        cls = cg.detector_sequences(scored[scored.target_object == t], raw)
        assert len(cls) == n[t] and (cls != cg.NOT_EVALUATED).all()
    seen = pd.concat([pd.read_csv(RESULTS / f"prompt_decision_{s}" / "scored.csv")
                      for s in ("v1", "low_v1", "high_v1")], ignore_index=True)
    fp = pd.read_csv(RESULTS / "eval_causal" / "frame_predictions.csv")
    assert len(cst.seen_type_decisions(seen, fp, "plain", "uss")) == 381
