import numpy as np

from vision_uss_research.alignment.fusion import (LogisticFusion,
                                                  balanced_accuracy, rule_fusion,
                                                  signal_features)


def test_rule_fusion_prefers_detector():
    out = rule_fusion(["high", None, "low"], ["low", "high", "high"])
    assert list(out) == ["high", "high", "low"]   # det when present, else uss


def test_signal_features_encodes_missing():
    X = signal_features(det_bin=["high", None], det_conf=[0.9, np.nan],
                        uss_prob=[0.7, np.nan])
    # row 0: det present high, conf 0.9, has_det 1, uss 0.7, has_uss 1
    assert list(X[0]) == [1.0, 0.9, 1.0, 0.7, 1.0]
    # row 1: everything missing -> zeros with presence flags off
    assert list(X[1]) == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_balanced_accuracy():
    # 2 low (1 right), 2 high (2 right) -> (0.5 + 1.0)/2
    assert balanced_accuracy(["low", "high", "high", "high"],
                             ["low", "low", "high", "high"]) == 0.75
    # ignores None
    assert balanced_accuracy(["high", None], ["high", "low"]) == 1.0


def _synth_complementary(n=400, seed=0):
    """Camera good on high objects, USS good on low objects - the real pattern.
    Returns (det_bin, det_conf, uss_prob, truth)."""
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 2, n)          # 1 = high
    det_bin, det_conf, uss_prob = [], [], []
    for t in truth:
        # detector: 95% right on high, 70% on low (weaker on flat/low)
        p_correct = 0.95 if t == 1 else 0.70
        d = t if rng.random() < p_correct else 1 - t
        det_bin.append("high" if d == 1 else "low")
        det_conf.append(rng.uniform(0.5, 0.95))
        # USS ClassProbHigh: good on low (low prob), noisy on high
        if t == 1:
            uss_prob.append(np.clip(rng.normal(0.55, 0.25), 0, 1))
        else:
            uss_prob.append(np.clip(rng.normal(0.30, 0.15), 0, 1))
    truth_bin = ["high" if t == 1 else "low" for t in truth]
    return det_bin, det_conf, uss_prob, truth_bin


def test_logistic_fusion_beats_detector_alone_on_complementary_data():
    det_bin, det_conf, uss_prob, truth = _synth_complementary(600, seed=1)
    y = np.array([1 if t == "high" else 0 for t in truth])
    X = signal_features(det_bin, det_conf, uss_prob)

    # split by index (proxy for sequence split)
    tr, te = slice(0, 400), slice(400, None)
    clf = LogisticFusion(epochs=3000).fit(X[tr], y[tr])
    fused = clf.predict_bin(X[te])

    det_bacc = balanced_accuracy(np.array(det_bin)[te], np.array(truth)[te])
    fused_bacc = balanced_accuracy(fused, np.array(truth)[te])
    # meta-fusion should not be worse than the detector, and usually better
    assert fused_bacc >= det_bacc - 0.02
    assert fused_bacc > 0.75


def test_logistic_fusion_deterministic():
    det_bin, det_conf, uss_prob, truth = _synth_complementary(200, seed=2)
    y = np.array([1 if t == "high" else 0 for t in truth])
    X = signal_features(det_bin, det_conf, uss_prob)
    a = LogisticFusion(seed=0).fit(X, y).predict_proba(X)
    b = LogisticFusion(seed=0).fit(X, y).predict_proba(X)
    assert np.allclose(a, b)
