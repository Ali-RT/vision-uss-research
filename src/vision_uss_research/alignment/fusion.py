"""Late fusion of camera + USS Low/High signals into one decision.

Two fusers:
- `rule_fusion`: transparent, no training - trust the detector when it fired,
  else the USS. The baseline (matches notebook 14).
- `LogisticFusion`: a tiny logistic-regression meta-classifier on the signal
  features (detector bin & confidence, USS ClassProbHigh). Trained by gradient
  descent in numpy - no sklearn - so it is dependency-free and unit-testable.
  It can learn the complementarity (defer to USS on low-confidence camera on
  the flat/low classes it handles better).

'high' is the positive class throughout.
"""

from __future__ import annotations

import numpy as np


def rule_fusion(det_bin, uss_bin):
    """Trust the detector when present, else USS; None when neither exists.
    Array-like in, numpy object array out."""
    d = np.asarray(det_bin, dtype=object)
    u = np.asarray(uss_bin, dtype=object)
    out = np.where(d != None, d, u)   # noqa: E711 - object array needs !=None
    return out


def signal_features(det_bin, det_conf, uss_prob) -> np.ndarray:
    """Feature matrix for the meta-classifier, robust to missing signals:
    [det_is_high, det_conf, has_detection, uss_prob, has_uss]. Missing values
    are encoded as 0 with an explicit presence flag, so the model can learn to
    ignore an absent signal."""
    d = np.asarray(det_bin, dtype=object)
    dc = np.asarray(det_conf, dtype=float)
    up = np.asarray(uss_prob, dtype=float)
    has_det = (d != None).astype(float)             # noqa: E711
    det_high = np.array([1.0 if b == "high" else 0.0 for b in d])
    det_conf = np.where(np.isfinite(dc), dc, 0.0) * has_det
    has_uss = np.isfinite(up).astype(float)
    uss_prob = np.where(np.isfinite(up), up, 0.0)
    return np.column_stack([det_high * has_det, det_conf, has_det,
                            uss_prob, has_uss])


class LogisticFusion:
    """Logistic regression (numpy gradient descent) mapping signal features to
    P(high). Standardizes features internally."""

    def __init__(self, lr: float = 0.1, epochs: int = 2000, l2: float = 1e-3,
                 seed: int = 0, class_weight: str | None = "balanced"):
        self.lr, self.epochs, self.l2, self.seed = lr, epochs, l2, seed
        # "balanced" weights each sample by inverse class frequency, so the
        # minority class (low, ~20%) is not drowned out - optimizes toward
        # balanced accuracy rather than raw accuracy on imbalanced data.
        self.class_weight = class_weight
        self.w = self.b = self.mu = self.sd = None

    def fit(self, X, y) -> "LogisticFusion":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        Xs = (X - self.mu) / self.sd
        rng = np.random.default_rng(self.seed)
        self.w = rng.normal(0, 0.01, Xs.shape[1])
        self.b = 0.0
        n = len(y)
        if self.class_weight == "balanced":
            n_pos, n_neg = y.sum(), (y == 0).sum()
            w_pos = n / (2 * n_pos) if n_pos else 1.0
            w_neg = n / (2 * n_neg) if n_neg else 1.0
            sw = np.where(y == 1, w_pos, w_neg)
        else:
            sw = np.ones(n)
        sw_sum = sw.sum()
        for _ in range(self.epochs):
            p = 1 / (1 + np.exp(-(Xs @ self.w + self.b)))
            grad_w = Xs.T @ (sw * (p - y)) / sw_sum + self.l2 * self.w
            grad_b = float((sw * (p - y)).sum() / sw_sum)
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X) -> np.ndarray:
        Xs = (np.asarray(X, dtype=float) - self.mu) / self.sd
        return 1 / (1 + np.exp(-(Xs @ self.w + self.b)))

    def predict_bin(self, X, threshold: float = 0.5):
        return np.where(self.predict_proba(X) >= threshold, "high", "low")


def balanced_accuracy(pred_bin, truth_bin) -> float:
    """Mean of low-recall and high-recall (robust to class imbalance)."""
    p = np.asarray(pred_bin, dtype=object)
    t = np.asarray(truth_bin, dtype=object)
    good = (p != None) & (t != None)   # noqa: E711
    p, t = p[good], t[good]
    accs = []
    for lab in ("low", "high"):
        m = t == lab
        if m.any():
            accs.append((p[m] == lab).mean())
    return float(np.mean(accs)) if accs else float("nan")
