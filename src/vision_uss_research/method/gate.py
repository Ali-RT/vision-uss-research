"""Open-set gate for the cascade (Step 4 protocol, STEP4_PROPOSAL.md revision 2).

Every function here implements one numbered rule of the approved design; the
section is cited in its docstring. Model code (DINOv2, the projection head)
imports torch lazily so the pure rules stay testable without it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

PAD_FRAC = 0.10            # §3.2 step 2
MIN_SIDE_PX = 8            # §3.2 step 3
CROP_PX = 224              # §3.2 step 5
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DINO_MODEL = "facebook/dinov2-small"
FEAT_DIM, HIDDEN_DIM, EMB_DIM = 384, 512, 128   # §3.3 head
TEMPERATURE = 0.1          # §3.3 loss
PER_CLASS = 16             # §3.3 sampler: crops per class per batch
STEPS_PER_EPOCH, EPOCHS = 500, 10
LR, WEIGHT_DECAY, SEED = 1e-3, 1e-4, 0
MIN_CLASS_CROPS, MIN_CLASS_SEQS = 20, 5         # §3.3 active classes
RETENTION = 0.95           # §3.5
PASS, UNKNOWN, NONE, BUCKET = "pass", "unknown", "none", "bucket"


# ---- §3.1 selected detection -------------------------------------------------
def select_detection(dets: pd.DataFrame, class_ids: dict[str, int]) -> pd.Series | None:
    """The selected detection of one frame: largest y1, then higher confidence,
    then lower class id. None when the frame has no detection."""
    if dets.empty:
        return None
    d = dets.assign(_cid=dets.pred.map(class_ids).fillna(10 ** 6))
    d = d.sort_values(["y1", "conf", "_cid"], ascending=[False, False, True], kind="mergesort")
    return d.iloc[0].drop(labels="_cid")


# ---- §3.2 crops ----------------------------------------------------------------
def pad_clip_box(x0: float, y0: float, x1: float, y1: float, img_w: int, img_h: int
                 ) -> tuple[int, int, int, int] | None:
    """Pad 10% of width (each side, horizontally) and 10% of height (each side,
    vertically), clip to the image; None (invalid crop) if a side < 8 px."""
    w, h = x1 - x0, y1 - y0
    bx0 = max(0.0, x0 - PAD_FRAC * w)
    by0 = max(0.0, y0 - PAD_FRAC * h)
    bx1 = min(float(img_w), x1 + PAD_FRAC * w)
    by1 = min(float(img_h), y1 + PAD_FRAC * h)
    box = (int(math.floor(bx0)), int(math.floor(by0)), int(math.ceil(bx1)), int(math.ceil(by1)))
    if box[2] - box[0] < MIN_SIDE_PX or box[3] - box[1] < MIN_SIDE_PX:
        return None
    return box


def crop_tensor(img, box: tuple[int, int, int, int]) -> np.ndarray:
    """RGB crop -> 224x224 bilinear antialiased (aspect not kept) -> [0,1] ->
    ImageNet-normalized, CHW float32. `img` is a PIL image."""
    from PIL import Image
    c = img.convert("RGB").crop(box).resize((CROP_PX, CROP_PX), Image.BILINEAR, reducing_gap=None)
    a = np.asarray(c, dtype=np.float32) / 255.0
    a = (a - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    return a.transpose(2, 0, 1).copy()


def dinov2_features(batch: np.ndarray, model, device: str) -> np.ndarray:
    """pooler_output (layer-normed CLS, 384-d) for a (B,3,224,224) batch; fp32, eval."""
    import torch
    with torch.inference_mode():
        out = model(pixel_values=torch.from_numpy(batch).to(device))
    return out.pooler_output.float().cpu().numpy()


# ---- §3.3 active classes, sampler, loss, head, prototypes ---------------------
def active_classes(crops: pd.DataFrame, class_ids: dict[str, int], heldout_types: set[str]
                   ) -> list[str]:
    """Detector classes with >= 20 valid training crops from >= 5 sequences; the
    held-out type's crops must already be absent (asserted)."""
    assert not set(crops.target_object) & heldout_types, "held-out crops in training data"
    g = crops.groupby("class_name").agg(n=("sequence_id", "size"), s=("sequence_id", "nunique"))
    ok = g[(g.n >= MIN_CLASS_CROPS) & (g.s >= MIN_CLASS_SEQS)].index
    return [c for c in sorted(class_ids, key=class_ids.get) if c in ok]


def sample_batch(index: dict[str, dict[str, np.ndarray]], classes: list[str],
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """16 crops per active class: sequence uniform within class, crop uniform
    within sequence, with replacement. `index[class][sequence]` = crop row ids.
    Returns (row ids, class positions)."""
    rows, labels = [], []
    for ci, c in enumerate(classes):
        seqs = list(index[c])
        for _ in range(PER_CLASS):
            s = seqs[rng.integers(len(seqs))]
            rows.append(index[c][s][rng.integers(len(index[c][s]))])
            labels.append(ci)
    return np.array(rows), np.array(labels)


def supcon_loss(z, labels, temperature: float = TEMPERATURE):
    """SupCon "out" (Khosla et al. 2020): z L2-normalized (B,D). Every sample
    is an anchor; positives = same label excluding self; denominator = all
    others; anchors without a positive are dropped from the mean."""
    import torch
    sim = z @ z.T / temperature
    b = z.shape[0]
    self_mask = torch.eye(b, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (labels[:, None] == labels[None, :]) & ~self_mask
    n_pos = pos.sum(1)
    keep = n_pos > 0
    mean_log_pos = torch.where(pos, log_prob, torch.zeros_like(log_prob)).sum(1)[keep] / n_pos[keep]
    return -mean_log_pos.mean()


def make_head():
    import torch
    torch.manual_seed(SEED)
    return torch.nn.Sequential(torch.nn.Linear(FEAT_DIM, HIDDEN_DIM), torch.nn.ReLU(),
                               torch.nn.Linear(HIDDEN_DIM, EMB_DIM))


def embed(head, feats: np.ndarray) -> np.ndarray:
    """L2-normalized head embedding; head=None means the training-free gate
    (normalized raw DINOv2 features)."""
    if head is None:
        z = feats.astype(np.float64)
    else:
        import torch
        with torch.inference_mode():
            z = head(torch.from_numpy(feats.astype(np.float32))).double().numpy()
    return z / np.linalg.norm(z, axis=1, keepdims=True)


def prototypes(z: np.ndarray, classes: pd.Series, seqs: pd.Series, active: list[str]
               ) -> dict[str, np.ndarray]:
    """Sequence-weighted: mean of normalized embeddings per sequence, equal-weight
    mean over sequences, renormalized."""
    out = {}
    df = pd.DataFrame({"c": classes.values, "s": seqs.values})
    for c in active:
        idx = np.flatnonzero(df.c.values == c)
        per_seq = [z[idx[df.s.values[idx] == s]].mean(0) for s in pd.unique(df.s.values[idx])]
        p = np.mean(per_seq, axis=0)
        out[c] = p / np.linalg.norm(p)
    return out


# ---- §3.4 gate rule ------------------------------------------------------------
def gate_frame(det_class: str | None, z: np.ndarray | None, protos: dict[str, np.ndarray],
               tau: float | None, buckets: set[str], class_ids: dict[str, int]) -> str:
    """Frame label: NONE (no detection), BUCKET, UNKNOWN or PASS.
    z None = invalid crop -> UNKNOWN. tau None = calibration unavailable (the
    caller must treat the rotation as not estimable)."""
    if det_class is None:
        return NONE
    if det_class in buckets:
        return BUCKET
    if det_class not in protos or z is None or tau is None:
        return UNKNOWN
    names = sorted(protos, key=lambda c: class_ids[c])          # tie -> lower class id
    sims = np.array([float(z @ protos[c]) for c in names])
    best = names[int(np.argmax(sims))]                          # argmax keeps first on ties
    return PASS if (best == det_class and float(z @ protos[det_class]) >= tau) else UNKNOWN


def confidence_frame(det_class: str | None, conf: float | None, has_proto: bool,
                     tau_conf: float | None, buckets: set[str]) -> str:
    """Comparator 6: same eligibility as the trained gate (no prototype -> UNKNOWN)."""
    if det_class is None:
        return NONE
    if det_class in buckets:
        return BUCKET
    if not has_proto or tau_conf is None:
        return UNKNOWN
    return PASS if conf >= tau_conf else UNKNOWN


# ---- §3.5 calibration ----------------------------------------------------------
@dataclass
class Calibration:
    tau: float | None          # None = unavailable (N = 0)
    n: int
    q: int
    n_agree: int
    feasible: bool
    retention: float | None    # measured joint retention at tau


def calibrate_tau(agree: np.ndarray, score: np.ndarray) -> Calibration:
    """Exact order statistic: q = ceil(0.95 N); tau = q-th largest agreement score
    among agreeing detections (inclusive); infeasible -> tau = -1; N = 0 ->
    unavailable."""
    agree = np.asarray(agree, bool)
    score = np.asarray(score, float)
    n = len(agree)
    if n == 0:
        return Calibration(None, 0, 0, 0, False, None)
    q = math.ceil(RETENTION * n)
    s = np.sort(score[agree])[::-1]
    if len(s) >= q:
        tau, feasible = float(s[q - 1]), True
    else:
        tau, feasible = -1.0, False
    retention = float((agree & (score >= tau)).sum() / n)
    return Calibration(tau, n, q, int(agree.sum()), feasible, retention)


def calibrate_conf(conf: np.ndarray) -> Calibration:
    """Comparator 6: tau_conf = q-th largest confidence, q = ceil(0.95 N)."""
    conf = np.asarray(conf, float)
    n = len(conf)
    if n == 0:
        return Calibration(None, 0, 0, 0, False, None)
    q = math.ceil(RETENTION * n)
    tau = float(np.sort(conf)[::-1][q - 1])
    return Calibration(tau, n, q, n, True, float((conf >= tau).sum() / n))


# ---- §3.6 sequence routing -----------------------------------------------------
def strict_majority(labels: list[str]) -> str:
    """The label held by more than 50% of the evaluated frames, else UNKNOWN.
    `labels` are frame labels where PASS frames carry their class name."""
    if not labels:
        return UNKNOWN
    vc = pd.Series(labels).value_counts()
    return vc.index[0] if vc.iat[0] * 2 > len(labels) else UNKNOWN
