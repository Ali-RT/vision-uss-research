"""Step 4 (M3): train one rotation's gate head, build prototypes, calibrate.

Inputs: the ground-truth crop features (gate_features.py gt), the rotation's
validation detector crops (gate_features.py det on eval_<tag>_val), and the
rotation's dataset manifest (val ground-truth boxes for calibration matching).

Produces, in --out:
  head.pt                 trained projection head (final step; no early stopping)
  prototypes.npz          trained-gate and training-free prototypes per active class
  calibration.json        tau (trained), tau (training-free), tau_conf, with
                          N, q, agreements, feasibility, measured retention,
                          insufficient-data flags; active classes; training log

Calibration population (§3.5): validation frames whose selected detection
matches a ground-truth box at IoU >= 0.5 with the same, named, active class.
CPU is sufficient (the head trains on cached 384-d features).

usage: python scripts/gate_train.py --rotation woodenboard \\
           --gt OUT/gate/gt_v1 --det OUT/gate/det_woodenboard_val \\
           --manifest .../dataset_v2holdwoodenboard/dataset_manifest.csv \\
           --heldout woodenboard --out OUT/gate/gate_woodenboard
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vision_uss_research.method import gate as g  # noqa: E402
from vision_uss_research.qa.flags import box_iou  # noqa: E402

KEYS = ["sequence_id", "camera", "frame_idx"]
MIN_CAL_TOTAL, MIN_CAL_PER_CLASS = 200, 20


def load(dirpath: Path) -> tuple[pd.DataFrame, np.ndarray]:
    return pd.read_csv(dirpath / "crops.csv"), np.load(dirpath / "features.npy")


def train_head(feats: np.ndarray, rows: pd.DataFrame, classes: list[str], log: list) -> object:
    import torch
    head = g.make_head()
    opt = torch.optim.AdamW(head.parameters(), lr=g.LR, weight_decay=g.WEIGHT_DECAY)
    index = {c: {s: grp.index.to_numpy() for s, grp in rows[rows.class_name == c].groupby("sequence_id")}
             for c in classes}
    rng = np.random.default_rng(g.SEED)
    x = torch.from_numpy(feats.astype(np.float32))
    for step in range(g.STEPS_PER_EPOCH * g.EPOCHS):
        ids, labels = g.sample_batch(index, classes, rng)
        z = torch.nn.functional.normalize(head(x[ids]), dim=1)
        loss = g.supcon_loss(z, torch.from_numpy(labels))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % g.STEPS_PER_EPOCH == 0:
            log.append({"epoch": (step + 1) // g.STEPS_PER_EPOCH, "train_loss": round(loss.item(), 5)})
    return head


def val_loss(head, feats, rows, classes) -> float | None:
    """Logged only (no checkpoint selection): one balanced batch, seed 1."""
    import torch
    rows = rows[rows.class_name.isin(classes)]
    if rows.empty:
        return None
    index = {c: {s: grp.index.to_numpy() for s, grp in rows[rows.class_name == c].groupby("sequence_id")}
             for c in classes if (rows.class_name == c).any()}
    present = list(index)
    ids, labels = g.sample_batch(index, present, np.random.default_rng(1))
    with torch.inference_mode():
        z = torch.nn.functional.normalize(head(torch.from_numpy(feats[ids].astype(np.float32))), dim=1)
        return round(float(g.supcon_loss(z, torch.from_numpy(labels))), 5)


def calibration_population(det: pd.DataFrame, manifest: pd.DataFrame, names: list[str],
                           active: list[str], buckets: set[str]) -> pd.DataFrame:
    """Correct known detections: selected detection matches a val GT box at
    IoU >= 0.5 with the same class, the class is named (not a bucket) and active."""
    gt = manifest[manifest.split == "val"].copy()
    gt["class_name"] = gt.class_id.map(lambda i: names[int(i)])
    gt_by = dict(tuple(gt.groupby(KEYS)))
    keep = []
    for r in det[(det.crop_status == "valid")].itertuples(index=False):
        if r.class_name in buckets or r.class_name not in active:
            continue
        frame = gt_by.get((r.sequence_id, r.camera, int(r.frame_idx)))
        if frame is None:
            continue
        same = frame[frame.class_name == r.class_name]
        if any(box_iou((r.x0, r.y0, r.x1, r.y1), (b.x0, b.y0, b.x1, b.y1)) >= 0.5
               for b in same.itertuples(index=False)):
            keep.append(r.feature_row)
    return det[det.feature_row.isin(keep)]


def agreement(z: np.ndarray, cls: pd.Series, protos: dict, class_ids: dict) -> tuple[np.ndarray, np.ndarray]:
    names = sorted(protos, key=lambda c: class_ids[c])
    p = np.stack([protos[c] for c in names])
    sims = z @ p.T
    best = np.array(names)[sims.argmax(1)]
    own = np.array([sims[i, names.index(c)] for i, c in enumerate(cls)])
    return best == cls.to_numpy(), own


def cal_json(c: g.Calibration, per_class: dict, flag: bool) -> dict:
    return {"tau": c.tau, "n": c.n, "q": c.q, "n_agree": c.n_agree, "feasible": c.feasible,
            "measured_retention": c.retention, "per_class_n": per_class,
            "unreliable_calibration": flag, "available": c.tau is not None}


def main() -> None:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--rotation", required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--det", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--heldout", required=True, help="comma-separated held-out types")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    heldout = set(a.heldout.split(","))
    names = json.loads((a.manifest.parent / "provenance.json").read_text())["class_names"]
    class_ids = {n: i for i, n in enumerate(names)}
    buckets = {"low_other", "high_other"}

    rows, feats = load(a.gt)
    col = f"split_{a.rotation}"
    rows = rows[(rows.crop_status == "valid")].copy()
    train = rows[rows[col] == "train"].reset_index(drop=True)
    val = rows[rows[col] == "val"].reset_index(drop=True)
    ftrain, fval = feats[train.feature_row.to_numpy()], feats[val.feature_row.to_numpy()]
    active = g.active_classes(train, class_ids, heldout)
    train_a = train[train.class_name.isin(active)].reset_index(drop=True)
    ftrain_a = ftrain[train.class_name.isin(active).to_numpy()]

    log: list = []
    head = train_head(ftrain_a, train_a, active, log)
    log.append({"val_loss_logged_only": val_loss(head, fval, val, active)})
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), a.out / "head.pt")

    protos = {"trained": g.prototypes(g.embed(head, ftrain_a), train_a.class_name, train_a.sequence_id,
                                      active),
              "training_free": g.prototypes(g.embed(None, ftrain_a), train_a.class_name,
                                            train_a.sequence_id, active)}
    np.savez(a.out / "prototypes.npz", **{f"{k}__{c}": v for k, p in protos.items() for c, v in p.items()})

    det, dfeats = load(a.det)
    cal = calibration_population(det, pd.read_csv(a.manifest), names, active, buckets)
    per_class = cal.class_name.value_counts().to_dict()
    flag = len(cal) < MIN_CAL_TOTAL or any(per_class.get(c, 0) < MIN_CAL_PER_CLASS
                                           for c in active if c not in buckets)
    fcal = dfeats[cal.feature_row.to_numpy()]
    out = {"rotation": a.rotation, "heldout": sorted(heldout), "active_classes": active,
           "train_crops": len(train_a), "train_crops_per_class": train_a.class_name.value_counts().to_dict(),
           "calibration_detections": len(cal), "training_log": log}
    for gate_name, head_ in (("trained", head), ("training_free", None)):
        agree, own = agreement(g.embed(head_, fcal), cal.class_name, protos[gate_name], class_ids) \
            if len(cal) else (np.array([], bool), np.array([]))
        out[gate_name] = cal_json(g.calibrate_tau(agree, own), per_class, flag)
    out["confidence"] = cal_json(g.calibrate_conf(cal.conf.to_numpy()), per_class, flag)
    for f in ("head.pt", "prototypes.npz"):
        out[f"sha256_{f}"] = hashlib.sha256((a.out / f).read_bytes()).hexdigest()
    (a.out / "calibration.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("rotation", "active_classes", "calibration_detections",
                                          "trained", "training_free", "confidence")}, indent=1, default=float))


if __name__ == "__main__":
    main()
