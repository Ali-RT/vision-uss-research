"""Step 4 (M2/M3): crop export + frozen DINOv2 features for the open-set gate.

Reads image content (crops are cut from frames) - this is NOT a metadata-only
step. Images come from a materialized local dataset directory (any rotation's
unpacked local_images.tar holds every labelled frame, under
images/<split>/<sequence>/<sequence>_<camera>_<frame>.jpg); a frame is found
by its stem, whatever split directory it sits in.

Modes
  gt   - training/calibration-free ground-truth crops: the union of the TRAIN
         and VAL ground-truth boxes of the given rotations. Columns
         split_<rotation> say which rotation may use a row and how; a row of a
         rotation's held-out types or of a reserved ID is never marked for
         that rotation (asserted). No visual screening.
  det  - detector crops: the selected detection (largest y1, then conf, then
         class id) of every frame listed in an inference_status.csv (status
         ok) or a frame-key CSV; frames without a detection are recorded
         with crop_status "no_detection".

Outputs (in --out): crops.csv (one row per crop: keys, class, source box,
padded+clipped box, transform version, crop_status valid|invalid|error,
sha256 of the RGB crop bytes, feature row) and features.npy (float32,
N x 384; zero rows for non-valid crops). The CSV and .npy are hashed at M2.

usage (Colab GPU):
  python scripts/gate_features.py gt --images /content/dataset/dataset_v2holdwoodenboard \\
      --rotation woodenboard=.../dataset_v2holdwoodenboard/dataset_manifest.csv ... \\
      --heldout woodenboard=woodenboard speedbump=speedbump bush=bush bollard=bollard \\
               treecubehose=tree,cubestandard,hose \\
      --reserved configs/splits/reserved_sessions_v1.csv --out OUT/gate/gt_v1
  python scripts/gate_features.py det --images ... --class-names-from MANIFEST_DIR \\
      --raw .../eval_holdwoodenboard_val/raw_predictions.csv \\
      --status .../eval_holdwoodenboard_val/inference_status.csv --out OUT/gate/det_woodenboard_val
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

TRANSFORM_VERSION = "pad10_clip_min8_rgb_224_bilinear_aa_imagenet_v1"
KEYS = ["sequence_id", "camera", "frame_idx"]
BATCH = 128


def stem(seq: str, cam: str, idx) -> str:
    return f"{seq}_{cam}_{int(idx):05d}"


def image_index(images_dir: Path) -> dict[str, Path]:
    return {p.stem: p for p in (images_dir / "images").rglob("*.jpg")}


def gt_rows(rotations: dict[str, Path], heldout: dict[str, set[str]], reserved: set[str]) -> pd.DataFrame:
    """Union of train/val GT boxes; split_<rotation> in {train, val, ''}."""
    mans = {rot: pd.read_csv(path) for rot, path in rotations.items()}
    # older manifests (the bollard rotation) predate the target_object column;
    # the target is a property of the sequence, so take it from the others
    seq_target = pd.concat([m[["sequence_id", "target_object"]] for m in mans.values()
                            if "target_object" in m]).drop_duplicates()
    assert not seq_target.sequence_id.duplicated().any(), "a sequence has two target types"
    seq_target = seq_target.set_index("sequence_id").target_object
    frames = []
    for rot, man_path in rotations.items():
        m = mans[rot]
        if "target_object" not in m:
            m = m.assign(target_object=m.sequence_id.map(seq_target))
            assert m.target_object.notna().all(), f"{rot}: sequences without a known target type"
        names = json.loads((man_path.parent / "provenance.json").read_text())["class_names"]
        m = m[m.split.isin(["train", "val"])].copy()
        m["class_name"] = m.class_id.map(lambda i: names[int(i)])
        bad = m[m.target_object.isin(heldout[rot]) | m.sequence_id.isin(reserved)]
        assert bad.empty, f"{rot}: {len(bad)} held-out/reserved rows in train/val"
        m["rotation"] = rot
        frames.append(m[KEYS + ["x0", "y0", "x1", "y1", "class_name", "target_object", "split", "rotation"]])
    allr = pd.concat(frames, ignore_index=True)
    box = KEYS + ["x0", "y0", "x1", "y1"]
    wide = allr.pivot_table(index=box, columns="rotation", values="split", aggfunc="first").fillna("")
    wide.columns = [f"split_{c}" for c in wide.columns]
    meta = allr.drop_duplicates(box).set_index(box)[["class_name", "target_object"]]
    # one GT box can carry a different class name in different rotations only
    # through class_groups; all rotations share dataset_v2's grouping, asserted:
    per = allr.groupby(box).class_name.nunique()
    assert (per <= 1).all(), "a GT box has different class names across rotations"
    out = meta.join(wide).reset_index()
    for rot in rotations:
        col = f"split_{rot}"
        assert not out.loc[out[col] != "", "target_object"].isin(heldout[rot]).any()
    return out


def det_rows(raw: pd.DataFrame, frames: pd.Series, class_ids: dict[str, int]) -> pd.DataFrame:
    """Selected detection per listed frame (image stem); no detection kept as a row."""
    rows = []
    by_img = dict(tuple(raw.groupby("image")))
    for img in frames:
        d = by_img.get(img)
        sel = None if d is None else g.select_detection(d, class_ids)
        seq, cam, idx = img.rsplit("_", 2)
        if sel is None:
            rows.append({"image": img, "sequence_id": seq, "camera": cam, "frame_idx": int(idx),
                         "class_name": None, "conf": np.nan, "x0": np.nan, "y0": np.nan,
                         "x1": np.nan, "y1": np.nan, "no_detection": True})
        else:
            rows.append({"image": img, "sequence_id": seq, "camera": cam, "frame_idx": int(idx),
                         "class_name": sel.pred, "conf": float(sel.conf), "x0": sel.x0, "y0": sel.y0,
                         "x1": sel.x1, "y1": sel.y1, "no_detection": False})
    return pd.DataFrame(rows)


def extract(rows: pd.DataFrame, idx: dict[str, Path], out: Path, device: str) -> None:
    import torch
    from PIL import Image
    from transformers import Dinov2Model
    model = Dinov2Model.from_pretrained(g.DINO_MODEL).to(device).eval().float()
    rev = getattr(model.config, "_commit_hash", None)
    feats = np.zeros((len(rows), g.FEAT_DIM), np.float32)
    status, sha, boxes = [], [], []
    batch, slots = [], []

    def flush():
        if batch:
            feats[slots] = g.dinov2_features(np.stack(batch), model, device)
            batch.clear(), slots.clear()

    for i, r in enumerate(rows.itertuples(index=False)):
        if getattr(r, "no_detection", False):
            status.append("no_detection"), sha.append(""), boxes.append("")
            continue
        p = idx.get(stem(r.sequence_id, r.camera, r.frame_idx))
        try:
            img = Image.open(p)
            img.load()
        except Exception as e:  # noqa: BLE001 - execution failure, counted
            status.append(f"error:{type(e).__name__}"), sha.append(""), boxes.append("")
            continue
        b = g.pad_clip_box(r.x0, r.y0, r.x1, r.y1, img.width, img.height)
        if b is None:
            status.append("invalid"), sha.append(""), boxes.append("")
            continue
        crop = img.convert("RGB").crop(b)
        sha.append(hashlib.sha256(np.asarray(crop).tobytes()).hexdigest())
        status.append("valid"), boxes.append(",".join(map(str, b)))
        batch.append(g.crop_tensor(img, b)), slots.append(i)
        if len(batch) == BATCH:
            flush()
    flush()
    rows = rows.assign(crop_status=status, crop_box=boxes, crop_sha256=sha,
                       transform=TRANSFORM_VERSION, feature_row=np.arange(len(rows)))
    out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / "crops.csv", index=False)
    np.save(out / "features.npy", feats)
    meta = {"model": g.DINO_MODEL, "model_revision": rev, "torch": torch.__version__,
            "transformers": __import__("transformers").__version__, "transform": TRANSFORM_VERSION,
            "rows": len(rows), "crop_status": rows.crop_status.value_counts().to_dict()}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    for f in ("crops.csv", "features.npy"):
        meta[f"sha256_{f}"] = hashlib.sha256((out / f).read_bytes()).hexdigest()
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gt", "det"])
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rotation", nargs="*", default=[], help="gt: rot=path/dataset_manifest.csv")
    ap.add_argument("--heldout", nargs="*", default=[], help="gt: rot=type[,type]")
    ap.add_argument("--reserved", type=Path, default=None)
    ap.add_argument("--class-names-from", type=Path, help="det: dataset dir with provenance.json")
    ap.add_argument("--raw", type=Path, help="det: raw_predictions.csv")
    ap.add_argument("--status", type=Path, help="det: inference_status.csv (ok frames)")
    ap.add_argument("--frames", type=Path, help="det: CSV with sequence_id,camera,frame_idx")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N rows only (never for M2/M3 outputs)")
    a = ap.parse_args()
    idx = image_index(a.images)
    if a.mode == "gt":
        rot = {k: Path(v) for k, v in (s.split("=", 1) for s in a.rotation)}
        held = {k: set(v.split(",")) for k, v in (s.split("=", 1) for s in a.heldout)}
        assert set(rot) == set(held), "every rotation needs its held-out types"
        reserved = set(pd.read_csv(a.reserved).sequence_id) if a.reserved else set()
        rows = gt_rows(rot, held, reserved)
    else:
        names = json.loads((a.class_names_from / "provenance.json").read_text())["class_names"]
        raw = pd.read_csv(a.raw)
        if a.status:
            st = pd.read_csv(a.status)
            frames = st.loc[st.status == "ok", "image"]
        else:
            f = pd.read_csv(a.frames)
            frames = pd.Series([stem(*k) for k in f[KEYS].itertuples(index=False)])
        rows = det_rows(raw, frames, {n: i for i, n in enumerate(names)})
    if a.limit:
        rows = rows.head(a.limit)
        print(f"SMOKE: limited to {a.limit} rows - not a protocol output")
    missing = sum(stem(r.sequence_id, r.camera, r.frame_idx) not in idx
                  for r in rows.itertuples(index=False))
    print(f"{a.mode}: {len(rows)} crops, {missing} without a local image (-> error rows)")
    extract(rows, idx, a.out, a.device)


if __name__ == "__main__":
    main()
