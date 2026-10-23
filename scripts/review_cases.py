"""Contact sheets for manual review of Step 4 evaluation cases (post hoc; no
setting of the frozen protocol depends on this).

One PNG per selected sequence: its evaluated frames, each with the detector's
selected box and class, the trained-gate and confidence-gate frame labels,
the USS crop rectangle, and the VLM names (USS crop / full frame); the header
gives truth and every arm's sequence decision. Plus review_sheet.csv with
blank columns for the reviewer.

Selection (known-type or held-out population of one rotation):
  hurt    - trained gate wrong, detector + constant-High right
  helped  - trained gate right, detector + constant-High wrong
  both    - both wrong
  changed - hurt + helped + both

usage (Colab, after the evaluate stage):
  python scripts/review_cases.py --artifacts OUT --rotation treecubehose \\
      --population known --select changed --out OUT/gate/gate_v1/review_known
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_evaluate as ge  # noqa: E402

ROT = {"treecubehose": ("dataset_v2holdtreecubehose", ["prompt_decision_tierc_v1"]),
       "woodenboard": ("dataset_v2holdwoodenboard", ["prompt_decision_v1", "prompt_decision_low_v1", "prompt_decision_high_v1"]),
       "speedbump": ("dataset_v2holdspeedbump", ["prompt_decision_v1", "prompt_decision_low_v1", "prompt_decision_high_v1"]),
       "bush": ("dataset_v2holdbush", ["prompt_decision_v1", "prompt_decision_low_v1", "prompt_decision_high_v1"]),
       "bollard": ("dataset_v2holdbollard", ["prompt_decision_v1", "prompt_decision_low_v1", "prompt_decision_high_v1"])}
ARMS_SHOWN = ("detector", "detector_high", "plain_mode", "gate_trained", "gate_conf", "vlm_uss", "vlm_full")


def select(sd: pd.DataFrame, kind: str) -> pd.DataFrame:
    g, d = sd.gate_trained == sd.truth, sd.detector_high == sd.truth
    masks = {"hurt": ~g & d, "helped": g & ~d, "both": ~g & ~d}
    masks["changed"] = masks["hurt"] | masks["helped"] | masks["both"]
    out = sd[masks[kind]].copy()
    out["case"] = np.select([masks["hurt"][out.index], masks["helped"][out.index]], ["hurt", "helped"], "both")
    return out


def sheet(seq: str, frames: pd.DataFrame, row: pd.Series, frames_root: Path, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.8), squeeze=False)
    for ax, (_, f) in zip(axes[0], frames.iterrows()):
        ax.axis("off")
        p = frames_root / seq / f.camera / f"{int(f.frame_idx):05d}.jpg"
        if p.exists():
            ax.imshow(Image.open(p).convert("RGB"))
        else:
            ax.text(.5, .5, f"missing {p.name}", ha="center", transform=ax.transAxes)
        if isinstance(f.crop, str) and f.crop:
            x0, y0, x1, y1 = map(float, f.crop.split(","))
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="cyan", lw=1.5, ls="--"))
        if not pd.isna(f.x0):
            ax.add_patch(Rectangle((f.x0, f.y0), f.x1 - f.x0, f.y1 - f.y0, fill=False, ec="red", lw=2))
        ax.set_title(f"{f.camera} #{int(f.frame_idx)} | det: {f.det_class} ({f.conf:.2f})\n"
                     f"gate: {f.gate_trained} | conf-gate: {f.gate_conf}\n"
                     f"VLM crop: {f.crop_name} | full: {f.full_name}", fontsize=8)
    dec = " | ".join(f"{a}={row[a]}" for a in ARMS_SHOWN)
    fig.suptitle(f"{seq} [{row.case}]  type={row.type}  TRUTH={row.truth}\n{dec}\n"
                 "red = detector selected box, cyan dashed = USS crop", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=80)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--rotation", default="treecubehose", choices=list(ROT))
    ap.add_argument("--population", default="known", choices=["known", "heldout"])
    ap.add_argument("--select", default="changed", choices=["hurt", "helped", "both", "changed"])
    ap.add_argument("--gate-run", default="gate_v1")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    A, gd = a.artifacts, a.artifacts / "gate" / a.gate_run
    ds, runs = ROT[a.rotation]
    sd = pd.read_csv(gd / "evaluation" / "sequence_decisions.csv", index_col=0)
    sd = sd[(sd.rotation == a.rotation) & (sd.population == a.population)]
    cases = select(sd, a.select)
    names = json.loads((A / "datasets" / ds / "provenance.json").read_text())["class_names"]
    det = pd.read_csv(gd / f"det_{a.rotation}_test" / "crops.csv")
    det = det[det.sequence_id.isin(set(cases.sequence_id))]
    feats = np.load(gd / f"det_{a.rotation}_test" / "features.npy")
    fl, _ = ge.frame_labels(det, feats, gd / f"gate_{a.rotation}", {n: i for i, n in enumerate(names)})
    fl = fl.join(det[["class_name", "conf", "x0", "y0", "x1", "y1"]])
    scored = pd.concat([pd.read_csv(A / "method" / r / "scored.csv") for r in runs], ignore_index=True)
    scored = scored.drop_duplicates(ge.KEYS)            # step-3 run precedence (amendment A1)
    fr = fl.merge(scored[ge.KEYS + ["crop", "crop_name", "full_name"]], on=ge.KEYS, how="left")
    frames_root = A / "frames" / "r30-85_n20"
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec, r in cases.iterrows():
        seq = r.sequence_id
        sheet(seq, fr[fr.sequence_id == seq].sort_values(["camera", "frame_idx"]), r, frames_root,
              a.out / f"{r.case}_{seq}.png")
        rows.append({"sequence_id": seq, "case": r.case, "type": r.type, "truth": r.truth,
                     **{arm: r[arm] for arm in ARMS_SHOWN},
                     "box_on_target": "", "label_correct": "", "vlm_name_plausible": "", "notes": ""})
    pd.DataFrame(rows).to_csv(a.out / "review_sheet.csv", index=False)
    print(f"{len(rows)} sheets -> {a.out}  ({cases.case.value_counts().to_dict()})")


if __name__ == "__main__":
    main()
