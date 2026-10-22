"""Step 4 M2 freeze: build (`build`) or verify (`check`) the machine-readable inventory.

build (Colab, after every M2 artifact exists): hashes every frozen input and
records the runtime, the active classes per rotation (computed from the valid
training crops with the §3.3 rule), and the code revision under review.

check (first cell of every M3 run): fails unless
  - the code paths are unchanged since the inventory's code_sha (only the
    inventory and documentation may differ),
  - every inventoried file still has its recorded sha256,
  - the runtime versions equal the recorded ones.

usage:
  python scripts/freeze_inventory.py build --artifacts OUT --out OUT/gate/gate_v1/freeze_inventory.json
  python scripts/freeze_inventory.py check --artifacts OUT --inventory configs/step4/freeze_inventory.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vision_uss_research.method import gate as g  # noqa: E402

ROTATIONS = {"woodenboard": ("dataset_v2holdwoodenboard", "woodenboard"),
             "speedbump": ("dataset_v2holdspeedbump", "speedbump"),
             "bush": ("dataset_v2holdbush", "bush"),
             "bollard": ("dataset_v2holdbollard", "bollard"),
             "treecubehose": ("dataset_v2holdtreecubehose", "tree,cubestandard,hose")}
GATE_RUN = "gate_v1"
PACKAGES = ("torch", "torchvision", "transformers", "accelerate", "ultralytics", "numpy", "pandas",
            "pillow", "scipy", "qwen-vl-utils", "opencv-python", "opencv-python-headless")
CODE_PATHS = ("src", "scripts", "notebooks", "requirements", "configs/experiments", "configs/splits",
              "configs/step4/tierc_frames_treecubehose.csv") + tuple(
              f"configs/step4/tierd_frames_{r}.csv" for r in ("woodenboard", "speedbump", "bush", "bollard"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def runtime() -> dict:
    out = {"python": platform.python_version()}
    for p in PACKAGES:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    return out


def files(artifacts: Path) -> dict[str, Path]:
    f = {}
    for rot, (ds, _) in ROTATIONS.items():
        d = artifacts / "datasets" / ds
        f[f"datasets/{ds}/dataset_manifest.csv"] = d / "dataset_manifest.csv"
        f[f"datasets/{ds}/provenance.json"] = d / "provenance.json"
        m = f"models/train_v1_hold{rot}/train/weights/best.pt"
        f[m] = artifacts / m
        for name in ("raw_predictions.csv", "inference_status.csv"):
            r = f"reports/eval_hold{rot}_val/{name}"
            f[r] = artifacts / r
        for name in ("crops.csv", "features.npy", "meta.json"):
            r = f"gate/{GATE_RUN}/det_{rot}_val/{name}"
            f[r] = artifacts / r
    for name in ("crops.csv", "features.npy", "meta.json"):
        r = f"gate/{GATE_RUN}/gt/{name}"
        f[r] = artifacts / r
    for r in ("method/prompt_uss_proj_v1/slots_at_frames.csv", "method/prompt_uss_proj_v1/sequence_status.csv"):
        f[r] = artifacts / r
    return f


def active_classes(artifacts: Path) -> dict:
    crops = pd.read_csv(artifacts / "gate" / GATE_RUN / "gt" / "crops.csv")
    crops = crops[crops.crop_status == "valid"]
    out = {}
    for rot, (ds, held) in ROTATIONS.items():
        names = json.loads((artifacts / "datasets" / ds / "provenance.json").read_text())["class_names"]
        train = crops[crops[f"split_{rot}"] == "train"]
        out[rot] = g.active_classes(train, {n: i for i, n in enumerate(names)}, set(held.split(",")))
    return out


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()


def build(artifacts: Path, out: Path) -> None:
    inv = {"code_sha": git("rev-parse", "HEAD"), "code_paths": list(CODE_PATHS),
           "dino_revision": g.DINO_REVISION, "runtime": runtime(),
           "active_classes": active_classes(artifacts),
           "files": {k: {"sha256": sha256(p), "bytes": p.stat().st_size} for k, p in files(artifacts).items()}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(inv, indent=1))
    print(json.dumps({k: v for k, v in inv.items() if k != "files"}, indent=1))
    print(f"{len(inv['files'])} files hashed -> {out}")


def check(artifacts: Path, inventory: Path) -> None:
    inv = json.loads(inventory.read_text())
    problems = []
    changed = git("diff", "--name-only", inv["code_sha"], "HEAD", "--", *inv["code_paths"])
    if changed:
        problems.append(f"code changed since {inv['code_sha'][:10]}: {changed.splitlines()[:5]}")
    dirty = git("status", "--porcelain", "--", *inv["code_paths"])
    if dirty:
        problems.append(f"uncommitted code changes: {dirty.splitlines()[:5]}")
    for rel, meta in inv["files"].items():
        p = artifacts / rel
        if not p.exists():
            problems.append(f"missing {rel}")
        elif sha256(p) != meta["sha256"]:
            problems.append(f"hash mismatch {rel}")
    now = runtime()
    for k, v in inv["runtime"].items():
        if now.get(k) != v:
            problems.append(f"runtime {k}: frozen {v}, now {now.get(k)}")
    if problems:
        print("FREEZE CHECK FAILED:\n  " + "\n  ".join(problems))
        raise SystemExit(1)
    print(f"freeze check OK: code {inv['code_sha'][:10]}, {len(inv['files'])} files, runtime matches")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "check"])
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--inventory", type=Path, default=REPO / "configs" / "step4" / "freeze_inventory.json")
    a = ap.parse_args()
    build(a.artifacts, a.out) if a.mode == "build" else check(a.artifacts, a.inventory)


if __name__ == "__main__":
    main()
