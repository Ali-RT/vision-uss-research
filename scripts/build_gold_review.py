"""Build gold-label review shards for the CUV test traces (labeling KPI ground truth).

Samples every --step-th frame within 8 m on the held-out test traces (same
trace split as notebook 17), splits them into one shard per reviewer, and gives
a share of frames (--overlap) to two reviewers to measure reviewer agreement.
Reviewers draw boxes from scratch - no pre-labels, so the gold set favours
neither pipeline. Each shard is a self-contained folder (index.html + images)
zipped for sending; reviewers open index.html in Chrome and export a CSV.

    uv run python scripts/build_gold_review.py --reviewers 4
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_uss_research.silver_gt import select_test_items  # noqa: E402

CATEGORY_TEXT = {
    "isopole": "ISO pole (75 mm)", "car": "balloon car", "wall": "wall",
    "dummyadult": "adult dummy", "dummychild": "child dummy",
    "box30": "box 30 cm", "box40": "box 40 cm", "curbhigh": "curbstone (high)",
    "woodencurb10": "wooden curb 10 cm", "metalbar": "horizontal metal bar",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", type=Path,
                        default=PROJECT_ROOT / "outputs/cuv_objectcheck/cuv_bundle_v1")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT /
                        "configs/experiments/cuv_finetune_v1.yaml")
    parser.add_argument("--name", default="gold_v1")
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--reviewers", type=int, default=4)
    parser.add_argument("--overlap", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefill", type=Path, default=None,
                        help="audit mode: silver.csv whose labels are shown as proposals")
    parser.add_argument("--sample", type=int, default=0,
                        help="audit mode: review only this many random prefilled frames")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    frames = pd.read_csv(args.bundle / "frames.csv")
    items = select_test_items(frames, cfg["split"], args.step)
    pre = {}
    if args.prefill is not None:
        sv = pd.read_csv(args.prefill)
        sv = sv[sv["status"].isin(["box", "not_visible"])]
        pre = {r.item_id: ({"status": "box", "x0": float(r.x0), "y0": float(r.y0),
                            "x1": float(r.x1), "y1": float(r.y1), "truncated": int(r.truncated)}
                           if r.status == "box" else {"status": r.status})
               for r in sv.itertuples()}
        items = items[items["item_id"].isin(pre)]
        if args.sample:
            items = items.sample(min(args.sample, len(items)), random_state=args.seed)

    # whole traces per reviewer (balanced by item count), then the overlap set
    rng = np.random.default_rng(args.seed)
    load = [0] * args.reviewers
    owner = {}
    for (side, trace), n in (items.groupby(["side", "trace"]).size()
                             .sort_values(ascending=False).items()):
        r = int(np.argmin(load))
        owner[(side, trace)] = r
        load[r] += n
    items["reviewer"] = [owner[(s, t)] for s, t in zip(items["side"], items["trace"])]
    dup = items.sample(frac=args.overlap, random_state=args.seed).copy() if args.overlap else items.iloc[:0].copy()
    dup["reviewer"] = (dup["reviewer"] + 1 + rng.integers(0, args.reviewers - 1, len(dup))) % args.reviewers
    dup["overlap"] = True
    items["overlap"] = items["item_id"].isin(dup["item_id"])
    allocation = pd.concat([items, dup], ignore_index=True)

    out = PROJECT_ROOT / "outputs" / "gold_review" / args.name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    template = (PROJECT_ROOT / "scripts" / "gold_review_template.html").read_text()
    for r in range(args.reviewers):
        shard = allocation[allocation["reviewer"] == r].sample(frac=1, random_state=r)
        sdir = out / f"shard_{r + 1}"
        (sdir / "images").mkdir(parents=True)
        recs = []
        for it in shard.itertuples():
            src = args.bundle / "frames" / f"{it.side}_{it.trace}" / f"{int(it.frame_idx):05d}.jpg"
            shutil.copyfile(src, sdir / "images" / f"{it.item_id}.jpg")
            recs.append({"id": it.item_id, "side": it.side, "trace": int(it.trace),
                         "frame": int(it.frame_idx), "category": it.label_class,
                         "what": CATEGORY_TEXT.get(it.label_class, it.label_class),
                         "dist": round(float(it.dist_m), 2),
                         **({"pre": pre[it.item_id]} if it.item_id in pre else {})})
        html = (template.replace("__SHARD__", f"{args.name} shard {r + 1}")
                .replace("__ITEMS__", json.dumps(recs)))
        (sdir / "index.html").write_text(html, encoding="utf-8")
        (sdir / "README.txt").write_text(
            f"Gold label review - shard {r + 1} ({len(recs)} frames)\n\n"
            "1. Unzip this folder anywhere.\n"
            "2. Open index.html in Chrome (double-click).\n"
            "3. Type your name at the top, then label every frame: drag a tight box\n"
            "   around the named object, or press N (not visible) / U (unsure).\n"
            "   The rules are on the right of the page.\n"
            "4. Work is saved in the browser as you go; you can close and come back.\n"
            "5. When done, press Export CSV and send the downloaded file back.\n")
        shutil.make_archive(str(out / f"{args.name}_shard_{r + 1}"), "zip",
                            root_dir=out, base_dir=f"shard_{r + 1}")
        print(f"shard {r + 1}: {len(recs)} frames")
    allocation.to_csv(out / "allocation.csv", index=False)
    print(f"\n{items['item_id'].nunique()} frames from {items.groupby(['side', 'trace']).ngroups} "
          f"test traces, {len(dup)} reviewed twice -> {out}")


if __name__ == "__main__":
    main()
