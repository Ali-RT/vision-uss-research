"""Gold-label review shards for the method-paper frames (PLAT_USS corpus).

Takes the frame lists the method is scored on (notebook 23/24 `scored.csv` or
`frames.csv`: sequence_id, camera, frame_idx, target_object, range_mm and the
pseudo-label box x0..y1) and builds review shards with the same HTML tool as
the CUV gold set. The ONLY proposal shown is the pseudo-label (key A): never a
Qwen/USS box, so the corrected gold favours neither arm of the comparison.
Accept (A), redraw, or press N (not visible). Already-reviewed frames can be
skipped with --exclude (a previous export).

Run where the frames are (Colab: --frames-root <artifacts>/frames/r30-85_n20):
    python scripts/build_plat_gold_review.py --frames-csv <scored.csv> \
        --frames-root <dir> --out <dir> --name plat_gold_v1 --per-shard 400
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

TEMPLATE = Path(__file__).resolve().parent / "gold_review_template.html"
README = """Gold label review - {name} shard {k} ({n} frames)

1. Unzip anywhere, open index.html in Chrome.
2. Type your name at the top.
3. Each frame names the STAGED obstacle the vehicle is approaching.
   The pseudo-label box is pre-filled (A accepts it as is).
   - Right object, tight box: press A (or just Next).
   - Wrong object or loose box (e.g. on the ego bumper strip, on a different
     parked car, on background): drag a new tight box.
   - Obstacle not in the image: N.  Unsure: U.  Cut by bottom/side edge: T.
4. Work is saved in the browser; Export CSV when done and send it back.
"""


def item_id(seq: str, cam: str, frame: int) -> str:
    return f"{seq}__{cam}__{int(frame):02d}"


def build_items(frames: pd.DataFrame) -> list[dict]:
    recs = []
    for r in frames.itertuples(index=False):
        rec = {"id": item_id(r.sequence_id, r.camera, r.frame_idx), "side": r.camera,
               "trace": int(str(r.sequence_id).split("_")[0]), "frame": int(r.frame_idx),
               "category": r.target_object,
               "what": f"the staged {r.target_object} the vehicle is approaching",
               "dist": round(float(r.range_mm) / 1000, 2) if pd.notna(r.range_mm) else None}
        if pd.notna(r.x0):
            rec.update(pre={"status": "box", "x0": float(r.x0), "y0": float(r.y0),
                            "x1": float(r.x1), "y1": float(r.y1), "truncated": 0},
                       preName="pseudo-label")
        recs.append(rec)
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames-csv", type=Path, nargs="+", required=True)
    ap.add_argument("--frames-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", default="plat_gold_v1")
    ap.add_argument("--per-shard", type=int, default=400)
    ap.add_argument("--exclude", type=Path, nargs="*", default=[],
                    help="earlier gold exports; their item_ids are skipped")
    a = ap.parse_args()

    cols = ["sequence_id", "camera", "frame_idx", "target_object", "range_mm",
            "x0", "y0", "x1", "y1"]
    frames = pd.concat([pd.read_csv(p, usecols=cols) for p in a.frames_csv])
    frames = frames.drop_duplicates(["sequence_id", "camera", "frame_idx"])
    done = set()
    for p in a.exclude:
        done |= set(pd.read_csv(p)["item_id"])
    frames["id"] = [item_id(s, c, f) for s, c, f in
                    zip(frames.sequence_id, frames.camera, frames.frame_idx)]
    frames = frames[~frames.id.isin(done)]
    # whole sequences stay in one shard; types interleave so a shard is a fair sample
    frames = frames.sort_values(["sequence_id", "frame_idx"])
    seqs = frames.sequence_id.drop_duplicates().sample(frac=1, random_state=0).tolist()
    order = {s: i for i, s in enumerate(seqs)}
    frames = frames.assign(o=frames.sequence_id.map(order)).sort_values(["o", "frame_idx"])

    out = a.out / a.name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    template = TEMPLATE.read_text()
    shards, cur, cur_seqs = [], [], 0
    for _, g in frames.groupby("o", sort=True):
        if cur and sum(len(x) for x in cur) + len(g) > a.per_shard:
            shards.append(pd.concat(cur)); cur = []
        cur.append(g)
    if cur:
        shards.append(pd.concat(cur))
    alloc = []
    for k, sh in enumerate(shards, 1):
        sdir = out / f"shard_{k}"
        (sdir / "images").mkdir(parents=True)
        missing = 0
        keep = []
        for r in sh.itertuples(index=False):
            src = a.frames_root / r.sequence_id / r.camera / f"{int(r.frame_idx):05d}.jpg"
            if not src.exists():
                missing += 1; continue
            shutil.copyfile(src, sdir / "images" / f"{r.id}.jpg")
            keep.append(r)
        recs = build_items(pd.DataFrame(keep))
        html = (template.replace("__SHARD__", f"{a.name} shard {k}")
                .replace("__ITEMS__", json.dumps(recs)))
        (sdir / "index.html").write_text(html, encoding="utf-8")
        (sdir / "README.txt").write_text(README.format(name=a.name, k=k, n=len(recs)))
        shutil.make_archive(str(out / f"{a.name}_shard_{k}"), "zip", root_dir=out,
                            base_dir=f"shard_{k}")
        alloc.append(pd.DataFrame(keep).assign(shard=k))
        print(f"shard {k}: {len(recs)} frames ({sh.sequence_id.nunique()} seqs)"
              + (f", {missing} frames missing on disk" if missing else ""))
    pd.concat(alloc).to_csv(out / "allocation.csv", index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
