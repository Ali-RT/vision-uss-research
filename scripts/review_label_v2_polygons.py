"""Contact sheets for the manual check of Label V2 "empty" polygons (step 1).

For a fixed sample of NON-reserved sequences (no reserved imagery is opened),
one PNG each:
  left  - every Label V2 polygon in its map frame ("empty" green, objects red,
          with object name and class code);
  right - USS topview, front and rear video frames at 30 / 55 / 85 % of the
          clip. The map-frame -> topview pixel transform is NOT known, so the
          polygons are shown beside the topview, not overlaid; alignment is
          judged visually and recorded as such.
plus review_sheet.csv with blank columns for the reviewer.

Strata (seed 0, at most one sequence per recording day within a stratum):
  mixed_code0  - "empty" (code 0) + an object
  empty_code3  - an "empty" row with code 3
  empty_code5  - an "empty" row with the undocumented code 5
  only_rough   - empty-only, rough-road scene
  only_other   - empty-only, scene NOT rough road (possible missing object)

usage (Colab, Drive mounted):
  python scripts/review_label_v2_polygons.py --raw-root RAW \
      --audit OUT/reports/audit_v1 \
      --reserved configs/splits/reserved_sessions_v1.csv --out OUT/reports/audit_v1/polygon_review
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vision_uss_research.sequences import sequence_files  # noqa: E402

PER_STRATUM = {"mixed_code0": 2, "empty_code3": 2, "empty_code5": 2,
               "only_rough": 2, "only_other": 4}
SEED = 0


def polygons(seq_dir: Path) -> list[dict]:
    csv = next(iter(sorted(seq_dir.glob("*_PAS_m_label_V2.csv"))), None)
    if csv is None:
        return []
    df = pd.read_csv(csv, encoding="utf-8-sig")
    code_col = [c for c in df.columns if c.startswith("class (")][0]
    out = []
    for _, r in df.iterrows():
        try:
            xs = [float(v) for v in str(r["PolygonX"]).split(",")]
            ys = [float(v) for v in str(r["PolygonY"]).split(",")]
        except ValueError:
            continue
        out.append({"name": str(r["Which object"]).strip(), "code": r[code_col], "x": xs, "y": ys})
    return out


def frames(video: Path, ratios=(0.30, 0.55, 0.85)):
    import cv2
    cap = cv2.VideoCapture(str(video))
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    imgs = []
    for r in ratios:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(r * max(n - 1, 0)))
        ok, img = cap.read()
        imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if ok else None)
    cap.release()
    return imgs


def sheet(seq_dir: Path, out_png: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = sequence_files(seq_dir)
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_axes([0.02, 0.08, 0.28, 0.82])
    for p in polygons(seq_dir):
        empty = p["name"].lower() == "empty"
        ax.fill(p["x"], p["y"], alpha=0.25, color="green" if empty else "red")
        ax.plot(p["x"] + p["x"][:1], p["y"] + p["y"][:1], color="green" if empty else "red")
        ax.text(sum(p["x"]) / len(p["x"]), sum(p["y"]) / len(p["y"]),
                f"{p['name']} ({p['code']})", fontsize=8, ha="center")
    ax.set_aspect("equal")
    ax.set_title("Label V2 polygons (map frame, mm)")
    views = [("topview", next(iter(sorted(seq_dir.glob("VIDEO_USS_TOPVIEW.*"))), None)),
             ("front", files.get("front_video")), ("rear", files.get("rear_video"))]
    for row, (cam, video) in enumerate(views):
        imgs = frames(video) if video else [None] * 3
        for col, img in enumerate(imgs):
            a = fig.add_axes([0.32 + col * 0.225, 0.66 - row * 0.31, 0.22, 0.28])
            a.axis("off")
            if img is not None:
                a.imshow(img)
            a.set_title(f"{cam} {(0.30, 0.55, 0.85)[col]:.0%}", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.savefig(out_png, dpi=80)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--inventory", type=Path, default=None,
                    help="inventory_metadata.csv; default: newest under <outputs>/inventory/*/ "
                         "or <outputs>/profiles/*/, where <outputs> is two levels above --audit")
    ap.add_argument("--reserved", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    lv = pd.read_csv(a.audit / "label_v2_objects.csv", dtype={"object": str})
    lv = lv[lv.error.fillna("") == ""]
    lv["name"] = lv.object.fillna("").str.lower().str.strip()
    reserved = set(pd.read_csv(a.reserved).sequence_id)
    lv = lv[~lv.sequence_id.isin(reserved)]
    sets = lv.groupby("sequence_id")["name"].agg(lambda s: "|".join(sorted(set(s))))
    inv_path = a.inventory
    if inv_path is None or not inv_path.exists():
        outputs = a.audit.resolve().parents[1]
        found = sorted(list(outputs.glob("inventory/*/inventory_metadata.csv"))
                       + list(outputs.glob("profiles/*/inventory_metadata.csv")))
        if not found:
            sys.exit(f"inventory_metadata.csv not found (given: {a.inventory}; searched {outputs})")
        inv_path = found[-1]
    print(f"inventory: {inv_path}")
    inv = pd.read_csv(inv_path).set_index("sequence_id")

    emp = lv[lv.name == "empty"]
    scene = inv.scene.reindex(sets.index).fillna("")
    only = sets == "empty"
    pools = {
        "mixed_code0": sets.index[sets.str.contains("empty") & ~only
                                  & sets.index.isin(emp[emp.class_code == 0].sequence_id)],
        "empty_code3": sets.index[sets.index.isin(emp[emp.class_code == 3].sequence_id)],
        "empty_code5": sets.index[sets.index.isin(emp[emp.class_code == 5].sequence_id)],
        "only_rough": sets.index[only & scene.str.contains("Rough_road")],
        "only_other": sets.index[only & ~scene.str.contains("Rough_road")],
    }
    picks = []
    for stratum, pool in pools.items():
        cand = pd.Series(pool).sample(frac=1, random_state=SEED)
        one_per_day = cand.groupby(cand.str.split("_").str[1]).head(1)
        picks += [(s, stratum) for s in one_per_day.head(PER_STRATUM[stratum])]

    rows = []
    for seq, group in picks:
        title = f"{seq} | {group} | scene: {inv.scene.get(seq, '')} | objects: {sets[seq]}"
        try:
            sheet(a.raw_root / seq, a.out / f"{seq}.png", title)
            err = ""
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"[:200]
        rows.append({"sequence_id": seq, "group": group, "scene": inv.scene.get(seq, ""),
                     "label_v2_objects": sets[seq], "day": seq.split("_")[1], "error": err,
                     "alignment_polygons_vs_topview": "", "empty_is_drivable_area_beside_object": "",
                     "visible_obstacle_in_frames": "",
                     "missing_annotation_suspected": "", "reviewer": "", "notes": ""})
    pd.DataFrame(rows).to_csv(a.out / "review_sheet.csv", index=False)
    print(f"{len(rows)} sheets -> {a.out} | errors {sum(bool(r['error']) for r in rows)}")


if __name__ == "__main__":
    main()
