"""Metadata-only audit for the reserved test sessions and Label V2 semantics.

Opens NO images and runs NO model. It reads CSV/JSON files only:

1. Reserved sessions (configs/splits/reserved_sessions_v1.csv): for each ID,
   whether any artifact already mentions it - an extracted frame folder, a
   SAM3/click label run, a dataset manifest, a method or report CSV. This
   documents prior exposure before the protocol is frozen.
2. Label V2 (every raw sequence's *_PAS_m_label_V2.csv): one row per labeled
   object with its object name, class code (0=traversable, 1=low, 2=unknown,
   3=high), whether its geometry is a point or a polygon, and the database
   path tokens (category folder, e.g. D12_Bicycle_stand, and 'TypeN'). This
   shows what "empty" means and whether the type-derived Low/High mapping
   agrees with the annotators' own class code.

usage (Colab, Drive mounted):
  python scripts/audit_reserved_and_label_v2.py --raw-root RAW --artifacts OUT \
      --reserved configs/splits/reserved_sessions_v1.csv --out OUT/reports/audit_v1
"""
from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

CLASS_COL = "class (0=traversable, 1=low, 2=unknown, 3=high)"


def label_v2_rows(seq_dir: Path) -> list[dict]:
    csvs = sorted(p for p in seq_dir.glob("*_PAS_m_label_V2.csv"))
    if not csvs:
        return [{"sequence_id": seq_dir.name, "error": "no_label_v2"}]
    try:
        df = pd.read_csv(csvs[0], encoding="utf-8-sig")
    except Exception as e:  # noqa: BLE001 - record and continue
        return [{"sequence_id": seq_dir.name, "error": f"{type(e).__name__}"}]
    rows = []
    for _, r in df.iterrows():
        db = str(r.get("NAMEoFFILE", ""))
        parts = re.split(r"[\\/]", db)
        cat = next((p for p in parts if re.match(r"^[A-Z]\d+_", p)), "")
        typ = next((p for p in parts if re.match(r"^Type\d+$", p, re.I)), "")
        point = (r.get("xmin") == r.get("xmax")) and (r.get("ymin") == r.get("ymax"))
        rows.append({"sequence_id": seq_dir.name, "error": "",
                     "object": str(r.get("Which object", "")).strip(),
                     "class_code": r.get(CLASS_COL),
                     "important_distance_mm": r.get("important Distance [mm]"),
                     "bbox_is_point": bool(point),
                     "has_polygon": isinstance(r.get("PolygonX"), str) and "," in r.get("PolygonX"),
                     "db_category": cat, "db_type": typ})
    return rows


def exposure(reserved: list[str], artifacts: Path) -> pd.DataFrame:
    """Which artifact files mention each reserved ID (CSV text search, no images)."""
    hits = {s: [] for s in reserved}
    frames = artifacts / "frames"
    for win in (frames.glob("*") if frames.exists() else []):
        for s in reserved:
            if (win / s).exists():
                hits[s].append(f"frames/{win.name}")
    for csv in list(artifacts.glob("labels/runs/*/*.csv")) + \
            list(artifacts.glob("experiments/*/propagated_boxes.csv")) + \
            list(artifacts.glob("datasets/*/dataset_manifest.csv")) + \
            list(artifacts.glob("method/*/*.csv")) + list(artifacts.glob("reports/*/*.csv")):
        try:
            text = csv.read_text(errors="ignore")
        except OSError:
            continue
        for s in reserved:
            if s in text:
                hits[s].append(str(csv.relative_to(artifacts)))
    return pd.DataFrame([{"sequence_id": s, "n_hits": len(h), "hits": "|".join(sorted(set(h)))}
                         for s, h in hits.items()])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--reserved", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    reserved = pd.read_csv(a.reserved).sequence_id.tolist()
    exp = exposure(reserved, a.artifacts)
    exp.to_csv(a.out / "reserved_exposure.csv", index=False)
    print(f"reserved {len(reserved)} | mentioned by any artifact: {(exp.n_hits > 0).sum()}")
    print(exp.hits.str.split("|").explode().replace("", pd.NA).dropna()
          .str.replace(r"/[^/]+$", "", regex=True).value_counts().to_string())

    seq_dirs = [p for p in a.raw_root.iterdir() if p.is_dir()]
    with ThreadPoolExecutor(a.workers) as pool:
        rows = [r for rs in pool.map(label_v2_rows, seq_dirs) for r in rs]
    lv = pd.DataFrame(rows)
    lv.to_csv(a.out / "label_v2_objects.csv", index=False)
    ok = lv[lv.error == ""]
    print(f"\nLabel V2: {lv.sequence_id.nunique()} sequences, {len(ok)} object rows, "
          f"errors {lv.loc[lv.error != '', 'error'].value_counts().to_dict()}")
    print("\nclass code by object (top 25):")
    print(pd.crosstab(ok.object.str.lower(), ok.class_code).loc[
        ok.object.str.lower().value_counts().head(25).index].to_string())
    e = ok[ok.object.str.lower() == "empty"]
    print(f"\n'empty' rows {len(e)}: class codes {e.class_code.value_counts().to_dict()} | "
          f"point bbox {e.bbox_is_point.mean():.2f} | polygon {e.has_polygon.mean():.2f}")
    print("\ndb_type values:", ok.db_type.value_counts().head(10).to_dict())
    print("db_category (top 15):", ok.db_category.value_counts().head(15).to_dict())


if __name__ == "__main__":
    main()
