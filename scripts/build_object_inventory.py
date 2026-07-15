"""Build an inventory of all object/label names across all sequences.

Iterates over every sequence in the manifest, collects object names from two sources
(metadata JSON "Label Objects" tags and Label V2 "Which object" rows), aggregates
per-object statistics (sequence counts, Label V2 class distribution, distance stats),
and matches each name against the height catalog in
configs/datasets/object_height_catalog.csv.

Outputs (under outputs/profiles/object_inventory/):
    inventory_sequences.csv  one row per sequence with its objects from both sources
    inventory_objects.csv    one row per unique object name with aggregate stats
    inventory_summary.csv    overview counts, per-source object counts, unmatched names
    inventory_errors.csv     sequences with missing/unreadable files

Usage:
    python scripts/build_object_inventory.py --profile colab_drive
    python scripts/build_object_inventory.py --profile local --limit 20
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vision_uss_research.settings import load_paths

LABEL_OBJECTS_SUBCATEGORY = "label objects"
CLASS_ID_TO_NAME = {"0": "traversable", "1": "low", "2": "unknown", "3": "high"}


def load_csv_dict(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        out_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_read_label_csv(path: Path) -> pd.DataFrame:
    attempts = [
        {"encoding": "utf-8-sig", "sep": ","},
        {"encoding": "utf-8", "sep": ","},
        {"encoding": "latin1", "sep": ","},
        {"encoding": "utf-8-sig", "sep": ";"},
        {"encoding": "latin1", "sep": ";"},
    ]

    last_error: Exception | None = None

    for kwargs in attempts:
        try:
            df = pd.read_csv(path, **kwargs)
            if len(df.columns) > 1:
                return df
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Could not read label CSV: {path}. Last error: {last_error}")


def norm_col(name: str) -> str:
    return str(name).strip().lower()


def find_column(columns: list[str], candidates: list[str]) -> str:
    norm_to_original = {norm_col(c): c for c in columns}

    for candidate in candidates:
        key = norm_col(candidate)
        if key in norm_to_original:
            return norm_to_original[key]

    for col in columns:
        low = norm_col(col)
        for candidate in candidates:
            if norm_col(candidate) in low:
                return col

    return ""


def normalize_object_name(name: str) -> str:
    """Normalization key for matching object names across sources and the catalog."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def load_height_catalog(catalog_csv: Path) -> dict[str, dict[str, str]]:
    """Map normalized object name (and every alias) -> catalog row."""
    key_to_row: dict[str, dict[str, str]] = {}

    if not catalog_csv.exists():
        return key_to_row

    for row in load_csv_dict(catalog_csv):
        names = [row.get("object_name", "")]
        names += str(row.get("aliases", "")).split("|")
        for name in names:
            key = normalize_object_name(name)
            if key:
                key_to_row.setdefault(key, row)

    return key_to_row


def extract_json_tags(metadata_path: Path) -> dict[str, list[str]]:
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    tags_by_subcategory: dict[str, list[str]] = defaultdict(list)
    for tag in metadata.get("tags", []):
        subcategory = str(tag.get("tagSubCategory", {}).get("name", "")).strip().lower()
        name = str(tag.get("name", "")).strip()
        if subcategory and name:
            tags_by_subcategory[subcategory].append(name)

    return dict(tags_by_subcategory)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory object/label names across all sequences"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Path profile name, e.g. local or colab_drive",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Default: manifests_dir/sequence_manifest.csv",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=PROJECT_ROOT / "configs" / "datasets" / "object_height_catalog.csv",
        help="Object height catalog CSV",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Optional override for the raw data root (default from profile)",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_paths(profile=args.profile)

    manifest_csv = (
        args.manifest.resolve()
        if args.manifest
        else (paths.manifests_dir / "sequence_manifest.csv").resolve()
    )
    raw_root = args.raw_root.resolve() if args.raw_root else paths.raw_data_root

    manifest_rows = load_csv_dict(manifest_csv)
    if args.limit and args.limit > 0:
        manifest_rows = manifest_rows[: args.limit]

    catalog = load_height_catalog(args.catalog)

    out_root = paths.outputs_dir / "profiles" / "object_inventory"

    sequence_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    # per normalized object name
    display_name: dict[str, str] = {}
    json_sequences: dict[str, set[str]] = defaultdict(set)
    label_v2_sequences: dict[str, set[str]] = defaultdict(set)
    label_v2_row_counts: Counter[str] = Counter()
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    distances_mm: dict[str, list[float]] = defaultdict(list)

    iterator = manifest_rows
    if not args.no_progress:
        iterator = tqdm(manifest_rows, desc="Inventorying sequences", unit="seq")

    for row in iterator:
        sequence_id = row.get("sequence_id", "")
        json_objects: list[str] = []
        v2_objects: list[str] = []
        v2_classes: list[str] = []
        v2_distances: list[str] = []
        seq_errors: list[str] = []

        # Source 1: metadata JSON "Label Objects" tags
        metadata_rel = str(row.get("metadata_json", "") or "").strip()
        if metadata_rel:
            metadata_path = raw_root / metadata_rel
            if metadata_path.exists():
                try:
                    tags = extract_json_tags(metadata_path)
                    json_objects = tags.get(LABEL_OBJECTS_SUBCATEGORY, [])
                except Exception as e:
                    seq_errors.append(f"metadata_json:{type(e).__name__}")
            else:
                seq_errors.append("metadata_json_not_found")
        else:
            seq_errors.append("missing_metadata_json_path")

        for name in json_objects:
            key = normalize_object_name(name)
            if not key:
                continue
            display_name.setdefault(key, name)
            json_sequences[key].add(sequence_id)

        # Source 2: Label V2 "Which object" rows
        label_rel = str(row.get("label_v2_csv", "") or "").strip()
        if label_rel:
            label_path = raw_root / label_rel
            if label_path.exists():
                try:
                    df = safe_read_label_csv(label_path)
                    columns = list(df.columns)
                    object_col = find_column(columns, ["Which object", "object"])
                    class_col = find_column(columns, ["class"])
                    distance_col = find_column(
                        columns, ["important Distance [mm]", "important distance", "distance"]
                    )

                    for _, label_row in df.iterrows():
                        name = str(label_row[object_col]).strip() if object_col else ""
                        if not name or name.lower() == "nan":
                            continue
                        key = normalize_object_name(name)
                        display_name.setdefault(key, name)
                        label_v2_sequences[key].add(sequence_id)
                        label_v2_row_counts[key] += 1
                        v2_objects.append(name)

                        if class_col:
                            class_value = str(label_row[class_col]).strip()
                            class_value = class_value.split(".")[0]  # "3.0" -> "3"
                            if class_value and class_value.lower() != "nan":
                                class_counts[key][class_value] += 1
                                v2_classes.append(class_value)

                        if distance_col:
                            distance = pd.to_numeric(label_row[distance_col], errors="coerce")
                            if pd.notna(distance):
                                distances_mm[key].append(float(distance))
                                v2_distances.append(str(int(distance)))
                except Exception as e:
                    seq_errors.append(f"label_v2:{type(e).__name__}")
            else:
                seq_errors.append("label_v2_csv_not_found")
        else:
            seq_errors.append("missing_label_v2_csv_path")

        sequence_rows.append(
            {
                "sequence_id": sequence_id,
                "sequence_dir": row.get("sequence_dir", ""),
                "json_label_objects": "|".join(json_objects),
                "label_v2_objects": "|".join(v2_objects),
                "label_v2_classes": "|".join(v2_classes),
                "label_v2_distances_mm": "|".join(v2_distances),
                "num_label_v2_rows": len(v2_objects),
                "errors": "|".join(seq_errors),
            }
        )

        if seq_errors:
            error_rows.append(
                {
                    "sequence_id": sequence_id,
                    "sequence_dir": row.get("sequence_dir", ""),
                    "errors": "|".join(seq_errors),
                }
            )

    # Aggregate per object
    object_rows: list[dict[str, Any]] = []
    all_keys = sorted(
        set(json_sequences) | set(label_v2_sequences),
        key=lambda k: -(len(json_sequences[k]) + len(label_v2_sequences[k])),
    )

    for key in all_keys:
        catalog_row = catalog.get(key)
        dists = sorted(distances_mm[key])
        classes = class_counts[key]
        class_dist = "|".join(
            f"{CLASS_ID_TO_NAME.get(class_id, class_id)}:{count}"
            for class_id, count in sorted(classes.items())
        )

        object_rows.append(
            {
                "object_name": display_name[key],
                "normalized_key": key,
                "n_sequences_json_tag": len(json_sequences[key]),
                "n_sequences_label_v2": len(label_v2_sequences[key]),
                "n_sequences_total": len(json_sequences[key] | label_v2_sequences[key]),
                "n_label_v2_rows": label_v2_row_counts[key],
                "label_v2_class_distribution": class_dist,
                "distance_mm_min": int(dists[0]) if dists else "",
                "distance_mm_median": int(dists[len(dists) // 2]) if dists else "",
                "distance_mm_max": int(dists[-1]) if dists else "",
                "catalog_matched": int(catalog_row is not None),
                "height_cm_min": catalog_row.get("height_cm_min", "") if catalog_row else "",
                "height_cm_max": catalog_row.get("height_cm_max", "") if catalog_row else "",
                "height_bin": catalog_row.get("height_bin", "") if catalog_row else "",
                "catalog_notes": catalog_row.get("notes", "") if catalog_row else "",
            }
        )

    unmatched = [r for r in object_rows if not r["catalog_matched"]]

    summary_rows: list[dict[str, Any]] = [
        {"group": "overview", "value": "manifest_rows", "count": len(manifest_rows)},
        {"group": "overview", "value": "sequences_with_errors", "count": len(error_rows)},
        {"group": "overview", "value": "unique_objects", "count": len(object_rows)},
        {"group": "overview", "value": "objects_unmatched_in_catalog", "count": len(unmatched)},
    ]
    for r in object_rows:
        summary_rows.append(
            {"group": "objects_by_sequence_count", "value": r["object_name"], "count": r["n_sequences_total"]}
        )
    for r in unmatched:
        summary_rows.append(
            {"group": "unmatched_objects", "value": r["object_name"], "count": r["n_sequences_total"]}
        )

    bin_counter: Counter[str] = Counter()
    for r in object_rows:
        bin_counter[r["height_bin"] or "UNMAPPED"] += r["n_sequences_total"]
    for value, count in bin_counter.most_common():
        summary_rows.append({"group": "sequences_by_height_bin", "value": value, "count": count})

    write_csv(sequence_rows, out_root / "inventory_sequences.csv")
    write_csv(object_rows, out_root / "inventory_objects.csv")
    write_csv(summary_rows, out_root / "inventory_summary.csv")
    write_csv(error_rows, out_root / "inventory_errors.csv")

    print(f"profile              : {paths.profile_name}")
    print(f"manifest_csv         : {manifest_csv}")
    print(f"catalog_csv          : {args.catalog}")
    print(f"output_root          : {out_root}")
    print(f"manifest_rows        : {len(manifest_rows)}")
    print(f"sequences_with_errors: {len(error_rows)}")
    print(f"unique_objects       : {len(object_rows)}")
    print(f"unmatched_in_catalog : {len(unmatched)}")
    print()
    print("Top objects by sequence count:")
    for r in object_rows[:30]:
        bin_note = r["height_bin"] or "UNMAPPED"
        print(
            f"  {r['object_name']:<35} seqs={r['n_sequences_total']:<6} "
            f"bin={bin_note:<8} classes=[{r['label_v2_class_distribution']}]"
        )
    if unmatched:
        print()
        print("Objects missing from the height catalog (add them to "
              "configs/datasets/object_height_catalog.csv):")
        for r in unmatched:
            print(f"  {r['object_name']} ({r['n_sequences_total']} sequences)")


if __name__ == "__main__":
    main()
