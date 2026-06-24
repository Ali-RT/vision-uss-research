from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vision_uss_research.settings import load_paths


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with out_path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text(text: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def normalize_tag_value(value: Any) -> str:
    return str(value or "").strip()


def extract_tags(payload: dict[str, Any]) -> list[dict[str, str]]:
    tags_out: list[dict[str, str]] = []

    for tag in payload.get("tags", []) or []:
        tag_name = normalize_tag_value(tag.get("name"))
        tag_id = normalize_tag_value(tag.get("id"))

        category = tag.get("tagCategory", {}) or {}
        subcategory = tag.get("tagSubCategory", {}) or {}

        category_name = normalize_tag_value(category.get("name"))
        category_id = normalize_tag_value(category.get("id"))

        subcategory_name = normalize_tag_value(subcategory.get("name"))
        subcategory_id = normalize_tag_value(subcategory.get("id"))

        tags_out.append(
            {
                "tag_id": tag_id,
                "tag_name": tag_name,
                "category_id": category_id,
                "category_name": category_name,
                "subcategory_id": subcategory_id,
                "subcategory_name": subcategory_name,
            }
        )

    return tags_out


def extract_custom_fields(payload: dict[str, Any]) -> list[dict[str, str]]:
    fields_out: list[dict[str, str]] = []

    for item in payload.get("customFieldValues", []) or []:
        field = item.get("field", {}) or {}

        fields_out.append(
            {
                "field_id": normalize_tag_value(field.get("id")),
                "field_name": normalize_tag_value(field.get("name")),
                "field_type": normalize_tag_value(field.get("type")),
                "value": safe_str(item.get("value")),
            }
        )

    return fields_out


def canonical_subcategory_name(name: str) -> str:
    name = str(name or "").strip()
    if "." in name:
        name = name.split(".")[-1].strip()
    return name


def values_for_subcategory(tags: list[dict[str, str]], subcategory_name: str) -> list[str]:
    target = subcategory_name.strip().lower()
    values = []
    for tag in tags:
        subcat = canonical_subcategory_name(tag["subcategory_name"]).lower()
        if subcat == target:
            if tag["tag_name"]:
                values.append(tag["tag_name"])
    return values


def pipe_join(values: list[str]) -> str:
    out = []
    for value in values:
        value = str(value).strip()
        if value and value not in out:
            out.append(value)
    return "|".join(out)


def infer_preferred_camera_from_direction(direction: str) -> str:
    value = direction.strip().lower()

    if value in {"forward", "forwards"}:
        return "front"

    if value in {"backward", "backwards", "reverse", "rearward"}:
        return "rear"

    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile all JSON metadata files")
    parser.add_argument("--profile", type=str, default="colab_drive")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    paths = load_paths(profile=args.profile)

    manifest_csv = (
        args.manifest.resolve()
        if args.manifest
        else (paths.manifests_dir / "sequence_manifest.csv").resolve()
    )

    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}")

    manifest_rows = load_csv(manifest_csv)

    if args.limit and args.limit > 0:
        manifest_rows = manifest_rows[: args.limit]

    out_root = paths.outputs_dir / "profiles" / "json_metadata"
    out_root.mkdir(parents=True, exist_ok=True)

    top_level_key_counter = Counter()
    top_level_key_nonempty_counter = Counter()
    tag_counter = Counter()
    subcategory_counter = Counter()
    category_counter = Counter()
    category_subcategory_counter = Counter()
    custom_field_counter = Counter()
    custom_field_value_counter = Counter()

    driving_direction_counter = Counter()
    preferred_camera_counter = Counter()
    weather_counter = Counter()
    scene_counter = Counter()
    approach_counter = Counter()
    label_object_counter = Counter()
    status_counter = Counter()

    flat_rows: list[dict[str, Any]] = []
    tag_rows: list[dict[str, Any]] = []
    custom_field_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    json_found = 0
    json_loaded = 0

    for row in tqdm(manifest_rows, desc="Profiling JSON metadata", unit="seq"):
        sequence_id = row.get("sequence_id", "")
        metadata_rel = (row.get("metadata_json") or "").strip()

        if not metadata_rel:
            error_rows.append(
                {
                    "sequence_id": sequence_id,
                    "metadata_json": "",
                    "error": "missing_metadata_path",
                }
            )
            continue

        metadata_path = paths.raw_data_root / metadata_rel

        if not metadata_path.exists():
            error_rows.append(
                {
                    "sequence_id": sequence_id,
                    "metadata_json": metadata_rel,
                    "error": "metadata_file_not_found",
                }
            )
            continue

        json_found += 1

        try:
            payload = load_json(metadata_path)
            json_loaded += 1
        except Exception as e:
            error_rows.append(
                {
                    "sequence_id": sequence_id,
                    "metadata_json": metadata_rel,
                    "error": f"{type(e).__name__}: {e}",
                }
            )
            continue

        for key, value in payload.items():
            top_level_key_counter[key] += 1
            if value not in [None, "", [], {}]:
                top_level_key_nonempty_counter[key] += 1

        tags = extract_tags(payload)
        custom_fields = extract_custom_fields(payload)

        for tag in tags:
            tag_counter[tag["tag_name"]] += 1
            subcategory_counter[tag["subcategory_name"]] += 1
            category_counter[tag["category_name"]] += 1
            category_subcategory_counter[
                (tag["category_name"], tag["subcategory_name"])
            ] += 1

            tag_rows.append(
                {
                    "sequence_id": sequence_id,
                    "metadata_json": metadata_rel,
                    **tag,
                }
            )

        for field in custom_fields:
            custom_field_counter[field["field_name"]] += 1
            if field["value"]:
                custom_field_value_counter[(field["field_name"], field["value"])] += 1

            custom_field_rows.append(
                {
                    "sequence_id": sequence_id,
                    "metadata_json": metadata_rel,
                    **field,
                }
            )

        driving_direction = pipe_join(values_for_subcategory(tags, "Driving Direction"))
        weather_tags = pipe_join(values_for_subcategory(tags, "Weather Conditions"))
        scene_tags = pipe_join(values_for_subcategory(tags, "Scene"))
        approach = pipe_join(values_for_subcategory(tags, "Approach"))
        label_objects = pipe_join(values_for_subcategory(tags, "Label Objects"))

        preferred_camera = infer_preferred_camera_from_direction(driving_direction)

        driving_direction_counter[driving_direction or "MISSING"] += 1
        preferred_camera_counter[preferred_camera] += 1
        status_counter[safe_str(payload.get("status")) or "MISSING"] += 1

        for value in weather_tags.split("|"):
            if value:
                weather_counter[value] += 1

        for value in scene_tags.split("|"):
            if value:
                scene_counter[value] += 1

        for value in approach.split("|"):
            if value:
                approach_counter[value] += 1

        for value in label_objects.split("|"):
            if value:
                label_object_counter[value] += 1

        avg_speed = ""
        for field in custom_fields:
            if field["field_name"].strip().lower() == "avg speed":
                avg_speed = field["value"]
                break

        flat_rows.append(
            {
                "sequence_id": sequence_id,
                "metadata_json": metadata_rel,
                "json_name": safe_str(payload.get("name")),
                "json_status": safe_str(payload.get("status")),
                "driving_direction": driving_direction,
                "preferred_camera_from_json": preferred_camera,
                "approach": approach,
                "weather_tags": weather_tags,
                "scene_tags": scene_tags,
                "label_objects_json": label_objects,
                "max_speed_kph": safe_str(payload.get("maxSpeedInKph")),
                "min_speed_kph": safe_str(payload.get("minSpeedInKph")),
                "avg_speed": avg_speed,
                "num_tags": len(tags),
                "num_custom_fields": len(custom_fields),
                "top_level_keys": pipe_join(sorted(payload.keys())),
            }
        )

    overview_rows = [
        {"metric": "manifest_rows", "value": len(manifest_rows)},
        {"metric": "json_found", "value": json_found},
        {"metric": "json_loaded", "value": json_loaded},
        {"metric": "json_errors", "value": len(error_rows)},
        {"metric": "flat_rows_written", "value": len(flat_rows)},
    ]

    top_level_rows = []
    for key, count in top_level_key_counter.most_common():
        top_level_rows.append(
            {
                "key": key,
                "present_count": count,
                "nonempty_count": top_level_key_nonempty_counter[key],
            }
        )

    tag_summary_rows = []
    for tag, count in tag_counter.most_common():
        tag_summary_rows.append({"tag_name": tag, "count": count})

    subcategory_summary_rows = []
    for subcat, count in subcategory_counter.most_common():
        subcategory_summary_rows.append({"subcategory_name": subcat, "count": count})

    category_subcategory_rows = []
    for (category, subcategory), count in category_subcategory_counter.most_common():
        category_subcategory_rows.append(
            {
                "category_name": category,
                "subcategory_name": subcategory,
                "count": count,
            }
        )

    custom_field_summary_rows = []
    for field, count in custom_field_counter.most_common():
        custom_field_summary_rows.append({"field_name": field, "count": count})

    custom_field_value_rows = []
    for (field, value), count in custom_field_value_counter.most_common(500):
        custom_field_value_rows.append(
            {
                "field_name": field,
                "value": value,
                "count": count,
            }
        )

    def counter_to_rows(group: str, counter: Counter, top_n: int | None = None) -> list[dict[str, Any]]:
        return [
            {"group": group, "value": value, "count": count}
            for value, count in counter.most_common(top_n)
        ]

    key_profile_rows: list[dict[str, Any]] = []
    key_profile_rows.extend(counter_to_rows("driving_direction", driving_direction_counter))
    key_profile_rows.extend(counter_to_rows("preferred_camera_from_json", preferred_camera_counter))
    key_profile_rows.extend(counter_to_rows("status", status_counter))
    key_profile_rows.extend(counter_to_rows("weather", weather_counter, 100))
    key_profile_rows.extend(counter_to_rows("scene", scene_counter, 100))
    key_profile_rows.extend(counter_to_rows("approach", approach_counter, 100))
    key_profile_rows.extend(counter_to_rows("label_objects", label_object_counter, 200))

    write_csv(overview_rows, out_root / "json_metadata_overview.csv")
    write_csv(top_level_rows, out_root / "json_top_level_key_summary.csv")
    write_csv(tag_summary_rows, out_root / "json_tag_summary.csv")
    write_csv(subcategory_summary_rows, out_root / "json_tag_subcategory_summary.csv")
    write_csv(category_subcategory_rows, out_root / "json_category_subcategory_summary.csv")
    write_csv(custom_field_summary_rows, out_root / "json_custom_field_summary.csv")
    write_csv(custom_field_value_rows, out_root / "json_custom_field_value_examples.csv")
    write_csv(key_profile_rows, out_root / "json_key_profile_summary.csv")
    write_csv(flat_rows, out_root / "json_sequence_metadata_flat.csv")
    write_csv(tag_rows, out_root / "json_all_tags_long.csv")
    write_csv(custom_field_rows, out_root / "json_all_custom_fields_long.csv")
    write_csv(error_rows, out_root / "json_metadata_errors.csv")

    report = f"""# JSON Metadata Profile

## Overview

- manifest rows: {len(manifest_rows)}
- JSON paths found: {json_found}
- JSON loaded: {json_loaded}
- JSON errors: {len(error_rows)}
- flat metadata rows written: {len(flat_rows)}

## Preferred Camera from JSON Driving Direction

{preferred_camera_counter}

## Driving Direction

{driving_direction_counter}

## Top Weather Tags

{weather_counter.most_common(20)}

## Top Scene Tags

{scene_counter.most_common(20)}

## Top Approach Tags

{approach_counter.most_common(20)}

## Top Label Objects

{label_object_counter.most_common(50)}

## Output Files

- json_metadata_overview.csv
- json_top_level_key_summary.csv
- json_tag_summary.csv
- json_tag_subcategory_summary.csv
- json_category_subcategory_summary.csv
- json_custom_field_summary.csv
- json_custom_field_value_examples.csv
- json_key_profile_summary.csv
- json_sequence_metadata_flat.csv
- json_all_tags_long.csv
- json_all_custom_fields_long.csv
- json_metadata_errors.csv
"""

    write_text(report, out_root / "json_metadata_report.md")

    print(f"profile                  : {paths.profile_name}")
    print(f"manifest_csv             : {manifest_csv}")
    print(f"output_root              : {out_root}")
    print(f"manifest_rows            : {len(manifest_rows)}")
    print(f"json_found               : {json_found}")
    print(f"json_loaded              : {json_loaded}")
    print(f"json_errors              : {len(error_rows)}")
    print(f"preferred_camera_summary : {dict(preferred_camera_counter)}")
    print(f"driving_direction_summary: {dict(driving_direction_counter)}")


if __name__ == "__main__":
    main()