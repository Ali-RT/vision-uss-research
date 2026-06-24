from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm


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


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def safe_join(values: list[str]) -> str:
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in out:
            out.append(text)
    return "|".join(out)


def canonical_subcategory_name(name: str) -> str:
    name = str(name or "").strip()
    if "." in name:
        name = name.split(".")[-1].strip()
    return name


def get_tags_by_subcategory(payload: dict[str, Any], subcategory_name: str) -> list[str]:
    values = []
    target = subcategory_name.strip().lower()

    for tag in payload.get("tags", []):
        sub = tag.get("tagSubCategory", {}) or {}
        sub_name = canonical_subcategory_name(str(sub.get("name", ""))).lower()

        if sub_name == target:
            tag_name = str(tag.get("name", "")).strip()
            if tag_name:
                values.append(tag_name)

    return values


def pick_first(values: list[str]) -> str:
    return values[0] if values else ""


def infer_preferred_camera(driving_direction: str) -> tuple[str, str]:
    value = driving_direction.strip().lower()

    if value in {"forward", "forwards"}:
        return "front", "json_driving_direction"
    if value in {"backward", "backwards", "reverse", "rearward"}:
        return "rear", "json_driving_direction"

    return "unknown", "no_direction_rule"


def build_enriched_manifest_rows(
    manifest_rows: list[dict[str, str]],
    raw_root: Path,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    enriched_rows: list[dict[str, Any]] = []

    iterator = manifest_rows
    if show_progress:
        iterator = tqdm(manifest_rows, desc="Enriching manifest from JSON", unit="seq")

    for row in iterator:
        metadata_rel = (row.get("metadata_json") or "").strip()
        metadata_path = raw_root / metadata_rel if metadata_rel else None

        metadata_exists = int(bool(metadata_path and metadata_path.exists()))
        json_loaded = 0

        driving_direction = ""
        approach = ""
        weather_tags = ""
        scene_tags = ""
        label_objects_json = ""
        sequence_name_json = ""
        sequence_status_json = ""
        max_speed_kph = ""
        min_speed_kph = ""
        avg_speed = ""
        preferred_camera_rule = "unknown"
        preferred_camera_rule_reason = "no_metadata"

        if metadata_exists:
            try:
                payload = load_json(metadata_path)
                json_loaded = 1

                driving_direction = pick_first(get_tags_by_subcategory(payload, "Driving Direction"))
                approach = pick_first(get_tags_by_subcategory(payload, "Approach"))
                weather_tags = safe_join(get_tags_by_subcategory(payload, "Weather Conditions"))
                scene_tags = safe_join(get_tags_by_subcategory(payload, "Scene"))
                label_objects_json = safe_join(get_tags_by_subcategory(payload, "Label Objects"))

                sequence_name_json = str(payload.get("name", "")).strip()
                sequence_status_json = str(payload.get("status", "")).strip()
                max_speed_kph = payload.get("maxSpeedInKph", "")
                min_speed_kph = payload.get("minSpeedInKph", "")

                for item in payload.get("customFieldValues", []):
                    field = item.get("field", {}) or {}
                    field_name = str(field.get("name", "")).strip().lower()
                    if field_name == "avg speed":
                        avg_speed = item.get("value", "")
                        break

                preferred_camera_rule, preferred_camera_rule_reason = infer_preferred_camera(
                    driving_direction
                )

            except Exception:
                json_loaded = 0
                preferred_camera_rule = "unknown"
                preferred_camera_rule_reason = "json_parse_error"

        enriched = dict(row)
        enriched.update(
            {
                "metadata_exists": metadata_exists,
                "json_loaded": json_loaded,
                "sequence_name_json": sequence_name_json,
                "sequence_status_json": sequence_status_json,
                "driving_direction": driving_direction,
                "approach": approach,
                "scene_tags": scene_tags,
                "weather_tags": weather_tags,
                "label_objects_json": label_objects_json,
                "max_speed_kph": max_speed_kph,
                "min_speed_kph": min_speed_kph,
                "avg_speed": avg_speed,
                "preferred_camera_rule": preferred_camera_rule,
                "preferred_camera_rule_reason": preferred_camera_rule_reason,
            }
        )
        enriched_rows.append(enriched)

    return enriched_rows


def build_metadata_profile_rows(enriched_rows: list[dict[str, Any]]) -> list[dict[str, str | int]]:
    direction_counter = Counter((r.get("driving_direction") or "MISSING") for r in enriched_rows)
    preferred_counter = Counter((r.get("preferred_camera_rule") or "MISSING") for r in enriched_rows)
    json_loaded_counter = Counter(str(r.get("json_loaded", "")) for r in enriched_rows)
    poc_ready_counter = Counter(str(r.get("is_poc_ready", "")) for r in enriched_rows)
    weather_counter = Counter()
    scene_counter = Counter()
    label_objects_counter = Counter()

    for row in enriched_rows:
        for item in str(row.get("weather_tags", "")).split("|"):
            item = item.strip()
            if item:
                weather_counter[item] += 1

        for item in str(row.get("scene_tags", "")).split("|"):
            item = item.strip()
            if item:
                scene_counter[item] += 1

        for item in str(row.get("label_objects_json", "")).split("|"):
            item = item.strip()
            if item:
                label_objects_counter[item] += 1

    profile_rows: list[dict[str, str | int]] = []

    def add_counter(group: str, counter: Counter, top_n: int | None = None) -> None:
        for key, count in counter.most_common(top_n):
            profile_rows.append({"group": group, "value": key, "count": count})

    add_counter("driving_direction", direction_counter)
    add_counter("preferred_camera_rule", preferred_counter)
    add_counter("json_loaded", json_loaded_counter)
    add_counter("is_poc_ready", poc_ready_counter)
    add_counter("weather_tags", weather_counter, top_n=50)
    add_counter("scene_tags", scene_counter, top_n=50)
    add_counter("label_objects_json", label_objects_counter, top_n=100)

    return profile_rows