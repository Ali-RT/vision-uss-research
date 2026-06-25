from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


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


def _norm_bool(text: str) -> bool:
    return str(text).strip().lower() in {"1", "true", "yes"}


def split_pipe_values(text: str) -> list[str]:
    if not text:
        return []
    return [x.strip() for x in str(text).split("|") if x.strip()]

def primary_label_object(text: str) -> str:
    items = split_pipe_values(text)

    cleaned: list[str] = []
    for item in items:
        name = item.split(":", 1)[0].strip()
        if name:
            cleaned.append(name)

    for name in cleaned:
        if name.lower() != "empty":
            return name

    if cleaned and cleaned[0].lower() == "empty":
        return "empty_control"

    return "unknown"


def primary_weather(text: str) -> str:
    items = split_pipe_values(text)
    return items[0] if items else "MISSING"


def make_sampling_row(row: dict[str, str]) -> dict[str, str]:
    label_objects_json = row.get("label_objects_json", "")
    target_object = primary_label_object(label_objects_json)

    return {
        "sequence_id": row.get("sequence_id", ""),
        "sequence_dir": row.get("sequence_dir", ""),
        "is_poc_ready": row.get("is_poc_ready", ""),
        "json_loaded": row.get("json_loaded", ""),
        "preferred_camera_rule": row.get("preferred_camera_rule", ""),
        "preferred_camera_rule_reason": row.get("preferred_camera_rule_reason", ""),
        "driving_direction": row.get("driving_direction", ""),
        "approach": row.get("approach", ""),
        "scene_tags": row.get("scene_tags", ""),
        "weather_tags": row.get("weather_tags", ""),
        "label_objects_json": label_objects_json,
        "target_object": target_object,
        "primary_label_object": target_object,
        "primary_weather": primary_weather(row.get("weather_tags", "")),
        "front_video": row.get("front_video", ""),
        "rear_video": row.get("rear_video", ""),
        "topview_video": row.get("topview_video", ""),
        "label_v2_csv": row.get("label_v2_csv", ""),
        "metadata_json": row.get("metadata_json", ""),
    }


def filter_base_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for row in rows:
        if not _norm_bool(row.get("is_poc_ready", "")):
            continue
        if not _norm_bool(row.get("json_loaded", "")):
            continue
        if not (row.get("label_v2_csv") or "").strip():
            continue
        out.append(row)
    return out


def stratified_sample(
    rows: list[dict[str, str]],
    group_keys: list[str],
    target_n: int,
    seed: int = 42,
) -> list[dict[str, str]]:
    if target_n <= 0 or not rows:
        return []

    rng = random.Random(seed)

    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = tuple((row.get(k, "") or "MISSING").strip() for k in group_keys)
        groups[key].append(row)

    for group_rows in groups.values():
        rng.shuffle(group_rows)

    group_items = list(groups.items())
    rng.shuffle(group_items)

    selected: list[dict[str, str]] = []

    # Round-robin across groups for diversity
    while len(selected) < target_n:
        progressed = False
        for _, group_rows in group_items:
            if group_rows and len(selected) < target_n:
                selected.append(group_rows.pop())
                progressed = True
        if not progressed:
            break

    return selected


def build_samples(
    enriched_rows: list[dict[str, str]],
    n_front: int,
    n_rear: int,
    n_unknown: int,
    seed: int = 42,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    base_rows = filter_base_rows(enriched_rows)

    front_rows = [r for r in base_rows if (r.get("preferred_camera_rule") or "").strip() == "front"]
    rear_rows = [r for r in base_rows if (r.get("preferred_camera_rule") or "").strip() == "rear"]
    unknown_rows = [r for r in base_rows if (r.get("preferred_camera_rule") or "").strip() == "unknown"]

    sampled_front = stratified_sample(
        front_rows,
        group_keys=["weather_tags", "approach", "target_object"],
        target_n=n_front,
        seed=seed,
    )
    sampled_rear = stratified_sample(
        rear_rows,
        group_keys=["weather_tags", "approach", "target_object"],
        target_n=n_rear,
        seed=seed + 1,
    )
    sampled_unknown = stratified_sample(
        unknown_rows,
        group_keys=["weather_tags", "approach", "target_object"],
        target_n=n_unknown,
        seed=seed + 2,
    )

    rule_ready_sample = [make_sampling_row(r) for r in (sampled_front + sampled_rear)]
    unknown_review_sample = [make_sampling_row(r) for r in sampled_unknown]

    return rule_ready_sample, unknown_review_sample