from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vision_uss_research.settings import load_paths


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
    # Try common encodings/separators. Label files may not all be perfectly consistent.
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

    # fallback: substring match
    for col in columns:
        low = norm_col(col)
        for candidate in candidates:
            if norm_col(candidate) in low:
                return col

    return ""


def nonempty_count(series: pd.Series) -> int:
    return int(series.notna().sum() - (series.astype(str).str.strip() == "").sum())


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile Label V2 CSV files")
    parser.add_argument("--profile", type=str, default="colab_drive")
    parser.add_argument(
        "--sample-csv",
        type=Path,
        default=None,
        help="Default: samples/rule_ready_sample.csv",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    paths = load_paths(profile=args.profile)

    sample_csv = (
        args.sample_csv.resolve()
        if args.sample_csv
        else (paths.samples_dir / "rule_ready_sample.csv").resolve()
    )

    rows = load_csv_dict(sample_csv)

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    out_root = paths.outputs_dir / "profiles" / "label_v2"
    out_root.mkdir(parents=True, exist_ok=True)

    sequence_rows: list[dict[str, Any]] = []
    column_rows: list[dict[str, Any]] = []
    label_long_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    all_column_counter = Counter()
    object_counter = Counter()
    class_counter = Counter()
    target_object_counter = Counter()
    rows_per_file_counter = Counter()

    for row in tqdm(rows, desc="Profiling label_v2 CSVs", unit="seq"):
        sequence_id = row.get("sequence_id", "")
        target_object = row.get("target_object", row.get("primary_label_object", ""))
        label_rel = str(row.get("label_v2_csv", "")).strip()

        target_object_counter[target_object or "MISSING"] += 1

        if not label_rel:
            error_rows.append(
                {
                    "sequence_id": sequence_id,
                    "label_v2_csv": "",
                    "error": "missing_label_v2_csv_path",
                }
            )
            continue

        label_path = paths.raw_data_root / label_rel

        if not label_path.exists():
            error_rows.append(
                {
                    "sequence_id": sequence_id,
                    "label_v2_csv": label_rel,
                    "error": "label_v2_csv_not_found",
                }
            )
            continue

        try:
            df = safe_read_label_csv(label_path)
        except Exception as e:
            error_rows.append(
                {
                    "sequence_id": sequence_id,
                    "label_v2_csv": label_rel,
                    "error": f"{type(e).__name__}: {e}",
                }
            )
            continue

        columns = list(df.columns)
        for col in columns:
            all_column_counter[col] += 1

        class_col = find_column(columns, ["class", "class_id", "Class"])
        object_col = find_column(columns, ["Which object", "object", "Object", "label_object"])
        distance_col = find_column(
            columns,
            ["important Distance [mm]", "important distance", "distance", "Distance"],
        )
        xmin_col = find_column(columns, ["xmin", "x_min", "bbox_xmin"])
        ymin_col = find_column(columns, ["ymin", "y_min", "bbox_ymin"])
        xmax_col = find_column(columns, ["xmax", "x_max", "bbox_xmax"])
        ymax_col = find_column(columns, ["ymax", "y_max", "bbox_ymax"])
        poly_x_col = find_column(columns, ["PolygonX", "polygon_x", "poly_x"])
        poly_y_col = find_column(columns, ["PolygonY", "polygon_y", "poly_y"])

        n_rows = len(df)
        rows_per_file_counter[n_rows] += 1

        object_nonempty = nonempty_count(df[object_col]) if object_col else 0
        class_nonempty = nonempty_count(df[class_col]) if class_col else 0
        distance_nonempty = nonempty_count(df[distance_col]) if distance_col else 0

        bbox_available = all([xmin_col, ymin_col, xmax_col, ymax_col])
        polygon_available = bool(poly_x_col and poly_y_col)

        if object_col:
            for value in df[object_col].dropna().astype(str):
                value = value.strip()
                if value:
                    object_counter[value] += 1

        if class_col:
            for value in df[class_col].dropna().astype(str):
                value = value.strip()
                if value:
                    class_counter[value] += 1

        sequence_rows.append(
            {
                "sequence_id": sequence_id,
                "preferred_camera_rule": row.get("preferred_camera_rule", ""),
                "driving_direction": row.get("driving_direction", ""),
                "target_object": target_object,
                "label_objects_json": row.get("label_objects_json", ""),
                "label_v2_csv": label_rel,
                "num_label_rows": n_rows,
                "num_columns": len(columns),
                "class_col": class_col,
                "object_col": object_col,
                "distance_col": distance_col,
                "xmin_col": xmin_col,
                "ymin_col": ymin_col,
                "xmax_col": xmax_col,
                "ymax_col": ymax_col,
                "polygon_x_col": poly_x_col,
                "polygon_y_col": poly_y_col,
                "bbox_available": int(bbox_available),
                "polygon_available": int(polygon_available),
                "object_nonempty_rows": object_nonempty,
                "class_nonempty_rows": class_nonempty,
                "distance_nonempty_rows": distance_nonempty,
                "columns": "|".join(columns),
            }
        )

        for col in columns:
            series = df[col]
            column_rows.append(
                {
                    "sequence_id": sequence_id,
                    "label_v2_csv": label_rel,
                    "column": col,
                    "nonempty_count": nonempty_count(series),
                    "sample_values": "|".join(
                        series.dropna().astype(str).str.strip().replace("", pd.NA).dropna().head(5)
                    ),
                }
            )

        # Long preview, capped to first 20 label rows per sequence
        for label_idx, label_row in df.head(20).iterrows():
            label_long_rows.append(
                {
                    "sequence_id": sequence_id,
                    "label_row_idx": int(label_idx),
                    "target_object": target_object,
                    "class_value": str(label_row[class_col]).strip() if class_col else "",
                    "object_value": str(label_row[object_col]).strip() if object_col else "",
                    "distance_value": str(label_row[distance_col]).strip() if distance_col else "",
                    "xmin": str(label_row[xmin_col]).strip() if xmin_col else "",
                    "ymin": str(label_row[ymin_col]).strip() if ymin_col else "",
                    "xmax": str(label_row[xmax_col]).strip() if xmax_col else "",
                    "ymax": str(label_row[ymax_col]).strip() if ymax_col else "",
                    "polygon_x": str(label_row[poly_x_col]).strip() if poly_x_col else "",
                    "polygon_y": str(label_row[poly_y_col]).strip() if poly_y_col else "",
                }
            )

    summary_rows: list[dict[str, Any]] = []

    summary_rows.append({"group": "overview", "value": "sample_rows", "count": len(rows)})
    summary_rows.append({"group": "overview", "value": "labels_loaded", "count": len(sequence_rows)})
    summary_rows.append({"group": "overview", "value": "label_errors", "count": len(error_rows)})

    for value, count in all_column_counter.most_common():
        summary_rows.append({"group": "columns", "value": value, "count": count})

    for value, count in object_counter.most_common(100):
        summary_rows.append({"group": "label_object_values", "value": value, "count": count})

    for value, count in class_counter.most_common(50):
        summary_rows.append({"group": "class_values", "value": value, "count": count})

    for value, count in target_object_counter.most_common(100):
        summary_rows.append({"group": "target_object_from_json", "value": value, "count": count})

    for value, count in rows_per_file_counter.most_common(50):
        summary_rows.append({"group": "rows_per_label_file", "value": value, "count": count})

    write_csv(summary_rows, out_root / "label_v2_summary.csv")
    write_csv(sequence_rows, out_root / "label_v2_sequence_profile.csv")
    write_csv(column_rows, out_root / "label_v2_column_profile.csv")
    write_csv(label_long_rows, out_root / "label_v2_long_preview.csv")
    write_csv(error_rows, out_root / "label_v2_errors.csv")

    print(f"profile             : {paths.profile_name}")
    print(f"sample_csv          : {sample_csv}")
    print(f"output_root         : {out_root}")
    print(f"sample_rows         : {len(rows)}")
    print(f"labels_loaded       : {len(sequence_rows)}")
    print(f"label_errors        : {len(error_rows)}")
    print(f"unique_columns      : {len(all_column_counter)}")
    print(f"top_objects         : {object_counter.most_common(20)}")
    print(f"class_values        : {class_counter.most_common(20)}")


if __name__ == "__main__":
    main()