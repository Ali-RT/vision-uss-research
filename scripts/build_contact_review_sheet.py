from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vision_uss_research.settings import load_paths


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict[str, str]], out_path: Path) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build review sheet for sampled contact sheets")
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument(
        "--sample-name",
        type=str,
        default="rule_ready_sample",
        choices=["rule_ready_sample", "unknown_camera_review_sample"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_paths(profile=args.profile)

    sample_csv = paths.samples_dir / f"{args.sample_name}.csv"
    summary_csv = paths.outputs_dir / "reviews" / args.sample_name / "contact_sheet_summary.csv"

    if not sample_csv.exists():
        raise FileNotFoundError(f"Missing sample CSV: {sample_csv}")
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing contact summary CSV: {summary_csv}")

    sample_rows = load_csv(sample_csv)
    summary_rows = load_csv(summary_csv)

    summary_by_seq = {row["sequence_id"]: row for row in summary_rows}

    review_rows: list[dict[str, str]] = []

    for row in sample_rows:
        seq_id = row["sequence_id"]
        summary = summary_by_seq.get(seq_id, {})

        seq_root = paths.outputs_dir / "reviews" / args.sample_name / seq_id
        preferred_camera = (row.get("preferred_camera_rule") or "").strip().lower()
        if preferred_camera not in {"front", "rear"}:
            preferred_camera = "front"

        chosen_contact_path = seq_root / f"{preferred_camera}_contact.jpg"
        topview_contact_path = seq_root / "topview_contact.jpg"
        pair_mid_path = seq_root / f"pair_{preferred_camera}_mid.jpg"

        review_rows.append(
            {
                "sequence_id": seq_id,
                "preferred_camera_rule": row.get("preferred_camera_rule", ""),
                "preferred_camera_rule_reason": row.get("preferred_camera_rule_reason", ""),
                "driving_direction": row.get("driving_direction", ""),
                "approach": row.get("approach", ""),
                "weather_tags": row.get("weather_tags", ""),
                "primary_label_object": row.get("primary_label_object", ""),
                "scene_tags": row.get("scene_tags", ""),
                "chosen_contact_saved": summary.get("chosen_contact_saved", ""),
                "topview_contact_saved": summary.get("topview_contact_saved", ""),
                "pair_mid_saved": summary.get("pair_mid_saved", ""),
                "chosen_contact_path": str(chosen_contact_path),
                "topview_contact_path": str(topview_contact_path),
                "pair_mid_path": str(pair_mid_path),
                "best_ratio": "",
                "object_visible_score_0_3": "",
                "scene_clarity_score_0_3": "",
                "topview_useful_score_0_3": "",
                "keep_for_next_stage": "",
                "candidate_quality": "",
                "notes": "",
            }
        )

    out_csv = paths.outputs_dir / "reviews" / args.sample_name / "contact_review_sheet.csv"
    write_csv(review_rows, out_csv)

    print(f"profile            : {paths.profile_name}")
    print(f"sample_name        : {args.sample_name}")
    print(f"sample_rows        : {len(sample_rows)}")
    print(f"review_sheet_csv   : {out_csv}")


if __name__ == "__main__":
    main()