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
        out_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Qwen review input CSV")
    parser.add_argument("--profile", type=str, default="colab_drive")
    parser.add_argument("--sample-name", type=str, default="rule_ready_sample")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_paths(profile=args.profile)

    review_sheet = paths.outputs_dir / "reviews" / args.sample_name / "contact_review_sheet.csv"
    board_summary = paths.outputs_dir / "reviews" / args.sample_name / "qwen_review_board_summary.csv"
    out_csv = paths.outputs_dir / "reviews" / args.sample_name / "qwen_review_input.csv"

    rows = load_csv(review_sheet)
    board_rows = load_csv(board_summary)

    board_by_seq = {r["sequence_id"]: r for r in board_rows}

    out_rows = []
    for row in rows:
        seq_id = row["sequence_id"]
        board = board_by_seq.get(seq_id, {})
        merged = dict(row)
        merged["qwen_review_board_path"] = board.get("qwen_review_board_path", "")
        merged["qwen_review_board_saved"] = board.get("qwen_review_board_saved", "")
        out_rows.append(merged)

    write_csv(out_rows, out_csv)

    print(f"profile       : {paths.profile_name}")
    print(f"review_sheet  : {review_sheet}")
    print(f"board_summary : {board_summary}")
    print(f"out_csv       : {out_csv}")
    print(f"rows_written  : {len(out_rows)}")


if __name__ == "__main__":
    main()