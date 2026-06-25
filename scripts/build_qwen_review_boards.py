from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vision_uss_research.settings import load_paths


RATIOS = ["0.15", "0.30", "0.45", "0.60", "0.75", "0.90"]

HEADER_H = 56
TILE_W = 320
TILE_H = 214
GRID_COLS = 3


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict[str, str | int]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        out_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def crop_tile(contact_img: np.ndarray, idx: int) -> np.ndarray:
    row = idx // GRID_COLS
    col = idx % GRID_COLS

    x0 = col * TILE_W
    y0 = HEADER_H + row * TILE_H
    x1 = x0 + TILE_W
    y1 = y0 + TILE_H

    tile = contact_img[y0:y1, x0:x1].copy()

    if tile.shape[:2] != (TILE_H, TILE_W):
        fixed = np.ones((TILE_H, TILE_W, 3), dtype=np.uint8) * 245
        h = min(TILE_H, tile.shape[0])
        w = min(TILE_W, tile.shape[1])
        fixed[:h, :w] = tile[:h, :w]
        return fixed

    return tile


def add_text(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float = 0.65,
    thickness: int = 2,
) -> None:
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def make_review_board(
    chosen_contact_path: Path,
    topview_contact_path: Path,
    row: dict[str, str],
    out_path: Path,
) -> bool:
    chosen = cv2.imread(str(chosen_contact_path))
    topview = cv2.imread(str(topview_contact_path))

    if chosen is None or topview is None:
        return False

    label_w = 140
    gap = 12
    header_h = 125
    row_h = TILE_H + 18

    board_w = label_w + TILE_W + gap + TILE_W
    board_h = header_h + len(RATIOS) * row_h

    board = np.ones((board_h, board_w, 3), dtype=np.uint8) * 255

    seq_id = row.get("sequence_id", "")
    cam = row.get("preferred_camera_rule", "")
    direction = row.get("driving_direction", "")
    target = row.get("target_object", row.get("primary_label_object", ""))
    objects = row.get("label_objects_json", "")
    weather = row.get("weather_tags", "")
    scene = row.get("scene_tags", "")

    add_text(board, "Qwen ratio review board", 16, 32, scale=0.85, thickness=2)
    add_text(board, f"seq={seq_id}", 16, 60, scale=0.48, thickness=1)
    add_text(board, f"camera={cam} direction={direction} target={target}", 16, 84, scale=0.50, thickness=1)
    add_text(board, f"objects={objects} weather={weather}", 16, 106, scale=0.45, thickness=1)

    add_text(board, "ratio", 24, header_h - 10, scale=0.60)
    add_text(board, "selected camera", label_w + 16, header_h - 10, scale=0.60)
    add_text(board, "top-view", label_w + TILE_W + gap + 16, header_h - 10, scale=0.60)

    for idx, ratio in enumerate(RATIOS):
        y = header_h + idx * row_h

        chosen_tile = crop_tile(chosen, idx)
        topview_tile = crop_tile(topview, idx)

        cv2.rectangle(board, (0, y), (board_w - 1, y + row_h - 1), (220, 220, 220), 1)
        add_text(board, ratio, 30, y + 112, scale=0.75, thickness=2)

        x_chosen = label_w
        x_top = label_w + TILE_W + gap

        board[y:y + TILE_H, x_chosen:x_chosen + TILE_W] = chosen_tile
        board[y:y + TILE_H, x_top:x_top + TILE_W] = topview_tile

    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), board))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build aligned Qwen review boards")
    parser.add_argument("--profile", type=str, default="colab_drive")
    parser.add_argument("--sample-name", type=str, default="rule_ready_sample")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_paths(profile=args.profile)

    review_sheet = paths.outputs_dir / "reviews" / args.sample_name / "contact_review_sheet.csv"
    if not review_sheet.exists():
        raise FileNotFoundError(f"Missing review sheet: {review_sheet}")

    rows = load_csv(review_sheet)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    out_root = paths.outputs_dir / "reviews" / args.sample_name
    summary_rows = []

    for row in tqdm(rows, desc="Building Qwen review boards", unit="seq"):
        seq_id = row["sequence_id"]
        chosen_contact_path = Path(row["chosen_contact_path"])
        topview_contact_path = Path(row["topview_contact_path"])
        board_path = out_root / seq_id / "qwen_ratio_review_board.jpg"

        ok = make_review_board(
            chosen_contact_path=chosen_contact_path,
            topview_contact_path=topview_contact_path,
            row=row,
            out_path=board_path,
        )

        summary_rows.append(
            {
                "sequence_id": seq_id,
                "qwen_review_board_path": str(board_path),
                "qwen_review_board_saved": int(ok),
            }
        )

    summary_csv = out_root / "qwen_review_board_summary.csv"
    write_csv(summary_rows, summary_csv)

    print(f"profile       : {paths.profile_name}")
    print(f"review_sheet  : {review_sheet}")
    print(f"summary_csv   : {summary_csv}")
    print(f"rows_processed: {len(rows)}")
    print(f"boards_saved  : {sum(int(r['qwen_review_board_saved']) for r in summary_rows)}")


if __name__ == "__main__":
    main()