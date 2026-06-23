from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vision_uss_research.settings import load_paths
from vision_uss_research.utils.video import (
    draw_text_block,
    get_video_info,
    make_side_by_side,
    read_frame_at_ratio,
    resize_with_padding,
    save_image,
)

RATIOS = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
TILE_W = 320
TILE_H = 214
GRID_COLS = 3
GRID_ROWS = 2


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict[str, str | int | float]], out_path: Path) -> None:
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


def blank_tile(message: str) -> any:
    import numpy as np
    import cv2

    img = np.ones((TILE_H, TILE_W, 3), dtype=np.uint8) * 245
    cv2.putText(
        img,
        message,
        (20, TILE_H // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return img


def build_contact_sheet(video_path: Path, camera_name: str, title: str) -> tuple[any, list[dict[str, str | int | float]], bool]:
    import numpy as np

    tiles = []
    rows = []

    info = get_video_info(video_path)
    if not info.opened:
        for ratio in RATIOS:
            tiles.append(blank_tile(f"{camera_name} missing"))
            rows.append(
                {
                    "camera_name": camera_name,
                    "ratio": ratio,
                    "frame_idx": -1,
                    "time_sec": "",
                    "success": 0,
                }
            )
        return make_grid(tiles, title), rows, False

    for ratio in RATIOS:
        frame, frame_idx, info = read_frame_at_ratio(video_path, ratio=ratio)
        if frame is None:
            tile = blank_tile(f"{camera_name} missing")
            success = 0
            time_sec = ""
        else:
            time_sec = round(info.duration_sec * ratio, 3) if info.duration_sec > 0 else ""
            tile = resize_with_padding(frame, TILE_W, TILE_H)
            tile = draw_text_block(
                tile,
                [
                    f"{camera_name}",
                    f"ratio={ratio:.2f} frame={frame_idx}",
                    f"time={time_sec}s",
                ],
            )
            success = 1

        tiles.append(tile)
        rows.append(
            {
                "camera_name": camera_name,
                "ratio": ratio,
                "frame_idx": frame_idx,
                "time_sec": time_sec,
                "success": success,
            }
        )

    return make_grid(tiles, title), rows, True


def make_grid(tiles: list[any], title: str) -> any:
    import numpy as np
    import cv2

    total = GRID_ROWS * GRID_COLS
    tiles = tiles[:total] + [blank_tile("missing")] * max(0, total - len(tiles))

    row_imgs = []
    for r in range(GRID_ROWS):
        row_tiles = tiles[r * GRID_COLS:(r + 1) * GRID_COLS]
        row_img = np.concatenate(row_tiles, axis=1)
        row_imgs.append(row_img)

    grid = np.concatenate(row_imgs, axis=0)

    header = np.ones((56, grid.shape[1], 3), dtype=np.uint8) * 255
    cv2.putText(
        header,
        title,
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    return np.concatenate([header, grid], axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate contact sheets for sampled subset")
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument(
        "--sample-csv",
        type=Path,
        default=None,
        help="Override sample CSV path",
    )
    parser.add_argument(
        "--sample-name",
        type=str,
        default="rule_ready_sample",
        choices=["rule_ready_sample", "unknown_camera_review_sample"],
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Limit number of sequences for smoke test; use 0 for all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_paths(profile=args.profile)

    sample_csv = (
        args.sample_csv.resolve()
        if args.sample_csv
        else (paths.samples_dir / f"{args.sample_name}.csv").resolve()
    )

    if not sample_csv.exists():
        raise FileNotFoundError(f"Sample CSV not found: {sample_csv}")

    rows = load_csv(sample_csv)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    out_root = paths.outputs_dir / "reviews" / args.sample_name
    summary_csv = out_root / "contact_sheet_summary.csv"

    summary_rows = []

    for row in tqdm(rows, desc=f"Generating {args.sample_name} contacts", unit="seq"):
        sequence_id = row["sequence_id"]
        preferred_camera = (row.get("preferred_camera_rule") or "").strip().lower()

        if preferred_camera not in {"front", "rear"}:
            preferred_camera = "front"

        chosen_video_rel = (row.get(f"{preferred_camera}_video") or "").strip()
        topview_video_rel = (row.get("topview_video") or "").strip()

        chosen_video_path = paths.raw_data_root / chosen_video_rel if chosen_video_rel else None
        topview_video_path = paths.raw_data_root / topview_video_rel if topview_video_rel else None

        seq_out = out_root / sequence_id
        seq_out.mkdir(parents=True, exist_ok=True)

        label_text = (
            f"{sequence_id} | cam={preferred_camera} | "
            f"dir={row.get('driving_direction', '')} | "
            f"approach={row.get('approach', '')} | "
            f"obj={row.get('primary_label_object', '')}"
        )

        chosen_contact_saved = 0
        topview_contact_saved = 0
        pair_mid_saved = 0

        if chosen_video_path and chosen_video_path.exists():
            chosen_sheet, chosen_sheet_rows, chosen_ok = build_contact_sheet(
                chosen_video_path, preferred_camera, label_text
            )
            chosen_contact_path = seq_out / f"{preferred_camera}_contact.jpg"
            save_image(chosen_contact_path, chosen_sheet)
            chosen_contact_saved = 1
        else:
            chosen_ok = False

        if topview_video_path and topview_video_path.exists():
            top_sheet, top_sheet_rows, top_ok = build_contact_sheet(
                topview_video_path, "topview", label_text
            )
            topview_contact_path = seq_out / "topview_contact.jpg"
            save_image(topview_contact_path, top_sheet)
            topview_contact_saved = 1
        else:
            top_ok = False

        pair_mid_path_rel = ""
        if chosen_video_path and chosen_video_path.exists() and topview_video_path and topview_video_path.exists():
            chosen_mid, _, _ = read_frame_at_ratio(chosen_video_path, ratio=0.60)
            top_mid, _, _ = read_frame_at_ratio(topview_video_path, ratio=0.60)

            if chosen_mid is not None and top_mid is not None:
                pair_mid = make_side_by_side(chosen_mid, top_mid)
                pair_mid = draw_text_block(
                    pair_mid,
                    [
                        f"{sequence_id}",
                        f"cam={preferred_camera} dir={row.get('driving_direction', '')}",
                        f"obj={row.get('primary_label_object', '')}",
                    ],
                    x=20,
                    y=30,
                )
                pair_mid_path = seq_out / f"pair_{preferred_camera}_mid.jpg"
                save_image(pair_mid_path, pair_mid)
                pair_mid_saved = 1
                pair_mid_path_rel = str(pair_mid_path)

        summary_rows.append(
            {
                "sequence_id": sequence_id,
                "preferred_camera_rule": preferred_camera,
                "driving_direction": row.get("driving_direction", ""),
                "approach": row.get("approach", ""),
                "primary_label_object": row.get("primary_label_object", ""),
                "weather_tags": row.get("weather_tags", ""),
                "chosen_video_exists": int(bool(chosen_video_path and chosen_video_path.exists())),
                "topview_video_exists": int(bool(topview_video_path and topview_video_path.exists())),
                "chosen_contact_saved": chosen_contact_saved,
                "topview_contact_saved": topview_contact_saved,
                "pair_mid_saved": pair_mid_saved,
                "pair_mid_path": pair_mid_path_rel,
            }
        )

    write_csv(summary_rows, summary_csv)

    print(f"profile             : {paths.profile_name}")
    print(f"sample_csv          : {sample_csv}")
    print(f"sequences_processed : {len(rows)}")
    print(f"summary_csv         : {summary_csv}")
    print(f"output_root         : {out_root}")


if __name__ == "__main__":
    main()