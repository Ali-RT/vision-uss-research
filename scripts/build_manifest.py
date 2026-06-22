from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vision_uss_research.inventory.discover import build_manifest, write_manifest_csv
from vision_uss_research.settings import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sequence manifest")
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Path profile name, e.g. local or colab_drive",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Optional override for raw data root",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output CSV path",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_paths(profile=args.profile)

    raw_root = args.raw_dir.resolve() if args.raw_dir else paths.raw_data_root.resolve()
    out_csv = (
        args.out.resolve()
        if args.out
        else (paths.manifests_dir / "sequence_manifest.csv").resolve()
    )

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw data root does not exist: {raw_root}")

    records = build_manifest(raw_root, show_progress=not args.no_progress)
    write_manifest_csv(records, out_csv)

    poc_ready = sum(1 for r in records if r.is_poc_ready)

    print(f"profile        : {paths.profile_name}")
    print(f"raw_root       : {raw_root}")
    print(f"manifest_csv   : {out_csv}")
    print(f"sequences_found: {len(records)}")
    print(f"poc_ready      : {poc_ready}")

    if records:
        print("\nfirst_5_sequence_ids:")
        for record in records[:5]:
            print(f"  - {record.sequence_id}")


if __name__ == "__main__":
    main()