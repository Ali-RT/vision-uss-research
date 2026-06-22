from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vision_uss_research.settings import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print resolved project paths")
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Path profile name, e.g. local or colab_drive",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_paths(profile=args.profile)

    print(f"profile_name  : {paths.profile_name}")
    print(f"project_root  : {paths.project_root}")
    print(f"raw_data_root : {paths.raw_data_root}")
    print(f"manifests_dir : {paths.manifests_dir}")
    print(f"processed_dir : {paths.processed_dir}")
    print(f"samples_dir   : {paths.samples_dir}")
    print(f"outputs_dir   : {paths.outputs_dir}")


if __name__ == "__main__":
    main()