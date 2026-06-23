from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vision_uss_research.metadata.enrich import (
    build_enriched_manifest_rows,
    build_metadata_profile_rows,
    load_csv,
    write_csv,
)
from vision_uss_research.settings import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich manifest with JSON metadata")
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Path profile name, e.g. local or colab_drive",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional override for input manifest CSV",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional override for enriched manifest CSV",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional override for metadata profile CSV",
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

    manifest_csv = (
        args.manifest.resolve()
        if args.manifest
        else (paths.manifests_dir / "sequence_manifest.csv").resolve()
    )
    enriched_csv = (
        args.out.resolve()
        if args.out
        else (paths.manifests_dir / "sequence_manifest_enriched.csv").resolve()
    )
    summary_csv = (
        args.summary_out.resolve()
        if args.summary_out
        else (paths.outputs_dir / "profiles" / "enriched_manifest_profile.csv").resolve()
    )

    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")

    manifest_rows = load_csv(manifest_csv)
    enriched_rows = build_enriched_manifest_rows(
        manifest_rows,
        raw_root=paths.raw_data_root,
        show_progress=not args.no_progress,
    )
    profile_rows = build_metadata_profile_rows(enriched_rows)

    write_csv(enriched_rows, enriched_csv)
    write_csv(profile_rows, summary_csv)

    json_ok = sum(1 for r in enriched_rows if str(r.get("json_loaded", "")) == "1")
    front_pref = sum(1 for r in enriched_rows if r.get("preferred_camera_rule") == "front")
    rear_pref = sum(1 for r in enriched_rows if r.get("preferred_camera_rule") == "rear")
    unknown_pref = sum(1 for r in enriched_rows if r.get("preferred_camera_rule") == "unknown")

    print(f"profile              : {paths.profile_name}")
    print(f"manifest_csv         : {manifest_csv}")
    print(f"enriched_manifest    : {enriched_csv}")
    print(f"profile_csv          : {summary_csv}")
    print(f"manifest_rows        : {len(manifest_rows)}")
    print(f"json_loaded          : {json_ok}")
    print(f"preferred_front      : {front_pref}")
    print(f"preferred_rear       : {rear_pref}")
    print(f"preferred_unknown    : {unknown_pref}")


if __name__ == "__main__":
    main()