from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vision_uss_research.sampling.subset import (
    build_samples,
    load_csv,
    write_csv,
)
from vision_uss_research.settings import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample first research subset from enriched manifest")
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--n-front", type=int, default=100)
    parser.add_argument("--n-rear", type=int, default=100)
    parser.add_argument("--n-unknown", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_paths(profile=args.profile)

    manifest_csv = (
        args.manifest.resolve()
        if args.manifest
        else (paths.manifests_dir / "sequence_manifest_enriched.csv").resolve()
    )

    if not manifest_csv.exists():
        raise FileNotFoundError(f"Enriched manifest not found: {manifest_csv}")

    enriched_rows = load_csv(manifest_csv)

    rule_ready_sample, unknown_review_sample = build_samples(
        enriched_rows=enriched_rows,
        n_front=args.n_front,
        n_rear=args.n_rear,
        n_unknown=args.n_unknown,
        seed=args.seed,
    )

    rule_ready_csv = paths.samples_dir / "rule_ready_sample.csv"
    unknown_review_csv = paths.samples_dir / "unknown_camera_review_sample.csv"

    write_csv(rule_ready_sample, rule_ready_csv)
    write_csv(unknown_review_sample, unknown_review_csv)

    front_count = sum(1 for r in rule_ready_sample if r["preferred_camera_rule"] == "front")
    rear_count = sum(1 for r in rule_ready_sample if r["preferred_camera_rule"] == "rear")

    print(f"profile                  : {paths.profile_name}")
    print(f"enriched_manifest        : {manifest_csv}")
    print(f"rule_ready_sample_csv    : {rule_ready_csv}")
    print(f"unknown_review_sample_csv: {unknown_review_csv}")
    print(f"rule_ready_rows          : {len(rule_ready_sample)}")
    print(f"  - front                : {front_count}")
    print(f"  - rear                 : {rear_count}")
    print(f"unknown_review_rows      : {len(unknown_review_sample)}")


if __name__ == "__main__":
    main()