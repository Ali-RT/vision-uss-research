from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ProjectPaths:
    profile_name: str
    project_root: Path
    raw_data_root: Path
    manifests_dir: Path
    processed_dir: Path
    samples_dir: Path
    outputs_dir: Path


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def load_paths(profile: str | None = None) -> ProjectPaths:
    repo_root = Path(__file__).resolve().parents[2]

    profile_name = profile or os.environ.get("VUSS_PROFILE", "local")
    config_path = repo_root / "configs" / "paths" / f"{profile_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Path config not found for profile '{profile_name}': {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_project_root = cfg["paths"]["project_root"]
    project_root = _resolve_path(raw_project_root, repo_root)

    paths = ProjectPaths(
        profile_name=cfg["profile_name"],
        project_root=project_root,
        raw_data_root=_resolve_path(cfg["paths"]["raw_data_root"], project_root),
        manifests_dir=_resolve_path(cfg["paths"]["manifests_dir"], project_root),
        processed_dir=_resolve_path(cfg["paths"]["processed_dir"], project_root),
        samples_dir=_resolve_path(cfg["paths"]["samples_dir"], project_root),
        outputs_dir=_resolve_path(cfg["paths"]["outputs_dir"], project_root),
    )

    return paths