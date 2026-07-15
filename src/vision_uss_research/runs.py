"""Experiment-run convention: run IDs and run.json provenance records.

Every pipeline run (labeling, dataset build, training, eval) writes its outputs
under <stage_root>/<run_id>/ together with a run.json capturing the git commit,
the frozen config, and the input artifacts it consumed. The chain
report -> model -> dataset -> label runs -> inventory -> git sha is what makes
a result reproducible.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


def new_run_id(stage: str, tag: str) -> str:
    """e.g. new_run_id('labels', 'sam3text_v1') -> '20260722_labels_sam3text_v1'."""
    return f"{time.strftime('%Y%m%d')}_{stage}_{tag}"


def git_sha(repo_root: Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, timeout=10,
        )
        sha = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}" if sha else "unknown"
    except Exception:
        return "unknown"


def start_run(run_dir: Path, config: dict, inputs: list[str] | None = None,
              repo_root: Path | None = None) -> dict:
    """Create run_dir and write run.json. `inputs` are the artifact paths/run IDs
    this run consumes (for the provenance chain). Returns the record."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": run_dir.name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": git_sha(repo_root),
        "config": config,
        "inputs": inputs or [],
        "status": "running",
    }
    (run_dir / "run.json").write_text(json.dumps(record, indent=1, default=str))
    return record


def finish_run(run_dir: Path, status: str = "completed",
               summary: dict | None = None) -> dict:
    """Mark the run finished; merge an optional summary (counts, metrics)."""
    run_dir = Path(run_dir)
    record = json.loads((run_dir / "run.json").read_text())
    record["status"] = status
    record["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if summary:
        record["summary"] = summary
    (run_dir / "run.json").write_text(json.dumps(record, indent=1, default=str))
    return record
