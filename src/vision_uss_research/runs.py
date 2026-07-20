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


def new_run_id(stage: str, tag: str, dated: bool = False) -> str:
    """Run id from a stage and a tag.

    Default is STABLE (`{stage}_{tag}`, e.g. 'dataset_v2'): re-running the same
    config targets the same run dir, so a build interrupted and resumed on a
    later day continues in place instead of minting a new, date-mismatched dir
    (the bug that made cross-run paths day-dependent). The creation timestamp is
    recorded inside run.json, not baked into the path. Bump the tag ('v2'->'v3')
    when you want a genuinely separate run.

    `dated=True` prepends the date for runs where one-per-invocation really is
    wanted (e.g. throwaway sweeps)."""
    base = f"{stage}_{tag}"
    return f"{time.strftime('%Y%m%d')}_{base}" if dated else base


def _run_created(run_dir: Path) -> float:
    """Sort key for 'latest': run.json's created timestamp, else dir mtime."""
    meta = run_dir / "run.json"
    if meta.exists():
        try:
            created = json.loads(meta.read_text()).get("created")
            if created:
                return time.mktime(time.strptime(created, "%Y-%m-%dT%H:%M:%S"))
        except Exception:
            pass
    try:
        return run_dir.stat().st_mtime
    except OSError:
        return 0.0


def latest_run(base_dir: Path, require_file: str | None = None,
               exclude_suffix: str | None = None) -> Path | None:
    """Newest run dir under base_dir by creation time (from run.json, not by
    name), so discovery is robust to stable AND legacy date-prefixed ids.
    `require_file`: only consider dirs containing this file. `exclude_suffix`:
    skip dirs whose name ends with it (e.g. '_smoke')."""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return None
    candidates = [
        d for d in base_dir.iterdir()
        if d.is_dir()
        and (require_file is None or (d / require_file).exists())
        and (exclude_suffix is None or not d.name.endswith(exclude_suffix))
    ]
    if not candidates:
        return None
    return max(candidates, key=_run_created)


def resolve_run(base_dir: Path, stage: str, tag: str | None = None,
                run_id: str | None = None, require_file: str | None = None,
                exclude_suffix: str | None = None) -> Path | None:
    """Pick a run dir: explicit `run_id` wins; else the stable `{stage}_{tag}`
    dir if it exists; else the newest run under base_dir. One resolution rule
    for every notebook, so 'which run does this consume' is never day-dependent."""
    base_dir = Path(base_dir)
    if run_id:
        return base_dir / run_id
    if tag:
        stable = base_dir / f"{stage}_{tag}"
        if stable.exists() and (require_file is None or (stable / require_file).exists()):
            return stable
    return latest_run(base_dir, require_file=require_file,
                      exclude_suffix=exclude_suffix)


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
