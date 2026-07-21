"""Tolerant append-CSV reading for long, resumable runs.

Resumable jobs append rows to one CSV across many sessions. When the row schema
gains a column between sessions, later rows are appended under the *old* header
and `pd.read_csv` dies with "Expected N fields, saw M". These helpers read such
a file anyway - mapping every row against whichever schema matches its width -
and can rewrite it in the current schema so the problem does not recur.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import pandas as pd


def read_csv_tolerant(path: Path, fields: list[str]) -> pd.DataFrame:
    """Read an append-CSV whose rows may follow the current `fields` schema or
    an older, narrower one recorded in the file's own header line.

    Rows are matched by field count: current schema first, then the header
    schema. Header lines (repeated or not) and unmatched widths are skipped.
    Returns a DataFrame with exactly `fields` as columns."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=fields)

    rows: list[dict] = []
    header: list[str] | None = None
    with path.open(newline="") as f:
        for rec in csv.reader(f):
            if not rec:
                continue
            # any line that looks like a header defines the legacy schema
            if rec[0] == fields[0] and len(set(rec) & set(fields)) > 1:
                if header is None:
                    header = rec
                continue
            if len(rec) == len(fields):
                rows.append(dict(zip(fields, rec)))
            elif header is not None and len(rec) == len(header):
                rows.append(dict(zip(header, rec)))
            # else: unknown width -> malformed, drop
    df = pd.DataFrame(rows)
    for col in fields:
        if col not in df.columns:
            df[col] = None
    return df[fields]


def repair_csv(path: Path, fields: list[str]) -> tuple[bool, int]:
    """Rewrite `path` in the current schema when it holds mixed-width rows.

    The original is kept as `<name>_backup_<timestamp>.csv`. Returns
    (repaired, n_rows). A file that already parses cleanly is left untouched."""
    path = Path(path)
    if not path.exists():
        return False, 0
    try:
        clean = pd.read_csv(path)
        if list(clean.columns) == list(fields):
            return False, len(clean)
    except Exception:
        pass  # unreadable or wrong columns -> repair below

    df = read_csv_tolerant(path, fields)
    backup = path.with_name(f"{path.stem}_backup_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    path.rename(backup)
    df.to_csv(path, index=False)
    return True, len(df)
