import csv

import pandas as pd
import pytest

from vision_uss_research.csv_io import read_csv_tolerant, repair_csv

OLD = ["sequence_id", "alignable", "error"]
NEW = ["sequence_id", "alignable", "error", "driving_direction", "required_camera"]


def _mixed_file(tmp_path):
    """The real failure: old header + old rows, then wider rows appended
    under it by a later version of the code."""
    p = tmp_path / "probe.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(OLD)
        w.writerow(["seqA", "1", ""])
        w.writerow(["seqB", "0", "no_mf4"])
        # newer rows: 5 fields, no new header
        w.writerow(["seqC", "1", "", "backward", "rear"])
        w.writerow(["seqD", "1", "", "forward", "front"])
    return p


def test_pandas_actually_fails_on_the_mixed_file(tmp_path):
    with pytest.raises(Exception):
        pd.read_csv(_mixed_file(tmp_path))


def test_read_csv_tolerant_recovers_both_schemas(tmp_path):
    df = read_csv_tolerant(_mixed_file(tmp_path), NEW)
    assert list(df.columns) == NEW
    assert len(df) == 4
    assert set(df["sequence_id"]) == {"seqA", "seqB", "seqC", "seqD"}
    # old rows get empty values for the new columns
    a = df[df["sequence_id"] == "seqA"].iloc[0]
    assert a["alignable"] == "1" and pd.isna(a["driving_direction"])
    # new rows keep theirs
    c = df[df["sequence_id"] == "seqC"].iloc[0]
    assert c["driving_direction"] == "backward" and c["required_camera"] == "rear"


def test_repair_csv_rewrites_and_backs_up(tmp_path):
    p = _mixed_file(tmp_path)
    repaired, n = repair_csv(p, NEW)
    assert repaired and n == 4
    # now plain pandas can read it, in the current schema
    df = pd.read_csv(p)
    assert list(df.columns) == NEW and len(df) == 4
    backups = list(tmp_path.glob("probe_backup_*.csv"))
    assert len(backups) == 1, "original must be preserved"
    # resume set is recoverable
    assert set(df["sequence_id"].astype(str)) == {"seqA", "seqB", "seqC", "seqD"}


def test_repair_is_a_noop_on_a_clean_file(tmp_path):
    p = tmp_path / "clean.csv"
    pd.DataFrame([{k: "x" for k in NEW}]).to_csv(p, index=False)
    before = p.read_text()
    repaired, n = repair_csv(p, NEW)
    assert not repaired and n == 1
    assert p.read_text() == before
    assert not list(tmp_path.glob("clean_backup_*.csv"))


def test_missing_file_returns_empty_frame(tmp_path):
    df = read_csv_tolerant(tmp_path / "nope.csv", NEW)
    assert list(df.columns) == NEW and df.empty
    assert repair_csv(tmp_path / "nope.csv", NEW) == (False, 0)
