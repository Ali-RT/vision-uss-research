import json
import time

from vision_uss_research.runs import (finish_run, latest_run, new_run_id,
                                      resolve_run, start_run)
from vision_uss_research.sequences import frames_window_id, nearest_frame


def test_frames_window_id():
    assert frames_window_id(20, (0.30, 0.85)) == "r30-85_n20"


def test_nearest_frame(tmp_path):
    for i in [0, 3, 7, 15]:
        (tmp_path / f"{i:05d}.jpg").write_bytes(b"x")
    assert nearest_frame(tmp_path, 7).name == "00007.jpg"
    assert nearest_frame(tmp_path, 10).name == "00007.jpg"   # |7-10| < |15-10|
    assert nearest_frame(tmp_path, 14).name == "00015.jpg"
    empty = tmp_path / "empty"
    empty.mkdir()
    assert nearest_frame(empty, 5) is None


def test_stable_and_dated_run_ids():
    # stable by default: re-running the same config resolves to the same dir
    assert new_run_id("dataset", "v2") == "dataset_v2"
    assert new_run_id("labels", "sam3text_v1") == "labels_sam3text_v1"
    # dated only when explicitly requested
    assert new_run_id("dataset", "v2", dated=True).endswith("_dataset_v2")
    assert new_run_id("dataset", "v2", dated=True)[:8].isdigit()


def test_latest_run_by_created_not_name(tmp_path):
    # a legacy date-named dir created LATER must still win over an older
    # stable-named dir - discovery uses run.json created, not lexical name
    old = tmp_path / "dataset_v2"
    new = tmp_path / "20200101_dataset_v2"   # name sorts earlier, created later
    for d, created in [(old, "2026-07-15T10:00:00"), (new, "2026-07-18T10:00:00")]:
        d.mkdir()
        (d / "run.json").write_text(json.dumps({"created": created}))
        (d / "dataset_manifest.csv").write_text("x")
    assert latest_run(tmp_path, require_file="dataset_manifest.csv") == new


def test_resolve_run_prefers_stable_then_latest(tmp_path):
    stable = tmp_path / "dataset_v2"
    stable.mkdir()
    (stable / "run.json").write_text(json.dumps({"created": "2026-07-15T10:00:00"}))
    (stable / "dataset_manifest.csv").write_text("x")
    # explicit run_id wins
    assert resolve_run(tmp_path, "dataset", run_id="dataset_v2") == stable
    # tag resolves to the stable dir
    assert resolve_run(tmp_path, "dataset", tag="v2",
                       require_file="dataset_manifest.csv") == stable
    # smoke dirs excluded from latest
    smoke = tmp_path / "inventory_v2_smoke"
    smoke.mkdir()
    (smoke / "run.json").write_text(json.dumps({"created": "2026-07-20T10:00:00"}))
    assert latest_run(tmp_path, exclude_suffix="_smoke") == stable


def test_run_lifecycle(tmp_path):
    run_id = new_run_id("labels", "sam3text_v1")
    assert run_id == "labels_sam3text_v1"

    run_dir = tmp_path / run_id
    record = start_run(run_dir, config={"objects": ["curbstone"]},
                       inputs=["inventory/20260714_inventory_v1"])
    assert record["status"] == "running"
    assert (run_dir / "run.json").exists()

    finish_run(run_dir, summary={"labeled": 42})
    saved = json.loads((run_dir / "run.json").read_text())
    assert saved["status"] == "completed"
    assert saved["summary"]["labeled"] == 42
    assert saved["config"]["objects"] == ["curbstone"]
    assert saved["inputs"] == ["inventory/20260714_inventory_v1"]
