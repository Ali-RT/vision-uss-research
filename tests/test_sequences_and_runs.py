import json

from vision_uss_research.runs import finish_run, new_run_id, start_run
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


def test_run_lifecycle(tmp_path):
    run_id = new_run_id("labels", "sam3text_v1")
    assert run_id.endswith("_labels_sam3text_v1")

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
