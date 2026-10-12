import numpy as np

from vision_uss_research.silver_gt import combine, parse_qwen


def test_parse_qwen_scales_back_and_handles_not_visible():
    vis, box = parse_qwen('```json\n{"visible": true, "bbox_2d": [322, 238, 644, 476]}\n```',
                          resized_wh=(644, 476), orig_wh=(640, 480))
    assert vis and np.allclose(box, [320, 240, 639, 479], atol=1.5)
    assert parse_qwen('{"visible": false}', (644, 476), (640, 480)) == (False, None)
    assert parse_qwen("no idea", (644, 476), (640, 480)) == (None, None)


def test_combine_agree_disagree_and_not_visible():
    a = combine([100, 100, 200, 200], True, np.array([105, 100, 205, 200]), (640, 480))
    assert a["status"] == "box" and a["x0"] == 100 and a["truncated"] == 0
    cut = combine([0, 300, 200, 479], True, np.array([0, 300, 200, 479]), (640, 480))
    assert cut["truncated"] == 1
    assert combine([100, 100, 200, 200], True, np.array([400, 10, 500, 60]), (640, 480))["status"] == "disputed"
    assert combine(None, False, None, (640, 480))["status"] == "not_visible"
    assert combine(None, True, np.array([1, 1, 50, 50]), (640, 480))["reason"] == "only qwen sees it"
