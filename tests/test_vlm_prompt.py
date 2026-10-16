import numpy as np

from vision_uss_research.method.vlm_prompt import (clean_name, name_prompt,
                                                   padded_crop, parse_name_box)


def test_parse_name_and_scaled_box():
    name, box = parse_name_box('```json\n{"name": "The Hedge", "bbox_2d": [10, 20, 110, 70]}\n```',
                               resized_wh=(200, 100), orig_wh=(400, 200))
    assert name == "hedge"
    assert np.allclose(box, [20, 40, 220, 140])


def test_parse_null_and_garbage():
    assert parse_name_box('{"name": null}', (100, 100), (100, 100)) == (None, None)
    assert parse_name_box("no json here", (100, 100), (100, 100)) == (None, None)
    name, box = parse_name_box('{"name": "car"}', (100, 100), (100, 100))
    assert name == "car" and box is None


def test_clean_name():
    assert clean_name("A traffic cone!") == "traffic cone"
    assert clean_name("the") is None
    assert clean_name(None) is None


def test_prompt_views_differ_only_by_uss_sentence():
    assert "ultrasonic" in name_prompt("crop") and "ultrasonic" not in name_prompt("full")


def test_padded_crop_clips():
    assert padded_crop((100, 100, 200, 200), 960, 640) == (80, 80, 220, 220)
    assert padded_crop((0, 0, 100, 100), 960, 640)[:2] == (0, 0)


def test_padded_crop_off_image_region_is_none():
    from vision_uss_research.method.uss_prompt import region_box
    reg = region_box(-400, 300, 960, 640)            # projected far left of the frame
    assert reg[2] < reg[0]                           # clipping inverts it
    assert padded_crop(reg, 960, 640) is None


def test_drive_over_and_lexicon():
    from vision_uss_research.method.vlm_prompt import (decision_prompt, name_to_bin,
                                                       parse_drive_over)
    assert "drive over" in decision_prompt("full") and "ultrasonic" in decision_prompt("crop")
    assert parse_drive_over('{"name": "curb", "drive_over": true}') is True
    assert parse_drive_over('{"name": "pole", "drive_over": false}') is False
    assert parse_drive_over('{"name": "pole"}') is None
    assert name_to_bin("Concrete curb") == "low"
    assert name_to_bin("speed bump") == "low"
    assert name_to_bin("traffic cone") == "high"
    assert name_to_bin(None) is None
