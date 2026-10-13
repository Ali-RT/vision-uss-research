import numpy as np

from vision_uss_research.click_track import (apply_cutoffs, mask_to_box, merge_by_nearest_click,
                                             stop_after_loss, to_gold_rows)


def test_mask_to_box_and_min_area():
    m = np.zeros((480, 640), bool)
    m[100:120, 200:260] = True
    assert mask_to_box(m) == (200.0, 100.0, 259.0, 119.0)
    assert mask_to_box(m, min_area=5000) is None


def test_stop_after_loss_blocks_relock():
    b = (1, 1, 5, 5)
    seq = [b, b, None, b, None, None, None, b, b]       # loss of 3 = left the view
    assert stop_after_loss(seq, stop_after=3) == [b, b, None, b, None, None, None, None, None]
    assert stop_after_loss([b, None, b], stop_after=3) == [b, None, b]  # short dropout kept


def test_merge_by_nearest_click_and_cutoffs():
    a, c = (0, 0, 1, 1), (9, 9, 10, 10)
    tracks = {10: {f: a for f in range(0, 30)}, 25: {f: c for f in range(15, 40)}}
    m = merge_by_nearest_click(list(range(0, 40)), tracks)
    assert m[12] == a and m[18] == c and m[35] == c and m[5] == a
    cut = apply_cutoffs(m, not_visible_after=33, not_visible_before=2)
    assert cut[34] is None and cut[1] is None and cut[20] == c


def test_to_gold_rows():
    df = to_gold_rows("rear", 3157, "curbhigh", {5: (1, 2, 3, 4), 6: None})
    assert df["status"].tolist() == ["box", "not_visible"]
    assert df["item_id"].iloc[0] == "rear_3157_00005"
