import pandas as pd

from vision_uss_research.qa.flags import FlagThresholds, box_iou, compute_flags


def _row(seq, obj, idx, x0, y0, x1, y1, area):
    return dict(sequence_id=seq, target_object=obj, camera="rear", frame_idx=idx,
                frame_path=f"{seq}/rear/{idx:05d}.jpg",
                x0=x0, y0=y0, x1=x1, y1=y1, mask_area_frac=area)


def test_box_iou():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_flags_big_jump_and_per_class_tiny():
    rows = [
        # legit tiny pole boxes (area 0.0006 > pole threshold 0.0003) - NOT flagged
        _row("s1", "pole", 0, 470, 300, 490, 380, 0.0006),
        _row("s1", "pole", 1, 471, 300, 491, 380, 0.0006),
        # exploded curbstone box
        _row("s2", "curbstone", 0, 0, 0, 940, 620, 0.7),
        # woodenboard jump between frames
        _row("s3", "woodenboard", 0, 100, 500, 300, 520, 0.004),
        _row("s3", "woodenboard", 1, 700, 100, 800, 140, 0.004),
        # curbstone below the DEFAULT tiny threshold - flagged
        _row("s4", "curbstone", 0, 10, 10, 14, 14, 0.0001),
    ]
    flags = compute_flags(pd.DataFrame(rows), FlagThresholds())
    by_seq = flags.groupby("sequence_id")["flags"].apply("|".join).to_dict()
    assert "s1" not in by_seq                       # per-class tiny threshold respected
    assert "big_box" in by_seq["s2"]
    assert "jump" in by_seq["s3"]
    assert "tiny_box" in by_seq["s4"]


def test_per_class_aspect_thresholds():
    rows = [
        # head-on curb: 900x40 px, aspect 22.5 - normal geometry, NOT flagged
        _row("c1", "curbstone", 0, 30, 500, 930, 540, 0.06),
        # vertical pole: 16x260 px, inverse aspect 16 - normal, NOT flagged
        _row("p1", "pole", 0, 470, 200, 486, 460, 0.007),
        # compact object with absurd aspect - flagged under the default limit
        _row("d1", "dummychild", 0, 100, 300, 500, 320, 0.013),
        # even a curb has limits: aspect ~90 -> flagged
        _row("c2", "curbstone", 0, 15, 500, 915, 510, 0.015),
    ]
    flags = compute_flags(pd.DataFrame(rows), FlagThresholds())
    flagged_seqs = set(flags["sequence_id"])
    assert "c1" not in flagged_seqs
    assert "p1" not in flagged_seqs
    assert "d1" in flagged_seqs
    assert "c2" in flagged_seqs


def test_cone_tiny_threshold():
    rows = [
        # cone at median area 0.0009 - legit, NOT flagged (per-class 0.0002)
        _row("k1", "cone", 0, 470, 380, 500, 410, 0.0009),
        # collapsed cone mask - still flagged
        _row("k2", "cone", 0, 480, 390, 484, 394, 0.00001),
    ]
    flags = compute_flags(pd.DataFrame(rows), FlagThresholds())
    flagged = set(flags["sequence_id"])
    assert "k1" not in flagged and "k2" in flagged
