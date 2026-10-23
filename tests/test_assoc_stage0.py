"""Stage 0 frame-identity rules of scripts/assoc_stage0_timing.py on synthetic data."""
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import assoc_stage0_timing as s0  # noqa: E402


def _frames(n=5, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (64, 96, 3), dtype=np.uint8) for _ in range(n)]


def _decoded(frames):
    sha = [hashlib.sha256(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()).hexdigest()
           for f in frames]
    return sha, [s0.gray(f) for f in frames]


def test_rig_and_day_parsing():
    assert s0.rig_of("110613_20190308_LB_XO2617_039_151540") == "LB_XO2617"
    assert s0.rig_of("128868_20190823_164047_LB-XN_8826_SIP6") == "LB-XN"
    assert s0.rig_of("142203_20201124_S-063091_170002") == "S-063091"
    assert s0.rig_of("1_2_3") == "unrecorded"
    assert s0.day_of("133670_102206_LB-XK_7683") == "unrecorded"
    assert s0.day_of("142203_20201124_S-063091_170002") == "20201124"


def test_exact_byte_match_identifies_frame():
    fr = _frames()
    sha, g = _decoded(fr)
    cached = cv2.imencode(".jpg", fr[3], [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
    r = s0.match(cached, sha, g, np.arange(5) * 0.4)
    assert (r["k"], r["status"], r["route"]) == (3, "verified_exact", "exact")


def test_identical_frames_with_different_pts_stay_unresolved():
    fr = _frames()
    fr[4] = fr[2].copy()
    sha, g = _decoded(fr)
    cached = cv2.imencode(".jpg", fr[2], [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
    r = s0.match(cached, sha, g, np.arange(5) * 0.4)
    assert r["status"] == "ambiguous_identical" and r["n_exact_hits"] == 2


def test_pixel_route_when_encoder_differs():
    fr = _frames()
    sha, g = _decoded(fr)
    cached = cv2.imencode(".jpg", fr[1], [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
    r = s0.match(cached, sha, g, np.arange(5) * 0.4)
    assert r["route"] == "pixel" and r["k"] == 1


def test_pixel_route_near_duplicates_are_weak_or_ambiguous():
    yy, xx = np.mgrid[0:64, 0:96]
    base = np.repeat(((xx * 2 + yy) % 256).astype(np.uint8)[..., None], 3, axis=2)   # smooth, JPEG-friendly
    fr = [base, np.clip(base.astype(int) + 1, 0, 255).astype(np.uint8), 255 - base]
    sha, g = _decoded(fr)
    cached = cv2.imencode(".jpg", base, [cv2.IMWRITE_JPEG_QUALITY, 60])[1].tobytes()
    r = s0.match(cached, sha, g, np.arange(3) * 0.4)
    assert r["status"] in ("pixel_weak", "ambiguous")


def test_sample_excludes_reserved(tmp_path):
    res = pd.read_csv(s0.RESERVED).sequence_id.tolist()
    rows = [{"sequence_id": f"{i}_202001{d:02d}_LB_XO2617_{i}", "camera": c}
            for i, d in enumerate(range(1, 26)) for c in ("front", "rear")]
    rows += [{"sequence_id": res[0], "camera": "front"}]
    m = tmp_path / "m.csv"
    pd.DataFrame(rows).to_csv(m, index=False)
    s = s0.build_sample(m)
    assert res[0] not in set(s.sequence_id)
    assert s.sequence_id.nunique() >= s0.MIN_SEQUENCES
    assert s.groupby("day").sequence_id.nunique().max() == 1


def test_sample_too_small_fails(tmp_path):
    m = tmp_path / "m.csv"
    pd.DataFrame([{"sequence_id": "1_20200101_LB_XO2617_1", "camera": "front"}]).to_csv(m, index=False)
    with pytest.raises(AssertionError):
        s0.build_sample(m)
