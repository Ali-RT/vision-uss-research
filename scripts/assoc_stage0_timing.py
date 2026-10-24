"""Association study, Stage 0: frame identity and timing audit (CPU only).

DEPTH_USS_ASSOCIATION_PROPOSAL.md §13 releases this audit and nothing else:
no model, no annotation, no association parameter. Cached imagery, frozen
Step 4 files and the extraction code are read, never modified; every output
goes under <artifacts>/assoc/stage0_timing_v1/.

Question: which decoded video frame IS each cached window JPEG
(frames/r30-85_n20/<seq>/<camera>/<i>.jpg), what is its real presentation
timestamp (PTS), and how far is its MF4 time from the nominal time that
notebook 20 assumed (window_video_seconds on a frame_count/fps duration)?

Per clip (sequence x camera):
  1. decode every frame with OpenCV (the extraction decoder) -> (index k,
     PTS, sha256 of its q92 re-encoding, gray image); PyAV PTS cross-check;
  2. identify each cached JPEG: exact byte match first, else a full-resolution
     pixel match under the declared tolerances below; bit-identical frames with
     different PTS, and weak pixel margins, are left unresolved, never guessed;
  3. record the index hypotheses (floor(r*N), round(r*N), OpenCV's own seek)
     next to the matched index - hypotheses are checked, never used;
  4. map the matched PTS to MF4 time through the camera channel, by exact
     sample-value match (tolerance below) - not by interpolation;
  5. residual = true MF4 time - nominal MF4 time (notebook 20 rule).
Camera-time mapping is audited separately from USS freshness: the freshness
table reports how often map-object and echo VALUES change; 100 Hz logging is
not taken as a 100 Hz measurement. Odometry displacement between nominal and
true time is reported in the channel's raw units; no speed unit is assumed.

usage:
  python scripts/assoc_stage0_timing.py sample --manifest M.csv --out configs/assoc/stage0_sample_v1.csv
  python scripts/assoc_stage0_timing.py audit --sample configs/assoc/stage0_sample_v1.csv \\
      --raw-root RAW --frames-root ARTIFACTS/frames/r30-85_n20 --out ARTIFACTS/assoc/stage0_timing_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

RESERVED = REPO / "configs" / "splits" / "reserved_sessions_v1.csv"
N_WINDOW, RATIO = 20, (0.30, 0.85)          # the cached window r30-85_n20
JPEG_QUALITY = 92                          # extract_sequence_frames default
# Declared before the audit run (full-resolution gray, mean absolute difference
# 0-255). Identity routes, strongest first:
#   exact  - the cached JPEG's bytes equal OpenCV's q92 re-encoding of decoded
#            frame k (same decoder + encoder => bit-identical); unique hash -> verified
#   pixel  - no byte match (e.g. a different OpenCV/libjpeg at extraction time):
#            nearest decoded frame by MAD, with MAD <= PIXEL_MAX_MAD and a margin
#            >= PIXEL_MARGIN to the next frame -> "pixel_unique" (strong, not exact);
#            margin in [PIXEL_WEAK_MARGIN, PIXEL_MARGIN) -> "pixel_weak" (unresolved,
#            typically a stationary vehicle); below that -> "ambiguous"
PIXEL_MAX_MAD = 4.0
PIXEL_MARGIN = 0.5
PIXEL_WEAK_MARGIN = 0.05
PTS_XCHECK_TOL_S = 0.001   # OpenCV POS_MSEC vs PyAV PTS, per decoded frame
CHANNEL_TOL_S = 0.002
IDENTIFIED = ("verified_exact", "pixel_unique")   # frame identity established    # camera-channel sample value vs decoded PTS
SAMPLE_SEED = 0
MIN_SEQUENCES = 20
MAP_SLOTS = 20
ODO = ("CEP_vhmCurrentPos_XPosition", "CEP_vhmCurrentPos_YPosition")


def reserved_ids() -> set[str]:
    ids = set(pd.read_csv(RESERVED).sequence_id)
    assert len(ids) == 154, f"reserved list has {len(ids)} ids, expected 154"
    return ids


def rig_of(seq: str) -> str:
    """Rig token of a sequence name: LB_XO2617 / LB-XO2617 / LB-XN / LB-XK /
    S-063091 ...; "unrecorded" when the name carries none."""
    p = seq.split("_")
    for i, tok in enumerate(p[1:], 1):
        if tok == "LB" and i + 1 < len(p) and p[i + 1].startswith("X"):
            return f"LB_{p[i + 1]}"
        if re.fullmatch(r"(LB-X\w+|S-\d+)", tok):
            return tok
    return "unrecorded"


def day_of(seq: str) -> str:
    """Recording date token (YYYYMMDD); "unrecorded" when absent."""
    return next((t for t in seq.split("_")[1:] if re.fullmatch(r"20\d{6}", t)), "unrecorded")


def build_sample(manifest: Path) -> pd.DataFrame:
    """One sequence per recording day (seeded), preferring sequences with both
    cameras cached; then every rig group with < 2 sequences topped up to 2.
    Uses manifest metadata only - no imagery, no sensor data."""
    m = pd.read_csv(manifest, usecols=["sequence_id", "camera"]).drop_duplicates()
    res = reserved_ids()
    m = m[~m.sequence_id.isin(res)]
    s = m.groupby("sequence_id").camera.agg(lambda c: ",".join(sorted(set(c)))).rename("cameras").reset_index()
    s["day"] = s.sequence_id.map(day_of)
    s["rig"] = s.sequence_id.map(rig_of)
    s["both"] = s.cameras.str.count(",") > 0
    rng = np.random.default_rng(SAMPLE_SEED)
    s["key"] = rng.random(len(s))
    s = s.sort_values(["both", "key"], ascending=[False, True])
    pick = s[s.day != "unrecorded"].groupby("day", sort=True).head(1)
    for rig, g in s.groupby("rig"):
        have = (pick.rig == rig).sum()
        if have < 2:
            pick = pd.concat([pick, g[~g.sequence_id.isin(pick.sequence_id)].head(2 - have)])
    pick = pick.sort_values("sequence_id")
    assert len(pick) >= MIN_SEQUENCES and not set(pick.sequence_id) & res
    rows = [{"sequence_id": r.sequence_id, "camera": c, "day": r.day, "rig": r.rig}
            for r in pick.itertuples() for c in r.cameras.split(",")]
    return pd.DataFrame(rows)


def gray(bgr: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def decode_all(video: Path):
    """Every frame in decode order with OpenCV (the extraction decoder): PTS from
    POS_MSEC, sha256 of its q92 JPEG re-encoding, full-resolution gray. PyAV
    PTS are read independently as a cross-check of the PTS and frame count."""
    import av
    import cv2
    cap = cv2.VideoCapture(str(video))
    pts, sha, g = [], [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        pts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        sha.append(hashlib.sha256(cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes()).hexdigest())
        g.append(gray(fr))
    cap.release()
    with av.open(str(video)) as c:
        st = c.streams.video[0]
        av_pts = [float(p.pts * st.time_base) for p in c.demux(st) if p.pts is not None]
    return np.array(pts), sha, g, np.sort(np.array(av_pts))


def opencv_nominal(video: Path):
    """notebook 20 / extract_sequence_frames rule, replayed: duration from
    frame_count / fps; for each ratio, the seek time and OpenCV's resulting
    frame position (hypothesis, not identity)."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    dur_ms = 1000.0 * n / max(fps, 1e-6)
    out = []
    for r in np.linspace(*RATIO, N_WINDOW):
        cap.set(cv2.CAP_PROP_POS_MSEC, r * dur_ms)
        ok, _ = cap.read()
        out.append(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1 if ok else -1)
    cap.release()
    return n, fps, dur_ms / 1000.0, out


def match(cached_bytes: bytes, sha: list[str], g: list[np.ndarray], pts: np.ndarray) -> dict:
    import cv2
    h = hashlib.sha256(cached_bytes).hexdigest()
    hits = [k for k, s in enumerate(sha) if s == h]
    if len(hits) == 1:
        return {"k": hits[0], "route": "exact", "status": "verified_exact", "n_exact_hits": 1}
    if len(hits) > 1:      # bit-identical decoded frames: identity unresolved unless one PTS
        same = np.ptp(pts[hits]) == 0
        return {"k": hits[0], "route": "exact", "n_exact_hits": len(hits),
                "status": "verified_exact" if same else "ambiguous_identical",
                "pts_min": float(pts[hits].min()), "pts_max": float(pts[hits].max())}
    c = gray(cv2.imdecode(np.frombuffer(cached_bytes, np.uint8), cv2.IMREAD_COLOR)).astype(np.int16)
    d = np.array([np.abs(x.astype(np.int16) - c).mean() for x in g])
    o = np.argsort(d)
    k1, margin = int(o[0]), float(d[o[1]] - d[o[0]]) if len(o) > 1 else np.inf
    rec = {"k": k1, "route": "pixel", "n_exact_hits": 0, "mad_best": round(float(d[k1]), 3),
           "mad_margin": round(margin, 3), "k_second": int(o[1]) if len(o) > 1 else -1}
    if d[k1] > PIXEL_MAX_MAD:
        rec["status"] = "no_match"
    elif margin >= PIXEL_MARGIN:
        rec["status"] = "pixel_unique"
    elif margin >= PIXEL_WEAK_MARGIN:
        rec["status"] = "pixel_weak"
    else:
        rec["status"] = "ambiguous"
    return rec


def change_intervals(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    live = v != 0
    ch = np.flatnonzero((np.diff(v) != 0) & live[1:] & live[:-1]) + 1
    return np.diff(t[ch])


def audit_clip(row, raw_root: Path, frames_root: Path, res: set[str]):
    import cv2
    from asammdf import MDF

    from vision_uss_research.alignment.mf4 import find_camera_channel, sorted_pairs, video_to_mf4
    from vision_uss_research.sequences import sequence_files, with_drive_retry

    seq, cam = row.sequence_id, row.camera
    assert seq not in res, f"reserved sequence {seq} reached the audit"   # before any file is opened
    clip = {"sequence_id": seq, "camera": cam, "day": row.day, "rig": row.rig, "error": ""}
    frames = []
    try:
        seq_dir = raw_root / seq
        found = with_drive_retry(sequence_files, seq_dir)
        video = found.get(f"{cam}_video")
        if video is None:
            raise FileNotFoundError("no_video")
        pts, sha, g, av_pts = decode_all(video)
        n_cv, fps_cv, dur_s, cv_idx = opencv_nominal(video)
        dpts = np.diff(pts)
        xok = len(av_pts) == len(pts) and bool(np.all(np.abs(av_pts - pts) <= PTS_XCHECK_TOL_S))
        clip.update(n_decoded=len(pts), n_pyav_packets=len(av_pts), pts_crosscheck_ok=xok,
                    n_identical_decoded=int(len(sha) - len(set(sha))), n_opencv=int(n_cv), fps_opencv=round(fps_cv, 4), duration_nominal_s=round(dur_s, 3),
                    pts_last=round(float(pts[-1]), 3), pts_nonmonotonic=int((dpts <= 0).sum()),
                    pts_duplicate=int(len(pts) - len(np.unique(pts))), pts_nan=int(np.isnan(pts).sum()),
                    dt_min=round(float(np.nanmin(dpts)), 3), dt_median=round(float(np.nanmedian(dpts)), 3),
                    dt_max=round(float(np.nanmax(dpts)), 3),
                    dt_cv=round(float(np.nanstd(dpts) / np.nanmean(dpts)), 3))
        # case-insensitive: several rigs store .MF4 (stage0_timing_v1 missed them)
        mf4 = next(iter(sorted(p for p in seq_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mf4")), None)
        if mf4 is None:
            raise FileNotFoundError("no_mf4")
        mdf = MDF(str(mf4))
        try:
            names = set(mdf.channels_db)
            ch = find_camera_channel(names, cam)
            clip["camera_channel"] = ch or ""
            if ch:
                sig = mdf.get(ch)
                cv_, ct = np.asarray(sig.samples, float), np.asarray(sig.timestamps, float)
                clip.update(channel_samples=len(cv_), channel_value_nonmonotonic=int((np.diff(cv_) <= 0).sum()),
                            channel_time_nonmonotonic=int((np.diff(ct) <= 0).sum()),
                            channel_count_equals_decoded=bool(len(cv_) == len(pts)))
                spts, smt = sorted_pairs(cv_, ct)
            odo = {c: mdf.get(c) for c in ODO if c in names}
            fresh = {}
            mi = []
            for n in range(1, MAP_SLOTS + 1):
                c = f"Map_Obj{n}_P1x"
                if c in names:
                    s = mdf.get(c)
                    mi.append(change_intervals(np.asarray(s.timestamps, float), np.asarray(s.samples, float)))
            mi = np.concatenate(mi) if mi else np.array([])
            ei = []
            for n in range(1, 13):
                c = f"CEP_{n}_DE_0_EchoDist"
                if c in names:
                    s = mdf.get(c)
                    ei.append(change_intervals(np.asarray(s.timestamps, float), np.asarray(s.samples, float)))
            ei = np.concatenate(ei) if ei else np.array([])
            for tag, x in (("mapobj", mi), ("echo", ei)):
                fresh[f"{tag}_change_n"] = len(x)
                for q in (50, 95):
                    fresh[f"{tag}_change_p{q}_s"] = round(float(np.percentile(x, q)), 4) if len(x) else np.nan
            clip.update(fresh)
        finally:
            mdf.close()

        nominal_s = np.linspace(*RATIO, N_WINDOW) * dur_s          # notebook 20 assumption
        cdir = frames_root / seq / cam
        for i, r in enumerate(np.linspace(*RATIO, N_WINDOW)):
            f = {"sequence_id": seq, "camera": cam, "frame_idx": i, "ratio": round(float(r), 4),
                 "nominal_video_s": round(float(nominal_s[i]), 4),
                 "hyp_floor_rN": int(np.floor(r * len(pts))), "hyp_round_rN": int(np.round(r * len(pts))),
                 "hyp_opencv_seek": cv_idx[i]}
            p = cdir / f"{i:05d}.jpg"
            if not p.exists():
                f["status"] = "cached_missing"
                frames.append(f)
                continue
            b = p.read_bytes()
            f["cached_sha256"] = hashlib.sha256(b).hexdigest()
            f.update(match(b, sha, g, pts))
            k = f["k"]
            f["true_pts_s"] = round(float(pts[k]), 4)
            f["video_residual_s"] = round(float(pts[k] - nominal_s[i]), 4)
            for h in ("hyp_floor_rN", "hyp_round_rN", "hyp_opencv_seek"):
                f[h + "_ok"] = bool(f[h] == k)
            if ch:
                j = int(np.argmin(np.abs(spts - pts[k])))
                f["channel_pts_err_s"] = round(float(spts[j] - pts[k]), 5)
                f["channel_ok"] = bool(abs(spts[j] - pts[k]) <= CHANNEL_TOL_S)
                f["true_mf4_s"] = round(float(smt[j]), 4)
                f["nominal_mf4_s"] = round(float(video_to_mf4(nominal_s[i], spts, smt)), 4)
                f["mf4_residual_s"] = round(f["true_mf4_s"] - f["nominal_mf4_s"], 4)
                if len(odo) == 2:
                    xy = [np.interp([f["nominal_mf4_s"], f["true_mf4_s"]], np.asarray(s.timestamps, float),
                                    np.asarray(s.samples, float)) for s in odo.values()]
                    f["odo_displacement_raw_units"] = round(float(np.hypot(xy[0][1] - xy[0][0], xy[1][1] - xy[1][0])), 1)
            f["pts_crosscheck_ok"] = xok
            if f["status"] in IDENTIFIED and not (f.get("channel_ok", False) and xok):
                f["status"] += "_unmapped"
            frames.append(f)
    except Exception as e:  # noqa: BLE001 - every failure is recorded, none silent
        clip["error"] = f"{type(e).__name__}:{e}"[:200]
    return clip, frames


def summarize(clips: pd.DataFrame, frames: pd.DataFrame) -> dict:
    ok = frames[frames.status.isin(IDENTIFIED)]
    weak = frames[frames.status == "pixel_weak"]
    full = frames.groupby(["sequence_id", "camera"]).status.apply(lambda s: s.isin(IDENTIFIED).all())

    def q(x):
        x = pd.Series(x).dropna()
        return {f"p{p}": round(float(np.percentile(x, p)), 3) for p in (0, 5, 25, 50, 75, 95, 100)} if len(x) else {}

    def col(df, c):
        return df[c] if c in df else pd.Series(dtype=float)
    return {
        "clips": len(clips), "clips_error": int((clips.error != "").sum()),
        "clip_errors": clips.loc[clips.error != "", "error"].str.split(":").str[0].value_counts().to_dict(),
        "sequences": int(clips.sequence_id.nunique()), "days": int(clips.day.nunique()),
        "rigs": clips.rig.value_counts().to_dict(),
        "frames": len(frames), "frame_status": frames.status.value_counts().to_dict(),
        "clips_all_frames_identified": int(full.sum()),
        "clips_all_frames_identified_frac": round(float(full.mean()), 4) if len(full) else None,
        "hypothesis_agreement_on_identified": {h: round(float(ok[h + "_ok"].mean()), 4) for h in
                                               ("hyp_floor_rN", "hyp_round_rN", "hyp_opencv_seek")} if len(ok) else {},
        "pixel_route_mad_best": q(col(ok[ok.route == "pixel"], "mad_best")) if len(ok) else {},
        "pixel_route_margin": q(col(frames[frames.route == "pixel"], "mad_margin")) if "route" in frames else {},
        "identified_video_residual_s": q(col(ok, "video_residual_s")),
        "identified_mf4_residual_s": q(col(ok, "mf4_residual_s")),
        "identified_abs_mf4_residual_s": q(col(ok, "mf4_residual_s").abs()),
        "identified_odo_displacement_raw_units": q(col(ok, "odo_displacement_raw_units")),
        "pixel_weak_odo_displacement_raw_units": q(col(weak, "odo_displacement_raw_units")),
        "pts_crosscheck_ok": col(clips, "pts_crosscheck_ok").value_counts().to_dict(),
        "channel_count_equals_decoded": col(clips, "channel_count_equals_decoded").value_counts().to_dict(),
        "frame_interval_cv": q(col(clips, "dt_cv")),
        "mapobj_change_p50_s": q(col(clips, "mapobj_change_p50_s")),
        "echo_change_p50_s": q(col(clips, "echo_change_p50_s")),
        "note": "audit diagnostics only; meeting any rate does not verify every frame or release later stages",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("sample")
    a1.add_argument("--manifest", type=Path, required=True)
    a1.add_argument("--out", type=Path, required=True)
    a2 = sub.add_parser("audit")
    a2.add_argument("--sample", type=Path, required=True)
    a2.add_argument("--raw-root", type=Path, required=True)
    a2.add_argument("--frames-root", type=Path, required=True)
    a2.add_argument("--out", type=Path, required=True)
    a2.add_argument("--workers", type=int, default=4)
    a2.add_argument("--limit", type=int, default=0, help="smoke: first N clips (output dir gets _smoke)")
    a = ap.parse_args()

    if a.cmd == "sample":
        s = build_sample(a.manifest)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        s.to_csv(a.out, index=False)
        print(f"{s.sequence_id.nunique()} sequences, {len(s)} clips, {s.day.nunique()} days, rigs {s.rig.value_counts().to_dict()}")
        print("sha256", hashlib.sha256(a.out.read_bytes()).hexdigest())
        return

    res = reserved_ids()
    s = pd.read_csv(a.sample, dtype=str)
    bad = set(s.sequence_id) & res
    assert not bad, f"sample contains reserved ids: {sorted(bad)[:5]}"   # asserted before opening anything
    out = a.out.with_name(a.out.name + "_smoke") if a.limit else a.out
    if a.limit:
        s = s.head(a.limit)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as pool:
        results = list(pool.map(lambda r: audit_clip(r, a.raw_root, a.frames_root, res), s.itertuples()))
    clips = pd.DataFrame([c for c, _ in results])
    frames = pd.DataFrame([f for _, fr in results for f in fr])
    if frames.empty:
        frames = pd.DataFrame(columns=["sequence_id", "camera", "status"])
    clips.to_csv(out / "clips_audit.csv", index=False)
    frames.to_csv(out / "frames_audit.csv", index=False)
    import av
    summary = summarize(clips, frames)
    meta = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "runtime_s": round(time.time() - t0, 1),
            "sample": str(a.sample), "sample_sha256": hashlib.sha256(a.sample.read_bytes()).hexdigest(),
            "reserved_excluded": len(res), "limit": a.limit,
            "tolerances": {"jpeg_quality": JPEG_QUALITY, "pixel_max_mad": PIXEL_MAX_MAD, "pixel_margin": PIXEL_MARGIN,
                           "pixel_weak_margin": PIXEL_WEAK_MARGIN, "pts_xcheck_tol_s": PTS_XCHECK_TOL_S,
                           "channel_tol_s": CHANNEL_TOL_S},
            "versions": {"python": sys.version.split()[0], "av": av.__version__,
                         "cv2": __import__("cv2").__version__, "asammdf": __import__("asammdf").__version__}}
    try:
        from vision_uss_research.runs import git_sha
        meta["git_sha"] = git_sha(REPO)
    except Exception:  # noqa: BLE001
        meta["git_sha"] = None
    (out / "summary.json").write_text(json.dumps({"meta": meta, "summary": summary}, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
