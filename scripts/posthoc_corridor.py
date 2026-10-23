"""POST HOC measurement (not part of the pre-registered Step 4 result):
is the detector's selected box in the vehicle's driving corridor?

Settings fixed before any corridor number was computed (2026-10-23):
  corridor     ground region of the driving path: |y| <= 1200 mm (the
               USS track rule's lateral limit), from the bumper of the frame's
               camera to 6000 mm beyond it, in that camera's facing direction
               (front: x 4200..10200 mm; rear: x -1300..-7300 mm)
  footprint    bottom-centre of the selected box ((x0+x1)/2, y1)
  inside       footprint inside the corridor polygon projected into the image
               (edges sampled every 100 mm; points behind the camera dropped)
  calibration  per (camera, day) ground-camera fits on known-type boxes, as in
               notebook 24's tier C mode (held-out types of every rotation
               excluded); a sequence whose own boxes are in a fit uses the
               other fold; global fit when a day has < 25 pairs
VALIDATION (added after the first smoke number, 2026-10-23): ground-truth
target boxes of the evaluated frames must lie inside. The geometric corridor
FAILED it: only 25% of bush tier D target footprints were inside (the USS-fit
cameras are too coarse and a box bottom is not always ground contact). It is
kept as `--mode geometric` for the record only.
`--mode empirical` (default): a target-location prior per rotation and camera
- the highest-density region (16 px grid) covering 90% of the footprints of
that rotation's TRAIN-split ground-truth boxes (never its test frames).
Reported: share of frames whose selected box lies outside the region, by
rotation / population / type / camera-vs-driving-direction, and how that
share differs between sequences each arm got right and wrong.

usage: python scripts/posthoc_corridor.py --results paper/results \\
           --det-dir paper/results/step4_det_test --out paper/results/posthoc_corridor
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MPath

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import uss_prompt_analysis as upa  # noqa: E402
from vision_uss_research.method.uss_prompt import (FRONT_BUMPER_MM, REAR_BUMPER_MM,  # noqa: E402
                                                   fit_ground_camera, track_target_slots)

LATERAL_MM, RANGE_MM, STEP_MM = 1200.0, 6000.0, 100.0
KEYS = ["sequence_id", "camera", "frame_idx"]
HELDOUT_ALL = {"woodenboard", "speedbump", "bush", "bollard", "tree", "cubestandard", "hose"}
ROTATIONS = ("woodenboard", "speedbump", "bush", "bollard", "treecubehose")


def corridor_ground(camera: str) -> tuple[np.ndarray, np.ndarray]:
    """Closed boundary of the corridor on the ground (vehicle frame, mm)."""
    if camera == "front":
        x0, x1 = FRONT_BUMPER_MM, FRONT_BUMPER_MM + RANGE_MM
    else:
        x0, x1 = REAR_BUMPER_MM, REAR_BUMPER_MM - RANGE_MM
    xs = np.arange(x0, x1 + np.sign(x1 - x0) * STEP_MM, np.sign(x1 - x0) * STEP_MM)
    ys = np.arange(-LATERAL_MM, LATERAL_MM + STEP_MM, STEP_MM)
    gx = np.r_[xs, np.full(len(ys), xs[-1]), xs[::-1], np.full(len(ys), xs[0])]
    gy = np.r_[np.full(len(xs), -LATERAL_MM), ys, np.full(len(xs), LATERAL_MM), ys[::-1]]
    return gx, gy


def polygon(model, camera: str) -> MPath | None:
    gx, gy = corridor_ground(camera)
    u, v, ok = model.project(gx, gy)
    if ok.sum() < 3:
        return None
    return MPath(np.c_[u[ok], v[ok]])


def calibrations(results: Path):
    slots, boxes = upa.load(results / "prompt_uss_proj_v1", results / "manifests" / "20260718_dataset_v2")
    types = (pd.read_csv(results / "manifests" / "dataset_v2daysplit" / "dataset_manifest.csv",
                         usecols=["sequence_id", "target_object"]).drop_duplicates("sequence_id"))
    pairs = (slots.merge(track_target_slots(slots), on=["sequence_id", "camera", "slot"])
             .merge(boxes, on=KEYS).merge(types, on="sequence_id"))
    pairs["day"] = pairs.sequence_id.str.split("_").str[1]
    pairs = pairs[((pairs.y1 - pairs.y0) >= 15) & ((pairs.x1 - pairs.x0) >= 8)]
    pairs = pairs[~pairs.target_object.isin(HELDOUT_ALL)]
    fold = {}
    for _, d in pairs.groupby(["camera", "day"]):
        for i, s in enumerate(sorted(d.sequence_id.unique())):
            fold[s] = i % 2
    pairs["fold"] = pairs.sequence_id.map(fold)

    def fit(d, cam):
        return fit_ground_camera(d.mid_x, d.mid_y, (d.x0 + d.x1) / 2, d.y1, cam)[0]
    glob_ = {c: fit(pairs[pairs.camera == c], c) for c in ("front", "rear")}
    by_fold = {k: fit(d, k[0]) for k, d in pairs.groupby(["camera", "day", "fold"]) if len(d) >= 25}
    by_day = {k: fit(d, k[0]) for k, d in pairs.groupby(["camera", "day"]) if len(d) >= 25}
    return fold, glob_, by_fold, by_day


def frame_inside(det: pd.DataFrame, cal) -> pd.Series:
    fold, glob_, by_fold, by_day = cal
    cache, out = {}, []
    for r in det.itertuples(index=False):
        if r.crop_status in ("no_detection",) or pd.isna(r.x0):
            out.append(np.nan)
            continue
        day = r.sequence_id.split("_")[1]
        f = fold.get(r.sequence_id)
        key = (r.camera, day, 1 - f) if f is not None else (r.camera, day)
        model = (by_fold.get(key) if f is not None else by_day.get(key)) or glob_[r.camera]
        pk = (id(model), r.camera)
        if pk not in cache:
            cache[pk] = polygon(model, r.camera)
        poly = cache[pk]
        out.append(np.nan if poly is None else bool(poly.contains_point(((r.x0 + r.x1) / 2, r.y1))))
    return pd.Series(out, index=det.index, dtype="object")


GRID_PX, MASS = 16, 0.90


class Region:
    """Highest-density footprint region on a GRID_PX grid covering MASS."""
    def __init__(self, u: np.ndarray, v: np.ndarray):
        iu, iv = (u // GRID_PX).astype(int), (v // GRID_PX).astype(int)
        cells = pd.Series(1, index=pd.MultiIndex.from_arrays([iu, iv])).groupby(level=[0, 1]).sum()
        cells = cells.sort_values(ascending=False)
        keep = cells.cumsum() / cells.sum() <= MASS
        keep.iloc[int(keep.sum())] = True if int(keep.sum()) < len(keep) else keep.iloc[-1]
        self.cells = set(cells.index[keep])

    def contains(self, u: float, v: float) -> bool:
        return (int(u // GRID_PX), int(v // GRID_PX)) in self.cells


def empirical_regions(results: Path) -> dict:
    ds = {"woodenboard": "dataset_v2holdwoodenboard", "speedbump": "dataset_v2holdspeedbump",
          "bush": "dataset_v2holdbush", "bollard": "dataset_v2holdbollard",
          "treecubehose": "dataset_v2holdtreecubehose"}
    out = {}
    for rot, d in ds.items():
        m = pd.read_csv(results / "manifests" / d / "dataset_manifest.csv")
        m = m[m.split == "train"]
        for cam, g in m.groupby("camera"):
            out[(rot, cam)] = Region(((g.x0 + g.x1) / 2).to_numpy(), g.y1.to_numpy())
    return out


def inside_empirical(det: pd.DataFrame, regions: dict, rot: str) -> pd.Series:
    out = [np.nan if (r.crop_status == "no_detection" or pd.isna(r.x0)) else
           bool(regions[(rot, r.camera)].contains((r.x0 + r.x1) / 2, r.y1)) for r in det.itertuples(index=False)]
    return pd.Series(out, index=det.index, dtype="object")


def gt_validation(results: Path, rot: str, inside_fn) -> float:
    """Share of the evaluated frames' ground-truth target boxes inside."""
    ds = {"woodenboard": "dataset_v2holdwoodenboard", "speedbump": "dataset_v2holdspeedbump",
          "bush": "dataset_v2holdbush", "bollard": "dataset_v2holdbollard",
          "treecubehose": "dataset_v2holdtreecubehose"}[rot]
    fl = REPO / "configs" / "step4" / (f"tierd_frames_{rot}.csv" if rot != "treecubehose"
                                        else "tierc_frames_treecubehose.csv")
    m = pd.read_csv(results / "manifests" / ds / "dataset_manifest.csv").merge(pd.read_csv(fl)[KEYS], on=KEYS)
    m = m.assign(crop_status="valid")
    return float(inside_fn(m).astype(bool).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=REPO / "paper" / "results")
    ap.add_argument("--det-dir", type=Path, required=True, help="folder with det_<rotation>_test/crops.csv")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["empirical", "geometric"], default="empirical")
    a = ap.parse_args()
    if a.mode == "geometric":
        cal = calibrations(a.results)
        inside_for = lambda rot: (lambda d: frame_inside(d, cal))  # noqa: E731
    else:
        regions = empirical_regions(a.results)
        inside_for = lambda rot: (lambda d: inside_empirical(d, regions, rot))  # noqa: E731
    validation = {rot: round(gt_validation(a.results, rot, inside_for(rot)), 3) for rot in ROTATIONS}
    print(f"[{a.mode}] ground-truth target footprints inside, per rotation: {validation}")
    direction = pd.read_csv(REPO / "configs" / "step4" / "driving_direction.csv").set_index("sequence_id")
    cam_dir = direction.driving_direction.map({"forward": "front", "backward": "rear"})
    sd = pd.read_csv(a.results / "step4_evaluation" / "sequence_decisions.csv", index_col=0)
    frames = []
    for rot in ROTATIONS:
        det = pd.read_csv(a.det_dir / f"det_{rot}_test" / "crops.csv")
        det = det[~det.crop_status.astype(str).str.startswith("error")]
        det["inside"] = inside_for(rot)(det)
        det["rotation"] = rot
        det["direction_camera"] = det.sequence_id.map(cam_dir)
        det["on_driving_camera"] = np.where(det.direction_camera.isna(), "unknown",
                                            np.where(det.camera == det.direction_camera, "yes", "no"))
        frames.append(det[KEYS + ["rotation", "class_name", "crop_status", "x0", "y0", "x1", "y1",
                                  "inside", "on_driving_camera"]])
    fr = pd.concat(frames, ignore_index=True)
    fr["record"] = fr.rotation + "|" + fr.sequence_id
    fr = fr.join(sd[["population", "type", "truth"]], on="record")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "validation.json").write_text(__import__("json").dumps({"mode": a.mode, "gt_inside": validation}, indent=1))
    fr.to_csv(a.out / "frames_corridor.csv", index=False)
    det_frames = fr[fr.inside.notna()]
    det_frames = det_frames.assign(outside=~det_frames.inside.astype(bool))
    by = (det_frames.groupby(["rotation", "population", "type"])
          .agg(frames_with_detection=("outside", "size"), outside_share=("outside", "mean"))
          .round(3).reset_index())
    by.to_csv(a.out / "outside_by_type.csv", index=False)
    cam = (det_frames.groupby(["on_driving_camera"]).outside.agg(["size", "mean"]).round(3))
    seq_out = det_frames.groupby("record").outside.mean().rename("seq_outside_share")
    s = sd.join(seq_out, how="inner")
    rows = []
    for arm in ("detector_high", "plain_mode", "gate_trained", "vlm_uss"):
        for pop in ("heldout", "known"):
            x = s[s.population == pop]
            ok = x[arm] == x.truth
            rows.append({"arm": arm, "population": pop, "n_right": int(ok.sum()), "n_wrong": int((~ok).sum()),
                         "outside_share_right": round(float(x.seq_outside_share[ok].mean()), 3),
                         "outside_share_wrong": round(float(x.seq_outside_share[~ok].mean()), 3)})
    arms = pd.DataFrame(rows)
    arms.to_csv(a.out / "outside_vs_errors.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"frames with a selected box: {len(det_frames)} | outside corridor: {det_frames.outside.mean():.3f}")
    print(cam.to_string())
    print(by.to_string(index=False))
    print(arms.to_string(index=False))


if __name__ == "__main__":
    main()
