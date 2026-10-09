"""Package CUV_ObjectCheck frames + labels for Colab (notebook 17).

Colab cannot read the OneDrive AVIs, and Drive folders degrade past ~10k
files, so this writes one zip: every frame within --max-dist m of the object
(moving traces only), per-trace folders, plus frames.csv with distance, the
tracked labels from track_cuv_labels.py, and trace metadata. Upload the zip to
<artifacts>/cuv_objectcheck/ on Drive.

    uv run --with openpyxl python scripts/export_cuv_bundle.py
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import pandas as pd

from prelabel_cuv_objectcheck import DEFAULT_DATA_ROOT, PROJECT_ROOT, hydrate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="fullvideo_cuv_v2_480")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--name", default="cuv_bundle_v1")
    parser.add_argument("--max-dist", type=float, default=8.0)
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / "outputs" / "prelabel" / args.run
    out_root = PROJECT_ROOT / "outputs" / "cuv_objectcheck"
    bundle = out_root / args.name
    (bundle / "frames").mkdir(parents=True, exist_ok=True)

    # frames_dist carries the per-frame detector box; the bundle's label is
    # the tracked one, so the detector columns are dropped before the merge
    fd = pd.read_csv(run_dir / "eval" / "frames_dist.csv",
                     usecols=["side", "trace", "object", "label_class",
                              "height_bin", "speed", "frame_idx", "n_frames",
                              "dist_m", "moved"])
    tr = (pd.read_csv(run_dir / "tracked" / "frames_tracked.csv",
                      usecols=["side", "trace", "frame_idx", "track_src",
                               "t_x0", "t_y0", "t_x1", "t_y1"])
          .rename(columns={"track_src": "label_src", "t_x0": "x0",
                           "t_y0": "y0", "t_x1": "x1", "t_y1": "y1"}))
    df = fd.merge(tr, on=["side", "trace", "frame_idx"], how="left")
    df = df[df["moved"] & (df["dist_m"] <= args.max_dist)].drop(columns="moved")

    for (side, trace), g in df.groupby(["side", "trace"]):
        tdir = bundle / "frames" / f"{side}_{trace}"
        want = set(g["frame_idx"].astype(int))
        if tdir.exists() and len(list(tdir.glob("*.jpg"))) == len(want):
            continue
        tdir.mkdir(exist_ok=True)
        folder = "Front" if side == "front" else "Rear"
        avi = next((args.data_root / folder).glob(f"*_{trace}_*.avi"))
        hydrate(avi)
        cap = cv2.VideoCapture(str(avi))
        idx, last = 0, max(want)
        while idx <= last:
            if not cap.grab():
                break
            if idx in want:
                ok, frame = cap.retrieve()
                if ok:
                    cv2.imwrite(str(tdir / f"{idx:05d}.jpg"), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 92])
            idx += 1
        cap.release()
        print(f"{side}_{trace}: {len(want)} frames", flush=True)

    df.to_csv(bundle / "frames.csv", index=False)
    (df.groupby(["side", "trace", "object", "label_class", "height_bin", "speed"])
       .agg(frames=("frame_idx", "size"), labeled=("label_src", "count"),
            min_dist_m=("dist_m", "min"), max_dist_m=("dist_m", "max"))
       .reset_index().to_csv(bundle / "traces.csv", index=False))
    zip_path = shutil.make_archive(str(out_root / args.name), "zip",
                                   root_dir=out_root, base_dir=args.name)
    size_mb = Path(zip_path).stat().st_size / 1e6
    print(f"\n{len(df)} frames, {int(df['label_src'].notna().sum())} labeled, "
          f"{df.groupby(['side', 'trace']).ngroups} traces -> {zip_path} "
          f"({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
