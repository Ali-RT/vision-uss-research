# MF4 <-> Video <-> USS-Distance Alignment (Track B - feasibility confirmed)

> **Track B is separate from the detector pipeline.** Notebooks 01-08 and the
> trained model do NOT depend on any of this - the detector predicts class and
> Low/High from pixels alone. Track B is an *optional enhancement* that adds
> metric information (geometric object height, camera/USS fusion) on top.
> Code lives in `src/vision_uss_research/alignment/` (imported by nothing in
> the pipeline); the EDA is `notebooks/10_mf4_alignment_eda.ipynb`; artifacts
> go to `<artifacts>/alignment/`. If Track B fails or is abandoned, the
> detector pipeline is unaffected.

Verified on the local sample `110613_20190308_LB_XO2617_039_151540` with
`asammdf` (2026-07). This unlocks per-frame USS distance -> the geometric height
estimate and the USS-fusion experiment.

## Key finding: the camera channels give an EXACT clock mapping

The MF4 has `WebCam` (front) and `WebCam3` (rear) channels. For each:

- **sample count == video frame count** (rear: 67 WebCam3 samples == 67 rear frames;
  front: 71 == 71).
- **sample VALUE == that frame's video PTS in seconds** (matched to 1e-3 against
  ffprobe pts_time: 0.000, 0.396, 1.189, 1.784, ...).
- **sample TIMESTAMP == that frame's MF4 master time**.

So the channel explicitly pairs (video_pts, mf4_time) per frame - no cycle-number
guessing, no linear approximation. This replaces the rough cycle->time hack from
the depth-failure notebook.

## Recipe

```python
from asammdf import MDF
import numpy as np

mf = MDF(mf4_path)
cam = mf.get("WebCam3")            # rear (WebCam for front); pick by driving direction
pts, mf4t = np.asarray(cam.samples,float), np.asarray(cam.timestamps,float)
o = np.argsort(pts); pts, mf4t = pts[o], mf4t[o]
video_to_mf4 = lambda v: np.interp(v, pts, mf4t)     # video seconds -> MF4 seconds

dsig = mf.get("sigPDC_Zone16_Distance")              # a USS distance on MF4 time
dt, dv = np.asarray(dsig.timestamps,float), np.asarray(dsig.samples,float)
dv[dv >= 4095] = np.nan                              # 0xFFF = no-object sentinel
good = ~np.isnan(dv)
dist_at = lambda mf4_time: np.interp(mf4_time, dt[good], dv[good])

# distance at video frame i (frame extracted at ratio r -> pts = r * duration):
# d = dist_at(video_to_mf4(frame_pts))
```

Verified: rear zone 16 falls 123 -> 68.8 -> 37.6 -> 21 over the last ~5 s of the
approach (object entering the zone), consistent with a backward approach.

## Distance-channel options (all on the MF4 master clock, populated in the sample)

- `sigPDC_Zone{15,16}_Distance` - production Park-Distance-Control zone distance
  (rear zones for a backward approach); 0xFFF/4095 = no object.
- `MAP_ObjBuff_elm{1..20}_obj_OVFAttributes_LastDetDist` - per-tracked-object last
  detected distance (mm, up to ~4900); 20 object slots.
- CusReplay `MAP_ClassifierMonitorNormed_ObjDist` (already used) - the
  classifier-monitored TARGET object, 0.41-3.59 m; richest but indexed by
  CusReplay CycleNr, so needs a cycle->MF4-time bridge if used.

## Remaining work (days, not weeks)

1. **Pick the channel/object matching the labeled target.** The object buffer has
   20 slots and PDC has many zones; select the one tracking the staged obstacle
   (by driving direction -> rear/front zones; cross-check against the Label V2
   object and the CusReplay target curve). ~1-2 days.
2. **Confirm units** per channel (PDC likely cm, ObjBuff mm) against the labeled
   2000 mm event and the CusReplay curve. Hours.
3. **Verify channel names are consistent across the full dataset** (this is one
   sequence; run over a sample of Drive sequences). ~1 day.
4. Then: calibrate focal from known-height classes (dummychild 125 cm, pole/
   isopole 108 cm) and emit per-detection height = box_px_h * dist / focal.

## Caveat

Confirmed on a single local sample. The mechanism (WebCam/WebCam3 pairing) is a
platform convention likely stable across the campaign, but channel availability
and naming must be spot-checked on more sequences before scaling.
