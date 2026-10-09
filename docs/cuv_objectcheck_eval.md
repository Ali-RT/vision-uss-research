# CUV_ObjectCheck: auto-labels and the fine-tuning decision

Collection: 2026-09-22, OneDrive `Data_CUV_ObjectCheck` - 73 traces with
video (59 rear, 14 front), one staged object per trace at 3 or 5 kph,
`tracelist-objectdata.xlsx` as ground truth. Detector:
`20260719_train_yolo_v1` (the paper's dataset_v2 model, 8 classes).

Question: can the trained detector label this collection, and does it need
fine-tuning (or more classes) to reach the Bosch SOP Midsize spec?

## Method

- `scripts/label_cuv_fullvideo.py` runs the detector on every frame
  (55,312), picks one box per frame with the approach-corridor prior, and
  writes per-frame CSVs, annotated MP4s, and a YOLO dataset of accepted
  frames (detector class agrees with the tracelist object). The label class
  always comes from the tracelist, never from the model.
- `scripts/eval_cuv_objectcheck.py` adds bumper-to-object distance per frame
  from MF4 `VHM_VehicleState_DrivenDist` plus the final USS gap
  (nearest `Map_ObjN_Dist`). Video and MF4 are start-aligned: all 73
  durations agree within 0.1 s (common trigger). Checks: driven distance
  plus gap matches the independent USS reading within ~3 cm below 1.5 m, and
  the six wall traces at 1.60 m show the wall base on the same image row.
- A frame is a HIT when its picked box has the correct Low/High bin. A
  trace meets spec when >= 50% of frames in [spec - 0.25 m, spec] are HITs.
- `scripts/scale_sweep_cuv.py` compares inference sizes.

## Key finding: inference size, not missing knowledge, for most classes

At imgsz 960 (the training size) the detector finds boxes, car and curbs far
away and loses them when they get near and large. The new 640x480 camera has
a narrower field of view than the training camera, so near objects upscaled
to 960 are larger than anything seen in training. At imgsz 480 the same
weights recover most of the near range:

| class | spec | traces meeting spec @960 | @480 | HIT 1.0-1.5 m @960 -> @480 |
|---|---|---|---|---|
| isopole | 2.0 m / 95% | 12/12 | 12/12 | 1.00 -> 1.00 |
| dummychild | 2.2 m / 90% | 6/6 | 6/6 | 0.93 -> 1.00 |
| dummyadult | 2.2 m / 90% | 5/6 | 6/6 | 0.99 -> 0.98 |
| box40 | 1.4 m / 90% | 0/6 | 6/6 | 0.00 -> 0.88 |
| car | 1.5 m / 95% | 7/12 | 11/12 | 0.66 -> 0.75 |
| box30 | 1.2 m / 90% | 0/6 | 4/6 | 0.01 -> 0.87 |
| curbhigh | 1.5 m / 90% | 0/6 | 4/6 | 0.09 -> 0.40 |
| wall | 1.6 m / 90% | 0/6 | 0/6 | 0 -> 0 |
| woodencurb10 | 1.0 m / 90% | 0/6 | 0/6 | 0 -> 0 |
| metalbar | 1.0 m / 90% | 0/6 | 0/6 | 0 -> 0 |

Imgsz 480 also removes the curbhigh error where grass behind the curb was
boxed as `high_other`: wrong-bin picks drop from 35% to 1%. The cost is
range: beyond ~4 m, 960 detects more (box30 0.95 -> 0.75 at 4-6 m). Every
spec distance is at or below 2.2 m, so 480 is the right size for this
camera. A 480+960 union keeps the 960 grass errors and is not better.

## Remaining gaps after the inference fix

- **Wall**: never detected at any size or distance; the few wall boxes are
  binned Low. Never a training class.
- **Wooden curb 10 cm**: detected only beyond ~4 m. At 1.0 m the board is at
  the bottom edge of the frame, so the 1.0 m spec is at the camera's
  geometric limit for a 10 cm object regardless of training.
- **Front car**: rear car hits 1.0 at 0.8-2.5 m, front only 0.17-0.72. The
  front view sees the lower body and a hard shadow, a different look.
- **Near range 0.5-1.5 m** for box30 and curbhigh: improved but not at 90%.
- **Metal bar**: invisible below ~5 m at 640x480; the USS reports no end gap
  either. Far boxes (5-12 m) are mostly the bar but some are doubtful.

## Tracking and gap filling (labels only)

The detector works frame by frame, so labels flicker. `scripts/track_cuv_labels.py`
links each trace's agreeing boxes into one track (one staged object per
trace) and fills gaps of up to 20 frames (0.67 s) between overlapping
anchors: first with a raw detection the picker skipped ("rescued", any
class), else by linear interpolation. For low-profile objects (curbs,
metal bar, 30/40 cm boxes) an exit rule drops bottom-edge boxes after the
object has left the frame; this removed 36 curbhigh false positives on the
asphalt strip behind the curb. Result: 13,521 labels (+8% over the 12,508
accepted; +13-22% for box30, curbs and metal bar, +2% for isopole).

Most missing frames sit in gaps longer than 20 frames (near-range misses),
which interpolation cannot fill safely; those need SAM3 propagation or the
fine-tune. Gap filling uses future frames, so it is a labeling tool only, not
a model of the in-car camera.

## Recommendation

1. **Switch inference to imgsz 480 for this camera now.** No training,
   four classes go from failing to meeting spec.
2. **Fine-tune: yes, targeted. New named classes: no.** The product output
   is Low/High, and the paper showed buckets work as trained classes. Map the
   new objects into existing classes: car, wall, boxes -> `high_other`;
   adult dummy -> `dummychild`; curb high, wooden curb -> `curbstone`.
   Add a named class only if a downstream function needs the object type.
3. Fine-tune from `best.pt` on this collection plus a replay of dataset_v2
   (to avoid forgetting the old camera), with scale augmentation covering the
   field-of-view difference. Split by trace (e.g. 2 of 6 traces per class
   held out, both speeds) so the spec eval stays clean.
4. Labels: the 480 auto-labels (12,508 accepted frames) cover isopole,
   dummies, car, boxes and curbs. Walls come from YOLOE-26 with the text
   prompt "wall" (open-vocabulary YOLO; 6/6 traces at 1.0-2.5 m, no false
   walls on 66 other frames, `scripts/wall_model_probe.py`); near-range gaps
   from SAM3 video tracking seeded by existing labels.
5. Drop metal bar from camera scope; leave its far labels out of training
   until reviewed. Set the camera's spec for 10 cm curbs at a distance the
   camera can see (about 1.2 m and beyond).

## Fine-tune workflow (notebook 17)

1. Locally: `scripts/export_cuv_bundle.py` packs every frame within 8 m of
   the object (27,096 frames, 72 traces) with distances and the tracked
   labels into `outputs/cuv_objectcheck/cuv_bundle_v1.zip` (2.0 GB).
2. Upload the zip to `<artifacts>/cuv_objectcheck/` on Drive.
3. Colab, GPU runtime: `notebooks/17_cuv_wall_sam3_finetune.ipynb`
   (config `configs/experiments/cuv_finetune_v1.yaml`). Stages: YOLOE wall
   labels -> SAM3 gap fill -> dataset (trace-level split, dataset_v2 replay)
   -> fine-tune (`MODEL_CHOICE` `yolo11s_ft` or `yolo26s`) -> spec eval on
   held-out traces vs the paper model, plus mAP on old and new val data.
   Every stage saves to Drive and resumes.

Smoke-tested end to end locally (two objects, 1 epoch, SAM3 off): every
stage runs; the YOLOE wall labels were checked visually at 0.2-3.7 m. The
SAM3 stage needs a GPU and access to `facebook/sam3`, so it has not run yet.

## Caveats

- 6 traces per class: 6/6 is consistent with a true rate as low as ~54%
  (95% CI), so these runs cannot certify 90/95%. They do show clear fails.
- HIT checks the Low/High bin of a corridor-picked box, not box overlap with
  a ground-truth box; accepted labels were spot-checked visually only.
- One site, one day, one vehicle. Front side: pole, car, one wall trace
  (3188, no vehicle motion, excluded); front wall 3189-3193 not delivered.

## Artifacts (`outputs/prelabel/`, gitignored)

- `fullvideo_cuv_v2_480/tracked/` - recommended labels: `yolo/` (13,521
  tracked labels), `videos/` (green detector, cyan rescued, yellow
  interpolated), `frames_tracked.csv`, `stats.csv`
- `fullvideo_cuv_v2_480/` - per-frame detector labels: `yolo/`, `videos/`,
  `frames.csv`,
  `detections.csv`, `eval/` (report.md, summary, per-trace, by-distance)
- `fullvideo_cuv_v1/` - same at imgsz 960, plus `eval/scale_sweep_*`
- `prelabel_cuv_v1/` - first 48-frame sample, `review.html`
