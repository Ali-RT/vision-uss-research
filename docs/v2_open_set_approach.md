# V2 Approach: Open-Set Traversability Detection + Paper Framing

Status 2026-07-16. Supersedes the v1 (6-class closed-set) scope described in
`low_object_detection_plan.md`; the pipeline (notebooks 01-07) is shared, v2 is
config-selected (`labeling_sam3text_v2extra.yaml`, `dataset_v2.yaml`).

## Model contract (one paragraph)

The model takes a **single RGB frame (960x640) from the front or rear bumper
camera** during a low-speed parking approach and returns **zero or more detected
obstacles, each as a bounding box in image pixels, a class, and a confidence**.
The class is one of six named obstacle types (curbstone, speedbump, woodenboard,
pole, dummychild, bicyclestand) or one of two open-set buckets (`low_other`,
`high_other`) trained on twelve additional obstacle types; every detection maps
deterministically to a deployment category - **"Low"** (drivable-over, below
~25 cm) or **"High"** for named classes, **"Low Unknown"** / **"High Unknown"**
for bucket detections. Combined with the ultrasonic distance at inference time,
each detection additionally yields a **geometric metric-height estimate**
(box pixel height x USS distance / calibrated focal length) at no training
cost. The model does **not** predict distance (USS territory), does not output
a traversable/empty signal (absence of detections is not a trained prediction),
operates in image space only (no top-view coordinates), and is trained on
mid-approach frames (ratio 0.30-0.85) - near-contact frames below the bumper
field of view are out of scope by design.

## V2 design

### Classes (8 training classes)

| id | class | bin | label source |
|---|---|---|---|
| 0 | curbstone | low | SAM3 text-prompt (batch) |
| 1 | speedbump | low | SAM3 text-prompt (batch) |
| 2 | woodenboard | low | human clicks |
| 3 | pole | high | SAM3 text-prompt (batch) |
| 4 | dummychild | high | human clicks |
| 5 | bicyclestand | high | human 2-corner boxes |
| 6 | low_other | low | batch: hose, step |
| 7 | high_other | high | batch: cone, bollard, car, tree, bush, stonelarge, stonemiddle, cubestandard, ubarrier, fence |

Key idea: "Unknown" cannot be bolted onto a closed-set detector, but it can be
**trained** - the bucket classes learn "obstacle-ish but not one of the six"
from a *diverse* set of real objects, which is what makes them fire on object
types outside the training set.

### Open-set evaluation (leave-object-out holdout)

`dataset_v2.yaml: holdout_objects` removes chosen object types from train/val
entirely (all their sequences forced to the test split). Bucket recall and
Low/High accuracy on the held-out type measures **generalization to never-seen
obstacles** - the honest version of the "Unknown" claim. Default holdout: cone;
the full result rotates the holdout (cone / bollard / bush / stone).

### Metric height without a regression head

At deployment the USS distance is available, so height follows from geometry:
`height_cm ~ box_pixel_height * distance / focal`. The focal constant is
calibrated once from known-height classes (dummychild 125 cm, isopole 108 cm).
Expected error +-20-30% - sufficient for the <25 cm decision; validated on the
known-height classes. A learned height regressor is a paper *experiment*
("does learning beat the geometry baseline?"), not a requirement: it would need
per-frame height labels (MF4-synced ObjDist) and is ill-posed without distance
input anyway.

### Anti-shortcut controls

- Report bucket performance stratified by scene/background tags (metadata) -
  answers "does Unknown key on the test track background?"
- Gold test split = human-clicked sequences, never trained on.
- Distance-stratified recall via CusReplay ObjDist (near-field story).

## Paper framing

**Central question:** can traversability semantics - "is this obstacle low
enough to drive over?" - be learned from automatically labeled data and
generalized to obstacle types never seen in training?

**Claims:**
1. **Scalable auto-labeling without human boxes**: SAM3 text-prompt seed +
   approach-corridor instance prior + tracker propagation over ~8k staged
   sequences; human effort measured in clicks (~170) not boxes; pseudo-label
   precision measured against the gold clicked test set.
2. **Open-set traversability, measured honestly**: leave-object-out holdout
   results for the bucket classes (the headline table).
3. **Metric height from camera-USS fusion for free**: geometric height vs
   monocular metric-depth foundation models on the same frames (the
   exploratory depth-failure study becomes the motivating baseline: foundation
   depth misses these obstacles; detector + USS distance beats it at a
   fraction of the compute).

**Positioning:** intersection of open-set detection (rarely evaluated on
safety-relevant traversability), foundation-model auto-labeling (SAM3 as
annotator), and camera-ultrasonic fusion (unique data asset: per-cycle USS
ground truth under staged obstacles). Target: WACV or IV/ITSC full paper;
autonomous-driving workshop as fallback. The leave-object-out protocol is a
citable contribution even if the raw data stays private.

**Working title:** "Drive-Over or Stop? Open-Set Traversability Detection for
Near-Field Parking from Automatically Labeled Camera-Ultrasonic Data"

**Known risks:**
- If held-out cone recall is ~0, claim 2 collapses to v1 scope. Run one holdout
  rotation EARLY, right after the v2extra labels land, before writing.
- Reviewers will probe background shortcuts -> the scene-tag stratification
  must be in the eval from the start.
- Test-track domain: single facility, single vehicle, 2019 imagery. State it;
  do not claim in-the-wild generality.

## Pipeline touchpoints

| step | notebook | config |
|---|---|---|
| labels v1 (named classes) | 02 | `labeling_sam3text.yaml` |
| labels clicks | 03 | in-notebook knobs |
| labels v2 buckets | 02, `CONFIG_NAME="labeling_sam3text_v2extra"` | `labeling_sam3text_v2extra.yaml` |
| QA gate | 04 | auto-discovers all runs |
| dataset v2 | 05, `CONFIG_NAME="dataset_v2"` | `dataset_v2.yaml` (class_groups, holdout) |
| train | 06 | `train_yolo_v1.yaml` |
| eval incl. Low/High + Unknown | 07 | CLASS_BINS / CATEGORY_NAMES in-notebook |
