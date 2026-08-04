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
obstacles** - the honest version of the "Unknown" claim.

**ROTATION RESULTS (2026-08, final - notebooks 05-07 x3, heldout_slice)**:

| held-out | seqs | frame recall | absorbed by | bucket share | seq Low/High |
|---|---|---|---|---|---|
| bollard | 177 | 0.404 | pole (99%) | 0.006 | 0.751 (100% of detected) |
| cone | 407 | 0.149 | pole (97%) | 0.001 | 0.506 (97% of detected) |
| bush | 867 | 0.001 | curbstone (11 det.) | 0.000 | 0.000 |

CORRECTION: the previously quoted "cone recall 0.42" was the run's AGGREGATE
high_other row, contaminated by known bucket members. Never quote aggregate
bucket metrics for held-out types - use nb07's heldout_slice.csv (which
exists for exactly this reason).

**Verdict - claim 2 reframed as a characterized negative result**: the
trained bucket contributes ~zero open-set capability in every regime (even
bush, whose in-bucket neighbor tree stayed in training). Generalization is
ABSORPTION into the visually nearest named class, decaying with distance
from the training support. The deterministic class->bin map keeps absorbed
detections safe when the neighbor shares the bin (bollard->pole); an
out-of-support obstacle (bush) is invisible or hallucinated as curbstone
(HIGH called LOW - the dangerous direction; curbstone precision 0.74->0.31
in the bush rotation). This SHARPENS the fusion motivation: USS detects the
hedge the camera cannot see.

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

## Track B status & findings (2026-07, notebooks 11-13)

- **Alignment solved**: MF4 WebCam/WebCam3 channels pair each video frame to
  master time exactly (`docs/mf4_video_alignment.md`). 4,631 alignable
  sequences; 1,424 with a clean-approach distance curve (notebook 11).
- **Focal chain validated**: geometric height on the ISO pole = ~4% error -
  the alignment -> distance -> focal pipeline is physically correct (nb12).
- **Fundamental limit on low objects**: naive box-height geometry overestimates
  flat objects ~5x, because a ground-hugging object's box top is a farther
  ground point, not a raised edge - a monocular ambiguity no ground-plane
  recompute fixes (it is algebraically identical under USS distance). This is a
  reportable finding: it MOTIVATES the learned appearance detector (99.8%
  Low/High), which does not share the ambiguity.
- **Geometric Low/High is a WEAK standalone signal (nb13, full run)**:
  calibrated threshold 738 mm, held-out balanced accuracy 0.64, AUC 0.70. The
  aggregate median height is right for tall objects (pole median 1122 vs 1080
  mm), but per-SEQUENCE height has huge variance (pole per-seq rel-err ~67%;
  curbstone called-low only 57% of the time despite a 570 mm median) because
  box height conflates vertical extent with ground-plane foreshortening, and
  that conflation depends on the per-sequence approach geometry. Conclusion:
  geometric height is NOT operationally useful for the drive-over decision -
  which is itself a clean finding that MOTIVATES the learned detector (99.8%).
  No further geometry work is warranted (the ambiguity is fundamental).

## Phase D RESULT (nb14, full gold-test run) - the headline holds

Per-sequence Low/High on 887 gold-test sequences with both signals:
- **USS alone: 62.9%** (production `ClassProbHigh`, dominant-object median).
- **Camera detector alone: 93.6%** (per-sequence aggregate vote; note this is
  the stricter per-sequence metric, vs nb07's 99.8% on IoU-matched detections).
- **Camera rescues USS on 93.0%** of the 329 sequences where USS is wrong -
  the quantified "camera assists USS" claim.

**Complementary failure modes (the fusion motivation, not just "camera wins"):**
- USS is GOOD at flat/low (curbstone 0.86, speedbump 0.86, woodenboard 0.88)
  and POOR at tall/vertical (bicyclestand 0.25, dummychild 0.60, pole 0.71,
  high_other/unknown 0.54).
- The camera is the mirror image. woodenboard is the one class where USS beats
  the camera (camera_gain -0.31) - flat/low, exactly USS's strength and the
  camera's (and geometry's) weakness.
- This mirrors the geometric-height finding: monocular vision struggles on
  flat/low, USS struggles on tall - so they genuinely complement.

**Meta-fusion result (nb15, class-balanced logistic on [det_bin, det_conf,
uss_prob], held-out 355 seqs)**:
- USS 0.734, detector 0.931, **meta-fusion 0.933** balanced accuracy - meta
  edges >= both signals on both metrics, so the clean "fusion >= both" figure
  holds, but the gain is MARGINAL (~0.2 pts).
- Interpretation: the camera detector is near-ceiling, so fusion adds little.
  The learned fuser does use USS (uss_prob weight 1.14) but det_is_high (2.38)
  dominates - USS only tips UNCERTAIN camera calls. It cannot rescue CONFIDENT
  camera errors (e.g. woodenboard: USS 1.00, camera 0.40, meta stays 0.40),
  which a global logistic model won't override on n=5.
- **Paper statement**: the camera alone is sufficient for the drive-over
  decision (93%); USS adds a marginal, confirmatory signal. The headline is
  nb14's "camera corrects the production USS on 93% of its errors", not the
  fusion delta. No further fusion tuning warranted (per-class rules on tiny
  counts = overfitting). EXPERIMENTAL PROGRAM COMPLETE.

## Phase D - the fusion experiment (original plan)

Combine three INDEPENDENT Low/High signals on the aligned test sequences:
1. the detector's class-based Low/High (99.8% on knowns);
2. the USS stack's own `ClassProbHigh` / `HeightProb` (the production baseline);
3. the geometric height score (nb13, class-agnostic).

Questions: does fusing camera + USS beat either alone in the ambiguous 10-30 cm
band? Does the camera correct the USS height classifier's errors (the practical
"camera assists USS" claim)? This does NOT depend on accurate metric height -
it is the camera-USS-complementarity result and the paper's headline. Build as
notebook 14 once nb13's real numbers are in.

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
