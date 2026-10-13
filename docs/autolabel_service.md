# Auto-labeling service: online detect + track, and how it is rated

The service labels videos where the object category of each video is known
(here: the CUV_ObjectCheck tracelist). Quality is rated against a human gold
set on held-out test traces.

## Pipeline (`src/vision_uss_research/online_labeler.py`)

Per frame, in one pass, no look-ahead:

1. **Detect** - the fine-tuned YOLO11s at imgsz 480 (`train_cuv_v1_*` weights).
2. **Track** - BoT-SORT (Ultralytics, camera-motion compensation) updated
   with that frame's detections; lost tracks keep a motion prediction for up
   to 30 frames.
3. **Target** - the job's object: a matched track of the category's detector
   class under the approach-corridor prior. A detection always beats a
   motion prediction; the prediction carries the target only when nothing of
   the category is detected, for at most 15 frames.
4. **Exit guard** (curbs, bars, boxes) - once the object leaves through the
   bottom edge, later bottom-edge boxes (the asphalt strip behind a curb) and
   predictions are dropped.

Output per frame: box, track id, source (`det` / `pred`), detector score and
track features (age, hit streak, frames since last match, competing tracks).
Run: `scripts/autolabel_online.py --render` (about 40 frames/s on an M-series
Mac).

First full run (`online_v1`, 72 traces, 27,096 frames within 8 m, first-run
fine-tuned weights): 87% of frames labeled vs 44% for the post-processed
labels in the bundle; 2.5% of labels are predictions. 37/72 traces have at
least one target track change (the tracker splitting one object into new
ids) - a tuning target for the tracker.

## Gold set (`scripts/build_gold_review.py`)

Every 10th frame within 8 m on the 24 test traces (never used in training):
883 frames, 4 shards of ~240 frames, 88 frames (10%) reviewed twice for
reviewer agreement. Reviewers draw boxes from scratch (no pre-labels), or mark
the object not visible / unsure. Each shard is a zip with `index.html`; the
exported `gold_*.csv` files come back for scoring.

## Silver ground truth (notebook 18)

Instead of drawing every gold box by hand: SAM3 (text prompt with the known
category, each frame on its own) and Qwen2.5-VL (asked to locate the named
object) label the same 883 frames. Boxes agreeing at IoU >= 0.5, or both
"not visible", become ground truth; the rest is disputed and left out. Both
models are independent of the service under test (detector + tracker); if
SAM3 is later added to the service, rate with Qwen and the audit instead. A
100-frame human audit (`build_gold_review.py --prefill silver.csv --sample 100`,
proposal shown, `A` accepts) measures silver accuracy, reported first in the
rating (`rate_autolabel.py --audit-dir`).

## Click-and-track ground truth (notebook 19)

The densest and most reliable option: one click per trace on the object
(~3 m), the SAM3 video tracker follows it forward until it leaves the view
and backward to where it appears - a box or "not visible" for every frame
(8,729 frames on the 24 test traces). Tracking runs in 200-frame chunks
re-seeded with the last box; once the object is lost for 8 frames it stays
"not visible", so the tracker cannot re-lock onto a look-alike. A filmstrip per
trace is reviewed; corrections are extra clicks (each frame takes the nearest
click's track) or "not visible after/before frame X". Output
`gold/clicktrack_v1/clicktrack_gold.csv` feeds `rate_autolabel.py --gold-dir`.
With every frame labeled, the 2-4 m band uses all its frames and false labels
per hour are exact.

## KPIs (`scripts/rate_autolabel.py`, `src/vision_uss_research/label_kpi.py`)

Per pipeline, per category x distance band (0-1, 1-2, 2-4, 4-8 m) and overall:

- **recall** - visible-object frames with a correct label (IoU >= 0.5 vs gold);
- **precision** - labels that are correct (a label where the object is not
  visible counts as wrong);
- **mean IoU** of correct labels; **labels on not-visible frames**;
- **tracking** - target track changes per trace;
- **reviewer agreement** - status agreement and box IoU on double-reviewed frames.

**Deck-style KPIs** (from "KPI Calculations", CC-DA/ESV4, 2018, adapted from
lidar parking markings to camera boxes; distance bands play the role of the
deck's ROI rings and are set with `--bands`):

- **TPR per sequence** - frames whose object is completely in view (gold
  "cut by image edge" excluded), TPR per trace, then the mean over traces,
  with the worst trace; pooled recall stays beside it;
- **object detected at least once** in each band, as a share of traces;
- **false labels per hour** - runs of consecutive wrong gold frames, and
  distinct track ids carrying a wrong label, per hour of video. The 24 test
  traces are only ~0.08 h of video, so one event moves the rate by ~12/h;
  quote counts with the rate.

**Label confidence**: a per-category logistic regression on the label's
features (detector score with a per-category slope, prediction flag, track
age, streak, box geometry), scored out-of-fold with folds grouped by trace.
Reported: calibration table and expected calibration error, and the
**auto-accept share** - labels that can ship without review at 95/98/99%
precision, overall and per category. The fitted model is saved as
`confidence_model.json` for scoring new jobs.

## Caveats

- Test traces come from the same day, site and vehicle as the training traces;
  a second session is needed before quoting the KPIs as service quality.
- The gold set is the reviewers' judgement; the double-reviewed frames bound
  how precise any KPI can be.
