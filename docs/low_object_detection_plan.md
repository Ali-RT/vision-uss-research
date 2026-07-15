# Low-Object Detection Plan (SAM3 Pseudo-Labels -> YOLO -> USS Fusion)

## Idea

Train a real-time camera detector (YOLO) that detects near-field obstacles and
classifies them as **low** (traversable-relevant, height below 25 cm, excluding the road
surface itself) vs **high**, using a SAM3-like promptable segmentation model to
auto-generate localization labels. The trained detector's low/high output then assists
the ultrasonic sensor (USS) height classifier, which is the known weak point of USS
perception.

This complements (does not replace) the USS-supervised benchmark direction in
`research_problem.md`. Note the supervision direction flips here: catalog knowledge +
SAM3 label the camera; the camera model then helps USS.

## Why this is a strong use case

1. **It targets the real USS weakness.** USS measures distance well but height poorly.
   The CusReplay logs contain the USS stack's own height classifier outputs
   (`MAP_ClassifierMonitorNormed_HeightProb`, `ClassProbHigh`) per cycle, so we can
   compare and fuse against the production-style baseline directly.
2. **The staged object catalog is a scientific instrument.** Boxes at exact 5 cm
   increments (10-60 cm) allow plotting detection recall and low/high accuracy as a
   function of true object height — a "height vs detectability" curve with the 25 cm
   decision boundary marked. This is the headline figure.
3. **Label V2 already encodes the target semantics.** Its class column is
   `0=traversable, 1=low, 2=unknown, 3=high`, so pseudo-labels can be cross-checked
   against human event labels.

## Two key insights that make auto-labeling work

1. **SAM3 localizes; the sequence tag classifies.** SAM3 cannot tell a 20 cm box from a
   30 cm box, but it does not have to: each sequence's metadata names the staged target
   object (`Label Objects` tag, Label V2 `Which object`), and the catalog maps that name
   to a height. YOLO class labels come from the catalog, not from SAM3.
2. **USS distance turns 2D masks into metric heights.** Mask pixel height + per-cycle
   USS `ObjDist` + camera geometry gives an approximate metric object height. Use it
   (a) to reject SAM3 false positives (a "20cmbox" mask implying 1.2 m height is wrong),
   and (b) optionally as a continuous height label for regression/ordinal heads.

## Object height catalog

Machine-readable version: `configs/datasets/object_height_catalog.csv`
(edit that file as the inventory discovers new object names).

| Object | Height | Bin (<25 cm = low) |
|---|---|---|
| 10cmbox | 10 cm | low |
| 15cmbox | 15 cm | low |
| 20cmbox | 20 cm | low |
| 25cmbox | 25 cm | high (boundary) |
| 30cmbox | 30 cm | high |
| 35cmbox | 35 cm | high |
| 40cmbox | 40 cm | high |
| 45cmbox | 45 cm | high |
| 50cmbox | 50 cm | high |
| 55cmbox | 55 cm | high |
| 60cmbox | 60 cm | high |
| child dummy | 125 cm | high |
| curblow | ~4-9 cm | low |
| curbhigh | ~10-18 cm | low |
| car | — | high |
| isopole | 108 cm | high |
| table (on side) | 76 cm | high |
| wall | — | high |

Boundary policy: "below 25 cm is low", so `25cmbox` is **high**. Errors will concentrate
at the 20/25/30 cm boundary; therefore train on **ordinal height bins** (e.g. <=10,
10-25, 25-60, >60 cm) and derive the binary low/high decision at the end, so one figure
shows where height resolution breaks down.

The catalog above is what we know so far. The actual data contains more object names
(e.g. `bicyclestand`, `empty`, `Tire_stopper (Radstopper)` were already observed). The
**first step is a full inventory over all sequences** — see `scripts/build_object_inventory.py`.

## Findings from the full inventory (2026-07-14, 8217 folders)

- **Scale:** 8055 complete sequences (98%); 73 empty folders, 89 with missing key files.
- **Class balance (sequences):** high 6119, low 2060, traversable 8205 (the paired
  `empty` row in nearly every file), 67 unlabeled. The low class has real volume:
  curbstone 1492, speedbump 302, woodenboard 109, curbstone_side 91, hose 57, step 9.
- **Distance gate CONFIRMED:** `important Distance [mm]` == 2000 for every row in the
  dataset. Label V2 cannot supervise distance; use CusReplay `ObjDist`.
- **Taxonomy confirmed:** NAMEoFFILE (on ~59% of rows, ~86% parseable) encodes
  `C_highObjectslarge` / `D_highObjectsmedium` / `E_highObjectssmall` /
  `F_lowObjectssmall` families + `<L><nn>_<category>` codes. Family agrees with the
  class column; class-high rows inside F sequences are real co-present objects
  (bush/pole/tree), not noise.
- **Class 5 resolved to an exclusion:** 160 rows, all on `empty` rows (many labelers,
  concentrated in stonemiddle campaigns). Excluded from training until clarified.
  Class 2 ("unknown") occurs exactly once in the whole dataset.
- **A second label schema exists** (~3700 rows) with extra columns (`Object Type`,
  `Target Object`, `Car Config`, ...). Some names embed metric sizes
  (`cone_medium_49cm`, `pole_round_300mm`) - a future source of height labels.
- Mapping rules live in `configs/datasets/label_v2_class_mapping.yaml`
  (class map, exclusions, canonical name merges, multi-object names).

## Pipeline phases

### Phase A — Taxonomy and audit (current)

- Run the inventory (`notebooks/build_object_inventory.ipynb`, or the manifest-based
  `scripts/build_object_inventory.py`) over all sequences to enumerate every object
  name in metadata JSON tags and Label V2 `Which object`, with sequence counts, Label V2
  class distribution, and distance stats per object. Smoke-test with a **random** folder
  sample first — consecutive folders are one test campaign and give a biased object mix.
- **Division of labor (important):** the low/high bin ground truth comes from the
  Label V2 class per sequence (1=low, 3=high), NOT from the catalog. The catalog
  contributes metric height where known (staged boxes, curbs, dummy, pole) for the
  height-vs-detectability analysis, plus a prior bin used only as a consistency check
  (`bin_agreement` column). A 1-1 object-name-to-bin mapping is not assumed.
- Extend `configs/datasets/object_height_catalog.csv` until every observed object name
  has a height where known (or an explicit `ignore`).
- Investigate undocumented Label V2 class values before mapping them (e.g. class `5`
  observed on `empty` rows in an early 200-folder sample; header documents only 0-3).
- **Label-driven taxonomy (pivot, 2026-07):** the staged-box height catalog belongs to a
  different campaign; the primary taxonomy now comes from the label files themselves:
  (a) the `class` column (0/1/3 -> traversable/low/high) and (b) the labeling team's own
  hierarchy embedded in the `NAMEoFFILE` path
  (`...\Labelled_Database\<customer>\<vehicle>\D_<family>\D<nn>_<category>\Type<n>\...`),
  mined by the inventory notebook. Cross-check family vs class; conflicts are label noise.
- **Distance gate:** if `important Distance [mm]` is (near-)constant at 2000 across all
  rows, it is a labeling-spec threshold, not a measurement - distance supervision must
  come from CusReplay `ObjDist` per cycle, not Label V2.
- Decide final ordinal bins after the full inventory.

### Phase B gate result (2026-07-14, notebooks/sam3_zero_shot_mask_test.ipynb, v2 run)

12 sequences, 2 per object, frames at ratios 0.40/0.60/0.80, both cameras when
driving direction unknown, score threshold 0.3:

| object | detection | mean top score | verdict |
|---|---|---|---|
| curbstone | 100% | 0.85 | PASS (5.7 masks/frame - distractors) |
| pole | 100% | 0.88 | PASS (10.2 masks/frame - distractors) |
| speedbump | 100% | 0.81 | PASS (was 67% at ratio 0.95 - FOV effect) |
| bicyclestand | 50% | 0.29 | MARGINAL |
| dummychild | 33% | 0.17 | FAIL with text prompt "child mannequin" |
| woodenboard | 17% | 0.16 | FAIL (5 cm plank flat on ground) |

Consequences:
- Text-prompted SAM3 auto-labeling covers the bulk of the dataset (~91% of low
  sequences via curbstone/speedbump/curbstone_side; most high classes).
- Frames must be sampled mid-approach (~0.4-0.8): at closest approach the object
  drops below the bumper camera FOV.
- Distractor suppression (approach corridor + temporal consistency + USS distance)
  is the core of the labeler, not an add-on (pole: 10 masks/frame).
- dummychild/bicyclestand/woodenboard need a fallback: prompt sweep first
  (e.g. "child" instead of "child mannequin"), then USS-corridor box/point prompts
  (visual prompting) instead of text, then manual labels for what remains.

### Phase B — SAM3 auto-labeling

Per sequence: sample frames along the approach -> prompt SAM3 with a text prompt derived
from the object tag ("cardboard box on the ground", "curb", "pole", "child mannequin")
-> filter candidate masks by:

- approach corridor (object roughly ahead of the active bumper),
- temporal consistency across frames,
- USS-distance height plausibility check (insight 2 above),

then export bbox/mask in YOLO format with the catalog class.

**Mandatory QA:** human-verify a random ~200-frame subset and report pseudo-label
precision/recall. Auto-labels without this number will not survive review.

### Phase C — YOLO training

- YOLOv8/YOLOv11 small variants (the story is real-time, near-bumper).
- Multi-task: detect + ordinal height-bin classification.
- Known pathologies: one staged target per sequence (use mosaic/copy-paste
  augmentation so the model does not learn "exactly one object per scene");
  near-duplicate frames within a sequence (split strictly by sequence).

### Phase D — Evaluation (the paper)

- Recall and low/high accuracy **per true height** (headline curve) and per distance bin.
- Confusion at the 20/25/30 cm boundary — expected, show it honestly.
- **Camera vs USS vs fusion:** compare YOLO low/high against USS
  `ClassProbHigh`/`HeightProb` on the same events; then a simple late fusion (logistic
  regression on both scores). The target result: fusion beats both alone in the
  ambiguous 10-30 cm band.

### Phase E — Tie-back to the depth study

`notebooks/depth_foundation_failure_analysis.ipynb` stays relevant: if zero-shot metric
depth models also miss <=25 cm objects, that justifies a dedicated detector rather than
"just use a depth foundation model".

## Risks

- **Curbs (4-9 cm) will hurt.** In 960x640 2019-era video a low curb is barely a texture
  edge; SAM3 masks for "curb" will be ragged. Boxes/poles/dummy should work; budget
  extra effort for curbs or accept and report lower quality.
- **Ceiling effect.** A student cannot exceed its labeler on localization. The
  contribution is height semantics (which SAM3 lacks), distillation to real-time, and
  USS fusion — keep the claims there.
- **Monocular height ambiguity.** A small box near and a big box far look identical; the
  model leans on ground-contact context. Report distance-stratified results; never claim
  the model "measures" height.
- **SAM3 license.** Fine for research; re-check terms before anything feeds a production
  system.

## Immediate next steps

1. Run the object inventory over all sequences (Colab, where the raw data lives):
   `python scripts/build_object_inventory.py --profile colab_drive`
2. Fill the height catalog for every discovered object name.
3. Zero-shot SAM3 sanity test on two sequences (the bicycle-stand sample + one low-box
   sequence): if SAM3 finds a 15 cm box reliably at 2-4 m, Phase B is viable.
