# USS Auto-Labeler Track (IEEE IV 2027 candidate)

Separate paper track from WACV 2660. IV 2027 deadline: **2026-11-15**
(notification 2027-01-15, 6 pages). Frozen WACV numbers are never touched;
all outputs of this track live under `outputs/autolabel/al_*` (Drive:
`vision_uss_research_artifacts/outputs/autolabel/`).

## Origin

An internal Bosch prototype (Aug 2024, `helpers_mf4.py` / `Mass_tester.py`,
mdfreader) generated BEV boxes for height-classification traces of a second
vehicle programme and compared them to human labels. The accompanying report
describes echo-level trigonometry; **the source does something simpler**:

1. choose one production USS map object slot `Map_Obj{n}_P1x/P2x/P1y/P2y`
   (vehicle frame, mm) whose mid-x decreases and whose final sample is nearest
   a bumper line (x = 4200 / -1300 mm) with y near 0;
2. take its final mid-x only (lateral offset ignored), project along the final
   ego yaw from `VHM_VehicleState_XPos/YPos`;
3. draw a **fixed 0.75 x 1.75 m** rectangle aligned with the ego heading.

Prototype defects: `obj_number = 8` hard-coded inside the search loop, a live
`breakpoint()`, `Mass_label_generator` calls `compute_bb_coords` with a missing
argument, and the corner metric matches corners by nearest neighbour (not
one-to-one, so optimistic). Its saved 25-trace evaluation reproduces exactly
only without the hard-code (Box_40cm_802: 0.1739 m, Wall_smooth_930: 0.7170 m);
`autolabel.uss_box.box_v1` is that version.

## Code

| file | role |
|---|---|
| `src/vision_uss_research/autolabel/geometry.py` | polygon IoU, oriented boxes, corner / centroid metrics (pure numpy) |
| `src/vision_uss_research/autolabel/uss_box.py` | MF4 signals, v1 port, one-change ablations, Label V2 target parsing, `evaluate_sequence` |
| `notebooks/16_uss_autolabel_eval.ipynb` | corpus run (Colab, resumable) + per-class, size-confound and ablation analysis |
| `tests/test_autolabel.py` | geometry + selection + v1-equivalence tests |

## Ground truth in this corpus

Every sequence's `*_PAS_m_label_V2.csv` carries human `PolygonX/PolygonY`
(world mm, same frame as `VHM_VehicleState_XPos/YPos` in m) per labelled
object, with class 0 traversable / 1 low / 3 high. The evaluation uses rows of
class 1 or 3 (not `empty`); the headline set is single-target sequences.

## Ablations (each ONE change vs v1)

`lat_pos` / `lat_neg` add the lateral offset with either sign (the Map_Obj y
convention is undocumented; three prototype traces disagree), `sync` pairs the
last valid object sample with the ego pose at that timestamp, `sel2` selects the
slot on its last valid sample, `seg_pos` / `seg_neg` size the box by the map
object's P1-P2 segment. `*_oracle_size_iou` re-centres the human polygon's own
shape on a variant's centre: IoU with localisation error only.

## Open questions the corpus run answers

1. Per-class label quality with CIs, at ~6k-sequence scale.
2. Whether "object height determines label quality" survives controlling for
   footprint size (first 4-trace signal: a wall scores v1 IoU 0.34 but
   oracle-size 0.80 - a size failure, not a localisation one).
3. Which fixes help, and which lateral sign is right.

## Results log

**`al_eval_v1` smoke (2026-09-30, 300 sampled sequences; 195 scored, 172
single-target).** Skips: 89 no MF4/label, 16 no target polygon.
- v1 median IoU: low 0.381 (CI 0.23-0.58, n=41, 32 of them curbstones),
  high 0.249 (0.21-0.27, n=131). **The draft's height gap reverses on this
  corpus.** v1 IoU peaks where the human box area is near the 1.31 m² template
  and falls on both sides - a template-fit artefact, not a height effect.
- Lateral sign: `lat_pos` (+1, y left) cuts the median centroid error
  0.50 -> 0.27 m (high 0.51 -> 0.26) and adds +0.024 mean IoU (CI 0.013-0.036);
  `lat_neg` is worse. `sync` and `sel2` change nothing; `seg_*` boxes lose
  ~0.2 IoU (the P1-P2 segment is the reflecting face, not the footprint).
- Small high objects (pole, dummychild) fail on localisation as well as size
  (oracle-size IoU 0.13 / 0.09); cars fail on size (8 m² vs 1.31 m² template).

**Depth shift (4 hand-checked traces):** moving the box so its near edge sits
on the map object HURTS on all four - the map object already sits near the
human box centre (box802: 0.04 m).

**Labelling protocol (answered 2026-10-05 by a colleague):** PLAT_USS polygons
were placed with the help of the synchronized reference webcam video (true for
past programmes in general). The second programme (Gravity/"Lucid") recorded no
video; its labels were likely estimated from the USS map view plus known object
dimensions (its `*_psol_map.png` tool screenshots show a map-only view). So
PLAT_USS ground truth = camera-assisted human estimates with a generous margin
(75 mm pole -> ~1.2 x 0.5 m box), not surveyed positions. The depth-shift
result is physical, not a map-view artefact: thin objects reflect near their
centre; only cars need an offset (1.54 m). Within-class box size is fairly
stable (relative IQR of area 0.33-0.60 for 11 of 13 classes vs 1.68 overall),
which the class size prior exploits.

**`al_eval_v2`** adds the depth-shift variant, stored face geometry, approach
tags, and cross-fitted class-size priors (`glob`, `cls`, `cls_depth`).

**`al_eval_v2` full run (2026-10-04, git f28452f): 8,216 folders, 5,595 scored,
4,991 single-target (3,960 high / 1,031 low).** Offline analysis:
`PYTHONPATH=src python scripts/autolabel_analysis.py outputs/autolabel/al_eval_v2`
(size / offset priors 5-fold cross-fitted by sequence-id hash).

| variant | median IoU all (95% CI) | high | low | IoU>=0.5 |
|---|---|---|---|---|
| v1 (prototype as run) | 0.238 (0.233-0.244) | 0.233 | 0.307 | 15.7% |
| + lateral offset | 0.265 (0.261-0.269) | 0.264 | 0.301 | 17.4% |
| + class size prior (`cls`) | 0.405 (0.395-0.415) | 0.427 | 0.315 | 35.7% |
| + class placement offset | 0.434 (0.424-0.445) | 0.463 | 0.332 | 40.0% |
| + class x approach offset/size | 0.468 (0.458-0.478) | 0.482 | 0.321 | 45.5% |

Findings:
- The prototype shows no "high objects label better" effect: v1 low 0.307
  (0.27-0.36) vs high 0.233 (0.229-0.238).
- After the fixes the low median lags, but it is ONE class in ONE geometry:
  curbstone (820 of 1,031 low) at 45 deg approaches (IoU ~0.15-0.17, centroid
  error 1.1-1.3 m - the USS map sees the nearest point of a long, oblique curb).
  Other low objects are among the best: woodenboard 0.64, speedbump 0.63-0.66;
  central low 0.64 with the approach-aware prior.
- Placement causes: lateral offset (displaced high: centroid 0.76 -> 0.26 m),
  object depth for cars (GT centre 1.54 m beyond the map object; car IoU
  0.14 -> 0.43 with the class offset). Depth shift for ALL classes hurts.
- Remaining-gap oracles (cls size): true orientation 0.477, true centre 0.562.

**Robustness (2026-10-05, `scripts/autolabel_robustness.py`) - headline numbers
now DAY-GROUPED.** Priors cross-fitted with every recording day (58) inside one
fold; CIs from a day-cluster bootstrap. Sequence-level folds leaked
near-duplicate approaches: final median 0.468 -> **0.439 (CI 0.395-0.481)**,
IoU>=0.5 45.5% -> **41.8%**; review policy t=0.6: 29.5% accepted at 64.9%,
t=0.7: 12.7% at 73.3%. Height test (OLS, low-minus-high IoU): prototype
+0.083 -> -0.010 [-0.07, +0.05] once box size is controlled; v2 -0.083 with
size+approach -> -0.036 [-0.08, +0.01] once shape (elongation, obliqueness)
is added. Threshold curve: v2 beats v1 at every t (0.3: 67% vs 35%; 0.7:
13% vs 5%). All 5,595 scored sequences (multi-target incl.): 0.429.
