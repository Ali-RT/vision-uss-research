# USS Auto-Labeler Track (IEEE IV 2027 candidate)

Separate paper track from WACV 2660. IV 2027 deadline: **2026-11-15**
(notification 2027-01-15, 6 pages). Frozen WACV numbers are never touched;
all outputs of this track live under `outputs/autolabel/al_*` (Drive:
`vision_uss_research_artifacts/outputs/autolabel/`).

## Origin

A Bosch prototype (M. K. Uludag, Aug 2024, `helpers_mf4.py` / `Mass_tester.py`,
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
human box centre (box802: 0.04 m). Likely the Label V2 polygons were drawn over
the USS map view, so the "ground truth" is human agreement on the USS map, not
a physical measurement. **Must be confirmed with the labelling team and stated
in the paper.**

**`al_eval_v2`** adds the depth-shift variant, stored face geometry, approach
tags, and cross-fitted class-size priors (`glob`, `cls`, `cls_depth`).
