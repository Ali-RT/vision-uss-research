# Action Plan

## Project Direction

The project will build a **vision-USS/top-view supervised benchmark** for near-field parking obstacle understanding.

The key discovery is that Label V2 geometry is top-view/map-space, not front/rear camera pixel annotation. Therefore, the next work should focus on dataset curation, top-view overlay validation, benchmark table construction, and baseline models for severity and distance prediction.

---

## Phase 0: Lock the Research Direction

### Goal

Define the paper scope and prevent incorrect assumptions.

### Decisions

- Treat front/rear camera video as visual evidence.
- Treat Label V2 polygons as top-view/map-space geometry.
- Treat `VIDEO_USS_TOPVIEW` as the visual space where Label V2 polygons should align.
- Do not treat Label V2 `xmin/xmax/ymin/ymax` as front/rear camera image boxes.
- Do not frame the first paper as a YOLO-style 2D object detection dataset.

### Deliverables

- `docs/research_problem.md`
- `docs/action_plan.md`

### Acceptance Criteria

- The paper goal is clearly stated.
- Supported and unsupported claims are documented.
- The first benchmark tasks are defined.

---

## Phase 1: Dataset and Coordinate-System Validation

### Goal

Validate the relationship between Label V2 labels and top-view visualization across many sequences.

### Already Completed

- Sequence manifest built.
- Metadata JSON loaded.
- JSON tag subcategory normalization fixed.
- Driving direction parsed.
- Preferred camera rule defined:
  - `forward -> front`
  - `backward -> rear`
- `label_objects_json` extracted.
- Rule-ready sample created.
- Label V2 profiled.
- One uploaded sequence confirmed that Label V2 polygons align with top-view/map coordinates.

### Next Work

Generate top-view overlays for many sequences.

For each sequence:

```text
read Label V2 CSV
parse PolygonX / PolygonY
convert millimeters to meters
load VIDEO_USS_TOPVIEW image or frame
draw obstacle polygon and empty/traversable polygon
save QA overlay
```

### Deliverables

```text
outputs/debug/topview_overlays/
processed/labels/normalized_label_v2_rows.csv
processed/labels/normalized_sequence_labels.csv
processed/labels/topview_overlay_summary.csv
```

### Acceptance Criteria

- Overlay generation runs on the 200 rule-ready samples.
- Most top-view polygons visually align with the top-view grid.
- Failed/missing overlays are logged.
- Object row, severity class, and distance are readable in the overlay.

---

## Phase 2: Build the Benchmark Dataset Table

### Goal

Create one clean benchmark row per sequence or per selected frame.

### Required Fields

```text
sequence_id
sequence_dir
selected_camera
selected_camera_video
selected_camera_frame_path
topview_image_or_frame_path
topview_overlay_path
metadata_json
label_v2_csv
driving_direction
weather_tags
scene_tags
approach_tags
label_objects_json
target_object
selected_label_object
severity_class_id
severity_class_name
distance_mm
distance_m
distance_bin
topview_polygon_x_m
topview_polygon_y_m
split
```

### Frame Sampling Strategy

Start simple:

```text
one selected frame per sequence
```

Use a deterministic frame ratio first, for example:

```text
frame_ratio = 0.30
```

Later, extend to multiple ratios or short clips.

### Deliverables

```text
processed/dataset/benchmark_samples.csv
processed/dataset/train.csv
processed/dataset/val.csv
processed/dataset/test.csv
processed/frames/selected_camera/
processed/frames/topview/
processed/frames/topview_overlays/
```

### Acceptance Criteria

- Every benchmark row has a selected front/rear camera.
- Every benchmark row has a valid Label V2 row.
- Every benchmark row has object type, severity class, and distance.
- Split files are created without sequence leakage.
- Missing files and parsing issues are logged.

---

## Phase 3: Define Benchmark Tasks

### Task A: Severity Prediction

Input:

```text
selected front/rear frame or short clip
```

Target:

```text
0 = traversable
1 = low
2 = unknown
3 = high
```

Metrics:

```text
accuracy
macro F1
per-class recall
confusion matrix
```

### Task B: Distance-Bin Prediction

Input:

```text
selected front/rear frame or short clip
```

Target:

```text
distance bin derived from important Distance [mm]
```

Suggested bins:

```text
0-0.5 m
0.5-1.0 m
1.0-1.5 m
1.5-2.0 m
>2.0 m
```

Metrics:

```text
accuracy
macro F1
mean absolute bin error
near/far confusion
```

### Task C: Object-Type Prediction

Input:

```text
selected front/rear frame or short clip
```

Target:

```text
target object class
```

Metrics:

```text
top-1 accuracy
macro F1
per-object recall
```

### Task D: Camera-to-Top-View Prediction

Input:

```text
selected front/rear frame or clip
```

Target:

```text
top-view occupancy, spatial bin, or polygon-derived object location
```

This is an advanced task after the first baselines.

---

## Phase 4: Baseline Modeling

### Baseline 1: RGB Image Baseline

Input:

```text
one selected camera frame
```

Model options:

```text
ResNet
EfficientNet
MobileNet
DINOv2 frozen features
```

Prediction heads:

```text
severity head
distance-bin head
object-type head
```

### Baseline 2: Video / Clip Baseline

Input:

```text
3-8 frames from selected camera video
```

Model options:

```text
frame encoder + temporal pooling
CNN + LSTM
VideoMAE
TimeSformer
lightweight 3D CNN
```

Targets:

```text
severity
distance bin
object type
```

### Baseline 3: Depth-Feature Baseline

Run pretrained depth models on selected camera frames:

```text
Depth Anything V2
Metric3D
UniDepth
Depth Pro
```

Compare:

```text
RGB only
depth only
RGB + depth
```

Targets:

```text
severity
distance bin
```

### Baseline 4: Sensor-Supervised Student

Training:

```text
camera frame or clip supervised by Label V2 distance, severity, and top-view labels
```

Inference:

```text
camera only
```

This is the strongest final research angle.

---

## Phase 5: Evaluation Design

### Goal

Make the experimental results scientifically defensible.

### Split Rules

Avoid leakage:

- split by sequence,
- do not put frames from the same sequence into multiple splits,
- preserve object/severity balance when possible,
- preserve front/rear balance when possible.

### Core Metrics

Severity:

```text
accuracy
macro F1
per-class recall
confusion matrix
```

Distance bin:

```text
accuracy
macro F1
mean absolute bin error
```

Distance regression, if used:

```text
MAE in meters
RMSE in meters
relative error
```

Object type:

```text
top-1 accuracy
macro F1
per-class recall
```

### Deliverables

```text
outputs/experiments/baseline_rgb/
outputs/experiments/baseline_depth/
outputs/experiments/baseline_video/
reports/evaluation_summary.md
```

---

## Phase 6: Literature Positioning

### Goal

Show that the work is timely and not redundant.

### Literature Areas

Cover:

- monocular metric depth,
- video depth,
- near-field parking perception,
- USS-camera fusion,
- BEV/top-view supervision,
- parking/surround-view datasets,
- weak supervision and privileged-information learning.

### Core Positioning

The project does not compete directly with generic monocular depth models.

It asks:

```text
Do modern image, video, and depth models transfer to near-field parking obstacle severity and distance prediction when supervised by USS/top-view map labels?
```

### Deliverable

```text
docs/literature_review.md
```

---

## Phase 7: Paper Contribution Package

### Expected Contributions

1. Scalable curation pipeline for raw vision-USS parking sequences.
2. Benchmark schema linking camera video, metadata JSON, top-view labels, severity, distance, and object type.
3. Coordinate-system analysis showing Label V2 labels are top-view/map-space.
4. Baselines for severity, object type, and distance-bin prediction.
5. Study of whether pretrained monocular depth improves near-field parking distance/severity prediction.

### Suggested Title

```text
Vision-USS Parking: A Top-View Supervised Benchmark for Near-Field Obstacle Severity and Distance Prediction
```

Alternative:

```text
Camera-Based Near-Field Parking Obstacle Understanding with USS-Derived Top-View Supervision
```

---

## Phase 8: Publication Strategy

### Workshop-First Route

Most realistic initial target:

```text
CVPR autonomous driving workshop
ICCV/ECCV autonomous driving workshop
IV workshop
ITSC workshop
WACV workshop
```

This route is appropriate if:

- the data cannot be fully public,
- the contribution is mainly dataset curation and benchmark definition,
- the baseline results are useful but not state-of-the-art enough for a main conference.

### Stronger Conference Route

More realistic if:

- the curated benchmark is large,
- baselines are strong,
- a subset or evaluation protocol can be shared,
- the depth/sensor-supervision result is novel and convincing.

---

## Immediate Next Step

Start Phase 1 implementation:

```text
Build top-view overlay generation for all 200 rule-ready samples.
```

### Next Script To Implement

```text
scripts/generate_topview_label_overlays.py
```

### Script Responsibilities

For each row in `rule_ready_sample.csv`:

1. Load metadata/sample row.
2. Load Label V2 CSV.
3. Select relevant label rows.
4. Parse `PolygonX` and `PolygonY`.
5. Convert from millimeters to meters.
6. Load `VIDEO_USS_TOPVIEW` still image or a frame from top-view video.
7. Map top-view meter coordinates to image pixels using axis/grid inference or explicit bounds.
8. Draw polygons, object names, severity class, and distance.
9. Save overlay image.
10. Write summary CSV.

### Expected Outputs

```text
outputs/debug/topview_overlays/{sequence_id}/topview_overlay.jpg
outputs/debug/topview_overlays/topview_overlay_summary.csv
```

### After That

Build:

```text
processed/dataset/benchmark_samples.csv
```

Then train the first baseline:

```text
selected camera image -> severity class + distance bin
```
