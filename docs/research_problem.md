# Research Problem

## Working Title

**Vision-USS Parking: A Top-View Supervised Benchmark for Near-Field Obstacle Severity and Distance Prediction**

## One-Sentence Summary

This project builds a curated benchmark from raw parking/USS measurement sequences to study whether front/rear camera video can predict near-field obstacle type, severity, and distance using USS/top-view map-space labels as supervision.

## Core Problem

Low-speed parking and near-field ADAS perception require reliable understanding of obstacles close to the vehicle, such as curbstones, poles, bushes, dummy children, barriers, speed bumps, bollards, and vehicle edges. These objects are difficult for camera-only perception because they may be small, partially visible, low height, close to the bumper, or poorly represented by standard far-field driving datasets.

The available raw data contains rich information, but it is not directly research-ready. Each sequence may include front/rear camera videos, top-view USS visualization, metadata JSON, Label V2 CSV/XLSX labels, and measurement files. The key challenge is converting this heterogeneous engineering data into a clean benchmark with correct coordinate interpretation and reproducible sample selection.

## Key Discovery

The Label V2 `xmin`, `xmax`, `ymin`, `ymax`, `PolygonX`, and `PolygonY` fields are not front/rear camera pixel coordinates.

They are best interpreted as **top-view/map-space coordinates**, likely stored in millimeters, while the `VIDEO_USS_TOPVIEW` visualization displays the same coordinate system in meters.

Therefore:

```text
Label V2 polygon / bbox  -> top-view or map-space geometry
front/rear camera video  -> visual evidence
VIDEO_USS_TOPVIEW        -> spatial visualization where polygons can be overlaid
```

This means the project should not be framed as a standard 2D object detection dataset unless a valid camera projection/calibration step is later recovered.

## Research Question

Can front/rear camera video predict near-field parking obstacle type, severity, and distance when supervised by USS-derived top-view/map-space labels?

## Supported Labels

The current data supports the following label types:

| Signal | Source | Use |
|---|---|---|
| Driving direction | Metadata JSON | Select front/rear camera |
| Preferred camera | Derived from driving direction | `forward -> front`, `backward -> rear` |
| Object tags | Metadata JSON / `label_objects_json` | Sequence-level target object selection |
| Object type | Label V2 `Which object` | Target object label |
| Severity class | Label V2 class column | Traversable / low / unknown / high |
| Distance | Label V2 `important Distance [mm]` | Distance regression or distance-bin prediction |
| Top-view polygon | Label V2 `PolygonX`, `PolygonY` | Top-view/map-space obstacle geometry |
| Top-view visualization | `VIDEO_USS_TOPVIEW` image/video | Visual QA and overlay |
| Weather / scene / approach | Metadata JSON | Dataset stratification and analysis |

## Unsupported Claims

The current evidence does **not** support the following claims:

```text
We have 2D camera-image bounding boxes for front/rear camera frames.
```

```text
This is directly YOLO-ready camera object detection data.
```

```text
Label V2 polygon coordinates can be drawn directly on front/rear images.
```

These claims require a valid projection/calibration transform from top-view/map coordinates to camera pixels.

## Proposed Benchmark Tasks

### Task A: Obstacle Severity Prediction

Input:

```text
selected front/rear frame or short camera clip
```

Target:

```text
0 = traversable
1 = low obstacle
2 = unknown
3 = high obstacle
```

This is the cleanest first task because it is directly supported by Label V2.

### Task B: Distance-Bin Prediction

Input:

```text
selected front/rear frame or short camera clip
```

Target:

```text
important Distance [mm], converted into bins
```

Example bins:

```text
0-0.5 m
0.5-1.0 m
1.0-1.5 m
1.5-2.0 m
>2.0 m
```

Distance bins should be used before exact regression because the labels may be noisy and the visible target may vary by frame.

### Task C: Object-Type Prediction

Input:

```text
selected front/rear frame or short camera clip
```

Target:

```text
curbstone, pole, bush, dummychild, speedbump, car, ubarrier, bollard, etc.
```

This can be based on both metadata JSON object tags and Label V2 `Which object`.

### Task D: Camera-to-Top-View Prediction

Input:

```text
front/rear camera frame or clip
```

Target:

```text
top-view obstacle location, occupancy cell, or polygon-derived spatial bin
```

This is the most advanced task and should come after the simpler severity and distance tasks.

## Main Contributions

The paper can contribute:

1. **A scalable curation pipeline** for raw vision-USS parking sequences.
2. **A benchmark dataset schema** linking front/rear camera video, metadata JSON, top-view USS visualization, Label V2 class, object type, distance, and top-view polygon geometry.
3. **A coordinate-system analysis** showing that Label V2 geometry is top-view/map-space, not camera-pixel annotation.
4. **Baseline models** for near-field obstacle severity, object type, and distance-bin prediction.
5. **An empirical study** evaluating whether modern monocular depth features improve near-field parking obstacle understanding.

## Paper Positioning

This work is not primarily a generic monocular depth paper. Instead, it studies near-field parking obstacle understanding using sensor-derived top-view supervision.

The paper should be positioned relative to:

- monocular metric depth estimation,
- video depth estimation,
- near-field parking perception,
- USS-camera fusion,
- BEV/top-view supervision,
- autonomous-driving dataset curation.

## Expected Outcome

A research-ready benchmark and baseline study showing how camera video can be paired with USS/top-view map-space labels to support near-field parking obstacle severity and distance prediction.

The first publishable version should target an autonomous driving / intelligent vehicles / computer vision workshop. A stronger venue becomes realistic if the benchmark is large, the baselines are strong, and at least part of the dataset or evaluation protocol can be released.
