# Pipeline Restructure Plan: Labeling -> Training -> Post-Training

Goal: a reproducible, Colab-first pipeline where every artifact traces back to a
run ID, a config snapshot, and a git commit. Motivated by real failures during the
labeling sprint: mixed CSV schemas across sessions, stale absolute frame paths,
overwritten artifacts, and helper code duplicated across five notebooks.

## 1. Repo layout (target)

```
vision-uss-research/
├── configs/
│   ├── paths/                     # local.yaml, colab_drive.yaml        (exists)
│   ├── datasets/                  # catalog + class mapping             (exists)
│   └── experiments/               # NEW: one frozen yaml per run type
│       ├── labeling_sam3text.yaml #   objects, prompts, thresholds, frame window
│       ├── dataset_v1.yaml        #   sources, class ids, split rules
│       └── train_yolo_v1.yaml     #   model, epochs, imgsz, augment, seed
├── src/vision_uss_research/
│   ├── settings.py                                                     (exists)
│   ├── runs.py                    # NEW: run-id + run.json (git sha, config snapshot)
│   ├── sequences.py               # NEW: camera selection, frame extraction,
│   │                              #      drive-retry  (now duplicated in 4 notebooks)
│   ├── labeling/
│   │   ├── boxes_io.py            # NEW: BOX_COLS, tolerant loader, append_csv, row_bbox
│   │   ├── corridor.py            # NEW: corridor prior + instance picker
│   │   └── sam3.py                # NEW: PCS + tracker wrappers (API-marked lines)
│   ├── qa/flags.py                # NEW: box_iou + flag rules (per-class thresholds)
│   └── datasets/yolo_export.py    # NEW: merge label runs -> YOLO dataset + provenance
├── notebooks/                     # thin drivers that import from src
│   ├── 01_build_object_inventory.ipynb
│   ├── 02_batch_sam3_labeling.ipynb
│   ├── 03_click_assisted_labeling.ipynb
│   ├── 04_label_qa_review.ipynb
│   ├── 05_build_yolo_dataset.ipynb        # NEW (merge + splits)
│   ├── 06_train_yolo.ipynb                # NEW
│   ├── 07_posttrain_eval.ipynb            # NEW
│   └── exploratory/                       # depth failure analysis, sam3 gate test
├── scripts/                       # CLI twins / one-off tools
├── requirements/colab.txt         # NEW: pinned transformers, ultralytics, etc.
└── docs/
```

Numbering = pipeline order. Shared logic lives in `src/`; notebooks stop carrying
their own copies (the mixed-schema and stale-path bugs both came from divergent
duplicates).

## 2. Artifact layout on Drive (target)

```
vision_uss_research_artifacts/
├── manifests/                       # stage 0, rarely redone
├── inventory/<run_id>/              # inventory outputs, versioned
├── frames/<window_id>/<seq>/<cam>/  # frame cache SHARED across runs;
│                                    # window_id = r30-85_n20 (params in the path)
├── labels/
│   ├── runs/<run_id>/               # batch SAM3 runs: boxes csv, seed_log,
│   │   └── run.json + config.yaml   #   failures, QA outputs for this run
│   └── clicks/<run_id>/             # click runs: clicks.json + boxes csv
├── datasets/<dataset_id>/           # merged YOLO datasets (images/, labels/,
│                                    #   splits, dataset.yaml, provenance.json)
├── models/<train_run_id>/           # weights, args, metrics, curves
└── reports/<eval_run_id>/           # post-training eval outputs
```

**Run convention.** Run ID = `YYYYMMDD_<stage>_<tag>` (e.g. `20260722_labels_sam3text_v1`).
Every run writes `run.json`: git commit, frozen config, input artifact IDs, timestamps.
Reproducibility chain: report -> model -> dataset (provenance.json) -> label runs ->
inventory run -> git sha. Frame paths in CSVs become RELATIVE to the frames root,
killing the stale-absolute-path class of bugs permanently.

## 3. Pipeline stages

- **L (labeling):** 01 inventory -> 02 batch SAM3 -> 03 click/box -> 04 QA gate.
- **D (dataset):** 05 merges selected label runs, applies class IDs from
  `configs/datasets/`, splits BY SEQUENCE, reserves ~20% of *clicked* sequences as
  the gold test set (never trained on), writes YOLO dir + `dataset.yaml` + provenance.
- **T (training):** 06 runs ultralytics YOLO (v8/11-s) from a config yaml; outputs to
  `models/<run_id>/`; fixed seeds; metrics logged.
- **P (post-training):** 07 evaluates on the gold test set: per-class mAP/recall,
  low/high binary accuracy, per-distance stratification (CusReplay ObjDist), confusion;
  later: fusion vs USS `ClassProbHigh`; ONNX export.

## 4. Colab convention

First cell of every notebook: clone-or-pull the repo, `pip install -r
requirements/colab.txt` (pinned versions - transformers/ultralytics drift has already
bitten us once), mount Drive, `load_paths("colab_drive")`, print the git sha that ends
up in run.json.

## 5. Reset & redo (reproducibility pass)

1. **Commit the current repo state first** (much of the work is untracked) - baseline tag.
2. **Archive, don't delete, Drive artifacts**: move everything current to
   `_archive_premigration/`. The clicks.json files are ~200 human clicks - they are
   imported into the new structure, NOT redone.
3. Implement the reorg (src modules -> notebook refactor -> new notebooks 05-07).
4. Redo the chain under proper run IDs:
   inventory v2 -> full batch labeling run -> click import + any re-clicks ->
   QA gate -> dataset v1 -> train v1 -> eval v1.

Implementation order (each independently mergeable):
- **PR1**: `src/` shared modules + `runs.py` + `requirements/colab.txt` + configs/experiments skeletons.
- **PR2**: regenerate notebooks 01-04 as thin drivers on src (notebook builders exist,
  so this is mostly mechanical); move gate/depth notebooks to `exploratory/`.
- **PR3**: 05 dataset merge/split + 06 training + 07 eval.
- **PR4**: Drive migration helper (archive + click import) + full redo run.
