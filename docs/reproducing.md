# Reproducing the Full Pipeline

Status 2026-08-02. The complete runbook to regenerate every result from raw
data, in order. Companion to `docs/run_tracking.md` (run-ID conventions) and
`docs/v2_open_set_approach.md` (the results themselves).

## Environments

| where | what runs there | setup |
|---|---|---|
| Colab (GPU) | notebooks 02, 06 (SAM3 labeling, YOLO training) | bootstrap cell in every notebook |
| Colab (CPU ok) | notebooks 01, 03-05, 07-15 | same bootstrap |
| Local Mac | unit tests, repo work | `.venv/bin/python -m pytest tests/ -q` (66 tests) |

Every notebook's **first cell is identical**: mounts Drive, `git pull
--ff-only` on `/content/vision-uss-research`, installs
`requirements/colab.txt`, adds `src/` to `sys.path`, loads
`configs/paths/colab_drive.yaml`. Knobs available in each: `ARTIFACTS_OVERRIDE`,
`RAW_ROOT_OVERRIDE`.

**Golden rules (each one was learned the hard way):**
1. After any `git pull` that changes `src/`, **restart the Colab runtime** -
   Python caches imported modules; the pulled code will not load otherwise.
2. Never materialize images onto Drive - Google Drive degrades past ~10k
   items per folder. Datasets live as **manifests** on Drive; images are
   copied to Colab local disk at train/eval time (notebooks 06/07 do this).
3. All notebooks are **resumable**: rerunning skips completed work (only rows
   with `error == ""` count as done; errored rows are retried). A Colab
   disconnect never loses more than one flush interval.
4. Drive I/O errors (Errno 5) are retried with backoff automatically
   (`with_drive_retry`); persistent failures are recorded and skipped, never
   fatal.
5. Locally, always use `.venv/bin/python` (plain `python3` resolves
   inconsistently on this Mac).

## Data layout (input)

Raw sequences on Drive: ~8,200 folders, each with front/rear `.mp4` videos, an
`.mf4` measurement file, a `NAMEoFFILE` label file, and a `CusReplay/` resim
tree containing `*_Classifier_Monitor_Normed.csv.xz` (the production USS
height classifier log). Path root set in `configs/paths/colab_drive.yaml`.

## Run tracking

Stable run IDs `{stage}_{tag}` (no dates). Every run writes `run.json` with
git SHA, config, input run IDs, and a summary - so any artifact chains back to
raw data: report -> model -> dataset -> labels -> inventory -> git commit.
Discovery via `resolve_run` / `latest_run` (newest `created` wins). See
`docs/run_tracking.md`.

## Track A - detector (notebooks 01-09, in order)

| step | notebook | config / knobs | output (Drive) | GPU |
|---|---|---|---|---|
| 1. Inventory | 01_build_object_inventory | `SMOKE_TEST` knob | `inventory_*` (objects, folders, label rows) | no |
| 2. Batch labels v1 | 02_batch_sam3_labeling | `labeling_sam3text.yaml` | `labels_sam3text` boxes.csv | yes |
| 3. Batch labels v2 buckets | 02 again, `CONFIG_NAME="labeling_sam3text_v2extra"` | `labeling_sam3text_v2extra.yaml` | `labels_sam3text_v2extra` | yes |
| 4. Click-assisted labels | 03_click_assisted_labeling | in-notebook (`REDO_OBJECTS`, box mode for bicyclestand) | `labels_clicks_v1` (gold) | yes |
| 5. QA gate | 04_label_qa_review | auto-discovers label runs | flag rates, galleries | no |
| 6. Dataset manifest | 05_build_yolo_dataset | `CONFIG_NAME="dataset_v2"` (`dataset_v2.yaml`: class_groups, cone holdout) | `dataset_v2` manifest + provenance | no |
| 7. Train | 06_train_yolo | `train_yolo_v1.yaml` (YOLO11s) | `train_v2` weights | yes |
| 8. Eval | 07_posttrain_eval | CLASS_BINS in-notebook | mAP, Low/High, open-set tables | yes |
| 9. Visualize (optional) | 08, 09 | - | galleries, videos, sim export | yes |

Human input required: **~170 clicks** in notebook 03 (woodenboard,
dummychild points; bicyclestand 2-corner boxes). Everything else is fully
automatic. Expected reference numbers: ~136k boxes; QA flag rate ~8% after
per-class thresholds; mAP50 0.79; matched-detection Low/High 99.8%;
per-sequence 93.6%; cone-holdout `high_other` mAP50 0.33 / recall 0.42.

## Track B - alignment + fusion (notebooks 10-15, in order)

| step | notebook | needs | output / reference numbers | GPU |
|---|---|---|---|---|
| 10. Alignment EDA | 10_mf4_alignment_eda | raw MF4s | 8,055 probed; 5,917 with MF4; 4,631 alignable (78.3%) | no |
| 11. Distance validation | 11_uss_distance_validation | nb10 | 1,424 accepted (monotonic approach); objbuff source | no |
| 12. Focal calibration | 12_focal_calibration_height | nb11 | focal ~164 px-units; pole aggregate err ~4% | no |
| 13. Geometric Low/High | 13_geometric_lowhigh_score | nb12 | bal-acc 0.64, AUC 0.70 (weak - a finding) | no |
| 14. Camera-USS fusion | 14_camera_uss_fusion | nb07 model + CusReplay | USS 62.9%, camera 93.6%, camera rescues 93.0% of USS errors | yes |
| 15. Meta-fusion | 15_meta_fusion | nb14 `fusion_signals.csv` | USS 0.734 / det 0.931 / meta 0.933 bal-acc (marginal gain) | no |

Notebooks 10/11 install `asammdf` themselves; they are the heavy jobs (hours,
fully resumable with on-Drive `progress.json` heartbeats).

## Key domain facts baked into the code (do not rediscover)

- Direction-relevant camera only: forward -> front (`WebCam`), backward ->
  rear (`WebCam3`). MF4 camera channels pair each frame PTS to master time
  exactly (`docs/mf4_video_alignment.md`).
- `NAMEoFFILE` distance field is a spec constant (2000mm) - never use it;
  real distance comes from MF4 `MAP_ObjBuff_*_LastDetDist` (mm, ~5m range) or
  CusReplay ObjDist.
- `distance_at` returns NaN outside the reading window - never clamps
  (clamping fabricates distances and corrupts heights).
- USS decision = dominant tracked object's median `ClassProbHigh`/255,
  threshold 0.5 (`uss_class.py`) - this is the production baseline.
- Monocular limit: flat objects' box-height overestimates ~5x (box top is a
  farther ground point). Fundamental; motivates the learned detector.

## Verification

```bash
.venv/bin/python -m pytest tests/ -q
```

66 tests cover every src module including the failure modes above (no-clamp,
resume-skips-errored, tiny-class splits, per-class QA thresholds, Drive-IO
survival, class-balanced fusion).
