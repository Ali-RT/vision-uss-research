# Run Tracking Convention

Every pipeline stage writes to `<artifacts>/<stage>/<run_id>/` with a `run.json`
recording the git sha, frozen config, creation timestamp, input runs consumed,
and a summary. The chain report -> model -> dataset -> label runs -> inventory ->
git sha is what makes a result reproducible.

## Run IDs are STABLE, not dated

`new_run_id(stage, tag)` returns `"{stage}_{tag}"` (e.g. `dataset_v2`,
`train_yolo_v1`) - **no date**. Rerunning the same config targets the *same*
run dir, so a build interrupted and resumed on a later day continues in place.

This replaced the earlier `{date}_{stage}_{tag}` scheme, which minted a new dir
whenever a run spilled across midnight and left cross-run paths day-dependent
(a model created today pointed at `datasets/20260717_dataset_v2`, but rerunning
notebook 05 the next day produced `datasets/20260718_dataset_v2`). The creation
time still lives in `run.json`; it is just no longer in the path.

- Want a genuinely separate run? **bump the tag** (`v2` -> `v3`) in the config.
- Want one-per-invocation (throwaway sweeps)? `new_run_id(..., dated=True)`.

## Discovery is by timestamp, not name

`latest_run(base, require_file=..., exclude_suffix=...)` returns the newest run
by `run.json` `created` (falling back to dir mtime) - robust to both stable and
legacy date-prefixed ids, so old runs still resolve after the switch.

`resolve_run(base, stage, tag=None, run_id=None, require_file=...)` is the single
rule every notebook uses to answer "which run do I consume":
1. explicit `run_id` wins;
2. else the stable `{stage}_{tag}` dir if it exists;
3. else the newest run.

## Cross-run references

A run records the runs it consumed in `run.json["inputs"]` as resolved run ids
(e.g. `["datasets/dataset_v2"]`). Because ids are now stable, these references
stay valid across days. Notebook 07 walks `model -> inputs[0] -> dataset` to find
what to evaluate, so the provenance is self-describing from the report backward.

## Notebook knobs

| notebook | pin input | pin own run |
|---|---|---|
| 02 labeling | `INVENTORY_RUN` | `RUN_ID_OVERRIDE` |
| 05 dataset | `LABEL_SOURCES` (auto) | `RUN_ID_OVERRIDE` |
| 06 train | `DATASET_RUN` | stable from config tag |
| 07 eval | `MODEL_RUN` | stable (`eval_test`) |

Leave them `None` for the default resolution; set them only to target a
specific historical run.
