# vision-uss-research

Research pipeline for converting large-scale multi-sensor parking/USS sequences into a reviewed, camera-selected, frame-level dataset for computer vision experiments.

## Current scope
- inventory raw sequence folders
- parse JSON metadata and labels
- infer preferred front/rear camera
- sample candidate sequences
- extract candidate frames
- curate reviewed dataset subsets

## Development setup
- local development on Mac/Linux
- large raw data stored on Google Drive and accessed from Colab
- environment and dependency management with uv