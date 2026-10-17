# Experiment configuration and provenance

Tracked metadata for the already-completed academic experiment:

- `phash-review-seed-42.csv`: adjudicated potential near-duplicate samples.
- `final-selection-seed-42-phash-reviewed.json`: frozen model selection based **only** on validation macro-F1.

These files are not raw images or checkpoints. They record how the published coursework results were obtained. Do **not** edit them or recreate the sealed evaluation to make the repository easier to run.

Per-run resolved configurations, checkpoints and predictions remain local under `runs/`; the data manifest remains under `data/manifests/`, both ignored by Git. See [experiment status](../docs/EXPERIMENT_STATUS.md).
