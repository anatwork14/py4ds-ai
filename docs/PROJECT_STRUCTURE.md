# Project structure and reproducibility

This repository uses a **lightweight Python `src/` layout** suitable for an undergraduate machine-learning assignment. There are no production API services, container deployments, agent managers or new benchmark runs.

## Directory map

| Path | Purpose | Versioned? |
|---|---|---|
| `src/py4ds_ai/data/` | Download, read-only audit, manifest/split preparation | Yes |
| `src/py4ds_ai/features/` | HOG, SIFT and Bag-of-Visual-Words feature extractors | Yes |
| `src/py4ds_ai/models/` | Linear classifiers, ResNet features, CNN training, selection and metrics | Yes |
| `scripts/` | Thin command-line wrappers, report renderer and saved-artifact verifier | Yes |
| `tests/` | Synthetic unit and contract tests | Yes |
| `configs/` | Frozen selection and pHash review ledger | Yes |
| `docs/` | Report, literature, dataset/repository audits, experiment log | Yes |
| `notebooks/` | Optional future exploration; no canonical experimental results | README only |
| `legacy/` | Original notebook, PDF, historical arrays and helper modules | Yes, unchanged blobs |
| `data/raw/` | Kaggle dataset, managed locally | No |
| `data/manifests/` | Locally persisted reviewed manifest | No |
| `runs/` | Metrics, checkpoints, predictions and plots | No |
| `artifacts/` | Optional local exported outputs | No |

## Environment

The `pyproject.toml` and `uv.lock` are the dependency source of truth; the original `legacy/requirements.txt` is kept only as an archive. Python 3.12 is the documented environment. See [README](../README.md) for Windows PowerShell setup, tests, and Ruff linting.

The current package has CLIs for data preparation, feature baselines, ResNet and CNN training, model selection and final evaluation. Those commands exist for research reproducibility, **not** as a recommendation to reexecute completed experiments.

## Scientific protocol

1. Preserve original input images and the reviewed data-manifest identity.
2. Split training/validation from the original `seg_train`; retain `seg_test` as sealed test.
3. Fit preprocessing and feature vocabularies on training only.
4. Compare all methods via validation macro-F1 and freeze a selected checkpoint.
5. Score the selected checkpoint once on held-out test; **already completed historically**.
6. Produce the final report from saved evidence and document limits.

The original study has completed step 5; it must **not** be repeated merely because files were reorganized. The new directory layout does not change model weights, predictions, split identities, or reported metrics.

## Locally stored evidence

The selected checkpoint was saved under `runs/seed-42-phash-reviewed/cnn18/layer4-imagenet-v1/best_checkpoint.pt`. Historical test metrics/predictions are under `runs/seed-42-phash-reviewed/final-evaluation/`, and the guard is under `runs/.test_evaluation_locks/`.

These items are not committed. For the scientific caveats and exact verification hashes, consult [FINAL_REPORT](FINAL_REPORT.md) and [EXPERIMENT_STATUS](EXPERIMENT_STATUS.md).

## Historical paths

In the original version, files such as `report.pdf`, `features/*.npy`, `modules/*.py`, `notebooks/main.ipynb` and `dataset_metadata.json` were at the repository root. Those paths in **historical audit records** refer to their original locations. Their byte-identical archived copies now live under `legacy/`. Historical commands that used the former paths should not be executed as current reproduction steps.

## Verification boundaries

The prior accepted revision reported 117 passing unit/synthetic tests. That is **historical** evidence, not a new test result for the restructured branch. Windows-native package installation and a final-tree pytest run must be performed locally before claiming verification on that platform. The documented baseline does not require CUDA training, Docker, CI/CD, or online services.
