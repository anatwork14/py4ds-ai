# CO3117 - Machine Learning Project (Semester 252, AY 2025-2026)

## 📌 Course Information
- **Course Name:** Machine Learning
- **Course Code:** CO3117
- **Semester:** 252
- **Academic Year:** 2025-2026

## 👨‍🏫 Instructor
- **Dr. Trương Vĩnh Lân**

## 👥 Team Members

| Name | Student ID | Role |
| :--- | :--- | :--- |
| **Nguyễn Văn An** | 2352013 | Leader |
| **Huỳnh Vương Khang** | 2350011 | Member |
| **Nguyễn Tấn Phát** | 2352889 | Member |
| **Lê Đặng Khánh Quỳnh** | 2353037 | Member |

## Project Objective

Build a reproducible six-class image-classification comparison for the Intel Natural Scene Classification dataset: classical HOG/SIFT-BoVW models, frozen ImageNet ResNet18 features, and trained ResNet18 models. The fixed alphabetical label map is `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`.

## Data and Split Protocol

Original files are kept read-only. The deterministic validation split is drawn from `seg_train`; `seg_test` remains the final test set; `seg_pred` is retained as an unlabeled prediction pool. Exact SHA-256 overlap is checked. pHash candidates are reviewed through the explicit ledger [`configs/phash-review-seed-42.csv`](configs/phash-review-seed-42.csv); 43 non-test rows were excluded conservatively, and all test rows were preserved. Only those flagged pairs were visually audited for leakage, not for exploratory analysis or model selection. Four near-duplicate candidate pairs within test remain a stated independence limitation. Exploratory sample montages use training images only.

The real-data audit and its caveats are recorded in [`docs/PHASE1_DATA_AUDIT.md`](docs/PHASE1_DATA_AUDIT.md). The dataset is not included in Git.

## Quick Start

Requires Python 3.12 and `uv`. **Platform scope:** the current lock pins `torch` and `torchvision` to the PyTorch `cu126` index and includes wheels for Python 3.11/3.12 on Linux (`manylinux_2_28`, x86_64 and aarch64) and Windows (x86_64). This describes wheel coverage in `uv.lock`, not verified end-to-end compatibility on every listed platform. Other platforms—especially macOS, for which the pinned PyTorch packages have no wheels in this lock—are not currently verified; the quick start is not a cross-platform compatibility claim.

```bash
uv sync --locked --extra dev

# After reviewing Kaggle's dataset terms and configuring user-managed credentials:
uv run --locked --extra dev py4ds-download \
  --output-dir data/raw --accept-terms \
  --extract-to data/raw/extracted

uv run --locked --extra dev py4ds-prepare \
  --data-root data/raw/extracted \
  --output-dir data/manifests/seed-42 \
  --val-fraction 0.15 --seed 42

# Apply the reviewed pHash ledger; this is the manifest used for comparisons:
uv run --locked --extra dev py4ds-prepare \
  --data-root data/raw/extracted \
  --output-dir data/manifests/seed-42-phash-reviewed \
  --val-fraction 0.15 --seed 42 \
  --phash-review-ledger configs/phash-review-seed-42.csv

uv run --locked --extra dev pytest -q
uv run --locked --extra dev ruff check src tests scripts
```

If the `kaggle` executable is not installed, the Kaggle CLI can be invoked ephemerally after setting up its normal credentials:

```bash
mkdir -p data/raw
cd data/raw
uvx --from kaggle kaggle datasets download puneet6060/intel-image-classification
```

Then safely extract the downloaded ZIP using `extract_archive_safely` (see the reproduction command in `docs/PHASE1_DATA_AUDIT.md`). The Kaggle CLI reported the license label `copyright-authors`; review the dataset terms before use or redistribution. Never commit the archive or extracted images.

## Research and Reproducibility

Follow [`AGENTS.md`](AGENTS.md), [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md), and [`docs/LITERATURE_REVIEW.md`](docs/LITERATURE_REVIEW.md). Each experiment must save its resolved configuration, split hash, predictions, metrics, plots, and environment provenance. Hyperparameter and checkpoint selection must use train/validation only; final test evaluation occurs only after selection is frozen.

**Status:** repository and data audits are recorded. Earlier HOG, SIFT-BoVW, frozen ResNet18, and CNN runs used the pre-review manifest and are superseded. All five clean-manifest runs are complete; validation-only selection chose the layer4-fine-tuned ResNet18 CNN (validation macro-F1 0.929137). Its one-time held-out evaluation achieved 0.930000 accuracy and 0.931004 macro-F1 on 3,000 test rows. The source-grounded results and caveats are in [`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md); the saved test artifacts remain local. Historical notebook/report scores are not current measurements.

To regenerate the report's summary and metric tables from the local frozen selection and run artifacts (without evaluating test again):

```bash
uv run --locked --extra dev python scripts/render_report.py
```

For the independent, **read-only** audit of the already-saved held-out
predictions (no image loading or model inference), run:

```bash
uv run --locked --extra dev python scripts/verify_saved_final_artifacts.py \
  --expected-metrics-sha256 4df9443c7ad28c9ddd36cc461d38f5f7ed1a0b43b819d2ea0e400dedf9b4967e \
  --expected-predictions-sha256 85d68f7fac4351716964ec0db773126805ac530c269ed37c607af8cae0486dde
```

See [the final academic reviewer gate](docs/FINAL_REVIEW_2026-10-08.md)
for the full Ubuntu acceptance checklist. These commands must not be confused
with a repeat of `py4ds-evaluate-final`, which remains prohibited for this
sealed test manifest.
