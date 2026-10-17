# CO3117 — Machine Learning Project (Semester 252, AY 2025–2026)

**Topic:** Intel Natural Scene Classification (six classes). This repository contains our course-assignment implementation, documentation, and historical experimental results.

**Main report:** [Final Evaluation Report](docs/FINAL_REPORT.md) · [Literature Review](docs/LITERATURE_REVIEW.md)

The reviewed implementation has been merged into `main`. Earlier local checks reported passing tests, but the final merged version has not yet been independently tested on native Windows. Saved test artifacts are not included in Git.

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

## Quick start — Windows PowerShell

Install [Git](https://git-scm.com/), [Python 3.12](https://www.python.org/downloads/), and [uv](https://docs.astral.sh/uv/). From PowerShell:

```powershell
git clone https://github.com/anatwork14/py4ds-ai.git
cd py4ds-ai
uv python install 3.12
uv sync --locked --extra dev --python 3.12
uv run --locked --extra dev python -m pytest -q
uv run --locked --extra dev ruff check src tests scripts
```

The lockfile includes Windows x86-64 wheels for the pinned CUDA-enabled PyTorch packages. This is **not** a claim of verified Windows/CUDA GPU training support. Unit tests can generally run without GPU acceleration.

## Dataset and reproducibility

The [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) is not distributed with the repository. Respect its usage conditions. Original images and saved run artifacts stay local and are ignored by Git.

The reviewed comparison uses a fixed split: **11,868 training**, **2,100 validation**, and **3,000 held-out test** images. The [pHash review ledger](configs/phash-review-seed-42.csv) records near-duplicate decisions, and the [data audit](docs/PHASE1_DATA_AUDIT.md) explains the protocol.

The following files are **historical legacy artifacts**, not the current source of evaluation results: `modules/`, `notebooks/main.ipynb`, `report.pdf`, and `features/`. Do not execute old preprocessing code against the source dataset, since it may move or delete images.

The maintained implementation resides in `src/py4ds_ai/`; command-line scripts are in `scripts/`, with synthetic/unit tests in `tests/`. The [implementation plan](docs/IMPLEMENTATION_PLAN.md) and [experiment log](docs/EXPERIMENT_STATUS.md) describe the design and runs.

## Results

Five approaches were compared using validation evidence: HOG/classical ML, SIFT–Bag-of-Visual-Words, frozen ResNet18 embeddings, ResNet18 classification-head training, and ResNet18 layer4 fine-tuning.

The selected **layer4 fine-tuned ResNet18** achieved **0.929137 validation macro-F1**. Its one historical held-out evaluation achieved **93.00% accuracy**, **0.931004 macro-F1**, and **0.929743 weighted-F1** on 3,000 test images. See [the final report](docs/FINAL_REPORT.md) for per-class results, citations, and limitations.

**Do not rerun the held-out test evaluator** for this frozen benchmark. If the original saved artifacts are available locally, the following commands safely rerender the report or audit saved prediction files without model inference:

```powershell
uv run --locked --extra dev python scripts/render_report.py
uv run --locked --extra dev python scripts/verify_saved_final_artifacts.py --expected-metrics-sha256 4df9443c7ad28c9ddd36cc461d38f5f7ed1a0b43b819d2ea0e400dedf9b4967e --expected-predictions-sha256 85d68f7fac4351716964ec0db773126805ac530c269ed37c607af8cae0486dde
```

**Limitations:** one split/seed; four potential near-duplicate test-image pairs; a historical dirty training worktree; incomplete comparable compute timings; and unverified image redistribution rights. Original dataset images and image-bearing outputs should not be published until their rights are established.
