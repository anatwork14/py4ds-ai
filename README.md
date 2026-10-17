# Intel Natural Scene Classification — CO3117 Machine Learning

Course assignment · Semester 252 (AY 2025–2026) · Python / scikit-learn / PyTorch

A reproducible comparison of **traditional machine learning**, **pretrained image features**, and **deep learning** on the six-class Intel Natural Scene Classification dataset.

**Classes:** `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`.

| Selected approach | Validation macro-F1 | Held-out accuracy | Held-out macro-F1 |
|---|---:|---:|---:|
| ResNet18, layer4 fine-tuned | 0.929137 | 0.930000 | 0.931004 |

These are **historical saved experiment results**. Only the selected model was scored once on the sealed test set (3,000 images). See the [final report](docs/FINAL_REPORT.md) for the complete comparison, per-class metrics, and limitations.

## Project overview

| Area | Content |
|---|---|
| Dataset and split | Train 11,868 · validation 2,100 · held-out test 3,000 |
| Classical ML | HOG and SIFT–Bag of Visual Words with linear classifiers |
| Transfer learning | Frozen ImageNet ResNet18 embeddings + linear classifier |
| Deep learning | ResNet18 head-only training and layer4 fine-tuning |
| Model selection | Validation macro-F1; one final held-out evaluation |
| Evaluation | Accuracy, macro/weighted F1, confusion matrix, classwise errors and Grad-CAM |

## Repository structure

```text
py4ds-ai/
├── configs/               # Frozen experiment selection and pHash review ledger
├── data/                  # Local datasets and generated manifests (ignored by Git)
│   └── README.md
├── docs/                  # Academic report, methodology, audits, references
│   └── PROJECT_STRUCTURE.md
├── legacy/                # Archived original notebook, PDF, helpers and arrays
├── notebooks/             # Workspace for new exploratory notebooks
├── scripts/               # Thin CLI entry points and report verification
├── src/py4ds_ai/
│   ├── data/              # Downloading, read-only auditing, splitting
│   ├── features/          # HOG and SIFT–BoVW
│   └── models/            # Classical baselines, ResNet, selection, metrics
├── tests/                 # Synthetic/unit/contract tests
├── pyproject.toml         # Python package, CLI definitions, test/lint setup
└── uv.lock               # Pinned dependency resolution
```

See [Project Structure & Reproducibility](docs/PROJECT_STRUCTURE.md) for artifact locations and the distinction between active and historical files.

## Getting started — Windows PowerShell

Install [Git](https://git-scm.com/), [uv](https://docs.astral.sh/uv/), and a compatible 64-bit Python 3.12 interpreter. Then:

```powershell
git clone https://github.com/anatwork14/py4ds-ai.git
cd py4ds-ai
uv python install 3.12
uv sync --locked --extra dev --python 3.12
uv run --locked --extra dev python -m pytest -q
uv run --locked --extra dev ruff check src tests scripts
```

The lock pins CUDA 12.6 PyTorch wheels and includes Windows x86-64 dependency resolution. **Windows installation and GPU execution have not been independently verified on the final revision.** Most synthetic/unit tests do not need model training or a GPU.

Linux users can run the same `uv` verification commands in a shell with Python 3.12; platform-specific package availability still applies.

## Dataset and experiments

The [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) is **not** in the repository. Follow [data/README.md](data/README.md) and the [dataset audit](docs/PHASE1_DATA_AUDIT.md) for source layout and the historical reviewed split. The original images must remain unchanged.

The tracked [pHash review ledger](configs/phash-review-seed-42.csv) and [frozen selection metadata](configs/final-selection-seed-42-phash-reviewed.json) identify the original experiment. They are *provenance records*, not an invitation to change splits, rerun tuning, or choose a different model.

**Do not rerun the held-out test evaluator** (`py4ds-evaluate-final` or `scripts/evaluate_final.py`) for this frozen project. Current experiment artifacts and checkpoints are stored locally, outside Git.

To **audit saved predictions** or rerender tables without model inference, when the original local artifacts are present:

```powershell
uv run --locked --extra dev python scripts/verify_saved_final_artifacts.py --expected-metrics-sha256 4df9443c7ad28c9ddd36cc461d38f5f7ed1a0b43b819d2ea0e400dedf9b4967e --expected-predictions-sha256 85d68f7fac4351716964ec0db773126805ac530c269ed37c607af8cae0486dde
uv run --locked --extra dev python scripts/render_report.py
```

Without the existing local `runs/` and `data/manifests/` evidence, these commands cannot reproduce the original scores from Git alone. They are not commands to generate a fresh experiment.

## Documentation

- [Final evaluation report](docs/FINAL_REPORT.md) — methodology, comparisons, metrics, citations and limitations.
- [Literature review](docs/LITERATURE_REVIEW.md) — HOG, SIFT/BoVW, residual networks, transfer learning, Grad-CAM.
- [Experiment log](docs/EXPERIMENT_STATUS.md) — historical validations, selected checkpoint and artifact identifiers.
- [Dataset preparation audit](docs/PHASE1_DATA_AUDIT.md) — immutable split and leakage adjudication.
- [Repository layout and usage](docs/PROJECT_STRUCTURE.md) — current code, local artifacts and archived files.
- [Original implementation plan](docs/IMPLEMENTATION_PLAN.md) — historical design and acceptance criteria.

## Course information

**CO3117 — Machine Learning**, Semester 252, AY 2025–2026  
**Instructor:** Dr. Trương Vĩnh Lân

| Member | Student ID |
|---|---|
| Nguyễn Văn An (leader) | 2352013 |
| Huỳnh Vương Khang | 2350011 |
| Nguyễn Tấn Phát | 2352889 |
| Lê Đặng Khánh Quỳnh | 2353037 |

## Limitations and archival notes

One split and seed, four potential within-test near-duplicate pairs, historical dirty training-worktree provenance, non-comparable compute timings, and unverified dataset redistribution rights limit interpretation. The project does **not** represent a new benchmark or proof of reproducible training from a clean commit.

The old submission PDF, notebook, helper modules and feature arrays were relocated **byte-for-byte** into [legacy/](legacy/README.md); their scores are not the current evaluation. The maintained code is in `src/py4ds_ai/`. Do not run the original legacy preprocessing over raw images because it may move or delete files.

No production deployment, agent orchestration, CI services, or new model training is required for this school assignment.
