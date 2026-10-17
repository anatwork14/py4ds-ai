# Phase 0 — Repository, notebook, report, and environment audit

## Status and scope

This is an audit record, not a new experiment report. The existing repository and draft PR #1 were inspected locally. No dataset was downloaded, no original image was changed, and no model was trained or evaluated during this phase.

Inspected: the main-branch README, metadata, requirements, `.gitignore`, four modules, complete notebook source and saved outputs, feature-array headers/checksums, and all 22 PDF pages (text extraction plus visual inspection of the figures). The three documentation files added by draft PR #1 were reviewed as a diff. The source PR remains a documentation-only draft; a non-blocking review comment was posted asking that exploratory sample montages be train-only because the existing notebook samples from all splits. The review readback showed state `COMMENTED`.[10]

## Repository and PR state

- Local project: `py4ds-ai`; base `main` commit `7cffec2` (`update readme`). The draft PR head is `docs/agent-research-and-evaluation-protocol` at `ec12ee1`.
- PR #1 is open and draft, targets `main`, and adds only `AGENTS.md`, `docs/IMPLEMENTATION_PLAN.md`, and `docs/LITERATURE_REVIEW.md` (333 insertions). No checks or earlier reviews were reported when inspected.[10]
- The implementation/audit work branch is `implementation/phase0-audit`, based on the PR head so its instructions remain available. The follow-up is therefore stacked on PR #1 until that draft is merged.
- The clone's `origin` points to `anatwork14/py4ds-ai`; `upstream` points to `PhatLavar/ML_ASSIGNMENT`. The README's clone, repository, and Colab links still point to `PhatLavar/ML_ASSIGNMENT` / the standalone Drive notebook.
- Date note: the PR documents use `2026-10-08` for code inspection/literature checking. The applicable task date is `2026-10-07`, while the machine's local clock read `2026-10-08 +07:00` and GitHub's PR `updatedAt` was `2026-10-07T23:07:16Z`. This is consistent if the documents use local time; record the timezone explicitly to avoid an apparent one-day discrepancy.[10]
- The checked-in tree has 17 tracked files on this branch: 14 from `main` plus the three PR documents. There is no test suite, package configuration, or lock file. `requirements.txt` is unpinned and does not list `pytest`.

## Existing artifacts and their evidentiary limits

- `features/` contains four tracked NumPy arrays. Header and SHA-256 inventory:

  | File | Shape / dtype | Bytes | SHA-256 |
  |---|---:|---:|---|
  | `X_train_resnet18.npy` | `(11839, 512)` / float32 | 24,246,400 | `9278ffd6a0b4856865dddab6a780c9e0f2cc72be1286d56c0043a79b2a0fcc05` |
  | `y_train_resnet18.npy` | `(11839,)` / int64 | 94,840 | `09024ae879bb86e729bc48ff590e054ec3089911ad1a844fe30d0db5d3bee209` |
  | `X_test_resnet18.npy` | `(2993, 512)` / float32 | 6,129,792 | `23a9ee64be77642018f57c5b92ec5eea5b37319ee4a95bde7eceef2ede5a2116` |
  | `y_test_resnet18.npy` | `(2993,)` / int64 | 24,072 | `ca92eeab3720c9e72cf0050d8434067e513e85b9e6ddbd37fda9248ad3397419` |

  No row-to-image IDs, transform/weight manifest, split hash, extraction log, model checkpoint, or classifier artifact is present. Keep these arrays as legacy artifacts for now; do not treat them as reproducible Phase 3 embeddings or use them to select a new model.
- The notebook has 24 cells (20 code cells) and six saved PNG outputs. Its outputs are historical Colab outputs, not a run performed in this environment. They contain no complete environment lock or run manifest.
- `report.pdf` is 22 PDF pages (21 printed pages plus cover). Its text was extracted and its six figures were visually inspected. The saved figures and notebook matrices are readable and their displayed class order is `buildings, forest, glacier, mountain, sea, street`.
- The notebook's embedded historical test results, also summarized in the report, are **not reproduced here** and are not accepted as current experiment results:

  | Historical notebook output | Accuracy | Weighted F1 |
  |---|---:|---:|
  | HOG + SVM | 0.685600 | 0.683679 |
  | HOG + Logistic Regression | 0.682259 | 0.681740 |
  | SIFT-BoVW + SVM | 0.595723 | 0.592727 |
  | SIFT-BoVW + Logistic Regression | 0.605413 | 0.603625 |
  | ResNet18 features + SVM | 0.8814 | 0.8809 |
  | ResNet18 features + Logistic Regression | 0.8720 | 0.8716 |

  Source locations: notebook cells 15 and 22 (zero-based); PDF printed pages 15–17. There are no saved per-image predictions, fitted estimators, training logs, or validation-selected configuration tying these numbers to a reproducible run. They must remain labeled historical/unverified.

## Confirmed methodological and engineering findings

### Data acquisition and mutation

- `modules/dataset_helper.py:10–11` selects the first ZIP returned by a filesystem glob without requiring an explicit archive when multiple ZIPs exist. Lines 37–38 call `extractall()` into the dataset directory without a path-safety check. Lines 49–66 move nested folders and delete `seg_pred`; lines 68–86 move sampled training files into `seg_val`.
- `notebooks/main.ipynb` cell 3 downloads a shared Google Drive file with `gdown`, then calls the mutating extractor. The local project has no archive or extracted image directories, and no download provenance/hash is saved.
- Notebook cell 9 deletes every file not present in `clean_df` (`os.remove`). The saved output says 106 files were deleted. This is irrecoverable cleanup of the extracted source, not a nondestructive manifest operation.
- `modules/eda_helper.py:35–36` catches every image-processing exception and returns null fields, hiding the cause. `clean_dataset()` at lines 75 and 93–95 keeps only 150×150 images. Lines 99–117 group by exact equality of a perceptual hash and unconditionally drop copies, preferring `test`, then `val`, then `train`; a perceptual-hash collision is not proof of byte-identical images.

### Split, leakage, and historical counts

- The notebook's saved EDA output reports 17,034 input rows, 16,979 images at 150×150, 55 non-square images, 51 redundant pHash copies, and 16,928 remaining rows. The notebook computes the pHash over all splits and then deletes removed source files. The report gives 2,993 test rows; the original split plot shows 3,000 test images, so seven original test examples are absent from the cleaned test set. This changes the benchmark that the report calls held out.
- From the saved clean total and train/test array counts, the validation count is 2,096 (derived, not independently recounted from raw images). The six test supports in the report sum to 2,993.
- The notebook uses the test split for final metrics across all model variants, while the report explicitly says validation was available but “not used in this project’s final evaluation.” No validation-based hyperparameter selection is recorded; selecting a winner from the test comparisons would contaminate the final estimate.
- The sample montage in notebook cell 5 is sampled from `df`, which contains train, validation, and test paths. It has no split annotation. Keep future human-facing examples and model-development plots train-only (or train/validation before lock); use test files only for the final predeclared evaluation.
- Figure 2 is generated from the pre-clean `df` and displays 3,000 test examples; the report does not clearly mark that split-count figure as pre-clean while later reporting cleaned counts. Rebuild the plot from the locked manifest and label its audit stage.

### Preprocessing and model evaluation

- The notebook's normalization code (cell 10) uses `clean_df` across all splits, including test. It averages per-image means and per-image standard deviations. The per-image means are compatible with a global mean only when each image has the same pixel count; the mean of per-image standard deviations is not the global pixel standard deviation. These values are not suitable as a train-only fitted transform, and the pretrained ResNet should use the selected weights' documented transform.
- The stored metadata values are precise, but the report's green-channel standard deviation is printed as 0.2347 in Table 3 and 0.2346 in the ResNet normalization subsection. Treat that as an unresolved reporting inconsistency; regenerate values from the chosen train-only implementation.
- `modules/traditional_helper.py` has reusable HOG and SIFT-BoVW components. Its label encoder and scaler are fit on train, and its SIFT vocabulary is sampled from train; those are useful fit-boundary behaviors to preserve. However, `run_traditional_pipeline()` (lines 143–172) evaluates only train/test, hardcodes `C=0.01`, does not tune on validation, and reports weighted F1 as its scalar score. HOG PCA is train-fitted, but the fitted pipeline is not persisted.
- `modules/deep_learning_helper.py` provides train/test loaders only (lines 44–68); it does not create a validation loader or retain sample IDs. Train transforms are stochastic (lines 26–34), whereas test transforms are deterministic. The model uses `ResNet18_Weights.DEFAULT` (lines 72–73) without explicitly selecting/persisting a weight ID or using the weight object's inference transform; it instead uses dataset metadata normalization and 150×150 resizing. Feature extraction is under `torch.no_grad()` but parameters are not explicitly marked `requires_grad=False`. The saved arrays cannot be aligned to source rows or regenerated from a cache fingerprint.
- The old pipeline is frozen feature extraction plus sklearn classifiers; it does not implement CNN-head training, fine-tuning, checkpoint selection, Grad-CAM, or a held-out final evaluation protocol.

### Report and literature review

- The PDF's results are consistent with the saved notebook classification reports and confusion matrices, but neither source provides a reproducible run bundle. The report has no bibliography/references section; its code-availability URL points to `PhatLavar/ML_ASSIGNMENT`. It makes an unsupported/uncounted statement that a small number of empty or corrupt files were excluded; the saved cleaning output does not quantify such rows.
- The PDF says “He et al. (2015)” while the draft literature review cites the CVPR 2016 version. This may reflect the 2015 preprint, but the final report should identify which version is cited and include a complete reference.[4]
- Bibliographic identifiers and linked primary publication pages were checked at title/venue/year level. This is not a systematic literature review or an independent reproduction of the papers' experiments:
  - Dalal & Triggs, HOG / CVPR 2005: DOI record matches the listed work.[1]
  - Lowe, SIFT / IJCV 2004: DOI record matches the listed work.[2]
  - Csurka et al., visual categorization with bags of keypoints / 2004: linked paper matches the listed work.[3]
  - He et al., residual learning / CVPR 2016: conference paper matches the listed work.[4]
  - Kornblith et al., transfer learning / CVPR 2019: conference paper matches the listed work.[5]
  - Selvaraju et al., Grad-CAM / ICCV 2017: conference paper matches the listed work.[6]
  - Optional references also match their linked primary venues: MobileNetV2 / CVPR 2018,[7] EfficientNet / PMLR 2019,[8] and Bag of Tricks / CVPR 2019.[9]
- The old report's results are not included in the new locked experiment comparison. The new report should generate metrics from saved predictions and cite the final, source-verified bibliography.

## Data and runtime availability

- No `dataset/seg_train`, `dataset/seg_test`, `dataset/seg_pred`, `data/raw`, or local ZIP was present. The `kaggle` CLI, Kaggle config, and `KAGGLE_API_TOKEN` were absent. Dataset license/terms and exact source release therefore remain unverified; no image-level audit or locked real-data manifest can yet be generated.
- Environment observed: Python 3.14.7, Linux x86_64, 8 CPU cores, 62 GiB RAM, NVIDIA GTX 1080 Ti with 11,264 MiB VRAM, and 27 GiB free disk. The base environment does not have NumPy, pandas, scikit-learn, PyTorch/torchvision, OpenCV, Pillow, ImageHash, matplotlib, seaborn, or pytest installed. No project test suite was present. These checks do not establish that the GPU training stack is compatible or that any experiment can run.

## Phase 0 commands and verification record

Commands/checks used:

```bash
gh pr view 1 --repo anatwork14/py4ds-ai --json number,title,state,isDraft,headRefName,baseRefName,body,updatedAt,headRefOid,baseRefOid,reviewDecision,statusCheckRollup,reviews
git diff --stat main...review/pr-1
git diff --check main...review/pr-1
uv run --with pypdf --with pdfplumber --with pypdfium2 python /mnt/ssd/genos2/.hermes/skills/productivity/pdf/scripts/pdf_read.py report.pdf --meta
uv run --with pypdf --with pdfplumber --with pypdfium2 python /mnt/ssd/genos2/.hermes/skills/productivity/pdf/scripts/pdf_read.py report.pdf --text
uv run --with pypdf --with pdfplumber --with pypdfium2 python /mnt/ssd/genos2/.hermes/skills/productivity/pdf/scripts/pdf_page_image.py report.pdf --pages 2-3 --dpi 150 --out-dir /mnt/ssd/genos2/.hermes/cache/scratch/py4ds-ai-report/pages
uv run --with pypdf --with pdfplumber --with pypdfium2 python /mnt/ssd/genos2/.hermes/skills/productivity/pdf/scripts/pdf_page_image.py report.pdf --pages 6-8,19 --dpi 120 --out-dir /mnt/ssd/genos2/.hermes/cache/scratch/py4ds-ai-report/figures
```

Notebook JSON and NPY headers were inspected with temporary Python standard-library probes (no project dependencies were installed); the saved audit conclusions and SHA-256 values are recorded above. PDF extraction/rendered pages and notebook PNGs were kept in the session scratch directory, not added to the repository.

- Passed: PR diff whitespace check; Python AST syntax parsing for all four existing modules; PDF text/metadata extraction; NPY header and SHA-256 inventory; notebook source/output parsing; visual review of all report and notebook figures.
- Not run: project unit tests (none exist), data audit on real images (data absent), model smoke/full experiments, GPU compatibility tests, and final test evaluation.
- No model, data, code, or new experimental metric was produced by this phase. The only Phase 0 deliverable is this audit document and the verified PR review comment.

## Phase 1 entry gate

Implement read-only discovery and auditing over an explicit user-supplied local dataset or an authenticated Kaggle CLI download. Preserve the original `seg_test` contents; save a deterministic manifest and exclusions instead of deleting/moving images. Use byte SHA-256 for exact duplicates, pHash only to flag review candidates, fit any preprocessing from train only, and keep the test split sealed. Until a real dataset path or user-managed Kaggle credentials are available, Phase 1 can be exercised only with synthetic fixtures and must not be reported as a completed real-data audit.

## Sources

[1] https://doi.org/10.1109/CVPR.2005.177 — Dalal and Triggs (2005), HOG for Human Detection
[2] https://doi.org/10.1023/B:VISI.0000029664.99615.94 — Lowe (2004), SIFT
[3] https://www.cs.princeton.edu/courses/archive/fall09/cos429/papers/csurka-eccv-04.pdf — Csurka et al. (2004), Visual Categorization with Bags of Keypoints
[4] https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html — He et al. (2016), Deep Residual Learning for Image Recognition
[5] https://openaccess.thecvf.com/content_CVPR_2019/html/Kornblith_Do_Better_ImageNet_Models_Transfer_Better_CVPR_2019_paper.html — Kornblith et al. (2019), Do Better ImageNet Models Transfer Better?
[6] https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html — Selvaraju et al. (2017), Grad-CAM
[7] https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html — Sandler et al. (2018), MobileNetV2
[8] https://proceedings.mlr.press/v97/tan19a.html — Tan and Le (2019), EfficientNet
[9] https://openaccess.thecvf.com/content_CVPR_2019/html/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019_paper.html — He et al. (2019), Bag of Tricks for Image Classification
[10] https://github.com/anatwork14/py4ds-ai/pull/1 — PR #1 — Research protocol and coding-agent instructions
