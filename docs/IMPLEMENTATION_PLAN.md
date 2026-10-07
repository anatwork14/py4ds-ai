# Implementation Plan and Acceptance Tests

**Scope:** implement an auditable comparison of classical image features, pretrained ResNet18 embeddings and end-to-end trained CNNs for the Intel 6-class scene dataset. This document is a **build specification**, not a claim that the scripts below already exist or were run.

## What is in the existing repository today (code inspection 2026-10-08)

- README.md: overall aim and run instructions, but mismatched clone/Colab references to PhatLavar/ML_ASSIGNMENT.
- dataset_metadata.json: six alphabetically sorted labels, 150×150 input, an RGB mean/std vector, claimed 16,928 clean labeled images, split folder names. Raw audit trace and normalization provenance not included.
- modules/dataset_helper.py: zip extraction, destructive nested-folder flattening, seg_pred deletion, per-class 15% validation split by moving files, and metadata generation.
- modules/eda_helper.py: PIL image metadata, perceptual hashes, strict square/150×150 filtering, duplicate drop (prioritizing test then val).
- modules/traditional_helper.py: grayscale contrast equalization, OpenCV HOG, SIFT-BoVW MiniBatchKMeans, StandardScaler, optional PCA, LinearSVC and LogisticRegression. Hardcoded hyperparameters and predictions on test (rather than a validation-first selection protocol).
- modules/deep_learning_helper.py: train/test ImageFolder loaders, stochastic train transform, ResNet18 ImageNet weight load, layer removal, inference feature extraction. No explicit validation loader in this module.
- features/: existing X_train_resnet18.npy, y_train_resnet18.npy, X_test_resnet18.npy, y_test_resnet18.npy; exact data/transform provenance not established, so do not treat them as reproducible until checked.
- notebooks/main.ipynb and report.pdf: exist in main, but their contents and reproducibility must be separately audited.
- requirements.txt: lists dependencies without versions; both correct environment recording and version compatibility testing are required.

## Target project layout (adapt rather than duplicate existing code)

- AGENTS.md: instructions, evaluation contract and quality gates.
- configs/: reproducible baseline/feature/deep configs; each documents seed, dataset location, paths, hyperparameters, device and budget.
- src/data/: read-only import, audit, hashes/duplicate review, class mapping, split manifest, dataset interfaces.
- src/features/: HOG, SIFT/visual-vocabulary, deterministic ResNet18 embeddings, provenance-aware cache.
- src/models/: classical search/train, CNN train/fine-tune, evaluation, Grad-CAM.
- src/common/: path/config handling, random seeding, logging, artifact registry.
- scripts/: prepare_data.py, run_classical.py, run_embeddings.py, run_deep.py, evaluate_final.py, render_report.py.
- tests/: unit/contract and minimal end-to-end tests using generated dummy images or a controlled subset.
- docs/: research references, methodology, result protocol, report and figure notes.
- runs/<run_id>/: generated per-run configs, metrics, logs, predictions, plots, model files, audit hashes; **ignored by Git**.
- data/raw/: original dataset archive/extracted source; read-only, **ignored by Git**. data/manifests/: small CSVs/JSON for split identities (can be versioned if data privacy/license allows).

The tested modules, not notebook cell side effects, must implement each step. The notebook is a readable report of calling those modules with outputs.

## Implementation stages, exact responsibilities and checks

### Stage 0 — Inventory and environment

**Input:** current repository. **Output:** docs/REPO_AUDIT.md, dependency lock/snapshot, CLI entry points and smoke test design.

- Read and summarize every notebook section and prior report; identify genuine existing results with source page/figure/cell and whether they can be reproduced. Don't import previous scores into the new test comparison before confirmation.
- Pin compatible package versions with environment resolution, record cpu/cuda/mps and GPU model if available.
- Require a reproducible source reference (commit SHA).
- Provide optional Kaggle API download instructions, but never embed credentials or fabricate successful downloads.
- Quality gate: clean install succeeds and basic import/CLI --help smoke tests pass in documented environment.

### Stage 1 — Data preparation (highest priority)

**Input:** Kaggle archive or user-supplied extracted local copy. **Output:** audit.csv, split_manifest.csv, exclusions.csv, class_to_idx.json, data_summary.json, sample montage.

- Discover either dataset/seg_train/seg_train/<class>/... and dataset/seg_test/seg_test/<class>/... or flattened equivalents, without moving them.
- Preserve seg_pred as an unlabeled inference pool and exclude from training/evaluation. Validate all six labels.
- Enumerate relative paths; inspect image readability, byte count, pixel mode/channel, resolution histogram, label distributions, exact SHA-256, perceptual hash; log invalids explicitly.
- Convert non-RGB to RGB at loading time when safe; resize rather than delete non-square images by default. Make exclusions configurable and justified.
- Check exact-duplicate and near-duplicate similarities within/across splits; flag potential label conflicts; do not misrepresent pHash identity as verified duplicates.
- Generate a stratified 15% validation subset of original seg_train using seed 42, while using original seg_test solely for final test. Exclusions from cross-split overlaps should be documented without editing original images.
- Persist manifest with columns sample_id, relative_path, source_split, assigned_split, class_name, class_idx, sha256, phash, audit_status, exclusion_reason; class IDs fixed to sorted six-label map.
- Quality gate: no train/val/test overlap among retained SHA-256s and adjudicated near-duplicates, consistent class IDs and reproducible manifest on repeated runs; audit counts reconcile input and excluded rows. Verify every file referenced by the manifest resolves.

### Stage 2 — Classical ML baselines

**Input:** clean manifest. **Output:** HOG/SIFT feature caches, fitted pipelines, validation search reports and per-image predictions.

- Implement HOG with explicit OpenCV compatible window/cell/block choices, documented pixel preprocessing, and output shape assertions.
- Implement SIFT keypoint extraction and descriptors; sample vocabulary fit only from training descriptors; vocabulary size can be selected on val; zero-descriptor images produce valid zero histogram.
- Pipelines must perform StandardScaler and PCA.fit only on training data (within CV fold if cross-validation is chosen). Avoid using validation data to fit KMeans vocab. Use float32 caches when safe and document dtype conversion.
- Mandatory classifiers: LinearSVC and LogisticRegression. Proposed search spaces, adjust based on runtime and validation findings: LinearSVC C in {0.01, 0.1, 1, 10}; LogisticRegression C in {0.01, 0.1, 1, 10}; HOG PCA components in {128, 256, 512} subject to actual feature/sample limits; BoVW vocabulary in {64, 128, 256}. Record feasible/failed configurations, not just winners.
- Selection metric: validation macro-F1; report accuracy and weighted F1 as well. Choose and freeze a single best classical model before final test.
- Important: if using GridSearchCV for an SIFT vocabulary, wrap **the vocabulary fit** inside an estimator/Pipeline per fold; precomputing a vocabulary on data that includes a CV holdout invalidates fold isolation.
- Quality gate: at least one deterministic HOG+SVM and SIFT-BoVW+classifier trial completed, with split-safe fit boundaries and saved training/tuning logs.

### Stage 3 — Frozen deep embeddings (hybrid)

**Input:** identical manifest, weight-defined transform config. **Output:** per-split embedding matrices, row-alignment manifests, hybrid results.

- Use torchvision ResNet18_Weights.IMAGENET1K_V1 (or pin documented equivalent). Read and record weights.transforms() for embedding inference; no stochastic transforms on cached features. Use model.eval and torch.inference_mode.
- Replace final FC with identity or flatten pooled features; check output shape (N, 512). Move batch to device; extract in stable manifest order; join back on unique sample_id.
- Key cache digest includes image SHA/sample order, split manifest digest, model weight ID, transformation parameters, dtype and code revision.
- Tune LogisticRegression/LinearSVC using training features and val results; fit scaler only on training embeddings. Test only after model selection is locked.
- Quality gate: no label/feature misalignment (ID equality check), stable features across repeat CPU inference within tolerances and cache invalidation on transformed/config changes.

### Stage 4 — Train and fine-tune CNN

**Input:** same manifest, trained-config freeze. **Output:** checkpoint, per-epoch logs, validation predictions and Grad-CAM samples.

- Minimum: ResNet18 ImageNet initialization, replace final fully connected classification layer with six outputs, freeze backbone and train head; controlled run unfreezing layer4 with smaller backbone learning rate if computationally feasible.
- Optional pedagogical baseline: small 3-block CNN trained from scratch, with a clearly identified scope and comparable budget.
- Use weight-compatible normalized train augmentation; deterministic val/test; logs show actual parameters, batch size, learning rates, weight decay, epochs and device.
- Correct epoch loop: model.train → batches → zero_grad(set_to_none=True) → forward → cross_entropy → backward → optimizer.step; aggregate weighted epoch loss and accuracy. Evaluation uses model.eval and torch.inference_mode; no update in val. Best checkpoint chosen by val macro-F1 (and deterministic tie rule), optional patience based on val.
- Ensure target.long, images.float32, consistent 3×H×W inputs and same device for model and batch; no NumPy-based gradient break.
- For fine-tuned network only, produce qualitative Grad-CAM overlays from selected class logit/final conv layer, tracking cases chosen by an a priori error-analysis rule.
- Quality gate: CPU smoke training over tiny batch proves loss computed, backward gradients finite, weights update, save/reload works and predictions are reproducible within declared tolerance; full training labeled NOT RUN until executed.

### Stage 5 — Final selection, evaluation and reporting

**Input:** frozen selected configs/checkpoints, original locked test manifest. **Output:** run-scoped final metrics, prediction CSVs, tables, confusion matrices, sample explanation plots, report.

- Before any final test, write selected_models.json with config IDs, validation scores, checkpoints and hashes. No tuning after seeing test.
- Evaluate each **predeclared comparator** on the same held-out set once, no per-model cherry-picked cleaning/split.
- Generate: raw and normalized six-by-six confusion matrices, accuracy, macro/weighted F1, per-class precision/recall/F1/support, train time, measured inference time with stated batch/device and optional confidence intervals.
- Label scores by split and experiment ID. If CSV results have missing fields or some models did not run, render NOT RUN; never fill a table with guesses.
- Discuss strongest/weakest class, likely visual overlaps, efficiency/accuracy trade-offs, domain shift, limitations of this single benchmark, and the cost of feature engineering vs learned representations.
- Test Grad-CAM for valid output shape/finite values; explain illustrative rather than causal guarantees.
- Quality gate: regenerating tables from saved predictions returns the same values as metrics JSON, to a stated precision.

### Stage 6 — Literature and delivery

**Output:** final report in Markdown (and optionally PDF, when toolchain available), source bibliography and linked tables/plots, corrected README with full steps and hardware/runtime transparency.

- Use docs/LITERATURE_REVIEW.md as source-backed foundation; summarize all cited papers faithfully with author/year, method, relevance and limitations. Verify any further publications before citing.
- Required report sections: title/abstract; problem/motivation; dataset/license and EDA; related literature; hypotheses; split/leakage safeguards; handcrafted/hybrid/CNN methods; tuning; full experimental setup; measured results; per-class/error analysis; Grad-CAM caution; discussion; limitations; reproducibility; citations; appendix with commands.
- State which artifacts are new measurements, which are earlier unreproduced history and which are literature benchmarks. Never present previous Kaggle users' metrics as current team's test numbers.
- Include a requirements compliance matrix mapping each instructor criterion to reproducible files and report sections.

## Experiment tracker (file-backed, never hard-coded metrics)

Each run must save:
- metadata.json: unique run ID, timestamp, commit, seed, dataset split hash, class mapping, hardware/device, package versions, initial/pretrained weight source.
- config.json: full resolved preprocessing/model/search configuration and changes from defaults.
- metrics.json: train/val/test split metrics separately with NOT RUN status until available.
- predictions_<split>.csv: sample_id, true_name, pred_name, confidence (when valid), and fold/run identifiers.
- figures/: loss/accuracy curves, confusion matrices, failure gallery, Grad-CAM, dataset plots.
- stdout.log and structured warnings; checkpoint reference and cache manifest.

## Proposed minimal execution CLI (names are specification until implemented)

1. python scripts/prepare_data.py --data-root data/raw --val-fraction 0.15 --seed 42
2. python scripts/run_classical.py --manifest data/manifests/split_manifest.csv --config configs/classical.yaml
3. python scripts/run_embeddings.py --manifest data/manifests/split_manifest.csv --config configs/embeddings.yaml
4. python scripts/run_deep.py --manifest data/manifests/split_manifest.csv --config configs/resnet18.yaml
5. python scripts/evaluate_final.py --selection runs/selected_models.json
6. python scripts/render_report.py --runs-dir runs --output docs/RESULTS.md
7. python -m pytest -q

**These commands are required targets, not verified runnable commands yet.** The coding agent must implement and execute them, document exact output/error and leave an auditable record.

## Test inventory (examples of executable assertions)

- Manifest: stable across runs, zero duplicate sample_id, 6 classes, labels match directories, train/val/test disjoint.
- Data audit: RGB conversion, non-square resized correctly, corrupt files flagged, no destructive operations, seg_pred untouched.
- Leakage: training fitted statistics invariant if test values change; val never passed to scaler/PCA/SIFT vocabulary fit.
- HOG: constant output length for all valid sample sizes; no NaN. SIFT: zero keypoints handled, histogram length equals vocab size.
- Deep embeddings: ordered stable sample IDs, shape N×512, no random val transform, no unexpected dtype/device errors.
- Training: small batch runs forward/backward, optimizer changes trainable weights but not frozen weights, state reloaded.
- Evaluation: synthetic known confusion matrix yields checked macro/weighted-F1, predicted IDs map to exact fixed label names.
- Comparison: best model chosen by validation not test; test metric read only after selection manifest is persisted.
- Reporting: tables/figures generated from metrics/predictions, not manual numeric text.

## Explicit completion criteria

The project is **not complete** just because scripts exist. It is complete only when independent reviewer can (a) acquire the dataset with allowed credentials, (b) regenerate the manifest without modifying raw files, (c) run classical + frozen ResNet + at least one trained ResNet experiment, (d) reproduce metric tables from prediction CSVs, (e) trace every scientific reference to a real source, (f) run smoke/unit tests, and (g) identify every unrun expensive experiment without fabricated numbers.

If time or GPU resources are limited, produce an honest subset with verified outputs; list remaining unrun work and minimize complexity rather than faking completion.
