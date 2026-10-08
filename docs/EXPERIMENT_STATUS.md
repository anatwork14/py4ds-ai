# Experiment status and reproducibility log

**Current comparison protocol:** `data/manifests/seed-42-phash-reviewed/split_manifest.csv`, SHA-256 `27c17f497e89a8f3b72975b45fcd1e44038e63edddbd4e721437a9a3500c982e`. It contains 11,868 train, 2,100 validation, and 3,000 test rows in fixed alphabetical class order. The 52-pair review ledger excluded 43 train/validation rows, preserved every test row, and left zero active cross-split pHash candidates; four test–test near-duplicate candidates remain. See [`PHASE1_DATA_AUDIT.md`](PHASE1_DATA_AUDIT.md).

**Important:** The validation scores in the historical table below were produced with the earlier, pre-review manifest (`ff407ae...`) and are superseded. They must not be used as the clean-protocol comparison or to select the final model. All five planned clean-manifest runs have completed. The selected model was evaluated on the held-out test once after selection and code were committed. Test images were viewed before that only in targeted pHash leakage adjudication, not for exploratory analysis or model selection.

## Reviewed-manifest validation runs

| Experiment | Selected candidate/checkpoint | Validation macro-F1 | Saved result |
|---|---|---:|---|
| Majority-class baseline (validation reference only) | training-majority label `mountain` | 0.050619 | `runs/seed-42-phash-reviewed/classical/hog/metrics.json` |
| HOG + linear classifiers | `hog-logistic_regression-C0.01` | 0.676893 | `runs/seed-42-phash-reviewed/classical/hog/metrics.json` |
| SIFT-BoVW + linear classifiers | `sift-bovw-128-logistic_regression-C1` | 0.592861 | `runs/seed-42-phash-reviewed/classical/sift-bovw/metrics.json` |
| Frozen ResNet18 embeddings + linear classifiers | `resnet18-hybrid-linear_svc-C0.1` | 0.903315 | `runs/seed-42-phash-reviewed/resnet18/imagenet-v1/metrics.json` |
| Pretrained ResNet18, frozen head | best epoch 4/5 | 0.908711 | `runs/seed-42-phash-reviewed/cnn18/head-imagenet-v1/metrics.json` |
| Pretrained ResNet18, layer4 fine-tuned | best epoch 5/5 | 0.929137 | `runs/seed-42-phash-reviewed/cnn18/layer4-imagenet-v1/metrics.json` |

All five runs used the same reviewed manifest SHA-256 `27c17f497e89a8f3b72975b45fcd1e44038e63edddbd4e721437a9a3500c982e`; the candidate run artifacts themselves report `test_status: NOT RUN`. Only the frozen winner was subsequently evaluated on test. The other four candidates have no test predictions or metrics.

The majority-class reference is recorded inside the HOG metrics artifact, uses the training-majority class, and is not a sixth selection candidate. Its test score was not computed.

## Frozen validation-only selection

`configs/final-selection-seed-42-phash-reviewed.json` records five candidates and selects the layer4-fine-tuned ResNet18 CNN by the declared highest-validation-macro-F1 rule. The selected checkpoint is `runs/seed-42-phash-reviewed/cnn18/layer4-imagenet-v1/best_checkpoint.pt`, SHA-256 `84fd9791276833d6dd8a5a6d1e17c1d89cc41eab8922002774c309a6ffbfd019`; validation accuracy is 0.928095 and macro-F1 is 0.929137. The selection and implementation were committed before final test access.

## Final held-out test evaluation (SCORED ONCE)

- Selected model: `resnet18-cnn-layer4`; test count: 3,000; accuracy: 0.930000; macro-F1: 0.931004; weighted-F1: 0.929743; error rows: 210.
- Evaluator code commit: `5303a1078a54119c51445fd0e1e12feb4c1167e4`; device: NVIDIA GeForce GTX 1080 Ti (`cuda:0`); evaluation time: 2.088 seconds.
- Guard record at `runs/.test_evaluation_locks/27c17f497e89a8f3b72975b45fcd1e44038e63edddbd4e721437a9a3500c982e.json` is `COMPLETED`; the metrics artifact records `test_evaluations: 1` and `test_status: SCORED ONCE`.
- Saved predictions, error rows, metrics, and confusion matrix are local under `runs/seed-42-phash-reviewed/final-evaluation/`. Metrics SHA-256: `4df9443c7ad28c9ddd36cc461d38f5f7ed1a0b43b819d2ea0e400dedf9b4967e`.
- Full class-level results, interpretation, and limitations are in [`FINAL_REPORT.md`](FINAL_REPORT.md).

## Superseded protocol (pre-review manifest)

- Manifest: `data/manifests/seed-42/split_manifest.csv`
- Manifest SHA-256: `ff407aeeb1f7c03561740ca876d82bf8c6b3febd261376a72f91be850235bdb9`
- Train/validation rows: 11,908 / 2,103; fixed class order: `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`.
- All supervised features and classifiers use only train and validation rows from this manifest. Preprocessing and feature fitting is train-only; model/parameter selection uses validation macro-F1. Every run reports `test_status: NOT RUN` and has no `predictions_test.csv`.

## Preliminary validation runs (superseded)

| Feature/model | Search performed | Selected validation experiment | Accuracy | Macro-F1 | Saved result |
|---|---|---|---:|---:|---|
| Majority baseline | Most frequent training class | `majority-class-mountain` | 0.178792 | 0.050558 | `runs/classical/hog-seed-42/metrics.json` |
| HOG + linear classifiers | HOG 128×128, 1,764 features; LinearSVC and LogisticRegression; C ∈ {0.01, 0.1, 1, 10} | LogisticRegression, C=0.01 | 0.675701 | 0.677715 | `runs/classical/hog-seed-42/metrics.json` |
| SIFT-BoVW + linear classifiers | Vocabulary sizes {64,128}; 30,000-descriptor training-only cap; LinearSVC and LogisticRegression; C ∈ {0.1,1} | 128 words + LinearSVC, C=0.1 | 0.579648 | 0.575782 | `runs/classical/sift-bovw-seed-42/metrics.json` |
| Frozen pretrained ResNet18 embeddings + linear classifiers | `IMAGENET1K_V1`, torchvision-defined transform, 512-d embeddings; LinearSVC and LogisticRegression; C ∈ {0.1,1} | LinearSVC, C=0.1 | 0.903471 | 0.904771 | `runs/resnet18/imagenet-v1-seed-42/metrics.json` |

These are model-selection results on validation, not unbiased final performance estimates. In particular, the best validation row is selected from several candidates and must not be presented as a test score.

## Historical reproduction commands (superseded)

```bash
uv sync --locked --extra dev

uv run --locked --extra dev py4ds-classical \
  --manifest data/manifests/seed-42/split_manifest.csv \
  --output-dir runs/classical/hog-seed-42 --seed 42

uv run --locked --extra dev py4ds-bovw \
  --manifest data/manifests/seed-42/split_manifest.csv \
  --output-dir runs/classical/sift-bovw-seed-42 \
  --vocab-sizes 64 128 --max-descriptors 30000 \
  --c-values 0.1 1.0 --seed 42

uv run --locked --extra dev py4ds-resnet \
  --manifest data/manifests/seed-42/split_manifest.csv \
  --output-dir runs/resnet18/imagenet-v1-seed-42 \
  --pretrained --device cuda:0 --batch-size 64 --num-workers 4 \
  --c-values 0.1 1.0 --seed 42
```

The commands refuse to overwrite an existing run directory. Choose a new `--output-dir` for repeats.

## Runtime and provenance

- Python 3.12.14; locked environment via `uv.lock`.
- Classical HOG, SIFT-BoVW, and scikit-learn classifier searches used CPU. ResNet18 embedding extraction and both CNN training runs used the local NVIDIA GeForce GTX 1080 Ti (`cuda:0`); the ResNet18 linear-classifier search ran on CPU. Device metadata, Python/package versions, seed, manifest hash, and class mapping are saved with each run.
- Before full extraction, CUDA initialization, allocation, synchronization, and an actual ResNet18 CUDA forward pass succeeded.
- The ResNet run used the explicit `ResNet18_Weights.IMAGENET1K_V1`; its configuration records the weight URL, transform, split hash, seed, batch size, worker count, device, and classifier search.
- Each saved run includes resolved configuration, metadata, feature matrices, validation predictions, metrics, selected model/configuration, fitted models, and validation plots. HOG and SIFT cache hashes and sample IDs are saved alongside features.

## Important reproducibility caveat

The run metadata records source commit `11334744c8634bc5a593e69fa271dc69bb626fd4` and `git_worktree_dirty: true`. The experiment code was not committed when those runs were launched. The implementation and final-evaluation guard are now committed and reviewed (up to `5303a1078a54119c51445fd0e1e12feb4c1167e4`), but the dirty-worktree flag still means the commit alone does not identify the exact historical training source tree. The saved configurations, metrics, checkpoint hashes, and predictions are the direct evidence for those runs.

## Remaining work and caveats

- Push the reviewed commits to the existing implementation branch/PR and verify the remote PR and CI status.
- Dataset-specific reuse terms remain unverified. Keep the archive, source images, and image-bearing figures local unless rights are confirmed.
- Four retained test–test near-duplicate candidates limit the independence assumption; do not remove them or retune against test.
- Training run metadata records commit `11334744c8634bc5a593e69fa271dc69bb626fd4` with a dirty worktree. The saved run artifacts are the direct evidence; that commit alone does not identify the exact training source tree.
