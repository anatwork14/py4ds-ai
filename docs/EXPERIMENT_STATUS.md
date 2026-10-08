# Experiment status and reproducibility log

**Current comparison protocol:** `data/manifests/seed-42-phash-reviewed/split_manifest.csv`, SHA-256 `27c17f497e89a8f3b72975b45fcd1e44038e63edddbd4e721437a9a3500c982e`. It contains 11,868 train, 2,100 validation, and 3,000 test rows in fixed alphabetical class order. The 52-pair review ledger excluded 43 train/validation rows, preserved every test row, and left zero active cross-split pHash candidates; four test–test near-duplicate candidates remain. See [`PHASE1_DATA_AUDIT.md`](PHASE1_DATA_AUDIT.md).

**Important:** The validation scores in the historical table below were produced with the earlier, pre-review manifest (`ff407ae...`) and are superseded. They must not be used as the clean-protocol comparison or to select the final model. All five planned clean-manifest runs have now completed. Test images were viewed only in targeted pHash leakage adjudication; they have not been used for feature/model selection or scored.

## Reviewed-manifest validation runs (test NOT RUN)

| Experiment | Selected candidate/checkpoint | Validation macro-F1 | Saved result |
|---|---|---:|---|
| HOG + linear classifiers | `hog-logistic_regression-C0.01` | 0.676893 | `runs/seed-42-phash-reviewed/classical/hog/metrics.json` |
| SIFT-BoVW + linear classifiers | `sift-bovw-128-logistic_regression-C1` | 0.592861 | `runs/seed-42-phash-reviewed/classical/sift-bovw/metrics.json` |
| Frozen ResNet18 embeddings + linear classifiers | `resnet18-hybrid-linear_svc-C0.1` | 0.903315 | `runs/seed-42-phash-reviewed/resnet18/imagenet-v1/metrics.json` |
| Pretrained ResNet18, frozen head | best epoch 4/5 | 0.908711 | `runs/seed-42-phash-reviewed/cnn18/head-imagenet-v1/metrics.json` |
| Pretrained ResNet18, layer4 fine-tuned | best epoch 5/5 | 0.929137 | `runs/seed-42-phash-reviewed/cnn18/layer4-imagenet-v1/metrics.json` |

All five runs used the same reviewed manifest SHA-256 `27c17f497e89a8f3b72975b45fcd1e44038e63edddbd4e721437a9a3500c982e`; each reports `test_status: NOT RUN`. These are validation-selection results, not estimates of held-out test performance. No `predictions_test.csv` exists under the clean-run directory.

## Frozen validation-only selection (test NOT RUN)

`configs/final-selection-seed-42-phash-reviewed.json` records five candidates and selects the layer4-fine-tuned ResNet18 CNN by the declared highest-validation-macro-F1 rule. The selected checkpoint is `runs/seed-42-phash-reviewed/cnn18/layer4-imagenet-v1/best_checkpoint.pt`, SHA-256 `84fd9791276833d6dd8a5a6d1e17c1d89cc41eab8922002774c309a6ffbfd019`; validation accuracy is 0.928095 and macro-F1 is 0.929137. The selection record and code are not committed yet. The test evaluator has not been invoked.

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

The run metadata records source commit `11334744c8634bc5a593e69fa271dc69bb626fd4` and `git_worktree_dirty: true`. The experiment code was not committed when these runs were launched. The saved files and settings are real, but the recorded Git commit alone does not identify the exact run code until the Phase 1–3 changes are committed and reviewed. Keep this caveat in any report.

## Remaining work

- Independently review the implementation and frozen selection file; verify the five-candidate inventory, manifest hash, and selected checkpoint digest, then commit the code and selection record.
- Run the held-out test evaluator exactly once from that committed state. Preserve all 3,000 test rows and report the four within-test near-duplicate pairs as an independence limitation.
- Generate test metrics, per-class/error analysis, comparison plots, and the source-verified final report from saved predictions. Keep dataset images and image-bearing figures local until reuse rights are confirmed.
- Push reviewable phase-based PRs, check each PR's actual CI state, and report any remaining license or reproducibility caveats.
