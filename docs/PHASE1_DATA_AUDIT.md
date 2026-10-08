# Phase 1 — Dataset acquisition and audit

Status: **real dataset acquired, extracted, and audited; manifest locked.** This document covers data preparation; completed model runs and the one-time final evaluation are recorded in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md) and [`FINAL_REPORT.md`](FINAL_REPORT.md).

## Provenance

- Dataset: Intel Image Classification, Kaggle identifier `puneet6060/intel-image-classification`.
- Source: [Kaggle dataset page](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).
- Acquisition and extraction date: 2026-10-08 (UTC+07:00 local time; exact download time was not captured).
- The requested `kaggle` executable was not installed. The same Kaggle CLI download was run with `uvx --from kaggle kaggle datasets download puneet6060/intel-image-classification` from `data/raw/`.
- The CLI reported `License(s): copyright-authors`. The dataset-specific reuse terms and underlying rights holders have **not** been independently verified; do not infer permission to redistribute or publish the images from this label.
- Archive: `data/raw/intel-image-classification.zip` (363,152,213 bytes); SHA-256 `3921405607c3e9d67a3459b6c629894c3a723aa1647f9310b6793da306e89337`.
- ZIP verification passed (`ZipFile.testzip()` returned `ok`) and reported 24,335 entries. The archive remains intact; safe extraction wrote to `data/raw/extracted/`. Both raw and extracted data are ignored by Git.

## License and redistribution status

- The Kaggle CLI reported `copyright-authors`. Kaggle's official CLI metadata documentation defines that label as “Data files © Original Authors”; the label does not identify the underlying rights holders or state a reuse permission. The dataset-specific reuse terms and image provenance remain unverified. As a conservative project policy, keep the archive, source images, and image-containing/Grad-CAM figures local; do not publish or redistribute them unless rights are confirmed with the dataset author and underlying rights holders. This is not legal advice. (Official metadata documentation checked 2026-10-08 (UTC+07:00 local time): [Kaggle CLI dataset metadata/license labels](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets_metadata.md#licenses).)

## Reproduction commands

```bash
# Download with a user-managed Kaggle CLI installation and credentials:
kaggle datasets download puneet6060/intel-image-classification

# Or run Kaggle CLI ephemerally when `kaggle` is not installed:
uvx --from kaggle kaggle datasets download puneet6060/intel-image-classification

# Safe extraction refuses an existing target and rejects unsafe ZIP entries:
uv run --locked --extra dev python -c \
  'from py4ds_ai.data.download import extract_archive_safely; print(extract_archive_safely("data/raw/intel-image-classification.zip", "data/raw/extracted"))'

# Create the deterministic audit and split bundle:
uv run --locked --extra dev py4ds-prepare \
  --data-root data/raw/extracted \
  --output-dir data/manifests/seed-42 \
  --val-fraction 0.15 --seed 42
```

The preparation command only reads source images; it does not move, delete, resize, or rewrite them. `seg_test` remains assigned to test, validation is selected from `seg_train`, and `seg_pred` is retained as an unlabeled prediction pool. Exploratory montages sample only assigned training images.

## Verified audit results

The locked output is `data/manifests/seed-42/`.

| Source split | Files |
|---|---:|
| `seg_train` | 14,034 |
| `seg_test` | 3,000 |
| `seg_pred` | 7,301 |
| **Total** | **24,335** |

| Assigned split/status | Rows |
|---|---:|
| Train | 11,908 |
| Validation | 2,103 |
| Test | 3,000 |
| Unlabeled prediction pool | 7,301 |
| Excluded from supervised train/validation | 23 |
| Readable images | 24,335 |
| Non-square images | 68 |
| Cross-split exact SHA-256 overlaps | 0 |

The 23 training exclusions comprise 20 exact-hash label-conflict files and 3 exact copies of `seg_test` files. The 20 conflicting files are preserved in raw data and recorded as excluded; the original test examples are retained. Five extra byte-identical file copies remain inside the assigned training split, but no image-content hash crosses between train, validation, and test. All six required class directories were found. The fixed mapping is `buildings=0, forest=1, glacier=2, mountain=3, sea=4, street=5`.

The perceptual-hash screen produced 177 candidate pairs at Hamming distance at most 4; this includes pairs involving the unlabeled prediction pool. Among the 95 pairs wholly within the supervised manifest, 33 were train–test, 3 validation–test, 12 train–validation, 41 train–train, 2 validation–validation, and 4 test–test.

A bounded, targeted visual audit examined the 36 cross-split test pairs, 12 train–validation pairs, and 4 within-test pairs. This was leakage adjudication of flagged pairs, not exploratory test analysis; no test examples were used for feature, hyperparameter, or model selection. Fifty-one pairs were judged the same photograph or a near-duplicate; one test/train glacier pair remained uncertain. Conservatively, the non-test endpoint was excluded for all 48 cross-split pairs, including the uncertain pair. Four within-test candidate pairs remain in the preserved test set and are a test-independence limitation. Decisions are recorded in `configs/phash-review-seed-42.csv` (52 pairs; SHA-256 `c5bf7e94e3052e1cf123e28fb8e633153da6b5da15ee3f42447414919043bbf7`). No source image was modified or added to Git.

## Lock and generated outputs

- Baseline `split_manifest.csv` SHA-256: `ff407aeeb1f7c03561740ca876d82bf8c6b3febd261376a72f91be850235bdb9`.
- Leakage-reviewed manifest: `data/manifests/seed-42-phash-reviewed/`; `split_manifest.csv` SHA-256 `27c17f497e89a8f3b72975b45fcd1e44038e63edddbd4e721437a9a3500c982e`.
- The reviewed manifest retains all 3,000 test rows and assigns 11,868 train, 2,100 validation, 7,301 prediction, and 66 excluded rows (the original 23 plus 43 pHash-review exclusions). The source dataset fingerprint is unchanged.
- The reviewed bundle has zero active cross-split pHash candidate pairs and zero cross-split exact SHA-256 overlaps; it retains four pHash candidates within test.
- `split.lock.json` records the source root, seed 42, 0.15 validation fraction, class mapping, manifest hash, dataset fingerprint, review-ledger hash, parent-assignment hash, decisions, and excluded sample IDs.
- The bundle contains `audit.csv`, `split_manifest.csv`, `exclusions.csv`, `leakage_candidates.csv`, `class_to_idx.json`, `data_summary.json`, `split.lock.json`, and plots including a training-only sample montage.

Commands verified at the initial Phase 1 implementation checkpoint:

- `uv lock --check --python 3.12.14` — passed.
- `uv run --locked --extra dev pytest -q` — **20 passed at that checkpoint**.
- `uv run --locked --extra dev ruff check src tests scripts` — **passed at that checkpoint**.

The latest combined project verification after adding HOG, BoVW, and ResNet18 embedding support is recorded in `docs/EXPERIMENT_STATUS.md`; it passed Ruff and **40 tests** before CNN-training work began.

## Remaining caveats

1. Review the Kaggle dataset terms; the CLI license label alone is not a legal assessment.
2. Four retained test–test near-duplicate candidates are reported as an independence limitation in `FINAL_REPORT.md`; do not remove them or retune against test rows.
3. The original seed-42 preparation was repeated successfully before this targeted review; the reviewed manifest is a deterministic derived lock whose source fingerprint matches the original dataset fingerprint.
The five validation experiments, validation-only selection, one-time held-out evaluation, and source-grounded final report are complete. Earlier pre-review runs remain superseded. Training run metadata records a dirty worktree at launch, so the exact historical training source tree is not fully identified by the recorded commit alone; see `EXPERIMENT_STATUS.md` and `FINAL_REPORT.md`. Dataset reuse rights remain unverified; keep source images and image-bearing artifacts local.
