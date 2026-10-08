# Intel Natural Scene Classification — Final Evaluation Report

**Report date:** 2026-10-08
**Status:** one held-out evaluation completed; selected model scored once.
**Current evaluation code commit:** `5303a1078a54119c51445fd0e1e12feb4c1167e4`.

## Executive summary

<!-- BEGIN GENERATED SUMMARY -->
Validation-only selection chose **Pretrained ResNet18, layer4 fine-tuned** (macro-F1 0.929137) from 5 candidates. In its single final evaluation on 3,000 held-out images, it achieved **93.00% accuracy**, **0.931004 macro-F1**, and **0.929743 weighted-F1**; 210 images were misclassified. These values are read from saved artifacts, not published benchmark results.
<!-- END GENERATED SUMMARY -->

The per-class metrics and highest-count confusions below are generated from the saved final-evaluation metrics. Error patterns are descriptive of this test set and do not establish why the model made those errors.

## Dataset and evaluation protocol

The project used the Intel Image Classification dataset distributed on Kaggle.[6] The comparison uses the pHash-reviewed manifest `data/manifests/seed-42-phash-reviewed/split_manifest.csv` (SHA-256 `27c17f497e89a8f3b72975b45fcd1e44038e63edddbd4e721437a9a3500c982e`) with 11,868 training, 2,100 validation, and 3,000 test rows. The fixed class order is `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`.

The review ledger `configs/phash-review-seed-42.csv` (SHA-256 `c5bf7e94e3052e1cf123e28fb8e633153da6b5da15ee3f42447414919043bbf7`) records 52 pHash-pair decisions. For cross-split candidates, non-test endpoints were conservatively excluded; all 3,000 test rows were preserved. The reviewed manifest has no active cross-split pHash candidates or exact-hash overlaps. Four near-duplicate candidate pairs within test remain an independence caveat. Flagged test images were viewed only for targeted leakage adjudication, not for exploratory analysis or model selection. Test scoring occurred only after the five-candidate validation selection was frozen and committed.

## Validation comparison and frozen selection

HOG was included as a handcrafted baseline. Dalal and Triggs evaluated HOG for human detection, not this six-class scene task; their paper's results are not evidence for the scores below.[1]

SIFT provides local descriptors.[2] Csurka et al. describe a bag-of-keypoints approach that quantizes local descriptors into fixed-length visual-word histograms.[3] In this project, the vocabulary is fit on training data only.

ResNet uses residual learning; the cited paper motivates the architecture family but does not report results on this dataset.[4]

<!-- BEGIN GENERATED VALIDATION TABLE -->
| Comparator/reference | Validation macro-F1 | Test status |
|---|---:|---|
| Majority-class baseline | 0.050619 | Not evaluated |
| HOG + Logistic Regression, C=0.01 | 0.676893 | Not evaluated |
| SIFT-BoVW, 128 words + Logistic Regression, C=1 | 0.592861 | Not evaluated |
| Pretrained ResNet18, frozen backbone / trained head | 0.908711 | Not evaluated |
| Pretrained ResNet18, layer4 fine-tuned | 0.929137 | Selected; evaluated once |
| Frozen ResNet18 embeddings + LinearSVC, C=0.1 | 0.903315 | Not evaluated |
<!-- END GENERATED VALIDATION TABLE -->

The frozen selection record is `configs/final-selection-seed-42-phash-reviewed.json` (SHA-256 `aad84528377fd219e1cc6223cce107186b8b907e69bda6c4094a13416e31050b`). Its rule is highest validation macro-F1, with ascending `run_id` as the tie-breaker. The majority-class row is a validation-only reference, not a sixth candidate; its test score remains `NOT RUN`. The selected checkpoint is `runs/seed-42-phash-reviewed/cnn18/layer4-imagenet-v1/best_checkpoint.pt` (SHA-256 `84fd9791276833d6dd8a5a6d1e17c1d89cc41eab8922002774c309a6ffbfd019`). Only the selected CNN was evaluated on test; test results for the other candidates do not exist.

## Final test results

<!-- BEGIN GENERATED TEST TABLES -->
| Metric | Result |
|---|---:|
| Test examples | 3,000 |
| Accuracy | 0.930000 |
| Macro-F1 | 0.931004 |
| Weighted-F1 | 0.929743 |
| Misclassified examples | 210 / 3,000 |
| Evaluations of this test split | 1 |
| Parameters (total / trainable) | 11,179,590 / 8,396,806 |
| Evaluation time | 2.088380 s |
| Seconds per image | 0.000696127 |

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| buildings | 437 | 0.892241 | 0.947368 | 0.918979 |
| forest | 474 | 0.997881 | 0.993671 | 0.995772 |
| glacier | 553 | 0.922179 | 0.857143 | 0.888472 |
| mountain | 525 | 0.883117 | 0.906667 | 0.894737 |
| sea | 510 | 0.940075 | 0.984314 | 0.961686 |
| street | 501 | 0.949686 | 0.904192 | 0.926380 |
<!-- END GENERATED TEST TABLES -->

Confusion-matrix row totals use the true class and columns use the predicted class. The matrix and normalized matrix are saved in `runs/seed-42-phash-reviewed/final-evaluation/metrics.json`. The most frequent off-diagonal counts are:

<!-- BEGIN GENERATED ERROR TABLE -->
| True class | Predicted class | Count |
|---|---|---:|
| glacier | mountain | 58 |
| street | buildings | 46 |
| mountain | glacier | 38 |
| buildings | street | 21 |
<!-- END GENERATED ERROR TABLE -->

No test-image examples or image-bearing saliency figures are included here. Grad-CAM produces coarse localization maps; we treat such maps as qualitative visualizations, not ground-truth evidence about model reasoning.[5]

## Reproducibility record

- Evaluation command used the locked environment and committed code: Python 3.12.14, `uv.lock`, device `cuda:0`, NVIDIA GeForce GTX 1080 Ti, batch size 64, four data-loader workers.
- The generated results table reports elapsed and per-image time from this local run; these are not hardware-independent benchmarks.
- Reproduction command from the project root (the single-use guard will reject a second run for this manifest):

  ```bash
  uv run --locked --extra dev python scripts/evaluate_final.py \
    --project-root . \
    --selection configs/final-selection-seed-42-phash-reviewed.json \
    --manifest data/manifests/seed-42-phash-reviewed/split_manifest.csv \
    --output-dir runs/seed-42-phash-reviewed/final-evaluation \
    --device cuda:0 --batch-size 64 --num-workers 4
  ```

- Guard record: `runs/.test_evaluation_locks/27c17f497e89a8f3b72975b45fcd1e44038e63edddbd4e721437a9a3500c982e.json`, state `COMPLETED`.
- Evaluation artifacts are local and ignored by Git: `runs/seed-42-phash-reviewed/final-evaluation/metrics.json`, `predictions_test.csv`, `test_errors.csv`, `evaluation_config.json`, and `figures/test_confusion_matrix.png`.
- SHA-256: metrics `4df9443c7ad28c9ddd36cc461d38f5f7ed1a0b43b819d2ea0e400dedf9b4967e`; predictions `85d68f7fac4351716964ec0db773126805ac530c269ed37c607af8cae0486dde`; error rows `d277964ae4cb8db0e13557a4664f1b7740f6998314f6ddd08f382d0ca2c81d04`.
- The run metadata for the training experiments records source commit `11334744c8634bc5a593e69fa271dc69bb626fd4` with `git_worktree_dirty: true`. Thus, saved configurations, metrics, checkpoint hashes, and predictions are the direct evidence for those runs; the commit ID alone does not fully identify the exact training source tree. The final evaluator itself ran from commit `5303a1078a54119c51445fd0e1e12feb4c1167e4`.

## Limitations and interpretation

1. Four test–test near-duplicate candidates remain, so test examples are not guaranteed independent.
2. This is one held-out split and one selected checkpoint, not a multi-seed estimate with confidence intervals. No test comparison was performed for the non-selected models.
3. Model selection considered five validation candidates; the test split was not used to choose among them.
4. Dataset-specific reuse terms and underlying image rights remain unverified. The archive, source images, and image-bearing figures remain local; this report contains aggregate metrics only.
5. Earlier notebook and legacy `report.pdf` numbers are historical and superseded; they were not reproduced as current results. See `docs/EXPERIMENT_STATUS.md` and `docs/PHASE1_DATA_AUDIT.md` for the detailed audit trail.

## Requirements compliance matrix

| Requirement | Status | Evidence or limitation |
|---|---|---|
| Nondestructive dataset preparation, fixed labels, and reviewed split manifest | Complete | `docs/PHASE1_DATA_AUDIT.md`; reviewed manifest hash above |
| Leakage review and preserved test rows | Complete with caveat | pHash ledger above; four within-test near-duplicate candidates remain |
| Classical, frozen-embedding, and fine-tuned CNN comparison | Complete on validation | Five validation candidates; the generated comparison table is sourced from saved run metrics |
| Majority-class baseline | Validation reference only | Included from the HOG metrics artifact; test status `NOT RUN` |
| Validation-only selection and final held-out evaluation | Complete for selected model | Frozen selection JSON; final metrics, prediction, and error artifacts below; exactly one test evaluation |
| Per-class, error, confusion, parameter, and timing analysis | Complete; artifacts local | Generated from final `metrics.json`; prediction/error CSVs are not tracked in Git |
| Reproducible training source provenance | Partial | Training metadata records `git_worktree_dirty: true`; the commit alone does not identify the exact training source tree |
| Dataset reuse terms and image rights | Unverified | Keep raw images, archive, and image-bearing figures local until terms are confirmed |

## Sources

[1] https://doi.org/10.1109/CVPR.2005.177 — Dalal and Triggs, HOG for Human Detection (CVPR 2005)
[2] https://doi.org/10.1023/B:VISI.0000029664.99615.94 — Lowe, Distinctive Image Features from Scale-Invariant Keypoints (IJCV 2004)
[3] https://www.cs.princeton.edu/courses/archive/fall09/cos429/papers/csurka-eccv-04.pdf — Csurka et al., Visual Categorization with Bags of Keypoints (ECCV workshop 2004)
[4] https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html — He et al., Deep Residual Learning for Image Recognition (CVPR 2016)
[5] https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html — Selvaraju et al., Grad-CAM (ICCV 2017)
[6] https://www.kaggle.com/puneet6060/intel-image-classification/metadata — Kaggle dataset metadata: Intel Image Classification
