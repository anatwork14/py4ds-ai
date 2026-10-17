# Intel Natural Scene Classification — Final Evaluation Report

**Report date:** 2026-10-08
**Status:** one held-out evaluation completed; selected model scored once.
**Historical one-time evaluation code commit:** `5303a1078a54119c51445fd0e1e12feb4c1167e4`.

## Executive summary

<!-- BEGIN GENERATED SUMMARY -->
Validation-only selection chose **Pretrained ResNet18, layer4 fine-tuned** (macro-F1 0.929137) from 5 candidates. In its single final evaluation on 3,000 held-out images, it achieved **93.00% accuracy**, **0.931004 macro-F1**, and **0.929743 weighted-F1**; 210 images were misclassified. These values are read from saved artifacts, not published benchmark results.
<!-- END GENERATED SUMMARY -->

The per-class metrics and highest-count confusions below are generated from the saved final-evaluation metrics. Error patterns are descriptive of this test set and do not establish why the model made those errors.

## Problem definition and research questions

**Task:** given one RGB photograph, predict exactly one of six scene classes:
`buildings`, `forest`, `glacier`, `mountain`, `sea`, or `street`.
This is supervised multiclass image classification, not the demand forecasting,
financial, credit-scoring, or anomaly-detection topics suggested as *examples*
in the assignment. The difficulty is that classes may share structure, texture,
lighting and objects; for instance a street scene can contain buildings and
a glacier scene can contain mountains.

The academic questions are: **(RQ1)** how do handcrafted descriptors compare
with learned pretrained representations on a common validation split?
**(RQ2)** does training a ResNet classification head or unfreezing its last
convolutional block improve validation macro-F1 over a frozen representation?
**(RQ3)** which classes account for the final selected model's mistakes?
Macro-F1 is the primary selection criterion because it gives each scene class
equal weight, regardless of frequency. Accuracy, weighted-F1 and per-class
scores are supplementary.

The hypotheses were documented before final test scoring in
[`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md): pretrained representations
could outperform handcrafted features (H1); fine-tuning might improve
performance while increasing compute and overfitting risk (H2); semantically
overlapping classes might generate systematic confusions (H3); and split
integrity and transform consistency matter to validity (H4).
The experiments provide evidence for H1 and an association relevant to H2/H3;
they do **not** isolate the effect of each preprocessing choice (H4) or
establish a causal explanation for particular mistakes.

## Dataset and evaluation protocol

The project used the Intel Image Classification dataset distributed on Kaggle.[6] The comparison uses the pHash-reviewed manifest `data/manifests/seed-42-phash-reviewed/split_manifest.csv` (SHA-256 `27c17f497e89a8f3b72975b45fcd1e44038e63edddbd4e721437a9a3500c982e`) with 11,868 training, 2,100 validation, and 3,000 test rows. The fixed class order is `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`.

The review ledger `configs/phash-review-seed-42.csv` (SHA-256 `c5bf7e94e3052e1cf123e28fb8e633153da6b5da15ee3f42447414919043bbf7`) records 52 pHash-pair decisions. For cross-split candidates, non-test endpoints were conservatively excluded; all 3,000 test rows were preserved. The reviewed manifest has no active cross-split pHash candidates or exact-hash overlaps. Four near-duplicate candidate pairs within test remain an independence caveat. Flagged test images were viewed only for targeted leakage adjudication, not for exploratory analysis or model selection. Test scoring occurred only after the five-candidate validation selection was frozen and committed.

## Literature synthesis and methodological choices

The reviewed literature supports a deliberate progression in representation
capacity, rather than a collection of unrelated classifiers:

- **Handcrafted gradients:** Dalal and Triggs' HOG [1] summarizes
  contrast-normalized local edge directions, offering spatial information
  without pretrained weights. The source paper addresses human detection, so
  its published metrics are not benchmark results for these scenes.
- **Local descriptors and quantization:** Lowe's SIFT [2] detects local
  keypoints; Csurka et al. [3] make variable-length descriptor sets comparable
  through visual-word histograms. This introduces vocabulary learning and can
  discard spatial arrangement; it is a useful contrast with structured HOG.
- **Learned representations and transfer:** residual networks [4] address
  optimization of deeper CNNs. Studies of ImageNet transfer [7] motivate
  comparing frozen features with fine-tuning, rather than assuming that
  fine-tuning always wins on a new domain.
- **Qualitative explanation:** Grad-CAM [5] localizes features relevant to
  one class logit, but does not provide a validated causal account of a
  prediction. Correct visual alignment with the actual model input is essential.

These sources motivate the representations and experiment design. They do not
supply any numerical Intel-scene results in the tables below. For full
author, venue, method and limitation notes see
[`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md).

## Implemented pipeline, preprocessing and tuning

1. **Data:** original images are left unchanged; a locked reviewed manifest
   partitions the original labeled training set into train and validation,
   while preserving the source test split for the final winner only. Label
   order and image-content hashes are checked. The pHash ledger removes
   reviewed cross-split leakage candidates but cannot prove universal
   independence.
2. **HOG/classical:** grayscale images are resized to 128 × 128. The default
   OpenCV HOG uses nine orientation bins, 16 × 16-pixel cells and 2 × 2-cell
   blocks, producing 1,764 features. The classifier pipeline fits
   `StandardScaler` on training data and searches Logistic Regression and
   LinearSVC regularization (`C`). The committed winning HOG comparator uses
   Logistic Regression with `C=0.01`.
3. **SIFT–BoVW/classical:** grayscale 128 × 128 images yield SIFT
   descriptors. MiniBatchKMeans learns visual-word centers from *training
   descriptors only*, then L2-normalized histograms represent images.
   Vocabulary size and classifier/`C` are validation-tuned; images with
   zero keypoints have zero histograms. The selected comparator uses
   128 words and Logistic Regression with `C=1`.
4. **Frozen deep/hybrid:** torchvision's ImageNet-pretrained ResNet18 and its
   weight-specific inference transform produce 512-dimensional image
   embeddings. A training-fitted scaler and a validation-tuned linear
   classifier complete the pipeline. The best recorded variant is
   LinearSVC with `C=0.1`.
5. **CNN head training/fine-tuning:** both variants use ImageNet ResNet18,
   a six-class linear head, cross-entropy loss and AdamW. Training uses
   random resized crops, horizontal flips, small rotations, and color
   jitter; validation uses the deterministic weight-specific 256-resize /
   224-center-crop and ImageNet normalization. In the head-only model the
   backbone is frozen; in the fine-tuned model `layer4` is also trainable,
   with a smaller learning rate than the classification head. Checkpoints
   and early stopping are selected on validation macro-F1.
6. **Evaluation:** each comparator's ranking is determined solely from
   validation. Only the frozen winner is loaded for one test evaluation.
   The output includes classwise precision, recall, F1, support, confusion
   counts and inference timing. All other test scores remain **NOT RUN**.

The saved `config.json` files for the seed-42 pHash-reviewed runs record
these configured search budgets: HOG used `C=[0.01, 0.1, 1, 10]` with
LinearSVC and LogisticRegression; SIFT–BoVW used vocabulary sizes `[64, 128]`,
`C=[0.1, 1]`, both learners, and `max_descriptors=30000`; frozen ResNet
embeddings used `C=[0.1, 1]` with the same two learners. These are configured
search spaces, not independent proof that every trial completed or produced a
saved result; trial-completion claims require per-trial evidence.

The saved CNN configuration records a maximum of five epochs, patience 2,
batch size 64, and head learning rate 0.001; the layer4 learning rate is
0.00001. Five is a cap, not a claim that all five epochs ran. The small
from-scratch CNN is optional and **NOT RUN**; it is not required for the
reported comparison.

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

## Interpretation of the validation comparison

On this *single fixed validation split*, the ranking is clear:
SIFT–BoVW (0.592861), HOG (0.676893), frozen ResNet embeddings
(0.903315), pretrained ResNet head training (0.908711), and layer4
fine-tuning (0.929137). A majority-class reference scored 0.050619.
Frozen deep features already exceed both handcrafted methods, consistent
with H1. This is evidence that pretrained transferable representations are
useful for this dataset, not proof that every ResNet beats every handcrafted
method or that every hyperparameter search received equal compute.

**HOG versus SIFT–BoVW:** HOG's higher validation score is compatible with
the idea that a spatial gradient grid retains scene-layout cues lost by
an unordered visual-word histogram. However, differences in extraction,
vocabulary capacity and tuning also exist. No controlled ablation isolates
the cause.

**Frozen versus trained ResNet:** head training (0.908711) only modestly
exceeds frozen embeddings with a linear SVM (0.903315); layer4
fine-tuning (0.929137) exceeds both on this validation run. These are
different learning procedures with different optimization costs, so the
difference cannot be attributed exclusively to unfreezing layer4.
There are no repeated seeds or confidence intervals establishing how
stable the ranking would be under new splits.

**Compute trade-offs:** handcrafted features avoid pretrained network
weights; SIFT requires descriptor extraction and vocabulary fitting;
frozen ResNet needs inference plus a lightweight classifier; fine-tuning
adds backpropagation, activation memory and optimizer state. These are
methodological costs, not measured head-to-head speed rankings: the
committed report does not provide comparable end-to-end wall times,
energy or peak memory across all candidates. The final selected model's
inference time below is hardware- and batch-specific and must not be
presented as evidence that it is faster than the other approaches.

**Final error analysis:** the most common off-diagonal categories below
include glacier→mountain and mountain→glacier, plus street→buildings
and buildings→street. Their shared visual structures are plausible
hypotheses, not demonstrated mechanisms. Forest has the strongest final
per-class F1; glacier has the weakest recall. Because only the winning
model was evaluated on test, these are **not** test-set comparisons with
other methods, and the frozen model must not be retuned from these errors.

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
- **Historical one-time evaluation command — provenance only; DO NOT RUN.** It records how the already-completed evaluation was invoked. The held-out evaluator must not be run again:

  ```bash
  # HISTORICAL PROVENANCE ONLY — DO NOT RUN
  uv run --locked --extra dev python scripts/evaluate_final.py \
    --project-root . \
    --selection configs/final-selection-seed-42-phash-reviewed.json \
    --manifest data/manifests/seed-42-phash-reviewed/split_manifest.csv \
    --output-dir runs/seed-42-phash-reviewed/final-evaluation \
    --device cuda:0 --batch-size 64 --num-workers 4
  ```

- Safe current commands from the project root render the report and audit the already-saved predictions against the original expected digests. They do not perform model inference or score test images:

  ```bash
  uv run --locked --extra dev python scripts/render_report.py
  uv run --locked --extra dev python scripts/verify_saved_final_artifacts.py \
    --expected-metrics-sha256 4df9443c7ad28c9ddd36cc461d38f5f7ed1a0b43b819d2ea0e400dedf9b4967e \
    --expected-predictions-sha256 85d68f7fac4351716964ec0db773126805ac530c269ed37c607af8cae0486dde
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

## Qualitative explanation integrity

The model-inspection pipeline uses Grad-CAM on the output of the final ResNet18 `layer4` residual block, targeting the saved validation prediction. Six corrected validation-only overlays were regenerated locally from the frozen selected checkpoint and saved validation predictions after the crop-alignment correction. Their local QA manifest records sample IDs, true/predicted labels, source paths and per-image hashes; the reviewer has not independently inspected the local image bytes. Twelve earlier overlays (six head-only and six layer4) predate the correction and are marked superseded locally. The corrected overlays are qualitative visualizations, not causal or ground-truth explanations. No test images were used. All image-bearing outputs remain local and are neither embedded here nor committed while redistribution rights remain unverified. Do not rerun the held-out evaluator or select a different model based on these visualizations.

## Scope of independent verification

The earlier GitHub-only reviewer could inspect tracked source and report text,
but did not have access to local artifact bytes; that review alone did not
independently verify the quoted test aggregates. On 2026-10-10, Hermes's
Ubuntu custody audit used `scripts/verify_saved_final_artifacts.py` to
recompute aggregates and verify sample IDs, order, and labels from the saved
prediction CSV against the immutable manifest. It exited 0 with
`VERIFIED_FROM_SAVED_PREDICTIONS`. This is a read-only arithmetic and identity
audit by the current orchestrator, not an independent reproduction of the
original training or evaluation. Current G4 report-renderer and final-SHA acceptance are recorded in
[`HERMES_G4_ACCEPTANCE.md`](HERMES_G4_ACCEPTANCE.md). Review history remains in
[`FINAL_REVIEW_2026-10-08.md`](FINAL_REVIEW_2026-10-08.md); current authority
and gate status are recorded in [`HERMES_MASTER_PLAN.md`](HERMES_MASTER_PLAN.md)
and [`HERMES_DECISION_LOG.md`](HERMES_DECISION_LOG.md).

## Four course criteria

| Instructor criterion | Where addressed | Evidence and qualification |
|---|---|---|
| Problem definition | Problem definition and research questions; Dataset and evaluation protocol | Six-class supervised scene classification, research questions, reviewed manifest and split counts. |
| Literature review | Literature synthesis and methodological choices; `docs/LITERATURE_REVIEW.md` | HOG, SIFT/BoVW, ResNet/transfer learning and Grad-CAM are attributed to their cited publications; paper benchmarks are not presented as Intel results. |
| Algorithm/pipeline | Implemented pipeline, preprocessing and tuning | Data/leakage controls, HOG, SIFT-BoVW, frozen ResNet features, head training and layer4 fine-tuning with validation-only selection. |
| Comparative evaluation | Validation comparison, frozen selection, final test results and limitations | Five candidate comparisons use validation macro-F1; only the frozen winner has one held-out test result. Single-split, dirty historical training-source and rights caveats remain disclosed. |

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
[6] https://www.kaggle.com/datasets/puneet6060/intel-image-classification — Kaggle dataset page: Intel Image Classification
[7] https://openaccess.thecvf.com/content_CVPR_2019/html/Kornblith_Do_Better_ImageNet_Models_Transfer_Better_CVPR_2019_paper.html — Kornblith, Shlens and Le, Do Better ImageNet Models Transfer Better? (CVPR 2019)
