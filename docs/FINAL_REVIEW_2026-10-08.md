# Independent final-review gate — CO3117 Machine Learning project

**Review date:** 2026-10-08 (Asia/Ho_Chi_Minh)  
**Scope:** [PR #3](https://github.com/anatwork14/py4ds-ai/pull/3), stacked on Phase 0 / protocol PRs.  
**Reviewer disposition:** **CONDITIONAL / NOT YET SIGNED OFF**, pending local Ubuntu verification below.  
**Target:** the four instructor requirements (problem, literature, implementation, evaluation), **not** production-service availability or CI.

## Decision and evidence levels

The project is academically meaningful and meets the *design/implementation scope*
for six-class natural-scene classification. Its strength is a controlled comparison
of handcrafted features, frozen pretrained image features, and fine-tuned CNNs
on an explicitly guarded split. Suggested coursework topics are examples;
scene classification is an appropriate supervised-learning problem.

- **Directly inspected (source and versioned metadata):** manifest and split
  safeguards, SIFT-BoVW/HOG extraction, training/validation separation,
  model selection, final evaluator's single-attempt guard, report renderer,
  source bibliography, summary report and regression tests.
- **Reported by server agent, not independently executed here:** 94 passing
  pytest tests at the earlier commit; Ruff/uv success; one final held-out
  evaluation; checkpoint and prediction file digests. These claims are not
  silently upgraded to reviewer-verified results.
- **Unavailable to this GitHub-only review:** source image archive,
  all ignored `runs/` artifacts, local checkpoint bytes, full saved test
  confusion matrix/prediction CSV, current Ubuntu interpreter/GPU environment,
  and test-run terminal logs.
- **Deliberately excluded:** GitHub Actions, a production deployment, new
  model architectures, retraining the winner, or any repeat of the held-out
  test evaluation. Local Ubuntu tests are the project's execution gate.

## Requirements traceability

| Instructor requirement | Finding | Primary evidence | Remaining academic action |
|---|---|---|---|
| 1. Problem definition/dataset | **Meets scope** | `README.md`, `docs/FINAL_REPORT.md`, manifest protocol | Discuss motivation, supervised 6-class goal and dataset-rights uncertainty |
| 2. Literature review | **Meets scope, documentation strengthened** | `docs/LITERATURE_REVIEW.md`, `docs/FINAL_REPORT.md` | Verify bibliography links and attribution in final rendered hand-in |
| 3. Algorithms/pipeline | **Implemented; explanation fix supplied** | `src/py4ds_ai/features/`, `src/py4ds_ai/models/`, tests | Run updated tests; recreate validation-only Grad-CAM illustrations |
| 4. Comparative evaluation | **Implemented, independent results check pending** | frozen `configs/final-selection-seed-42-phash-reviewed.json`, `docs/FINAL_REPORT.md` | Validate saved metrics/checkpoint/predictions and report render on Ubuntu |

## Corrections committed during final review

1. **Scientific metric consistency:** `scripts/render_report.py` now derives
   class precision, recall and F1, macro-F1 and weighted-F1 from the saved
   confusion matrix, rejecting mismatches instead of checking only finiteness.
   It also checks selected-candidate identity and the selected checkpoint hash.
   Synthetic fixtures and negative regression tests were corrected/added in
   `tests/test_report_rendering.py`.
2. **Grad-CAM geometry:** `src/py4ds_ai/models/cnn_training.py` now
   inverse-normalizes the actual validation tensor (which includes the
   weight-specific resize and center crop) for background pixels, instead
   of distorting the source image with a separate direct 224×224 resize.
   A non-square-image alignment regression test was added to
   `tests/test_cnn_training.py`.
3. **Submission narrative:** `docs/FINAL_REPORT.md` now explicitly maps
   the research questions, literature, train/validation/test protocol,
   actual implemented feature engineering, tuning approach, comparative
   interpretation, measured-versus-unmeasured efficiency, and limitations
   to the four coursework requirements.

These edits do **not** change trained weights, the selected model, frozen
manifest, validation metrics, or held-out test metrics.

## Mandatory Ubuntu acceptance steps

From a working tree containing this review's commits, with the same locked
environment and locally available ignored data/run artifacts:

```bash
git status --short
uv lock --check --python 3.12.14
uv run --locked --extra dev ruff check src tests scripts
uv run --locked --extra dev pytest -q
uv run --locked --extra dev python scripts/render_report.py
uv run --locked --extra dev python scripts/verify_saved_final_artifacts.py \
  --expected-metrics-sha256 4df9443c7ad28c9ddd36cc461d38f5f7ed1a0b43b819d2ea0e400dedf9b4967e \
  --expected-predictions-sha256 85d68f7fac4351716964ec0db773126805ac530c269ed37c607af8cae0486dde
```

**Interpretation:** all commands must exit zero. Fix code/test failures on the
PR branch; do not reinterpret a failure as a license to rerun the final
evaluator. The report renderer only reads the local frozen selection and
saved result JSON; it does not run inference or score test images.
Inspect `git diff -- docs/FINAL_REPORT.md` after rendering for unexplained
changes. Repeated rendering should produce the same content.

Validate provenance **locally**, without distributing restricted images:

```bash
sha256sum data/manifests/seed-42-phash-reviewed/split_manifest.csv
sha256sum configs/final-selection-seed-42-phash-reviewed.json
sha256sum runs/seed-42-phash-reviewed/cnn18/layer4-imagenet-v1/best_checkpoint.pt
sha256sum runs/seed-42-phash-reviewed/final-evaluation/metrics.json
sha256sum runs/seed-42-phash-reviewed/final-evaluation/predictions_test.csv
cat runs/.test_evaluation_locks/27c17f497e89a8f3b72975b45fcd1e44038e63edddbd4e721437a9a3500c982e.json
```

Compare these digests to the recorded report values. Note that editing the
tracked selection JSON will change its digest and invalidate the report
provenance statement. Do **not** recreate or overwrite the selected JSON.

For a stronger independent results check, have an Ubuntu reviewer recompute
the full confusion matrix and the accuracy/macro/weighted/per-class scores
from the existing `predictions_test.csv` and compare every entry to
`metrics.json`. This reads prior predictions; it is **not** another model
evaluation. The report renderer's arithmetic check alone does not establish
that the confusion matrix agrees with the underlying prediction CSV.

For Grad-CAM, independently regenerate **validation-only** illustrations
from the saved best checkpoint after the crop-alignment fix; do not use or
redistribute the old, geometrically unverified overlays, and do not access
test images for any additional visual/model-selection work.

## Outcome interpretation and non-blocking limitations

- The five validation macro-F1 scores are SIFT–BoVW 0.592861, HOG
  0.676893, frozen ResNet + SVM 0.903315, ResNet head 0.908711 and
  ResNet layer4 fine-tuned 0.929137. The majority baseline is 0.050619.
- The frozen validation winner's **reported** single held-out result is
  macro-F1 0.931004 and accuracy 0.930000 on 3,000 test rows.
  No other candidate was scored on test; do not fabricate a test comparison.
- These are single-split measurements, not statistically significant,
  repeated-seed estimates. A larger F1 does not alone imply a better
  deployment-time cost profile; only the winner's inference timing is
  available in the report.
- Four within-test near-duplicate candidate pairs remain; the historical
  training worktree was dirty; redistribution rights and image ownership
  have not been independently established. Continue to disclose all three.
  Do not commit raw image content, restricted training checkpoints or
  image-bearing saliency outputs before rights are resolved.

## Final reviewer verdict and stop conditions

**Conditional pass on scope; final reproducibility sign-off pending.**
The project has real academic value without any GitHub Actions, deployment
stack, additional architecture, or further held-out evaluation. The next
commit should only follow confirmed Ubuntu failures, documentation corrections,
or a verified artifact-consistency issue. Maintain the sealed final-test guard
as `COMPLETED` and preserve the validation-only selection record.

**Approve the final coursework hand-in only after:** (a) all five Ubuntu
commands exit successfully, (b) the local artifact digests and aggregate
metrics match the recorded evidence, (c) the corrected Grad-CAM regression
passes and any shared Grad-CAM figures were regenerated from validation
images only, and (d) the submitted report retains the above caveats.
