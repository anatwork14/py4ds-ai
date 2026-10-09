# Grad-CAM alignment QA — TASK 04 / G3

**Status:** evidence ready for reviewer; no model-selection change.
**Date:** 2026-10-09 (Asia/Ho_Chi_Minh).
**Scope:** validation examples only. No `seg_test` image was opened, transformed or scored; no training or final evaluator was run.

## Alignment and provenance findings

- The implementation fix is commit `c2b50da` (`fix(gradcam): overlay heatmaps on exact normalized validation crop`, 2026-10-08 23:03:06 +07). `_gradcam_overlay_base` inverse-normalizes the actual model tensor, so its background uses the same weight-specific resize and center crop as validation inference.
- The existing non-square synthetic regression compares that background pixel-for-pixel with the expected `Resize(256)` + `CenterCrop(224)` crop. The focused command `uv run --locked --extra dev pytest -q tests/test_cnn_training.py::test_gradcam_background_matches_nonsquare_validation_crop tests/test_gradcam.py` exited 0: **2 passed in 3.18s**. Test log SHA-256: `a4976dddf28090a73229d07dc62e40cd1f9f7144c58af758069c31465cb68072`.
- The six saved `head-imagenet-v1` overlays were written 2026-10-08 12:12:48–12:12:50 +07; the six saved `layer4-imagenet-v1` overlays were written 2026-10-08 15:47:10 +07. All twelve predate the correction commit. Their PNG bytes remain unchanged; a `README-SUPERSEDED.md` marker in each directory says not to use them as spatially validated illustrations.
- Regenerated six overlays from the already-frozen selected layer4 checkpoint (`84fd9791276833d6dd8a5a6d1e17c1d89cc41eab8922002774c309a6ffbfd019`) using only the 2,100 locked validation rows and saved validation predictions. The selection rule remains: up to six validation errors in manifest order, then correct validation rows if needed. The six selected rows were existing validation errors; no new predictions were used to select or tune a model.
- All six outputs are RGB, **224×224 px**. The local JSON manifest records each sample ID, true/predicted class, Grad-CAM target class, caption, selection rule, caveat, output path, dimensions and SHA-256. Visual inspection of `0030-buildings-pred-street.png` and `0119-buildings-pred-sea.png` found no obvious stretch or crop mismatch. Grad-CAM remains qualitative and is not causal evidence or ground truth.

## Artifacts and protected-state check

- Local-only figure directory (ignored by Git): `runs/seed-42-phash-reviewed/cnn18/layer4-imagenet-v1/figures/gradcam-validation-v2/gradcam/`.
- Figure manifest: `runs/seed-42-phash-reviewed/cnn18/layer4-imagenet-v1/figures/gradcam-validation-v2/gradcam_validation_manifest.json`; SHA-256 `be7904cb14f6839a8b4b12a48927ea9acd97bf0ff889567267b61d92ca9c4535`.
- Generation audit log: `/mnt/ssd/genos2/.hermes/cache/scratch/py4ds-ai-g3-regenerate-validation-gradcam.log`; SHA-256 `40d1e29a62a41252db557853bae55e73f46a3b7f3c402ae509329090c8d36182`.
- The harness hashed the reviewed manifest, frozen selection, selected checkpoint, validation predictions, saved final metrics/predictions/errors/config and evaluation guard before and after rendering, and asserted every digest was unchanged. The guard remains `COMPLETED`.
- No raw image or image-containing figure was committed or uploaded: dataset redistribution rights remain unverified. The full local manifest retains image hashes and captions for authorized inspection.

The one-off rendering harness is `/mnt/ssd/genos2/.hermes/cache/scratch/py4ds-ai-g3-regenerate-validation-gradcam.py`; it loaded the frozen checkpoint and saved validation predictions, computed Grad-CAM for the predetermined validation examples, and did not call the training or final-evaluation path.
