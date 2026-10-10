# Hermes G4 academic report review — 2026-10-10

**Reviewer:** Hermes, sole technical/research orchestrator under owner mandate `1558312501085478916`. Codex supplied the bounded narrative candidate; Hermes independently inspected and exercised it. No external ChatGPT GO is required.

**Decision:** report/evidence content ACCEPTED; this note's containing commit is the G4 source checkpoint and must pass the exact-SHA checks before the GitHub acceptance is issued. G5 requires its own clean-environment/integration gate. Operational watchdog remediation is separate and remains in progress.

## Real candidate and independent checks

- Worker `G4-REPORT-20261010-01`, base `b99b4e221ae21f28c9bd0aa5db9a2abf0b79f19b`, branch `agent/g4-report-20261010-01`, isolated worktree, tmux `py4ds-codex-worker`. Actual start 10:04:39 +07, end 10:07:56 +07, exit 0. Effective CLI turn context identifies `gpt-6-luna`, effort `high`, sandbox `workspace-write`, network false. This is CLI metadata, not independent provider-backend attestation.
- Independently inspected complete diff: report narrative only. All four generated blocks are byte-identical to the base. Candidate rendered twice from existing artifacts: SHA256 `48489c7ff6d88d9b6f00d244e6afac0db6aa23ff925d10c52c11e091d6e4800e` both times. Hermes then replaced the obsolete pending-review sentence with a pointer to this custody note; no numeric block changed.
- Integrated report rendered twice: identical SHA256 `52617b577a376d6e23e5db40c720601b041fc908983e71d7227b59a87dd345e1`.
- `uv lock --check` exit 0; locked Ruff on `src tests scripts` exit 0; CPU synthetic suite 117 passed in 12.80s, exit 0. Unit tests involving evaluator code use synthetic fixtures only, never the real sealed dataset.
- Saved verifier with original expected metrics/predictions hashes: exit 0, `VERIFIED_FROM_SAVED_PREDICTIONS`, matching 3,000 saved rows and 210 errors. No images, checkpoints or inference were loaded by that verifier.
- Protected custody comparison: 115 manifest/config/checkpoint/array/CSV/JSON files, 651,103,653 bytes, zero mismatches against the before-worker snapshot. Test guard remains `COMPLETED`. Original metrics digest `4df9443c7ad28c9ddd36cc461d38f5f7ed1a0b43b819d2ea0e400dedf9b4967e`; prediction digest `85d68f7fac4351716964ec0db773126805ac530c269ed37c607af8cae0486dde`.
- A verification subprocess launched from the Hermes Python kernel inherited its Python 3.14 `PYTHONPATH`, contaminating Python 3.12 imports and producing 18 collection errors. Failure log is preserved; direct sanitized invocation unsetting `PYTHONPATH`, `PYTHONHOME`, `VIRTUAL_ENV` passed. This is an orchestration environment failure, not concealed as a passing run or fixed by changing project dependencies.

## Academic traceability and source review

1. Problem/data: six-class supervised scene classification, immutable reviewed manifest, source/rights disclosure, fixed alphabetical labels and explicit data exclusions.
2. Literature: original HOG/SIFT/visual-word papers and CVF ResNet/transfer/Grad-CAM pages retrieved and inspected on 2026-10-10. No paper benchmark is substituted for Intel results. DOI/publisher availability is not a claim of unrestricted image rights.
3. Implementation: audited tested train-only fitting, deterministic validation embeddings, controlled head/layer4 training, single-use real evaluation protection and corrected validation crop-aligned Grad-CAM. Existing correct pipeline is reused, not rewritten.
4. Evaluation: five validation candidates plus majority reference; one historically frozen winner's test result. Report tables regenerated from saved artifacts, identity/aggregates reconciled. Actual saved search records contain 8 HOG trials, 8 BoVW trials and 4 embedding-classifier trials; both CNN histories record five completed epochs. No experiments repeated to verify these records.

Legacy notebook was parsed without execution; legacy PDF contains 22 pages and 41,182 extracted text characters. These historical artifacts are not current measurements. Unsafe legacy helpers remain archival and must not be used with protected source data; G5 must make that entry-point distinction unambiguous.

## Caveats retained

Historical dirty training worktree, one split/seed, four within-test near-duplicate candidates, incomplete comparable compute measurements and unresolved image redistribution rights remain disclosed. Six corrected validation overlays and twelve superseded overlays remain local. Same-UID Codex isolation is governance/write-sandbox isolation, not adversarial credential isolation. Independent saved-arithmetic checks are not an independent reproduction of training or final inference.

Durable logs/source retrieval index/model metadata/checks live in `../py4ds-ai-runtime/bootstrap-20261010/` and `../py4ds-ai-runtime/tasks/G4-REPORT-20261010-01/`. GitHub checkpoint supplies the full containing SHA, exact-SHA command outcomes and final verdict. No held-out evaluation, training, frozen artifact edit or image publication was performed.
