# CO3117 — Agent execution plan and final-review control board

**Issued:** 2026-10-08 (Asia/Ho_Chi_Minh). **Authority:** final reviewer operating through [PR #3](https://github.com/anatwork14/py4ds-ai/pull/3).  
**Scope:** academic six-class Intel image classification; classical ML and deep learning. **Target:** the four instructor criteria (problem, literature, algorithm pipeline, comparative evaluation).  
**Decision:** CONDITIONAL ACCEPTANCE of project scope; not yet a validated final submission.  
**Workflow:** Ubuntu server is the execution environment; GitHub PR comments are the evidence/review interface. **No GitHub Actions.**

> **GitHub Issues are disabled in this repository as of the creation of this plan** (GitHub API 410). Until the owner enables **Settings → General → Features → Issues**, treat the numbered tasks below as issue-equivalent work items and post task-specific evidence on PR #3. Once Issues are enabled, copy each task into a separate GitHub Issue, assign an executor, and link it from PR #3. Do not falsely claim Issues were created.

## Immutable experimental boundary

- The original `seg_test` final evaluation already ran once and its guard is `COMPLETED`. **NEVER run `py4ds-evaluate-final`, `scripts/evaluate_final.py`, or another scored test inference** on this manifest. Read-only auditing of the previously saved prediction CSV is allowed.
- Preserve original Kaggle data, the reviewed split manifest and hash, frozen selection JSON, final metrics/predictions and checkpoint bytes. Do not rerun training, repeat seeds, change hyperparameters, move validation rows, rebuild manifest or select a new winner without new explicit user authorization.
- Review and fixes may update documentation, source validators, synthetic tests, or **validation-only** Grad-CAM illustrations derived from an existing checkpoint.
- Do not publish raw images, checkpoints, Kaggle archive, credentials, `runs/` caches or copyrighted images until redistribution rights are established.
- Test metrics for four unselected models and the majority baseline remain **NOT RUN**. Do not fill them in using test data. Keep the four within-test near-duplicate pairs, historically dirty training worktree, one-split uncertainty and unresolved image rights plainly disclosed.
- No GitHub Actions, cloud infrastructure, Docker deployment, extra algorithm for a cosmetic score gain, broad refactor, or speed claims without real comparable measurements.

## GitHub PR stack and branch policy

`main` ← **PR #1** (`docs/agent-research-and-evaluation-protocol`, currently draft) ← **PR #2** (`implementation/phase0-audit`) ← **PR #3** (`implementation/phase1-data-manifest`).

Agent changes must be made in the existing PR #3 branch, or in a focused branch/PR explicitly linked to #3, without rewriting the historical execution records. Before every push, inspect `git branch --show-current`, `git rev-parse HEAD`, `git status --short`, `git fetch`, and recent PR feedback. Pull/reconcile rather than overwriting an existing dirty Ubuntu worktree. Never force-push. Merge **in bottom-to-top dependency order (#1 → #2 → #3)** only after reviewer GO; never merge or mark draft ready unilaterally.

## Ordered work items and non-negotiable exit gates

### TASK 01 — [P0][G0] Align Ubuntu worktree and evidence

**Owner:** Ubuntu execution agent. **Depends on:** none.

- [ ] Confirm Ubuntu git branch/head, clean/dirty status and any uncommitted code; reconcile differences with PR #3 without discarding work.
- [ ] Check actual Python and `uv` versions, `uv.lock` and that local `runs/` / manifest / checkpoint paths are present. Report absent paths without synthesizing replacements.
- [ ] Inventory PR #1/#2/#3 state and review comments, including #1's draft state, and identify actual conflicts (do not infer a conflict from a transient mergeability status).
- [ ] Post a checkpoint with head SHA, environment, exact paths/status and blocker list on PR #3.

**Exit:** reviewer can identify exactly which git revision and immutable experiment artifacts the operator is using. No model execution.

### TASK 02 — [P0][G1] Execute reproducible local code checks

**Owner:** Ubuntu execution agent. **Depends on:** TASK 01 GO.

```bash
uv lock --check --python 3.12.14
uv run --locked --extra dev ruff check src tests scripts
uv run --locked --extra dev pytest -q
git diff --check
```

- [ ] Record each command's actual exit code, test count, failure excerpt and UTC/+07 timestamp; provide a server-log path or paste concise logs in PR #3.
- [ ] Address test/lint failures with minimal changes plus a regression test, and rerun affected tests and full suite.
- [ ] Explicitly verify recently added `tests/test_report_rendering.py`, `tests/test_cnn_training.py`, and `tests/test_saved_evidence_verification.py`; **do not reuse the old “94 passed” figure as a new test result**.

**Exit:** all four commands succeed at a recorded SHA; reviewer posts GO. CI is not required.

### TASK 03 — [P0][G2] Verify sealed result artifacts *read-only*

**Owner:** Ubuntu evidence auditor. **Depends on:** TASK 02 GO.

```bash
sha256sum data/manifests/seed-42-phash-reviewed/split_manifest.csv
sha256sum configs/final-selection-seed-42-phash-reviewed.json
sha256sum runs/seed-42-phash-reviewed/cnn18/layer4-imagenet-v1/best_checkpoint.pt
sha256sum runs/seed-42-phash-reviewed/final-evaluation/metrics.json
sha256sum runs/seed-42-phash-reviewed/final-evaluation/predictions_test.csv
uv run --locked --extra dev python scripts/verify_saved_final_artifacts.py \
  --expected-metrics-sha256 4df9443c7ad28c9ddd36cc461d38f5f7ed1a0b43b819d2ea0e400dedf9b4967e \
  --expected-predictions-sha256 85d68f7fac4351716964ec0db773126805ac530c269ed37c607af8cae0486dde
```

- [ ] Compare digests to `docs/FINAL_REPORT.md`; inspect the manifest/selected checkpoint and completed test guard.
- [ ] Verify saved prediction **sample ID, order and true class** against the immutable test-manifest rows; recalculate 6×6 confusion matrix, accuracy, macro-F1, weighted-F1, per-class precision/recall/F1 and errors from the saved CSV.
- [ ] Do not interpret the validator passing as a fresh test evaluation or proof the original training source tree was clean. If hashes disagree, STOP; preserve existing files and report discrepancy without overwriting artifacts.

**Exit:** integrity evidence is posted and reviewer independently checks claimed consistency. **No additional test inference.**

### TASK 04 — [P1][G3] Correct and review qualitative explanation

**Owner:** model-explanation agent. **Depends on:** TASK 02 GO; no new test work.

- [ ] Confirm `_gradcam_overlay_base` inverse normalization matches the model's deterministic resize/crop using non-square synthetic regression.
- [ ] Determine whether old saved Grad-CAM files were created before geometry correction; mark old illustrations superseded and regenerate **validation-only** overlays from the frozen existing checkpoint if legally permissible. Do not touch `seg_test`.
- [ ] Review captions: true/pred class, selection rule and Grad-CAM qualitative-not-causal caveat. Validate exported image dimensions, not just that a file exists.
- [ ] Do not make model-selection decisions from explanations.

**Exit:** corrected validation figure manifest or explicit NOT REGENERATED with justification; reviewer GO.

### TASK 05 — [P1][G4] Finish report and scientific interpretation

**Owner:** report agent. **Depends on:** TASK 03 GO; TASK 04 if illustrating Grad-CAM.

```bash
uv run --locked --extra dev python scripts/render_report.py
git diff -- docs/FINAL_REPORT.md
```

- [ ] Run renderer twice; confirm generated blocks are stable and no numbers change unexpectedly.
- [ ] Proofread `docs/FINAL_REPORT.md` and `docs/LITERATURE_REVIEW.md` against the **four instructor criteria**, with genuine source citations.
- [ ] Verify dataset/problem statement, six labels, split counts, preprocessing, HOG/SIFT-BoVW, frozen embeddings, head versus layer4 training, hyperparameter search evidence, selection rule and results are explained accessibly.
- [ ] Emphasize **validation** comparison across all algorithms, with **test** numbers only for the frozen winner. Explain HOG vs BoVW, frozen-vs-fine-tuned outcomes as possible mechanisms rather than causal proofs.
- [ ] Confirm measured versus inferred computational trade-offs, single-split uncertainty, potential data duplication, licensing and exact provenance caveats, plus executable reproduction instructions.
- [ ] Keep generated numerical blocks under their markers; do not hand-edit tracked test scores.

**Exit:** independently checkable academic final report; no fabricated claims.

### TASK 06 — [P1][G5] Review, reconcile and integrate PR stack

**Owner:** repo integration agent; **reviewer has final approval authority**. **Depends on:** TASK 01–05 GO.

- [ ] Review PR #1, #2, #3 comments, changed-file diffs and merge readiness. Clear blockers by traceable commits.
- [ ] Require all relevant Ubuntu evidence with exact latest SHA; ensure prior PR reviews still apply after any rebases.
- [ ] Ask reviewer for explicit GO before changing #1's draft state or merging any PR.
- [ ] Preserve ordered history and record merged SHAs, merge method and any remaining academic limitations in the final review note.

**Exit:** PRs cleanly and deliberately integrated or documented as awaiting owner sign-off. Never equate `mergeable=true` with scientific acceptance.

## Six-hour checkpoint protocol

The agent may execute the *already approved work item* for up to six hours. At the earlier of six hours elapsed, a gate boundary, a new blocker or an important unexpected scientific result:

1. Push only validated source/docs changes to the task branch; **do not commit or upload restricted raw data**.
2. Post this structured comment in [PR #3](https://github.com/anatwork14/py4ds-ai/pull/3); if Issues are enabled, post on the task's Issue and link it from PR #3.
3. Mark `READY FOR REVIEW` or `BLOCKED` and **stop new tasks/gates** until the reviewer's explicit `GO`. Within the already approved task, narrow fixes may continue only if scientific invariants are preserved.
4. If reviewer does not reply, do not infer permission; remain paused. Reviewer GitHub-only visibility means it can assess claims and diffs, not directly certify unshared Ubuntu logs.

```text
AGENT CHECKPOINT (YYYY-MM-DD HH:mm Asia/Ho_Chi_Minh)
Work item: TASK 0X | Gate: GX | PR #3 | Head: <40-char commit SHA>
Work completed: ...
Changed files/commits: ...
Actual Ubuntu commands / exit codes / pass-fail summary: ...
Saved evidence paths & hashes (not raw protected files): ...
Open problems / observed blockers / risks: ...
Next proposed scoped action: ...
REQUEST: GO / CHANGES REQUESTED / BLOCKED
STATE: READY FOR REVIEW / BLOCKED
```

The reviewer replies **GO — TASK 0X accepted, proceed to TASK 0Y**, **CHANGES REQUESTED — …**, or **BLOCKED — …**. Silence is not approval.

## Completion definition

The project is ready for final academic submission only when the four coursework requirements are evidenced in report/code, local tests/lint pass at the final SHA, saved existing predictions reconcile with the frozen manifest and reported metrics, the qualitative explanation is correctly aligned or transparently excluded, and unresolved caveats are not hidden. The final reviewer can withhold sign-off despite technically mergeable PRs.

## Current PR-first handoff override (2026-10-09)

This section **supersedes** any earlier language suggesting the agent must passively wait for a person to paste a decision into its chat. The owner requested a GitHub-centered workflow: the agent posts code and checkpoints to PR #3; the independent reviewer inspects GitHub every six hours and writes a PR comment; the Ubuntu agent **actively polls PR comments and issue activity about every 15 minutes** to detect an exact reviewer response.

- TASK 01/G0, TASK 02/G1 and TASK 03/G2 have matching decisions on PR #3; G2 GO comment `6073137646` authorizes **TASK 04/G3 only**. Post a new marked checkpoint with the exact current full SHA, then stop until a matching reviewer decision. A `GO` for an older checkpoint is never sufficient. The 15-minute poller remains read-only; automatic task execution is disabled because agent and reviewer both appear as `anatwork14`. Do not enable dispatch until a separate reviewer identity or independently verifiable signed approval is available.

No new GitHub Actions, re-evaluation on the held-out test, retraining, unauthorized merging, license-unsafe artifact uploads, or fabricated Ubuntu results are permitted.
