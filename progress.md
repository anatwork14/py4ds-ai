# CO3117 academic project — agent/reviewer progress

> **Current governance, 2026-10-10:** Hermes is the sole technical/research reviewer and final decision maker; Codex (GPT-6 Luna high) is the implementation worker Hermes must launch and supervise inside tmux. External ChatGPT GO and old 15-minute PR approval poller are retired as authorization mechanisms. Read [Hermes charter](docs/HERMES_ORCHESTRATION_CHARTER.md) and [bootstrap runbook](docs/HERMES_BOOTSTRAP_RUNBOOK.md). **GitHub edits here do not prove Hermes gateway, scheduled jobs or tmux Codex worker are installed/running.** Hermes must verify real server state and report proof. TASK 05/G4 was not fully signed off at handoff; current HEAD must be rechecked after governance commits.

**Last updated:** 2026-10-09, Asia/Ho_Chi_Minh
**Status:** HERMES AUTHORITY TRANSFER IN PROGRESS — G4 unaccepted at handoff; no claim that Ubuntu Hermes/Codex processes have started.
**Primary PR:** [#3](https://github.com/anatwork14/py4ds-ai/pull/3)  
**Task definitions:** [docs/AGENT_EXECUTION_PLAN.md](docs/AGENT_EXECUTION_PLAN.md)  
**Final-review criteria:** [docs/FINAL_REVIEW_2026-10-08.md](docs/FINAL_REVIEW_2026-10-08.md)

This is an evidence ledger, **not** a claim that test scripts ran in the latest commit. Only check an item after attaching a real SHA, command output, and reviewer GO in PR #3 / its future linked GitHub Issue.

## Completed in GitHub (reviewer-verified source changes, NOT Ubuntu test runs)

- [x] Problem and experiment scope reviewed against the four CO3117 instructor requirements.
- [x] Read-only source review and code modifications: metric consistency checks, Grad-CAM crop alignment, saved-prediction audit and synthetic regression tests.
- [x] `docs/FINAL_REPORT.md` expanded to include problem, literature synthesis, algorithms, tuning and validation interpretation.
- [x] `docs/FINAL_REVIEW_2026-10-08.md` records the local acceptance criteria.
- [x] `docs/AGENT_EXECUTION_PLAN.md` and `AGENTS.md` specify mandatory agent gates and six-hour checkpoints.
- [x] Instructions posted to PR #3.
- [ ] GitHub Issues enabled (blocked by repo setting; GitHub API returned HTTP 410).

## Required execution gates (must execute on Ubuntu; each gate needs reviewer GO)

| Task | Priority | Gate | Current state | Evidence |
|---|---|---|---|---|
| TASK 01: Ubuntu worktree/artifact inventory | P0 | G0 | **ACCEPTED** | G0 checkpoint and reviewer decision on PR #3 |
| TASK 02: locked uv, Ruff, pytest, diff checks | P0 | G1 | **ACCEPTED** | `c77c521688ae67bbc3695a5966ba10ceb73e76c9`; 117 tests passed |
| TASK 03: frozen manifest/prediction/metric audit | P0 | G2 | **ACCEPTED** | Read-only saved-prediction verification at `c77c521`; no new scoring |
| TASK 04: Grad-CAM validation-only explanation QA | P1 | G3 | **ACCEPTED** | `0b98d73113f6b38a5e5e904a47f41e2c60d8f2e6`; six local replacement overlays, 12 old overlays marked superseded |
| TASK 05: rendered report and literature/assignment QA | P1 | G4 | **IN PROGRESS — DOCS UPDATED; VERIFY/COMMIT PENDING** | G3 reviewer GO; double-render identical; final validation/checkpoint pending |
| TASK 06: PR #1 → #2 → #3 integration review | P1 | G5 | **BLOCKED — WAIT FOR G4 GO** | Do not start before a fresh matching reviewer decision |

## Protected final-test evidence

- [x] Repository documents one already-completed one-time final-test evaluation for the selected ResNet18-layer4 model.
- [x] Ubuntu auditor independently verified the held-out report numbers against saved `predictions_test.csv`, `metrics.json`, checkpoint SHA-256 and locked manifest.
- [x] Evidence audit confirmed evaluation guard remains `COMPLETED`.
- [x] Old misaligned validation Grad-CAM images marked superseded; corrected validation-only overlays regenerated from the frozen checkpoint and dimensions verified. See [`docs/GRADCAM_QA.md`](docs/GRADCAM_QA.md).

**Do not rerun the held-out evaluator to fill any of the above checkboxes.** Saved-prediction inspection is read-only and does not score images.

## Checkpoint log

Add a brief log line at each agent handoff and link the primary GitHub comment/Issue, e.g.:

| Time (+07) | Task | Head SHA | Agent evidence | Reviewer decision |
|---|---|---|---|---|
| 2026-10-08 | Planning | See PR #3 | Source plan + direct GitHub instructions | Await TASK 01 checkpoint |

## Stop and approval conditions

- **P0 stop:** any unknown/changed manifest, mismatched checkpoint/hash, prior result tampering, inconsistent test evidence, unexpected test scoring or failing code check. Preserve artifacts, comment **BLOCKED** and request review.
- **P1 stop:** inaccurate citations, misleading report claims, unverified image rights when intending redistribution, PR-stack merge conflicts, stale review evidence or geometrically invalid figure.
- **Final sign-off:** all tasks evidence-backed, all required Ubuntu commands pass at the final SHA, no active scientific/report blocker, and reviewer explicitly approves the ordered PR integration.

**Academic delivery does not require GitHub Actions or a production deployment.**

## PR-first communication record (historical checkpoints; current gate is below)

- [x] `docs/PR_REVIEW_HANDOFF.md` defines SHA-pinned agent checkpoints and reviewer decisions on PR #3.
- [x] `scripts/check_pr_review.py` and `tests/test_pr_review_handoff.py` implement read-only review polling and stale-head checks.
- [x] TASK 01/G0, TASK 02/G1 and TASK 03/G2 were completed and accepted; see the linked PR checkpoint comments.
- [x] TASK 04/G3 was accepted by reviewer decision [6073541055](https://github.com/anatwork14/py4ds-ai/pull/3#issuecomment-6073541055), authorizing TASK 05/G4 only.
- [ ] G4 checkpoint not yet posted. Update the read-only 15-minute poller to the new G4 checkpoint after posting; automatic task execution remains disabled because the reviewer and agent share GitHub identity `anatwork14`.

## Current execution status — 2026-10-09

This is the single authoritative live status; historical checkpoint comments remain linked in PR #3.

- TASK 01/G0, TASK 02/G1 and TASK 03/G2 were completed and accepted. TASK 04/G3 was accepted by reviewer comment [6073541055](https://github.com/anatwork14/py4ds-ai/pull/3#issuecomment-6073541055), authorizing TASK 05/G4 only.
- G4 report rendering ran twice at the unchanged G3 SHA; outputs were byte-identical (SHA-256 `8aa36d3d887ccd126a7e8c62b28fd450948ff68bbb7777539b85d230b4b104e4`). Citation numbering/authors were cross-checked against the literature-review entries, and the report now maps the four course criteria explicitly.
- Stale Grad-CAM text and old unchecked G1/G2 checklist states were corrected. The six corrected Grad-CAM image files remain local pending rights verification.
- After G4 commit/checkpoint, stop and wait for reviewer GO. TASK 06/G5 is not authorized yet. Never rerun final test evaluation or alter frozen results.


## Hermes transition verification (new authority; not yet executed on Ubuntu)
- [x] Hermes authority charter and bootstrap runbook authored on PR #3.
- [x] Legacy AGENTS, prior execution plan and PR handoff annotated as superseded for review authorization.
- [ ] Hermes confirms active repo HEAD, inherited G4 work/evidence and protected digests; creates detailed own master plan, task board and decision log.
- [ ] Hermes installs/verifies its own gateway and lightweight watchdog plus hourly review; attaches real run logs/heartbeats.
- [ ] Hermes launches one tmux Codex worker using requested gpt-6-luna / high in a scoped worktree; verifies model availability, PID, log, test and task boundaries.
- [ ] Hermes completes G4 at CURRENT SHA with report double-render and QA, then self-reviews before G5.
- [ ] Hermes independently handles G5 and accepts/repairs/merges stacked PRs only when scientifically justified; no new held-out evaluation.
- [ ] Old ChatGPT review automation and PR-comment GO poller are disabled/retired (verify separately; do not assume GitHub docs stop server cron).
