# CO3117 academic project — agent/reviewer progress

**Last updated:** 2026-10-09, Asia/Ho_Chi_Minh
**Status:** FINAL REVIEW OPEN — G3 evidence ready; waiting for reviewer GO.
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

| Task | Priority | Gate | State | Last proven execution SHA / evidence |
|---|---|---|---|---|
| TASK 01: Ubuntu worktree/artifact inventory | P0 | G0 | **NOT STARTED** | — |
| TASK 02: locked uv, Ruff, pytest, diff checks | P0 | G1 | **NOT STARTED** | — |
| TASK 03: frozen manifest/prediction/metric audit | P0 | G2 | **NOT STARTED** | — |
| TASK 04: Grad-CAM validation-only explanation QA | P1 | G3 | **NOT STARTED** | — |
| TASK 05: rendered report and literature/assignment QA | P1 | G4 | **NOT STARTED** | — |
| TASK 06: PR #1 → #2 → #3 merge readiness | P1 | G5 | **NOT STARTED** | — |

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

## PR-first communication upgrade (2026-10-09)

- [x] [`docs/PR_REVIEW_HANDOFF.md`](docs/PR_REVIEW_HANDOFF.md) defines SHA-pinned agent checkpoints and reviewer GO/CHANGES_REQUESTED/BLOCKED comments on PR #3.
- [x] `scripts/check_pr_review.py` and `tests/test_pr_review_handoff.py` added for read-only GitHub polling and anti-stale-review checks.
- [x] TASK 01/G0 GO **was posted** on [PR #3](https://github.com/anatwork14/py4ds-ai/pull/3#issuecomment-6071463946), authorizing TASK 02/G1. This supersedes the older TASK 01 “NOT STARTED” status above; the historic table is retained as an unaudited execution snapshot.
- [ ] TASK 02/G1 Ubuntu execution checkpoint posted with the new machine-readable marker and a full SHA.
- [ ] Agent confirms the **15-minute PR/Issue polling loop or server timer is actually running**. A script checked into Git is *not* proof a background process exists.
- [ ] Reviewer posts an exact TASK 02 checkpoint-matching decision; subsequent tasks remain gated.

## Current execution status — 2026-10-09

This section supersedes the historical gate table and interim G0/G1 handoff snapshot above.

- TASK 01/G0, TASK 02/G1 and TASK 03/G2 have matching PR decisions; the G2 GO is [comment 6073137646](https://github.com/anatwork14/py4ds-ai/pull/3#issuecomment-6073137646) at head `c77c521688ae67bbc3695a5966ba10ceb73e76c9`.
- TASK 04/G3 is authorized and its local evidence is ready for a SHA-pinned PR checkpoint. The focused geometry tests passed; old figures are marked superseded; six replacement validation-only overlays were generated and audited. No test images, training, or final evaluator were used.
- The 15-minute PR poller is read-only. Automatic task execution remains disabled because agent and reviewer both appear as `anatwork14`; do not dispatch from comments until a separate reviewer identity or independently verifiable signed approval is available.
- After the G3 checkpoint is posted, wait for a matching reviewer GO before TASK 05/G4. Do not merge PRs or alter frozen results.
