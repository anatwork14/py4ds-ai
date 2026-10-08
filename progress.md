# CO3117 academic project — agent/reviewer progress

**Last updated:** 2026-10-08, Asia/Ho_Chi_Minh  
**Status:** FINAL REVIEW OPEN — awaiting Ubuntu evidence.  
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
- [ ] Ubuntu auditor independently verified the held-out report numbers against saved `predictions_test.csv`, `metrics.json`, checkpoint SHA-256 and locked manifest.
- [ ] Evidence audit confirmed evaluation guard remains `COMPLETED`.
- [ ] Old misaligned validation Grad-CAM images clearly excluded or regenerated using corrected transforms.

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
