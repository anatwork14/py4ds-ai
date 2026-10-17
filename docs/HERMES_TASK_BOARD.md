# Hermes live task board

Authority: direct owner mandate `1558312501085478916`; Hermes charter. Updated 2026-10-10. Trusted local transitions never dispatch from GitHub comments.

| Task | Priority/gate | Dependencies | State | Executor / evidence |
|---|---|---|---|---|
| H00 custody/source/academic audit | P0 | none | ACCEPTED | Hermes; repository/PRs/code/notebook/PDF/source papers audited; protected snapshot matched |
| H01 monitoring/bootstrap | P0 | H00 | FIX_REQUIRED | Real gateway plus watchdog/hourly/report canaries completed; alert helper correction still under review |
| G4-REPORT-20261010-01 | P1/G4A | H00 | ACCEPTED | Codex gpt-6-luna/high/workspace-write exit 0; narrative-only candidate independently checked |
| G4B academic/evidence review | P0/G4 | G4A + H00 | ACCEPTED | Four generated blocks unchanged; original expected digests; double render; 117 fixture tests |
| G4C checkpoint/acceptance | P1/G4 | G4B | ACCEPTED | SHA `99dd935d04ee34f99c70be1544bfd661ecb1492e`; verified PR #3 comment `6093259139` |
| OPS-WATCHDOG-20261010-02 | P1/ops | H01 gap | FIX_REQUIRED | Codex exit 0; 13 tests. Hermes rejected completed-task false alarms and inadequate missing-PID test |
| OPS-FIX-20261010-03 | P1/ops | rejected candidate | RUNNING | Same single tmux slot, separate worktree, focused regression-first repairs; no commit/push |
| G5A PR integration audit | P1/G5 | G4 ACCEPT | IN PROGRESS | Full stack/diffs reviewed; legacy warning added; no new images/experiments |
| G5B clean-environment smoke | P0/G5 | G5A | BASELINE PASS; FINAL TREE PENDING | Clean locked environment at `99dd935`: 117 passed in 20.55s, Ruff; ten safe help CLIs exit 0 |
| G5C ordered integration | P1/G5 | current G5 checks | PENDING | Charter authority; normal #1 -> #2 -> #3 merges with fresh readback and bases |
| D01 readiness/sign-off | P1 | G5 | PENDING | Four criteria, final SHA/checks, scientific hashes, limits, operational disposition |

G0–G3 historical acceptances remain valid history. Frozen one-test result/data/checkpoint/rights/no-CI limits remain binding. Runtime evidence resides in the sibling `py4ds-ai-runtime`; full contracts are in `HERMES_MASTER_PLAN.md`. No task is accepted merely because a worker exits zero.
