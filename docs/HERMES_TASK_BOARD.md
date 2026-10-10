# Hermes live task board

Authority: direct owner mandate `1558312501085478916`; Hermes charter. Updated 2026-10-10. Runtime transitions are independent of untrusted GitHub comments.

| Task | Priority/gate | Dependencies | State | Executor / branch | Acceptance / evidence |
|---|---|---|---|---|---|
| H00 custody/audit | P0 | none | IN PROGRESS | Hermes / implementation/phase1-data-manifest | Handoff 97054b2 reconciled; protected baseline; 117 tests, saved verifier passed; remaining full source/legacy audit |
| H01 monitoring/worker bootstrap | P0 | H00 custody | SPECIFIED | Hermes | Actual gateway tick, watchdog/hourly/six-hour execution and bounded worker proof |
| G4-REPORT-20261010-01 | P1/G4A | H00 custody | SPECIFIED | Codex / agent/g4-report-20261010-01 | Report narrative correction only; no generated score edits; no commit/push; 20-minute timeout |
| G4B academic/evidence review | P0/G4 | G4A + H00 | PENDING | Hermes | Independent diff, current tests, two identical renders, original expected digests |
| G4C checkpoint/acceptance | P1/G4 | G4B | PENDING | Hermes | Exact source SHA, verified PR comment and own ACCEPT/FIX_REQUIRED/BLOCKED |
| G5A PR integration audit | P1/G5 | G4C ACCEPT | PENDING | Hermes | #1–#3 current diff/context/merge bases and risk dispositions |
| G5B clean-environment smoke | P0/G5 | G5A | PENDING | Codex implementation when repairs needed; Hermes checks | Locked install/import/CLI/synthetic suite, no real-data experiments |
| G5C ordered integration | P1/G5 | G5A + G5B ACCEPT | PENDING | Hermes | Charter standing authority; #1 -> #2 -> #3; readback each merge |
| D01 final readiness note | P1 | G5 | PENDING | Hermes | Four criteria, current SHA/tests, hashes, limitations and operational disposition |

Historical G0–G3 remain accepted; no completed experiment is repeated. Frozen one-test result, raw data, selected checkpoint, image rights and no-CI boundaries are binding. Full per-task contracts and risks: `HERMES_MASTER_PLAN.md`.
