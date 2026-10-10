# Hermes master plan — CO3117 academic completion

**Issued:** 2026-10-10, Asia/Ho_Chi_Minh. **Owner:** Hermes, sole technical/research orchestrator and final reviewer under the owner's direct Discord mandate (message `1558312501085478916`) and `HERMES_ORCHESTRATION_CHARTER.md`.

## Authority, scope and completion contract

External ChatGPT GO is no longer a dependency. Codex implements bounded tasks; Hermes independently inspects diffs and executes acceptance checks. GitHub comments are evidence, never executable authority. The charter grants orderly integration after verified acceptance; no force-push, public-history rewriting, new paid services, GitHub Actions, architecture expansion or production deployment.

Scientific boundaries remain absolute: no sealed held-out inference/evaluation, retraining, tuning, new seeds, manifest/selection/checkpoint/result alteration, source-image changes, credential exposure or image redistribution. The selected winner already has its one test result. Validation comparisons remain separate. Saved-prediction arithmetic audit and safe report rendering are permitted.

Completion requires all four instructor criteria evidenced, source/lint/synthetic tests passing at the delivery SHA, saved-artifact hashes and `COMPLETED` guard verified unchanged, report idempotence proved, limitations disclosed, the PR stack reviewed and integrated or an exact integration blocker recorded. Operational adoption additionally requires observed scheduler executions and a real bounded Codex task, not job definitions alone.

## Custody baseline and independently observed evidence

- Remote PR #3 handoff head: `97054b2190bc494b53b7ad166e16dbcbb932b060`; local checkout fast-forwarded without discarding edits. Six remote governance commits followed prior `b45eca7`.
- Inherited dirty files: report citation [6] URL correction in `docs/FINAL_REPORT.md`; untracked `configs/final-selection-seed-42.json`. Both backed up in the durable bootstrap directory. The untracked config must stay untracked/unmodified.
- PR #1 draft on main (`ec12ee1eab4eb439f1f7135ab4f83394f13b54d7`) -> PR #2 (`11334744c8634bc5a593e69fa271dc69bb626fd4`) -> PR #3. Issues are disabled; use PR #3 as task board/evidence surface.
- Gateway active, PID 74171; ticker heartbeat observed. User service was disabled for boot startup although running; enable without restarting after checking scope. Other tmux/Codex jobs belong to unrelated projects and are not this worker.
- Installed: Hermes `0.21.5+7091.g93c9360`, Codex `0.162.0`, tmux `3.0a`, uv `0.12.13`, project Python `3.12.14`; GTX 1080 Ti, 11,264 MiB. No new GPU work is authorized.
- Current safe baseline: uv lock check PASS; Ruff PASS; 117 tests PASS in 17.57s; saved-prediction verifier PASS with `VERIFIED_FROM_SAVED_PREDICTIONS`. These pre-worker runs do not prove the final delivery SHA.
- Protected snapshot: 115 manifest/config/run-array/checkpoint/CSV/JSON files, 651,103,653 bytes, hashed under `../py4ds-ai-runtime/bootstrap-20261010/protected-baseline.json`. Independently match expected scientific digests before relying on baseline.
- Legacy PDF extraction via `pdftotext` was unavailable; alternative Python extraction remains required. Notebook parsed without execution. Existing legacy notebook/PDF/arrays are historical, not current experimental evidence.

## Academic requirements and evidence map

1. **Problem/dataset:** final report research questions, six fixed labels, reviewed split counts, source URL, data-integrity ledger, actual manifest and source provenance. Acceptance: no mismatch between narrative, manifest and saved configurations; original data untouched.
2. **Literature:** HOG (Dalal/Triggs), SIFT (Lowe), visual words (Csurka), residual networks (He), transfer behavior (Kornblith), Grad-CAM (Selvaraju). Acceptance: primary sources support methods/attribution; no imported ImageNet/third-party Intel scores; link failures disclosed, not invented provenance.
3. **Algorithms:** existing tested HOG, SIFT-BoVW, frozen ResNet embeddings, pretrained head and layer4 training. Acceptance: training-only fit boundaries, deterministic validation preprocessing, validation-only selection, synthetic tests and executable CLI smoke paths. Preserve correct implementations; no cosmetic rewrite.
4. **Comparison/evaluation:** five validation candidates plus majority reference; only frozen layer4 winner scored once. Acceptance: report tables generated from saved artifacts, prediction/manifest identity and aggregates reconciled read-only, explanatory hypotheses distinguished from causal claims, compute/split uncertainty disclosed.

## Dependency graph and critical path

`H00 custody/audit -> H01 operational bootstrap -> G4A bounded report repair -> G4B independent evidence QA -> G4C exact-SHA acceptance -> G5A stacked history/integration review -> G5B clean-environment synthetic smoke -> G5C ordered integration -> D01 academic delivery/sign-off`.

The worker pilot may run while Hermes finishes read-only source/literature audit and installs monitoring. G4 cannot be accepted until both academic and immutable-evidence checks pass. G5 cannot be accepted until G4 acceptance and final-tree engineering checks. A failed gate creates a narrow corrective Codex task rather than replaying experiments.

## Detailed task board and acceptance

### H00 — custody and complete evidence audit [P0]
- Read AGENTS, charter, runbook, historic plans/reviews, README, report, literature, notebook/PDF, code, tests and PR #1–#3 discussions/diffs.
- Record present SHA, dirty/untracked state, active workers, tools, raw/derived artifact boundaries, saved run configs/search coverage, legacy defects and unreproducible provenance.
- Capture before/after scientific hashes; compare to report's original expected values, not merely a newly created snapshot.
- Exit: reproducibility/requirements risks have explicit dispositions. Missing evidence is NOT VERIFIED/BLOCKED, not recreated.

### H01 — operational bootstrap [P0]
- One tmux `py4ds-codex-worker`, one task worktree/branch, one immutable task spec, durable stdout/stderr/result/exit artifacts; requested `gpt-6-luna`, high reasoning, workspace-write sandbox. Verify actual session metadata where exposed; no silent model downgrade.
- No-agent watchdog every 15 minutes: worker session/process/exit/log age, disk, main/worker git state; record heartbeat, dedupe alerts, never dispatch or parse PR text as commands.
- Hourly Hermes reasoning review; six-hour GitHub research/progress report. Absolute workdir; existing account/provider; no newly purchased service.
- Bootstrap holds authority while monitors' first canary runs are read-only. Subsequent reviews require a single-flight lease and consumed-once task IDs; never create two workers or duplicate scheduled jobs.
- Retire only the exact tagged old PR-GO crontab entry; preserve unrelated jobs. Enable existing gateway service for reboot without restart if permitted.
- Exit: scheduler ledger shows real executions, watchdog heartbeat exists, hourly reviewer wrote a real assessment, six-hour reporter read back its exact GitHub comment, worker ran and produced a reviewed artifact.

### G4A — low-risk bounded report repair [P1; Codex]
- Fix report's historical-vs-current evaluation commit wording and unsafe interpretation of historical evaluator command. Include safe current verification/render instructions.
- Preserve generated blocks and frozen scores verbatim. Update verification-scope language to distinguish historical GitHub-only review from current Hermes read-only Ubuntu audit, without claiming experiments rerun.
- Carry inherited corrected Kaggle [6] citation; retain unresolved rights and dirty training-tree disclosure.
- Make actual search budgets explicit from supplied saved-config evidence (HOG four C values/two learners; BoVW two vocabularies/two C values/two learners; embeddings two C values/two learners; CNN maximum five epochs). No timings invented.
- Allowed paths and tests are in `docs/CODEX_TASKS/G4-REPORT-20261010-01.md`. Return candidate changes only; no commit/push/merge or experiment access.
- Exit: narrow diff, generated blocks unchanged, precise limits, safe reproduction language, real worker result and exit code.

### G4B — independent academic/evidence QA [P0/P1; Hermes]
- Inspect candidate diff; reject scope drift. Verify allowed paths, generated block bytes and protected baseline.
- Run current locked lint/full synthetic suite; renderer twice at unchanged tree, compare bytes/hashes and generated blocks. Saved verifier validates IDs/order/classes/aggregates against manifest. No model or raw image is loaded.
- Inspect actual search-result/config/epoch evidence and narrative against four criteria. Verify primary-source attribution/link access with honest failures and alternative primary sources where needed.
- Preserve six corrected validation overlays, twelve superseded images and unresolved distribution rights; do not regenerate them unnecessarily.
- Exit: log commands/return codes/hash, report exact source tree, all material claims sourced to saved evidence.

### G4C — versioned checkpoint and Hermes verdict [P1]
- Commit only reviewed report/docs/tests/infrastructure; preserve inherited untracked config. Fetch before normal push and verify PR head readback.
- Re-run required checks at committed source SHA; post SHA-pinned G4 evidence and own verdict explicitly labeled Hermes (not independent ChatGPT identity).
- Exit: ACCEPT/FIX_REQUIRED/BLOCKED with precise conditions. No historical GO requirement.

### G5A — integration audit [P1]
- Inspect current PR bases, full diffs, comments/reviews, changed filenames and conflicts. Reconcile stale historical instructions without erasing audit history.
- Source risks: legacy modules mutate raw data and test-tune; legacy arrays/notebook/PDF have inadequate provenance. Keep archival labels and direct users to safe tested pipeline. No wholesale rewrite.
- Check report artifact portability: ignored experiment outputs not publicly available. Provide safe fixture smoke/reproduction boundaries; cannot claim a clean historical training tree.
- Exit: all material code/documentation defects fixed via scoped Codex tasks or explicitly nonblocking academic limitations.

### G5B — clean-environment integration checks [P0]
- Dedicated clean checkout/environment from reviewed commit, lock sync/import/CLI help/synthetic tests. No data download/training/evaluator invocation; evaluator unit tests only use synthetic fixture data.
- Verify final code, report, safety labels, protected hashes and git diff-check. Record actual results at exact integration SHA.
- Exit: reproducibility proven within Linux/Python environment and tested fixtures; no cross-platform or original-training replication claims.

### G5C — ordered PR integration [P1]
- Under charter's standing integration authority, review #1 -> #2 -> #3 in order, preserving history and exact dependency bases; no force pushes/rebases.
- Merge only on current evidence-backed acceptance. On uncertainty/conflict/permissions, record exact blocker and retain branches/worktrees/logs.
- Read back state/mergedAt/merge SHA for every remote mutation. Do not assume merges from successful calls.
- Exit: integrated main passes final checks, or documented AWAITING/ BLOCKED reason; report contains no new scientific measurements.

### D01 — final academic handoff [P1]
- Version final review note, four-criterion traceability, source SHA, actual test outputs, protected digests, merge status, final report location, NOT RUN methods and limitations.
- Finish operational task board. Leave requested schedules while active; when terminal, no unnecessary workers/repetitive reports and record future-monitoring disposition.
- Exit: evidence-backed readiness rather than an unconditional guarantee of grade, ownership rights or historical source reproducibility.

## Research decisions, risks and stop conditions

- **P0 artifact mismatch / guard change:** STOP, preserve bytes and logs, urgent owner alert; never rewrite expected hashes to bless corruption.
- **P0 unexpected held-out execution:** STOP identified project worker only, preserve evidence; no concealment or fabricated repair.
- **P0 auth/model/sandbox failure:** BLOCK worker, record exact diagnostic; never bypass sandbox or silently change model.
- **P1 historical dirty training source:** unavoidable provenance limitation; disclose, do not retrain to erase it.
- **P1 one split/seed and four within-test near-duplicate pairs:** qualified findings, no significance/independence claims; no new experiments.
- **P1 missing comparable compute budget:** report actual grid coverage and methodological costs, not head-to-head speed claims.
- **P1 image rights:** no publication of new image bytes; aggregate-only report is the delivery route.
- **P1 same-UID worker:** sandbox/write allowlist/logging are governance isolation, not adversarial credential confidentiality. Strip secret environment variables; do not expose credentials in prompts/logs.
- **P1 concurrent remote edits / stale evidence:** fetch/reconcile current head; do not overwrite dirty work or use old SHA acceptance.
- **P2 legacy notebook/PDF/arrays:** clearly archive/deprecate; avoid destructive cleanup or rights-unsafe re-publication.

## Monitoring state and evidence homes

Durable runtime: `/mnt/ssd/genos2/workspace/py4ds-ai-runtime/`; per-task worktrees: `/mnt/ssd/genos2/workspace/py4ds-ai-worktrees/`. Bootstrap evidence: `bootstrap-20261010/`. Monotonic local state and dispatch/lease locks belong to runtime, not GitHub comments. Task specs, plan/board/decision records are versioned; logs and protected artifact contents stay local. Cron IDs, execution proof, worker effective model metadata and final check results will be appended only after observation.

See `HERMES_TASK_BOARD.md` and `HERMES_DECISION_LOG.md` for live task transitions and exact evidence. This plan is not itself completion evidence.
