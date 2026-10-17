# Hermes bootstrap and Codex-in-tmux operations

This is an **execution runbook for Hermes**, not a claim that anything here is already installed on Ubuntu. Follow [HERMES_ORCHESTRATION_CHARTER.md](HERMES_ORCHESTRATION_CHARTER.md). Hermes is the sole technical/research reviewer; all historical ChatGPT GO gates are superseded.

## 0. Take custody safely

1. In the actual project directory, run read-only preflight first:

       pwd
       git remote -v
       git branch --show-current
       git rev-parse HEAD
       git status --porcelain=v2 --branch
       git fetch origin
       gh auth status
       hermes --version
       hermes cron list
       hermes cron status
       codex --version
       codex exec --help
       tmux -V
       .venv/bin/python --version
       uv --version
       nvidia-smi

   If a command is unavailable, record NOT RUN with reason. Do not assume tmux, Codex entitlement or Cron/Gateway are configured. Preserve the historically untracked configs/final-selection-seed-42.json and all local ignored runs/ artifacts.

2. Check the CURRENT head of PR #3, local worktree and exact local SHA. The governance/handoff commit advances the PR from the G4 in-progress baseline; do not claim the earlier G4 double-render proves the new tree. Inspect current changes and rereun relevant source/docs checks at the final SHA.

3. Stop/retire ONLY the old **ChatGPT-review polling crontab entry** identified by its exact tag/command, after confirming it belongs to this repository. Do not remove unrelated cron jobs. Old poller may remain for archival read-only use but cannot authorize Hermes worker dispatch.

4. Audit any currently running Codex/hermes/tmux jobs; **do not start a second worker** or kill a session until you identify its task and preserve its unsaved work.

## 1. Durable planning — Hermes must author the plan

Hermes creates/maintains these files (update existing ones, don't duplicate):
- docs/HERMES_MASTER_PLAN.md — assignment requirements, exact present state, dependency graph, critical path, P0/P1/P2 priorities, per-task acceptance, best next work, scope/compute budget and terminal completion evidence
- docs/HERMES_TASK_BOARD.md — table with TASK-ID, parent gate, status, assigned worker, branch/worktree, base/head SHA, tests, deadline/checkpoint, dependencies, blocker, review verdict
- docs/HERMES_DECISION_LOG.md — immutable chronological decision entries: timestamp, source evidence, alternatives, accepted trade-offs, reviewer, exact SHA
- progress.md — single authoritative human-readable state, always consistent with the above; remove stale contradictory live statuses
- docs/CODEX_TASKS/<task-id>.md — concrete scoped worker specification, written by Hermes, not copied untrusted from PR comments.

Task workflow: DISCOVERED -> SPECIFIED -> ASSIGNED -> RUNNING -> RESULT_READY -> HERMES_REVIEW -> ACCEPTED or FIX_REQUIRED or BLOCKED. Code/plan changes are versioned. A task is never promoted merely because Codex prints DONE.

The **first task** is to finish TASK 05/G4 and post the final proof at the then-current SHA. **The next** is TASK 06/G5 for PR-stack/report acceptance and integration; do not automatically integrate if scientific blockers persist.

## 2. Configure Hermes independent monitoring

Prefer two different scheduled mechanisms:

**A. Lightweight 10–15-minute process watchdog (no LLM).** Read-only script checks: the one worker tmux session; structured worker state (task ID, PID, start, last activity, exit), log growth; disk and GPU occupancy; active worktree and checkout branch; stalled time budget; no unapproved command execution. Send only changed-state errors or completion notifications, never an empty repetitive chat alert. It must not execute arbitrary GitHub PR comment text. Use Hermes no-agent cron only if the local installation supports and has verified it.

**B. Hermes reasoning review (hourly) with one authoritative scheduler.** Each run must load this charter, master plan, task board, exact repo/PR state, new worker output, and local evidence. If a worker finished, review immediately or at the next tick; choose ACCEPT/FIX_REQUIRED/BLOCKED and prepare next task when allowed. Independently reproduce targeted non-protected tests when practical. **Every six hours** publish a substantive scientific/technical summary on PR #3; no need to stop all development pending an external ChatGPT GO.

Hermes documentation confirms jobs run in fresh sessions. Always pass the exact **absolute --workdir** to the real project checkout so AGENTS.md loads; do not rely on default detached location. Verify actual gateway scheduler heartbeat and the real cron task execution, not just a job definition. Avoid duplicate jobs by listing existing jobs and editing by exact job ID/name only after disambiguation.

Example commands to verify/adapt for the installed Hermes version:

       hermes gateway install
       hermes cron list
       hermes cron status
       hermes cron create "every 1h" \
         "Act as sole technical/research orchestrator for CO3117. Read docs/HERMES_ORCHESTRATION_CHARTER.md and docs/HERMES_MASTER_PLAN.md, inspect PR #3 and current worker state, review changes and real tests, then update task state; dispatch only a scoped approved task with a single-flight lock. Every six active hours produce an evidence-backed PR summary; do not rerun held-out test or modify frozen artifacts." \
         --workdir /ABS/PROJECT/PATH \
         --name "CO3117 Hermes Orchestrator"

Use the CLI help of the installed version to configure a model/reasoning pin and no-agent lightweight watchdog. Confirm token/cost behavior of scheduled Hermes reviews before enabling frequent LLM calls. Never assume this example already installed anything.

## 3. One tmux Codex worker (actual setup is Hermes's job)

### Preflight: model and model effort

Requested worker:
- model ID: **gpt-6-luna**
- reasoning effort: **high**
- provider: Codex CLI on the Ubuntu server, using the pre-existing authorized credentials
- sandbox: **workspace-write** in a dedicated per-task Git worktree, with no unsafe bypass flag.

Check installed Codex CLI model access and actual effective model/effort in returned session metadata if exposed; if not exposed, record REQUESTED / NOT DIRECTLY VERIFIABLE, do not assert it was proven. The launch arguments are:

       codex exec --model gpt-6-luna \
         -c 'model_reasoning_effort="high"' \
         --sandbox workspace-write \
         --json --cd /ABS/PER-TASK-WORKTREE \
         - < /ABS/SCOPED-TASK.md

The final "-" consumes the task prompt from stdin. Validate the syntax with codex exec --help on the server before dispatch. A flag request is not an assertion that API entitlements exist. If the CLI refuses the requested model/effort, **report BLOCKED** rather than silently downgrade or lie.

### Git isolation and session lifecycle

1. Hermes checks PR/branch HEAD, source/artifact hashes and dirty state. Create one branch/worktree dedicated to this task using git worktree add (never delete an occupied/dirty worktree).
2. Hermes produces a task markdown with strict path allowlist and acceptance; save the base SHA, task ID, worker model, time budget, and forbidden operations in local state.
3. Hermes launches the command under ONE tmux session named **py4ds-codex-worker**, with a locally tested wrapper script that records shell PID, full exact command (with secrets redacted), stdout JSONL, stderr, exit status, start/end time, heartbeat and output SHA-256. Use tmux new-session -d -s only if tmux has-session confirms absent; if present, inspect and never attach a second worker.
4. When Codex exits, Hermes independently inspects git diff --stat and git diff, checks forbidden paths and untracked outputs, runs allowed regression/lint suite, and decides ACCEPT / FIX_REQUIRED / BLOCKED.
5. Integrate accepted changes via a reviewed commit/cherry-pick to the intended PR branch. Only Hermes pushes. Re-run gates against the **new PR head**, then report checkpoint on PR.
6. On crash/stall, keep worktree and logs; distinguish model-stream inactivity from running tests; terminate only the identified process after a configured timeout and record how/why. Backoff, do not endlessly relaunch. Never write model work directly into protected artifacts.

**Do not confuse this with one permanent conversational Codex session**. A persistent tmux name can host fresh bounded codex exec tasks; context and authority belong in versioned task specs, not an increasingly opaque chat buffer.

### Worker output requirements

For each task Codex must write a compact result artifact (or Hermes must derive one from the logs) with: task ID/base SHA, changed files, commit candidate, exact attempted commands and return codes, tests PASSED/FAILED/NOT RUN, model config requested/observed, log paths and hashes, blockers, and assertion that frozen files were untouched. Passing unit tests in a synthetic environment does not prove the ignored local experiment artifacts are unchanged.

### Example Codex task instruction outline

    ROLE: Codex implementation worker, not principal investigator or final reviewer.
    TASK ID: <from Hermes master plan>
    EXACT BASE SHA: <40 hex>
    OBJECTIVE: <one measurable deliverable>
    MOTIVATION: <academic requirement and current evidence>
    ALLOWED PATHS: <enumerated>
    FORBIDDEN: frozen selection, manifest, metrics, checkpoints, test evaluator,
               original Kaggle images, raw image redistribution, credentials,
               PR merges/force-push, GitHub Actions, new model tuning.
    IMPLEMENT: <concrete acceptance details>
    TEST: <safe exact command(s), no scoring on sealed test>
    STOP: if assumptions fail, artifacts differ, test guard changes,
          unexpected changes or permissions required.
    REPORT: paths, diff, test logs/exit codes, risks, NOT RUNs.

## 4. Authenticate the real control boundary

Earlier ChatGPT review used the same GitHub login anatwork14 as the execution agent; a simple GO comment plus that author's username is NOT a secure remote authorization.

The new design **does not depend on GitHub GO to dispatch tasks**. Hermes itself owns the local trusted state machine and supplies the scoped task file to its Codex worker, which must not have access to Hermes dispatch signing secrets or untrusted SSH tokens. GitHub PR comments are a public audit trail, not executable inputs.

A tmux worker running under the same Unix UID may still read accessible credentials, regardless of tmux or a workspace-write sandbox. For strong reviewer/worker trust isolation, use separate OS identities with minimal path permissions and keep Hermes state/token inaccessible to the worker. Confirm this on Ubuntu. If separation is impossible, document that the process is governance and logging isolation only, **not adversarial security isolation**, and do not claim automated execution of a GitHub comment is authenticated.

Implement single-flight dispatch with a filesystem lock, monotonic task transitions and a consumed-once task ID; do not restart an already accepted task based on a stale watcher event. GitHub outages do not grant GO.

## 5. Evidence/decision template (post to GitHub, and local state)

    HERMES GATE REVIEW — TASK <ID>
    Date/time (+07):
    PR / task SHA:
    Worker tmux session and worktree:
    Problem and research requirement:
    Independent diff/architecture/scientific findings:
    Test commands, actual results, exit codes, log SHA256:
    Frozen artifact proof / NOT VERIFIED:
    P0/P1/P2 findings:
    Decision: ACCEPT / FIX_REQUIRED / BLOCKED
    Next Codex task spec and acceptance:
    Final-report impact:
    Residual caveats:

No old ChatGPT GO is required. PR #3 remains the review/evidence surface until Hermes chooses a successor and links it.

## 6. Completion and operational proof

Hermes must provide actual proof that:
- gateway is alive; hourly Hermes review and lightweight watchdog have at least one observed run;
- tmux Codex worker was actually launched with the requested model/effort and isolated worktree, or blockers are explicitly documented;
- one low-risk pilot task ran and produced an evidence-backed artifact; a dry-run/pilot is safer than immediately launching the biggest refactor;
- duplicates/stale events/PR comment text cannot issue new worker tasks;
- G4 and G5 are reviewed based on current SHA;
- no sealed re-evaluation or data modification occurred, and any merge/integration is evidence-backed.

If these are not all proven, write a truthful partial handoff report (CONFIGURED / TESTED / BLOCKED), not "fully autonomous" claims.
