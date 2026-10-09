# PR-first agent ↔ reviewer handoff (CO3117)

**Authoritative work surface:** [PR #3](https://github.com/anatwork14/py4ds-ai/pull/3), branch `implementation/phase1-data-manifest`.  
**Primary assignment:** [AGENT_EXECUTION_PLAN.md](AGENT_EXECUTION_PLAN.md) TASK 01–06.  
**Current permission:** G0/TASK 01 has GO in PR #3 comment [6071463946](https://github.com/anatwork14/py4ds-ai/pull/3#issuecomment-6071463946). **TASK 02/G1 is the currently authorized work**, subject to its own checkpoint/review.

## Division of labor: no direct chat is required for each handoff

- **Ubuntu execution agent:** reads the PR and any linked task issues, performs *one authorized task* locally, pushes source/test changes into the existing PR branch (or a specifically linked code PR), posts a **marked checkpoint comment**, then pauses development and polls for an exact marked reviewer decision.
- **GitHub reviewer:** runs a separate **scheduled six-hour GitHub-only review**, reads new commits/diffs/checkpoints and connected issues, posts a **marked decision on the authoritative PR**, and advises only within the assignment and available evidence.
- **Owner:** supervises protected decisions. The GitHub process does not give the reviewer access to ignored files or the server. A reported Ubuntu pass is not proof the reviewer executed Ubuntu commands.
- **GitHub Issues:** at initial setup they were disabled (API 410); when enabled, agents may track TASK 01–06 as Issues and must link those issues from the PR checkpoint. **All GO/CHANGES_REQUESTED/BLOCKED decisions for this PR still go on PR #3**, so there is one unambiguous approval channel. The poller additionally reads linked task-Issue activity for context.

**Important:** a ChatGPT scheduled review and an external Ubuntu agent are separate processes. Posting GO does not wake a stopped external agent by itself. The Ubuntu owner/agent must configure a live polling loop or a systemd/cron service capable of checking and dispatching its agent. Never report unattended monitoring as installed unless the server actually runs it.

## Durable PR checkpoint markers (required from TASK 02 onward)

The executing agent posts a normal PR comment through `gh pr comment 3 --body-file checkpoint.md`, with an exact marker at the start:

```md
<!-- CO3117_AGENT_CHECKPOINT v1 id=G1-<unique-identifier> sha=<FULL_40_HEX_HEAD_SHA> task=TASK-02 -->
### AGENT CHECKPOINT — TASK 02 / G1
Timestamp: YYYY-MM-DD HH:mm Asia/Ho_Chi_Minh
PR: #3
Checkpoint ID: G1-<unique-identifier>
Head SHA: <FULL_40_HEX_HEAD_SHA>
Commands and exit codes: ...
Test counts/failures and actual local log paths: ...
Commits and file diff summary: ...
Frozen artifact integrity: ...
Blockers and remaining questions: ...
Proposed next gate: TASK 03 / G2
STATE: READY FOR REVIEW / BLOCKED
```

The unique identifier can be `G1-YYYYMMDD-HHMM-<sha12>` (letters, digits, hyphens, underscores and periods only). Verify the full SHA with `git rev-parse HEAD` **after pushing** and ensure it equals the PR head. The checkpoint must pin exactly the code/version for which the tests were run; if source changes afterward, rerun affected checks and post a new checkpoint.

## Reviewer decisions and exact-SHA matching

A GO/rejection is posted **as a PR comment after the corresponding checkpoint**:

```md
<!-- CO3117_REVIEW_DECISION v1 checkpoint=G1-<unique-identifier> sha=<FULL_40_HEX_HEAD_SHA> verdict=GO -->
REVIEWER DECISION: GO — TASK 02 accepted. Authorized next gate: TASK 03 / G2.
Evidence reviewed: ... 
Limitations/remaining risks: ...
```

Supported verdicts are `GO`, `CHANGES_REQUESTED`, and `BLOCKED`. A rejected gate requires specific actionable findings. No approval can be inferred from silence, generic praise, comments on a different SHA, a mergeable PR, or an earlier GO (such as G0's existing approval).

**Parsing note:** labels and markers are coordination conventions, not cryptographic authentication. Reviewer and agent may appear as the same GitHub account (`anatwork14`); the agent must not post `CO3117_REVIEW_DECISION` markers itself. Use separate GitHub identities for stronger provenance if feasible.

## Recommended 15-minute Ubuntu polling

The repo includes **[scripts/check_pr_review.py](../scripts/check_pr_review.py)**, a read-only tool using the existing `gh` login. It checks the *latest checkpoint*, exact checkpoint ID, full SHA **against the current GitHub PR head**, latest matching reviewer verdict, and (when enabled) open CO3117 task-Issue comments.

After posting a TASK 02 checkpoint, run:

```bash
.venv/bin/python scripts/check_pr_review.py \
  --repo anatwork14/py4ds-ai --pr 3 \
  --checkpoint-id G1-REPLACE-WITH-YOURS \
  --head-sha REPLACE_WITH_EXACT_40_CHAR_HEAD_SHA \
  --task TASK-02
```

Exit statuses:

| Exit | Verdict | Allowed action |
|---|---|---|
| `0` | `GO` | Confirm the reviewer explicitly authorized the next task; update `progress.md` and resume only that task |
| `10` | Waiting, no checkpoint, or stale checkpoint | **Do not advance**; synchronize/poll again |
| `11` | `BLOCKED` | Preserve state and escalate the blocker |
| `12` | `CHANGES_REQUESTED` | Address only listed corrections in the same approved task; publish a new SHA-pinned checkpoint |
| `2` | GitHub/auth/API error | Stop; fix communication/auth; do not assume approval |

For an **active foreground agent session**, a simple 15-minute checking loop suffices:

```bash
while true; do
  .venv/bin/python scripts/check_pr_review.py \
    --checkpoint-id G1-REPLACE-WITH-YOURS \
    --head-sha REPLACE_WITH_EXACT_40_CHAR_HEAD_SHA \
    --task TASK-02
  result=$?
  if [ "$result" -eq 0 ] || [ "$result" -eq 2 ] || [ "$result" -eq 11 ] || [ "$result" -eq 12 ]; then
    break
  fi
  sleep 900
done
```

If the agent's chat/process terminates, this loop also terminates. For true unattended checking, **have the Ubuntu agent install/verify a user systemd timer or cron entry at 15-minute intervals**, keeping the current checkpoint ID/SHA in a state file. It may notify/resume the agent through an existing authorized runner, but must **not blindly launch arbitrary code or start the next task without a matching GO**. Do not install a new proprietary orchestration service.

Polling can be silent when no result changes. When the PR head advances, the poller returns `HEAD_MOVED` (exit 10); the agent must rerun the relevant checks at the new head and publish a new checkpoint before using any prior GO. Reviewer checks are scheduled **every six hours**, but a reviewer may comment sooner; agent checks every ~15 minutes so it can react as soon as the comment appears. A review schedule is not a guaranteed instant reply.

## Required workflow after a checkpoint

1. Post marked checkpoint on PR #3 (and cross-link the task Issue when available).
2. Set `WAITING_FOR_REVIEW`; **stop writing new task/gate code**.
3. Check PR decision and issue status every ~15 minutes (with a verified running polling process/timer).
4. If `GO` matches the **latest checkpoint SHA and ID**, read the review text, acknowledge if needed and execute only the named next task.
5. For `CHANGES_REQUESTED`, fix *only* the named issues, rerun verification and post a new checkpoint.
6. For `BLOCKED`, invalid review marker, or GitHub network failures, stop and preserve all experimental artifacts. Never infer a GO.

## Immutable scientific boundary

Never rerun the final held-out test evaluator, retune or retrain the frozen winner, change the locked manifest/selection/metrics, commit raw images/credentials/checkpoints, add GitHub Actions or merge PRs without review. Saved final predictions may be audited read-only.

This protocol supersedes older statements implying that the Ubuntu agent cannot poll GitHub; it *can* poll **if it actually runs a loop/timer**, but GitHub polling is not a substitute for independently checking Ubuntu evidence.
