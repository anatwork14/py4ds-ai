"""Read-only GitHub PR review-gate poller for an Ubuntu execution agent.

Run with a server timer/cron or manually. This checker never starts training,
runs evaluations, merges PRs, or advances the agent on its own.

Only a review decision matching the exact checkpoint ID and commit SHA can
authorize the next coursework gate. GitHub Issues are supplementary context;
the authoritative decision is always posted on the tracked pull request.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from typing import Any

CHECKPOINT_RE = re.compile(
    r"<!-- CO3117_AGENT_CHECKPOINT v1 id=(?P<id>[A-Za-z0-9_.-]+)"
    r" sha=(?P<sha>[0-9a-f]{40}) task=(?P<task>TASK-0[1-6]) -->"
)
DECISION_RE = re.compile(
    r"<!-- CO3117_REVIEW_DECISION v1 checkpoint=(?P<id>[A-Za-z0-9_.-]+)"
    r" sha=(?P<sha>[0-9a-f]{40}) verdict=(?P<verdict>GO|CHANGES_REQUESTED|BLOCKED) -->"
)
_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def _fetch_pages(path: str) -> list[dict[str, Any]]:
    """Use gh's existing credentials and collect all REST pages as JSON arrays."""
    proc = subprocess.run(
        ["gh", "api", "--paginate", "--slurp", path],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError(f"GitHub request failed for {path}: {proc.stderr.strip()}")
    pages = json.loads(proc.stdout)
    if not isinstance(pages, list):
        raise ValueError("Expected a list of GitHub API pages")
    items: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, list):
            raise ValueError("Expected each GitHub API page to be an array")
        if not all(isinstance(item, dict) for item in page):
            raise ValueError("Expected JSON comment/issue objects")
        items.extend(page)
    return items


def reviewer_decision(
    comments: list[dict[str, Any]], checkpoint_id: str, head_sha: str, task: str
) -> dict[str, str]:
    """Reject stale or other-checkpoint GO comments, including old G0 approvals."""
    ordered = sorted(comments, key=lambda comment: int(comment.get("id", 0)))
    checkpoints: list[tuple[int, dict[str, str]]] = []
    for comment in ordered:
        marker = CHECKPOINT_RE.search(str(comment.get("body") or ""))
        if marker:
            checkpoints.append((int(comment["id"]), marker.groupdict()))
    if not checkpoints:
        return {"verdict": "AWAITING_CHECKPOINT", "reason": "No marked agent checkpoint"}
    latest_id, latest = checkpoints[-1]
    if (latest["id"], latest["sha"], latest["task"]) != (
        checkpoint_id, head_sha, task
    ):
        return {
            "verdict": "STALE_CHECKPOINT",
            "reason": "Supplied checkpoint is not the most recent PR checkpoint",
        }
    for comment in reversed(ordered):
        comment_id = int(comment.get("id", 0))
        if comment_id <= latest_id:
            break
        marker = DECISION_RE.search(str(comment.get("body") or ""))
        if marker and (marker["id"], marker["sha"]) == (checkpoint_id, head_sha):
            return {
                "verdict": marker["verdict"],
                "reason": "Matched exact checkpoint and SHA",
                "url": str(comment.get("html_url") or ""),
            }
    return {"verdict": "WAITING_FOR_REVIEW", "reason": "No matching reviewer decision yet"}


def _issue_summary(repo: str) -> dict[str, Any]:
    try:
        issues = _fetch_pages(f"repos/{repo}/issues?state=open&per_page=100")
    except RuntimeError as exc:
        # The repository historically has GitHub Issues disabled (HTTP 410).
        return {"status": "unavailable", "reason": str(exc)}
    work_items = [
        item for item in issues
        if "pull_request" not in item
        and (
            "CO3117" in str(item.get("title", "")).upper()
            or "TASK 0" in str(item.get("title", "")).upper()
            or "TASK-0" in str(item.get("title", "")).upper()
        )
    ]
    # Show issue comments for context, but do not authorize GO based on them.
    result = []
    for item in work_items[:12]:
        number = int(item["number"])
        comments = _fetch_pages(
            f"repos/{repo}/issues/{number}/comments?per_page=100"
        )
        latest = comments[-1] if comments else None
        result.append({
            "number": number,
            "title": item.get("title"),
            "url": item.get("html_url"),
            "comment_count": len(comments),
            "latest_comment_url": latest.get("html_url") if latest else None,
        })
    return {"status": "available", "open_tracking_issues": result}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only poll for an exact CO3117 PR checkpoint review"
    )
    parser.add_argument("--repo", default="anatwork14/py4ds-ai")
    parser.add_argument("--pr", type=int, default=3)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--task", choices=[f"TASK-0{i}" for i in range(1, 7)], required=True)
    parser.add_argument(
        "--skip-issues", action="store_true",
        help="Only query PR comments; GitHub Issues are supplementary",
    )
    args = parser.parse_args(argv)
    if not _REPO_RE.fullmatch(args.repo):
        parser.error("Expected --repo owner/name")
    if args.pr < 1 or not re.fullmatch(r"[0-9a-f]{40}", args.head_sha):
        parser.error("Expected positive PR number and full lowercase 40-char SHA")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.checkpoint_id):
        parser.error("Invalid checkpoint ID")

    try:
        comments = _fetch_pages(f"repos/{args.repo}/issues/{args.pr}/comments?per_page=100")
        result: dict[str, Any] = reviewer_decision(
            comments, args.checkpoint_id, args.head_sha, args.task
        )
        result.update({
            "checkpoint_id": args.checkpoint_id,
            "head_sha": args.head_sha,
            "task": args.task,
            "pr_url": f"https://github.com/{args.repo}/pull/{args.pr}",
        })
        if not args.skip_issues:
            result["issues"] = _issue_summary(args.repo)
    except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"verdict": "ERROR", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return {
        "GO": 0,
        "CHANGES_REQUESTED": 12,
        "BLOCKED": 11,
    }.get(result["verdict"], 10)


if __name__ == "__main__":
    sys.exit(main())
