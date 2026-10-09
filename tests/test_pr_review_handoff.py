"""Offline unit tests for the GitHub-only agent/reviewer checkpoint protocol."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
handoff = importlib.import_module("check_pr_review")

HEAD = "a" * 40
OTHER_HEAD = "b" * 40
CHECKPOINT = "<!-- CO3117_AGENT_CHECKPOINT v1 id=G1-A sha=" + HEAD + " task=TASK-02 -->"


def _comment(number: int, body: str) -> dict:
    return {"id": number, "body": body, "html_url": f"https://example.test/comment/{number}"}


def _review(
    verdict: str, *, checkpoint: str = "G1-A", sha: str = HEAD
) -> str:
    return (
        "<!-- CO3117_REVIEW_DECISION v1 checkpoint="
        + checkpoint + " sha=" + sha + " verdict=" + verdict + " -->"
    )


def test_no_checkpoint_is_not_approval() -> None:
    assert handoff.reviewer_decision([], "G1-A", HEAD, "TASK-02")["verdict"] == (
        "AWAITING_CHECKPOINT"
    )


def test_older_go_cannot_approve_new_gate() -> None:
    comments = [
        _comment(10, "<!-- CO3117_AGENT_CHECKPOINT v1 id=G0-A sha="
                 + OTHER_HEAD + " task=TASK-01 -->"),
        _comment(11, _review("GO", checkpoint="G0-A", sha=OTHER_HEAD)),
        _comment(12, CHECKPOINT),
    ]
    assert handoff.reviewer_decision(comments, "G1-A", HEAD, "TASK-02")["verdict"] == (
        "WAITING_FOR_REVIEW"
    )


def test_exact_sha_and_checkpoint_required() -> None:
    comments = [
        _comment(10, CHECKPOINT),
        _comment(11, _review("GO", sha=OTHER_HEAD)),
        _comment(12, _review("GO", checkpoint="some-other-gate")),
    ]
    assert handoff.reviewer_decision(comments, "G1-A", HEAD, "TASK-02")["verdict"] == (
        "WAITING_FOR_REVIEW"
    )


def test_matched_review_verdicts_are_returned() -> None:
    for verdict in ("GO", "BLOCKED", "CHANGES_REQUESTED"):
        result = handoff.reviewer_decision(
            [_comment(10, CHECKPOINT), _comment(11, _review(verdict))],
            "G1-A",
            HEAD,
            "TASK-02",
        )
        assert result["verdict"] == verdict
        assert result["url"] == "https://example.test/comment/11"


def test_new_checkpoint_invalidates_prior_go() -> None:
    comments = [
        _comment(10, CHECKPOINT),
        _comment(11, _review("GO")),
        _comment(12, "<!-- CO3117_AGENT_CHECKPOINT v1 id=G2-A sha="
                 + OTHER_HEAD + " task=TASK-03 -->"),
    ]
    result = handoff.reviewer_decision(comments, "G1-A", HEAD, "TASK-02")
    assert result["verdict"] == "STALE_CHECKPOINT"


def test_latest_matching_verdict_wins() -> None:
    comments = [
        _comment(10, CHECKPOINT),
        _comment(11, _review("CHANGES_REQUESTED")),
        _comment(12, _review("GO")),
    ]
    assert handoff.reviewer_decision(comments, "G1-A", HEAD, "TASK-02")["verdict"] == "GO"


def test_invalid_marker_is_ignored() -> None:
    comments = [_comment(10, "I think this is GO"), _comment(11, _review("GO"))]
    assert handoff.reviewer_decision(comments, "G1-A", HEAD, "TASK-02")["verdict"] == (
        "AWAITING_CHECKPOINT"
    )
