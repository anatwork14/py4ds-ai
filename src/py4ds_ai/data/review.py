"""Apply explicit, reproducible perceptual-hash review decisions to a split."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from .manifest import PreparedManifest

SUPERVISED_SPLITS = {"train", "validation", "test"}
ALLOWED_VERDICTS = {
    "same_or_near_duplicate",
    "uncertain_possible_duplicate",
    "different_image",
}
LEDGER_COLUMNS = {
    "candidate_id",
    "sample_id_a",
    "sample_id_b",
    "assigned_split_a",
    "assigned_split_b",
    "hamming_distance",
    "verdict",
    "rationale",
}


def _pair_key(row: dict[str, Any]) -> tuple[str, str]:
    first, second = sorted((str(row["sample_id_a"]), str(row["sample_id_b"])))
    return first, second


def _is_reviewable(candidate: dict[str, Any]) -> bool:
    split_a = candidate["assigned_split_a"]
    split_b = candidate["assigned_split_b"]
    if split_a not in SUPERVISED_SPLITS or split_b not in SUPERVISED_SPLITS:
        return False
    return split_a != split_b or split_a == "test"


def apply_reviewed_phash_candidates(
    prepared: PreparedManifest,
    ledger_path: str | Path,
) -> PreparedManifest:
    """Exclude reviewed training/validation near-duplicates without changing test rows.

    The ledger must contain exactly every cross-split supervised pHash candidate and
    every test-test candidate from ``prepared``. Uncertain cross-split pairs are
    excluded conservatively; test rows are never removed.
    """
    if "phash_review" in prepared.config:
        raise ValueError("A pHash review has already been applied to this manifest")

    path = Path(ledger_path).expanduser().resolve()
    raw_ledger = path.read_bytes()
    try:
        decoded = raw_ledger.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Review ledger must be UTF-8 CSV") from exc
    reader = csv.DictReader(decoded.splitlines())
    if reader.fieldnames is None or not LEDGER_COLUMNS.issubset(reader.fieldnames):
        missing = sorted(LEDGER_COLUMNS - set(reader.fieldnames or []))
        raise ValueError(f"Review ledger is missing required columns: {missing}")
    ledger_rows = list(reader)

    expected: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in prepared.leakage_candidates:
        if not _is_reviewable(candidate):
            continue
        key = _pair_key(candidate)
        if key in expected:
            raise ValueError(f"Duplicate pHash candidate pair in manifest: {key}")
        expected[key] = candidate

    decisions: dict[tuple[str, str], dict[str, str]] = {}
    for ledger_row in ledger_rows:
        key = _pair_key(ledger_row)
        candidate = expected.get(key)
        if candidate is None:
            raise ValueError(f"Review ledger contains an ineligible candidate pair: {key}")
        if key in decisions:
            raise ValueError(f"Review ledger contains a duplicate candidate pair: {key}")
        candidate_splits = {
            str(candidate["sample_id_a"]): str(candidate["assigned_split_a"]),
            str(candidate["sample_id_b"]): str(candidate["assigned_split_b"]),
        }
        ledger_splits = {
            str(ledger_row["sample_id_a"]): str(ledger_row["assigned_split_a"]),
            str(ledger_row["sample_id_b"]): str(ledger_row["assigned_split_b"]),
        }
        if ledger_splits != candidate_splits:
            raise ValueError(f"Review ledger split metadata does not match candidate {key}")
        try:
            ledger_distance = int(ledger_row["hamming_distance"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid pHash distance for candidate {key}") from exc
        if ledger_distance != int(candidate["hamming_distance"]):
            raise ValueError(f"Review ledger pHash distance does not match candidate {key}")
        verdict = ledger_row["verdict"].strip()
        if verdict not in ALLOWED_VERDICTS:
            raise ValueError(f"Unsupported review verdict for candidate {key}: {verdict!r}")
        decisions[key] = {"verdict": verdict, "candidate_id": ledger_row["candidate_id"].strip()}

    missing_pairs = sorted(set(expected) - set(decisions))
    if missing_pairs:
        raise ValueError(
            "Review ledger does not cover every eligible candidate "
            f"({len(missing_pairs)} missing; first: {missing_pairs[0]})"
        )

    rows = deepcopy(prepared.rows)
    row_by_id = {str(row["sample_id"]): row for row in rows}
    if len(row_by_id) != len(rows):
        raise ValueError("Prepared manifest sample_id values are not unique")
    exclusion_reasons: dict[str, set[str]] = {}
    verdict_counts: Counter[str] = Counter()
    test_test_count = 0

    for key, candidate in expected.items():
        verdict = decisions[key]["verdict"]
        verdict_counts[verdict] += 1
        split_a = str(candidate["assigned_split_a"])
        split_b = str(candidate["assigned_split_b"])
        if split_a == split_b == "test":
            test_test_count += 1
            continue
        if verdict == "different_image":
            continue

        sample_a = str(candidate["sample_id_a"])
        sample_b = str(candidate["sample_id_b"])
        if "test" in {split_a, split_b}:
            excluded_id = sample_b if split_a == "test" else sample_a
            reference_id = sample_a if split_a == "test" else sample_b
            relation = "test"
        elif {split_a, split_b} == {"train", "validation"}:
            excluded_id = sample_a if split_a == "train" else sample_b
            reference_id = sample_b if split_a == "train" else sample_a
            relation = "validation"
        else:
            raise AssertionError(f"Unexpected reviewable split pair: {split_a}, {split_b}")

        row = row_by_id.get(excluded_id)
        if row is None:
            raise ValueError(f"Reviewed sample is absent from the manifest: {excluded_id}")
        if row["source_split"] != "seg_train" or row["assigned_split"] not in {
            "train",
            "validation",
        }:
            raise ValueError(f"Refusing to exclude non-training source row: {excluded_id}")
        exclusion_reasons.setdefault(excluded_id, set()).add(
            f"{verdict}_of_{relation}:{reference_id}"
        )

    for sample_id, reasons in exclusion_reasons.items():
        row = row_by_id[sample_id]
        row["assigned_split"] = "excluded"
        row["exclusion_reason"] = "phash_review:" + ";".join(sorted(reasons))

    parent_assignments = sorted(
        (str(row["sample_id"]), str(row["assigned_split"])) for row in prepared.rows
    )
    parent_digest = hashlib.sha256(
        json.dumps(parent_assignments, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    config = deepcopy(prepared.config)
    config["phash_review"] = {
        "policy": (
            "exclude non-test endpoints of visually confirmed or uncertain cross-split "
            "near-duplicate candidates; preserve all test rows"
        ),
        "ledger_sha256": hashlib.sha256(raw_ledger).hexdigest(),
        "parent_assignment_sha256": parent_digest,
        "reviewed_candidate_count": len(decisions),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "test_test_candidate_count": test_test_count,
        "excluded_sample_ids": sorted(exclusion_reasons),
        "excluded_count": len(exclusion_reasons),
    }

    leakage_candidates = deepcopy(prepared.leakage_candidates)
    for candidate in leakage_candidates:
        for side in ("a", "b"):
            sample_id = str(candidate[f"sample_id_{side}"])
            if sample_id in row_by_id:
                candidate[f"assigned_split_{side}"] = row_by_id[sample_id]["assigned_split"]

    return PreparedManifest(
        rows=rows,
        class_to_idx=deepcopy(prepared.class_to_idx),
        data_root=prepared.data_root,
        config=config,
        leakage_candidates=leakage_candidates,
    )
