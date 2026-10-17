from __future__ import annotations

import csv
from pathlib import Path

import pytest

from py4ds_ai.data.manifest import PreparedManifest
from py4ds_ai.data.review import apply_reviewed_phash_candidates


def _row(sample_id: str, split: str, source_split: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "relative_path": sample_id,
        "source_split": source_split,
        "assigned_split": split,
        "class_name": "forest",
        "class_idx": 1,
        "file_bytes": 1,
        "extension": ".jpg",
        "width": 8,
        "height": 8,
        "mode": "RGB",
        "channels": 3,
        "sha256": f"sha-{sample_id}",
        "phash": "0000000000000000",
        "audit_status": "ok",
        "exclusion_reason": "",
    }


def _candidate(a: str, b: str, split_a: str, split_b: str) -> dict[str, object]:
    return {
        "sample_id_a": a,
        "sample_id_b": b,
        "source_split_a": "seg_test" if split_a == "test" else "seg_train",
        "source_split_b": "seg_test" if split_b == "test" else "seg_train",
        "assigned_split_a": split_a,
        "assigned_split_b": split_b,
        "class_a": "forest",
        "class_b": "forest",
        "sha256_equal": False,
        "hamming_distance": 2,
        "requires_review": True,
    }


def _write_ledger(path: Path, candidates: list[dict[str, object]]) -> None:
    fields = [
        "candidate_id",
        "sample_id_a",
        "sample_id_b",
        "assigned_split_a",
        "assigned_split_b",
        "hamming_distance",
        "verdict",
        "rationale",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidates)


def test_reviewed_phash_candidates_exclude_non_test_side_conservatively(tmp_path: Path) -> None:
    candidates = [
        _candidate("t1", "tr1", "test", "train"),
        _candidate("t1", "v1", "test", "validation"),
        _candidate("tr2", "v2", "train", "validation"),
        _candidate("t1", "t2", "test", "test"),
        _candidate("t1", "tr3", "test", "train"),
    ]
    verdicts = [
        "same_or_near_duplicate",
        "uncertain_possible_duplicate",
        "same_or_near_duplicate",
        "same_or_near_duplicate",
        "different_image",
    ]
    ledger_rows = [
        {
            "candidate_id": f"R{index:02d}",
            "sample_id_a": item["sample_id_a"],
            "sample_id_b": item["sample_id_b"],
            "assigned_split_a": item["assigned_split_a"],
            "assigned_split_b": item["assigned_split_b"],
            "hamming_distance": item["hamming_distance"],
            "verdict": verdicts[index - 1],
            "rationale": "synthetic test review",
        }
        for index, item in enumerate(candidates, 1)
    ]
    ledger = tmp_path / "review.csv"
    _write_ledger(ledger, ledger_rows)
    prepared = PreparedManifest(
        rows=[
            _row("t1", "test", "seg_test"),
            _row("t2", "test", "seg_test"),
            _row("tr1", "train", "seg_train"),
            _row("tr2", "train", "seg_train"),
            _row("tr3", "train", "seg_train"),
            _row("v1", "validation", "seg_train"),
            _row("v2", "validation", "seg_train"),
        ],
        class_to_idx={
            "buildings": 0,
            "forest": 1,
            "glacier": 2,
            "mountain": 3,
            "sea": 4,
            "street": 5,
        },
        data_root=tmp_path,
        config={"seed": 42, "phash_hamming_threshold": 4},
        leakage_candidates=candidates,
    )

    reviewed = apply_reviewed_phash_candidates(prepared, ledger)

    rows = {row["sample_id"]: row for row in reviewed.rows}
    assert rows["tr1"]["assigned_split"] == "excluded"
    assert rows["v1"]["assigned_split"] == "excluded"
    assert rows["tr2"]["assigned_split"] == "excluded"
    assert rows["tr3"]["assigned_split"] == "train"
    assert rows["v2"]["assigned_split"] == "validation"
    assert rows["t1"]["assigned_split"] == rows["t2"]["assigned_split"] == "test"
    assert {row["assigned_split"] for row in prepared.rows if row["sample_id"] == "tr1"} == {
        "train"
    }
    assert reviewed.config["phash_review"]["reviewed_candidate_count"] == 5
    assert reviewed.config["phash_review"]["excluded_sample_ids"] == ["tr1", "tr2", "v1"]


def test_review_ledger_must_cover_every_eligible_candidate(tmp_path: Path) -> None:
    candidate = _candidate("t1", "tr1", "test", "train")
    prepared = PreparedManifest(
        rows=[_row("t1", "test", "seg_test"), _row("tr1", "train", "seg_train")],
        class_to_idx={
            "buildings": 0,
            "forest": 1,
            "glacier": 2,
            "mountain": 3,
            "sea": 4,
            "street": 5,
        },
        data_root=tmp_path,
        config={"seed": 42, "phash_hamming_threshold": 4},
        leakage_candidates=[candidate],
    )
    ledger = tmp_path / "review.csv"
    _write_ledger(ledger, [])

    with pytest.raises(ValueError, match="does not cover every eligible candidate"):
        apply_reviewed_phash_candidates(prepared, ledger)
