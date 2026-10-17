"""Deterministic artifacts and immutable split-lock handling."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .manifest import CLASS_NAMES, PreparedManifest, assert_no_cross_split_hash_overlap
from .plots import write_dataset_plots

AUDIT_COLUMNS = (
    "sample_id",
    "relative_path",
    "source_split",
    "assigned_split",
    "class_name",
    "class_idx",
    "file_bytes",
    "extension",
    "width",
    "height",
    "mode",
    "channels",
    "sha256",
    "phash",
    "audit_status",
    "exclusion_reason",
)
MANIFEST_COLUMNS = (
    "sample_id",
    "relative_path",
    "source_split",
    "assigned_split",
    "class_name",
    "class_idx",
    "sha256",
    "phash",
    "audit_status",
    "exclusion_reason",
)
LEAKAGE_COLUMNS = (
    "sample_id_a",
    "sample_id_b",
    "source_split_a",
    "source_split_b",
    "assigned_split_a",
    "assigned_split_b",
    "class_a",
    "class_b",
    "sha256_equal",
    "hamming_distance",
    "requires_review",
)


def _csv_bytes(rows: Iterable[dict[str, Any]], columns: tuple[str, ...]) -> bytes:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _dataset_fingerprint(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["sample_id"]):
        digest.update(row["sample_id"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    split_names = ("train", "validation", "test", "prediction", "excluded")
    source_names = ("seg_train", "seg_test", "seg_pred")
    by_split = Counter(row["assigned_split"] for row in rows)
    by_source = Counter(row["source_split"] for row in rows)
    if sum(by_split.values()) != len(rows):
        raise AssertionError("Assigned-split counts do not reconcile to the manifest row count")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise AssertionError("Manifest sample_id values are not unique")
    by_class = {
        class_name: {
            split_name: sum(
                row["class_name"] == class_name and row["assigned_split"] == split_name
                for row in rows
            )
            for split_name in split_names
        }
        for class_name in CLASS_NAMES
    }
    dimensions = [row for row in rows if row["width"] is not None and row["height"] is not None]
    return {
        "total_files": len(rows),
        "counts_by_source_split": {name: by_source[name] for name in source_names},
        "counts_by_assigned_split": {name: by_split[name] for name in split_names},
        "counts_by_class_and_split": by_class,
        "counts_by_audit_status": dict(
            sorted(Counter(row["audit_status"] for row in rows).items())
        ),
        "readable_images": len(dimensions),
        "non_square_images": sum(row["width"] != row["height"] for row in dimensions),
        "train_validation_exclusions": sum(
            row["source_split"] == "seg_train" and row["assigned_split"] == "excluded"
            for row in rows
        ),
        "test_rows_flagged": sum(
            row["source_split"] == "seg_test" and bool(row["exclusion_reason"]) for row in rows
        ),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_source_paths(prepared: PreparedManifest) -> None:
    root = prepared.data_root.resolve()
    for row in prepared.rows:
        candidate = root / row["relative_path"]
        if row["audit_status"] == "symlink_ignored":
            if not candidate.is_symlink():
                raise ValueError(f"Source symlink changed after audit: {row['relative_path']}")
            continue
        source = candidate.resolve()
        if not source.is_relative_to(root):
            raise ValueError(f"Manifest path escapes the dataset root: {row['relative_path']}")
        if not source.is_file():
            raise FileNotFoundError(f"Manifest source file is missing: {row['relative_path']}")
        expected_hash = row["sha256"]
        if expected_hash and _sha256_file(source) != expected_hash:
            raise ValueError(f"Source file changed after audit: {row['relative_path']}")


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "audit": output_dir / "audit.csv",
        "manifest": output_dir / "split_manifest.csv",
        "exclusions": output_dir / "exclusions.csv",
        "leakage_candidates": output_dir / "leakage_candidates.csv",
        "class_to_idx": output_dir / "class_to_idx.json",
        "summary": output_dir / "data_summary.json",
        "lock": output_dir / "split.lock.json",
        "class_counts_plot": output_dir / "plots" / "class_counts.png",
        "image_dimensions_plot": output_dir / "plots" / "image_dimensions.png",
        "audit_status_plot": output_dir / "plots" / "audit_status.png",
        "sample_montage_plot": output_dir / "plots" / "sample_montage.png",
        "sample_montage_manifest": output_dir / "plots" / "sample_montage_manifest.csv",
    }


def write_artifacts(
    prepared: PreparedManifest,
    output_dir: str | Path,
    *,
    montage_per_class: int = 3,
) -> dict[str, Path]:
    """Write an audit bundle once; reject conflicting rewrites of a locked split."""
    if montage_per_class < 1:
        raise ValueError("montage_per_class must be at least 1")
    if prepared.class_to_idx != {name: index for index, name in enumerate(CLASS_NAMES)}:
        raise AssertionError("Class mapping differs from the required fixed alphabetical mapping")
    assert_no_cross_split_hash_overlap(prepared.rows)
    _validate_source_paths(prepared)

    target = Path(output_dir).expanduser().resolve()
    source_root = prepared.data_root.resolve()
    if target == source_root or target.is_relative_to(source_root):
        raise ValueError("Artifact output must be outside the read-only dataset root")

    audit_bytes = _csv_bytes(prepared.rows, AUDIT_COLUMNS)
    manifest_bytes = _csv_bytes(prepared.rows, MANIFEST_COLUMNS)
    exclusion_bytes = _csv_bytes(
        [row for row in prepared.rows if row["assigned_split"] == "excluded"], AUDIT_COLUMNS
    )
    leakage_bytes = _csv_bytes(prepared.leakage_candidates, LEAKAGE_COLUMNS)
    manifest_hash = _sha256_bytes(manifest_bytes)
    dataset_hash = _dataset_fingerprint(prepared.rows)
    output_config = {"montage_per_class": montage_per_class}
    lock = {
        "schema_version": 1,
        "source_root": str(source_root),
        "dataset_fingerprint_sha256": dataset_hash,
        "manifest_sha256": manifest_hash,
        "row_count": len(prepared.rows),
        "split_config": prepared.config,
        "output_config": output_config,
        "class_to_idx": prepared.class_to_idx,
    }

    paths = _artifact_paths(target)
    if target.exists():
        if not paths["lock"].is_file():
            raise FileExistsError(
                f"Output directory exists without a split lock; choose a new directory: {target}"
            )
        existing_lock = json.loads(paths["lock"].read_text(encoding="utf-8"))
        if existing_lock != lock:
            raise FileExistsError(
                f"Existing output is locked to a different manifest/configuration: {target}"
            )
        if (
            not paths["manifest"].is_file()
            or _sha256_bytes(paths["manifest"].read_bytes()) != manifest_hash
        ):
            raise ValueError(
                f"Locked split manifest is missing or has been modified: {paths['manifest']}"
            )
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise ValueError(f"Locked artifact bundle is incomplete: {missing}")
        return paths

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    staged_paths = _artifact_paths(staging)
    try:
        payloads = {
            staged_paths["audit"]: audit_bytes,
            staged_paths["manifest"]: manifest_bytes,
            staged_paths["exclusions"]: exclusion_bytes,
            staged_paths["leakage_candidates"]: leakage_bytes,
            staged_paths["class_to_idx"]: _json_bytes(prepared.class_to_idx),
        }
        summary = {
            "schema_version": 1,
            "source_root": str(source_root),
            "dataset_fingerprint_sha256": dataset_hash,
            "manifest_sha256": manifest_hash,
            "split_config": prepared.config,
            "counts": _counts(prepared.rows),
            "counts_by_assigned_split": _counts(prepared.rows)["counts_by_assigned_split"],
            "leakage_candidate_count": len(prepared.leakage_candidates),
            "cross_split_exact_sha256_overlap": 0,
        }
        payloads[staged_paths["summary"]] = _json_bytes(summary)
        for path, content in payloads.items():
            path.write_bytes(content)
        write_dataset_plots(prepared, staging / "plots", montage_per_class=montage_per_class)
        staged_paths["lock"].write_bytes(_json_bytes(lock))
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return _artifact_paths(target)
