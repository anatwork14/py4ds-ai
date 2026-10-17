"""Read and verify train/validation rows from the immutable manifest."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from py4ds_ai.data.manifest import CLASS_NAMES


@dataclass(frozen=True)
class ImageRow:
    """A verified labeled image reference from one supervised split."""

    sample_id: str
    path: Path
    class_name: str
    class_idx: int
    sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_train_validation(
    manifest_path: str | Path,
) -> tuple[list[ImageRow], list[ImageRow], dict[str, int], str]:
    """Validate the split lock and source hashes, reading no test image paths."""
    manifest_file = Path(manifest_path).expanduser().resolve()
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Split manifest does not exist: {manifest_file}")
    bundle_dir = manifest_file.parent
    lock_path = bundle_dir / "split.lock.json"
    class_map_path = bundle_dir / "class_to_idx.json"
    if not lock_path.is_file() or not class_map_path.is_file():
        raise FileNotFoundError(
            "Manifest bundle must contain split.lock.json and class_to_idx.json"
        )

    manifest_bytes = manifest_file.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    lock: dict[str, Any] = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("manifest_sha256") != manifest_hash:
        raise ValueError("manifest hash differs from split lock")
    source_root = Path(lock["source_root"]).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Locked dataset root is not a directory: {source_root}")

    class_to_idx = json.loads(class_map_path.read_text(encoding="utf-8"))
    expected_mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    if class_to_idx != expected_mapping or lock.get("class_to_idx") != expected_mapping:
        raise ValueError("Class mapping differs from the required fixed alphabetical mapping")

    reader = csv.DictReader(manifest_bytes.decode("utf-8").splitlines())
    required_columns = {
        "sample_id",
        "relative_path",
        "source_split",
        "assigned_split",
        "class_name",
        "class_idx",
        "sha256",
        "audit_status",
    }
    if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
        raise ValueError("Split manifest is missing required columns")

    rows = list(reader)
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Split manifest sample_id values are not unique")

    selected: dict[str, list[ImageRow]] = {"train": [], "validation": []}
    for row in rows:
        split = row["assigned_split"]
        if split not in selected:
            continue
        if row["source_split"] != "seg_train":
            raise ValueError(f"{split} rows must come from seg_train: {row['sample_id']}")
        if row["audit_status"] != "ok":
            raise ValueError(f"Non-readable row assigned to {split}: {row['sample_id']}")
        class_name = row["class_name"]
        if class_name not in class_to_idx or int(row["class_idx"]) != class_to_idx[class_name]:
            raise ValueError(f"Invalid class mapping in row: {row['sample_id']}")
        relative = PurePosixPath(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in row["relative_path"]:
            raise ValueError(f"Unsafe manifest path: {row['relative_path']}")
        candidate = source_root.joinpath(*relative.parts)
        if candidate.is_symlink():
            raise ValueError(f"Manifest source became a symlink: {row['relative_path']}")
        source = candidate.resolve()
        if not source.is_relative_to(source_root):
            raise ValueError(
                f"Manifest path escapes the locked dataset root: {row['relative_path']}"
            )
        if not source.is_file():
            raise FileNotFoundError(f"Manifest source file is missing: {row['relative_path']}")
        actual_hash = _sha256_file(source)
        if actual_hash != row["sha256"]:
            raise ValueError(f"Source file changed after audit: {row['relative_path']}")
        selected[split].append(
            ImageRow(
                sample_id=row["sample_id"],
                path=source,
                class_name=class_name,
                class_idx=class_to_idx[class_name],
                sha256=actual_hash,
            )
        )

    if not selected["train"] or not selected["validation"]:
        raise ValueError("Locked manifest must contain non-empty train and validation splits")
    return selected["train"], selected["validation"], class_to_idx, manifest_hash
