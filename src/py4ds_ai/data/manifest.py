"""Read-only inventory and deterministic split assignment for the Intel dataset."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import imagehash
from PIL import Image, UnidentifiedImageError

CLASS_NAMES = ("buildings", "forest", "glacier", "mountain", "sea", "street")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class PreparedManifest:
    """In-memory manifest rows and the fixed class mapping."""

    rows: list[dict[str, Any]]
    class_to_idx: dict[str, int]
    data_root: Path
    config: dict[str, Any]
    leakage_candidates: list[dict[str, Any]] = field(default_factory=list)


def _discover_split_dir(data_root: Path, split_name: str) -> Path | None:
    outer = data_root / split_name
    if outer.is_symlink():
        raise ValueError(f"Refusing symlink split directory: {outer}")
    if not outer.is_dir():
        return None
    nested = outer / split_name
    if nested.is_symlink():
        raise ValueError(f"Refusing symlink nested split directory: {nested}")
    return nested if nested.is_dir() else outer


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit_file(path: Path, data_root: Path, source_split: str, split_dir: Path) -> dict[str, Any]:
    relative_path = path.relative_to(data_root).as_posix()
    subpath = path.relative_to(split_dir)
    class_name = (
        None
        if source_split == "seg_pred"
        else (subpath.parts[0] if len(subpath.parts) > 1 else None)
    )
    class_to_idx = {name: index for index, name in enumerate(CLASS_NAMES)}
    file_bytes = path.stat().st_size
    row: dict[str, Any] = {
        "sample_id": relative_path,
        "relative_path": relative_path,
        "source_split": source_split,
        "assigned_split": "excluded",
        "class_name": class_name,
        "class_idx": class_to_idx.get(class_name) if class_name is not None else None,
        "file_bytes": file_bytes,
        "extension": path.suffix.lower(),
        "width": None,
        "height": None,
        "mode": None,
        "channels": None,
        "sha256": _hash_file(path),
        "phash": None,
        "audit_status": "ok",
        "exclusion_reason": "",
    }
    if row["extension"] not in IMAGE_EXTENSIONS:
        row["audit_status"] = "unsupported_extension"
        row["exclusion_reason"] = "unsupported_extension"
        return row
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            row["width"], row["height"] = image.size
            row["mode"] = image.mode
            row["channels"] = len(image.getbands())
            row["phash"] = str(imagehash.phash(image))
    except (OSError, UnidentifiedImageError, ValueError, Image.DecompressionBombError) as exc:
        row["audit_status"] = "unreadable"
        row["exclusion_reason"] = f"unreadable:{type(exc).__name__}"
    return row


def _audit_symlink(
    path: Path, data_root: Path, source_split: str, split_dir: Path
) -> dict[str, Any]:
    relative_path = path.relative_to(data_root).as_posix()
    subpath = path.relative_to(split_dir)
    class_name = (
        None
        if source_split == "seg_pred"
        else (subpath.parts[0] if len(subpath.parts) > 1 else None)
    )
    class_to_idx = {name: index for index, name in enumerate(CLASS_NAMES)}
    return {
        "sample_id": relative_path,
        "relative_path": relative_path,
        "source_split": source_split,
        "assigned_split": "excluded",
        "class_name": class_name,
        "class_idx": class_to_idx.get(class_name) if class_name is not None else None,
        "file_bytes": None,
        "extension": path.suffix.lower(),
        "width": None,
        "height": None,
        "mode": None,
        "channels": None,
        "sha256": "",
        "phash": None,
        "audit_status": "symlink_ignored",
        "exclusion_reason": "symlink_not_followed",
    }


def assert_no_cross_split_hash_overlap(rows: list[dict[str, Any]]) -> None:
    """Raise when an exact file hash appears in more than one supervised split."""
    hash_to_split: dict[str, tuple[str, str]] = {}
    for row in rows:
        split_name = row["assigned_split"]
        digest = row["sha256"]
        if split_name not in {"train", "validation", "test"} or not digest:
            continue
        previous = hash_to_split.get(digest)
        if previous is not None and previous[0] != split_name:
            raise AssertionError(
                f"Exact SHA-256 leakage between {previous[0]} ({previous[1]}) "
                f"and {split_name} ({row['sample_id']})"
            )
        hash_to_split[digest] = (split_name, row["sample_id"])


class _BKNode:
    def __init__(self, value: int, row: dict[str, Any]) -> None:
        self.value = value
        self.rows = [row]
        self.children: dict[int, _BKNode] = {}


class _BKTree:
    """BK-tree for Hamming-distance searches over fixed-width perceptual hashes."""

    def __init__(self) -> None:
        self.root: _BKNode | None = None

    def add(self, value: int, row: dict[str, Any]) -> None:
        if self.root is None:
            self.root = _BKNode(value, row)
            return
        node = self.root
        while True:
            distance = (value ^ node.value).bit_count()
            if distance == 0:
                node.rows.append(row)
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _BKNode(value, row)
                return
            node = child

    def query(self, value: int, radius: int) -> list[tuple[dict[str, Any], int]]:
        if self.root is None:
            return []
        matches: list[tuple[dict[str, Any], int]] = []
        pending = [self.root]
        while pending:
            node = pending.pop()
            distance = (value ^ node.value).bit_count()
            if distance <= radius:
                matches.extend((row, distance) for row in node.rows)
            low, high = distance - radius, distance + radius
            pending.extend(child for edge, child in node.children.items() if low <= edge <= high)
        return matches


def find_phash_candidates(
    rows: list[dict[str, Any]],
    *,
    hamming_threshold: int = 4,
) -> list[dict[str, Any]]:
    """Return perceptual-hash neighbors for human review; never exclude a row."""
    if not 0 <= hamming_threshold <= 64:
        raise ValueError("hamming_threshold must be between 0 and 64")
    tree = _BKTree()
    candidates: list[dict[str, Any]] = []
    ordered_rows = sorted(rows, key=lambda row: row["sample_id"])
    for row in ordered_rows:
        hash_text = row.get("phash")
        if not hash_text:
            continue
        if len(hash_text) != 16:
            raise ValueError(f"Expected 64-bit phash as 16 hexadecimal characters: {hash_text!r}")
        try:
            hash_value = int(hash_text, 16)
        except ValueError as exc:
            raise ValueError(f"Invalid hexadecimal phash: {hash_text!r}") from exc
        for neighbor, distance in tree.query(hash_value, hamming_threshold):
            same_sha256 = row.get("sha256") == neighbor.get("sha256")
            if same_sha256:
                continue
            left, right = sorted((row, neighbor), key=lambda item: item["sample_id"])
            candidates.append(
                {
                    "sample_id_a": left["sample_id"],
                    "sample_id_b": right["sample_id"],
                    "source_split_a": left["source_split"],
                    "source_split_b": right["source_split"],
                    "assigned_split_a": left["assigned_split"],
                    "assigned_split_b": right["assigned_split"],
                    "class_a": left["class_name"],
                    "class_b": right["class_name"],
                    "sha256_equal": False,
                    "hamming_distance": distance,
                    "requires_review": True,
                }
            )
        tree.add(hash_value, row)
    candidates.sort(
        key=lambda item: (item["sample_id_a"], item["sample_id_b"], item["hamming_distance"])
    )
    return candidates


def build_manifest(
    data_root: str | Path,
    *,
    val_fraction: float = 0.15,
    seed: int = 42,
    phash_hamming_threshold: int = 4,
) -> PreparedManifest:
    """Audit supported files and assign a stable validation subset without moving them."""
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root is not a directory: {root}")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1")

    class_to_idx = {name: index for index, name in enumerate(CLASS_NAMES)}
    rows: list[dict[str, Any]] = []
    for source_split in ("seg_train", "seg_test", "seg_pred"):
        split_dir = _discover_split_dir(root, source_split)
        if split_dir is None:
            if source_split in {"seg_train", "seg_test"}:
                raise FileNotFoundError(
                    f"Required split directory is missing: {root / source_split}"
                )
            continue
        if source_split in {"seg_train", "seg_test"}:
            observed_classes = {
                child.name
                for child in split_dir.iterdir()
                if child.is_dir() and not child.is_symlink()
            }
            missing_classes = sorted(set(CLASS_NAMES) - observed_classes)
            if missing_classes:
                missing = ", ".join(missing_classes)
                raise ValueError(
                    f"{source_split} is missing expected class directories: {missing}"
                )
        for path in sorted(split_dir.rglob("*")):
            if path.is_symlink():
                rows.append(_audit_symlink(path, root, source_split, split_dir))
            elif path.is_file():
                rows.append(_audit_file(path, root, source_split, split_dir))

    if not rows:
        raise ValueError(f"No files found in required dataset splits under {root}")

    for row in rows:
        if row["source_split"] == "seg_pred":
            row["assigned_split"] = "prediction"
        elif row["source_split"] == "seg_test":
            row["assigned_split"] = "test"
            if row["class_name"] not in class_to_idx:
                row["audit_status"] = "unknown_class"
                row["exclusion_reason"] = "unknown_class_preserved_in_test"
            elif row["audit_status"] != "ok":
                row["exclusion_reason"] = row["exclusion_reason"] or row["audit_status"]
        elif row["audit_status"] != "ok":
            row["exclusion_reason"] = row["exclusion_reason"] or row["audit_status"]
        elif row["class_name"] not in class_to_idx:
            row["audit_status"] = "unknown_class"
            row["exclusion_reason"] = "unknown_class"

    test_by_sha: dict[str, list[dict[str, Any]]] = {}
    train_by_sha: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        digest = row["sha256"]
        if row["source_split"] == "seg_test" and digest:
            test_by_sha.setdefault(digest, []).append(row)
        elif (
            row["source_split"] == "seg_train"
            and row["audit_status"] == "ok"
            and row["class_name"] in class_to_idx
            and digest
        ):
            train_by_sha.setdefault(digest, []).append(row)

    for test_group in test_by_sha.values():
        labels = {row["class_name"] for row in test_group}
        if len(labels) > 1:
            for row in test_group:
                if row["audit_status"] == "ok":
                    row["audit_status"] = "label_conflict"
                row["exclusion_reason"] = "label_conflict_preserved_in_test"

    class_groups: dict[str, list[list[dict[str, Any]]]] = {name: [] for name in CLASS_NAMES}
    for digest, train_group in train_by_sha.items():
        test_matches = test_by_sha.get(digest, [])
        labels = {row["class_name"] for row in train_group}
        if test_matches:
            test_sample_id = min(row["sample_id"] for row in test_matches)
            reason = f"duplicate_of_test:{test_sample_id}"
            if len(labels) > 1:
                reason = f"label_conflict+{reason}"
            for row in train_group:
                row["assigned_split"] = "excluded"
                row["exclusion_reason"] = reason
                if len(labels) > 1:
                    row["audit_status"] = "label_conflict"
            continue
        if len(labels) != 1:
            for row in train_group:
                row["assigned_split"] = "excluded"
                row["audit_status"] = "label_conflict"
                row["exclusion_reason"] = "label_conflict_within_train"
            continue
        class_name = next(iter(labels))
        class_groups[class_name].append(train_group)

    for class_name, class_idx in class_to_idx.items():
        groups = sorted(
            class_groups[class_name], key=lambda group: min(row["sample_id"] for row in group)
        )
        rng = random.Random(seed + class_idx)
        rng.shuffle(groups)
        total_count = sum(len(group) for group in groups)
        validation_groups: list[list[dict[str, Any]]] = []
        validation_count = 0
        if total_count > 1 and len(groups) > 1:
            target_count = min(total_count - 1, max(1, round(total_count * val_fraction)))
            for group in groups:
                proposed_count = validation_count + len(group)
                if total_count - proposed_count < 1:
                    continue
                if proposed_count <= target_count or abs(target_count - proposed_count) < abs(
                    target_count - validation_count
                ):
                    validation_groups.append(group)
                    validation_count = proposed_count
                if validation_count >= target_count:
                    break
            if not validation_groups:
                candidates = [group for group in groups if total_count - len(group) >= 1]
                if candidates:
                    best = min(
                        candidates,
                        key=lambda group: (abs(target_count - len(group)), groups.index(group)),
                    )
                    validation_groups.append(best)

        validation_ids = {row["sample_id"] for group in validation_groups for row in group}
        for group in groups:
            for row in group:
                row["assigned_split"] = (
                    "validation" if row["sample_id"] in validation_ids else "train"
                )

    rows.sort(key=lambda row: row["relative_path"])
    assert_no_cross_split_hash_overlap(rows)
    leakage_candidates = find_phash_candidates(rows, hamming_threshold=phash_hamming_threshold)
    config = {
        "schema_version": 1,
        "seed": seed,
        "val_fraction": val_fraction,
        "phash_hamming_threshold": phash_hamming_threshold,
        "split_strategy": "class-stratified deterministic SHA-256 groups",
        "test_protocol": "seg_test remains test; matching seg_train files are excluded",
        "class_names": list(CLASS_NAMES),
    }
    return PreparedManifest(
        rows=rows,
        class_to_idx=class_to_idx,
        data_root=root,
        config=config,
        leakage_candidates=leakage_candidates,
    )
