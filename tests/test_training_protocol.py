from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image

from py4ds_ai.models.data import load_train_validation


def _write_bundle(root: Path) -> tuple[Path, dict[str, Path]]:
    data_root = root / "raw"
    split_dir = root / "manifests" / "seed-42"
    split_dir.mkdir(parents=True)
    files = {
        "train": data_root / "seg_train" / "seg_train" / "buildings" / "train.png",
        "validation": data_root / "seg_train" / "seg_train" / "buildings" / "validation.png",
    }
    for index, path in enumerate(files.values()):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), (index * 30, 40, 80)).save(path)

    rows = []
    for sample_id, path, assigned in (
        ("train-row", files["train"], "train"),
        ("validation-row", files["validation"], "validation"),
        # Deliberately nonexistent: train/validation loading must not inspect test files.
        ("sealed-test-row", data_root / "missing-test.png", "test"),
    ):
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if assigned != "test" else "sealed"
        rows.append(
            {
                "sample_id": sample_id,
                "relative_path": path.relative_to(data_root).as_posix()
                if assigned != "test"
                else "missing-test.png",
                "source_split": "seg_test" if assigned == "test" else "seg_train",
                "assigned_split": assigned,
                "class_name": "buildings",
                "class_idx": 0,
                "sha256": digest,
                "phash": "",
                "audit_status": "ok",
                "exclusion_reason": "",
            }
        )

    manifest_path = split_dir / "split_manifest.csv"
    columns = list(rows[0])
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    class_map = {
        "buildings": 0,
        "forest": 1,
        "glacier": 2,
        "mountain": 3,
        "sea": 4,
        "street": 5,
    }
    (split_dir / "class_to_idx.json").write_text(
        json.dumps(class_map),
        encoding="utf-8",
    )
    lock = {
        "source_root": str(data_root.resolve()),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "class_to_idx": {
            "buildings": 0,
            "forest": 1,
            "glacier": 2,
            "mountain": 3,
            "sea": 4,
            "street": 5,
        },
    }
    (split_dir / "split.lock.json").write_text(json.dumps(lock), encoding="utf-8")
    return manifest_path, files


def test_loader_verifies_lock_and_reads_only_train_validation(tmp_path: Path) -> None:
    manifest_path, source_files = _write_bundle(tmp_path)

    train_rows, validation_rows, class_to_idx, manifest_hash = load_train_validation(manifest_path)

    assert [row.sample_id for row in train_rows] == ["train-row"]
    assert [row.sample_id for row in validation_rows] == ["validation-row"]
    assert train_rows[0].path == source_files["train"].resolve()
    assert class_to_idx["buildings"] == 0
    assert manifest_hash == hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def test_loader_rejects_manifest_modified_after_lock(tmp_path: Path) -> None:
    manifest_path, _ = _write_bundle(tmp_path)
    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")

    try:
        load_train_validation(manifest_path)
    except ValueError as exc:
        assert "manifest hash differs from split lock" in str(exc)
    else:
        raise AssertionError("modified manifest unexpectedly passed split-lock validation")
