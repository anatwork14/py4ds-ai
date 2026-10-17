from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from py4ds_ai.data.manifest import CLASS_NAMES, build_manifest
from py4ds_ai.data.outputs import write_artifacts


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 60), color).save(path, format="PNG", pnginfo=None)


def _make_dataset(root: Path) -> None:
    for class_index, class_name in enumerate(CLASS_NAMES):
        train_dir = root / "seg_train" / "seg_train" / class_name
        test_dir = root / "seg_test" / "seg_test" / class_name
        _write_image(train_dir / "a.png", (class_index * 20, 40, 80))
        _write_image(train_dir / "b.png", (class_index * 20, 90, 130))
        _write_image(test_dir / "test.png", (120, class_index * 20, 30))


def test_writer_emits_locked_artifacts_and_train_only_montage(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    _make_dataset(data_root)
    prepared = build_manifest(data_root, val_fraction=0.5, seed=42)
    output_dir = tmp_path / "manifest-output"

    write_artifacts(prepared, output_dir, montage_per_class=1)

    required = [
        "audit.csv",
        "split_manifest.csv",
        "exclusions.csv",
        "leakage_candidates.csv",
        "class_to_idx.json",
        "data_summary.json",
        "split.lock.json",
        "plots/class_counts.png",
        "plots/image_dimensions.png",
        "plots/audit_status.png",
        "plots/sample_montage.png",
        "plots/sample_montage_manifest.csv",
    ]
    assert all((output_dir / relative).is_file() for relative in required)

    summary = json.loads((output_dir / "data_summary.json").read_text(encoding="utf-8"))
    assert sum(summary["counts_by_assigned_split"].values()) == len(prepared.rows)
    assert summary["counts"]["total_files"] == len(prepared.rows)

    row_by_id = {row["sample_id"]: row for row in prepared.rows}
    with (output_dir / "plots/sample_montage_manifest.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        montage_rows = list(csv.DictReader(stream))
    assert montage_rows
    assert all(row_by_id[row["sample_id"]]["assigned_split"] == "train" for row in montage_rows)

    original_manifest = (output_dir / "split_manifest.csv").read_bytes()
    repeated = build_manifest(data_root, val_fraction=0.5, seed=42)
    assert repeated.rows == prepared.rows
    write_artifacts(repeated, output_dir, montage_per_class=1)
    assert (output_dir / "split_manifest.csv").read_bytes() == original_manifest


def test_locked_output_rejects_a_different_split_configuration(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    _make_dataset(data_root)
    output_dir = tmp_path / "manifest-output"
    first = build_manifest(data_root, val_fraction=0.5, seed=42)
    second = build_manifest(data_root, val_fraction=0.5, seed=7)
    write_artifacts(first, output_dir, montage_per_class=1)

    with pytest.raises(FileExistsError, match="locked"):
        write_artifacts(second, output_dir, montage_per_class=1)


def test_writer_rejects_source_files_changed_after_audit(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    _make_dataset(data_root)
    prepared = build_manifest(data_root, val_fraction=0.5, seed=42)
    changed_source = data_root / "seg_train" / "seg_train" / "buildings" / "a.png"
    _write_image(changed_source, (255, 1, 2))
    output_dir = tmp_path / "manifest-output"

    with pytest.raises(ValueError, match="changed after audit"):
        write_artifacts(prepared, output_dir, montage_per_class=1)
    assert not output_dir.exists()
