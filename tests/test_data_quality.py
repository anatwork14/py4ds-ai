from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from py4ds_ai.data.manifest import CLASS_NAMES, build_manifest


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), color).save(path, format="PNG")


def _dataset_with_one_per_class(root: Path) -> None:
    for class_index, class_name in enumerate(CLASS_NAMES):
        _write_image(
            root / "seg_train" / "seg_train" / class_name / "train.png",
            (class_index * 25, 50, 80),
        )
        _write_image(
            root / "seg_test" / "seg_test" / class_name / "test.png",
            (100, class_index * 25, 60),
        )


def test_corrupt_training_file_is_flagged_and_left_in_place(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    _dataset_with_one_per_class(data_root)
    corrupt_path = data_root / "seg_train" / "seg_train" / "forest" / "broken.jpg"
    corrupt_path.write_bytes(b"")
    original_hash = hashlib.sha256(corrupt_path.read_bytes()).hexdigest()

    prepared = build_manifest(data_root, val_fraction=0.5, seed=42)

    row = next(
        item for item in prepared.rows if item["relative_path"].endswith("forest/broken.jpg")
    )
    assert row["audit_status"] == "unreadable"
    assert row["assigned_split"] == "excluded"
    assert row["sha256"] == original_hash
    assert corrupt_path.read_bytes() == b""


def test_seg_pred_is_preserved_as_unlabeled_prediction_pool(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    _dataset_with_one_per_class(data_root)
    prediction_path = data_root / "seg_pred" / "seg_pred" / "unlabeled.png"
    _write_image(prediction_path, (15, 25, 35))
    original_bytes = prediction_path.read_bytes()

    prepared = build_manifest(data_root, val_fraction=0.5, seed=42)

    row = next(item for item in prepared.rows if item["relative_path"].endswith("unlabeled.png"))
    assert row["source_split"] == "seg_pred"
    assert row["assigned_split"] == "prediction"
    assert row["class_name"] is None
    assert row["class_idx"] is None
    assert prediction_path.read_bytes() == original_bytes


def test_exact_duplicates_inside_train_are_assigned_as_one_group(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    _dataset_with_one_per_class(data_root)
    source = data_root / "seg_train" / "seg_train" / "buildings" / "train.png"
    duplicate = data_root / "seg_train" / "seg_train" / "buildings" / "train_copy.png"
    duplicate.write_bytes(source.read_bytes())
    _write_image(
        data_root / "seg_train" / "seg_train" / "buildings" / "second.png",
        (230, 40, 70),
    )

    prepared = build_manifest(data_root, val_fraction=0.5, seed=42)

    copies = [
        row
        for row in prepared.rows
        if row["source_split"] == "seg_train"
        and row["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    ]
    assert len(copies) == 2
    assert copies[0]["assigned_split"] in {"train", "validation"}
    assert copies[0]["assigned_split"] == copies[1]["assigned_split"]
