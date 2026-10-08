from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from py4ds_ai.data.manifest import CLASS_NAMES, build_manifest


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color).save(path, format="JPEG", quality=95)


def test_exact_train_test_duplicates_exclude_train_copy_only(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    duplicate_train = data_root / "seg_train" / "seg_train" / "buildings" / "train_0.jpg"
    for class_index, class_name in enumerate(CLASS_NAMES):
        train_dir = data_root / "seg_train" / "seg_train" / class_name
        test_dir = data_root / "seg_test" / "seg_test" / class_name
        for image_index in range(3):
            _write_image(
                train_dir / f"train_{image_index}.jpg", (class_index * 20, image_index * 30, 80)
            )
        test_dir.mkdir(parents=True, exist_ok=True)
        if class_name == "buildings":
            test_copy = test_dir / "test_duplicate.jpg"
            test_copy.write_bytes(duplicate_train.read_bytes())
        else:
            _write_image(test_dir / "test.jpg", (120, class_index * 20, 40))

    prepared = build_manifest(data_root, val_fraction=0.34, seed=42)
    rows = {row["relative_path"]: row for row in prepared.rows}

    train_copy = rows["seg_train/seg_train/buildings/train_0.jpg"]
    test_copy = rows["seg_test/seg_test/buildings/test_duplicate.jpg"]
    assert train_copy["assigned_split"] == "excluded"
    assert "duplicate_of_test" in train_copy["exclusion_reason"]
    assert test_copy["assigned_split"] == "test"

    retained_train_hashes = {
        row["sha256"] for row in prepared.rows if row["assigned_split"] in {"train", "validation"}
    }
    test_hashes = {row["sha256"] for row in prepared.rows if row["assigned_split"] == "test"}
    assert retained_train_hashes.isdisjoint(test_hashes)
    assert test_copy["sha256"] == hashlib.sha256(duplicate_train.read_bytes()).hexdigest()
