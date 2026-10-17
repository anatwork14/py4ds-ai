from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from py4ds_ai.data.manifest import CLASS_NAMES, build_manifest


def _write_image(path: Path, color: tuple[int, int, int], size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format="JPEG", quality=95)


def _make_minimal_dataset(root: Path) -> None:
    for class_index, class_name in enumerate(CLASS_NAMES):
        train_root = root / "seg_train" / "seg_train" / class_name
        test_root = root / "seg_test" / "seg_test" / class_name
        _write_image(train_root / "square.jpg", (20 + class_index, 40, 60), (150, 150))
        _write_image(train_root / "wide.jpg", (70, 80 + class_index, 90), (200, 100))
        _write_image(test_root / "test.jpg", (100, 110, 120 + class_index), (150, 150))


def _snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_non_square_images_are_audited_without_modifying_source(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    _make_minimal_dataset(data_root)
    before = _snapshot(data_root)

    prepared = build_manifest(data_root, val_fraction=0.5, seed=42)

    after = _snapshot(data_root)
    assert after == before
    rows = {row["relative_path"]: row for row in prepared.rows}
    wide_row = rows["seg_train/seg_train/buildings/wide.jpg"]
    assert (wide_row["width"], wide_row["height"]) == (200, 100)
    assert (wide_row["mode"], wide_row["channels"]) == ("RGB", 3)
    assert wide_row["audit_status"] == "ok"
    assert wide_row["assigned_split"] in {"train", "validation"}
    assert prepared.class_to_idx == {name: index for index, name in enumerate(CLASS_NAMES)}


def test_missing_expected_class_directory_fails_loudly(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    for class_index, class_name in enumerate(CLASS_NAMES[:-1]):
        _write_image(
            data_root / "seg_train" / "seg_train" / class_name / "train.jpg",
            (class_index, 20, 30),
            (64, 64),
        )
        _write_image(
            data_root / "seg_test" / "seg_test" / class_name / "test.jpg",
            (class_index, 40, 50),
            (64, 64),
        )

    with pytest.raises(ValueError, match="missing expected class directories"):
        build_manifest(data_root)


def test_symlink_images_are_recorded_without_reading_targets(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    _make_minimal_dataset(data_root)
    outside_image = tmp_path / "outside.jpg"
    _write_image(outside_image, (240, 20, 10), (64, 64))
    link_path = data_root / "seg_train" / "seg_train" / "buildings" / "external.jpg"
    link_path.symlink_to(outside_image)

    prepared = build_manifest(data_root, val_fraction=0.5, seed=42)

    link_row = next(row for row in prepared.rows if row["relative_path"].endswith("external.jpg"))
    assert link_row["audit_status"] == "symlink_ignored"
    assert link_row["assigned_split"] == "excluded"
    assert link_row["sha256"] == ""
    assert link_path.is_symlink()


def test_decompression_bomb_is_recorded_as_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "dataset"
    _make_minimal_dataset(data_root)
    oversized = data_root / "seg_train" / "seg_train" / "buildings" / "oversized.jpg"
    oversized.write_bytes(b"oversized-image-fixture")

    def raise_decompression_bomb(*args: object, **kwargs: object) -> None:
        raise Image.DecompressionBombError("image exceeds pixel limit")

    original_open = Image.open

    def open_with_bomb(path: Path, *args: object, **kwargs: object) -> object:
        if Path(path) == oversized:
            return raise_decompression_bomb()
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Image, "open", open_with_bomb)
    prepared = build_manifest(data_root, val_fraction=0.5, seed=42)

    row = next(row for row in prepared.rows if row["relative_path"].endswith("oversized.jpg"))
    assert row["audit_status"] == "unreadable"
    assert row["assigned_split"] == "excluded"
    assert row["exclusion_reason"] == "unreadable:DecompressionBombError"
