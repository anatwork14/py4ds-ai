from __future__ import annotations

from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from py4ds_ai.data.manifest import CLASS_NAMES, build_manifest, find_phash_candidates


def test_perceptual_hash_matches_are_review_flags_not_exclusions() -> None:
    rows = [
        {
            "sample_id": "train/a.jpg",
            "sha256": "sha-a",
            "source_split": "seg_train",
            "phash": "0000000000000000",
            "assigned_split": "train",
            "class_name": "buildings",
        },
        {
            "sample_id": "test/b.jpg",
            "sha256": "sha-b",
            "source_split": "seg_test",
            "phash": "0000000000000001",
            "assigned_split": "test",
            "class_name": "buildings",
        },
    ]

    candidates = find_phash_candidates(rows, hamming_threshold=1)

    assert len(candidates) == 1
    assert candidates[0]["sample_id_a"] == "test/b.jpg"
    assert candidates[0]["sample_id_b"] == "train/a.jpg"
    assert candidates[0]["hamming_distance"] == 1
    assert candidates[0]["requires_review"] is True
    assert rows[0]["assigned_split"] == "train"
    assert rows[1]["assigned_split"] == "test"


def _write_png_variant(path: Path, color: tuple[int, int, int], note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = PngInfo()
    metadata.add_text("variant", note)
    Image.new("RGB", (64, 64), color).save(path, format="PNG", pnginfo=metadata)


def test_manifest_flags_phash_matches_without_excluding_different_files(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    for class_index, class_name in enumerate(CLASS_NAMES):
        train_dir = data_root / "seg_train" / "seg_train" / class_name
        test_dir = data_root / "seg_test" / "seg_test" / class_name
        color = (20 + class_index, 60, 100)
        _write_png_variant(train_dir / "train_a.png", color, f"train-a-{class_name}")
        _write_png_variant(train_dir / "train_b.png", color, f"train-b-{class_name}")
        _write_png_variant(test_dir / "test.png", color, f"test-{class_name}")

    prepared = build_manifest(data_root, val_fraction=0.5, seed=42, phash_hamming_threshold=0)
    rows = {row["relative_path"]: row for row in prepared.rows}
    train_row = rows["seg_train/seg_train/buildings/train_a.png"]
    test_row = rows["seg_test/seg_test/buildings/test.png"]

    assert train_row["sha256"] != test_row["sha256"]
    assert train_row["phash"] == test_row["phash"]
    assert train_row["assigned_split"] in {"train", "validation"}
    assert test_row["assigned_split"] == "test"
    assert any(
        candidate["sample_id_a"] == test_row["sample_id"]
        and candidate["sample_id_b"] == train_row["sample_id"]
        for candidate in prepared.leakage_candidates
    )
