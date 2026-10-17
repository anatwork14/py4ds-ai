from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw

from py4ds_ai.data.manifest import CLASS_NAMES, build_manifest
from py4ds_ai.data.outputs import write_artifacts
from py4ds_ai.models.classical_cli import run_hog_search


def _write_pattern(path: Path, class_index: int, variant: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (64, 64), (20 + variant, 30, 40))
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (4 + class_index * 2, 5 + variant, 24 + class_index * 2, 24 + variant),
        fill=(220, 200 - class_index * 10, 100 + variant),
    )
    draw.line((0, 63 - class_index, 63, class_index), fill=(200, 40 + variant, 20), width=3)
    image.save(path)


def _make_six_class_dataset(root: Path) -> None:
    for class_index, class_name in enumerate(CLASS_NAMES):
        train_dir = root / "seg_train" / "seg_train" / class_name
        test_dir = root / "seg_test" / "seg_test" / class_name
        for variant in range(2):
            _write_pattern(train_dir / f"train-{variant}.png", class_index, variant)
        _write_pattern(train_dir / "validation-source.png", class_index, 5)
        _write_pattern(test_dir / "test.png", class_index, 9)


def test_hog_run_writes_validation_artifacts_without_reading_test_files(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    _make_six_class_dataset(data_root)
    prepared = build_manifest(data_root, val_fraction=0.34, seed=42)
    manifest_dir = tmp_path / "manifest"
    write_artifacts(prepared, manifest_dir, montage_per_class=1)
    test_files = list((data_root / "seg_test").rglob("*.png"))
    for path in test_files:
        path.unlink()

    output_dir = tmp_path / "run"
    result = run_hog_search(manifest_dir / "split_manifest.csv", output_dir, c_values=(0.1,))

    assert result["status"] == "completed"
    assert result["test_status"] == "NOT RUN"
    assert result["train_count"] > 0
    assert result["validation_count"] == len(CLASS_NAMES)
    assert result["trial_count"] == 2
    assert (output_dir / "best_model.joblib").is_file()
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["test_status"] == "NOT RUN"
    with (output_dir / "predictions_validation.csv").open(encoding="utf-8") as stream:
        predictions = list(csv.DictReader(stream))
    assert len(predictions) == (1 + result["trial_count"]) * result["validation_count"]
    assert all("seg_test" not in row["sample_id"] for row in predictions)
