from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image

from py4ds_ai.data.manifest import CLASS_NAMES
from py4ds_ai.models.cnn_training import (
    _early_stop_reached,
    build_resnet18_classifier,
    run_cnn_experiment,
)


def _write_synthetic_bundle(root: Path) -> Path:
    data_root = root / "raw"
    bundle = root / "manifests" / "seed-42"
    bundle.mkdir(parents=True)
    rows = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        for sample_no in range(2):
            path = data_root / "seg_train" / "seg_train" / class_name / f"train-{sample_no}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 24), (25 + class_idx * 20 + sample_no, 80, 160)).save(path)
            rows.append(
                {
                    "sample_id": f"{class_name}-train-{sample_no}",
                    "relative_path": path.relative_to(data_root).as_posix(),
                    "source_split": "seg_train",
                    "assigned_split": "train",
                    "class_name": class_name,
                    "class_idx": class_idx,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "phash": "",
                    "audit_status": "ok",
                    "exclusion_reason": "",
                }
            )
        path = data_root / "seg_train" / "seg_train" / class_name / "validation.png"
        Image.new("RGB", (28, 32), (35 + class_idx * 20, 100, 150)).save(path)
        rows.append(
            {
                "sample_id": f"{class_name}-validation",
                "relative_path": path.relative_to(data_root).as_posix(),
                "source_split": "seg_train",
                "assigned_split": "validation",
                "class_name": class_name,
                "class_idx": class_idx,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "phash": "",
                "audit_status": "ok",
                "exclusion_reason": "",
            }
        )
    rows.append(
        {
            "sample_id": "sealed-test-row",
            "relative_path": "missing-test.png",
            "source_split": "seg_test",
            "assigned_split": "test",
            "class_name": CLASS_NAMES[0],
            "class_idx": 0,
            "sha256": "sealed",
            "phash": "",
            "audit_status": "ok",
            "exclusion_reason": "",
        }
    )
    manifest = bundle / "split_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    mapping = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    (bundle / "class_to_idx.json").write_text(json.dumps(mapping), encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (bundle / "split.lock.json").write_text(
        json.dumps(
            {
                "source_root": str(data_root.resolve()),
                "manifest_sha256": digest,
                "dataset_fingerprint_sha256": "synthetic",
                "class_to_idx": mapping,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_early_stopping_requires_configured_patience() -> None:
    assert not _early_stop_reached(2, patience=3)
    assert _early_stop_reached(3, patience=3)
    assert not _early_stop_reached(3, patience=0)


def test_classifier_freezes_backbone_or_only_unfreezes_layer4() -> None:
    head_model = build_resnet18_classifier(pretrained=False, fine_tune_layer4=False)
    head_trainable = {name for name, param in head_model.named_parameters() if param.requires_grad}
    assert head_trainable
    assert all(name.startswith("fc.") for name in head_trainable)
    assert head_model.fc.out_features == len(CLASS_NAMES)

    fine_tune_model = build_resnet18_classifier(pretrained=False, fine_tune_layer4=True)
    fine_tune_trainable = {
        name for name, param in fine_tune_model.named_parameters() if param.requires_grad
    }
    assert any(name.startswith("layer4.") for name in fine_tune_trainable)
    assert all(name.startswith(("layer4.", "fc.")) for name in fine_tune_trainable)


def test_one_epoch_run_saves_checkpoint_and_validation_artifacts(tmp_path: Path) -> None:
    manifest = _write_synthetic_bundle(tmp_path)
    output = tmp_path / "runs" / "cnn-smoke"

    result = run_cnn_experiment(
        manifest,
        output,
        pretrained=False,
        epochs=1,
        batch_size=6,
        num_workers=0,
        device="cpu",
        seed=42,
        fine_tune_layer4=False,
        generate_gradcam=False,
    )

    assert result["status"] == "completed"
    assert result["train_count"] == 12
    assert result["validation_count"] == 6
    assert result["test_status"] == "NOT RUN"
    assert (output / "best_checkpoint.pt").is_file()
    assert (output / "history.csv").is_file()
    assert (output / "predictions_validation.csv").is_file()
    assert (output / "figures" / "learning_curves.png").is_file()
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["test_status"] == "NOT RUN"
    assert len(metrics["epochs"]) == 1
    assert not (output / "predictions_test.csv").exists()
