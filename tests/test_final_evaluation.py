from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from py4ds_ai.data.manifest import CLASS_NAMES
from py4ds_ai.models.cnn_training import build_resnet18_classifier
from py4ds_ai.models.evaluate_final import evaluate_final


def _write_bundle(root: Path) -> tuple[Path, str]:
    project = root / "project"
    source_root = project / "data" / "raw"
    bundle = project / "data" / "manifests" / "seed-42"
    bundle.mkdir(parents=True)
    rows = []
    for index, name in enumerate(CLASS_NAMES):
        rows.append(
            {
                "sample_id": f"train-{name}",
                "relative_path": f"seg_train/seg_train/{name}/missing-train.png",
                "source_split": "seg_train",
                "assigned_split": "train",
                "class_name": name,
                "class_idx": index,
                "sha256": "unused-train",
                "audit_status": "ok",
            }
        )
        rows.append(
            {
                "sample_id": f"validation-{name}",
                "relative_path": f"seg_train/seg_train/{name}/missing-validation.png",
                "source_split": "seg_train",
                "assigned_split": "validation",
                "class_name": name,
                "class_idx": index,
                "sha256": "unused-validation",
                "audit_status": "ok",
            }
        )
        path = source_root / "seg_test" / "seg_test" / name / "test.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (24, 32), (50 + index * 20, 90, 140)).save(path)
        rows.append(
            {
                "sample_id": f"test-{name}",
                "relative_path": path.relative_to(source_root).as_posix(),
                "source_split": "seg_test",
                "assigned_split": "test",
                "class_name": name,
                "class_idx": index,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "audit_status": "ok",
            }
        )
    manifest = bundle / "split_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    mapping = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    (bundle / "class_to_idx.json").write_text(json.dumps(mapping), encoding="utf-8")
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (bundle / "split.lock.json").write_text(
        json.dumps(
            {
                "source_root": str(source_root.resolve()),
                "manifest_sha256": manifest_hash,
                "class_to_idx": mapping,
            }
        ),
        encoding="utf-8",
    )
    return manifest, manifest_hash


def _write_selection(project_root: Path, manifest_hash: str) -> Path:
    run = project_root / "runs" / "cnn-smoke"
    run.mkdir(parents=True)
    config = {
        "architecture": "torchvision ResNet18",
        "weights": "none",
        "fine_tune_layer4": False,
    }
    (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
    model = build_resnet18_classifier(pretrained=False, fine_tune_layer4=False)
    checkpoint = run / "best_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "manifest_sha256": manifest_hash,
            "class_to_idx": {name: index for index, name in enumerate(CLASS_NAMES)},
            "config": config,
        },
        checkpoint,
    )
    selection = project_root / "configs" / "selection.json"
    selection.parent.mkdir(parents=True)
    selection.write_text(
        json.dumps(
            {
                "test_status": "NOT RUN",
                "manifest_sha256": manifest_hash,
                "selected_model": {
                    "model_id": "resnet18-cnn-head",
                    "checkpoint_path": "runs/cnn-smoke/best_checkpoint.pt",
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "config_path": "runs/cnn-smoke/config.json",
                    "validation_macro_f1": 0.9,
                    "test_status": "NOT RUN",
                },
            }
        ),
        encoding="utf-8",
    )
    return selection


def test_final_evaluation_reads_only_test_and_seals_second_attempt(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    manifest, manifest_hash = _write_bundle(tmp_path)
    selection = _write_selection(project_root, manifest_hash)
    output = project_root / "runs" / "final-test"

    result = evaluate_final(
        selection,
        manifest,
        output,
        project_root=project_root,
        device="cpu",
        batch_size=3,
        num_workers=0,
    )

    assert result["test_status"] == "SCORED ONCE"
    assert result["test_evaluations"] == 1
    assert result["test_count"] == len(CLASS_NAMES)
    assert result["manifest_sha256"] == manifest_hash
    predictions = list(csv.DictReader((output / "predictions_test.csv").open()))
    assert len(predictions) == len(CLASS_NAMES)
    assert {row["sample_id"] for row in predictions} == {f"test-{name}" for name in CLASS_NAMES}
    assert (output / "test_errors.csv").is_file()
    assert (output / "figures" / "test_confusion_matrix.png").is_file()
    assert not (output / "predictions_validation.csv").exists()

    with pytest.raises(FileExistsError, match="already exists"):
        evaluate_final(
            selection,
            manifest,
            project_root / "runs" / "final-test-repeat",
            project_root=project_root,
            device="cpu",
            batch_size=3,
            num_workers=0,
        )


def test_wrong_checkpoint_class_order_is_rejected_before_test_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = tmp_path / "project"
    manifest, manifest_hash = _write_bundle(tmp_path)
    selection_path = _write_selection(project_root, manifest_hash)
    checkpoint_path = project_root / "runs" / "cnn-smoke" / "best_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["class_to_idx"] = {name: index for index, name in enumerate(reversed(CLASS_NAMES))}
    torch.save(checkpoint, checkpoint_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["selected_model"]["checkpoint_sha256"] = hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    (project_root / "data" / "raw" / "seg_test" / "seg_test" / CLASS_NAMES[0] / "test.png").unlink()
    real_torch_load = torch.load
    load_options: list[dict] = []

    def recording_load(*args, **kwargs):
        load_options.append(kwargs)
        return real_torch_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)

    with pytest.raises(ValueError, match="Checkpoint class mapping"):
        evaluate_final(
            selection_path,
            manifest,
            project_root / "runs" / "final-test",
            project_root=project_root,
            device="cpu",
            batch_size=3,
            num_workers=0,
        )
    assert load_options and load_options[0].get("weights_only") is True
    assert not (project_root / "runs" / ".test_evaluation_locks" / f"{manifest_hash}.json").exists()
