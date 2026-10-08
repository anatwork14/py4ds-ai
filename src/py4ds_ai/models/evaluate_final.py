"""Single sealed evaluation of a preselected checkpoint on manifest test rows."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from py4ds_ai.data.manifest import CLASS_NAMES
from py4ds_ai.models.cnn_training import (
    _resolve_device,
    _save_confusion_matrix,
    _transforms,
    build_resnet18_classifier,
)
from py4ds_ai.models.data import ImageRow, _sha256_file
from py4ds_ai.models.metrics import classification_metrics


class _TestDataset(Dataset):
    def __init__(self, rows: Sequence[ImageRow], transform) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row.path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, row.class_idx, row.sample_id


def load_sealed_test(
    manifest_path: str | Path,
) -> tuple[list[ImageRow], dict[str, int], str]:
    """Read and hash-verify only assigned test image files from the locked manifest."""
    manifest_file = Path(manifest_path).expanduser().resolve()
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Split manifest does not exist: {manifest_file}")
    bundle = manifest_file.parent
    lock_path = bundle / "split.lock.json"
    class_map_path = bundle / "class_to_idx.json"
    if not lock_path.is_file() or not class_map_path.is_file():
        raise FileNotFoundError("Manifest bundle requires split.lock.json and class_to_idx.json")
    manifest_hash = _sha256_file(manifest_file)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("manifest_sha256") != manifest_hash:
        raise ValueError("Manifest hash differs from split lock")
    class_to_idx = json.loads(class_map_path.read_text(encoding="utf-8"))
    expected_mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    if class_to_idx != expected_mapping or lock.get("class_to_idx") != expected_mapping:
        raise ValueError("Class mapping differs from the fixed alphabetical mapping")
    source_root = Path(lock["source_root"]).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Locked dataset root is not a directory: {source_root}")

    rows: list[ImageRow] = []
    seen_ids: set[str] = set()
    with manifest_file.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            "sample_id",
            "relative_path",
            "source_split",
            "assigned_split",
            "class_name",
            "class_idx",
            "sha256",
            "audit_status",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Split manifest is missing required columns")
        for item in reader:
            if item["assigned_split"] != "test":
                continue
            sample_id = item["sample_id"]
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate test sample_id: {sample_id}")
            seen_ids.add(sample_id)
            if item["source_split"] != "seg_test" or item["audit_status"] != "ok":
                raise ValueError(f"Invalid source/audit status for test row: {sample_id}")
            class_name = item["class_name"]
            if class_name not in class_to_idx or int(item["class_idx"]) != class_to_idx[class_name]:
                raise ValueError(f"Invalid class mapping for test row: {sample_id}")
            relative = PurePosixPath(item["relative_path"])
            if relative.is_absolute() or ".." in relative.parts or "\\" in item["relative_path"]:
                raise ValueError(f"Unsafe test path: {item['relative_path']}")
            candidate = source_root.joinpath(*relative.parts)
            if candidate.is_symlink():
                raise ValueError(f"Test source became a symlink: {item['relative_path']}")
            source = candidate.resolve()
            if not source.is_relative_to(source_root):
                raise ValueError(f"Test path escapes locked dataset root: {item['relative_path']}")
            if not source.is_file():
                raise FileNotFoundError(f"Test image is missing: {item['relative_path']}")
            actual_hash = _sha256_file(source)
            if actual_hash != item["sha256"]:
                raise ValueError(f"Test source changed after audit: {item['relative_path']}")
            rows.append(
                ImageRow(
                    sample_id=sample_id,
                    path=source,
                    class_name=class_name,
                    class_idx=class_to_idx[class_name],
                    sha256=actual_hash,
                )
            )
    if not rows:
        raise ValueError("Locked manifest contains no test rows")
    if {row.class_idx for row in rows} != set(range(len(CLASS_NAMES))):
        raise ValueError("Test split must contain all six fixed classes")
    return rows, class_to_idx, manifest_hash


def _project_file(project_root: Path, relative_path: str, name: str) -> Path:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts or "\\" in relative_path:
        raise ValueError(f"Unsafe {name} path in selection record: {relative_path}")
    candidate = project_root.joinpath(*relative.parts).resolve()
    if not candidate.is_relative_to(project_root):
        raise ValueError(f"{name} path escapes project root: {relative_path}")
    return candidate


def _write_predictions(
    path: Path,
    rows: Sequence[ImageRow],
    predictions: np.ndarray,
    confidence: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    fields = ["sample_id", "true_name", "predicted_name", "confidence"]
    fields.extend(f"prob_{name}" for name in CLASS_NAMES)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(rows):
            record: dict[str, Any] = {
                "sample_id": row.sample_id,
                "true_name": row.class_name,
                "predicted_name": CLASS_NAMES[int(predictions[index])],
                "confidence": f"{float(confidence[index]):.12g}",
            }
            record.update(
                {
                    f"prob_{name}": f"{float(probabilities[index, class_index]):.12g}"
                    for class_index, name in enumerate(CLASS_NAMES)
                }
            )
            writer.writerow(record)


def evaluate_final(
    selection_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    project_root: str | Path,
    device: str = "auto",
    batch_size: int = 64,
    num_workers: int = 4,
) -> dict[str, Any]:
    """Score the frozen selected CNN on test once; an on-disk guard prevents repeats."""
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    root = Path(project_root).expanduser().resolve()
    selection_file = Path(selection_path).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    target = Path(output_dir).expanduser().resolve()
    if not selection_file.is_relative_to(root) or not manifest_file.is_relative_to(root):
        raise ValueError("Selection and manifest must be inside project_root")
    if target == root or not target.is_relative_to(root):
        raise ValueError("Output directory must be a subdirectory of project_root")
    if target.exists():
        raise FileExistsError(f"Final evaluation output already exists: {target}")

    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    if selection.get("test_status") != "NOT RUN":
        raise ValueError("Selection record is not in the pre-test state")
    manifest_hash = _sha256_file(manifest_file)
    lock_path = manifest_file.parent / "split.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("manifest_sha256") != manifest_hash:
        raise ValueError("Manifest hash differs from split lock")
    if selection.get("manifest_sha256") != manifest_hash:
        raise ValueError("Frozen selection uses a different manifest")

    selected = selection.get("selected_model")
    if not isinstance(selected, dict):
        raise ValueError("Selection record has no selected_model entry")
    if selected.get("test_status") != "NOT RUN":
        raise ValueError("Selected model has already used the test set")
    if not selected.get("model_id", "").startswith("resnet18-cnn-"):
        raise ValueError("Final evaluator currently supports only the selected ResNet18 CNN")
    checkpoint_path = _project_file(root, selected.get("checkpoint_path", ""), "checkpoint")
    config_path = _project_file(root, selected.get("config_path", ""), "config")
    if not checkpoint_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("Selected checkpoint or its run configuration is missing")
    checkpoint_hash = _sha256_file(checkpoint_path)
    if checkpoint_hash != selected.get("checkpoint_sha256"):
        raise ValueError("Selected checkpoint SHA-256 differs from the frozen selection record")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("architecture") != "torchvision ResNet18":
        raise ValueError("Selected checkpoint is not a torchvision ResNet18")
    weights = config.get("weights")
    if weights not in ("none", "IMAGENET1K_V1"):
        raise ValueError(f"Unsupported or unversioned ResNet18 weights: {weights}")
    fine_tune_layer4 = bool(config.get("fine_tune_layer4", False))
    selected_device = _resolve_device(device)
    model = build_resnet18_classifier(
        pretrained=False,
        fine_tune_layer4=fine_tune_layer4,
    ).to(selected_device)
    checkpoint = torch.load(checkpoint_path, map_location=selected_device, weights_only=True)
    if checkpoint.get("manifest_sha256") != manifest_hash:
        raise ValueError("Checkpoint was trained with a different manifest")
    expected_class_to_idx = {name: index for index, name in enumerate(CLASS_NAMES)}
    if checkpoint.get("class_to_idx") != expected_class_to_idx:
        raise ValueError("Checkpoint class mapping differs from the fixed alphabetical mapping")
    checkpoint_config = checkpoint.get("config", {})
    if checkpoint_config.get("weights") != weights:
        raise ValueError("Checkpoint and run configuration disagree on pretrained weights")
    if bool(checkpoint_config.get("fine_tune_layer4", False)) != fine_tune_layer4:
        raise ValueError("Checkpoint and run configuration disagree on fine-tuning mode")
    model.load_state_dict(checkpoint["model_state_dict"])
    _, test_transform = _transforms(pretrained=weights != "none")

    target.parent.mkdir(parents=True, exist_ok=True)
    guard_dir = root / "runs" / ".test_evaluation_locks"
    guard_dir.mkdir(parents=True, exist_ok=True)
    guard_path = guard_dir / f"{manifest_hash}.json"
    guard_record = {
        "state": "IN_PROGRESS",
        "manifest_sha256": manifest_hash,
        "selected_model_id": selected["model_id"],
        "selection_path": str(selection_file.relative_to(root)),
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output_dir": str(target.relative_to(root)),
    }
    try:
        with guard_path.open("x", encoding="utf-8") as stream:
            json.dump(guard_record, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as exc:
        raise FileExistsError(
            f"Test evaluation guard already exists for this manifest: {guard_path}"
        ) from exc

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        test_rows, class_to_idx, loaded_manifest_hash = load_sealed_test(manifest_file)
        if loaded_manifest_hash != manifest_hash:
            raise ValueError("Manifest changed while preparing test evaluation")
        generator = torch.Generator().manual_seed(int(config.get("seed", 42)))
        loader = DataLoader(
            _TestDataset(test_rows, test_transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=selected_device.type == "cuda",
            generator=generator,
            persistent_workers=num_workers > 0,
        )
        model.eval()
        labels_parts: list[np.ndarray] = []
        prediction_parts: list[np.ndarray] = []
        probability_parts: list[np.ndarray] = []
        sample_ids: list[str] = []
        if selected_device.type == "cuda":
            torch.cuda.synchronize(selected_device)
        started = time.perf_counter()
        with torch.inference_mode():
            for images, labels, batch_ids in loader:
                images = images.to(selected_device, dtype=torch.float32, non_blocking=True)
                logits = model(images)
                probabilities = torch.softmax(logits, dim=1)
                confidence, predictions = probabilities.max(dim=1)
                labels_parts.append(labels.numpy().astype(np.int64, copy=False))
                prediction_parts.append(predictions.cpu().numpy().astype(np.int64, copy=False))
                probability_parts.append(probabilities.cpu().numpy().astype(np.float32, copy=False))
                sample_ids.extend(batch_ids)
        if selected_device.type == "cuda":
            torch.cuda.synchronize(selected_device)
        elapsed = time.perf_counter() - started
        labels = np.concatenate(labels_parts)
        predictions = np.concatenate(prediction_parts)
        probabilities = np.concatenate(probability_parts)
        confidence = probabilities.max(axis=1)
        if len(labels) != len(test_rows) or sample_ids != [row.sample_id for row in test_rows]:
            raise RuntimeError("Test prediction order/count differs from the locked manifest")
        metrics = classification_metrics(labels, predictions)
        confusion = np.asarray(metrics["confusion_matrix"], dtype=np.float64)
        row_sums = confusion.sum(axis=1, keepdims=True)
        normalized_confusion = np.divide(
            confusion,
            row_sums,
            out=np.zeros_like(confusion),
            where=row_sums != 0,
        )
        metrics["normalized_confusion_matrix"] = normalized_confusion.tolist()
        metrics["test_status"] = "SCORED ONCE"
        metrics["test_evaluations"] = 1
        metrics["test_count"] = len(test_rows)
        metrics["selected_model_id"] = selected["model_id"]
        metrics["selection_validation_macro_f1"] = selected["validation_macro_f1"]
        metrics["manifest_sha256"] = manifest_hash
        metrics["checkpoint_sha256"] = checkpoint_hash
        metrics["evaluation_seconds"] = elapsed
        metrics["seconds_per_image"] = elapsed / len(test_rows)
        metrics["parameters_total"] = sum(parameter.numel() for parameter in model.parameters())
        metrics["parameters_trainable"] = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        metrics["device"] = (
            torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda" else "CPU"
        )
        metrics["class_to_idx"] = class_to_idx
        metrics["created_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

        _write_predictions(
            staging / "predictions_test.csv",
            test_rows,
            predictions,
            confidence,
            probabilities,
        )
        errors = [
            (row, int(prediction))
            for row, prediction in zip(test_rows, predictions, strict=True)
            if row.class_idx != int(prediction)
        ]
        with (staging / "test_errors.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("sample_id", "true_name", "predicted_name"),
                lineterminator="\n",
            )
            writer.writeheader()
            for row, prediction in errors:
                writer.writerow(
                    {
                        "sample_id": row.sample_id,
                        "true_name": row.class_name,
                        "predicted_name": CLASS_NAMES[prediction],
                    }
                )
        (staging / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (staging / "evaluation_config.json").write_text(
            json.dumps(
                {
                    "selection_path": str(selection_file.relative_to(root)),
                    "manifest_path": str(manifest_file.relative_to(root)),
                    "manifest_sha256": manifest_hash,
                    "selected_model_id": selected["model_id"],
                    "checkpoint_path": selected["checkpoint_path"],
                    "checkpoint_sha256": checkpoint_hash,
                    "test_status": "SCORED ONCE",
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "device": str(selected_device),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        figure_dir = staging / "figures"
        figure_dir.mkdir()
        _save_confusion_matrix(
            metrics,
            figure_dir / "test_confusion_matrix.png",
            "Final test confusion matrix — selected model",
        )
        os.replace(staging, target)
        guard_record["state"] = "COMPLETED"
        guard_record["completed_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        temporary_guard = guard_path.with_suffix(".tmp")
        temporary_guard.write_text(
            json.dumps(guard_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary_guard, guard_path)
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        guard_record["state"] = "INCOMPLETE_TEST_ATTEMPT"
        guard_record["error_type"] = type(exc).__name__
        temporary_guard = guard_path.with_suffix(".tmp")
        temporary_guard.write_text(
            json.dumps(guard_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary_guard, guard_path)
        raise

    return {
        "status": "completed",
        "output_dir": str(target),
        "selected_model_id": selected["model_id"],
        "test_status": "SCORED ONCE",
        "test_evaluations": 1,
        "test_count": len(test_rows),
        "manifest_sha256": manifest_hash,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen selected model on the locked test split exactly once."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = evaluate_final(
            args.selection,
            args.manifest,
            args.output_dir,
            project_root=args.project_root,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
