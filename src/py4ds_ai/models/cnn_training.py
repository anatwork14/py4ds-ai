"""Train and validate a ResNet18 classifier without loading the held-out test split."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from py4ds_ai.data.manifest import CLASS_NAMES
from py4ds_ai.models.data import ImageRow, load_train_validation
from py4ds_ai.models.gradcam import gradcam_heatmap
from py4ds_ai.models.metrics import classification_metrics
from py4ds_ai.models.resnet_features import set_deterministic


class _LabeledImageDataset(Dataset):
    def __init__(self, rows: Sequence[ImageRow], transform) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row.path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, row.class_idx


def build_resnet18_classifier(
    *,
    pretrained: bool = True,
    fine_tune_layer4: bool = False,
) -> nn.Module:
    """Build six-class ResNet18 with a frozen backbone, optionally unfreezing layer4."""
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    if fine_tune_layer4:
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
    return model


def _early_stop_reached(no_improvement_epochs: int, *, patience: int) -> bool:
    """Return whether the configured number of consecutive non-improving epochs elapsed."""
    if no_improvement_epochs < 0 or patience < 0:
        raise ValueError("no_improvement_epochs and patience must be non-negative")
    return patience > 0 and no_improvement_epochs >= patience


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _metadata(manifest_path: Path, manifest_hash: str, device: torch.device, seed: int) -> dict:
    lock = json.loads((manifest_path.parent / "split.lock.json").read_text(encoding="utf-8"))
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    is_cuda = device.type == "cuda"
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "git_worktree_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        "python": __import__("sys").version,
        "platform": __import__("platform").platform(),
        "seed": seed,
        "device": torch.cuda.get_device_name(device) if is_cuda else "CPU",
        "gpu_capability": list(torch.cuda.get_device_capability(device)) if is_cuda else None,
        "torch_cuda_build": torch.version.cuda,
        "split_manifest_sha256": manifest_hash,
        "dataset_fingerprint_sha256": lock.get("dataset_fingerprint_sha256"),
        "class_to_idx": lock["class_to_idx"],
        "packages": {
            package: _package_version(package)
            for package in ("numpy", "Pillow", "torch", "torchvision")
        },
    }


def _worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def _resolve_device(device: str) -> torch.device:
    selected = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return selected


def _transforms(pretrained: bool):
    reference_weights = ResNet18_Weights.IMAGENET1K_V1
    reference_transform = reference_weights.transforms()
    mean = reference_transform.mean
    std = reference_transform.std
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    validation_transform = (
        reference_weights.transforms()
        if pretrained
        else transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    )
    return train_transform, validation_transform


def _set_training_mode(model: nn.Module, fine_tune_layer4: bool) -> None:
    model.train()
    if fine_tune_layer4:
        for module in (model.conv1, model.bn1, model.layer1, model.layer2, model.layer3):
            module.eval()
        model.layer4.train()
    else:
        model.eval()
    model.fc.train()


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
    fine_tune_layer4: bool,
) -> tuple[float, float]:
    _set_training_mode(model, fine_tune_layer4)
    total_loss = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        images = images.to(device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_function(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Training loss became non-finite")
        loss.backward()
        optimizer.step()
        batch_count = labels.shape[0]
        total_loss += float(loss.detach().item()) * batch_count
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        count += batch_count
    if count == 0:
        raise ValueError("Training loader produced no examples")
    return total_loss / count, correct / count


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    loss_function = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    labels_all: list[np.ndarray] = []
    predictions_all: list[np.ndarray] = []
    confidence_all: list[np.ndarray] = []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            labels_device = labels.to(device, dtype=torch.long, non_blocking=True)
            logits = model(images)
            total_loss += float(loss_function(logits, labels_device).item())
            probabilities = torch.softmax(logits, dim=1)
            confidence, predictions = probabilities.max(dim=1)
            labels_all.append(labels.numpy().astype(np.int64, copy=False))
            predictions_all.append(predictions.cpu().numpy().astype(np.int64, copy=False))
            confidence_all.append(confidence.cpu().numpy().astype(np.float32, copy=False))
    if not labels_all:
        raise ValueError("Validation loader produced no examples")
    labels = np.concatenate(labels_all)
    predictions = np.concatenate(predictions_all)
    confidence = np.concatenate(confidence_all)
    return total_loss / len(labels), labels, predictions, confidence


def _write_predictions(
    path: Path,
    rows: Sequence[ImageRow],
    predictions: np.ndarray,
    confidence: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("sample_id", "true_name", "predicted_name", "confidence"),
            lineterminator="\n",
        )
        writer.writeheader()
        for row, prediction, score in zip(rows, predictions, confidence, strict=True):
            writer.writerow(
                {
                    "sample_id": row.sample_id,
                    "true_name": row.class_name,
                    "predicted_name": CLASS_NAMES[int(prediction)],
                    "confidence": f"{float(score):.12g}",
                }
            )


def _save_confusion_matrix(metrics: dict, output_path: Path, title: str) -> None:
    matrix = np.asarray(metrics["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title)
    for row_index in range(len(CLASS_NAMES)):
        for column_index in range(len(CLASS_NAMES)):
            axis.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
                fontsize=8,
            )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _save_gradcam_examples(
    model: nn.Module,
    rows: Sequence[ImageRow],
    transform,
    predictions: np.ndarray,
    output_dir: Path,
    device: torch.device,
    *,
    max_examples: int = 6,
) -> list[dict[str, Any]]:
    errors = [
        i
        for i, (row, prediction) in enumerate(zip(rows, predictions, strict=True))
        if row.class_idx != int(prediction)
    ]
    selected = errors[:max_examples]
    if len(selected) < max_examples:
        selected.extend(
            i
            for i, (row, prediction) in enumerate(zip(rows, predictions, strict=True))
            if row.class_idx == int(prediction) and i not in selected
        )
    selected = selected[:max_examples]
    result: list[dict[str, Any]] = []
    destination = output_dir / "gradcam"
    destination.mkdir(parents=True, exist_ok=True)
    for index in selected:
        row = rows[index]
        with Image.open(row.path) as image:
            original = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            tensor = transform(image.convert("RGB"))
        predicted = int(predictions[index])
        heatmap = gradcam_heatmap(model, tensor, target_class=predicted, device=str(device))
        color = plt.get_cmap("jet")(heatmap)[..., :3]
        base = np.asarray(original, dtype=np.float32) / 255.0
        overlay = np.clip(0.55 * base + 0.45 * color, 0.0, 1.0)
        filename = f"{index:04d}-{row.class_name}-pred-{CLASS_NAMES[predicted]}.png"
        Image.fromarray(np.uint8(np.round(overlay * 255.0))).save(destination / filename)
        result.append(
            {
                "sample_id": row.sample_id,
                "true_name": row.class_name,
                "predicted_name": CLASS_NAMES[predicted],
                "selected_class_for_gradcam": CLASS_NAMES[predicted],
                "overlay": f"figures/gradcam/{filename}",
            }
        )
    return result


def run_cnn_experiment(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    pretrained: bool = True,
    epochs: int = 5,
    patience: int = 3,
    batch_size: int = 64,
    num_workers: int = 4,
    device: str = "auto",
    seed: int = 42,
    fine_tune_layer4: bool = False,
    learning_rate: float = 1e-3,
    backbone_learning_rate: float = 1e-5,
    weight_decay: float = 1e-4,
    generate_gradcam: bool = True,
) -> dict[str, Any]:
    """Train a ResNet18 head or layer4 fine-tune using locked train/validation rows only."""
    if epochs <= 0 or patience < 0 or batch_size <= 0 or num_workers < 0:
        raise ValueError(
            "epochs and batch_size must be positive; patience and num_workers must be non-negative"
        )
    if learning_rate <= 0 or backbone_learning_rate <= 0 or weight_decay < 0:
        raise ValueError("Learning rates must be positive and weight_decay non-negative")
    manifest_file = Path(manifest_path).expanduser().resolve()
    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Run output already exists; choose a new path: {target}")
    if target == manifest_file.parent or target.is_relative_to(manifest_file.parent):
        raise ValueError("Run output must not be written inside the locked manifest bundle")
    lock = json.loads((manifest_file.parent / "split.lock.json").read_text(encoding="utf-8"))
    source_root = Path(lock["source_root"]).expanduser().resolve()
    if target == source_root or target.is_relative_to(source_root):
        raise ValueError("Run output must not be written inside the read-only dataset root")
    train_rows, validation_rows, class_to_idx, manifest_hash = load_train_validation(manifest_file)
    expected_labels = set(range(len(CLASS_NAMES)))
    if {row.class_idx for row in train_rows} != expected_labels:
        raise ValueError("Training split must contain all six fixed classes")
    if {row.class_idx for row in validation_rows} != expected_labels:
        raise ValueError("Validation split must contain all six fixed classes")

    selected_device = _resolve_device(device)
    set_deterministic(seed)
    model = build_resnet18_classifier(
        pretrained=pretrained,
        fine_tune_layer4=fine_tune_layer4,
    ).to(selected_device)
    train_transform, validation_transform = _transforms(pretrained)
    train_dataset = _LabeledImageDataset(train_rows, train_transform)
    validation_dataset = _LabeledImageDataset(validation_rows, validation_transform)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=selected_device.type == "cuda",
        worker_init_fn=_worker_seed,
        generator=generator,
        persistent_workers=num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=selected_device.type == "cuda",
        worker_init_fn=_worker_seed,
        persistent_workers=num_workers > 0,
    )
    optimizer_groups: list[dict[str, Any]] = [
        {"params": list(model.fc.parameters()), "lr": learning_rate, "name": "fc"}
    ]
    if fine_tune_layer4:
        optimizer_groups.insert(
            0,
            {
                "params": list(model.layer4.parameters()),
                "lr": backbone_learning_rate,
                "name": "layer4",
            },
        )
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=weight_decay)
    loss_function = nn.CrossEntropyLoss()

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    started = time.perf_counter()
    try:
        weights_id = ResNet18_Weights.IMAGENET1K_V1.name if pretrained else "none"
        config = {
            "architecture": "torchvision ResNet18",
            "weights": weights_id,
            "weights_url": ResNet18_Weights.IMAGENET1K_V1.url if pretrained else None,
            "fine_tune_layer4": fine_tune_layer4,
            "trainable_parts": ["fc", "layer4"] if fine_tune_layer4 else ["fc"],
            "epochs": epochs,
            "patience": patience,
            "early_stopping_metric": "validation macro-F1; consecutive non-improving epochs",
            "batch_size": batch_size,
            "num_workers": num_workers,
            "head_learning_rate": learning_rate,
            "layer4_learning_rate": backbone_learning_rate if fine_tune_layer4 else None,
            "weight_decay": weight_decay,
            "optimizer": "AdamW",
            "loss": "CrossEntropyLoss",
            "train_augmentation": [
                "RandomResizedCrop(224, scale=(0.75,1.0))",
                "RandomHorizontalFlip(0.5)",
                "RandomRotation(10)",
                "ColorJitter(0.1,0.1,0.1,0.03)",
            ],
            "validation_transform": (
                "ResNet18_Weights.IMAGENET1K_V1.transforms()"
                if pretrained
                else "Resize(256), CenterCrop(224), ImageNet normalization"
            ),
            "selection_metric": "validation macro-F1; ties keep earliest epoch",
            "seed": seed,
            "device": torch.cuda.get_device_name(selected_device)
            if selected_device.type == "cuda"
            else "CPU",
            "test_status": "NOT RUN; test rows are not loaded by this experiment",
            "gradcam_selection_rule": (
                "up to six validation errors in manifest order, then correct validation rows"
                if generate_gradcam
                else "disabled"
            ),
        }
        (staging / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        metadata = _metadata(manifest_file, manifest_hash, selected_device, seed)
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        history: list[dict[str, Any]] = []
        best_macro_f1 = -1.0
        best_epoch = 0
        no_improvement_epochs = 0
        stopped_early = False
        best_checkpoint_path = staging / "best_checkpoint.pt"
        best_validation: dict[str, Any] | None = None
        best_predictions: np.ndarray | None = None
        best_confidence: np.ndarray | None = None
        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            train_loss, train_accuracy = _train_epoch(
                model,
                train_loader,
                optimizer,
                loss_function,
                selected_device,
                fine_tune_layer4,
            )
            validation_loss, labels, predictions, confidence = _evaluate(
                model,
                validation_loader,
                selected_device,
            )
            validation_metrics = classification_metrics(labels, predictions)
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_macro_f1": validation_metrics["macro_f1"],
                "validation_weighted_f1": validation_metrics["weighted_f1"],
                "epoch_seconds": time.perf_counter() - epoch_start,
            }
            history.append(record)
            print(
                f"epoch={epoch}/{epochs} train_loss={train_loss:.5f} "
                f"train_acc={train_accuracy:.4f} val_macro_f1="
                f"{validation_metrics['macro_f1']:.4f}",
                flush=True,
            )
            if validation_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = validation_metrics["macro_f1"]
                best_epoch = epoch
                best_validation = validation_metrics
                best_predictions = predictions.copy()
                best_confidence = confidence.copy()
                checkpoint = {
                    "schema_version": 1,
                    "epoch": epoch,
                    "model_state_dict": {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    },
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_macro_f1": best_macro_f1,
                    "manifest_sha256": manifest_hash,
                    "class_to_idx": class_to_idx,
                    "config": config,
                }
                temporary_checkpoint = staging / ".best_checkpoint.tmp.pt"
                torch.save(checkpoint, temporary_checkpoint)
                os.replace(temporary_checkpoint, best_checkpoint_path)
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1
                if _early_stop_reached(no_improvement_epochs, patience=patience):
                    stopped_early = epoch < epochs
                    if stopped_early:
                        print(
                            f"early_stopping epoch={epoch} best_epoch={best_epoch} "
                            f"patience={patience}",
                            flush=True,
                        )
                    break

        if best_validation is None or best_predictions is None or best_confidence is None:
            raise RuntimeError("Training completed without a validation checkpoint")
        checkpoint_data = torch.load(
            best_checkpoint_path, map_location=selected_device, weights_only=True
        )
        model.load_state_dict(checkpoint_data["model_state_dict"])
        _write_predictions(
            staging / "predictions_validation.csv",
            validation_rows,
            best_predictions,
            best_confidence,
        )
        with (staging / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(history)
        metrics = {
            "schema_version": 1,
            "test_status": "NOT RUN",
            "selection_metric": "validation macro-F1; ties keep earliest epoch",
            "best_epoch": best_epoch,
            "best_validation": best_validation,
            "epochs_completed": len(history),
            "stopped_early": stopped_early,
            "early_stopping_patience": patience,
            "epochs": history,
            "duration_seconds": time.perf_counter() - started,
        }
        (staging / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (staging / "best_model_config.json").write_text(
            json.dumps(
                {
                    "architecture": "torchvision ResNet18",
                    "weights": weights_id,
                    "fine_tune_layer4": fine_tune_layer4,
                    "best_epoch": best_epoch,
                    "validation_macro_f1": best_macro_f1,
                    "manifest_sha256": manifest_hash,
                    "class_to_idx": class_to_idx,
                    "test_status": "NOT RUN",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        figure_dir = staging / "figures"
        figure_dir.mkdir()
        epochs_axis = [record["epoch"] for record in history]
        figure, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(epochs_axis, [record["train_loss"] for record in history], label="train")
        axes[0].plot(
            epochs_axis, [record["validation_loss"] for record in history], label="validation"
        )
        axes[0].set_title("Cross-entropy loss")
        axes[1].plot(epochs_axis, [record["train_accuracy"] for record in history], label="train")
        axes[1].plot(
            epochs_axis, [record["validation_accuracy"] for record in history], label="validation"
        )
        axes[1].set_title("Accuracy")
        axes[2].plot(
            epochs_axis,
            [record["validation_macro_f1"] for record in history],
            label="validation",
        )
        axes[2].set_title("Validation macro-F1")
        for axis in axes:
            axis.set_xlabel("Epoch")
            axis.legend()
        figure.tight_layout()
        figure.savefig(figure_dir / "learning_curves.png", dpi=160)
        plt.close(figure)
        _save_confusion_matrix(
            best_validation,
            figure_dir / "best_validation_confusion_matrix.png",
            f"Validation confusion matrix — epoch {best_epoch}",
        )
        gradcam_examples: list[dict[str, Any]] = []
        if generate_gradcam:
            gradcam_examples = _save_gradcam_examples(
                model,
                validation_rows,
                validation_transform,
                best_predictions,
                figure_dir,
                selected_device,
            )
        (staging / "gradcam_examples.json").write_text(
            json.dumps(gradcam_examples, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "status": "completed",
        "output_dir": str(target),
        "manifest_sha256": manifest_hash,
        "train_count": len(train_rows),
        "validation_count": len(validation_rows),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "best_validation_macro_f1": best_macro_f1,
        "test_status": "NOT RUN",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train/fine-tune ResNet18 and select checkpoints using validation only."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Locked split_manifest.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="New run output directory")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Stop after this many non-improving validation macro-F1 epochs; 0 disables.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a PyTorch device string")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--fine-tune-layer4", action="store_true")
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the explicitly versioned ImageNet1K V1 weights (default: true).",
    )
    parser.add_argument(
        "--gradcam",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save validation Grad-CAM overlays for selected examples (default: true).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = run_cnn_experiment(
            args.manifest,
            args.output_dir,
            pretrained=args.pretrained,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            seed=args.seed,
            fine_tune_layer4=args.fine_tune_layer4,
            learning_rate=args.learning_rate,
            backbone_learning_rate=args.backbone_learning_rate,
            weight_decay=args.weight_decay,
            generate_gradcam=args.gradcam,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
