"""Run train/validation-only ResNet18 embeddings and hybrid classifiers."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import joblib
import matplotlib
import numpy as np
import torch
from torchvision.models import ResNet18_Weights

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from py4ds_ai.data.manifest import CLASS_NAMES
from py4ds_ai.models.classical import validation_search
from py4ds_ai.models.classical_cli import _json_write, _metadata, _write_predictions
from py4ds_ai.models.data import load_train_validation
from py4ds_ai.models.metrics import classification_metrics
from py4ds_ai.models.resnet_features import (
    build_resnet18_encoder,
    extract_embeddings,
    set_deterministic,
)


def run_resnet_search(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    pretrained: bool = True,
    batch_size: int = 64,
    num_workers: int = 4,
    device: str = "auto",
    c_values: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    seed: int = 42,
) -> dict[str, Any]:
    """Extract fixed ResNet18 embeddings and select hybrid classifier on validation."""
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
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
    y_train = np.asarray([row.class_idx for row in train_rows], dtype=np.int64)
    y_validation = np.asarray([row.class_idx for row in validation_rows], dtype=np.int64)
    expected_labels = set(range(len(CLASS_NAMES)))
    if set(y_train) != expected_labels or set(y_validation) != expected_labels:
        raise ValueError("Train and validation splits must each contain all six fixed classes")

    if device == "auto":
        selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    set_deterministic(seed)
    model, transform, weight_id = build_resnet18_encoder(pretrained=pretrained)
    selected_device_name = (
        torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda" else "CPU"
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        weight_url = ResNet18_Weights.IMAGENET1K_V1.url if pretrained else None
        config = {
            "architecture": "torchvision ResNet18",
            "weights": weight_id,
            "weights_url": weight_url,
            "embedding_dimension": 512,
            "input_transform": "ResNet18_Weights.IMAGENET1K_V1.transforms()",
            "batch_size": batch_size,
            "num_workers": num_workers,
            "device": selected_device_name,
            "device_type": selected_device.type,
            "classifier_search": ["linear_svc", "logistic_regression"],
            "C_values": list(c_values),
            "selection_metric": "validation macro-F1",
            "seed": seed,
            "test_status": "NOT RUN; test split is not loaded by this search",
        }
        _json_write(staging / "config.json", config)
        metadata = _metadata(manifest_file, manifest_hash, target)
        metadata.update(
            {
                "seed": seed,
                "device": selected_device_name,
                "torch_cuda_build": torch.version.cuda,
                "gpu_capability": list(torch.cuda.get_device_capability(selected_device))
                if selected_device.type == "cuda"
                else None,
                "packages": {
                    **metadata["packages"],
                    "torch": torch.__version__,
                    "torchvision": __import__("torchvision").__version__,
                },
            }
        )
        _json_write(staging / "metadata.json", metadata)

        started = time.perf_counter()
        X_train = extract_embeddings(
            model,
            train_rows,
            transform,
            batch_size=batch_size,
            num_workers=num_workers,
            device=str(selected_device),
        )
        X_validation = extract_embeddings(
            model,
            validation_rows,
            transform,
            batch_size=batch_size,
            num_workers=num_workers,
            device=str(selected_device),
        )
        embedding_seconds = time.perf_counter() - started
        np.savez_compressed(
            staging / "resnet18_train.npz",
            X=X_train,
            y=y_train,
            sample_id=np.asarray([row.sample_id for row in train_rows]),
            sha256=np.asarray([row.sha256 for row in train_rows]),
            manifest_sha256=manifest_hash,
            weight_id=weight_id,
        )
        np.savez_compressed(
            staging / "resnet18_validation.npz",
            X=X_validation,
            y=y_validation,
            sample_id=np.asarray([row.sample_id for row in validation_rows]),
            sha256=np.asarray([row.sha256 for row in validation_rows]),
            manifest_sha256=manifest_hash,
            weight_id=weight_id,
        )
        torch.save(
            {
                "architecture": "resnet18",
                "weight_id": weight_id,
                "weights_url": weight_url,
                "state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
            },
            staging / "encoder_state.pt",
        )

        search_started = time.perf_counter()
        trials, best = validation_search(
            X_train,
            y_train,
            X_validation,
            y_validation,
            c_values=c_values,
            experiment_prefix="resnet18-hybrid",
            seed=seed,
        )
        search_seconds = time.perf_counter() - search_started
        models_dir = staging / "models"
        models_dir.mkdir()
        prediction_path = staging / "predictions_validation.csv"
        records: list[dict[str, Any]] = []
        for trial in trials:
            record = {
                "experiment_id": trial.experiment_id,
                "feature": "pretrained ResNet18 512-d embedding",
                "weights": weight_id,
                "classifier": trial.classifier,
                "C": trial.c_value,
                "train": classification_metrics(y_train, trial.model.predict(X_train)),
                "validation": classification_metrics(y_validation, trial.predictions),
                "fit_seconds": trial.fit_seconds,
                "prediction_score_type": trial.score_type,
            }
            records.append(record)
            _write_predictions(
                prediction_path,
                trial.experiment_id,
                validation_rows,
                trial.predictions,
                trial.prediction_scores,
                trial.score_type,
            )
            joblib.dump(trial.model, models_dir / f"{trial.experiment_id}.joblib")

        best_metrics = classification_metrics(y_validation, best.predictions)
        metrics = {
            "schema_version": 1,
            "test_status": "NOT RUN",
            "selection_metric": "validation macro-F1",
            "embedding_extraction_seconds": embedding_seconds,
            "classifier_search_seconds": search_seconds,
            "trials": records,
            "best_experiment_id": best.experiment_id,
            "best_validation": best_metrics,
        }
        _json_write(staging / "metrics.json", metrics)
        _json_write(
            staging / "best_model_config.json",
            {
                "experiment_id": best.experiment_id,
                "architecture": "torchvision ResNet18",
                "weights": weight_id,
                "classifier": best.classifier,
                "C": best.c_value,
                "selection_metric": "validation macro-F1",
                "validation_macro_f1": best.validation_macro_f1,
                "split_manifest_sha256": manifest_hash,
                "class_to_idx": class_to_idx,
                "test_status": "NOT RUN",
            },
        )
        shutil.copy2(
            models_dir / f"{best.experiment_id}.joblib", staging / "best_classifier.joblib"
        )
        figure_dir = staging / "figures"
        figure_dir.mkdir()
        names = [record["experiment_id"] for record in records]
        scores = [record["validation"]["macro_f1"] for record in records]
        figure, axis = plt.subplots(figsize=(12, 5))
        axis.bar(range(len(names)), scores, color="#3976a8")
        axis.set_xticks(range(len(names)), names, rotation=45, ha="right")
        axis.set_ylabel("Validation macro-F1")
        axis.set_title("Pretrained ResNet18 embeddings — validation-only classifier selection")
        axis.set_ylim(0, 1)
        figure.tight_layout()
        figure.savefig(figure_dir / "validation_macro_f1.png", dpi=160)
        plt.close(figure)

        matrix = np.asarray(best_metrics["confusion_matrix"])
        figure, axis = plt.subplots(figsize=(7, 6))
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
        axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title(f"Validation confusion matrix — {best.experiment_id}")
        for row_index in range(len(CLASS_NAMES)):
            for column_index in range(len(CLASS_NAMES)):
                axis.text(column_index, row_index, str(matrix[row_index, column_index]),
                          ha="center", va="center", fontsize=8)
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(figure_dir / "best_validation_confusion_matrix.png", dpi=160)
        plt.close(figure)
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
        "trial_count": len(trials),
        "best_experiment_id": best.experiment_id,
        "best_validation_macro_f1": best.validation_macro_f1,
        "embedding_extraction_seconds": embedding_seconds,
        "test_status": "NOT RUN",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract ResNet18 features and tune a hybrid classifier on train/validation."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Locked split_manifest.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="New run output directory")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a PyTorch device string")
    parser.add_argument(
        "--pretrained", action=argparse.BooleanOptionalAction, default=True,
        help="Use the explicitly versioned ImageNet1K V1 weights (default: true).",
    )
    parser.add_argument("--c-values", type=float, nargs="+", default=(0.01, 0.1, 1.0, 10.0))
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = run_resnet_search(
            args.manifest,
            args.output_dir,
            pretrained=args.pretrained,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            c_values=args.c_values,
            seed=args.seed,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
