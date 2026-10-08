"""Run HOG and majority baselines using train/validation only."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from py4ds_ai.data.manifest import CLASS_NAMES
from py4ds_ai.features.hog import HOGConfig, extract_hog
from py4ds_ai.models.classical import majority_baseline, validation_search
from py4ds_ai.models.data import ImageRow, load_train_validation
from py4ds_ai.models.metrics import classification_metrics


def _json_write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _extract_matrix(rows: list[ImageRow], config: HOGConfig, split_name: str) -> np.ndarray:
    features: list[np.ndarray] = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        features.append(extract_hog(row.path, config))
        if index % 1000 == 0 or index == total:
            print(f"HOG {split_name}: {index}/{total}", flush=True)
    return np.stack(features).astype(np.float32, copy=False)


def _write_predictions(
    path: Path,
    experiment_id: str,
    rows: list[ImageRow],
    predictions: np.ndarray,
    scores: np.ndarray,
    score_type: str,
) -> None:
    already_exists = path.exists()
    with path.open("a" if already_exists else "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "experiment_id",
                "sample_id",
                "true_name",
                "predicted_name",
                "prediction_score",
                "score_type",
            ),
            lineterminator="\n",
        )
        if not already_exists:
            writer.writeheader()
        for row, prediction, score in zip(rows, predictions, scores, strict=True):
            writer.writerow(
                {
                    "experiment_id": experiment_id,
                    "sample_id": row.sample_id,
                    "true_name": row.class_name,
                    "predicted_name": CLASS_NAMES[int(prediction)],
                    "prediction_score": f"{float(score):.12g}",
                    "score_type": score_type,
                }
            )


def _metadata(manifest_path: Path, manifest_hash: str, output_dir: Path) -> dict[str, Any]:
    lock = json.loads((manifest_path.parent / "split.lock.json").read_text(encoding="utf-8"))
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "schema_version": 1,
        "run_id": output_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "git_worktree_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        "python": sys.version,
        "platform": platform.platform(),
        "device": "CPU",
        "seed": 42,
        "split_manifest_sha256": manifest_hash,
        "dataset_fingerprint_sha256": lock.get("dataset_fingerprint_sha256"),
        "class_to_idx": lock["class_to_idx"],
        "packages": {
            name: _package_version(name)
            for name in ("numpy", "scikit-learn", "opencv-python", "matplotlib")
        },
    }


def run_hog_search(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    c_values: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    seed: int = 42,
) -> dict[str, Any]:
    """Extract HOG for train/validation and run validation-only classifier search."""
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
    if set(y_train) != set(range(len(CLASS_NAMES))):
        raise ValueError("Training split does not contain all six fixed classes")
    if set(y_validation) != set(range(len(CLASS_NAMES))):
        raise ValueError("Validation split does not contain all six fixed classes")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        config = HOGConfig()
        resolved_config = {
            "feature": "HOG",
            "image_size_width_height": list(config.image_size),
            "orientations": config.orientations,
            "pixels_per_cell_width_height": list(config.pixels_per_cell),
            "cells_per_block_width_height": list(config.cells_per_block),
            "feature_count": config.feature_count,
            "classifiers": ["linear_svc", "logistic_regression"],
            "C_values": list(c_values),
            "selection_metric": "validation macro-F1",
            "seed": seed,
            "test_status": "NOT RUN; final test is intentionally unavailable to this search",
        }
        _json_write(staging / "config.json", resolved_config)
        _json_write(staging / "metadata.json", _metadata(manifest_file, manifest_hash, target))

        feature_start = time.perf_counter()
        X_train = _extract_matrix(train_rows, config, "train")
        X_validation = _extract_matrix(validation_rows, config, "validation")
        feature_seconds = time.perf_counter() - feature_start
        np.savez_compressed(
            staging / "hog_train.npz",
            X=X_train,
            y=y_train,
            sample_id=np.asarray([row.sample_id for row in train_rows]),
            sha256=np.asarray([row.sha256 for row in train_rows]),
            manifest_sha256=manifest_hash,
        )
        np.savez_compressed(
            staging / "hog_validation.npz",
            X=X_validation,
            y=y_validation,
            sample_id=np.asarray([row.sample_id for row in validation_rows]),
            sha256=np.asarray([row.sha256 for row in validation_rows]),
            manifest_sha256=manifest_hash,
        )

        baseline_predictions, baseline_class = majority_baseline(y_train, len(y_validation))
        baseline_metrics = classification_metrics(y_validation, baseline_predictions)
        baseline_id = f"majority-class-{CLASS_NAMES[baseline_class]}"
        prediction_path = staging / "predictions_validation.csv"
        _write_predictions(
            prediction_path,
            baseline_id,
            validation_rows,
            baseline_predictions,
            np.ones(len(y_validation), dtype=np.float64),
            "constant_baseline_score",
        )

        search_start = time.perf_counter()
        trials, best = validation_search(
            X_train,
            y_train,
            X_validation,
            y_validation,
            c_values=c_values,
            seed=seed,
        )
        search_seconds = time.perf_counter() - search_start
        records: list[dict[str, Any]] = [
            {
                "experiment_id": baseline_id,
                "feature": "none",
                "classifier": "majority_baseline",
                "validation": baseline_metrics,
                "selected_majority_class": CLASS_NAMES[baseline_class],
                "fit_seconds": 0.0,
            }
        ]
        models_dir = staging / "models"
        models_dir.mkdir()
        for trial in trials:
            train_predictions = trial.model.predict(X_train)
            train_metrics = classification_metrics(y_train, train_predictions)
            validation_metrics = classification_metrics(y_validation, trial.predictions)
            records.append(
                {
                    "experiment_id": trial.experiment_id,
                    "feature": "HOG",
                    "classifier": trial.classifier,
                    "C": trial.c_value,
                    "train": train_metrics,
                    "validation": validation_metrics,
                    "fit_seconds": trial.fit_seconds,
                    "prediction_score_type": trial.score_type,
                }
            )
            _write_predictions(
                prediction_path,
                trial.experiment_id,
                validation_rows,
                trial.predictions,
                trial.prediction_scores,
                trial.score_type,
            )
            joblib.dump(trial.model, models_dir / f"{trial.experiment_id}.joblib")

        metrics = {
            "schema_version": 1,
            "selection_metric": "validation macro-F1",
            "test_status": "NOT RUN",
            "feature_extraction_seconds": feature_seconds,
            "search_seconds": search_seconds,
            "majority_baseline": baseline_metrics,
            "majority_class": CLASS_NAMES[baseline_class],
            "trials": records[1:],
            "best_classical_experiment_id": best.experiment_id,
            "best_classical_validation": classification_metrics(y_validation, best.predictions),
        }
        _json_write(staging / "metrics.json", metrics)
        _json_write(
            staging / "best_classical_config.json",
            {
                "experiment_id": best.experiment_id,
                "feature": "HOG",
                "classifier": best.classifier,
                "C": best.c_value,
                "selection_metric": "validation macro-F1",
                "validation_macro_f1": best.validation_macro_f1,
                "split_manifest_sha256": manifest_hash,
                "class_to_idx": class_to_idx,
            },
        )
        shutil.copy2(models_dir / f"{best.experiment_id}.joblib", staging / "best_model.joblib")

        figure_dir = staging / "figures"
        figure_dir.mkdir()
        plot_names = [row["experiment_id"] for row in records]
        plot_scores = [row["validation"]["macro_f1"] for row in records]
        figure, axis = plt.subplots(figsize=(12, 5))
        axis.bar(range(len(plot_names)), plot_scores, color="#3976a8")
        axis.set_xticks(range(len(plot_names)), plot_names, rotation=45, ha="right")
        axis.set_ylabel("Validation macro-F1")
        axis.set_title("HOG and majority baseline — train/validation only")
        axis.set_ylim(0, 1)
        figure.tight_layout()
        figure.savefig(figure_dir / "validation_macro_f1.png", dpi=160)
        plt.close(figure)

        matrix = np.asarray(metrics["best_classical_validation"]["confusion_matrix"])
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
        "best_classical_experiment_id": best.experiment_id,
        "best_validation_macro_f1": best.validation_macro_f1,
        "test_status": "NOT RUN",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run HOG and majority baselines using only locked train/validation data."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Locked split_manifest.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="New run output directory")
    parser.add_argument(
        "--c-values",
        type=float,
        nargs="+",
        default=(0.01, 0.1, 1.0, 10.0),
        help="Positive classifier regularization values; selection uses validation macro-F1.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = run_hog_search(
            args.manifest,
            args.output_dir,
            c_values=args.c_values,
            seed=args.seed,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
