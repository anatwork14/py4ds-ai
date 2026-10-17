"""Run training-only SIFT-BoVW vocabulary and classifier searches."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from py4ds_ai.data.manifest import CLASS_NAMES
from py4ds_ai.features.sift_bovw import SIFTBoVW, extract_sift_descriptors
from py4ds_ai.models.classical import validation_search
from py4ds_ai.models.classical_cli import (
    _json_write,
    _metadata,
    _write_predictions,
)
from py4ds_ai.models.data import ImageRow, load_train_validation
from py4ds_ai.models.metrics import classification_metrics


def _extract_descriptors(rows: list[ImageRow], split: str) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for index, row in enumerate(rows, start=1):
        result.append(extract_sift_descriptors(row.path))
        if index % 1000 == 0 or index == len(rows):
            print(f"SIFT {split}: {index}/{len(rows)}", flush=True)
    return result


def _check_sizes(vocab_sizes: Sequence[int], max_descriptors: int) -> None:
    if not vocab_sizes or any(size <= 0 for size in vocab_sizes):
        raise ValueError("vocab_sizes must contain positive integers")
    if len(set(vocab_sizes)) != len(vocab_sizes):
        raise ValueError("vocab_sizes must not contain duplicates")
    if max_descriptors < max(vocab_sizes):
        raise ValueError("max_descriptors must be at least the largest vocabulary size")


def run_bovw_search(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    vocab_sizes: Sequence[int] = (64, 128, 256),
    c_values: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    max_descriptors: int = 120_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Tune SIFT-BoVW and linear classifiers on train/validation, never test."""
    if not c_values:
        raise ValueError("c_values must not be empty")
    _check_sizes(vocab_sizes, max_descriptors)
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
    if set(y_train) != set(range(len(CLASS_NAMES))) or set(y_validation) != set(
        range(len(CLASS_NAMES))
    ):
        raise ValueError("Train and validation splits must each contain all six fixed classes")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        config = {
            "feature": "SIFT bag of visual words",
            "image_size_width_height": [128, 128],
            "descriptor": "OpenCV SIFT, 128-dimensional float32",
            "vocab_sizes": list(vocab_sizes),
            "max_descriptors_per_vocabulary": max_descriptors,
            "descriptor_sampling": "deterministic uniform sample from training descriptors only",
            "histogram_normalization": "L2 per image; zero-descriptor image maps to zero vector",
            "classifiers": ["linear_svc", "logistic_regression"],
            "C_values": list(c_values),
            "selection_metric": "validation macro-F1",
            "seed": seed,
            "test_status": "NOT RUN; final test is intentionally unavailable to this search",
        }
        _json_write(staging / "config.json", config)
        _json_write(staging / "metadata.json", _metadata(manifest_file, manifest_hash, target))

        feature_start = time.perf_counter()
        train_descriptors = _extract_descriptors(train_rows, "train")
        validation_descriptors = _extract_descriptors(validation_rows, "validation")
        extraction_seconds = time.perf_counter() - feature_start
        records: list[dict[str, Any]] = []
        candidates = []
        prediction_path = staging / "predictions_validation.csv"
        models_dir = staging / "models"
        models_dir.mkdir()
        cache_dir = staging / "features"
        cache_dir.mkdir()

        for vocab_size in vocab_sizes:
            bovw = SIFTBoVW(
                vocab_size=vocab_size,
                max_descriptors=max_descriptors,
                seed=seed,
            ).fit(train_descriptors)
            X_train = bovw.transform(train_descriptors)
            X_validation = bovw.transform(validation_descriptors)
            np.savez_compressed(
                cache_dir / f"bovw-{vocab_size}-train.npz",
                X=X_train,
                y=y_train,
                sample_id=np.asarray([row.sample_id for row in train_rows]),
                manifest_sha256=manifest_hash,
                descriptor_count=bovw.descriptor_count_,
            )
            np.savez_compressed(
                cache_dir / f"bovw-{vocab_size}-validation.npz",
                X=X_validation,
                y=y_validation,
                sample_id=np.asarray([row.sample_id for row in validation_rows]),
                manifest_sha256=manifest_hash,
                descriptor_count=bovw.descriptor_count_,
            )
            joblib.dump(bovw, models_dir / f"sift-vocabulary-{vocab_size}.joblib")
            trials, _ = validation_search(
                X_train,
                y_train,
                X_validation,
                y_validation,
                c_values=c_values,
                experiment_prefix=f"sift-bovw-{vocab_size}",
                seed=seed,
            )
            candidates.extend((trial, bovw) for trial in trials)
            for trial in trials:
                train_metrics = classification_metrics(y_train, trial.model.predict(X_train))
                validation_metrics = classification_metrics(y_validation, trial.predictions)
                records.append(
                    {
                        "experiment_id": trial.experiment_id,
                        "feature": "SIFT-BoVW",
                        "vocab_size": vocab_size,
                        "classifier": trial.classifier,
                        "C": trial.c_value,
                        "train": train_metrics,
                        "validation": validation_metrics,
                        "fit_seconds": trial.fit_seconds,
                        "prediction_score_type": trial.score_type,
                        "training_descriptors_used": bovw.descriptor_count_,
                        "training_descriptors_available": bovw.available_descriptor_count_,
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

        best_trial, best_bovw = min(
            candidates,
            key=lambda item: (
                -item[0].validation_macro_f1,
                -item[0].validation_accuracy,
                item[0].experiment_id,
            ),
        )
        search_seconds = time.perf_counter() - feature_start - extraction_seconds
        best_record = next(
            item for item in records if item["experiment_id"] == best_trial.experiment_id
        )
        metrics = {
            "schema_version": 1,
            "selection_metric": "validation macro-F1",
            "test_status": "NOT RUN",
            "sift_extraction_seconds": extraction_seconds,
            "total_search_and_vectorization_seconds": search_seconds,
            "training_descriptors_available": best_bovw.available_descriptor_count_,
            "training_descriptors_used": best_bovw.descriptor_count_,
            "trials": records,
            "best_bovw_experiment_id": best_trial.experiment_id,
            "best_bovw_validation": best_record["validation"],
        }
        _json_write(staging / "metrics.json", metrics)
        _json_write(
            staging / "best_bovw_config.json",
            {
                "experiment_id": best_trial.experiment_id,
                "feature": "SIFT-BoVW",
                "vocab_size": int(best_trial.experiment_id.split("-")[2]),
                "classifier": best_trial.classifier,
                "C": best_trial.c_value,
                "selection_metric": "validation macro-F1",
                "validation_macro_f1": best_trial.validation_macro_f1,
                "split_manifest_sha256": manifest_hash,
                "class_to_idx": class_to_idx,
            },
        )
        joblib.dump(
            {"bovw": best_bovw, "classifier_pipeline": best_trial.model},
            staging / "best_model.joblib",
        )

        figure_dir = staging / "figures"
        figure_dir.mkdir()
        figure, axis = plt.subplots(figsize=(14, 5))
        names = [item["experiment_id"] for item in records]
        scores = [item["validation"]["macro_f1"] for item in records]
        axis.bar(range(len(names)), scores, color="#3976a8")
        axis.set_xticks(range(len(names)), names, rotation=75, ha="right")
        axis.set_ylabel("Validation macro-F1")
        axis.set_title("SIFT-BoVW classifier search — train/validation only")
        axis.set_ylim(0, 1)
        figure.tight_layout()
        figure.savefig(figure_dir / "validation_macro_f1.png", dpi=160)
        plt.close(figure)

        best_cm = np.asarray(best_record["validation"]["confusion_matrix"])
        figure, axis = plt.subplots(figsize=(7, 6))
        image = axis.imshow(best_cm, cmap="Blues")
        axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
        axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title(f"Validation confusion matrix — {best_trial.experiment_id}")
        for row_index in range(len(CLASS_NAMES)):
            for column_index in range(len(CLASS_NAMES)):
                axis.text(
                    column_index,
                    row_index,
                    str(best_cm[row_index, column_index]),
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(figure_dir / "best_validation_confusion_matrix.png", dpi=160)
        plt.close(figure)
        (staging / "feature_extraction_summary.json").write_text(
            json.dumps(
                {
                    "train_sample_count": len(train_rows),
                    "validation_sample_count": len(validation_rows),
                    "zero_descriptor_train_images": sum(
                        len(value) == 0 for value in train_descriptors
                    ),
                    "zero_descriptor_validation_images": sum(
                        len(value) == 0 for value in validation_descriptors
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        shutil.move(str(staging), str(target))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "status": "completed",
        "output_dir": str(target),
        "manifest_sha256": manifest_hash,
        "train_count": len(train_rows),
        "validation_count": len(validation_rows),
        "trial_count": len(records),
        "best_bovw_experiment_id": best_trial.experiment_id,
        "best_validation_macro_f1": best_trial.validation_macro_f1,
        "test_status": "NOT RUN",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune SIFT-BoVW vocabularies and classifiers on locked train/validation splits."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Locked split_manifest.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="New run output directory")
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=(64, 128, 256))
    parser.add_argument("--c-values", type=float, nargs="+", default=(0.01, 0.1, 1.0, 10.0))
    parser.add_argument("--max-descriptors", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = run_bovw_search(
            args.manifest,
            args.output_dir,
            vocab_sizes=args.vocab_sizes,
            c_values=args.c_values,
            max_descriptors=args.max_descriptors,
            seed=args.seed,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
