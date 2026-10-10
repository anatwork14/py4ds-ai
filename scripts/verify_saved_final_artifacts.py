"""Audit the existing final-test predictions; never load images or run the model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

CLASS_NAMES = ("buildings", "forest", "glacier", "mountain", "sea", "street")
REQUIRED_PREDICTION_FIELDS = {"sample_id", "true_name", "predicted_name"}
REQUIRED_ERROR_FIELDS = {"sample_id", "true_name", "predicted_name"}


def _read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return result


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing columns in {path}: {sorted(required)}")
        return list(reader)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _equal_number(actual: Any, expected: float, name: str) -> None:
    if isinstance(actual, bool) or not isinstance(actual, (float, int)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(float(actual)) or not math.isclose(
        float(actual), expected, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"{name} does not agree with saved predictions")


def verify_saved_artifacts(
    selection_path: Path,
    metrics_path: Path,
    predictions_path: Path,
    errors_path: Path,
    guard_path: Path,
    *,
    manifest_path: Path | None = None,
    expected_metrics_sha256: str | None = None,
    expected_predictions_sha256: str | None = None,
) -> dict[str, Any]:
    """Independently recompute reported scores from the *saved* per-image prediction CSV."""
    if expected_metrics_sha256 is not None and _sha256(metrics_path) != (
        expected_metrics_sha256
    ):
        raise ValueError("Saved metrics SHA-256 differs from documented value")
    if expected_predictions_sha256 is not None and _sha256(predictions_path) != (
        expected_predictions_sha256
    ):
        raise ValueError("Saved prediction SHA-256 differs from documented value")

    selection = _read_json(selection_path)
    metrics = _read_json(metrics_path)
    guard = _read_json(guard_path)
    selected = selection.get("selected_model")
    if not isinstance(selected, dict):
        raise ValueError("Selection is missing selected_model")
    if metrics.get("selected_model_id") != selected.get("model_id"):
        raise ValueError("Final selected model differs from frozen selection")
    if metrics.get("manifest_sha256") != selection.get("manifest_sha256"):
        raise ValueError("Final manifest differs from frozen selection")
    if metrics.get("checkpoint_sha256") != selected.get("checkpoint_sha256"):
        raise ValueError("Final checkpoint differs from frozen selection")
    if metrics.get("class_names") != list(CLASS_NAMES):
        raise ValueError("Metrics do not use the fixed six-class order")
    if guard.get("state") != "COMPLETED" or metrics.get("test_status") != "SCORED ONCE":
        raise ValueError("The final evaluation was not recorded as completed")
    if metrics.get("test_evaluations") != 1:
        raise ValueError("The recorded test evaluation count is not one")
    if metrics.get("manifest_sha256") not in guard_path.name:
        raise ValueError("Guard filename does not match the frozen manifest")

    predictions = _read_csv(predictions_path, REQUIRED_PREDICTION_FIELDS)
    if not predictions:
        raise ValueError("Saved prediction CSV is empty")
    if manifest_path is not None:
        if _sha256(manifest_path) != selection.get("manifest_sha256"):
            raise ValueError("Locked split manifest SHA-256 differs from frozen selection")
        manifest_rows = _read_csv(
            manifest_path,
            {"sample_id", "class_name", "assigned_split", "source_split", "audit_status"},
        )
        test_rows = [row for row in manifest_rows if row["assigned_split"] == "test"]
        if len(test_rows) != len(predictions):
            raise ValueError("Locked manifest test count differs from saved predictions")
        for manifest_row, prediction_row in zip(test_rows, predictions, strict=True):
            if manifest_row["source_split"] != "seg_test" or (
                manifest_row["audit_status"] != "ok"
            ):
                raise ValueError("Locked manifest has an invalid test source/audit status")
            if (
                manifest_row["sample_id"] != prediction_row["sample_id"]
                or manifest_row["class_name"] != prediction_row["true_name"]
            ):
                raise ValueError(
                    "Saved prediction IDs, order or true labels differ from locked manifest"
                )
    matrix = [[0] * len(CLASS_NAMES) for _ in CLASS_NAMES]
    name_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
    seen_ids: set[str] = set()
    computed_errors: list[dict[str, str]] = []
    for row in predictions:
        sample_id = row["sample_id"]
        if not sample_id or sample_id in seen_ids:
            raise ValueError("Saved predictions have a blank or duplicate sample_id")
        seen_ids.add(sample_id)
        true_name = row["true_name"]
        predicted_name = row["predicted_name"]
        if true_name not in name_to_index or predicted_name not in name_to_index:
            raise ValueError(f"Saved predictions contain an invalid class for {sample_id}")
        matrix[name_to_index[true_name]][name_to_index[predicted_name]] += 1
        if true_name != predicted_name:
            computed_errors.append(
                {"sample_id": sample_id, "true_name": true_name,
                 "predicted_name": predicted_name}
            )

    if metrics.get("test_count") != len(predictions):
        raise ValueError("Saved prediction count differs from final metrics")
    if metrics.get("confusion_matrix") != matrix:
        raise ValueError("Saved confusion matrix differs from prediction CSV")
    recorded_errors = _read_csv(errors_path, REQUIRED_ERROR_FIELDS)
    if [
        {field: row[field] for field in ("sample_id", "true_name", "predicted_name")}
        for row in recorded_errors
    ] != computed_errors:
        raise ValueError("Saved test_errors.csv differs from prediction CSV")

    n = len(predictions)
    diag = sum(matrix[i][i] for i in range(len(CLASS_NAMES)))
    _equal_number(metrics.get("accuracy"), diag / n, "accuracy")
    per_class = metrics.get("per_class")
    if not isinstance(per_class, list) or len(per_class) != len(CLASS_NAMES):
        raise ValueError("Per-class metrics are missing or incomplete")
    f1_values: list[float] = []
    supports: list[int] = []
    for i, name in enumerate(CLASS_NAMES):
        support = sum(matrix[i])
        predicted_count = sum(matrix[j][i] for j in range(len(CLASS_NAMES)))
        tp = matrix[i][i]
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (support + predicted_count) if support + predicted_count else 0.0
        recorded = per_class[i]
        if not isinstance(recorded, dict) or recorded.get("class_name") != name or (
            recorded.get("support") != support
        ):
            raise ValueError(f"Per-class label/support differs for {name}")
        for key, score in (("precision", precision), ("recall", recall), ("f1", f1)):
            _equal_number(recorded.get(key), score, f"{name} {key}")
        f1_values.append(f1)
        supports.append(support)

    _equal_number(metrics.get("macro_f1"), sum(f1_values) / len(CLASS_NAMES), "macro-F1")
    _equal_number(
        metrics.get("weighted_f1"),
        sum(f1 * support for f1, support in zip(f1_values, supports, strict=True)) / n,
        "weighted-F1",
    )
    normalized = metrics.get("normalized_confusion_matrix")
    if normalized is not None:
        if not isinstance(normalized, list) or len(normalized) != len(CLASS_NAMES):
            raise ValueError("Normalized confusion matrix has invalid shape")
        for i, row in enumerate(normalized):
            if not isinstance(row, list) or len(row) != len(CLASS_NAMES):
                raise ValueError("Normalized confusion matrix has invalid shape")
            for j, actual in enumerate(row):
                expected = matrix[i][j] / supports[i] if supports[i] else 0.0
                _equal_number(actual, expected, f"normalized matrix ({i},{j})")

    return {
        "status": "VERIFIED_FROM_SAVED_PREDICTIONS",
        "test_count": n,
        "misclassified_count": len(computed_errors),
        "selected_model_id": selected["model_id"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "note": "No images, checkpoints or model inference were accessed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only consistency check of previously saved final-test artifacts."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--selection", type=Path,
        default=Path("configs/final-selection-seed-42-phash-reviewed.json"),
    )
    parser.add_argument(
        "--metrics", type=Path,
        default=Path("runs/seed-42-phash-reviewed/final-evaluation/metrics.json"),
    )
    parser.add_argument(
        "--predictions", type=Path,
        default=Path("runs/seed-42-phash-reviewed/final-evaluation/predictions_test.csv"),
    )
    parser.add_argument(
        "--errors", type=Path,
        default=Path("runs/seed-42-phash-reviewed/final-evaluation/test_errors.csv"),
    )
    parser.add_argument("--guard", type=Path, default=None)
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="Locked split manifest (defaults to the path in the frozen selection)",
    )
    parser.add_argument("--expected-metrics-sha256", type=str, default=None)
    parser.add_argument("--expected-predictions-sha256", type=str, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()

    def resolve(value: Path) -> Path:
        return value if value.is_absolute() else root / value

    selection_path = resolve(args.selection)
    selection = _read_json(selection_path)
    guard_path = (
        resolve(args.guard) if args.guard is not None
        else root / "runs/.test_evaluation_locks" / f"{selection['manifest_sha256']}.json"
    )
    manifest_relative = args.manifest or Path(selection["manifest_path"])
    result = verify_saved_artifacts(
        selection_path, resolve(args.metrics), resolve(args.predictions),
        resolve(args.errors), guard_path,
        manifest_path=resolve(manifest_relative),
        expected_metrics_sha256=args.expected_metrics_sha256,
        expected_predictions_sha256=args.expected_predictions_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
