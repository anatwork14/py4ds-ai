from __future__ import annotations

import csv
import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
verify_module = importlib.import_module("verify_saved_final_artifacts")


def _fixture(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    selection_path = root / "selection.json"
    metrics_path = root / "metrics.json"
    predictions_path = root / "predictions_test.csv"
    errors_path = root / "test_errors.csv"
    guard_path = root / "manifest-hash.json"
    names = list(verify_module.CLASS_NAMES)
    rows = [
        {"sample_id": f"sample-{index}", "true_name": name,
         "predicted_name": "mountain" if name == "glacier" else name}
        for index, name in enumerate(names)
    ]
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "true_name", "predicted_name"])
        writer.writeheader()
        writer.writerows(rows)
    with errors_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "true_name", "predicted_name"])
        writer.writeheader()
        writer.writerow(rows[2])

    matrix = [[int(i == j) for j in range(6)] for i in range(6)]
    matrix[2][2] = 0
    matrix[2][3] = 1
    per_class = []
    for i, name in enumerate(names):
        support = sum(matrix[i])
        predicted = sum(matrix[j][i] for j in range(6))
        tp = matrix[i][i]
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        per_class.append(
            {"class_name": name, "support": support, "precision": precision,
             "recall": recall, "f1": f1}
        )
    metrics = {
        "selected_model_id": "resnet18-cnn-layer4",
        "manifest_sha256": "manifest-hash",
        "checkpoint_sha256": "checkpoint-hash",
        "test_status": "SCORED ONCE",
        "test_evaluations": 1,
        "test_count": len(rows),
        "class_names": names,
        "confusion_matrix": matrix,
        "accuracy": 5 / 6,
        "macro_f1": sum(item["f1"] for item in per_class) / 6,
        "weighted_f1": sum(item["f1"] for item in per_class) / 6,
        "per_class": per_class,
        "normalized_confusion_matrix": [[float(v) for v in row] for row in matrix],
    }
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    selection_path.write_text(json.dumps({
        "manifest_sha256": "manifest-hash",
        "selected_model": {
            "model_id": "resnet18-cnn-layer4",
            "checkpoint_sha256": "checkpoint-hash",
        },
    }), encoding="utf-8")
    guard_path.write_text(json.dumps({"state": "COMPLETED"}), encoding="utf-8")
    return selection_path, metrics_path, predictions_path, errors_path, guard_path


def _verify(paths: tuple[Path, Path, Path, Path, Path]) -> dict:
    return verify_module.verify_saved_artifacts(*paths)


def test_saved_predictions_reproduce_final_metrics_without_images(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    result = _verify(paths)

    assert result["status"] == "VERIFIED_FROM_SAVED_PREDICTIONS"
    assert result["test_count"] == 6
    assert result["misclassified_count"] == 1
    assert result["accuracy"] == 5 / 6


def test_saved_evidence_rejects_tampered_confusion_matrix(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    metrics = json.loads(paths[1].read_text(encoding="utf-8"))
    metrics["confusion_matrix"][2][3] = 0
    paths[1].write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(ValueError, match="confusion matrix differs"):
        _verify(paths)


def test_saved_evidence_rejects_tampered_predictions(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    data = paths[2].read_text(encoding="utf-8").replace("sample-0", "sample-1")
    paths[2].write_text(data, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate sample_id"):
        _verify(paths)


def test_saved_evidence_rejects_inconsistent_f1(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    metrics = json.loads(paths[1].read_text(encoding="utf-8"))
    metrics["macro_f1"] = 0.99
    paths[1].write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(ValueError, match="macro-F1 does not agree"):
        _verify(paths)


def test_saved_evidence_rejects_incomplete_guard(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths[4].write_text(json.dumps({"state": "INCOMPLETE_TEST_ATTEMPT"}), encoding="utf-8")

    with pytest.raises(ValueError, match="not recorded as completed"):
        _verify(paths)


def test_saved_evidence_rejects_wrong_recorded_sha256(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    with pytest.raises(ValueError, match="prediction SHA-256"):
        verify_module.verify_saved_artifacts(
            *paths,
            expected_predictions_sha256="0" * 64,
        )
