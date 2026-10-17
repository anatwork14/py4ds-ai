from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

MODEL_LABELS = {
    "hog-logistic_regression-C0.01": "HOG + Logistic Regression, C=0.01",
    "sift-bovw-128-logistic_regression-C1": "SIFT-BoVW, 128 words + Logistic Regression, C=1",
    "resnet18-hybrid-linear_svc-C0.1": "Frozen ResNet18 embeddings + LinearSVC, C=0.1",
    "resnet18-cnn-head": "Pretrained ResNet18, frozen backbone / trained head",
    "resnet18-cnn-layer4": "Pretrained ResNet18, layer4 fine-tuned",
}

BLOCKS = {
    "summary": ("<!-- BEGIN GENERATED SUMMARY -->", "<!-- END GENERATED SUMMARY -->"),
    "validation": (
        "<!-- BEGIN GENERATED VALIDATION TABLE -->",
        "<!-- END GENERATED VALIDATION TABLE -->",
    ),
    "test": ("<!-- BEGIN GENERATED TEST TABLES -->", "<!-- END GENERATED TEST TABLES -->"),
    "errors": ("<!-- BEGIN GENERATED ERROR TABLE -->", "<!-- END GENERATED ERROR TABLE -->"),
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _resolve(root: Path, path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _format_validation(root: Path, selection: dict[str, Any]) -> str:
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Selection must contain at least one candidate")
    if selection.get("candidate_count") != len(candidates):
        raise ValueError("candidate_count does not match selection candidates")

    manifest_hash = selection.get("manifest_sha256")
    candidate_rows: list[tuple[dict[str, Any], float]] = []
    baseline: dict[str, Any] | None = None
    for candidate in candidates:
        if candidate.get("manifest_sha256") != manifest_hash:
            raise ValueError("Candidate manifest hash does not match selection manifest")
        metrics = _load_json(_resolve(root, candidate["metrics_path"]))
        metric_group = metrics.get(candidate["metrics_key"])
        if not isinstance(metric_group, dict):
            raise ValueError(f"Missing metrics key {candidate['metrics_key']!r}")
        score = _finite_number(metric_group.get("macro_f1"), "validation macro-F1")
        recorded_score = _finite_number(
            candidate.get("validation_macro_f1"), "candidate validation macro-F1"
        )
        if not math.isclose(score, recorded_score, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Candidate validation macro-F1 does not match saved metrics")
        candidate_rows.append((candidate, score))
        if baseline is None and isinstance(metrics.get("majority_baseline"), dict):
            baseline = metrics["majority_baseline"]

    selected = selection.get("selected_model")
    if not isinstance(selected, dict):
        raise ValueError("Selection is missing selected_model")
    selected_id = selected.get("model_id")
    selected_run = selected.get("run_id")
    if not any(
        candidate.get("model_id") == selected_id and candidate.get("run_id") == selected_run
        for candidate, _ in candidate_rows
    ):
        raise ValueError("Selected model is not present in the candidate list")
    winner = min(candidate_rows, key=lambda row: (-row[1], str(row[0].get("run_id", ""))))
    if winner[0].get("run_id") != selected_run:
        raise ValueError("Frozen selection is not the highest validation macro-F1 candidate")
    selected_score = _finite_number(
        selected.get("validation_macro_f1"), "selected validation macro-F1"
    )
    if not math.isclose(winner[1], selected_score, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Selected validation macro-F1 does not match candidate metrics")
    if selected != winner[0]:
        raise ValueError("Frozen selected_model must exactly match the winning candidate")
    if baseline is None:
        raise ValueError("No saved validation majority-baseline metrics were found")

    baseline_score = _finite_number(baseline.get("macro_f1"), "majority-baseline macro-F1")
    lines = [
        "| Comparator/reference | Validation macro-F1 | Test status |",
        "|---|---:|---|",
        f"| Majority-class baseline | {baseline_score:.6f} | Not evaluated |",
    ]
    for candidate, score in sorted(candidate_rows, key=lambda row: str(row[0].get("run_id", ""))):
        model_id = str(candidate.get("model_id", "unknown"))
        label = MODEL_LABELS.get(model_id, model_id)
        status = "Selected; evaluated once" if model_id == selected_id else "Not evaluated"
        lines.append(f"| {label} | {score:.6f} | {status} |")
    return "\n".join(lines)


def _validate_final_metrics(
    selection: dict[str, Any], metrics: dict[str, Any]
) -> tuple[list[str], list[list[int]]]:
    selected = selection["selected_model"]
    if metrics.get("selected_model_id") != selected.get("model_id"):
        raise ValueError("Final metrics do not match the frozen selected model")
    if metrics.get("manifest_sha256") != selection.get("manifest_sha256"):
        raise ValueError("Final metrics manifest hash does not match selection")
    checkpoint_hash = selected.get("checkpoint_sha256")
    if checkpoint_hash is not None and metrics.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Final metrics checkpoint hash does not match frozen selection")
    if metrics.get("test_status") != "SCORED ONCE" or metrics.get("test_evaluations") != 1:
        raise ValueError("Final report requires exactly one completed test evaluation")

    names = metrics.get("class_names")
    matrix = metrics.get("confusion_matrix")
    per_class = metrics.get("per_class")
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
        raise ValueError("Final metrics must include ordered class names")
    if not isinstance(matrix, list) or len(matrix) != len(names):
        raise ValueError("Confusion matrix must be square and match class names")

    checked_matrix: list[list[int]] = []
    for row in matrix:
        if not isinstance(row, list) or len(row) != len(names):
            raise ValueError("Confusion matrix must be square and match class names")
        checked_row: list[int] = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Confusion matrix entries must be nonnegative integers")
            checked_row.append(value)
        checked_matrix.append(checked_row)

    test_count = metrics.get("test_count")
    if isinstance(test_count, bool) or not isinstance(test_count, int) or test_count <= 0:
        raise ValueError("test_count must be a positive integer")
    if sum(sum(row) for row in checked_matrix) != test_count:
        raise ValueError("Confusion matrix total does not match test_count")
    if not isinstance(per_class, list) or len(per_class) != len(names):
        raise ValueError("Per-class metrics must match the class list")
    class_f1: list[float] = []
    for index, row in enumerate(per_class):
        support = sum(checked_matrix[index])
        if not isinstance(row, dict) or row.get("class_name") != names[index] or (
            row.get("support") != support
            or isinstance(row.get("support"), bool)
        ):
            raise ValueError("Per-class names/support do not match the confusion matrix")
        true_positives = checked_matrix[index][index]
        predicted_count = sum(matrix_row[index] for matrix_row in checked_matrix)
        precision = true_positives / predicted_count if predicted_count else 0.0
        recall = true_positives / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        for key, expected in (("precision", precision), ("recall", recall), ("f1", f1)):
            actual = _finite_number(row.get(key), f"per-class {key}")
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Per-class {key} does not match the confusion matrix")
        class_f1.append(f1)

    accuracy = _finite_number(metrics.get("accuracy"), "test accuracy")
    diagonal = sum(checked_matrix[i][i] for i in range(len(names)))
    if not math.isclose(accuracy, diagonal / test_count, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Accuracy does not match the confusion matrix")
    macro_f1 = _finite_number(metrics.get("macro_f1"), "test macro-F1")
    weighted_f1 = _finite_number(metrics.get("weighted_f1"), "test weighted-F1")
    if not math.isclose(macro_f1, sum(class_f1) / len(names), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Macro-F1 does not match the confusion matrix")
    expected_weighted = sum(
        score * sum(checked_matrix[index])
        for index, score in enumerate(class_f1)
    ) / test_count
    if not math.isclose(weighted_f1, expected_weighted, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Weighted-F1 does not match the confusion matrix")
    for key in ("parameters_total", "parameters_trainable"):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("evaluation_seconds", "seconds_per_image"):
        value = _finite_number(metrics.get(key), key)
        if value < 0:
            raise ValueError(f"{key} must be nonnegative")
    return names, checked_matrix


def _render_test_tables(metrics: dict[str, Any], names: list[str], matrix: list[list[int]]) -> str:
    test_count = metrics["test_count"]
    correct = sum(matrix[i][i] for i in range(len(names)))
    rows = [
        "| Metric | Result |",
        "|---|---:|",
        f"| Test examples | {test_count:,} |",
        f"| Accuracy | {metrics['accuracy']:.6f} |",
        f"| Macro-F1 | {metrics['macro_f1']:.6f} |",
        f"| Weighted-F1 | {metrics['weighted_f1']:.6f} |",
        f"| Misclassified examples | {test_count - correct:,} / {test_count:,} |",
        f"| Evaluations of this test split | {metrics['test_evaluations']} |",
        (
            f"| Parameters (total / trainable) | "
            f"{metrics['parameters_total']:,} / {metrics['parameters_trainable']:,} |"
        ),
        f"| Evaluation time | {metrics['evaluation_seconds']:.6f} s |",
        f"| Seconds per image | {metrics['seconds_per_image']:.9f} |",
        "",
        "| Class | Support | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in metrics["per_class"]:
        rows.append(
            f"| {item['class_name']} | {item['support']:,} | "
            f"{item['precision']:.6f} | {item['recall']:.6f} | {item['f1']:.6f} |"
        )
    return "\n".join(rows)


def _render_error_table(names: list[str], matrix: list[list[int]], limit: int = 4) -> str:
    errors = sorted(
        (
            (matrix[true_idx][pred_idx], true_idx, pred_idx)
            for true_idx in range(len(names))
            for pred_idx in range(len(names))
            if true_idx != pred_idx and matrix[true_idx][pred_idx] > 0
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )[:limit]
    lines = ["| True class | Predicted class | Count |", "|---|---|---:|"]
    lines.extend(f"| {names[i]} | {names[j]} | {count} |" for count, i, j in errors)
    return "\n".join(lines)


def _render_summary(selection: dict[str, Any], metrics: dict[str, Any]) -> str:
    model_id = metrics["selected_model_id"]
    label = MODEL_LABELS.get(model_id, model_id)
    validation_f1 = _finite_number(
        selection["selected_model"].get("validation_macro_f1"), "selected validation macro-F1"
    )
    correct = sum(metrics["confusion_matrix"][i][i] for i in range(len(metrics["class_names"])))
    errors = metrics["test_count"] - correct
    return (
        f"Validation-only selection chose **{label}** (macro-F1 {validation_f1:.6f}) from "
        f"{selection['candidate_count']} candidates. In its single final evaluation on "
        f"{metrics['test_count']:,} held-out images, "
        f"it achieved **{metrics['accuracy']:.2%} accuracy**, "
        f"**{metrics['macro_f1']:.6f} macro-F1**, "
        f"and **{metrics['weighted_f1']:.6f} weighted-F1**; "
        f"{errors:,} images were misclassified. "
        "These values are read from saved artifacts, not published benchmark results."
    )


def _replace_block(document: str, name: str, body: str) -> str:
    start_marker, end_marker = BLOCKS[name]
    if document.count(start_marker) != 1 or document.count(end_marker) != 1:
        raise ValueError(f"Report must contain exactly one {name} generated block")
    start = document.index(start_marker) + len(start_marker)
    end = document.index(end_marker)
    if end < start:
        raise ValueError(f"Malformed {name} generated block")
    return document[:start] + "\n" + body + "\n" + document[end:]


def render_report(
    project_root: Path | str,
    report_path: Path | str,
    selection_path: Path | str,
    final_metrics_path: Path | str,
) -> str:
    root = Path(project_root).resolve()
    report_file = _resolve(root, report_path)
    selection = _load_json(_resolve(root, selection_path))
    metrics = _load_json(_resolve(root, final_metrics_path))
    names, matrix = _validate_final_metrics(selection, metrics)
    validation_table = _format_validation(root, selection)
    test_tables = _render_test_tables(metrics, names, matrix)
    error_table = _render_error_table(names, matrix)
    summary = _render_summary(selection, metrics)

    document = report_file.read_text(encoding="utf-8")
    for name, body in (
        ("summary", summary),
        ("validation", validation_table),
        ("test", test_tables),
        ("errors", error_table),
    ):
        document = _replace_block(document, name, body)
    report_file.write_text(document, encoding="utf-8")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render report result blocks from saved experiment artifacts."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path, default=Path("docs/FINAL_REPORT.md"))
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("configs/final-selection-seed-42-phash-reviewed.json"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("runs/seed-42-phash-reviewed/final-evaluation/metrics.json"),
    )
    args = parser.parse_args()
    output = render_report(args.project_root, args.report, args.selection, args.metrics)
    print(f"Rendered result blocks in {args.report} ({len(output)} characters)")


if __name__ == "__main__":
    main()
