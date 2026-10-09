from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
render_report = importlib.import_module("render_report").render_report


MARKERS = (
    "<!-- BEGIN GENERATED SUMMARY -->\n<!-- END GENERATED SUMMARY -->\n"
    "<!-- BEGIN GENERATED VALIDATION TABLE -->\n<!-- END GENERATED VALIDATION TABLE -->\n"
    "<!-- BEGIN GENERATED TEST TABLES -->\n<!-- END GENERATED TEST TABLES -->\n"
    "<!-- BEGIN GENERATED ERROR TABLE -->\n<!-- END GENERATED ERROR TABLE -->\n"
)


def _write_fixture(root: Path, *, test_evaluations: int = 1) -> tuple[Path, Path, Path, Path]:
    report_path = root / "docs/FINAL_REPORT.md"
    selection_path = root / "configs/selection.json"
    candidate_metrics_path = root / "runs/candidate/metrics.json"
    final_metrics_path = root / "runs/final/metrics.json"

    for path in (report_path, selection_path, candidate_metrics_path, final_metrics_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(MARKERS, encoding="utf-8")
    candidate_metrics_path.write_text(
        json.dumps(
            {
                "best": {"macro_f1": 0.5},
                "majority_baseline": {"macro_f1": 0.1},
            }
        ),
        encoding="utf-8",
    )
    selection_path.write_text(
        json.dumps(
            {
                "candidate_count": 1,
                "manifest_sha256": "manifest-hash",
                "candidates": [
                    {
                        "metrics_path": "runs/candidate/metrics.json",
                        "metrics_key": "best",
                        "manifest_sha256": "manifest-hash",
                        "model_id": "candidate-a",
                        "run_id": "runs/candidate",
                        "test_status": "NOT RUN",
                        "validation_macro_f1": 0.5,
                        "checkpoint_sha256": "checkpoint-hash",
                    }
                ],
                "selected_model": {
                    "metrics_path": "runs/candidate/metrics.json",
                    "metrics_key": "best",
                    "manifest_sha256": "manifest-hash",
                    "model_id": "candidate-a",
                    "run_id": "runs/candidate",
                    "test_status": "NOT RUN",
                    "validation_macro_f1": 0.5,
                    "checkpoint_sha256": "checkpoint-hash",
                },
            }
        ),
        encoding="utf-8",
    )
    final_metrics_path.write_text(
        json.dumps(
            {
                "accuracy": 2 / 3,
                "macro_f1": 2 / 3,
                "weighted_f1": 2 / 3,
                "evaluation_seconds": 1.5,
                "seconds_per_image": 0.5,
                "parameters_total": 2,
                "parameters_trainable": 1,
                "checkpoint_sha256": "checkpoint-hash",
                "class_names": ["class-a", "class-b"],
                "confusion_matrix": [[1, 1], [0, 1]],
                "manifest_sha256": "manifest-hash",
                "per_class": [
                    {
                        "class_name": "class-a",
                        "precision": 1.0,
                        "recall": 0.5,
                        "f1": 2 / 3,
                        "support": 2,
                    },
                    {
                        "class_name": "class-b",
                        "precision": 0.5,
                        "recall": 1.0,
                        "f1": 2 / 3,
                        "support": 1,
                    },
                ],
                "selected_model_id": "candidate-a",
                "test_count": 3,
                "test_evaluations": test_evaluations,
                "test_status": "SCORED ONCE",
            }
        ),
        encoding="utf-8",
    )
    return root, report_path, selection_path, final_metrics_path


def test_render_report_generates_results_from_saved_artifacts(tmp_path: Path) -> None:
    root, report_path, selection_path, final_metrics_path = _write_fixture(tmp_path)

    rendered = render_report(root, report_path, selection_path, final_metrics_path)

    assert "candidate-a" in rendered
    assert "| Comparator/reference | Validation macro-F1 | Test status |" in rendered
    assert "0.500000" in rendered
    assert "0.666667" in rendered
    assert "Parameters (total / trainable) | 2 / 1" in rendered
    assert "Evaluation time | 1.500000 s" in rendered
    assert "Seconds per image | 0.500000" in rendered
    assert "| class-a | class-b | 1 |" in rendered
    assert rendered == report_path.read_text(encoding="utf-8")


def test_render_report_rejects_selection_score_mismatch(tmp_path: Path) -> None:
    root, report_path, selection_path, final_metrics_path = _write_fixture(tmp_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["selected_model"]["validation_macro_f1"] = 0.4
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    original = report_path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="validation macro-F1"):
        render_report(root, report_path, selection_path, final_metrics_path)

    assert report_path.read_text(encoding="utf-8") == original


def test_render_report_rejects_repeated_test_evaluation_without_writing(tmp_path: Path) -> None:
    root, report_path, selection_path, final_metrics_path = _write_fixture(
        tmp_path, test_evaluations=2
    )
    original = report_path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one"):
        render_report(root, report_path, selection_path, final_metrics_path)

    assert report_path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("field", ["macro_f1", "weighted_f1"])
def test_render_report_rejects_inconsistent_aggregate_f1(tmp_path: Path, field: str) -> None:
    root, report_path, selection_path, final_metrics_path = _write_fixture(tmp_path)
    metrics = json.loads(final_metrics_path.read_text(encoding="utf-8"))
    metrics[field] = 0.7
    final_metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    original = report_path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="F1 does not match the confusion matrix"):
        render_report(root, report_path, selection_path, final_metrics_path)

    assert report_path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("field", ["precision", "recall", "f1"])
def test_render_report_rejects_inconsistent_per_class_scores(
    tmp_path: Path, field: str
) -> None:
    root, report_path, selection_path, final_metrics_path = _write_fixture(tmp_path)
    metrics = json.loads(final_metrics_path.read_text(encoding="utf-8"))
    metrics["per_class"][0][field] = 0.3
    final_metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    original = report_path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match=f"Per-class {field} does not match"):
        render_report(root, report_path, selection_path, final_metrics_path)

    assert report_path.read_text(encoding="utf-8") == original


def test_render_report_rejects_checkpoint_hash_mismatch(tmp_path: Path) -> None:
    root, report_path, selection_path, final_metrics_path = _write_fixture(tmp_path)
    metrics = json.loads(final_metrics_path.read_text(encoding="utf-8"))
    metrics["checkpoint_sha256"] = "another-checkpoint"
    final_metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    original = report_path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint hash"):
        render_report(root, report_path, selection_path, final_metrics_path)

    assert report_path.read_text(encoding="utf-8") == original


def test_render_report_rejects_modified_selected_candidate(tmp_path: Path) -> None:
    root, report_path, selection_path, final_metrics_path = _write_fixture(tmp_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["selected_model"]["checkpoint_sha256"] = "different-hash"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    original = report_path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint hash does not match frozen selection"):
        render_report(root, report_path, selection_path, final_metrics_path)

    assert report_path.read_text(encoding="utf-8") == original
