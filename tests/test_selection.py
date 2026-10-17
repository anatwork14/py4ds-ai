from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from py4ds_ai.models.selection import freeze_selection


def _write_run(
    runs_root: Path,
    name: str,
    score: float,
    *,
    manifest_hash: str,
    test_status: str = "NOT RUN",
) -> None:
    run = runs_root / name
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(
        json.dumps(
            {
                "best_validation": {"macro_f1": score},
                "best_epoch": 3,
                "test_status": test_status,
            }
        ),
        encoding="utf-8",
    )
    (run / "metadata.json").write_text(
        json.dumps({"split_manifest_sha256": manifest_hash}),
        encoding="utf-8",
    )
    (run / "config.json").write_text(
        json.dumps({"architecture": "torchvision ResNet18", "fine_tune_layer4": True}),
        encoding="utf-8",
    )
    (run / "best_checkpoint.pt").write_bytes(b"synthetic checkpoint")


def _write_manifest_bundle(root: Path) -> tuple[Path, str]:
    bundle = root / "data" / "manifests" / "seed-42"
    bundle.mkdir(parents=True)
    manifest = bundle / "split_manifest.csv"
    manifest.write_text("assigned_split\ntrain\nvalidation\ntest\n", encoding="utf-8")
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (bundle / "split.lock.json").write_text(
        json.dumps({"manifest_sha256": manifest_hash}), encoding="utf-8"
    )
    return manifest, manifest_hash


def test_freeze_selection_records_validation_winner_and_checkpoint_hash(tmp_path: Path) -> None:
    manifest, manifest_hash = _write_manifest_bundle(tmp_path)
    runs_root = tmp_path / "runs"
    _write_run(runs_root, "candidate-a", 0.89, manifest_hash=manifest_hash)
    _write_run(runs_root, "candidate-b", 0.93, manifest_hash=manifest_hash)
    output = tmp_path / "configs" / "final_selection.json"

    result = freeze_selection(runs_root, manifest, output)

    assert result["selected_model"]["run_id"] == "runs/candidate-b"
    assert result["selected_model"]["validation_macro_f1"] == pytest.approx(0.93)
    assert result["manifest_sha256"] == manifest_hash
    assert result["test_status"] == "NOT RUN"
    assert result["selected_model"]["checkpoint_sha256"] == hashlib.sha256(
        b"synthetic checkpoint"
    ).hexdigest()
    assert json.loads(output.read_text(encoding="utf-8")) == result


def test_freeze_selection_does_not_follow_predictable_temp_symlink(tmp_path: Path) -> None:
    manifest, manifest_hash = _write_manifest_bundle(tmp_path)
    runs_root = tmp_path / "runs"
    _write_run(runs_root, "candidate", 0.91, manifest_hash=manifest_hash)
    output = tmp_path / "configs" / "final_selection.json"
    output.parent.mkdir()
    protected = tmp_path / "protected.json"
    protected.write_text("keep this content", encoding="utf-8")
    predictable_temp = output.with_name(f".{output.name}.tmp")
    predictable_temp.symlink_to(protected)

    freeze_selection(runs_root, manifest, output)

    assert protected.read_text(encoding="utf-8") == "keep this content"
    assert output.is_file()
    assert predictable_temp.is_symlink()


def test_freeze_selection_rejects_mixed_manifests_and_test_scores(tmp_path: Path) -> None:
    manifest, manifest_hash = _write_manifest_bundle(tmp_path)
    runs_root = tmp_path / "runs"
    _write_run(runs_root, "valid", 0.89, manifest_hash=manifest_hash)
    _write_run(runs_root, "mismatched", 0.91, manifest_hash="different")

    with pytest.raises(ValueError, match="manifest"):
        freeze_selection(runs_root, manifest, tmp_path / "selection.json")

    (runs_root / "mismatched" / "metadata.json").write_text(
        json.dumps({"split_manifest_sha256": manifest_hash}), encoding="utf-8"
    )
    (runs_root / "mismatched" / "metrics.json").write_text(
        json.dumps({"best_validation": {"macro_f1": 0.91}, "test_status": "USED"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="test"):
        freeze_selection(runs_root, manifest, tmp_path / "selection.json")


def test_freeze_selection_accepts_nested_clean_run_root(tmp_path: Path) -> None:
    manifest, manifest_hash = _write_manifest_bundle(tmp_path)
    runs_root = tmp_path / "runs" / "seed-42-phash-reviewed"
    _write_run(runs_root, "candidate", 0.92, manifest_hash=manifest_hash)
    output = tmp_path / "configs" / "final_selection.json"

    result = freeze_selection(runs_root, manifest, output)

    assert result["selected_model"]["run_id"] == "runs/seed-42-phash-reviewed/candidate"
    assert result["manifest_sha256"] == manifest_hash
