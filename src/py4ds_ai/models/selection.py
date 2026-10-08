"""Freeze a final model using validation scores only, before test evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validation_record(metrics: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    for key in ("best_classical_validation", "best_bovw_validation", "best_validation"):
        value = metrics.get(key)
        if isinstance(value, dict) and "macro_f1" in value:
            return key, value
    return None


def freeze_selection(
    runs_root: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Record the highest validation macro-F1 candidate without reading test images."""
    run_root = Path(runs_root).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"Runs directory does not exist: {run_root}")
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Split manifest does not exist: {manifest_file}")
    if output_file.exists():
        raise FileExistsError(
            f"Selection record already exists; refusing to overwrite: {output_file}"
        )
    try:
        project_root = Path(os.path.commonpath((run_root, manifest_file)))
    except ValueError as exc:
        raise ValueError("Runs directory and manifest must share a project root") from exc
    if not run_root.is_relative_to(project_root) or not manifest_file.is_relative_to(project_root):
        raise ValueError("Runs directory and manifest must be inside a common project root")
    if not output_file.is_relative_to(project_root):
        raise ValueError("Selection output must be inside the project root")

    manifest_hash = _sha256_file(manifest_file)
    lock_path = manifest_file.parent / "split.lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"Split lock is missing: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("manifest_sha256") != manifest_hash:
        raise ValueError("Manifest hash differs from its split lock")

    candidates: list[dict[str, Any]] = []
    for metrics_path in sorted(run_root.rglob("metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        extracted = _validation_record(metrics)
        if extracted is None:
            continue
        if metrics.get("test_status") != "NOT RUN":
            raise ValueError(f"Candidate has a test status other than NOT RUN: {metrics_path}")
        run_dir = metrics_path.parent
        metadata_path = run_dir / "metadata.json"
        config_path = run_dir / "config.json"
        if not metadata_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(f"Candidate is missing metadata.json or config.json: {run_dir}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("split_manifest_sha256") != manifest_hash:
            raise ValueError(f"Candidate manifest hash mismatch: {run_dir}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        metric_key, validation = extracted
        score = float(validation["macro_f1"])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"Invalid validation macro-F1 in {metrics_path}")
        run_id = str(run_dir.relative_to(project_root))
        checkpoint = run_dir / "best_checkpoint.pt"
        model_id = (
            metrics.get("best_classical_experiment_id")
            or metrics.get("best_bovw_experiment_id")
            or metrics.get("best_experiment_id")
            or f"resnet18-cnn-{'layer4' if config.get('fine_tune_layer4') else 'head'}"
        )
        candidates.append(
            {
                "run_id": run_id,
                "model_id": model_id,
                "validation_macro_f1": score,
                "validation_accuracy": validation.get("accuracy"),
                "metrics_key": metric_key,
                "metrics_path": str(metrics_path.relative_to(project_root)),
                "config_path": str(config_path.relative_to(project_root)),
                "checkpoint_path": (
                    str(checkpoint.relative_to(project_root)) if checkpoint.is_file() else None
                ),
                "checkpoint_sha256": _sha256_file(checkpoint) if checkpoint.is_file() else None,
                "manifest_sha256": manifest_hash,
                "test_status": "NOT RUN",
            }
        )
    if not candidates:
        raise ValueError(f"No validation-only model candidates found under {run_root}")
    candidates.sort(key=lambda item: (-item["validation_macro_f1"], item["run_id"]))
    selected = candidates[0]
    project_manifest = str(manifest_file.relative_to(project_root))
    result = {
        "schema_version": 1,
        "selection_created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selection_rule": "highest validation macro-F1; ties broken by ascending run_id",
        "manifest_path": project_manifest,
        "manifest_sha256": manifest_hash,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selected_model": selected,
        "test_status": "NOT RUN",
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_file.parent,
            prefix=f".{output_file.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary_path, output_file)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze the highest validation macro-F1 candidate before final test evaluation."
    )
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = freeze_selection(args.runs_root, args.manifest, args.output)
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
