from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_prepare_data_script_exposes_required_help() -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/prepare_data.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--data-root" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--val-fraction" in result.stdout
    assert "--seed" in result.stdout


def test_prepare_data_cli_reports_missing_root_without_traceback(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    missing_root = tmp_path / "no-such-dataset"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_data.py",
            "--data-root",
            str(missing_root),
            "--output-dir",
            str(tmp_path / "output"),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Dataset root is not a directory" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "output").exists()


def test_download_cli_help_exposes_explicit_terms_and_safe_extraction() -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/download_data.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--accept-terms" in result.stdout
    assert "--extract-to" in result.stdout


def test_download_cli_refuses_without_terms_acknowledgement(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "raw"
    result = subprocess.run(
        [sys.executable, "scripts/download_data.py", "--output-dir", str(output_dir)],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "review the Kaggle terms" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output_dir.exists()
