from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from py4ds_ai.models.bovw_cli import run_bovw_search


def test_run_bovw_search_rejects_empty_c_values_before_loading_manifest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="c_values"):
        run_bovw_search(
            tmp_path / "manifest-bundle" / "missing-manifest.csv",
            tmp_path / "output",
            vocab_sizes=(2,),
            c_values=(),
            max_descriptors=2,
        )


def test_run_bovw_script_exposes_vocabulary_and_split_options() -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/run_bovw.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--manifest" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--vocab-sizes" in result.stdout
    assert "--max-descriptors" in result.stdout
