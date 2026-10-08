from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_run_classical_script_exposes_reproducible_options() -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/run_classical.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--manifest" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--c-values" in result.stdout
    assert "--seed" in result.stdout
