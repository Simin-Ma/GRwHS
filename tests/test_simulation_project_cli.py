from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_simulation_second_cli_help_runs() -> None:
    cmd = [sys.executable, "-m", "simulation_second.src.run_blueprint", "--help"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "run-benchmark" in proc.stdout
    assert "build-tables" in proc.stdout


def test_old_simulation_project_cli_route_is_not_active() -> None:
    cmd = [sys.executable, "-m", "simulation_project.src.run_experiment", "--help"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0
