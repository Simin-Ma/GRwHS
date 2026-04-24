from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_simulation_cli_help_runs() -> None:
    cmd = [sys.executable, "-m", "simulation_project.src.run_experiment", "--help"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "--experiment {all,1,2,3,3a,3b,3c,3d,4,5,analysis}" in proc.stdout
    assert "--skip-analysis" in proc.stdout
    assert "--no-archive-artifacts" in proc.stdout


def test_simulation_cli_profile_flag_removed() -> None:
    cmd = [sys.executable, "-m", "simulation_project.src.run_experiment", "--experiment", "1", "--profile", "laptop"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "unrecognized arguments: --profile laptop" in (proc.stderr or "")
