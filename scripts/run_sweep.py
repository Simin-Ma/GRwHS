"""Sweep launcher for simulation_project experiments.

On Windows, process-pool parallelism is disabled by default for stability.
Set SIM_ALLOW_WINDOWS_PROCESS_POOL=1 to force-enable it from a spawn-safe script entrypoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation_project.src.run_sweep import _cli  # noqa: E402


if __name__ == "__main__":
    _cli()
