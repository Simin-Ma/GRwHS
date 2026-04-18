"""Sweep launcher for simulation_project experiments."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation_project.src.run_sweep import _cli


if __name__ == "__main__":
    _cli()
