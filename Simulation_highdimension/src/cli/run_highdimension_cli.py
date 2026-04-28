from __future__ import annotations

import sys
from pathlib import Path

from simulation_second.src.cli.run_blueprint_cli import main as simulation_second_main


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "highdimension.yaml"


def main(argv: list[str] | None = None) -> int:
    """Run high-dimensional benchmark commands through the shared benchmark CLI."""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--config" not in args:
        args = ["--config", str(_default_config_path()), *args]
    return simulation_second_main(args)
