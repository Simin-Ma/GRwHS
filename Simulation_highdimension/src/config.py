from __future__ import annotations

from pathlib import Path

from simulation_second.src.config import BenchmarkConfig, load_benchmark_config


def default_config_path() -> Path:
    """Return the default high-dimensional benchmark YAML path."""
    return Path(__file__).resolve().parents[1] / "config" / "highdimension.yaml"


def load_highdimension_config(path: str | Path | None = None) -> BenchmarkConfig:
    """Load the high-dimensional benchmark config.

    This reuses the `simulation_second` config parser so the high-dimensional
    suite stays compatible with the same runner, table builder, and pairing
    protocol as the low-dimensional benchmark.
    """
    return load_benchmark_config(default_config_path() if path is None else path)
