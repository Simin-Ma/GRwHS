from __future__ import annotations

# Public experiment entrypoint facade.
from .experiments import (
    run_ga_v2_group_separation,
    run_ga_v2_complexity_mismatch,
    run_ga_v2_correlation_stress,
)
from .experiments.orchestration import _cli, run_all_experiments

__all__ = [
    "run_ga_v2_group_separation",
    "run_ga_v2_complexity_mismatch",
    "run_ga_v2_correlation_stress",
    "run_all_experiments",
    "_cli",
]


if __name__ == "__main__":
    _cli()

