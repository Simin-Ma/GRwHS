from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "run_ga_v2_group_separation",
    "run_ga_v2_complexity_mismatch",
    "run_ga_v2_correlation_stress",
    "run_all_experiments",
]

_LAZY_EXPORTS = {
    "run_ga_v2_group_separation": (
        "simulation_project.src.experiments.exp_ga_v2_group_separation",
        "run_ga_v2_group_separation",
    ),
    "run_ga_v2_complexity_mismatch": (
        "simulation_project.src.experiments.exp_ga_v2_complexity_mismatch",
        "run_ga_v2_complexity_mismatch",
    ),
    "run_ga_v2_correlation_stress": (
        "simulation_project.src.experiments.exp_ga_v2_correlation_stress",
        "run_ga_v2_correlation_stress",
    ),
    "run_all_experiments": (
        "simulation_project.src.experiments.orchestration",
        "run_all_experiments",
    ),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(str(name))
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
