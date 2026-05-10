from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "run_exp1_kappa_profile_regimes",
    "run_exp2_group_separation",
    "run_exp3_linear_benchmark",
    "run_exp3a_main_benchmark",
    "run_exp3b_boundary_stress",
    "run_exp3c_highdim_stress",
    "run_exp3d_within_group_mixed",
    "run_exp4_variant_ablation",
    "run_exp5_prior_sensitivity",
    "run_ga_v2_group_separation",
    "run_ga_v2_complexity_mismatch",
    "run_ga_v2_correlation_stress",
    "run_all_experiments",
]

_LAZY_EXPORTS = {
    "run_exp1_kappa_profile_regimes": (
        "simulation_project.src.experiments.exp1",
        "run_exp1_kappa_profile_regimes",
    ),
    "run_exp2_group_separation": (
        "simulation_project.src.experiments.exp2",
        "run_exp2_group_separation",
    ),
    "run_exp3_linear_benchmark": (
        "simulation_project.src.experiments.exp3",
        "run_exp3_linear_benchmark",
    ),
    "run_exp3a_main_benchmark": (
        "simulation_project.src.experiments.exp3",
        "run_exp3a_main_benchmark",
    ),
    "run_exp3b_boundary_stress": (
        "simulation_project.src.experiments.exp3",
        "run_exp3b_boundary_stress",
    ),
    "run_exp3c_highdim_stress": (
        "simulation_project.src.experiments.exp3",
        "run_exp3c_highdim_stress",
    ),
    "run_exp3d_within_group_mixed": (
        "simulation_project.src.experiments.exp3",
        "run_exp3d_within_group_mixed",
    ),
    "run_exp4_variant_ablation": (
        "simulation_project.src.experiments.exp4",
        "run_exp4_variant_ablation",
    ),
    "run_exp5_prior_sensitivity": (
        "simulation_project.src.experiments.exp5",
        "run_exp5_prior_sensitivity",
    ),
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
