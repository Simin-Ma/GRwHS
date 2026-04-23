from __future__ import annotations

# Public experiment entrypoint facade.
from .experiments import (
    run_exp1_kappa_profile_regimes,
    run_exp2_group_separation,
    run_exp3_linear_benchmark,
    run_exp3a_main_benchmark,
    run_exp3b_boundary_stress,
    run_exp3c_highdim_stress,
    run_exp3d_within_group_mixed,
    run_exp4_variant_ablation,
    run_exp5_prior_sensitivity,
)
from .experiments.orchestration import _cli, run_all_experiments

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
    "run_all_experiments",
    "_cli",
]


if __name__ == "__main__":
    _cli()

