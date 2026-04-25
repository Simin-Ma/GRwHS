from __future__ import annotations

from typing import Any, Dict, List

from .blueprint import FAMILY_SPECS
from .schemas import (
    DEFAULT_METHOD_ROSTER,
    OPERATIONAL_METRICS,
    PREDICTIVE_METRICS,
    PRIMARY_METRICS,
    UNCERTAINTY_METRICS,
    ConvergenceGateSpec,
    SettingSpec,
)


DEFAULT_CONVERGENCE_GATE = ConvergenceGateSpec()

TABLE_SPECS: Dict[str, Dict[str, List[str]]] = {
    "main_benchmark_leaderboard": {
        "rows": ["headline settings from the main benchmark families"],
        "columns": [
            "winner",
            "mse_overall",
            "mse_signal",
            "coverage_95",
            "runner_up",
            "delta_overall_vs_runner_up",
            "GIGG_MMLE_overall_mse",
            "GIGG_MMLE_over_GR_RHS_mse_ratio",
        ],
    },
    "full_appendix_benchmark_table": {
        "rows": ["setting x method"],
        "columns": [
            "n_runs",
            "n_ok",
            "n_converged",
            "mse_overall",
            "mse_signal",
            "mse_null",
            "coverage_95",
            "avg_ci_length",
            "runtime_mean",
        ],
    },
    "group_size_family_comparison": {
        "rows": ["equal-size", "small-size", "large-size", "unequal-size"],
        "columns": [
            "winner",
            "LASSO_CV_over_GR_RHS_mse_ratio",
            "GIGG_MMLE_over_GR_RHS_mse_ratio",
            "interpretation",
        ],
    },
}

OPTIONAL_STRESS_LINES: List[Dict[str, Any]] = [
    {
        "name": "paired_decoy",
        "role": "supporting robustness benchmark",
        "status": "design placeholder in simulation_second",
    },
    {
        "name": "size_imbalance",
        "role": "supporting robustness benchmark",
        "status": "design placeholder in simulation_second",
    },
    {
        "name": "weak_identification",
        "role": "supporting inference benchmark",
        "status": "design placeholder in simulation_second",
    },
]


def build_main_suite(*, n_test: int = 100) -> List[SettingSpec]:
    return [
        SettingSpec(
            setting_id="setting_1_classical_equal_medium",
            label="Setting 1: Classical Reference, Equal Groups, Medium Correlation",
            family="classical_reference",
            group_sizes=(10, 10, 10, 10, 10),
            active_groups=(0, 1, 2),
            n_train=500,
            n_test=n_test,
            rho_within=0.6,
            rho_between=0.2,
            role="credibility anchor",
            notes="Family A random blueprint with shared mild support and concentration.",
        ),
        SettingSpec(
            setting_id="setting_2_classical_equal_high",
            label="Setting 2: Classical Reference, Equal Groups, High Correlation",
            family="classical_reference",
            group_sizes=(10, 10, 10, 10, 10),
            active_groups=(0, 1, 2),
            n_train=500,
            n_test=n_test,
            rho_within=0.8,
            rho_between=0.2,
            role="classical paper-style stress point",
            notes="Family A random blueprint under higher within-group correlation.",
        ),
        SettingSpec(
            setting_id="setting_3_single_mode_equal",
            label="Setting 3: Single-Mode Heterogeneous, Equal Groups",
            family="single_mode_heterogeneous",
            group_sizes=(10, 10, 10, 10, 10),
            active_groups=(0, 1, 2),
            n_train=500,
            n_test=n_test,
            rho_within=0.6,
            rho_between=0.2,
            role="transition from neutral to mechanism-sensitive benchmark",
            notes="Family B random blueprint with shared heterogeneous mode family.",
        ),
        SettingSpec(
            setting_id="setting_4_single_mode_unequal",
            label="Setting 4: Single-Mode Heterogeneous, Unequal Groups",
            family="single_mode_heterogeneous",
            group_sizes=(30, 10, 5, 3, 2),
            active_groups=(0, 1, 2),
            n_train=500,
            n_test=n_test,
            rho_within=0.6,
            rho_between=0.2,
            role="group-size heterogeneity benchmark",
            notes="Tests whether unequal group sizes make the group layer more informative.",
        ),
        SettingSpec(
            setting_id="setting_5_multimode_equal",
            label="Setting 5: Main Showcase, Multi-Mode Heterogeneous Active Groups",
            family="multimode_heterogeneous",
            group_sizes=(10, 10, 10, 10, 10),
            active_groups=(0, 1, 2),
            n_train=500,
            n_test=n_test,
            rho_within=0.8,
            rho_between=0.2,
            role="main showcase setting",
            notes="Family C random blueprint with active-group-specific hyperparameters.",
        ),
        SettingSpec(
            setting_id="setting_6_multimode_large_groups",
            label="Setting 6: Main Showcase, Multi-Mode Heterogeneous, Larger Groups",
            family="multimode_heterogeneous",
            group_sizes=(25, 25),
            active_groups=(0, 1),
            n_train=500,
            n_test=n_test,
            rho_within=0.8,
            rho_between=0.2,
            role="wide-group multimode benchmark",
            notes="Large groups make coordinate-only sparsity less aligned with the DGP.",
        ),
    ]


def get_setting_by_id(setting_id: str, *, n_test: int = 100) -> SettingSpec:
    wanted = str(setting_id).strip()
    for setting in build_main_suite(n_test=n_test):
        if setting.setting_id == wanted:
            return setting
    raise KeyError(f"Unknown setting_id: {setting_id!r}")


def build_suite_manifest(*, n_test: int = 100) -> Dict[str, Any]:
    return {
        "package": "simulation_second",
        "description": "Second-generation benchmark suite built from the GR-RHS blueprint.",
        "methods": list(DEFAULT_METHOD_ROSTER),
        "convergence_gate": DEFAULT_CONVERGENCE_GATE.to_dict(),
        "signal_families": {name: spec.to_dict() for name, spec in FAMILY_SPECS.items()},
        "metrics": {
            "primary": list(PRIMARY_METRICS),
            "uncertainty": list(UNCERTAINTY_METRICS),
            "predictive": list(PREDICTIVE_METRICS),
            "operational": list(OPERATIONAL_METRICS),
        },
        "tables": TABLE_SPECS,
        "optional_stress_lines": OPTIONAL_STRESS_LINES,
        "settings": [setting.to_dict() for setting in build_main_suite(n_test=n_test)],
    }

