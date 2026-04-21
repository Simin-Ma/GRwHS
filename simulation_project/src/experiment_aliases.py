from __future__ import annotations

from typing import Any


SWEEP_EXPERIMENT_ALIASES: dict[str, str] = {
    "all": "all",
    "exp1-5": "all",
    "exp1_to_exp5": "all",
    "pipeline": "all",
    "1": "exp1",
    "exp1": "exp1",
    "exp1_kappa_profile_regimes": "exp1",
    "2": "exp2",
    "exp2": "exp2",
    "exp2_group_separation": "exp2",
    "3": "exp3",
    "exp3": "exp3",
    "exp3_linear_benchmark": "exp3",
    "exp3a": "exp3a",
    "exp3a_main_benchmark": "exp3a",
    "exp3b": "exp3b",
    "exp3b_boundary_stress": "exp3b",
    "4": "exp4",
    "exp4": "exp4",
    "exp4_variant_ablation": "exp4",
    "5": "exp5",
    "exp5": "exp5",
    "exp5_prior_sensitivity": "exp5",
}

CLI_EXPERIMENT_CHOICES: tuple[str, ...] = ("all", "1", "2", "3", "3a", "3b", "4", "5", "analysis")

CLI_CHOICE_TO_KEY: dict[str, str] = {
    "all": "all",
    "1": "exp1",
    "2": "exp2",
    "3": "exp3",
    "3a": "exp3a",
    "3b": "exp3b",
    "4": "exp4",
    "5": "exp5",
    "analysis": "analysis",
}


def normalize_sweep_experiment(value: Any) -> str:
    key = str(value).strip().lower()
    if key not in SWEEP_EXPERIMENT_ALIASES:
        raise ValueError(f"unknown experiment alias: {value!r}")
    return SWEEP_EXPERIMENT_ALIASES[key]


def cli_choice_to_key(choice: str) -> str:
    key = str(choice).strip().lower()
    if key not in CLI_CHOICE_TO_KEY:
        raise ValueError(f"unknown cli experiment choice: {choice!r}")
    return CLI_CHOICE_TO_KEY[key]
