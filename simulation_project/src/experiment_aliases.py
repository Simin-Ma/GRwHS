from __future__ import annotations

from typing import Any


SWEEP_EXPERIMENT_ALIASES: dict[str, str] = {
    "all": "all",
    "pipeline": "all",
    "ga_v2a": "ga_v2a",
    "ga_v2_group_separation": "ga_v2a",
    "ga_v2b": "ga_v2b",
    "ga_v2_complexity_mismatch": "ga_v2b",
    "ga_v2c": "ga_v2c",
    "ga_v2_correlation_stress": "ga_v2c",
}

CLI_EXPERIMENT_CHOICES: tuple[str, ...] = ("all", "ga_v2a", "ga_v2b", "ga_v2c", "analysis")

CLI_CHOICE_TO_KEY: dict[str, str] = {
    "all": "all",
    "ga_v2a": "ga_v2a",
    "ga_v2b": "ga_v2b",
    "ga_v2c": "ga_v2c",
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
