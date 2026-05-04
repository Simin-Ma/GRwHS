from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from simulation_project.src.experiments.fitting import _fit_all_methods as legacy_fit_all_methods
from simulation_project.src.utils import FitResult, SamplerConfig

from .schemas import ConvergenceGateSpec


def _rhs_sampler_strategy_for_package(package: str | None) -> str:
    text = str("" if package is None else package).strip().lower()
    if "highdimension" in text:
        return "high_dim"
    return "low_dim"


def _grrhs_defaults_for_package(package: str | None) -> dict[str, Any]:
    strategy = _rhs_sampler_strategy_for_package(package)
    if strategy == "high_dim":
        return {
            "tau_target": "groups",
            "sampler_backend": "collapsed_profile",
            "use_local_scale": False,
            "progress_bar": False,
        }
    return {
        "tau_target": "groups",
        "sampler_backend": "gibbs_staged",
        "progress_bar": False,
    }


def _gigg_mmle_defaults_for_package(package: str | None) -> dict[str, Any]:
    strategy = _rhs_sampler_strategy_for_package(package)
    if strategy == "high_dim":
        return {
            "exact_highdim_fastpath": True,
        }
    return {
        "exact_highdim_fastpath": False,
    }


def sampler_config_from_gate(gate: ConvergenceGateSpec) -> SamplerConfig:
    return SamplerConfig(
        chains=int(gate.chains),
        warmup=int(gate.warmup),
        post_warmup_draws=int(gate.post_warmup_draws),
        adapt_delta=float(gate.adapt_delta),
        max_treedepth=int(gate.max_treedepth),
        strict_adapt_delta=float(gate.strict_adapt_delta),
        strict_max_treedepth=int(gate.strict_max_treedepth),
        max_divergence_ratio=float(gate.max_divergence_ratio),
        rhat_threshold=float(gate.rhat_threshold),
        ess_threshold=float(gate.ess_threshold),
    )


def fit_benchmark_methods(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    task: str,
    seed: int,
    p0: int,
    p0_groups: int,
    methods: Sequence[str],
    gate: ConvergenceGateSpec,
    grrhs_kwargs: dict[str, Any] | None = None,
    gigg_config: dict[str, Any] | None = None,
    method_jobs: int = 1,
    benchmark_package: str = "simulation_second",
) -> dict[str, FitResult]:
    grrhs_merged = {
        **_grrhs_defaults_for_package(benchmark_package),
        **dict(grrhs_kwargs or {}),
    }
    gigg_merged = {
        **_gigg_mmle_defaults_for_package(benchmark_package),
        **dict(gigg_config or {}),
    }
    return legacy_fit_all_methods(
        X,
        y,
        groups,
        task=str(task),
        seed=int(seed),
        p0=int(p0),
        p0_groups=int(p0_groups),
        sampler=sampler_config_from_gate(gate),
        grrhs_kwargs=grrhs_merged,
        methods=[str(method) for method in methods],
        gigg_config=gigg_merged,
        bayes_min_chains=int(gate.bayes_min_chains),
        enforce_bayes_convergence=bool(gate.enforce_bayes_convergence),
        max_convergence_retries=int(gate.max_convergence_retries),
        method_jobs=int(method_jobs),
        rhs_sampler_strategy=_rhs_sampler_strategy_for_package(benchmark_package),
    )
