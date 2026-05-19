from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from simulation_second.src.bayes_kernel.experiments.fitting import _fit_all_methods as legacy_fit_all_methods
from simulation_second.src.bayes_kernel.utils import FitResult, SamplerConfig

from .schemas import ConvergenceGateSpec


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


def _rhs_sampler_strategy(
    X: np.ndarray,
    methods: Sequence[str],
    grrhs_kwargs: dict[str, Any] | None,
) -> str:
    method_set = {str(method).strip() for method in methods}
    if any("HighDim" in method for method in method_set):
        return "high_dim"
    if np.asarray(X).shape[1] > np.asarray(X).shape[0]:
        return "high_dim"
    backend = str((grrhs_kwargs or {}).get("sampler_backend", "")).strip().lower()
    if backend == "collapsed_profile":
        return "high_dim"
    return "low_dim"


def fit_real_data_methods(
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
) -> dict[str, FitResult]:
    methods_use = [str(method) for method in methods]
    grrhs_use = dict(grrhs_kwargs or {})
    return legacy_fit_all_methods(
        np.asarray(X, dtype=float),
        np.asarray(y, dtype=float).reshape(-1),
        groups,
        task=str(task),
        seed=int(seed),
        p0=int(p0),
        p0_groups=int(p0_groups),
        sampler=sampler_config_from_gate(gate),
        grrhs_kwargs=grrhs_use,
        methods=methods_use,
        gigg_config=dict(gigg_config or {}),
        bayes_min_chains=int(gate.bayes_min_chains),
        enforce_bayes_convergence=bool(gate.enforce_bayes_convergence),
        max_convergence_retries=int(gate.max_convergence_retries),
        method_jobs=int(method_jobs),
        rhs_sampler_strategy=_rhs_sampler_strategy(np.asarray(X, dtype=float), methods_use, grrhs_use),
    )

