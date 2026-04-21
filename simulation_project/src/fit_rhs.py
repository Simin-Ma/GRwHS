from __future__ import annotations

from typing import Sequence

import numpy as np

from simulation_project.src.core.models.baselines import RegularizedHorseshoeRegression

from .fit_helpers import as_int_groups, fit_error_result
from .utils import (
    FitResult,
    SamplerConfig,
    diagnostics_summary_for_method,
    logistic_pseudo_sigma,
    timed_call,
    rhs_style_tau0,
)

_SIMULATION_RHS_BACKEND = "numpyro"


def _build_rhs(
    *,
    task: str,
    seed: int,
    n: int,
    p: int,
    p0: int,
    pseudo_sigma: float,
    sampler: SamplerConfig,
    adapt_delta: float,
    max_treedepth: int,
    progress_bar: bool,
) -> RegularizedHorseshoeRegression:
    likelihood = "logistic" if str(task).lower() == "logistic" else "gaussian"
    tau0 = rhs_style_tau0(n=n, p=p, p0=p0)
    if likelihood == "logistic":
        tau0 *= float(pseudo_sigma)
    return RegularizedHorseshoeRegression(
        scale_global=float(tau0),
        likelihood=likelihood,
        backend=_SIMULATION_RHS_BACKEND,
        num_warmup=int(sampler.warmup),
        num_samples=int(sampler.post_warmup_draws),
        num_chains=int(sampler.chains),
        target_accept_prob=float(adapt_delta),
        max_tree_depth=int(max_treedepth),
        chain_method="sequential",
        progress_bar=bool(progress_bar),
        seed=int(seed),
    )


def fit_rhs(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    task: str,
    seed: int,
    p0: int,
    sampler: SamplerConfig,
    progress_bar: bool = True,
) -> FitResult:
    tracked = ["beta", "tau", "lambda", "c"]
    n, p = int(X.shape[0]), int(X.shape[1])

    try:
        pseudo_sigma = 1.0
        if str(task).lower() == "logistic":
            pseudo_sigma = logistic_pseudo_sigma(y)
        model = _build_rhs(
            task=task,
            seed=seed,
            n=n,
            p=p,
            p0=p0,
            pseudo_sigma=pseudo_sigma,
            sampler=sampler,
            adapt_delta=float(sampler.adapt_delta),
            max_treedepth=int(sampler.max_treedepth),
            progress_bar=bool(progress_bar),
        )
        if str(getattr(model, "backend", "")).strip().lower() != _SIMULATION_RHS_BACKEND:
            raise RuntimeError(
                f"Simulation benchmark RHS backend must be '{_SIMULATION_RHS_BACKEND}'. "
                f"Received '{getattr(model, 'backend', None)}'."
            )
        model, runtime = timed_call(model.fit, X, y, groups=as_int_groups(groups))
        beta_draws = getattr(model, "coef_samples_", None)
        beta_mean = getattr(model, "coef_", None)

        rhat_max, ess_min, div_ratio, converged, details = diagnostics_summary_for_method(
            model=model,
            tracked_params=tracked,
            beta_draws=beta_draws,
            config=sampler,
        )

        if np.isfinite(div_ratio) and div_ratio >= float(sampler.max_divergence_ratio):
            strict = _build_rhs(
                task=task,
                seed=seed + 999,
                n=n,
                p=p,
                p0=p0,
                pseudo_sigma=pseudo_sigma,
                sampler=sampler,
                adapt_delta=float(sampler.strict_adapt_delta),
                max_treedepth=int(sampler.strict_max_treedepth),
                progress_bar=bool(progress_bar),
            )
            if str(getattr(strict, "backend", "")).strip().lower() != _SIMULATION_RHS_BACKEND:
                raise RuntimeError(
                    f"Simulation benchmark RHS backend must be '{_SIMULATION_RHS_BACKEND}'. "
                    f"Received '{getattr(strict, 'backend', None)}'."
                )
            strict, runtime2 = timed_call(strict.fit, X, y, groups=as_int_groups(groups))
            beta_draws = getattr(strict, "coef_samples_", None)
            beta_mean = getattr(strict, "coef_", None)
            rhat_max, ess_min, div_ratio, converged, details = diagnostics_summary_for_method(
                model=strict,
                tracked_params=tracked,
                beta_draws=beta_draws,
                config=sampler,
            )
            runtime += runtime2

        return FitResult(
            method="RHS",
            status="ok",
            beta_mean=None if beta_mean is None else np.asarray(beta_mean, dtype=float),
            beta_draws=None if beta_draws is None else np.asarray(beta_draws, dtype=float),
            kappa_draws=None,
            group_scale_draws=None,
            runtime_seconds=float(runtime),
            rhat_max=float(rhat_max),
            bulk_ess_min=float(ess_min),
            divergence_ratio=float(div_ratio),
            converged=bool(converged),
            diagnostics=details,
        )
    except Exception as exc:
        return fit_error_result("RHS", f"{type(exc).__name__}: {exc}")
