from __future__ import annotations

from typing import Sequence

import numpy as np

from simulation_project.src.core.models.baselines import RegularizedHorseshoeGibbs

from .helpers import fit_error_result
from ...utils import FitResult, SamplerConfig, diagnostics_summary_for_method, rhs_style_tau0, timed_call


def _build_rhs_gibbs(
    *,
    n: int,
    p: int,
    p0: int,
    sampler: SamplerConfig,
    progress_bar: bool,
    seed: int,
) -> RegularizedHorseshoeGibbs:
    return RegularizedHorseshoeGibbs(
        scale_global=float(rhs_style_tau0(n=int(n), p=int(p), p0=int(max(p0, 1)))),
        num_warmup=int(sampler.warmup),
        num_samples=int(sampler.post_warmup_draws),
        num_chains=int(sampler.chains),
        thinning=1,
        seed=int(seed),
        progress_bar=bool(progress_bar),
    )


def fit_rhs_gibbs(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    task: str,
    seed: int,
    p0: int,
    sampler: SamplerConfig,
    progress_bar: bool = True,
    method_name: str = "RHS_HighDim",
) -> FitResult:
    del groups
    if str(task).strip().lower() != "gaussian":
        return fit_error_result(str(method_name), "NotImplementedError: RHS_HighDim currently supports Gaussian likelihood only.")

    tracked = ["beta", "tau", "lambda", "c", "sigma"]
    n, p = int(X.shape[0]), int(X.shape[1])

    try:
        model = _build_rhs_gibbs(
            n=n,
            p=p,
            p0=p0,
            sampler=sampler,
            progress_bar=bool(progress_bar),
            seed=seed,
        )
        model, runtime = timed_call(model.fit, X, y)
        beta_draws = getattr(model, "coef_samples_", None)
        beta_mean = getattr(model, "coef_", None)

        rhat_max, ess_min, div_ratio, converged, details = diagnostics_summary_for_method(
            model=model,
            tracked_params=tracked,
            beta_draws=beta_draws,
            config=sampler,
        )
        details = dict(details or {})
        details["rhs_impl"] = "rhs_gibbs_woodbury"
        details["rhs_sampler_name"] = str(method_name)
        details["rhs_sampler_strategy"] = "high_dim"
        details["rhs_defaults"] = {
            "slab_df": float(model.slab_df),
            "slab_scale": float(model.slab_scale),
            "global_scale": float(model.scale_global),
        }

        return FitResult(
            method=str(method_name),
            status="ok",
            beta_mean=None if beta_mean is None else np.asarray(beta_mean, dtype=float),
            beta_draws=None if beta_draws is None else np.asarray(beta_draws, dtype=float),
            kappa_draws=None,
            group_scale_draws=None,
            tau_draws=None if getattr(model, "tau_samples_", None) is None else np.asarray(model.tau_samples_, dtype=float),
            runtime_seconds=float(runtime),
            rhat_max=float(rhat_max),
            bulk_ess_min=float(ess_min),
            divergence_ratio=float(div_ratio),
            converged=bool(converged),
            diagnostics=details,
        )
    except Exception as exc:
        return fit_error_result(str(method_name), f"{type(exc).__name__}: {exc}")
