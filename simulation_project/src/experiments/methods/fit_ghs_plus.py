from __future__ import annotations

from typing import Sequence

import numpy as np

from simulation_project.src.core.models.baselines import GroupedHorseshoePlus

from .helpers import as_int_groups, fit_error_result, scaled_iteration_budget
from ...utils import FitResult, SamplerConfig, diagnostics_summary_for_method, timed_call

# GHS+ Gibbs mixes far more slowly than HMC for the group-level shrinkage
# parameters (lambda_g). Scale Gibbs iterations by this multiplier relative
# to the HMC sampler budget so that beta coefficients reach adequate ESS.
_GHS_ITER_MULT = 4
_GHS_ITER_FLOOR = 2000
_GHS_ITER_CAP = 5000


def _ghs_iters(
    sampler: SamplerConfig,
    *,
    iter_mult: int = _GHS_ITER_MULT,
    iter_floor: int = _GHS_ITER_FLOOR,
    iter_cap: int = _GHS_ITER_CAP,
) -> tuple[int, int]:
    return scaled_iteration_budget(
        sampler,
        iter_mult=iter_mult,
        iter_floor=iter_floor,
        iter_cap=iter_cap,
    )


def fit_ghs_plus(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    task: str,
    seed: int,
    p0: int,
    sampler: SamplerConfig,
    iter_mult: int = _GHS_ITER_MULT,
    iter_floor: int = _GHS_ITER_FLOOR,
    iter_cap: int = _GHS_ITER_CAP,
    fit_intercept: bool = True,
    tau0: float = 1.0,
    group_scale_prior: float = 1.0,
    local_scale_prior: float = 1.0,
    progress_bar: bool = True,
) -> FitResult:
    """Fit Xu et al. (2016) HBGHS using the paper-style Gaussian Gibbs setup.

    The underlying sampler matches the hierarchical grouped horseshoe:
      tau ~ C+(0, 1), lambda_g ~ C+(0, 1), delta_j ~ C+(0, 1)
    with centered/scaled Gaussian regression handled inside the model when
    ``fit_intercept=True``.

    The previous project-specific RHS-style tau calibration is intentionally not
    used here so the default wrapper stays aligned to the paper's formulation.
    """
    if str(task).lower() == "logistic":
        return fit_error_result(
            "GHS_plus",
            "NotImplementedError: Group Horseshoe+ is gaussian-only in this repository",
        )

    try:
        ghs_burnin, ghs_draws = _ghs_iters(
            sampler,
            iter_mult=iter_mult,
            iter_floor=iter_floor,
            iter_cap=iter_cap,
        )
        model = GroupedHorseshoePlus(
            fit_intercept=bool(fit_intercept),
            tau0=float(tau0),
            group_scale_prior=float(group_scale_prior),
            local_scale_prior=float(local_scale_prior),
            iters=int(ghs_burnin + ghs_draws),
            burnin=int(ghs_burnin),
            thin=1,
            seed=int(seed),
            num_chains=int(sampler.chains),
            progress_bar=bool(progress_bar),
        )
        model, runtime = timed_call(model.fit, X, y, groups=as_int_groups(groups))
        beta_draws = getattr(model, "coef_samples_", None)
        beta_mean = getattr(model, "coef_mean_", None)
        group_draws = getattr(model, "group_lambda_samples_", None)

        # Convergence gated on beta only: group_scale (lambda_g) mixes ~10x
        # slower than beta in Gibbs and is a nuisance parameter for this study.
        rhat_max, ess_min, div_ratio, converged, details = diagnostics_summary_for_method(
            model=model,
            tracked_params=["beta"],
            beta_draws=beta_draws,
            config=sampler,
        )
        diagnostics = dict(details or {})
        diagnostics["paper_alignment"] = {
            "paper": "Xu et al. (2016)",
            "model": "HBGHS",
            "sampler": "gaussian_gibbs",
            "tau0": float(tau0),
            "group_scale_prior": float(group_scale_prior),
            "local_scale_prior": float(local_scale_prior),
            "fit_intercept": bool(fit_intercept),
        }

        return FitResult(
            method="GHS_plus",
            status="ok",
            beta_mean=None if beta_mean is None else np.asarray(beta_mean, dtype=float),
            beta_draws=None if beta_draws is None else np.asarray(beta_draws, dtype=float),
            kappa_draws=None,
            group_scale_draws=None if group_draws is None else np.asarray(group_draws, dtype=float),
            runtime_seconds=float(runtime),
            rhat_max=float(rhat_max),
            bulk_ess_min=float(ess_min),
            divergence_ratio=float(div_ratio),
            converged=bool(converged),
            diagnostics=diagnostics,
        )
    except Exception as exc:
        return fit_error_result("GHS_plus", f"{type(exc).__name__}: {exc}")

