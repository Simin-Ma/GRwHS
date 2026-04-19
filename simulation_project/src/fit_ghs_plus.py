from __future__ import annotations

from typing import Sequence

import numpy as np

from simulation_project.src.core.models.baselines import GroupedHorseshoePlus

from .utils import FitResult, SamplerConfig, diagnostics_summary_for_method, rhs_style_tau0, timed_call

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
    mult = max(1, int(iter_mult))
    floor = max(10, int(iter_floor))
    cap = max(floor, int(iter_cap))
    burnin = min(max(int(sampler.warmup) * mult, floor), cap)
    draws = min(max(int(sampler.post_warmup_draws) * mult, floor), cap)
    return burnin, draws


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
) -> FitResult:
    if str(task).lower() == "logistic":
        return FitResult(
            method="GHS_plus",
            status="error",
            beta_mean=None,
            beta_draws=None,
            kappa_draws=None,
            group_scale_draws=None,
            runtime_seconds=float("nan"),
            rhat_max=float("nan"),
            bulk_ess_min=float("nan"),
            divergence_ratio=float("nan"),
            converged=False,
            error="NotImplementedError: Group Horseshoe+ is gaussian-only in this repository",
            diagnostics={},
        )

    try:
        n, p = int(X.shape[0]), int(X.shape[1])
        tau0 = rhs_style_tau0(n=n, p=p, p0=p0)
        ghs_burnin, ghs_draws = _ghs_iters(
            sampler,
            iter_mult=iter_mult,
            iter_floor=iter_floor,
            iter_cap=iter_cap,
        )
        model = GroupedHorseshoePlus(
            fit_intercept=False,
            tau0=float(tau0),
            iters=int(ghs_burnin + ghs_draws),
            burnin=int(ghs_burnin),
            thin=1,
            seed=int(seed),
            num_chains=int(sampler.chains),
            progress_bar=False,
        )
        model, runtime = timed_call(model.fit, X, y, groups=[list(map(int, g)) for g in groups])
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
            diagnostics=details,
        )
    except Exception as exc:
        return FitResult(
            method="GHS_plus",
            status="error",
            beta_mean=None,
            beta_draws=None,
            kappa_draws=None,
            group_scale_draws=None,
            runtime_seconds=float("nan"),
            rhat_max=float("nan"),
            bulk_ess_min=float("nan"),
            divergence_ratio=float("nan"),
            converged=False,
            error=f"{type(exc).__name__}: {exc}",
            diagnostics={},
        )
