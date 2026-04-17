from __future__ import annotations

from typing import Sequence

import numpy as np

from grrhs.models.baselines import GroupedHorseshoePlus

from .utils import FitResult, SamplerConfig, diagnostics_summary_for_method, rhs_style_tau0, timed_call


def fit_ghs_plus(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    task: str,
    seed: int,
    p0: int,
    sampler: SamplerConfig,
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
        model = GroupedHorseshoePlus(
            fit_intercept=False,
            tau0=float(tau0),
            iters=int(sampler.warmup + sampler.post_warmup_draws),
            burnin=int(sampler.warmup),
            thin=1,
            seed=int(seed),
            num_chains=int(sampler.chains),
            progress_bar=False,
        )
        model, runtime = timed_call(model.fit, X, y, groups=[list(map(int, g)) for g in groups])
        beta_draws = getattr(model, "coef_samples_", None)
        beta_mean = getattr(model, "coef_mean_", None)
        group_draws = getattr(model, "group_lambda_samples_", None)

        rhat_max, ess_min, div_ratio, converged, details = diagnostics_summary_for_method(
            model=model,
            tracked_params=["beta", "group_scale"],
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
