from __future__ import annotations

from typing import Sequence

import numpy as np

from grrhs.models.gigg_regression import GIGGRegression

from .utils import FitResult, SamplerConfig, diagnostics_summary_for_method, timed_call


def fit_gigg_mmle(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    task: str,
    seed: int,
    sampler: SamplerConfig,
) -> FitResult:
    if str(task).lower() == "logistic":
        return FitResult(
            method="GIGG_MMLE",
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
            error="NotImplementedError: GIGG_MMLE is gaussian-only in this repository",
            diagnostics={},
        )

    tracked = ["beta", "gamma2"]
    try:
        model = GIGGRegression(
            method="mmle",
            n_burn_in=int(sampler.warmup),
            n_samples=int(sampler.post_warmup_draws),
            n_thin=1,
            seed=int(seed),
            num_chains=int(sampler.chains),
            fit_intercept=False,
            store_lambda=True,
        )
        model, runtime = timed_call(model.fit, X, y, groups=[list(map(int, g)) for g in groups])

        beta_draws = getattr(model, "coef_samples_", None)
        beta_mean = getattr(model, "coef_mean_", None)
        gamma2_draws = getattr(model, "gamma2_samples_", None)

        rhat_max, ess_min, div_ratio, converged, details = diagnostics_summary_for_method(
            model=model,
            tracked_params=tracked,
            beta_draws=beta_draws,
            config=sampler,
        )

        return FitResult(
            method="GIGG_MMLE",
            status="ok",
            beta_mean=None if beta_mean is None else np.asarray(beta_mean, dtype=float),
            beta_draws=None if beta_draws is None else np.asarray(beta_draws, dtype=float),
            kappa_draws=None,
            group_scale_draws=None if gamma2_draws is None else np.asarray(gamma2_draws, dtype=float),
            runtime_seconds=float(runtime),
            rhat_max=float(rhat_max),
            bulk_ess_min=float(ess_min),
            divergence_ratio=float(div_ratio),
            converged=bool(converged),
            diagnostics=details,
        )
    except Exception as exc:
        return FitResult(
            method="GIGG_MMLE",
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
