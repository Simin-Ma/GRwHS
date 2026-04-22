from __future__ import annotations

import numpy as np
from sklearn.linear_model import LassoCV, LinearRegression

from .helpers import fit_error_result
from ...utils import FitResult, timed_call


def fit_ols(X: np.ndarray, y: np.ndarray, *, task: str, seed: int) -> FitResult:
    if str(task).lower() != "gaussian":
        return fit_error_result("OLS", "NotImplementedError: OLS baseline is gaussian-only in this pipeline")
    try:
        model = LinearRegression(fit_intercept=False)
        model, runtime = timed_call(model.fit, X, y)
        beta = np.asarray(model.coef_, dtype=float).reshape(-1)
        return FitResult(
            method="OLS",
            status="ok",
            beta_mean=beta,
            beta_draws=None,
            kappa_draws=None,
            group_scale_draws=None,
            runtime_seconds=float(runtime),
            rhat_max=float("nan"),
            bulk_ess_min=float("nan"),
            divergence_ratio=float("nan"),
            converged=True,
            diagnostics={},
        )
    except Exception as exc:
        return fit_error_result("OLS", f"{type(exc).__name__}: {exc}")


def fit_lasso_cv(X: np.ndarray, y: np.ndarray, *, task: str, seed: int) -> FitResult:
    if str(task).lower() != "gaussian":
        return fit_error_result("LASSO_CV", "NotImplementedError: LASSO_CV baseline is gaussian-only in this pipeline")
    try:
        model = LassoCV(
            cv=5,
            fit_intercept=False,
            n_jobs=None,
            random_state=int(seed),
            max_iter=10000,
        )
        model, runtime = timed_call(model.fit, X, y)
        beta = np.asarray(model.coef_, dtype=float).reshape(-1)
        return FitResult(
            method="LASSO_CV",
            status="ok",
            beta_mean=beta,
            beta_draws=None,
            kappa_draws=None,
            group_scale_draws=None,
            runtime_seconds=float(runtime),
            rhat_max=float("nan"),
            bulk_ess_min=float("nan"),
            divergence_ratio=float("nan"),
            converged=True,
            diagnostics={},
        )
    except Exception as exc:
        return fit_error_result("LASSO_CV", f"{type(exc).__name__}: {exc}")

