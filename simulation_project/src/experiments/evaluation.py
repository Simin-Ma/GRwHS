from __future__ import annotations

import numpy as np

from ..utils import FitResult


def _evaluate_row(
    result: FitResult,
    beta0: np.ndarray,
    *,
    X_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute MSE, CI coverage, and (optionally) held-out log predictive density."""
    from .analysis.metrics import ci_length_and_coverage, compute_test_lpd, mse_null_signal_overall

    nan = float("nan")
    if result.beta_mean is None:
        return {
            "mse_null": nan,
            "mse_signal": nan,
            "mse_overall": nan,
            "avg_ci_length": nan,
            "coverage_95": nan,
            "lpd_test": nan,
        }
    m = mse_null_signal_overall(result.beta_mean, beta0)
    ci_len, cov = ci_length_and_coverage(beta0, result.beta_draws)
    lpd = nan
    if X_train is not None and y_train is not None and X_test is not None and y_test is not None:
        train_resid2 = float(np.mean((np.asarray(y_train) - np.asarray(X_train) @ result.beta_mean) ** 2))
        lpd = compute_test_lpd(result.beta_mean, X_test, y_test, sigma2_hat=train_resid2)
    return {
        "mse_null": m["mse_null"],
        "mse_signal": m["mse_signal"],
        "mse_overall": m["mse_overall"],
        "avg_ci_length": ci_len,
        "coverage_95": cov,
        "lpd_test": lpd,
    }


def _kappa_group_means(result: FitResult, n_groups: int) -> list[float]:
    """Posterior mean kappa_g per group for GR_RHS; NaN for other methods."""
    if result.kappa_draws is None:
        return [float("nan")] * n_groups
    kd = np.asarray(result.kappa_draws, dtype=float)
    if kd.ndim > 2:
        kd = kd.reshape(-1, kd.shape[-1])
    if kd.shape[-1] != n_groups:
        return [float("nan")] * n_groups
    return [float(np.mean(kd[:, g])) for g in range(n_groups)]


def _kappa_group_prob_gt(result: FitResult, n_groups: int, threshold: float = 0.5) -> list[float]:
    if result.kappa_draws is None:
        return [float("nan")] * n_groups
    kd = np.asarray(result.kappa_draws, dtype=float)
    if kd.ndim > 2:
        kd = kd.reshape(-1, kd.shape[-1])
    if kd.shape[-1] != n_groups:
        return [float("nan")] * n_groups
    return [float(np.mean(kd[:, g] > float(threshold))) for g in range(n_groups)]



