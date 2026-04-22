from __future__ import annotations

from typing import Any, Sequence

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
    from .analysis.metrics import ci_length_and_coverage, compute_test_lpd, compute_test_lpd_ppd, mse_null_signal_overall

    nan = float("nan")
    if result.beta_mean is None:
        return {
            "mse_null": nan,
            "mse_signal": nan,
            "mse_overall": nan,
            "avg_ci_length": nan,
            "coverage_95": nan,
            "lpd_test": nan,
            "lpd_test_ppd": nan,
            "lpd_test_plugin": nan,
        }
    m = mse_null_signal_overall(result.beta_mean, beta0)
    ci_len, cov = ci_length_and_coverage(beta0, result.beta_draws)
    lpd_plugin = nan
    lpd_ppd = nan
    if X_train is not None and y_train is not None and X_test is not None and y_test is not None:
        train_resid2 = float(np.mean((np.asarray(y_train) - np.asarray(X_train) @ result.beta_mean) ** 2))
        lpd_plugin = compute_test_lpd(result.beta_mean, X_test, y_test, sigma2_hat=train_resid2)
        lpd_ppd = compute_test_lpd_ppd(result.beta_draws, X_test, y_test, sigma2_hat=train_resid2)
    lpd = lpd_ppd if np.isfinite(lpd_ppd) else lpd_plugin
    return {
        "mse_null": m["mse_null"],
        "mse_signal": m["mse_signal"],
        "mse_overall": m["mse_overall"],
        "avg_ci_length": ci_len,
        "coverage_95": cov,
        "lpd_test": lpd,
        "lpd_test_ppd": lpd_ppd,
        "lpd_test_plugin": lpd_plugin,
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


def _bridge_ratio_diagnostics(
    result: FitResult,
    *,
    groups: Sequence[Sequence[int]],
    X: np.ndarray,
    y: np.ndarray,
    signal_group_mask: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """
    Posterior-bridge diagnostic proxy:
      rho_g = ||E(beta_g|Y)||_2 / (E(kappa_g|Y) * ||Y_g||_2)

    For grouped regression experiments, Y_g is represented by the group score
    vector X_g^T y / n (same coefficient scale as beta).
    """
    nan = float("nan")
    out: dict[str, Any] = {
        "bridge_ratio_mean": nan,
        "bridge_ratio_min": nan,
        "bridge_ratio_max": nan,
        "bridge_ratio_p95": nan,
        "bridge_ratio_violations": nan,
        "bridge_ratio_null_mean": nan,
        "bridge_ratio_signal_mean": nan,
        "bridge_ratio_by_group": "",
    }
    if result.beta_mean is None or result.kappa_draws is None:
        return out

    beta = np.asarray(result.beta_mean, dtype=float).reshape(-1)
    kd = np.asarray(result.kappa_draws, dtype=float)
    if kd.ndim > 2:
        kd = kd.reshape(-1, kd.shape[-1])
    if kd.ndim == 1:
        kd = kd.reshape(-1, 1)
    n_groups = int(len(groups))
    if kd.shape[-1] != n_groups:
        return out

    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if X_arr.ndim != 2 or y_arr.ndim != 1 or X_arr.shape[0] != y_arr.shape[0]:
        return out
    n = max(int(X_arr.shape[0]), 1)
    group_scores = (X_arr.T @ y_arr) / float(n)
    kappa_mean = np.mean(kd, axis=0)

    ratios_raw: list[float] = []
    ratios_clip: list[float] = []
    for gid, g in enumerate(groups):
        idx = np.asarray(list(g), dtype=int)
        if idx.size == 0:
            ratios_raw.append(nan)
            ratios_clip.append(nan)
            continue
        num = float(np.linalg.norm(beta[idx], ord=2))
        denom = float(kappa_mean[gid]) * float(np.linalg.norm(group_scores[idx], ord=2))
        if not np.isfinite(num) or not np.isfinite(denom) or denom <= 0.0:
            ratios_raw.append(nan)
            ratios_clip.append(nan)
            continue
        rr = num / denom
        ratios_raw.append(float(rr))
        ratios_clip.append(float(min(max(rr, 0.0), 1.0)))

    vec = np.asarray(ratios_clip, dtype=float)
    raw_vec = np.asarray(ratios_raw, dtype=float)
    finite = vec[np.isfinite(vec)]
    if finite.size == 0:
        return out

    out["bridge_ratio_mean"] = float(np.mean(finite))
    out["bridge_ratio_min"] = float(np.min(finite))
    out["bridge_ratio_max"] = float(np.max(finite))
    out["bridge_ratio_p95"] = float(np.quantile(finite, 0.95))
    out["bridge_ratio_violations"] = int(np.sum(np.isfinite(raw_vec) & (raw_vec > 1.0 + 1e-6)))
    out["bridge_ratio_by_group"] = "|".join(
        f"{float(v):.6g}" if np.isfinite(v) else "nan"
        for v in vec.tolist()
    )

    if signal_group_mask is not None:
        mask = np.asarray(signal_group_mask, dtype=bool).reshape(-1)
        if mask.size == vec.size:
            null_vals = vec[(~mask) & np.isfinite(vec)]
            sig_vals = vec[(mask) & np.isfinite(vec)]
            out["bridge_ratio_null_mean"] = float(np.mean(null_vals)) if null_vals.size else nan
            out["bridge_ratio_signal_mean"] = float(np.mean(sig_vals)) if sig_vals.size else nan
    return out



