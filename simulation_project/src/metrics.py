from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from .utils import flatten_draws, posterior_ci95


def mse_null_signal_overall(beta_hat: np.ndarray, beta_true: np.ndarray) -> dict[str, float]:
    b = np.asarray(beta_hat, dtype=float).reshape(-1)
    t = np.asarray(beta_true, dtype=float).reshape(-1)
    signal = np.abs(t) > 1e-12
    null = ~signal
    out = {
        "mse_overall": float(np.mean((b - t) ** 2)),
        "mse_signal": float(np.mean((b[signal] - t[signal]) ** 2)) if np.any(signal) else float("nan"),
        "mse_null": float(np.mean((b[null] - t[null]) ** 2)) if np.any(null) else float("nan"),
    }
    return out


def ci_length_and_coverage(beta_true: np.ndarray, beta_draws: Optional[np.ndarray]) -> tuple[float, float]:
    ci = posterior_ci95(beta_draws)
    if ci is None:
        return float("nan"), float("nan")
    low = np.asarray(ci[0], dtype=float)
    high = np.asarray(ci[1], dtype=float)
    width = float(np.mean(high - low))
    t = np.asarray(beta_true, dtype=float)
    cover = (t >= low) & (t <= high)
    return width, float(np.mean(cover))


def group_l2_score(beta_hat: np.ndarray, groups: Sequence[Sequence[int]]) -> np.ndarray:
    b = np.asarray(beta_hat, dtype=float)
    return np.asarray([float(np.sum(b[np.asarray(g, dtype=int)] ** 2)) for g in groups], dtype=float)


def group_l2_error(beta_hat: np.ndarray, beta_true: np.ndarray, groups: Sequence[Sequence[int]]) -> np.ndarray:
    b = np.asarray(beta_hat, dtype=float)
    t = np.asarray(beta_true, dtype=float)
    return np.asarray([
        float(np.sum((b[np.asarray(g, dtype=int)] - t[np.asarray(g, dtype=int)]) ** 2))
        for g in groups
    ])


def group_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    s = np.asarray(scores, dtype=float).reshape(-1)
    y = np.asarray(labels, dtype=int).reshape(-1)
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def prob_above(draws: Optional[np.ndarray], threshold: float) -> float:
    flat = flatten_draws(draws, scalar=False)
    if flat is None:
        return float("nan")
    return float(np.mean(flat > float(threshold)))


def compute_test_lpd(
    beta_hat: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    sigma2_hat: float,
) -> float:
    """Plug-in log predictive density on a held-out test set.

    Uses the posterior mean (or OLS/LASSO estimate) for beta and the training
    residual variance as a plug-in for sigma^2.  Valid for all methods.
    """
    if beta_hat is None:
        return float("nan")
    b = np.asarray(beta_hat, dtype=float).reshape(-1)
    Xt = np.asarray(X_test, dtype=float)
    yt = np.asarray(y_test, dtype=float).reshape(-1)
    s2 = max(float(sigma2_hat), 1e-8)
    resid = yt - Xt @ b
    return float(-0.5 * np.log(2.0 * np.pi * s2) - 0.5 * float(np.mean(resid ** 2)) / s2)
