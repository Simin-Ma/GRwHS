# grwhs/metrics/uncertainty.py
from __future__ import annotations
import numpy as np
from typing import Dict, Iterable, Optional, Tuple


def gaussian_nll(y_true: np.ndarray, mu: np.ndarray, sigma2: np.ndarray | float, *, average: str = "mean") -> float:
    """Keep a consistent `from .uncertainty import gaussian_nll` import path."""
    y_true = np.asarray(y_true).reshape(-1)
    mu = np.asarray(mu).reshape(-1)
    if np.isscalar(sigma2):
        s2 = float(sigma2) * np.ones_like(y_true)
    else:
        s2 = np.asarray(sigma2).reshape(-1)
    s2 = np.maximum(s2, 1e-30)
    nll = 0.5 * (np.log(2 * np.pi * s2) + ((y_true - mu) ** 2) / s2)
    return float(np.sum(nll) if average == "sum" else np.mean(nll))


def predictive_interval_coverage(
    y_true: np.ndarray,
    *,
    samples: Optional[np.ndarray] = None,
    pred_mean: Optional[np.ndarray] = None,
    pred_std: Optional[np.ndarray] = None,
    alphas: Iterable[float] = (0.1, 0.2),
) -> Dict[str, Dict[str, float]]:
    """
    Coverage & width for two interfaces:
      1) samples: array (S, n) predictive draws → empirical quantiles
      2) pred_mean + pred_std: Normal approx → quantile via Φ^{-1} (use np.percentile on standard normal)
    Returns dict keyed by e.g. "90%" with {"coverage": ..., "avg_width": ...}
    """
    y = np.asarray(y_true).reshape(-1)
    n = y.size
    out: Dict[str, Dict[str, float]] = {}

    if samples is not None:
        draws = np.asarray(samples)
        if draws.ndim == 1:
            draws = draws[None, :]
        S, n2 = draws.shape
        assert n2 == n, "samples shape must be (S, n)"
        for a in alphas:
            q_low = np.quantile(draws, a / 2.0, axis=0)
            q_high = np.quantile(draws, 1.0 - a / 2.0, axis=0)
            cover = np.mean((y >= q_low) & (y <= q_high))
            width = np.mean(q_high - q_low)
            out[f"{int((1-a)*100)}%"] = {"coverage": float(cover), "avg_width": float(width)}
        return out

    if pred_mean is not None and pred_std is not None:
        mu = np.asarray(pred_mean).reshape(-1)
        sd = np.asarray(pred_std).reshape(-1)
        sd = np.maximum(sd, 1e-30)
        # Use standard normal quantiles
        # Offline approximation: use np.percentile on many N(0,1) samples with a fixed RNG for stability
        rng = np.random.default_rng(42)
        std_norm = rng.standard_normal(10_000)
        for a in alphas:
            lo_q = float(np.percentile(std_norm, 100 * (a / 2.0)))
            hi_q = float(np.percentile(std_norm, 100 * (1.0 - a / 2.0)))
            q_low = mu + lo_q * sd
            q_high = mu + hi_q * sd
            cover = np.mean((y >= q_low) & (y <= q_high))
            width = np.mean(q_high - q_low)
            out[f"{int((1-a)*100)}%"] = {"coverage": float(cover), "avg_width": float(width)}
        return out

    raise ValueError("Provide either `samples` or (`pred_mean` and `pred_std`).")


def parameter_ci_coverage(
    beta_true: np.ndarray,
    beta_samples: np.ndarray,
    *,
    alpha: float = 0.1,
    use_median_length: bool = False,
) -> Dict[str, float]:
    """
    Parameter-wise credible interval coverage:
      - beta_samples: (S, p)
      - beta_true: (p,)
    Returns:
      overall_coverage, avg_length (or median_length), and per-parameter coverage_mean
    """
    b0 = np.asarray(beta_true).reshape(-1)
    draws = np.asarray(beta_samples)
    if draws.ndim != 2:
        raise ValueError("beta_samples must be 2D (S, p)")
    S, p = draws.shape
    if p != b0.size:
        raise ValueError("beta_samples second dim must match beta_true length.")
    ql = np.quantile(draws, alpha / 2.0, axis=0)
    qh = np.quantile(draws, 1.0 - alpha / 2.0, axis=0)
    cover_vec = (b0 >= ql) & (b0 <= qh)
    lengths = qh - ql
    overall = float(np.mean(cover_vec))
    avg_len = float(np.median(lengths) if use_median_length else np.mean(lengths))
    return {
        "coverage": overall,
        "avg_length": avg_len,
        "per_param_coverage_mean": float(np.mean(cover_vec)),
    }


def zscore_calibration_summary(
    y_true: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
) -> Dict[str, float]:
    """
    Compute standardized residuals z = (y - μ) / σ and summarize:
      mean(z), std(z), mean(|z|), fraction(|z|<=1), fraction(|z|<=2)
    Perfect calibration (Gaussian) → mean≈0, std≈1, |z|<=1 about 68%, |z|<=2 about 95%.
    """
    y = np.asarray(y_true).reshape(-1)
    mu = np.asarray(pred_mean).reshape(-1)
    sd = np.asarray(pred_std).reshape(-1)
    sd = np.maximum(sd, 1e-30)
    z = (y - mu) / sd
    return {
        "z_mean": float(np.mean(z)),
        "z_std": float(np.std(z, ddof=1) if z.size > 1 else 0.0),
        "mean_abs_z": float(np.mean(np.abs(z))),
        "frac_absz_le1": float(np.mean(np.abs(z) <= 1.0)),
        "frac_absz_le2": float(np.mean(np.abs(z) <= 2.0)),
    }
