from __future__ import annotations

import math
from typing import Dict

import numpy as np

_EPS = 1e-12


def generate_null_group(pg: int, sigma2: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.normal(loc=0.0, scale=math.sqrt(float(sigma2)), size=int(pg))


def generate_signal_group_distributed(pg: int, mu_g: float, sigma2: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    beta_val = math.sqrt((2.0 * float(sigma2) * float(mu_g)) / max(int(pg), 1))
    beta = np.full(int(pg), beta_val, dtype=float)
    y = rng.normal(loc=beta, scale=math.sqrt(float(sigma2)), size=int(pg))
    return y, beta


def profile_theta_kappa(kappa: np.ndarray, rho: float) -> np.ndarray:
    """
    0415 profile specialization:
      theta(kappa) = kappa * rho^2 / (kappa + (1-kappa)*rho^2)
    """
    kap = np.clip(np.asarray(kappa, dtype=float), _EPS, 1.0 - _EPS)
    rho2 = float(rho) ** 2
    den = kap + (1.0 - kap) * rho2
    return (kap * rho2) / np.maximum(den, _EPS)


def profile_psi_kappa(kappa: np.ndarray, rho: float) -> np.ndarray:
    """
    0415 profile specialization:
      psi(kappa) = kappa * rho^2 / (rho^2 + kappa)
    """
    kap = np.clip(np.asarray(kappa, dtype=float), _EPS, 1.0 - _EPS)
    rho2 = float(rho) ** 2
    den = rho2 + kap
    return (kap * rho2) / np.maximum(den, _EPS)


def kappa_posterior_grid(
    y_group: np.ndarray,
    tau: float,
    sigma2: float,
    alpha_kappa: float,
    beta_kappa: float,
    grid_size: int = 2001,
) -> Dict[str, np.ndarray]:
    y = np.asarray(y_group, dtype=float).reshape(-1)
    pg = y.size
    grid = np.linspace(1e-5, 1.0 - 1e-5, int(grid_size))
    rho = float(tau) / math.sqrt(max(float(sigma2), _EPS))
    Sg = float(np.sum(y * y) / (2.0 * max(float(sigma2), _EPS)))
    theta = profile_theta_kappa(grid, rho=rho)
    psi = profile_psi_kappa(grid, rho=rho)

    # 0415 profile log-likelihood kernel:
    #   ell_g(kappa) = -(p_g/2) * log(1 + theta(kappa)) + S_g * psi(kappa)
    ll = -0.5 * float(pg) * np.log1p(theta) + Sg * psi
    lp = (float(alpha_kappa) - 1.0) * np.log(grid) + (float(beta_kappa) - 1.0) * np.log1p(-grid)
    logp = ll + lp
    logp = logp - float(np.max(logp))
    w = np.exp(logp)
    w = w / np.trapezoid(w, grid)

    return {"kappa": grid, "density": w}


def posterior_summary_from_grid(
    kappa_grid: np.ndarray,
    density: np.ndarray,
    *,
    tail_threshold: float | None = None,
    window_lower: float | None = None,
    window_upper: float | None = None,
) -> Dict[str, float]:
    g = np.asarray(kappa_grid, dtype=float)
    d = np.asarray(density, dtype=float)
    mean = float(np.trapezoid(g * d, g))

    dg = np.diff(g)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (d[1:] + d[:-1]) * dg)])
    cdf = cdf / max(float(cdf[-1]), _EPS)
    median = float(np.interp(0.5, cdf, g))
    var = float(np.trapezoid(((g - mean) ** 2) * d, g))
    sd = math.sqrt(max(var, 0.0))

    out = {
        "post_mean_kappa": mean,
        "post_median_kappa": median,
        "post_sd_kappa": sd,
    }

    if tail_threshold is not None:
        mask = g > float(tail_threshold)
        out["tail_prob"] = float(np.trapezoid(d[mask], g[mask])) if np.any(mask) else 0.0

    if window_lower is not None and window_upper is not None:
        lo = float(window_lower)
        hi = float(window_upper)
        mask = (g >= lo) & (g <= hi)
        out["window_prob"] = float(np.trapezoid(d[mask], g[mask])) if np.any(mask) else 0.0

    return out

