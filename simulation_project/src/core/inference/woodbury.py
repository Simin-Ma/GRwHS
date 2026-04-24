"""Woodbury identity utilities and fast Gaussian beta samplers."""
from __future__ import annotations

import math

import numpy as np
from numpy.random import Generator


def woodbury_inverse(a: np.ndarray, u: np.ndarray, c: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute inverse using the Woodbury identity."""
    inv_a = np.linalg.inv(a)
    middle = np.linalg.inv(c + v @ inv_a @ u)
    return inv_a - inv_a @ u @ middle @ v @ inv_a


def beta_sample_woodbury(
    X: np.ndarray,
    y: np.ndarray,
    sigma2: float,
    prior_var: np.ndarray,
    rng: Generator,
    *,
    jitter: float = 1e-10,
) -> np.ndarray:
    """Sample beta from its Gaussian posterior using the Woodbury sampler.

    Complexity is O(n^2 p + n^3), versus O(p^3) for direct Cholesky on the
    p-by-p precision matrix. This path is efficient when n << p.

    Posterior:
      Cov(beta | y) = D - D X^T M^{-1} X D
      Mean(beta | y) = D X^T M^{-1} y
      with D = diag(prior_var), M = X D X^T + sigma2 * I_n.

    Algorithm (Bhattacharya et al., 2016):
      u ~ N(0, D)
      delta ~ N(0, sigma2 * I_n)
      w = M^{-1}(y - X u - delta)
      beta = u + D X^T w
    """
    n, p = X.shape
    D = np.maximum(prior_var, jitter)

    XD = X * D                                          # n x p (broadcast D by row)
    M = XD @ X.T                                        # n x n
    np.fill_diagonal(M, M.diagonal() + sigma2)          # M += sigma2 * I

    u = rng.standard_normal(p) * np.sqrt(D)             # u ~ N(0, D)
    delta = rng.standard_normal(n) * math.sqrt(max(sigma2, jitter))  # delta ~ N(0, sigma2 * I)
    w = np.linalg.solve(M, y - X @ u - delta)           # n
    return u + D * (X.T @ w)                            # p


def beta_sample_cholesky(
    XtX: np.ndarray,
    Xty: np.ndarray,
    sigma2: float,
    prior_var: np.ndarray,
    rng: Generator,
    *,
    jitter: float = 1e-10,
) -> np.ndarray:
    """Sample beta via Cholesky on the p-by-p posterior precision.

    Posterior precision: X^T X / sigma2 + diag(1 / prior_var).
    Complexity O(p^3). Prefer beta_sample_woodbury when n << p.
    """
    from scipy.linalg import cho_factor, cho_solve, solve_triangular

    D = np.maximum(prior_var, jitter)
    prior_prec = 1.0 / D
    precision = XtX / sigma2 + np.diag(prior_prec)
    np.fill_diagonal(precision, precision.diagonal() + jitter)
    chol, lower = cho_factor(precision, lower=True, check_finite=False)
    mean = cho_solve((chol, lower), Xty / sigma2, check_finite=False)
    z = rng.standard_normal(precision.shape[0])
    noise = solve_triangular(chol, z, lower=lower, check_finite=False)
    return mean + noise


def beta_sample_block_cholesky(
    Xg: np.ndarray,
    residual_without_g: np.ndarray,
    sigma2: float,
    prior_var_g: np.ndarray,
    rng: Generator,
    *,
    jitter: float = 1e-10,
) -> np.ndarray:
    """Sample a coefficient block beta_g from its exact Gaussian conditional.

    Conditional posterior for a block g:
      beta_g | beta_-g, y ~ N(m_g, Q_g^{-1})
      Q_g = X_g^T X_g / sigma2 + diag(1 / prior_var_g)
      m_g = Q_g^{-1} X_g^T (y - X_-g beta_-g) / sigma2

    This is useful for interwoven/block-refresh moves that improve coupling
    between strong signal groups and hierarchical shrinkage parameters.
    """
    from scipy.linalg import cho_factor, cho_solve, solve_triangular

    Xg_arr = np.asarray(Xg, dtype=float)
    resid_arr = np.asarray(residual_without_g, dtype=float).reshape(-1)
    prior = np.maximum(np.asarray(prior_var_g, dtype=float).reshape(-1), jitter)
    sigma2_use = float(max(sigma2, jitter))

    XtX_g = Xg_arr.T @ Xg_arr
    Xty_g = Xg_arr.T @ resid_arr
    precision = XtX_g / sigma2_use + np.diag(1.0 / prior)
    np.fill_diagonal(precision, precision.diagonal() + jitter)
    chol, lower = cho_factor(precision, lower=True, check_finite=False)
    mean = cho_solve((chol, lower), Xty_g / sigma2_use, check_finite=False)
    z = rng.standard_normal(int(prior.shape[0]))
    noise = solve_triangular(chol, z, lower=lower, check_finite=False)
    return mean + noise

