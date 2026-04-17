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
    """Sample 尾 ~ N(渭_尾, 危_尾) via Bhattacharya (2016) fast algorithm.

    Complexity O(n虏p + n鲁) versus O(p鲁) for the direct Cholesky approach.
    Efficient when n << p.

    Posterior: 危_尾 = D - D X岬€ M鈦宦?X D,  渭_尾 = D X岬€ M鈦宦?y
    where D = diag(prior_var), M = X D X岬€ + 蟽虏I.

    Algorithm (Bhattacharya et al., 2016, Biometrika):
        u  ~ N(0, D)
        未  ~ N(0, 蟽虏I_n)
        w   = M鈦宦?(y 鈭?Xu 鈭?未)
        尾   = u + D X岬€ w
    """
    n, p = X.shape
    D = np.maximum(prior_var, jitter)

    XD = X * D                                          # n脳p  (broadcast D as row)
    M = XD @ X.T                                        # n脳n
    np.fill_diagonal(M, M.diagonal() + sigma2)          # M += 蟽虏I

    u = rng.standard_normal(p) * np.sqrt(D)             # u ~ N(0, D)
    delta = rng.standard_normal(n) * math.sqrt(max(sigma2, jitter))  # 未 ~ N(0, 蟽虏I)
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
    """Sample 尾 from its Gaussian posterior via Cholesky on the p脳p precision.

    Posterior precision: X岬€ X / 蟽虏 + diag(1/prior_var).
    Complexity O(p鲁); use beta_sample_woodbury for n << p.
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

