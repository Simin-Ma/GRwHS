r"""Shrinkage diagnostics utilities aligned with GRRHS theory."""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
from numpy.random import Generator, default_rng
from scipy.sparse.linalg import LinearOperator, cg

ArrayLike = np.ndarray

_EPS_DEFAULT = 1e-12


def _ensure_array(x: ArrayLike, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        raise ValueError(f"{name} must not be scalar; got shape {arr.shape}.")
    return arr


def regularized_lambda(lambda_raw: ArrayLike, tau: float, c: float) -> np.ndarray:
    r"""Compute RHS regularized local scales :math:`\tilde{\lambda}_j^2`.

    Parameters
    ----------
    lambda_raw : array-like
        Local half-Cauchy draws (:math:`\lambda_j` > 0).
    tau : float
        Global scale (:math:`\tau` > 0).
    c : float
        Slab width (>0).

    Returns
    -------
    np.ndarray
        Squared regularized scales (:math:`\tilde{\lambda}_j^2`).
    """

    lam = _ensure_array(lambda_raw, name="lambda_raw")
    if c <= 0:
        raise ValueError("c must be positive for regularized lambda computation.")
    if tau <= 0:
        raise ValueError("tau must be positive for regularized lambda computation.")

    lam2 = np.square(lam)
    tau2 = float(tau) ** 2
    c2 = float(c) ** 2
    denom = c2 + tau2 * lam2
    return (c2 * lam2) / np.maximum(denom, _EPS_DEFAULT)


def prior_precision(phi_j: ArrayLike, tau: float, tilde_lambda_sq: ArrayLike, sigma: float) -> np.ndarray:
    r"""Prior precision :math:`d_j` for coefficients.

    Computes :math:`(\phi_{g(j)}^2\, \tau^2\, \tilde{\lambda}_j^2\, \sigma^2)^{-1}`.
    """

    phi = _ensure_array(phi_j, name="phi_j")
    tls = _ensure_array(tilde_lambda_sq, name="tilde_lambda_sq")
    if phi.shape != tls.shape:
        raise ValueError("phi_j and tilde_lambda_sq must share the same shape.")
    if tau <= 0 or sigma <= 0:
        raise ValueError("tau and sigma must be positive for prior precision.")

    tau2 = float(tau) ** 2
    sigma2 = float(sigma) ** 2
    denom = np.maximum(np.square(phi), _EPS_DEFAULT) * tau2 * np.maximum(tls, _EPS_DEFAULT) * sigma2
    return 1.0 / denom


def shrinkage_kappa(XtX_diag: ArrayLike, sigma2: float, prior_prec: ArrayLike) -> np.ndarray:
    r"""Ridge-style shrinkage factor :math:`\kappa_j`.

    With :math:`q_j = \sigma^{-2} (X^\top X)_{jj}` and :math:`d_j` the prior precision,
    returns :math:`\kappa_j = q_j / (q_j + d_j)`.
    """

    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive when computing shrinkage factors.")

    q = _ensure_array(XtX_diag, name="XtX_diag") / float(sigma2)
    d = _ensure_array(prior_prec, name="prior_prec")
    if q.shape != d.shape:
        raise ValueError("XtX_diag and prior_prec must share the same shape.")

    denom = q + d
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.divide(q, denom, out=np.zeros_like(q), where=denom > 0)
    return kappa


def variance_budget_omegas(
    phi_j: ArrayLike,
    tau: float,
    tilde_lambda: ArrayLike,
    *,
    eps: float = 1e-8,
) -> Dict[str, np.ndarray]:
    r"""Compute normalized variance budget contributions (priors-only)."""

    if eps <= 0:
        raise ValueError("eps must be positive for variance budget computation.")

    phi = _ensure_array(phi_j, name="phi_j")
    tilde = _ensure_array(tilde_lambda, name="tilde_lambda")
    if phi.shape != tilde.shape:
        raise ValueError("phi_j and tilde_lambda must share shape.")
    if tau <= 0:
        raise ValueError("tau must be positive for variance budget computation.")

    phi_clip = np.maximum(phi, eps)
    tau_clip = max(float(tau), eps)
    tilde_clip = np.maximum(tilde, eps)

    term_group = 2.0 * np.log(phi_clip)
    term_tau = 2.0 * np.log(tau_clip)
    term_lambda = 2.0 * np.log(tilde_clip)

    def _sym_clip(x: np.ndarray) -> np.ndarray:
        sign = np.where(x >= 0.0, 1.0, -1.0)
        return sign * np.maximum(np.abs(x), eps)

    clipped_group = _sym_clip(term_group)
    clipped_tau = _sym_clip(term_tau)
    clipped_lambda = _sym_clip(term_lambda)

    denom = clipped_group + clipped_tau + clipped_lambda
    denom = np.where(np.abs(denom) < eps, np.sign(denom + eps) * 3.0 * eps, denom)

    omega_group = clipped_group / denom
    omega_tau = clipped_tau / denom
    omega_lambda = clipped_lambda / denom

    return {
        "omega_group": omega_group,
        "omega_tau": omega_tau,
        "omega_lambda": omega_lambda,
    }


def _ensure_rng(rng: Optional[Union[int, Generator]]) -> Generator:
    if isinstance(rng, Generator):
        return rng
    return default_rng(rng)


def estimate_kappa_hutchinson_cg(
    X: ArrayLike,
    sigma: float,
    prior_prec: ArrayLike,
    *,
    num_probes: int = 10,
    tol: float = 1e-3,
    maxiter: Optional[int] = None,
    probe_space: str = "coef",
    rng: Optional[Union[int, Generator]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    r"""
    Estimate coefficient-space shrinkage :math:`\kappa_j` via Hutchinson + CG.

    Optionally also estimates the diagonal of the hat matrix in data-space
    when ``probe_space == 'data'``.
    """

    X = np.asarray(X, dtype=float)
    prior = _ensure_array(prior_prec, name="prior_prec")
    if prior.ndim != 1:
        raise ValueError("prior_prec must be a one-dimensional array.")
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional design matrix.")
    n, p = X.shape
    if prior.shape[0] != p:
        raise ValueError("prior_prec length must match number of columns in X.")
    if sigma <= 0:
        raise ValueError("sigma must be positive in Hutchinson estimator.")
    if num_probes <= 0:
        raise ValueError("num_probes must be positive in Hutchinson estimator.")
    if tol <= 0:
        raise ValueError("tol must be positive in Hutchinson estimator.")

    probe_space = probe_space.lower()
    if probe_space not in {"coef", "data"}:
        raise ValueError("probe_space must be 'coef' or 'data'.")

    sigma2 = float(sigma) ** 2
    sigma2_inv = 1.0 / sigma2
    rng_obj = _ensure_rng(rng)

    def matvec(vec: np.ndarray) -> np.ndarray:
        return prior * vec + sigma2_inv * (X.T @ (X @ vec))

    linear_op = LinearOperator((p, p), matvec=matvec)
    accum = np.zeros(p, dtype=float)
    hat_accum = np.zeros(n, dtype=float) if probe_space == "data" else None
    dense_cache: Optional[np.ndarray] = None

    def solve(vec: np.ndarray) -> np.ndarray:
        nonlocal dense_cache
        sol, info = cg(linear_op, vec, rtol=tol, atol=0.0, maxiter=maxiter)
        if info != 0:
            if dense_cache is None:
                dense_cache = sigma2_inv * (X.T @ X) + np.diag(prior)
            sol = np.linalg.solve(dense_cache, vec)
        return sol

    for _ in range(num_probes):
        r = rng_obj.choice([-1.0, 1.0], size=p).astype(float)
        s = solve(r)
        v = X @ s
        u = sigma2_inv * (X.T @ v)
        accum += r * u

        if hat_accum is not None:
            z = rng_obj.choice([-1.0, 1.0], size=n).astype(float)
            b = sigma2_inv * (X.T @ z)
            s_data = solve(b)
            h_vec = sigma2_inv * (X @ s_data)
            hat_accum += z * h_vec

    kappa_est = accum / float(num_probes)
    hat_diag_est = None if hat_accum is None else hat_accum / float(num_probes)
    return kappa_est, hat_diag_est


def slab_spike_ratio(tau: float, lambda_raw: ArrayLike, c: float) -> np.ndarray:
    r"""Compute slab-to-spike ratio :math:`r_j = (\tau^2 \lambda_j^2) / c^2`."""

    lam = _ensure_array(lambda_raw, name="lambda_raw")
    if tau <= 0 or c <= 0:
        raise ValueError("tau and c must be positive for slab-spike ratio.")
    tau2 = float(tau) ** 2
    c2 = float(c) ** 2
    return (tau2 * np.square(lam)) / c2


def edf_by_group(kappa: ArrayLike, group_index: ArrayLike, *, G: Optional[int] = None) -> np.ndarray:
    """Aggregate shrinkage factors to effective degrees of freedom per group."""

    kap = _ensure_array(kappa, name="kappa")
    gidx = np.asarray(group_index, dtype=int)
    if kap.shape != gidx.shape:
        raise ValueError("kappa and group_index must share the same shape.")
    if kap.ndim != 1:
        raise ValueError("kappa and group_index must be one-dimensional arrays.")

    if G is None:
        if gidx.size == 0:
            return np.zeros(0, dtype=float)
        G = int(gidx.max()) + 1
    if G <= 0:
        raise ValueError("Number of groups G must be positive.")

    edf = np.zeros(G, dtype=float)
    np.add.at(edf, gidx, kap)
    return edf


def compute_kappa(XtX_diag: ArrayLike, sigma2: float, prior_prec: ArrayLike) -> np.ndarray:
    r"""Convenience wrapper returning :math:`\kappa_j` given design diagonal."""

    return shrinkage_kappa(XtX_diag, sigma2, prior_prec)
