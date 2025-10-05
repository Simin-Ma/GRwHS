r"""Shrinkage diagnostics utilities aligned with GRwHS theory."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

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
    """Compute normalized variance budget contributions (priors-only)."""

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

    L_star = term_group + term_tau + term_lambda
    sign = np.where(L_star >= 0, 1.0, -1.0)
    sign[L_star == 0.0] = 1.0
    L_star_safe = np.where(np.abs(L_star) < eps, sign * eps, L_star)

    omega_group = term_group / L_star_safe
    omega_tau = term_tau / L_star_safe
    omega_lambda = term_lambda / L_star_safe

    return {
        "omega_group": omega_group,
        "omega_tau": omega_tau,
        "omega_lambda": omega_lambda,
    }


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
