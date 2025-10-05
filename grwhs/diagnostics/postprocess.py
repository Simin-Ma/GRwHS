# grwhs/diagnostics/postprocess.py
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .shrinkage import (
    regularized_lambda,
    prior_precision,
    shrinkage_kappa,
    variance_budget_omegas,
    slab_spike_ratio,
    edf_by_group,
)

Array = np.ndarray


@dataclass
class DiagnosticsResult:
    """
    Container for post-processed diagnostics.

    Attributes
    ----------
    per_coeff:
        Dict[str, np.ndarray] of shape (p,), median stats per coefficient.
        Keys include: 'kappa', 'omega_group', 'omega_tau', 'omega_lambda',
                      'r', 'Pr_r_gt_1', optionally quantiles with suffixes.
    per_group:
        Dict[str, np.ndarray] of shape (G,), median stats per group.
        Keys include: 'edf', optionally quantiles with suffixes.
    samples_used:
        Number of posterior draws processed after burn-in/thin.
    meta:
        Misc metadata (eps, c, shapes, etc.).
    pandas:
        Optional dict with 'per_coeff'/'per_group' DataFrames (if pandas installed).
    """
    per_coeff: Dict[str, Array]
    per_group: Dict[str, Array]
    samples_used: int
    meta: Dict[str, Any]
    pandas: Optional[Dict[str, Any]] = None


def anchor_log_group_scales(log_phi_samples: Array) -> Array:
    """
    Anchor group log-scales per draw:
      z_g* = z_g - mean(z_{1..G})

    Args:
        log_phi_samples: shape (T, G)

    Returns:
        anchored: shape (T, G)
    """
    Z = np.asarray(log_phi_samples)
    if Z.ndim != 2:
        raise ValueError(f"log_phi_samples must be (T,G), got {Z.shape}")
    Z_bar = Z.mean(axis=1, keepdims=True)
    return Z - Z_bar


def _quantiles(x: Array, qs=(0.05, 0.5, 0.95)) -> Tuple[Array, ...]:
    return tuple(np.quantile(x, q, axis=0) for q in qs)


def _safe_optional_pandas(per_coeff: Dict[str, Array], per_group: Dict[str, Array]):
    try:
        import pandas as pd  # type: ignore

        p = per_coeff["kappa"].size
        coeff_cols = {k: v for k, v in per_coeff.items()}
        df_coeff = pd.DataFrame(coeff_cols)
        df_coeff.index.name = "j"

        G = per_group["edf"].size
        group_cols = {k: v for k, v in per_group.items()}
        df_group = pd.DataFrame(group_cols)
        df_group.index.name = "g"

        return {"per_coeff": df_coeff, "per_group": df_group}
    except Exception:
        return None


def compute_diagnostics_from_samples(
    *,
    X: Array,
    group_index: Array,
    c: float,
    eps: float,
    # Samples: shapes (T, p) or (T, G) or (T,)
    beta: Optional[Array] = None,             # unused for diagnostics here, but kept for extensibility
    lambda_: Array,                           # (T, p)
    tau: Array,                               # (T,)
    phi: Array,                               # (T, G)
    sigma: Array,                             # (T,)
) -> DiagnosticsResult:
    """
    End-to-end diagnostics per paper's Algorithm "Post-processing".

    Given posterior samples of (phi_g, tau, lambda_j, sigma) and design X,
    compute for each draw:
      tilde_lambda_j^2, d_j, q_j, kappa_j,
      L*_j and omegas, r_j and indicator(r_j > 1), and edf_g=sum kappa_j in group.

    Finally aggregate to median/quantiles and probability summaries.

    Args
    ----
    X : (n, p) design matrix; only diag(X^T X) is required.
    group_index : (p,) int array mapping coefficient j -> group g in [0, G-1].
    c : slab width (>0)
    eps : small positive for numerical stability in omega budget.
    beta : (T, p) optional posterior samples of beta (not needed here).
    lambda_ : (T, p) local scales samples (>0)
    tau : (T,) global scale samples (>0)
    phi : (T, G) group scales samples (>0)
    sigma : (T,) noise scale samples (>0)

    Returns
    -------
    DiagnosticsResult
    """
    X = np.asarray(X)
    gidx = np.asarray(group_index).astype(int)
    n, p = X.shape
    if gidx.shape != (p,):
        raise ValueError(f"group_index must be shape (p,), got {gidx.shape}")

    # shapes
    lam = np.asarray(lambda_)
    tau_s = np.asarray(tau)
    phi_s = np.asarray(phi)
    sig = np.asarray(sigma)

    if lam.ndim != 2 or lam.shape[1] != p:
        raise ValueError(f"lambda_ must be (T,p), got {lam.shape}")
    T = lam.shape[0]

    if tau_s.shape != (T,):
        raise ValueError(f"tau must be (T,), got {tau_s.shape}")
    G = int(gidx.max()) + 1
    if phi_s.shape != (T, G):
        raise ValueError(f"phi must be (T,G={G}), got {phi_s.shape}")
    if sig.shape != (T,):
        raise ValueError(f"sigma must be (T,), got {sig.shape}")

    # Precompute diagonal of X^T X
    XtX_diag = np.sum(X * X, axis=0)  # shape (p,)

    # Storage for per-draw/per-j diagnostics
    K = np.empty((T, p), dtype=float)
    OMG = np.empty((T, p), dtype=float)
    OMT = np.empty((T, p), dtype=float)
    OML = np.empty((T, p), dtype=float)
    R = np.empty((T, p), dtype=float)
    # per-group EDF
    EDF = np.empty((T, G), dtype=float)

    # Map phi per draw to length-p vector by indexing group
    for t in range(T):
        phi_j = phi_s[t, gidx]           # (p,)
        tau_t = tau_s[t]                 # scalar
        sig_t = sig[t]                   # scalar
        lam_t = lam[t]                   # (p,)

        # tilde_lambda^2 and tilde (not squared) for omegas
        tls = regularized_lambda(lam_t, tau_t, c)  # (p,) squared
        tilde = np.sqrt(tls)

        d = prior_precision(phi_j, tau_t, tls, sig_t)  # (p,)
        kappa = shrinkage_kappa(XtX_diag, sig_t * sig_t, d)  # (p,)
        omegas = variance_budget_omegas(phi_j, tau_t, tilde, eps=eps)
        r = slab_spike_ratio(tau_t, lam_t, c)

        # group edf
        edf = edf_by_group(kappa, gidx, G=G)

        # write
        K[t] = kappa
        OMG[t] = omegas["omega_group"]
        OMT[t] = omegas["omega_tau"]
        OML[t] = omegas["omega_lambda"]
        R[t] = r
        EDF[t] = edf

    # Aggregate quantiles (median + 5%/95%) and probabilities
    k_lo, k_md, k_hi = _quantiles(K)
    og_lo, og_md, og_hi = _quantiles(OMG)
    ot_lo, ot_md, ot_hi = _quantiles(OMT)
    ol_lo, ol_md, ol_hi = _quantiles(OML)
    r_lo, r_md, r_hi = _quantiles(R)

    # Event probability Pr(r_j > 1)
    pr_r_gt_1 = (R > 1.0).mean(axis=0)

    # Group EDF
    edf_lo, edf_md, edf_hi = _quantiles(EDF)

    per_coeff = {
        "kappa": k_md,
        "kappa_lo": k_lo,
        "kappa_hi": k_hi,
        "omega_group": og_md,
        "omega_group_lo": og_lo,
        "omega_group_hi": og_hi,
        "omega_tau": ot_md,
        "omega_tau_lo": ot_lo,
        "omega_tau_hi": ot_hi,
        "omega_lambda": ol_md,
        "omega_lambda_lo": ol_lo,
        "omega_lambda_hi": ol_hi,
        "r": r_md,
        "r_lo": r_lo,
        "r_hi": r_hi,
        "Pr_r_gt_1": pr_r_gt_1,
        # helpful indices
        "group_index": gidx.copy(),
    }

    per_group = {
        "edf": edf_md,
        "edf_lo": edf_lo,
        "edf_hi": edf_hi,
    }

    meta = {
        "T": int(T),
        "n": int(n),
        "p": int(p),
        "G": int(G),
        "eps": float(eps),
        "c": float(c),
        "has_beta_samples": beta is not None,
        "notes": "Diagnostics computed per theory: kappa, omegas (priors-only), r, edf",
    }

    pandas_payload = _safe_optional_pandas(per_coeff, per_group)

    return DiagnosticsResult(
        per_coeff=per_coeff,
        per_group=per_group,
        samples_used=T,
        meta=meta,
        pandas=pandas_payload,
    )
