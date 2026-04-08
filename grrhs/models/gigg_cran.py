"""CRAN gigg (R package) compatible sampler port (Python).

This module provides a nearly line-by-line translation of the CRAN `gigg` package
Gibbs samplers into NumPy-based Python for reference and compatibility.

Primary upstream reference:
  - CRAN source `gigg_0.2.1`, file `src/gigg_sampler.cpp`
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.random import Generator, default_rng


def chol_solve(M: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Solve M * U = V for SPD M via Cholesky (CRAN gigg::chol_solve analogue)."""
    mat = np.asarray(M, dtype=float)
    vec = np.asarray(V, dtype=float)
    chol = np.linalg.cholesky(mat)
    b_star = np.linalg.solve(chol.T, vec)
    return np.linalg.solve(chol, b_star)


def quick_solve(XtX_inv: np.ndarray, D_pos: np.ndarray, vec_draw: np.ndarray) -> np.ndarray:
    """
    CRAN gigg::quick_solve port.

    Computes the solution to (XtX + D) * U = vec_draw given XtX_inv = (XtX)^{-1} and D_pos = sqrt(diag(D)).
    """
    XtX_inv = np.asarray(XtX_inv, dtype=float)
    D_pos = np.asarray(D_pos, dtype=float).reshape(-1)
    vec_draw = np.asarray(vec_draw, dtype=float).reshape(-1)
    p = int(XtX_inv.shape[0])
    alpha_mat = np.zeros((p, p), dtype=float)
    beta_mat = np.zeros((p, p), dtype=float)
    beta_mat[:, 0] = XtX_inv @ vec_draw
    for j in range(p):
        alpha_mat[:, j] = D_pos[j] * XtX_inv[:, j]
    for k in range(p - 1):
        denom = 1.0 + D_pos[k] * alpha_mat[k, k]
        store_k_vec = (D_pos[k] / denom) * alpha_mat[:, k]
        for j in range(p - k - 1, 0, -1):
            alpha_mat[:, j + k] = alpha_mat[:, j + k] - alpha_mat[k, j + k] * store_k_vec
        beta_mat[:, k + 1] = beta_mat[:, k] - beta_mat[k, k] * store_k_vec
    denom_last = 1.0 + D_pos[p - 1] * alpha_mat[p - 1, p - 1]
    return beta_mat[:, p - 1] - (D_pos[p - 1] * beta_mat[p - 1, p - 1] / denom_last) * alpha_mat[:, p - 1]


def _digamma_approx(x: float) -> float:
    xx = float(x)
    if xx <= 0.0:
        xx = 1e-12
    out = 0.0
    while xx < 6.0:
        out -= 1.0 / xx
        xx += 1.0
    inv = 1.0 / xx
    inv2 = inv * inv
    out += math.log(xx) - 0.5 * inv - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))
    return out


def _trigamma_approx(x: float) -> float:
    xx = float(x)
    if xx <= 0.0:
        xx = 1e-12
    out = 0.0
    while xx < 6.0:
        out += 1.0 / (xx * xx)
        xx += 1.0
    inv = 1.0 / xx
    inv2 = inv * inv
    out += inv + 0.5 * inv2 + inv2 * inv * (1.0 / 6.0 - inv2 * (1.0 / 30.0 - inv2 / 42.0))
    return out


def digamma_inv(y: float, precision: float = 1e-8, max_iter: int = 200) -> float:
    """CRAN gigg::digamma_inv port (Newton iterations)."""
    yy = float(y)
    if yy >= -2.22:
        x_old = math.exp(yy) + 0.5
    else:
        x_old = -1.0 / (yy - _digamma_approx(1.0))
    x_new = x_old - (_digamma_approx(x_old) - yy) / max(_trigamma_approx(x_old), 1e-12)
    it = 0
    while abs(x_new - x_old) >= float(precision) and it < max_iter:
        x_old = x_new
        x_new = x_old - (_digamma_approx(x_old) - yy) / max(_trigamma_approx(x_old), 1e-12)
        it += 1
    return float(x_new)


def rgig_cpp(chi: float, psi: float, lambda_param: float, *, rng: Optional[Generator] = None) -> float:
    """
    Python port of CRAN gigg::rgig_cpp (scalar).

    NOTE: This is a direct algorithmic port and uses rejection sampling; it can be slow in worst cases.
    """
    if rng is None:
        rng = default_rng()
    chi = float(chi)
    psi = float(psi)
    lam = float(lambda_param)
    alpha = math.sqrt(psi / chi)
    beta = math.sqrt(chi * psi)
    final_draw = 0.0

    def target(x: float) -> float:
        if x <= 0.0:
            return 0.0
        return (x ** (lam - 1.0)) * math.exp(-(beta / 2.0) * (x + 1.0 / x))

    if (lam > 1.0) or (beta > 1.0):
        m = (math.sqrt((lam - 1.0) ** 2 + beta**2) + (lam - 1.0)) / beta
        a = -2.0 * (lam + 1.0) / beta - m
        b = 2.0 * (lam - 1.0) * m / beta - 1.0
        c = m
        p = b - a**2 / 3.0
        q = 2.0 * a**3 / 27.0 - a * b / 3.0 + c
        phi = math.acos(-(q / 2.0) * math.sqrt(-27.0 / (p**3)))
        x_minus = math.sqrt(-(4.0 / 3.0) * p) * math.cos(phi / 3.0 + (4.0 / 3.0) * math.pi) - a / 3.0
        x_plus = math.sqrt(-(4.0 / 3.0) * p) * math.cos(phi / 3.0) - a / 3.0
        v_plus = math.sqrt(target(m))
        u_minus = (x_minus - m) * math.sqrt(target(x_minus))
        u_plus = (x_plus - m) * math.sqrt(target(x_plus))
        while True:
            u_draw = rng.uniform(u_minus, u_plus)
            v_draw = rng.uniform(0.0, v_plus)
            x_draw = u_draw / v_draw + m
            if (x_draw > 0.0) and (v_draw**2 <= target(x_draw)):
                final_draw = x_draw
                break
    elif (0.0 <= lam <= 1.0) and (min(0.5, (2.0 / 3.0) * math.sqrt(1.0 - lam)) <= beta <= 1.0):
        m = beta / ((1.0 - lam) + math.sqrt((1.0 - lam) ** 2 + beta**2))
        x_plus = ((1.0 + lam) + math.sqrt((1.0 + lam) ** 2 + beta**2)) / beta
        v_plus = math.sqrt(target(m))
        u_plus = x_plus * math.sqrt(target(x_plus))
        while True:
            u_draw = rng.uniform(0.0, u_plus)
            v_draw = rng.uniform(0.0, v_plus)
            x_draw = u_draw / v_draw
            if (x_draw > 0.0) and (v_draw**2 <= target(x_draw)):
                final_draw = x_draw
                break
    elif (0.0 <= lam < 1.0) and (0.0 < beta <= (2.0 / 3.0) * math.sqrt(1.0 - lam)):
        m = beta / ((1.0 - lam) + math.sqrt((1.0 - lam) ** 2 + beta**2))
        x0 = beta / (1.0 - lam)
        x_star = max(x0, 2.0 / beta)
        k1 = target(m)
        A1 = k1 * x0
        if x0 < 2.0 / beta:
            k2 = math.exp(-beta)
            if lam == 0.0:
                A2 = k2 * math.log(2.0 / (beta**2))
            else:
                A2 = k2 * ((2.0 / beta) ** lam - x0**lam) / lam
        else:
            k2 = 0.0
            A2 = 0.0
        k3 = x_star ** (lam - 1.0)
        A3 = 2.0 * k3 * math.exp(-x_star * beta / 2.0) / beta
        A = A1 + A2 + A3
        while True:
            u_draw = rng.uniform(0.0, 1.0)
            v_draw = rng.uniform(0.0, A)
            if v_draw <= A1:
                x_draw = x0 * v_draw / A1
                h = k1
            elif v_draw <= A1 + A2:
                v_draw = v_draw - A1
                if lam == 0.0:
                    x_draw = beta * math.exp(v_draw * math.exp(beta))
                else:
                    x_draw = (x0**lam + v_draw * lam / k2) ** (1.0 / lam)
                h = k2 * (x_draw ** (lam - 1.0))
            else:
                v_draw = v_draw - (A1 + A2)
                x_draw = -2.0 / beta * math.log(math.exp(-x_star * beta / 2.0) - v_draw * beta / (2.0 * k3))
                h = k3 * math.exp(-x_draw * beta / 2.0)
            if (x_draw > 0.0) and (u_draw * h <= target(x_draw)):
                final_draw = x_draw
                break
    return float(final_draw / alpha)


def _inv_gamma_draw(shape: float, rate: float, rng: Generator) -> float:
    # CRAN uses 1 / rgamma(shape, scale=1/rate)
    return float(1.0 / rng.gamma(shape=float(shape), scale=float(1.0 / rate)))


def _validate_grp_idx(grp_idx: np.ndarray, p: int) -> Tuple[int, np.ndarray]:
    idx = np.asarray(grp_idx, dtype=int).reshape(-1)
    if idx.size != p:
        raise ValueError("grp_idx must have length p.")
    if np.any(idx < 1):
        raise ValueError("grp_idx must be a 1..G sequence with no skips.")
    uniq = np.unique(idx)
    if np.any(np.diff(uniq) > 1):
        raise ValueError("grp_idx must not skip group ids (1..G with no gaps).")
    # CRAN requires grp_idx sorted (nondecreasing).
    if not np.all(idx[:-1] <= idx[1:]):
        raise ValueError("CRAN gigg expects grp_idx to be sorted (groups contiguous).")
    return int(uniq.size), idx


def _grp_sizes_from_idx(grp_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vals, counts = np.unique(grp_idx, return_counts=True)
    if vals.size == 0:
        raise ValueError("grp_idx is empty.")
    grp_size = counts.astype(int)
    grp_size_cs = np.cumsum(grp_size).astype(int)
    return grp_size, grp_size_cs


def gigg_fixed_gibbs_sampler(
    X: np.ndarray,
    C: np.ndarray,
    Y: np.ndarray,
    *,
    grp_idx: np.ndarray,
    alpha_inits: np.ndarray,
    beta_inits: np.ndarray,
    lambda_sq_inits: np.ndarray,
    gamma_sq_inits: np.ndarray,
    eta_inits: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    tau_sq_init: float = 1.0,
    sigma_sq_init: float = 1.0,
    nu_init: float = 1.0,
    n_burn_in: int = 500,
    n_samples: int = 1000,
    n_thin: int = 1,
    stable_const: float = 1e-7,
    verbose: bool = True,
    btrick: bool = False,
    stable_solve: bool = False,
    seed: int = 0,
) -> Dict[str, Any]:
    """Nearly line-by-line Python port of CRAN `gigg_fixed_gibbs_sampler`."""
    rng = default_rng(int(seed))
    X = np.asarray(X, dtype=float)
    C = np.asarray(C, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    n, M = X.shape
    if C.ndim != 2 or C.shape[0] != n:
        raise ValueError("C must be (n, K) matrix with same n as X.")
    K = int(C.shape[1])
    J, grp_idx = _validate_grp_idx(grp_idx, M)
    grp_size, grp_size_cs = _grp_sizes_from_idx(grp_idx)
    grp_idx0 = grp_idx - 1

    alpha = np.asarray(alpha_inits, dtype=float).reshape(K)
    beta = np.asarray(beta_inits, dtype=float).reshape(M)
    lambda_sq = np.asarray(lambda_sq_inits, dtype=float).reshape(M)
    gamma_sq = np.asarray(gamma_sq_inits, dtype=float).reshape(J)
    eta = np.asarray(eta_inits, dtype=float).reshape(J)
    a = np.asarray(p, dtype=float).reshape(J)
    b = np.asarray(q, dtype=float).reshape(J)
    tau_sq = float(tau_sq_init)
    sigma_sq = float(sigma_sq_init)
    nu = float(nu_init)

    tX = X.T
    XtX = tX @ X
    tC = C.T
    CtC = tC @ C
    CtCinv = np.linalg.inv(CtC)
    CtCinvtC = CtCinv @ tC
    alpha_term1 = CtCinvtC @ Y
    alpha_term2 = CtCinvtC @ X

    alpha_store = np.zeros((K, n_samples), dtype=float)
    beta_store = np.zeros((M, n_samples), dtype=float)
    lambda_store = np.zeros((M, n_samples), dtype=float)
    gamma_store = np.zeros((J, n_samples), dtype=float)
    tau_store = np.zeros(n_samples, dtype=float)
    sigma_store = np.zeros(n_samples, dtype=float)
    eta_store = np.zeros((J, n_samples), dtype=float)
    nu_store = np.zeros(n_samples, dtype=float)

    tau_shape_const = (float(M) + 1.0) / 2.0
    sigma_shape_const = (float(n) + 1.0) / 2.0
    # With beta | sigma_sq, ... ~ N(0, sigma_sq * D), the conjugate sigma_sq conditional
    # includes both RSS and the beta quadratic form. CRAN uses a Half-Cauchy style
    # augmentation which adds the 1/nu term.
    sigma_shape_const = (float(n) + float(M) + 1.0) / 2.0

    def _draw_alpha() -> None:
        nonlocal alpha
        if K == 0:
            return
        mean = alpha_term1 - alpha_term2 @ beta
        cov = sigma_sq * CtCinv
        chol = np.linalg.cholesky(cov)
        alpha = mean + chol @ rng.normal(size=K)

    def _draw_beta_standard() -> None:
        nonlocal beta
        ytilde = Y - (C @ alpha if K else 0.0)
        D_no_sigma = tau_sq * gamma_sq[grp_idx0] * lambda_sq
        D_no_sigma = np.maximum(D_no_sigma, 1e-300)
        precision = XtX.copy()
        precision.flat[:: M + 1] += 1.0 / D_no_sigma
        mean_rhs = tX @ ytilde
        mean = chol_solve(precision, mean_rhs) if stable_solve else np.linalg.solve(precision, mean_rhs)
        cholP = np.linalg.cholesky(precision)
        noise = np.linalg.solve(cholP.T, rng.normal(size=M))
        beta = mean + math.sqrt(max(sigma_sq, 0.0)) * noise

    def _draw_beta_btrick() -> None:
        nonlocal beta
        ytilde = Y - (C @ alpha if K else 0.0)
        D_no_sigma = tau_sq * gamma_sq[grp_idx0] * lambda_sq
        D_no_sigma = np.maximum(D_no_sigma, 1e-300)
        D = sigma_sq * D_no_sigma
        u = rng.normal(size=M) * np.sqrt(D)
        delta = rng.normal(size=n)
        v = (X @ u) / math.sqrt(max(sigma_sq, 1e-300)) + delta
        theta = (D[:, None] * tX)  # (M,n)
        mat = (X @ theta) / max(sigma_sq, 1e-300) + np.eye(n, dtype=float)
        rhs = ytilde / math.sqrt(max(sigma_sq, 1e-300)) - v
        w = np.linalg.solve(mat, rhs)
        beta = u + (theta @ w) / math.sqrt(max(sigma_sq, 1e-300))

    def _draw_tau_sq() -> None:
        nonlocal tau_sq
        denom = np.maximum(sigma_sq * gamma_sq[grp_idx0] * lambda_sq, 1e-300)
        quad = float(np.sum((beta * beta) / denom))
        rate = quad / 2.0 + 1.0 / max(nu, 1e-300)
        tau_sq = _inv_gamma_draw(tau_shape_const, rate, rng)

    def _draw_sigma_sq() -> None:
        nonlocal sigma_sq
        resid = Y - (C @ alpha if K else 0.0) - X @ beta
        rss = float(resid @ resid)
        denom = np.maximum(tau_sq * gamma_sq[grp_idx0] * lambda_sq, 1e-300)
        beta_quad = float(np.sum((beta * beta) / denom))
        rate = (rss + beta_quad) / 2.0 + 1.0 / max(nu, 1e-300)
        sigma_sq = _inv_gamma_draw(sigma_shape_const, rate, rng)

    def _draw_gamma_lambda_eta() -> None:
        nonlocal gamma_sq, lambda_sq, eta
        for j in range(J):
            start = 0 if j == 0 else int(grp_size_cs[j - 1])
            end = int(grp_size_cs[j])
            stable_psi = float(np.sum(beta[start:end] ** 2 / np.maximum(lambda_sq[start:end], 1e-300)))
            stable_psi *= 1.0 / max(sigma_sq * tau_sq, 1e-300)
            stable_psi = max(stable_psi, float(stable_const))
            group_half = float(grp_size[j]) / 2.0
            if group_half < float(a[j]):
                gamma_sq[j] = rgig_cpp(stable_psi, 2.0 * float(eta[j]), float(a[j]) - group_half, rng=rng)
            else:
                gamma_sq[j] = 1.0 / rgig_cpp(2.0 * float(eta[j]), stable_psi, group_half - float(a[j]), rng=rng)
            for pos in range(start, end):
                rate = float(eta[j]) + float(beta[pos] ** 2) / (2.0 * max(sigma_sq * tau_sq * gamma_sq[j], 1e-300))
                lambda_sq[pos] = _inv_gamma_draw(float(b[j]) + 0.5, rate, rng)
            eta[j] = 1.0

    def _draw_nu() -> None:
        nonlocal nu
        rate = (1.0 / tau_sq) + (1.0 / sigma_sq)
        nu = _inv_gamma_draw(1.0, rate, rng)

    # burn-in
    for cnt in range(int(n_burn_in)):
        _draw_alpha()
        (_draw_beta_btrick if btrick else _draw_beta_standard)()
        _draw_tau_sq()
        _draw_sigma_sq()
        _draw_gamma_lambda_eta()
        _draw_nu()
        if verbose and (cnt + 1) % 500 == 0:
            print(f"{cnt+1} Burn-in Draws")
    if verbose:
        print("Burn-in Iterations Complete")

    # sampling
    cnt = 0
    total_saved = 0
    while total_saved < int(n_samples):
        _draw_alpha()
        (_draw_beta_btrick if btrick else _draw_beta_standard)()
        _draw_tau_sq()
        _draw_sigma_sq()
        _draw_gamma_lambda_eta()
        _draw_nu()
        if cnt % int(n_thin) == 0:
            tau_store[total_saved] = tau_sq
            sigma_store[total_saved] = sigma_sq
            nu_store[total_saved] = nu
            alpha_store[:, total_saved] = alpha
            beta_store[:, total_saved] = beta
            lambda_store[:, total_saved] = lambda_sq
            gamma_store[:, total_saved] = gamma_sq
            eta_store[:, total_saved] = eta
            total_saved += 1
            if verbose and (total_saved % 500 == 0):
                print(f"{total_saved} Samples Drawn")
        cnt += 1

    return {
        "alphas": alpha_store,
        "betas": beta_store,
        "lambda_sqs": lambda_store,
        "gamma_sqs": gamma_store,
        "tau_sqs": tau_store,
        "sigma_sqs": sigma_store,
        "a": a,
        "b": b,
        "X": X,
        "C": C,
        "Y": Y,
        "grp_idx": grp_idx,
        "n_burn_in": int(n_burn_in),
        "n_samples": int(n_samples),
        "n_thin": int(n_thin),
    }


def gigg_mmle_gibbs_sampler(
    X: np.ndarray,
    C: np.ndarray,
    Y: np.ndarray,
    *,
    grp_idx: np.ndarray,
    alpha_inits: np.ndarray,
    beta_inits: np.ndarray,
    lambda_sq_inits: np.ndarray,
    gamma_sq_inits: np.ndarray,
    eta_inits: np.ndarray,
    p_inits: np.ndarray,
    q_inits: np.ndarray,
    tau_sq_init: float = 1.0,
    sigma_sq_init: float = 1.0,
    nu_init: float = 1.0,
    n_burn_in: int = 500,
    n_samples: int = 1000,
    n_thin: int = 1,
    stable_const: float = 1e-7,
    verbose: bool = True,
    btrick: bool = False,
    stable_solve: bool = False,
    seed: int = 0,
    mmle_samp_size: int = 1000,
    mmle_tol_scale: float = 1e-4,
    max_mmle_iters: int = 50000,
) -> Dict[str, Any]:
    """
    Nearly line-by-line Python port of CRAN `gigg_mmle_gibbs_sampler`.

    Differences vs the C++:
    - Adds `max_mmle_iters` hard stop to prevent infinite loops in degenerate cases.
    """
    rng = default_rng(int(seed))
    X = np.asarray(X, dtype=float)
    C = np.asarray(C, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    n, M = X.shape
    if C.ndim != 2 or C.shape[0] != n:
        raise ValueError("C must be (n, K) matrix with same n as X.")
    K = int(C.shape[1])
    J, grp_idx = _validate_grp_idx(grp_idx, M)
    grp_size, grp_size_cs = _grp_sizes_from_idx(grp_idx)
    grp_idx0 = grp_idx - 1

    alpha = np.asarray(alpha_inits, dtype=float).reshape(K)
    beta = np.asarray(beta_inits, dtype=float).reshape(M)
    lambda_sq = np.asarray(lambda_sq_inits, dtype=float).reshape(M)
    gamma_sq = np.asarray(gamma_sq_inits, dtype=float).reshape(J)
    eta = np.asarray(eta_inits, dtype=float).reshape(J)
    p_vec = np.asarray(p_inits, dtype=float).reshape(J)
    q_vec = np.asarray(q_inits, dtype=float).reshape(J)
    tau_sq = float(tau_sq_init)
    sigma_sq = float(sigma_sq_init)
    nu = float(nu_init)

    tX = X.T
    XtX = tX @ X
    tC = C.T
    CtC = tC @ C
    CtCinv = np.linalg.inv(CtC)
    CtCinvtC = CtCinv @ tC
    alpha_term1 = CtCinvtC @ Y
    alpha_term2 = CtCinvtC @ X

    alpha_store = np.zeros((K, n_samples), dtype=float)
    beta_store = np.zeros((M, n_samples), dtype=float)
    lambda_store = np.zeros((M, n_samples), dtype=float)
    gamma_store = np.zeros((J, n_samples), dtype=float)
    tau_store = np.zeros(n_samples, dtype=float)
    sigma_store = np.zeros(n_samples, dtype=float)
    eta_store = np.zeros((J, n_samples), dtype=float)
    nu_store = np.zeros(n_samples, dtype=float)

    tau_shape_const = (float(M) + 1.0) / 2.0
    sigma_shape_const = (float(n) + float(M) + 1.0) / 2.0

    def _draw_alpha() -> None:
        nonlocal alpha
        if K == 0:
            return
        mean = alpha_term1 - alpha_term2 @ beta
        cov = sigma_sq * CtCinv
        chol = np.linalg.cholesky(cov)
        alpha = mean + chol @ rng.normal(size=K)

    def _draw_beta_standard() -> None:
        nonlocal beta
        ytilde = Y - (C @ alpha if K else 0.0)
        D_no_sigma = tau_sq * gamma_sq[grp_idx0] * lambda_sq
        D_no_sigma = np.maximum(D_no_sigma, 1e-300)
        precision = XtX.copy()
        precision.flat[:: M + 1] += 1.0 / D_no_sigma
        mean_rhs = tX @ ytilde
        mean = chol_solve(precision, mean_rhs) if stable_solve else np.linalg.solve(precision, mean_rhs)
        cholP = np.linalg.cholesky(precision)
        noise = np.linalg.solve(cholP.T, rng.normal(size=M))
        beta = mean + math.sqrt(max(sigma_sq, 0.0)) * noise

    def _draw_beta_btrick() -> None:
        nonlocal beta
        ytilde = Y - (C @ alpha if K else 0.0)
        D_no_sigma = tau_sq * gamma_sq[grp_idx0] * lambda_sq
        D_no_sigma = np.maximum(D_no_sigma, 1e-300)
        D = sigma_sq * D_no_sigma
        u = rng.normal(size=M) * np.sqrt(D)
        delta = rng.normal(size=n)
        v = (X @ u) / math.sqrt(max(sigma_sq, 1e-300)) + delta
        theta = (D[:, None] * tX)  # (M,n)
        mat = (X @ theta) / max(sigma_sq, 1e-300) + np.eye(n, dtype=float)
        rhs = ytilde / math.sqrt(max(sigma_sq, 1e-300)) - v
        w = np.linalg.solve(mat, rhs)
        beta = u + (theta @ w) / math.sqrt(max(sigma_sq, 1e-300))

    def _draw_tau_sq() -> None:
        nonlocal tau_sq
        denom = np.maximum(sigma_sq * gamma_sq[grp_idx0] * lambda_sq, 1e-300)
        quad = float(np.sum((beta * beta) / denom))
        rate = quad / 2.0 + 1.0 / max(nu, 1e-300)
        tau_sq = _inv_gamma_draw(tau_shape_const, rate, rng)

    def _draw_sigma_sq() -> None:
        nonlocal sigma_sq
        resid = Y - (C @ alpha if K else 0.0) - X @ beta
        rss = float(resid @ resid)
        denom = np.maximum(tau_sq * gamma_sq[grp_idx0] * lambda_sq, 1e-300)
        beta_quad = float(np.sum((beta * beta) / denom))
        rate = (rss + beta_quad) / 2.0 + 1.0 / max(nu, 1e-300)
        sigma_sq = _inv_gamma_draw(sigma_shape_const, rate, rng)

    def _draw_gamma_lambda_eta() -> None:
        nonlocal gamma_sq, lambda_sq, eta
        for j in range(J):
            start = 0 if j == 0 else int(grp_size_cs[j - 1])
            end = int(grp_size_cs[j])
            stable_psi = float(np.sum(beta[start:end] ** 2 / np.maximum(lambda_sq[start:end], 1e-300)))
            stable_psi *= 1.0 / max(sigma_sq * tau_sq, 1e-300)
            stable_psi = max(stable_psi, float(stable_const))
            group_half = float(grp_size[j]) / 2.0
            if group_half < float(p_vec[j]):
                gamma_sq[j] = rgig_cpp(stable_psi, 2.0 * float(eta[j]), float(p_vec[j]) - group_half, rng=rng)
            else:
                gamma_sq[j] = 1.0 / rgig_cpp(2.0 * float(eta[j]), stable_psi, group_half - float(p_vec[j]), rng=rng)
            for pos in range(start, end):
                rate = float(eta[j]) + float(beta[pos] ** 2) / (2.0 * max(sigma_sq * tau_sq * gamma_sq[j], 1e-300))
                lambda_sq[pos] = _inv_gamma_draw(float(q_vec[j]) + 0.5, rate, rng)
            eta[j] = 1.0

    def _draw_nu() -> None:
        nonlocal nu
        rate = (1.0 / tau_sq) + (1.0 / sigma_sq)
        nu = _inv_gamma_draw(1.0, rate, rng)

    # burn-in
    for cnt in range(int(n_burn_in)):
        _draw_alpha()
        (_draw_beta_btrick if btrick else _draw_beta_standard)()
        _draw_tau_sq()
        _draw_sigma_sq()
        _draw_gamma_lambda_eta()
        _draw_nu()
        if verbose and (cnt + 1) % 500 == 0:
            print(f"{cnt+1} Burn-in Draws")
    if verbose:
        print("Burn-in Iterations Complete")

    # MMLE loop
    mmle_samp_size = int(mmle_samp_size)
    terminate_mmle = float(mmle_tol_scale) * float(J)
    lambda_mmle_store = np.zeros((mmle_samp_size, M), dtype=float)
    gamma_mmle_store = np.zeros((mmle_samp_size, J), dtype=float)
    eta_mmle_store = np.zeros((mmle_samp_size, J), dtype=float)
    q_new = q_vec.copy()
    p_new = p_vec.copy()
    delta_mmle = float("inf")
    cnt = 0
    while delta_mmle >= terminate_mmle and cnt < int(max_mmle_iters):
        _draw_alpha()
        (_draw_beta_btrick if btrick else _draw_beta_standard)()
        _draw_tau_sq()
        _draw_sigma_sq()
        _draw_gamma_lambda_eta()
        _draw_nu()
        eta_mmle_store[cnt % mmle_samp_size] = eta
        gamma_mmle_store[cnt % mmle_samp_size] = gamma_sq
        lambda_mmle_store[cnt % mmle_samp_size] = lambda_sq
        if (cnt + 1) % mmle_samp_size == 0:
            for j in range(J):
                start = 0 if j == 0 else int(grp_size_cs[j - 1])
                end = int(grp_size_cs[j])
                log_lambda_mmle_sum = float(np.sum(np.log(lambda_mmle_store[:, start:end])))
                overflow_check = float(np.sum(np.log(eta_mmle_store[:, j]))) / mmle_samp_size - log_lambda_mmle_sum / (
                    float(grp_size[j]) * float(mmle_samp_size)
                )
                q_new[j] = min(float(digamma_inv(overflow_check, precision=1e-8)), 4.0)
                p_new[j] = 1.0 / float(n)
            delta_mmle = float(np.sum((q_new - q_vec) ** 2) + np.sum((p_new - p_vec) ** 2))
            q_vec[:] = q_new
            p_vec[:] = p_new
        cnt += 1
        if verbose and (cnt % 500 == 0):
            print(f"{cnt} MMLE Draws")
    if verbose:
        print("MMLE Estimate Found")

    # sampling
    cnt = 0
    total_saved = 0
    while total_saved < int(n_samples):
        _draw_alpha()
        (_draw_beta_btrick if btrick else _draw_beta_standard)()
        _draw_tau_sq()
        _draw_sigma_sq()
        _draw_gamma_lambda_eta()
        _draw_nu()
        if cnt % int(n_thin) == 0:
            tau_store[total_saved] = tau_sq
            sigma_store[total_saved] = sigma_sq
            nu_store[total_saved] = nu
            alpha_store[:, total_saved] = alpha
            beta_store[:, total_saved] = beta
            lambda_store[:, total_saved] = lambda_sq
            gamma_store[:, total_saved] = gamma_sq
            eta_store[:, total_saved] = eta
            total_saved += 1
            if verbose and (total_saved % 500 == 0):
                print(f"{total_saved} Samples Drawn")
        cnt += 1

    return {
        "alphas": alpha_store,
        "betas": beta_store,
        "lambda_sqs": lambda_store,
        "gamma_sqs": gamma_store,
        "tau_sqs": tau_store,
        "sigma_sqs": sigma_store,
        "a": p_vec,
        "b": q_vec,
        "X": X,
        "C": C,
        "Y": Y,
        "grp_idx": grp_idx,
        "n_burn_in": int(n_burn_in),
        "n_samples": int(n_samples),
        "n_thin": int(n_thin),
        "mmle_iters": int(cnt),
        "mmle_delta": float(delta_mmle),
    }


def _post_summary(draws: np.ndarray, *, axis: int = 1) -> Dict[str, np.ndarray]:
    arr = np.asarray(draws, dtype=float)
    return {
        "mean": np.mean(arr, axis=axis),
        "lcl.95": np.quantile(arr, 0.025, axis=axis),
        "ucl.95": np.quantile(arr, 0.975, axis=axis),
    }


def gigg(
    X: np.ndarray,
    C: np.ndarray,
    Y: np.ndarray,
    *,
    method: str = "mmle",
    grp_idx: np.ndarray,
    alpha_inits: Optional[np.ndarray] = None,
    beta_inits: Optional[np.ndarray] = None,
    a: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    sigma_sq_init: float = 1.0,
    tau_sq_init: float = 1.0,
    n_burn_in: int = 500,
    n_samples: int = 1000,
    n_thin: int = 1,
    verbose: bool = True,
    btrick: bool = False,
    stable_solve: bool = True,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    CRAN-like `gigg()` wrapper (ports the behavior in CRAN `R/gigg_main.R`).

    Returns a dict similar to the R list, including posterior summaries.
    """
    X = np.asarray(X, dtype=float)
    C = np.asarray(C, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    n, p_dim = X.shape
    if C.shape[0] != n:
        raise ValueError("X and C must have the same number of rows and match len(Y).")

    method = str(method).lower()
    grp_idx_arr = np.asarray(grp_idx, dtype=int).reshape(-1)
    G = int(np.unique(grp_idx_arr).size)
    if alpha_inits is None:
        alpha_inits = np.zeros(C.shape[1], dtype=float)
    if beta_inits is None:
        beta_inits = np.zeros(p_dim, dtype=float)
    if a is None:
        a = np.full(G, 0.5, dtype=float)
    if b is None:
        b = np.full(G, 0.5, dtype=float)

    lambda_sq_inits = np.ones(p_dim, dtype=float)
    gamma_sq_inits = np.ones(G, dtype=float)
    eta_inits = np.ones(G, dtype=float)
    nu_init = 1.0
    stable_const = 1e-7

    if method == "fixed":
        draws = gigg_fixed_gibbs_sampler(
            X,
            C,
            Y,
            grp_idx=grp_idx_arr,
            alpha_inits=np.asarray(alpha_inits, dtype=float),
            beta_inits=np.asarray(beta_inits, dtype=float),
            lambda_sq_inits=lambda_sq_inits,
            gamma_sq_inits=gamma_sq_inits,
            eta_inits=eta_inits,
            p=np.asarray(a, dtype=float),
            q=np.asarray(b, dtype=float),
            tau_sq_init=float(tau_sq_init),
            sigma_sq_init=float(sigma_sq_init),
            nu_init=float(nu_init),
            n_burn_in=int(n_burn_in),
            n_samples=int(n_samples),
            n_thin=int(n_thin),
            stable_const=float(stable_const),
            verbose=bool(verbose),
            btrick=bool(btrick),
            stable_solve=bool(stable_solve),
            seed=int(seed),
        )
    elif method == "mmle":
        draws = gigg_mmle_gibbs_sampler(
            X,
            C,
            Y,
            grp_idx=grp_idx_arr,
            alpha_inits=np.asarray(alpha_inits, dtype=float),
            beta_inits=np.asarray(beta_inits, dtype=float),
            lambda_sq_inits=lambda_sq_inits,
            gamma_sq_inits=gamma_sq_inits,
            eta_inits=eta_inits,
            p_inits=np.full(G, 1.0 / float(n), dtype=float),
            q_inits=np.asarray(b, dtype=float),
            tau_sq_init=float(tau_sq_init),
            sigma_sq_init=float(sigma_sq_init),
            nu_init=float(nu_init),
            n_burn_in=int(n_burn_in),
            n_samples=int(n_samples),
            n_thin=int(n_thin),
            stable_const=float(stable_const),
            verbose=bool(verbose),
            btrick=bool(btrick),
            stable_solve=bool(stable_solve),
            seed=int(seed),
        )
    else:
        raise ValueError("method must be either 'fixed' or 'mmle'.")

    alpha_draws = _post_summary(draws["alphas"], axis=1)
    beta_draws = _post_summary(draws["betas"], axis=1)
    sigma_sq_draws = _post_summary(draws["sigma_sqs"].reshape(1, -1), axis=1)

    return {
        "draws": draws,
        "beta.hat": beta_draws["mean"],
        "beta.lcl.95": beta_draws["lcl.95"],
        "beta.ucl.95": beta_draws["ucl.95"],
        "alpha.hat": alpha_draws["mean"],
        "alpha.lcl.95": alpha_draws["lcl.95"],
        "alpha.ucl.95": alpha_draws["ucl.95"],
        "sigma_sq.hat": sigma_sq_draws["mean"].reshape(-1),
        "sigma_sq.lcl.95": sigma_sq_draws["lcl.95"].reshape(-1),
        "sigma_sq.ucl.95": sigma_sq_draws["ucl.95"].reshape(-1),
    }


def _groups_to_contiguous_grp_idx(groups: Sequence[Sequence[int]], p: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (perm, grp_idx_perm) such that grp_idx_perm is sorted (contiguous groups)."""
    order: List[int] = []
    grp_idx_sorted = np.empty(p, dtype=int)
    for g, idxs in enumerate(groups, start=1):
        idx_list = [int(i) for i in idxs]
        order.extend(idx_list)
        for j in idx_list:
            grp_idx_sorted[j] = g
    perm = np.array(order, dtype=int)
    grp_idx_perm = grp_idx_sorted[perm]
    return perm, grp_idx_perm


@dataclass
class GIGGRegressionCRAN:
    """Runner-friendly wrapper that uses the CRAN-compatible samplers."""

    method: str = "fixed"
    n_burn_in: int = 500
    n_samples: int = 1000
    n_thin: int = 1
    seed: int = 0
    num_chains: int = 1
    btrick: bool = False
    stable_solve: bool = True
    fit_intercept: bool = True
    store_lambda: bool = True
    a: Optional[np.ndarray] = None
    b: Optional[np.ndarray] = None

    coef_samples_: Optional[np.ndarray] = field(default=None, init=False)
    alpha_samples_: Optional[np.ndarray] = field(default=None, init=False)
    tau2_samples_: Optional[np.ndarray] = field(default=None, init=False)
    sigma2_samples_: Optional[np.ndarray] = field(default=None, init=False)
    gamma2_samples_: Optional[np.ndarray] = field(default=None, init=False)
    lambda_samples_: Optional[np.ndarray] = field(default=None, init=False)
    tau_samples_: Optional[np.ndarray] = field(default=None, init=False)
    sigma_samples_: Optional[np.ndarray] = field(default=None, init=False)
    gamma_samples_: Optional[np.ndarray] = field(default=None, init=False)
    coef_mean_: Optional[np.ndarray] = field(default=None, init=False)
    alpha_mean_: Optional[np.ndarray] = field(default=None, init=False)
    intercept_: float = field(default=0.0, init=False)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        groups: Sequence[Sequence[int]],
        C: Optional[np.ndarray] = None,
    ) -> "GIGGRegressionCRAN":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n, p = X.shape
        if y.shape[0] != n:
            raise ValueError("X and y must have compatible shapes.")
        if C is None and bool(self.fit_intercept):
            C_arr = np.ones((n, 1), dtype=float)
        else:
            C_arr = np.zeros((n, 0), dtype=float) if C is None else np.asarray(C, dtype=float)
        k = int(C_arr.shape[1])

        perm, grp_idx_perm = _groups_to_contiguous_grp_idx(groups, p)
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(p, dtype=int)
        Xp = X[:, perm]

        def run_chain(chain_seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
            alpha0 = np.zeros(k, dtype=float)
            beta0 = np.zeros(p, dtype=float)
            G = len(groups)
            a_vec = np.full(G, 0.5, dtype=float) if self.a is None else np.asarray(self.a, dtype=float).reshape(G)
            b_vec = np.full(G, 0.5, dtype=float) if self.b is None else np.asarray(self.b, dtype=float).reshape(G)
            out = gigg(
                Xp,
                C_arr,
                y,
                method=str(self.method).lower(),
                grp_idx=grp_idx_perm,
                alpha_inits=alpha0,
                beta_inits=beta0,
                a=a_vec,
                b=b_vec,
                n_burn_in=int(self.n_burn_in),
                n_samples=int(self.n_samples),
                n_thin=int(self.n_thin),
                verbose=False,
                btrick=bool(self.btrick),
                stable_solve=bool(self.stable_solve),
                seed=int(chain_seed),
            )
            draws = out["draws"]
            beta_draws = np.asarray(draws["betas"], dtype=float).T[:, inv_perm]
            alpha_draws = np.asarray(draws["alphas"], dtype=float).T
            tau2 = np.asarray(draws["tau_sqs"], dtype=float).reshape(-1)
            sigma2 = np.asarray(draws["sigma_sqs"], dtype=float).reshape(-1)
            gamma2 = np.asarray(draws["gamma_sqs"], dtype=float).T
            lam = np.asarray(draws["lambda_sqs"], dtype=float).T[:, inv_perm] if self.store_lambda else None
            return beta_draws, alpha_draws, tau2, sigma2, gamma2, lam

        chains = max(1, int(self.num_chains))
        results = [run_chain(int(self.seed) + i) for i in range(chains)]
        beta_stack = np.stack([r[0] for r in results], axis=0)
        alpha_stack = np.stack([r[1] for r in results], axis=0)
        tau2_stack = np.stack([r[2] for r in results], axis=0)
        sigma2_stack = np.stack([r[3] for r in results], axis=0)
        gamma2_stack = np.stack([r[4] for r in results], axis=0)
        lam_stack = None if results[0][5] is None else np.stack([r[5] for r in results], axis=0)

        self.coef_samples_ = beta_stack if chains > 1 else beta_stack[0]
        self.alpha_samples_ = alpha_stack if chains > 1 else alpha_stack[0]
        self.tau2_samples_ = tau2_stack if chains > 1 else tau2_stack[0]
        self.sigma2_samples_ = sigma2_stack if chains > 1 else sigma2_stack[0]
        self.gamma2_samples_ = gamma2_stack if chains > 1 else gamma2_stack[0]
        self.lambda_samples_ = None if lam_stack is None else (lam_stack if chains > 1 else lam_stack[0])

        self.tau_samples_ = np.sqrt(np.maximum(self.tau2_samples_, 0.0))
        self.sigma_samples_ = np.sqrt(np.maximum(self.sigma2_samples_, 0.0))
        self.gamma_samples_ = np.sqrt(np.maximum(self.gamma2_samples_, 0.0))

        beta_flat = beta_stack.reshape(-1, p)
        self.coef_mean_ = beta_flat.mean(axis=0)
        if k == 0:
            self.alpha_mean_ = None
            self.intercept_ = 0.0
        else:
            alpha_flat = alpha_stack.reshape(-1, k)
            self.alpha_mean_ = alpha_flat.mean(axis=0)
            self.intercept_ = float(self.alpha_mean_[0]) if self.alpha_mean_.size else 0.0
        return self

    def predict(self, X: np.ndarray, C: Optional[np.ndarray] = None) -> np.ndarray:
        if self.coef_mean_ is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        X_arr = np.asarray(X, dtype=float)
        yhat = X_arr @ self.coef_mean_
        if self.alpha_mean_ is not None and self.alpha_mean_.size:
            C_arr = np.ones((X_arr.shape[0], 1), dtype=float) if C is None else np.asarray(C, dtype=float)
            yhat = yhat + C_arr @ self.alpha_mean_
        return yhat
