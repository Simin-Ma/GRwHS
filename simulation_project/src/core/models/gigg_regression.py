from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from pathlib import Path
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from numpy.random import Generator, default_rng

from ..utils.parallel_runtime import can_use_process_pool

try:
    from scipy.stats import geninvgauss as _scipy_geninvgauss
    _SCIPY_GIG_AVAILABLE = True
except Exception:
    _scipy_geninvgauss = None
    _SCIPY_GIG_AVAILABLE = False

try:
    from numba import njit as _njit
    _NUMBA_AVAILABLE = True
except Exception:
    _njit = None
    _NUMBA_AVAILABLE = False

_POS_FLOOR = 1e-8
_POS_CAP = 1e3
_BETA_CAP = 1e3
_GIG_MAX_REJECTION_DRAWS = 20000
_GIG_BATCH_SIZE = 256
_GIG_FALLBACK_LOGNORMAL_SIGMA = 0.35

_GIG_SAMPLER_STATS: dict[str, int] = {
    "scipy_success": 0,
    "rejection_success": 0,
    "fallback_used": 0,
}
def _digamma_approx(x: float) -> float:
    """Lightweight digamma approximation with recurrence + asymptotic expansion."""
    xx = float(x)
    if xx <= 0.0:
        xx = _POS_FLOOR
    out = 0.0
    while xx < 6.0:
        out -= 1.0 / xx
        xx += 1.0
    inv = 1.0 / xx
    inv2 = inv * inv
    out += math.log(xx) - 0.5 * inv - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))
    return out


def _trigamma_approx(x: float) -> float:
    """Lightweight trigamma approximation with recurrence + asymptotic expansion."""
    xx = float(x)
    if xx <= 0.0:
        xx = _POS_FLOOR
    out = 0.0
    while xx < 6.0:
        out += 1.0 / (xx * xx)
        xx += 1.0
    inv = 1.0 / xx
    inv2 = inv * inv
    out += inv + 0.5 * inv2 + inv2 * inv * (1.0 / 6.0 - inv2 * (1.0 / 30.0 - inv2 / 42.0))
    return out


def _chol_solve(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Solve mat * x = vec for SPD mat via Cholesky."""
    chol = np.linalg.cholesky(mat)
    y = np.linalg.solve(chol, vec)
    return np.linalg.solve(chol.T, y)


def _digamma_inv(y: float, tol: float = 1e-8, max_iter: int = 50) -> float:
    """Invert the digamma function via Newton iterations (Minka, 2000)."""
    if y >= -2.22:
        x = math.exp(y) + 0.5
    else:
        x = -1.0 / (y - _digamma_approx(1.0))
    for _ in range(max_iter):
        prev = x
        trig = max(_trigamma_approx(prev), 1e-12)
        x = prev - (_digamma_approx(prev) - y) / trig
        if abs(x - prev) < tol:
            break
    return max(x, tol)


def _normalise_groups(groups: Sequence[Sequence[int]], p: int) -> List[List[int]]:
    normalised: List[List[int]] = []
    covered = np.zeros(p, dtype=bool)
    for block in groups:
        idx = [int(i) for i in block]
        if not idx:
            raise ValueError("Each group must contain at least one feature.")
        if min(idx) < 0 or max(idx) >= p:
            raise ValueError("Group indices must lie in [0, p).")
        normalised.append(idx)
        covered[idx] = True
    if not np.all(covered):
        missing = np.where(~covered)[0].tolist()
        raise ValueError(f"Some features are not assigned to a group: {missing}")
    return normalised


def _groups_from_grp_idx(grp_idx: Sequence[int], p: int) -> List[List[int]]:
    grp = np.asarray(grp_idx, dtype=int).reshape(-1)
    if grp.size != p:
        raise ValueError("grp_idx length must equal number of columns in X.")
    gmin = int(grp.min())
    if gmin == 1:
        grp0 = grp - 1
    elif gmin == 0:
        grp0 = grp
    else:
        raise ValueError("grp_idx must be 0-based or 1-based contiguous labels.")
    unique = np.unique(grp0)
    expected = np.arange(int(unique.size), dtype=int)
    if not np.array_equal(unique, expected):
        raise ValueError("grp_idx labels must be contiguous with no gaps.")
    groups: List[List[int]] = []
    for gid in expected:
        groups.append(np.where(grp0 == gid)[0].astype(int).tolist())
    return _normalise_groups(groups, p)


def _clip_positive_scalar(value: float, *, floor: float = _POS_FLOOR, cap: float = _POS_CAP) -> float:
    val = float(value)
    if not np.isfinite(val):
        return float(floor)
    return float(min(max(val, floor), cap))


def _clip_positive_array(values: np.ndarray, *, floor: float = _POS_FLOOR, cap: float = _POS_CAP) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.where(np.isfinite(arr), arr, floor)
    return np.clip(arr, floor, cap)


def _soft_cap_positive_scalar(value: float, *, floor: float, cap: float) -> float:
    x = float(value)
    if not np.isfinite(x):
        x = float(floor)
    x = max(x, float(floor))
    c = float(max(cap, floor + 1e-12))
    span = c - float(floor)
    # Smoothly saturates below cap without creating a hard boundary pile-up.
    return float(floor + span * (1.0 - math.exp(-(x - floor) / max(span, 1e-12))))


def _soft_cap_positive_array(values: np.ndarray, *, floor: float, cap: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.where(np.isfinite(arr), arr, floor)
    arr = np.maximum(arr, floor)
    c = float(max(cap, floor + 1e-12))
    span = c - float(floor)
    return floor + span * (1.0 - np.exp(-(arr - floor) / max(span, 1e-12)))


def _record_gig_sampler_event(key: str) -> None:
    _GIG_SAMPLER_STATS[key] = int(_GIG_SAMPLER_STATS.get(key, 0)) + 1


def _target_log_scalar_standardized(x: float, lam: float, beta: float) -> float:
    xv = max(float(x), _POS_FLOOR)
    return (lam - 1.0) * math.log(xv) - 0.5 * beta * (xv + 1.0 / xv)


def _gig_mode_standardized(lam: float, beta: float) -> float:
    b = max(float(beta), _POS_FLOOR)
    lam_val = float(lam)
    disc = math.sqrt((lam_val - 1.0) ** 2 + b * b)
    mode = ((lam_val - 1.0) + disc) / b
    return _clip_positive_scalar(mode, floor=_POS_FLOOR, cap=_POS_CAP)


if _NUMBA_AVAILABLE:
    @_njit(cache=True)
    def _first_accept_case1_numba(u_draw: np.ndarray, v_draw: np.ndarray, m: float, lam: float, beta: float) -> tuple[int, float]:
        for i in range(u_draw.shape[0]):
            vv = float(v_draw[i])
            if vv <= 0.0:
                continue
            x = float(u_draw[i] / vv + m)
            if (not np.isfinite(x)) or (x <= 0.0):
                continue
            lhs = 2.0 * math.log(max(vv, _POS_FLOOR))
            rhs = (lam - 1.0) * math.log(x) - 0.5 * beta * (x + 1.0 / x)
            if lhs <= rhs:
                return i, x
        return -1, 0.0


    @_njit(cache=True)
    def _first_accept_case2_numba(u_draw: np.ndarray, v_draw: np.ndarray, lam: float, beta: float) -> tuple[int, float]:
        for i in range(u_draw.shape[0]):
            vv = float(v_draw[i])
            if vv <= 0.0:
                continue
            x = float(u_draw[i] / vv)
            if (not np.isfinite(x)) or (x <= 0.0):
                continue
            lhs = 2.0 * math.log(max(vv, _POS_FLOOR))
            rhs = (lam - 1.0) * math.log(x) - 0.5 * beta * (x + 1.0 / x)
            if lhs <= rhs:
                return i, x
        return -1, 0.0


    @_njit(cache=True)
    def _first_accept_case3_numba(
        u_draw: np.ndarray,
        v_draw: np.ndarray,
        lam: float,
        beta: float,
        x0: float,
        A1: float,
        A2: float,
        k1: float,
        k2: float,
        k3: float,
        x_star: float,
    ) -> tuple[int, float]:
        a12 = A1 + A2
        base_inner = math.exp(-x_star * beta / 2.0)
        for i in range(u_draw.shape[0]):
            u = max(float(u_draw[i]), _POS_FLOOR)
            v = float(v_draw[i])
            if v <= A1:
                x = x0 * v / max(A1, _POS_FLOOR)
                h = k1
            elif v <= a12:
                vv = v - A1
                if lam == 0.0:
                    x = beta * math.exp(vv * math.exp(beta))
                else:
                    x = (x0 ** lam + vv * lam / max(k2, _POS_FLOOR)) ** (1.0 / lam)
                h = k2 * (x ** (lam - 1.0))
            else:
                vv = v - a12
                inner = base_inner - vv * beta / (2.0 * max(k3, _POS_FLOOR))
                if inner <= 0.0:
                    continue
                x = -2.0 / beta * math.log(inner)
                h = k3 * math.exp(-x * beta / 2.0)
            if (not np.isfinite(x)) or (not np.isfinite(h)) or (x <= 0.0) or (h <= 0.0):
                continue
            lhs = math.log(u) + math.log(max(h, _POS_FLOOR))
            rhs = (lam - 1.0) * math.log(x) - 0.5 * beta * (x + 1.0 / x)
            if lhs <= rhs:
                return i, x
        return -1, 0.0
else:
    def _first_accept_case1_numba(u_draw: np.ndarray, v_draw: np.ndarray, m: float, lam: float, beta: float) -> tuple[int, float]:
        return -1, 0.0


    def _first_accept_case2_numba(u_draw: np.ndarray, v_draw: np.ndarray, lam: float, beta: float) -> tuple[int, float]:
        return -1, 0.0


    def _first_accept_case3_numba(
        u_draw: np.ndarray,
        v_draw: np.ndarray,
        lam: float,
        beta: float,
        x0: float,
        A1: float,
        A2: float,
        k1: float,
        k2: float,
        k3: float,
        x_star: float,
    ) -> tuple[int, float]:
        return -1, 0.0


def _first_accept_case1_numpy(u_draw: np.ndarray, v_draw: np.ndarray, *, m: float, lam: float, beta: float) -> tuple[int, float]:
    v_safe = np.asarray(v_draw, dtype=float)
    u_safe = np.asarray(u_draw, dtype=float)
    valid = v_safe > 0.0
    if not np.any(valid):
        return -1, 0.0
    x = np.zeros_like(v_safe, dtype=float)
    x[valid] = u_safe[valid] / v_safe[valid] + float(m)
    valid = valid & np.isfinite(x) & (x > 0.0)
    if not np.any(valid):
        return -1, 0.0
    idx_valid = np.flatnonzero(valid)
    xv = x[valid]
    lhs = 2.0 * np.log(np.maximum(v_safe[valid], _POS_FLOOR))
    rhs = (float(lam) - 1.0) * np.log(xv) - 0.5 * float(beta) * (xv + 1.0 / xv)
    acc = np.flatnonzero(lhs <= rhs)
    if acc.size == 0:
        return -1, 0.0
    pick = int(idx_valid[int(acc[0])])
    return pick, float(x[pick])


def _first_accept_case2_numpy(u_draw: np.ndarray, v_draw: np.ndarray, *, lam: float, beta: float) -> tuple[int, float]:
    v_safe = np.asarray(v_draw, dtype=float)
    u_safe = np.asarray(u_draw, dtype=float)
    valid = v_safe > 0.0
    if not np.any(valid):
        return -1, 0.0
    x = np.zeros_like(v_safe, dtype=float)
    x[valid] = u_safe[valid] / v_safe[valid]
    valid = valid & np.isfinite(x) & (x > 0.0)
    if not np.any(valid):
        return -1, 0.0
    idx_valid = np.flatnonzero(valid)
    xv = x[valid]
    lhs = 2.0 * np.log(np.maximum(v_safe[valid], _POS_FLOOR))
    rhs = (float(lam) - 1.0) * np.log(xv) - 0.5 * float(beta) * (xv + 1.0 / xv)
    acc = np.flatnonzero(lhs <= rhs)
    if acc.size == 0:
        return -1, 0.0
    pick = int(idx_valid[int(acc[0])])
    return pick, float(x[pick])


def _first_accept_case3_numpy(
    u_draw: np.ndarray,
    v_draw: np.ndarray,
    *,
    lam: float,
    beta: float,
    x0: float,
    A1: float,
    A2: float,
    k1: float,
    k2: float,
    k3: float,
    x_star: float,
) -> tuple[int, float]:
    u = np.asarray(u_draw, dtype=float)
    v = np.asarray(v_draw, dtype=float)
    x = np.zeros_like(v, dtype=float)
    h = np.zeros_like(v, dtype=float)

    m1 = v <= A1
    if np.any(m1):
        x[m1] = float(x0) * v[m1] / max(float(A1), _POS_FLOOR)
        h[m1] = float(k1)

    m2 = (~m1) & (v <= (float(A1) + float(A2)))
    if np.any(m2):
        vv = v[m2] - float(A1)
        if float(lam) == 0.0:
            x[m2] = float(beta) * np.exp(vv * math.exp(float(beta)))
        else:
            x[m2] = (float(x0) ** float(lam) + vv * float(lam) / max(float(k2), _POS_FLOOR)) ** (1.0 / float(lam))
        h[m2] = float(k2) * (x[m2] ** (float(lam) - 1.0))

    m3 = (~m1) & (~m2)
    if np.any(m3):
        vv = v[m3] - (float(A1) + float(A2))
        inner = math.exp(-float(x_star) * float(beta) / 2.0) - vv * float(beta) / (2.0 * max(float(k3), _POS_FLOOR))
        ok = inner > 0.0
        if np.any(ok):
            idx3 = np.flatnonzero(m3)
            idx_ok = idx3[ok]
            x[idx_ok] = -2.0 / float(beta) * np.log(inner[ok])
            h[idx_ok] = float(k3) * np.exp(-x[idx_ok] * float(beta) / 2.0)

    valid = np.isfinite(x) & np.isfinite(h) & (x > 0.0) & (h > 0.0)
    if not np.any(valid):
        return -1, 0.0
    idx_valid = np.flatnonzero(valid)
    xv = x[valid]
    hv = h[valid]
    uv = np.maximum(u[valid], _POS_FLOOR)
    lhs = np.log(uv) + np.log(np.maximum(hv, _POS_FLOOR))
    rhs = (float(lam) - 1.0) * np.log(xv) - 0.5 * float(beta) * (xv + 1.0 / xv)
    acc = np.flatnonzero(lhs <= rhs)
    if acc.size == 0:
        return -1, 0.0
    pick = int(idx_valid[int(acc[0])])
    return pick, float(x[pick])


def _try_sample_gig_scipy(*, lam: float, beta: float, alpha: float, rng: Generator) -> Optional[float]:
    if not _SCIPY_GIG_AVAILABLE:
        return None
    try:
        y = float(_scipy_geninvgauss.rvs(float(lam), float(beta), random_state=rng))
    except Exception:
        return None
    if not np.isfinite(y) or y <= 0.0:
        return None
    _record_gig_sampler_event("scipy_success")
    return _clip_positive_scalar(y / max(alpha, _POS_FLOOR), floor=_POS_FLOOR, cap=_POS_CAP)


def _gig_fallback_draw(*, lam: float, beta: float, alpha: float, rng: Generator) -> float:
    mode = _gig_mode_standardized(lam, beta)
    center = math.log(max(mode, _POS_FLOOR))
    y = float(rng.lognormal(mean=center, sigma=_GIG_FALLBACK_LOGNORMAL_SIGMA))
    _record_gig_sampler_event("fallback_used")
    return _clip_positive_scalar(y / max(alpha, _POS_FLOOR), floor=_POS_FLOOR, cap=_POS_CAP)


def _rgig_cpp_scalar(
    chi: float,
    psi: float,
    lambda_param: float,
    rng: Generator,
    *,
    max_iter: int = _GIG_MAX_REJECTION_DRAWS,
    batch_size: int = _GIG_BATCH_SIZE,
) -> float:
    """Sample from standardized GIG via fast backend + batched rejection + safe fallback."""
    chi_pos = max(float(chi), _POS_FLOOR)
    psi_pos = max(float(psi), _POS_FLOOR)
    lam = max(float(lambda_param), _POS_FLOOR)
    alpha = math.sqrt(psi_pos / chi_pos)
    beta = math.sqrt(chi_pos * psi_pos)

    scipy_draw = _try_sample_gig_scipy(lam=lam, beta=beta, alpha=alpha, rng=rng)
    if scipy_draw is not None:
        return scipy_draw

    max_draws = max(1, int(max_iter))
    batch = max(8, int(batch_size))

    if (lam > 1.0) or (beta > 1.0):
        m = (math.sqrt((lam - 1.0) ** 2 + beta**2) + (lam - 1.0)) / beta
        a = -2.0 * (lam + 1.0) / beta - m
        b = 2.0 * (lam - 1.0) * m / beta - 1.0
        c = m
        p = b - a**2 / 3.0
        q = 2.0 * a**3 / 27.0 - a * b / 3.0 + c
        cos_arg = max(min(-(q / 2.0) * math.sqrt(-27.0 / (p**3)), 1.0), -1.0)
        phi = math.acos(cos_arg)
        x_minus = math.sqrt(-(4.0 / 3.0) * p) * math.cos(phi / 3.0 + (4.0 / 3.0) * math.pi) - a / 3.0
        x_plus = math.sqrt(-(4.0 / 3.0) * p) * math.cos(phi / 3.0) - a / 3.0
        v_plus = math.sqrt(math.exp(_target_log_scalar_standardized(m, lam, beta)))
        u_minus = (x_minus - m) * math.sqrt(math.exp(_target_log_scalar_standardized(x_minus, lam, beta)))
        u_plus = (x_plus - m) * math.sqrt(math.exp(_target_log_scalar_standardized(x_plus, lam, beta)))

        draws_used = 0
        while draws_used < max_draws:
            bs = min(batch, max_draws - draws_used)
            u_draw = rng.uniform(u_minus, u_plus, size=bs)
            v_draw = rng.uniform(0.0, v_plus, size=bs)
            if _NUMBA_AVAILABLE:
                idx, x_val = _first_accept_case1_numba(np.asarray(u_draw, dtype=float), np.asarray(v_draw, dtype=float), float(m), float(lam), float(beta))
            else:
                idx, x_val = _first_accept_case1_numpy(np.asarray(u_draw, dtype=float), np.asarray(v_draw, dtype=float), m=float(m), lam=float(lam), beta=float(beta))
            if idx >= 0:
                _record_gig_sampler_event("rejection_success")
                return _clip_positive_scalar(float(x_val) / alpha, floor=_POS_FLOOR, cap=_POS_CAP)
            draws_used += bs

    elif (0.0 <= lam <= 1.0) and (min(0.5, (2.0 / 3.0) * math.sqrt(max(1.0 - lam, 0.0))) <= beta <= 1.0):
        m = beta / ((1.0 - lam) + math.sqrt((1.0 - lam) ** 2 + beta**2))
        x_plus = ((1.0 + lam) + math.sqrt((1.0 + lam) ** 2 + beta**2)) / beta
        v_plus = math.sqrt(math.exp(_target_log_scalar_standardized(m, lam, beta)))
        u_plus = x_plus * math.sqrt(math.exp(_target_log_scalar_standardized(x_plus, lam, beta)))

        draws_used = 0
        while draws_used < max_draws:
            bs = min(batch, max_draws - draws_used)
            u_draw = rng.uniform(0.0, u_plus, size=bs)
            v_draw = rng.uniform(0.0, v_plus, size=bs)
            if _NUMBA_AVAILABLE:
                idx, x_val = _first_accept_case2_numba(np.asarray(u_draw, dtype=float), np.asarray(v_draw, dtype=float), float(lam), float(beta))
            else:
                idx, x_val = _first_accept_case2_numpy(np.asarray(u_draw, dtype=float), np.asarray(v_draw, dtype=float), lam=float(lam), beta=float(beta))
            if idx >= 0:
                _record_gig_sampler_event("rejection_success")
                return _clip_positive_scalar(float(x_val) / alpha, floor=_POS_FLOOR, cap=_POS_CAP)
            draws_used += bs

    elif (0.0 <= lam < 1.0) and (0.0 < beta <= (2.0 / 3.0) * math.sqrt(max(1.0 - lam, 0.0))):
        x0 = beta / max(1.0 - lam, _POS_FLOOR)
        x_star = max(x0, 2.0 / beta)
        m = beta / ((1.0 - lam) + math.sqrt((1.0 - lam) ** 2 + beta**2))
        k1 = math.exp(_target_log_scalar_standardized(m, lam, beta))
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

        draws_used = 0
        while draws_used < max_draws:
            bs = min(batch, max_draws - draws_used)
            u_draw = rng.uniform(0.0, 1.0, size=bs)
            v_draw = rng.uniform(0.0, A, size=bs)
            if _NUMBA_AVAILABLE:
                idx, x_val = _first_accept_case3_numba(
                    np.asarray(u_draw, dtype=float),
                    np.asarray(v_draw, dtype=float),
                    float(lam),
                    float(beta),
                    float(x0),
                    float(A1),
                    float(A2),
                    float(k1),
                    float(k2),
                    float(k3),
                    float(x_star),
                )
            else:
                idx, x_val = _first_accept_case3_numpy(
                    np.asarray(u_draw, dtype=float),
                    np.asarray(v_draw, dtype=float),
                    lam=float(lam),
                    beta=float(beta),
                    x0=float(x0),
                    A1=float(A1),
                    A2=float(A2),
                    k1=float(k1),
                    k2=float(k2),
                    k3=float(k3),
                    x_star=float(x_star),
                )
            if idx >= 0:
                _record_gig_sampler_event("rejection_success")
                return _clip_positive_scalar(float(x_val) / alpha, floor=_POS_FLOOR, cap=_POS_CAP)
            draws_used += bs

    # Fast-fail fallback for extreme parameter regions where rejection is too slow.
    return _gig_fallback_draw(lam=lam, beta=beta, alpha=alpha, rng=rng)


def _sample_gig_scalar(lambda_param: float, chi: float, psi: float, rng: Generator) -> float:
    """Sample one GIG(lambda, chi, psi) draw with a CRAN-style scalar sampler."""
    draw = _rgig_cpp_scalar(chi=max(chi, _POS_FLOOR), psi=max(psi, _POS_FLOOR), lambda_param=max(lambda_param, _POS_FLOOR), rng=rng)
    return _clip_positive_scalar(draw, floor=_POS_FLOOR, cap=_POS_CAP)


def _sample_invgamma_scalar(shape: float, scale: float, rng: Generator, *, floor: float) -> float:
    """Sample scalar inverse-gamma by gamma transform."""
    a = max(float(shape), 1e-12)
    b = max(float(scale), floor)
    # If G ~ Gamma(a, theta=1/b), then 1/G ~ InvGamma(a, scale=b).
    g = rng.gamma(shape=a, scale=1.0 / b)
    return _clip_positive_scalar(1.0 / max(float(g), floor), floor=floor, cap=_POS_CAP)


def _sample_invgamma_vector(shape: np.ndarray | float, scale: np.ndarray, rng: Generator, *, floor: float) -> np.ndarray:
    """Vectorized inverse-gamma sampling by gamma transform."""
    shape_arr = np.asarray(shape, dtype=float)
    scale_arr = np.asarray(scale, dtype=float)
    shape_arr = np.maximum(shape_arr, 1e-12)
    scale_arr = np.maximum(scale_arr, floor)
    g = rng.gamma(shape=shape_arr, scale=1.0 / scale_arr)
    return _clip_positive_array(1.0 / np.maximum(np.asarray(g, dtype=float), floor), floor=floor, cap=_POS_CAP)


def _fit_gigg_chain_task(payload: dict) -> dict:
    if payload.get("X_file"):
        X_arr = np.load(str(payload["X_file"]), allow_pickle=False)
    else:
        X_arr = np.asarray(payload["X"], dtype=float)
    if payload.get("y_file"):
        y_arr = np.load(str(payload["y_file"]), allow_pickle=False)
    else:
        y_arr = np.asarray(payload["y"], dtype=float)
    if payload.get("C_file"):
        C_arr = np.load(str(payload["C_file"]), allow_pickle=False)
    else:
        C_arr = np.asarray(payload["C"], dtype=float) if payload["C"] is not None else None
    if payload.get("alpha_inits_file"):
        alpha_inits_arr = np.load(str(payload["alpha_inits_file"]), allow_pickle=False)
    else:
        alpha_inits_arr = np.asarray(payload["alpha_inits"], dtype=float) if payload["alpha_inits"] is not None else None
    if payload.get("beta_inits_file"):
        beta_inits_arr = np.load(str(payload["beta_inits_file"]), allow_pickle=False)
    else:
        beta_inits_arr = np.asarray(payload["beta_inits"], dtype=float) if payload["beta_inits"] is not None else None
    if payload.get("a_file"):
        a_arr = np.load(str(payload["a_file"]), allow_pickle=False)
    else:
        a_arr = np.asarray(payload["a"], dtype=float) if payload["a"] is not None else None
    if payload.get("b_file"):
        b_arr = np.load(str(payload["b_file"]), allow_pickle=False)
    else:
        b_arr = np.asarray(payload["b"], dtype=float) if payload["b"] is not None else None

    model = GIGGRegression(
        method=str(payload["method"]),
        n_burn_in=int(payload["n_burn_in"]),
        n_samples=int(payload["n_samples"]),
        n_thin=int(payload["n_thin"]),
        jitter=float(payload["jitter"]),
        seed=int(payload["seed"]),
        num_chains=1,
        a_value=payload["a_value"],
        b_init=float(payload["b_init"]),
        b_floor=float(payload["b_floor"]),
        b_max=float(payload["b_max"]),
        tau_sq_init=float(payload["tau_sq_init"]),
        sigma_sq_init=float(payload["sigma_sq_init"]),
        mmle_update=str(payload["mmle_update"]),
        mmle_burnin_only=bool(payload["mmle_burnin_only"]),
        share_group_hyper=bool(payload["share_group_hyper"]),
        mmle_samp_size=int(payload.get("mmle_samp_size", 1000)),
        mmle_tol_scale=float(payload.get("mmle_tol_scale", 1e-4)),
        mmle_max_iters=int(payload.get("mmle_max_iters", 50000)),
        mmle_step_size=float(payload.get("mmle_step_size", 1.0)),
        mmle_update_every=int(payload.get("mmle_update_every", 1)),
        mmle_window=int(payload.get("mmle_window", 1)),
        fit_intercept=bool(payload.get("fit_intercept", True)),
        store_lambda=bool(payload["store_lambda"]),
        btrick=bool(payload["btrick"]),
        stable_solve=bool(payload["stable_solve"]),
        init_strategy=str(payload.get("init_strategy", "ridge")),
        init_ridge=float(payload.get("init_ridge", 1.0)),
        init_scale_blend=float(payload.get("init_scale_blend", 0.5)),
        randomize_group_order=bool(payload.get("randomize_group_order", False)),
        lambda_vectorized_update=bool(payload.get("lambda_vectorized_update", False)),
        extra_beta_refresh_prob=float(payload.get("extra_beta_refresh_prob", 0.0)),
        lambda_constraint_mode=str(payload.get("lambda_constraint_mode", "soft")),
        q_constraint_mode=str(payload.get("q_constraint_mode", "soft")),
        lambda_cap=float(payload.get("lambda_cap", _POS_CAP)),
        lambda_soft_cap=float(payload.get("lambda_soft_cap", payload.get("lambda_cap", _POS_CAP))),
        progress_bar=bool(payload.get("progress_bar", False)),
    )
    fitted = model.fit(
        X_arr,
        y_arr,
        groups=payload["groups"],
        C=C_arr,
        alpha_inits=alpha_inits_arr,
        beta_inits=beta_inits_arr,
        a=a_arr,
        b=b_arr,
        method=str(payload["method"]),
    )
    return {
        "coef_samples": None if fitted.coef_samples_ is None else np.asarray(fitted.coef_samples_),
        "alpha_samples": None if fitted.alpha_samples_ is None else np.asarray(fitted.alpha_samples_),
        "tau_samples": None if fitted.tau_samples_ is None else np.asarray(fitted.tau_samples_),
        "sigma_samples": None if fitted.sigma_samples_ is None else np.asarray(fitted.sigma_samples_),
        "gamma_samples": None if fitted.gamma_samples_ is None else np.asarray(fitted.gamma_samples_),
        "tau2_samples": None if fitted.tau2_samples_ is None else np.asarray(fitted.tau2_samples_),
        "sigma2_samples": None if fitted.sigma2_samples_ is None else np.asarray(fitted.sigma2_samples_),
        "gamma2_samples": None if fitted.gamma2_samples_ is None else np.asarray(fitted.gamma2_samples_),
        "lambda_samples": None if fitted.lambda_samples_ is None else np.asarray(fitted.lambda_samples_),
        "b_samples": None if fitted.b_samples_ is None else np.asarray(fitted.b_samples_),
        "coef_mean": None if fitted.coef_mean_ is None else np.asarray(fitted.coef_mean_),
        "alpha_mean": None if fitted.alpha_mean_ is None else np.asarray(fitted.alpha_mean_),
        "b_mean": None if fitted.b_mean_ is None else np.asarray(fitted.b_mean_),
        "intercept": float(fitted.intercept_),
    }


@dataclass
class GIGGRegression:
    """CRAN-style GIGG Gibbs sampler with backward-compatible fit interface."""

    method: str = "mmle"
    n_burn_in: int = 500
    n_samples: int = 1000
    n_thin: int = 1
    jitter: float = 1e-7
    seed: int = 0
    num_chains: int = 1
    a_value: Optional[float] = None
    a_fixed_default: float = 0.5
    b_init: float = 0.5
    b_floor: float = 1e-12
    b_max: float = 4.0
    tau_sq_init: float = 1.0
    sigma_sq_init: float = 1.0
    tau_scale: Optional[float] = None
    sigma_scale: Optional[float] = None
    mmle_enabled: Optional[bool] = None
    mmle_update: str = "paper_lambda_only"
    mmle_burnin_only: bool = False
    force_a_1_over_n: bool = True
    share_group_hyper: bool = False
    mmle_samp_size: int = 1000
    mmle_tol_scale: float = 1e-4
    mmle_max_iters: int = 50000
    mmle_step_size: float = 1.0
    mmle_update_every: int = 1
    mmle_window: int = 1
    fit_intercept: bool = True
    store_lambda: bool = True
    btrick: bool = False
    stable_solve: bool = True
    init_strategy: str = "zero"  # original R package starts alpha/beta at zero by default
    init_ridge: float = 1.0
    init_scale_blend: float = 0.0
    randomize_group_order: bool = False
    lambda_vectorized_update: bool = False
    extra_beta_refresh_prob: float = 0.0
    lambda_constraint_mode: str = "none"  # original R sampler does not cap lambda_sq
    q_constraint_mode: str = "hard"  # original R MMLE uses an explicit q <= 4 cap
    lambda_cap: float = _POS_CAP
    lambda_soft_cap: float = _POS_CAP
    progress_bar: bool = True

    rng_: Generator = field(init=False, repr=False)
    coef_samples_: Optional[np.ndarray] = field(default=None, init=False)
    alpha_samples_: Optional[np.ndarray] = field(default=None, init=False)
    tau_samples_: Optional[np.ndarray] = field(default=None, init=False)
    sigma_samples_: Optional[np.ndarray] = field(default=None, init=False)
    gamma_samples_: Optional[np.ndarray] = field(default=None, init=False)
    tau2_samples_: Optional[np.ndarray] = field(default=None, init=False)
    sigma2_samples_: Optional[np.ndarray] = field(default=None, init=False)
    gamma2_samples_: Optional[np.ndarray] = field(default=None, init=False)
    lambda_samples_: Optional[np.ndarray] = field(default=None, init=False)
    b_samples_: Optional[np.ndarray] = field(default=None, init=False)
    coef_mean_: Optional[np.ndarray] = field(default=None, init=False)
    alpha_mean_: Optional[np.ndarray] = field(default=None, init=False)
    b_mean_: Optional[np.ndarray] = field(default=None, init=False)
    intercept_: float = field(default=0.0, init=False)
    auto_intercept_: bool = field(default=False, init=False)

    # Backward-compatible aliases used elsewhere in the project.
    iters: Optional[int] = None
    burnin: Optional[int] = None
    thin: Optional[int] = None

    def __post_init__(self) -> None:
        if self.num_chains <= 0:
            raise ValueError("num_chains must be a positive integer.")
        if self.tau_scale is not None:
            self.tau_sq_init = float(self.tau_scale)
        if self.sigma_scale is not None:
            self.sigma_sq_init = float(self.sigma_scale)
        if self.mmle_enabled is not None and not bool(self.mmle_enabled):
            self.method = "fixed"
        self.method = str(self.method).lower()
        if self.method not in {"fixed", "mmle"}:
            raise ValueError("method must be one of {'fixed', 'mmle'}.")
        if self.burnin is not None:
            self.n_burn_in = int(max(0, self.burnin))
        if self.thin is not None:
            self.n_thin = int(max(1, self.thin))
        self.n_burn_in = int(max(0, self.n_burn_in))
        self.n_thin = int(max(1, self.n_thin))
        if self.iters is not None:
            total = int(max(0, self.iters))
            kept = max(0, total - self.n_burn_in)
            self.n_samples = max(1, kept // self.n_thin)
        else:
            self.n_samples = int(max(0, self.n_samples))
        init_mode = str(self.init_strategy).strip().lower()
        if init_mode not in {"ridge", "zero"}:
            raise ValueError("init_strategy must be one of {'ridge','zero'}.")
        self.init_strategy = init_mode
        self.init_ridge = float(max(self.init_ridge, 0.0))
        self.init_scale_blend = float(min(max(self.init_scale_blend, 0.0), 1.0))
        self.mmle_step_size = float(min(max(self.mmle_step_size, 0.0), 1.0))
        self.mmle_update_every = int(max(1, self.mmle_update_every))
        self.mmle_window = int(max(1, self.mmle_window))
        self.extra_beta_refresh_prob = float(min(max(self.extra_beta_refresh_prob, 0.0), 1.0))
        mode = str(self.lambda_constraint_mode).strip().lower()
        if mode not in {"hard", "soft", "none"}:
            raise ValueError("lambda_constraint_mode must be one of {'hard','soft','none'}.")
        self.lambda_constraint_mode = mode
        q_mode = str(self.q_constraint_mode).strip().lower()
        if q_mode not in {"hard", "soft", "none"}:
            raise ValueError("q_constraint_mode must be one of {'hard','soft','none'}.")
        self.q_constraint_mode = q_mode
        self.lambda_cap = float(max(self.lambda_cap, self.jitter * 10.0))
        self.lambda_soft_cap = float(max(self.lambda_soft_cap, self.jitter * 10.0))
        self.burnin = self.n_burn_in
        self.iters = self.n_burn_in + (self.n_samples * self.n_thin)
        self.thin = self.n_thin

    def _stabilize_lambda_scalar(self, value: float) -> float:
        floor = max(self.jitter, _POS_FLOOR)
        if self.lambda_constraint_mode == "none":
            return _clip_positive_scalar(value, floor=floor, cap=max(floor, value if np.isfinite(value) else floor))
        if self.lambda_constraint_mode == "soft":
            return _soft_cap_positive_scalar(value, floor=floor, cap=self.lambda_soft_cap)
        return _clip_positive_scalar(value, floor=floor, cap=self.lambda_cap)

    def _stabilize_lambda_array(self, values: np.ndarray) -> np.ndarray:
        floor = max(self.jitter, _POS_FLOOR)
        if self.lambda_constraint_mode == "none":
            arr = np.asarray(values, dtype=float)
            arr = np.where(np.isfinite(arr), arr, floor)
            return np.maximum(arr, floor)
        if self.lambda_constraint_mode == "soft":
            return _soft_cap_positive_array(values, floor=floor, cap=self.lambda_soft_cap)
        return _clip_positive_array(values, floor=floor, cap=self.lambda_cap)

    def _stabilize_q_scalar(self, value: float) -> float:
        floor = max(self.b_floor, self.jitter)
        cap = max(self.b_max, floor + 1e-12)
        if self.q_constraint_mode == "none":
            val = float(value)
            if not np.isfinite(val):
                return float(floor)
            return float(np.maximum(val, floor))
        if self.q_constraint_mode == "soft":
            return _soft_cap_positive_scalar(value, floor=floor, cap=cap)
        return _clip_positive_scalar(value, floor=floor, cap=cap)

    def _stabilize_q_array(self, values: np.ndarray) -> np.ndarray:
        floor = max(self.b_floor, self.jitter)
        cap = max(self.b_max, floor + 1e-12)
        if self.q_constraint_mode == "none":
            arr = np.asarray(values, dtype=float)
            arr = np.where(np.isfinite(arr), arr, floor)
            return np.maximum(arr, floor)
        if self.q_constraint_mode == "soft":
            return _soft_cap_positive_array(values, floor=floor, cap=cap)
        return _clip_positive_array(values, floor=floor, cap=cap)

    def _ridge_initial_state(
        self,
        *,
        X: np.ndarray,
        y_arr: np.ndarray,
        C_arr: np.ndarray,
        beta_default: np.ndarray,
        alpha_default: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Ridge-based warm start for beta/alpha to reduce burn-in drift."""
        k = int(C_arr.shape[1])
        p = int(X.shape[1])
        if self.init_strategy != "ridge":
            return alpha_default, beta_default
        if k > 0:
            Z = np.hstack([C_arr, X])
        else:
            Z = X
        d = int(Z.shape[1])
        pen = np.full(d, float(self.init_ridge), dtype=float)
        if k > 0:
            # Avoid over-penalizing intercept/control coefficients.
            pen[:k] = 0.0 if self.auto_intercept_ else 0.1 * float(self.init_ridge)
        lhs = Z.T @ Z + np.diag(pen)
        if self.jitter > 0.0:
            lhs = lhs + np.eye(d, dtype=float) * max(self.jitter, 1e-10)
        rhs = Z.T @ y_arr
        try:
            coef = _chol_solve(lhs, rhs)
        except Exception:
            coef = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        coef = np.nan_to_num(np.asarray(coef, dtype=float), nan=0.0, posinf=_BETA_CAP, neginf=-_BETA_CAP)
        coef = np.clip(coef, -_BETA_CAP, _BETA_CAP)
        if k > 0:
            alpha0 = coef[:k].copy()
            beta0 = coef[k : (k + p)].copy()
        else:
            alpha0 = np.zeros(0, dtype=float)
            beta0 = coef[:p].copy()
        return alpha0, beta0

    @staticmethod
    def _flatten_param_draws(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        data = np.asarray(arr, dtype=float)
        if data.ndim == 0:
            return data.reshape(1, 1)
        if data.ndim == 1:
            return data.reshape(1, -1)
        if data.ndim == 2:
            return data
        return data.reshape(-1, *data.shape[2:])

    @staticmethod
    def _stack_chain_draws(arrays: List[Optional[np.ndarray]]) -> Optional[np.ndarray]:
        if not arrays or arrays[0] is None:
            return None
        return np.stack([np.asarray(arr) for arr in arrays], axis=0)

    @staticmethod
    def _resolve_groups(
        p: int,
        groups: Optional[Sequence[Sequence[int]]],
        grp_idx: Optional[Sequence[int]],
    ) -> List[List[int]]:
        if groups is not None:
            return _normalise_groups(groups, p)
        if grp_idx is None:
            raise ValueError("Either groups or grp_idx must be provided for GIGG.")
        return _groups_from_grp_idx(grp_idx, p)

    def _fit_multichain(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        groups: List[List[int]],
        C: Optional[np.ndarray],
        alpha_inits: Optional[np.ndarray],
        beta_inits: Optional[np.ndarray],
        a: Optional[np.ndarray],
        b: Optional[np.ndarray],
        method: str,
    ) -> "GIGGRegression":
        tmpdir_obj: tempfile.TemporaryDirectory[str] | None = None
        data_paths: dict[str, Path] = {}
        process_ok = False
        process_reason = ""
        try:
            if int(self.num_chains) > 1:
                process_ok, process_reason = can_use_process_pool()
                if process_ok:
                    tmpdir_obj = tempfile.TemporaryDirectory(prefix="gigg_shared_")
                    tmpdir = Path(tmpdir_obj.name)
                    x_path = tmpdir / "X.npy"
                    y_path = tmpdir / "y.npy"
                    np.save(x_path, np.asarray(X, dtype=float), allow_pickle=False)
                    np.save(y_path, np.asarray(y, dtype=float), allow_pickle=False)
                    data_paths["X_file"] = x_path
                    data_paths["y_file"] = y_path
                    if C is not None:
                        c_path = tmpdir / "C.npy"
                        np.save(c_path, np.asarray(C, dtype=float), allow_pickle=False)
                        data_paths["C_file"] = c_path
                    if alpha_inits is not None:
                        ai_path = tmpdir / "alpha_inits.npy"
                        np.save(ai_path, np.asarray(alpha_inits, dtype=float), allow_pickle=False)
                        data_paths["alpha_inits_file"] = ai_path
                    if beta_inits is not None:
                        bi_path = tmpdir / "beta_inits.npy"
                        np.save(bi_path, np.asarray(beta_inits, dtype=float), allow_pickle=False)
                        data_paths["beta_inits_file"] = bi_path
                    if a is not None:
                        a_path = tmpdir / "a.npy"
                        np.save(a_path, np.asarray(a, dtype=float), allow_pickle=False)
                        data_paths["a_file"] = a_path
                    if b is not None:
                        b_path = tmpdir / "b.npy"
                        np.save(b_path, np.asarray(b, dtype=float), allow_pickle=False)
                        data_paths["b_file"] = b_path

            payloads: List[dict] = []
            for chain_idx in range(int(self.num_chains)):
                payloads.append(
                    {
                        "method": method,
                        "n_burn_in": self.n_burn_in,
                        "n_samples": self.n_samples,
                        "n_thin": self.n_thin,
                        "jitter": self.jitter,
                        "seed": int(self.seed) + chain_idx,
                        "a_value": self.a_value,
                        "b_init": self.b_init,
                        "b_floor": self.b_floor,
                        "b_max": self.b_max,
                        "tau_sq_init": self.tau_sq_init,
                        "sigma_sq_init": self.sigma_sq_init,
                        "mmle_update": self.mmle_update,
                        "mmle_burnin_only": self.mmle_burnin_only,
                        "share_group_hyper": self.share_group_hyper,
                        "mmle_samp_size": self.mmle_samp_size,
                        "mmle_tol_scale": self.mmle_tol_scale,
                        "mmle_max_iters": self.mmle_max_iters,
                        "mmle_step_size": self.mmle_step_size,
                        "mmle_update_every": self.mmle_update_every,
                        "mmle_window": self.mmle_window,
                        "fit_intercept": self.fit_intercept,
                        "store_lambda": self.store_lambda,
                        "btrick": self.btrick,
                        "stable_solve": self.stable_solve,
                        "init_strategy": self.init_strategy,
                        "init_ridge": self.init_ridge,
                        "init_scale_blend": self.init_scale_blend,
                        "randomize_group_order": self.randomize_group_order,
                        "lambda_vectorized_update": self.lambda_vectorized_update,
                        "extra_beta_refresh_prob": self.extra_beta_refresh_prob,
                        "lambda_constraint_mode": self.lambda_constraint_mode,
                        "q_constraint_mode": self.q_constraint_mode,
                        "lambda_cap": self.lambda_cap,
                        "lambda_soft_cap": self.lambda_soft_cap,
                        "progress_bar": bool(self.progress_bar),
                        "X": None if data_paths else np.asarray(X, dtype=float),
                        "y": None if data_paths else np.asarray(y, dtype=float),
                        "C": None if data_paths else (None if C is None else np.asarray(C, dtype=float)),
                        "groups": [list(g) for g in groups],
                        "alpha_inits": None if data_paths else (None if alpha_inits is None else np.asarray(alpha_inits, dtype=float)),
                        "beta_inits": None if data_paths else (None if beta_inits is None else np.asarray(beta_inits, dtype=float)),
                        "a": None if data_paths else (None if a is None else np.asarray(a, dtype=float)),
                        "b": None if data_paths else (None if b is None else np.asarray(b, dtype=float)),
                        "X_file": str(data_paths["X_file"]) if "X_file" in data_paths else None,
                        "y_file": str(data_paths["y_file"]) if "y_file" in data_paths else None,
                        "C_file": str(data_paths["C_file"]) if "C_file" in data_paths else None,
                        "alpha_inits_file": str(data_paths["alpha_inits_file"]) if "alpha_inits_file" in data_paths else None,
                        "beta_inits_file": str(data_paths["beta_inits_file"]) if "beta_inits_file" in data_paths else None,
                        "a_file": str(data_paths["a_file"]) if "a_file" in data_paths else None,
                        "b_file": str(data_paths["b_file"]) if "b_file" in data_paths else None,
                    }
                )

            if int(self.num_chains) <= 1:
                chain_results = [_fit_gigg_chain_task(payload) for payload in payloads]
            else:
                try:
                    if process_ok:
                        with ProcessPoolExecutor(max_workers=int(self.num_chains)) as executor:
                            fut_map = {executor.submit(_fit_gigg_chain_task, payloads[i]): i for i in range(len(payloads))}
                            chain_results = [None] * len(payloads)
                            fut_iter = as_completed(fut_map)
                            if bool(self.progress_bar):
                                from simulation_project.src.core.utils.logging_utils import progress as _progress

                                fut_iter = _progress(fut_iter, total=len(payloads), desc="GIGG chains")
                            for fut in fut_iter:
                                chain_results[fut_map[fut]] = fut.result()
                    else:
                        if bool(self.progress_bar):
                            print(f"[WARN] GIGG process pool disabled ({process_reason}). Running chains sequentially.")
                            from simulation_project.src.core.utils.logging_utils import progress as _progress

                            chain_results = []
                            for i in _progress(range(len(payloads)), total=len(payloads), desc="GIGG chains [serial]"):
                                chain_results.append(_fit_gigg_chain_task(payloads[int(i)]))
                        else:
                            chain_results = [_fit_gigg_chain_task(payload) for payload in payloads]
                except Exception:
                    if bool(self.progress_bar):
                        from simulation_project.src.core.utils.logging_utils import progress as _progress

                        chain_results = []
                        for i in _progress(range(len(payloads)), total=len(payloads), desc="GIGG chains [fallback]"):
                            chain_results.append(_fit_gigg_chain_task(payloads[int(i)]))
                    else:
                        chain_results = [_fit_gigg_chain_task(payload) for payload in payloads]

            lead = chain_results[0]
            self.rng_ = default_rng(self.seed)
            self.coef_samples_ = self._stack_chain_draws([item["coef_samples"] for item in chain_results])
            self.alpha_samples_ = self._stack_chain_draws([item["alpha_samples"] for item in chain_results])
            self.tau_samples_ = self._stack_chain_draws([item["tau_samples"] for item in chain_results])
            self.sigma_samples_ = self._stack_chain_draws([item["sigma_samples"] for item in chain_results])
            self.gamma_samples_ = self._stack_chain_draws([item["gamma_samples"] for item in chain_results])
            self.tau2_samples_ = self._stack_chain_draws([item["tau2_samples"] for item in chain_results])
            self.sigma2_samples_ = self._stack_chain_draws([item["sigma2_samples"] for item in chain_results])
            self.gamma2_samples_ = self._stack_chain_draws([item["gamma2_samples"] for item in chain_results])
            self.lambda_samples_ = self._stack_chain_draws([item["lambda_samples"] for item in chain_results])
            self.b_samples_ = self._stack_chain_draws([item["b_samples"] for item in chain_results])

            coef_draws = self._flatten_param_draws(self.coef_samples_)
            alpha_draws = self._flatten_param_draws(self.alpha_samples_)
            b_draws = self._flatten_param_draws(self.b_samples_)
            self.coef_mean_ = None if coef_draws is None else coef_draws.mean(axis=0)
            self.alpha_mean_ = None if alpha_draws is None else alpha_draws.mean(axis=0)
            self.b_mean_ = None if b_draws is None else b_draws.mean(axis=0)
            self.intercept_ = float(lead["intercept"])
            return self
        finally:
            if tmpdir_obj is not None:
                tmpdir_obj.cleanup()

    def _draw_beta_standard(
        self,
        *,
        X: np.ndarray,
        y_tilde: np.ndarray,
        sigma_sq: float,
        local_scale: np.ndarray,
        rng: Generator,
    ) -> np.ndarray:
        p = int(X.shape[1])
        prior_prec = 1.0 / local_scale
        precision_beta = (X.T @ X) + np.diag(prior_prec)
        if self.jitter > 0.0:
            precision_beta = precision_beta + np.eye(p) * self.jitter
        mean_beta = _chol_solve(precision_beta, X.T @ y_tilde)
        noise_beta = _chol_solve(precision_beta, rng.normal(size=p))
        beta = mean_beta + math.sqrt(sigma_sq) * noise_beta
        return np.clip(np.nan_to_num(beta, nan=0.0, posinf=_BETA_CAP, neginf=-_BETA_CAP), -_BETA_CAP, _BETA_CAP)

    def _draw_beta_btrick(
        self,
        *,
        X: np.ndarray,
        y_tilde: np.ndarray,
        sigma_sq: float,
        local_scale: np.ndarray,
        rng: Generator,
    ) -> np.ndarray:
        """
        Bhattacharya et al. (2016) fast Gaussian draw for beta block.

        For theta = beta / sigma with prior theta ~ N(0, D), D=diag(local_scale):
          y/sigma = X theta + eps, eps~N(0, I)
        draw theta from posterior using n x n solve, then beta = sigma * theta.
        """
        n = int(X.shape[0])
        sigma = math.sqrt(_clip_positive_scalar(sigma_sq, floor=self.jitter))
        y_star = y_tilde / sigma
        d = _clip_positive_array(local_scale, floor=self.jitter)

        u = rng.normal(size=d.shape[0]) * np.sqrt(d)
        delta = rng.normal(size=n)
        v = X @ u + delta
        x_d = X * d[np.newaxis, :]
        mat = x_d @ X.T + np.eye(n, dtype=float)
        rhs = y_star - v
        if self.jitter > 0.0:
            mat = mat + np.eye(n, dtype=float) * self.jitter
        if self.stable_solve:
            w = _chol_solve(mat, rhs)
        else:
            w = np.linalg.solve(mat, rhs)
        theta = u + d * (X.T @ w)
        beta = sigma * theta
        return np.clip(np.nan_to_num(beta, nan=0.0, posinf=_BETA_CAP, neginf=-_BETA_CAP), -_BETA_CAP, _BETA_CAP)

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        *,
        groups: Optional[Sequence[Sequence[int]]] = None,
        C: Optional[np.ndarray] = None,
        Y: Optional[np.ndarray] = None,
        grp_idx: Optional[Sequence[int]] = None,
        alpha_inits: Optional[np.ndarray] = None,
        beta_inits: Optional[np.ndarray] = None,
        a: Optional[Sequence[float]] = None,
        b: Optional[Sequence[float]] = None,
        method: Optional[str] = None,
    ) -> "GIGGRegression":
        X = np.asarray(X, dtype=float)
        yy = y if y is not None else Y
        if yy is None:
            raise ValueError("Response y (or Y) is required.")
        y_arr = np.asarray(yy, dtype=float).reshape(-1)
        n, p = X.shape
        if y_arr.shape[0] != n:
            raise ValueError("X and y must have compatible shapes.")

        normalised_groups = self._resolve_groups(p, groups, grp_idx)
        group_id = np.empty(p, dtype=int)
        for gid, idxs in enumerate(normalised_groups):
            group_id[idxs] = gid
        group_sizes = np.array([len(g) for g in normalised_groups], dtype=int)
        G = group_sizes.size

        C_arr = np.zeros((n, 0), dtype=float) if C is None else np.asarray(C, dtype=float)
        self.auto_intercept_ = False
        if C is None and bool(self.fit_intercept):
            C_arr = np.ones((n, 1), dtype=float)
            self.auto_intercept_ = True
        if C_arr.ndim != 2 or C_arr.shape[0] != n:
            raise ValueError("C must be a 2D array with n rows.")
        k = int(C_arr.shape[1])

        method_eff = str(self.method if method is None else method).lower()
        if method_eff not in {"fixed", "mmle"}:
            raise ValueError("method must be one of {'fixed', 'mmle'}.")

        a_vec: np.ndarray
        if a is not None:
            a_vec = np.asarray(a, dtype=float).reshape(-1)
            if a_vec.size != G:
                raise ValueError("a must have length equal to number of groups.")
        elif self.a_value is not None:
            a_vec = np.full(G, float(self.a_value), dtype=float)
        elif method_eff == "fixed":
            a_vec = np.full(G, float(self.a_fixed_default), dtype=float)
        else:
            a_vec = np.full(G, 1.0 / max(n, 1), dtype=float)

        if b is not None:
            b_vec = np.asarray(b, dtype=float).reshape(-1)
            if b_vec.size != G:
                raise ValueError("b must have length equal to number of groups.")
        else:
            b_vec = np.full(G, max(self.b_init, self.b_floor), dtype=float)

        if beta_inits is not None:
            beta = np.asarray(beta_inits, dtype=float).reshape(-1)
            if beta.size != p:
                raise ValueError("beta_inits must have length p.")
        else:
            beta = np.zeros(p, dtype=float)

        if alpha_inits is not None:
            alpha = np.asarray(alpha_inits, dtype=float).reshape(-1)
            if alpha.size != k:
                raise ValueError("alpha_inits must have length ncol(C).")
        else:
            alpha = np.zeros(k, dtype=float)

        if beta_inits is None or alpha_inits is None:
            alpha_warm, beta_warm = self._ridge_initial_state(
                X=X,
                y_arr=y_arr,
                C_arr=C_arr,
                beta_default=beta,
                alpha_default=alpha,
            )
            if beta_inits is None:
                beta = beta_warm
            if alpha_inits is None:
                alpha = alpha_warm

        if self.num_chains > 1:
            return self._fit_multichain(
                X,
                y_arr,
                groups=normalised_groups,
                C=C_arr,
                alpha_inits=alpha,
                beta_inits=beta,
                a=a_vec,
                b=b_vec,
                method=method_eff,
            )

        _progress = None
        if bool(self.progress_bar):
            from simulation_project.src.core.utils.logging_utils import progress as _progress

        self.rng_ = default_rng(self.seed)
        rng = self.rng_
        group_arrays = [np.asarray(idxs, dtype=int) for idxs in normalised_groups]
        group_order = np.arange(G, dtype=int)

        lambda_sq = np.ones(p, dtype=float)
        gamma_sq = np.ones(G, dtype=float)
        eta = np.ones(G, dtype=float)
        p_vec = _clip_positive_array(a_vec, floor=self.jitter, cap=_POS_CAP)
        q_vec = self._stabilize_q_array(b_vec)
        tau_sq = _clip_positive_scalar(float(self.tau_sq_init), floor=self.jitter)
        sigma_sq = _clip_positive_scalar(float(self.sigma_sq_init), floor=self.jitter)
        nu = 1.0

        CtC = C_arr.T @ C_arr if k > 0 else None
        tau_shape = 0.5 * (p + 1.0)
        sigma_shape = 0.5 * (n + 1.0)

        kept = self.n_samples
        coef_draws = np.zeros((kept, p), dtype=float)
        alpha_draws = np.zeros((kept, k), dtype=float) if k > 0 else None
        tau2_draws = np.zeros(kept, dtype=float)
        sigma2_draws = np.zeros(kept, dtype=float)
        gamma2_draws = np.zeros((kept, G), dtype=float)
        lambda_draws = np.zeros((kept, p), dtype=float) if self.store_lambda else None
        b_draws = np.zeros((kept, G), dtype=float)
        keep_idx = 0

        def _gibbs_step() -> None:
            nonlocal alpha, beta, lambda_sq, gamma_sq, tau_sq, sigma_sq, nu
            if k > 0:
                resid_alpha = y_arr - X @ beta
                if self.stable_solve:
                    mean_alpha = _chol_solve(CtC, C_arr.T @ resid_alpha)
                    noise_alpha = _chol_solve(CtC, rng.normal(size=k))
                    alpha = mean_alpha + math.sqrt(sigma_sq) * noise_alpha
                else:
                    cov_alpha = np.linalg.inv(CtC)
                    mean_alpha = cov_alpha @ (C_arr.T @ resid_alpha)
                    alpha = mean_alpha + math.sqrt(sigma_sq) * (np.linalg.cholesky(cov_alpha) @ rng.normal(size=k))

            y_tilde = y_arr - (C_arr @ alpha if k > 0 else 0.0)
            local_scale = np.maximum(tau_sq * gamma_sq[group_id] * lambda_sq, self.jitter)
            if self.btrick:
                beta = self._draw_beta_btrick(
                    X=X,
                    y_tilde=y_tilde,
                    sigma_sq=sigma_sq,
                    local_scale=local_scale,
                    rng=rng,
                )
            else:
                beta = self._draw_beta_standard(
                    X=X,
                    y_tilde=y_tilde,
                    sigma_sq=sigma_sq,
                    local_scale=local_scale,
                    rng=rng,
                )

            gl_param_expand_diag_inv = np.maximum(1.0 / np.maximum(local_scale, self.jitter), self.jitter)
            tau_rate = _clip_positive_scalar(
                0.5 * tau_sq * float(np.sum((beta**2) * gl_param_expand_diag_inv))
                + 1.0 / _clip_positive_scalar(nu, floor=self.jitter),
                floor=self.jitter,
            )
            try:
                tau_sq = _sample_invgamma_scalar(tau_shape, tau_rate, rng, floor=self.jitter)
            except Exception:
                tau_sq = tau_rate / max(tau_shape + 1.0, 1.0)
            tau_sq = _clip_positive_scalar(tau_sq, floor=self.jitter)

            resid = y_arr - X @ beta - (C_arr @ alpha if k > 0 else 0.0)
            rss = float(resid @ resid)
            scale_sigma = _clip_positive_scalar(0.5 * rss + 1.0 / _clip_positive_scalar(nu, floor=self.jitter))
            try:
                sigma_sq = _sample_invgamma_scalar(sigma_shape, scale_sigma, rng, floor=self.jitter)
            except Exception:
                sigma_sq = scale_sigma / max(sigma_shape + 1.0, 1.0)
            sigma_sq = _clip_positive_scalar(sigma_sq, floor=self.jitter)

            gid_iter = group_order

            for gid in gid_iter:
                idxs = group_arrays[int(gid)]
                psi = _clip_positive_scalar(
                    float(np.sum((beta[idxs] ** 2) / np.maximum(lambda_sq[idxs], self.jitter)))
                    / _clip_positive_scalar(tau_sq, floor=self.jitter),
                    floor=self.jitter,
                )
                group_half = 0.5 * float(group_sizes[gid])
                shape_gap = float(abs(float(p_vec[gid]) - group_half))
                if shape_gap < self.jitter:
                    shape_gap = self.jitter
                if group_half < float(p_vec[gid]):
                    gamma_draw = _sample_gig_scalar(
                        lambda_param=shape_gap,
                        chi=psi,
                        psi=max(2.0 * float(eta[gid]), self.jitter),
                        rng=rng,
                    )
                else:
                    inv_draw = _sample_gig_scalar(
                        lambda_param=shape_gap,
                        chi=max(2.0 * float(eta[gid]), self.jitter),
                        psi=psi,
                        rng=rng,
                    )
                    gamma_draw = 1.0 / _clip_positive_scalar(float(inv_draw), floor=self.jitter)
                gamma_sq[gid] = max(float(gamma_draw), self.jitter)

                lam_shape = max(float(q_vec[gid]) + 0.5, 1e-6)
                lam_denom = _clip_positive_scalar(2.0 * tau_sq * gamma_sq[gid], floor=self.jitter)
                lam_scale_vec = np.maximum(float(eta[gid]) + (beta[idxs] ** 2) / lam_denom, self.jitter)
                if self.lambda_vectorized_update:
                    try:
                        draw_vec = _sample_invgamma_vector(lam_shape, lam_scale_vec, rng, floor=self.jitter)
                    except Exception:
                        draw_vec = lambda_sq[idxs]
                    lambda_sq[idxs] = np.maximum(np.asarray(draw_vec, dtype=float), self.jitter)
                else:
                    for j in idxs:
                        lam_scale = _clip_positive_scalar(float(eta[gid]) + float(beta[j] ** 2) / lam_denom, floor=self.jitter)
                        try:
                            draw = _sample_invgamma_scalar(lam_shape, lam_scale, rng, floor=self.jitter)
                        except Exception:
                            draw = lambda_sq[j]
                        lambda_sq[j] = max(float(draw), self.jitter)

            eta.fill(1.0)
            nu_scale = _clip_positive_scalar(1.0 / _clip_positive_scalar(tau_sq, floor=self.jitter) + 1.0 / _clip_positive_scalar(sigma_sq, floor=self.jitter), floor=self.jitter)
            try:
                nu = _sample_invgamma_scalar(1.0, nu_scale, rng, floor=self.jitter)
            except Exception:
                nu = nu_scale / 2.0
            nu = _clip_positive_scalar(nu, floor=self.jitter)

            lambda_sq = np.maximum(lambda_sq, self.jitter)
            gamma_sq = np.maximum(gamma_sq, self.jitter)
            p_vec[:] = _clip_positive_array(p_vec, floor=self.jitter, cap=_POS_CAP)
            q_vec[:] = self._stabilize_q_array(q_vec)

        burnin_iter = range(self.n_burn_in)
        if _progress is not None:
            burnin_iter = _progress(burnin_iter, total=int(self.n_burn_in), desc="GIGG burn-in")
        for burnin_idx in burnin_iter:
            _gibbs_step()

        if method_eff == "mmle":
            mmle_samp_size = int(max(1, self.mmle_samp_size))
            terminate_mmle = float(self.mmle_tol_scale) * float(G)
            delta_mmle = float("inf")
            mmle_cnt = 0
            lambda_mmle_store = np.zeros((mmle_samp_size, p), dtype=float)
            eta_mmle_store = np.ones((mmle_samp_size, G), dtype=float)
            max_mmle_iters = max(1, int(self.mmle_max_iters))
            mmle_iter = range(max_mmle_iters)
            if _progress is not None:
                mmle_iter = _progress(mmle_iter, total=int(max_mmle_iters), desc="GIGG MMLE")
            for _ in mmle_iter:
                _gibbs_step()
                lambda_mmle_store[mmle_cnt % mmle_samp_size] = lambda_sq
                eta_mmle_store[mmle_cnt % mmle_samp_size] = eta
                mmle_cnt += 1
                if mmle_cnt % mmle_samp_size == 0:
                    q_new = q_vec.copy()
                    p_new = np.full(G, 1.0 / max(float(n), 1.0), dtype=float)
                    for gid, idxs in enumerate(group_arrays):
                        overflow_check = float(np.mean(np.log(np.maximum(eta_mmle_store[:, gid], self.jitter))))
                        overflow_check -= float(np.mean(np.log(np.maximum(lambda_mmle_store[:, idxs], self.jitter))))
                        try:
                            est = _digamma_inv(overflow_check)
                        except Exception:
                            est = float(q_vec[gid])
                        q_new[gid] = min(max(float(est), self.b_floor), self.b_max)
                    delta_mmle = float(np.sum((q_new - q_vec) ** 2 + (p_new - p_vec) ** 2))
                    q_vec[:] = self._stabilize_q_array(q_new)
                    p_vec[:] = p_new
                    if delta_mmle < terminate_mmle:
                        break

        sample_iters = kept * self.n_thin
        sample_iter = range(sample_iters)
        if _progress is not None:
            sample_iter = _progress(sample_iter, total=int(sample_iters), desc="GIGG sample")
        for it in sample_iter:
            _gibbs_step()
            if it % self.n_thin == 0:
                coef_draws[keep_idx] = beta
                if alpha_draws is not None:
                    alpha_draws[keep_idx] = alpha
                tau2_draws[keep_idx] = tau_sq
                sigma2_draws[keep_idx] = sigma_sq
                gamma2_draws[keep_idx] = gamma_sq
                if lambda_draws is not None:
                    lambda_draws[keep_idx] = lambda_sq
                b_draws[keep_idx] = q_vec
                keep_idx += 1

        self.coef_samples_ = coef_draws if kept else None
        self.alpha_samples_ = alpha_draws
        self.tau2_samples_ = tau2_draws if kept else None
        self.sigma2_samples_ = sigma2_draws if kept else None
        self.gamma2_samples_ = gamma2_draws if kept else None
        self.tau_samples_ = None if self.tau2_samples_ is None else np.sqrt(np.maximum(self.tau2_samples_, self.jitter))
        self.sigma_samples_ = None if self.sigma2_samples_ is None else np.sqrt(np.maximum(self.sigma2_samples_, self.jitter))
        self.gamma_samples_ = None if self.gamma2_samples_ is None else np.sqrt(np.maximum(self.gamma2_samples_, self.jitter))
        self.lambda_samples_ = lambda_draws if lambda_draws is not None else None
        self.b_samples_ = b_draws if kept else None
        self.coef_mean_ = coef_draws.mean(axis=0) if kept else beta
        self.alpha_mean_ = None if alpha_draws is None else alpha_draws.mean(axis=0)
        self.b_mean_ = b_draws.mean(axis=0) if kept else q_vec.copy()
        if self.auto_intercept_ and self.alpha_mean_ is not None and self.alpha_mean_.size > 0:
            self.intercept_ = float(self.alpha_mean_[0])
        else:
            self.intercept_ = 0.0
        return self

    def predict(self, X: np.ndarray, C: Optional[np.ndarray] = None) -> np.ndarray:
        if self.coef_mean_ is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        X_arr = np.asarray(X, dtype=float)
        y_hat = X_arr @ self.coef_mean_
        if C is not None and self.alpha_mean_ is not None and self.alpha_mean_.size > 0:
            C_arr = np.asarray(C, dtype=float)
            if C_arr.ndim != 2 or C_arr.shape[0] != X_arr.shape[0]:
                raise ValueError("C must be a 2D array with same row count as X.")
            y_hat = y_hat + C_arr @ self.alpha_mean_
        return y_hat + self.intercept_
