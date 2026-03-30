from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from numpy.random import Generator, default_rng
# NOTE:
# We avoid scipy.stats samplers here because repeated scalar rvs calls can be
# prohibitively slow in some runtime environments.

_POS_FLOOR = 1e-8
_POS_CAP = 1e3
_BETA_CAP = 1e3


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
    gmax = int(grp.max())
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


def _rgig_cpp_scalar(chi: float, psi: float, lambda_param: float, rng: Generator, *, max_iter: int = 200000) -> float:
    """Python port of CRAN gigg::rgig_cpp scalar sampler."""
    chi_pos = max(float(chi), _POS_FLOOR)
    psi_pos = max(float(psi), _POS_FLOOR)
    lam = float(lambda_param)
    alpha = math.sqrt(psi_pos / chi_pos)
    beta = math.sqrt(chi_pos * psi_pos)

    def _target_log(x: float) -> float:
        return (lam - 1.0) * math.log(x) - 0.5 * beta * (x + 1.0 / x)

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
        v_plus = math.sqrt(math.exp(_target_log(m)))
        u_minus = (x_minus - m) * math.sqrt(math.exp(_target_log(x_minus)))
        u_plus = (x_plus - m) * math.sqrt(math.exp(_target_log(x_plus)))
        for _ in range(max_iter):
            u_draw = rng.uniform(u_minus, u_plus)
            v_draw = rng.uniform(0.0, v_plus)
            if v_draw <= 0.0:
                continue
            x_draw = u_draw / v_draw + m
            if x_draw <= 0.0:
                continue
            if (v_draw**2) <= math.exp(_target_log(x_draw)):
                return _clip_positive_scalar(x_draw / alpha, floor=_POS_FLOOR, cap=_POS_CAP)
    elif (0.0 <= lam <= 1.0) and (min(0.5, (2.0 / 3.0) * math.sqrt(max(1.0 - lam, 0.0))) <= beta <= 1.0):
        m = beta / ((1.0 - lam) + math.sqrt((1.0 - lam) ** 2 + beta**2))
        x_plus = ((1.0 + lam) + math.sqrt((1.0 + lam) ** 2 + beta**2)) / beta
        v_plus = math.sqrt(math.exp(_target_log(m)))
        u_plus = x_plus * math.sqrt(math.exp(_target_log(x_plus)))
        for _ in range(max_iter):
            u_draw = rng.uniform(0.0, u_plus)
            v_draw = rng.uniform(0.0, v_plus)
            if v_draw <= 0.0:
                continue
            x_draw = u_draw / v_draw
            if x_draw <= 0.0:
                continue
            if (v_draw**2) <= math.exp(_target_log(x_draw)):
                return _clip_positive_scalar(x_draw / alpha, floor=_POS_FLOOR, cap=_POS_CAP)
    elif (0.0 <= lam < 1.0) and (0.0 < beta <= (2.0 / 3.0) * math.sqrt(max(1.0 - lam, 0.0))):
        m = beta / ((1.0 - lam) + math.sqrt((1.0 - lam) ** 2 + beta**2))
        x0 = beta / max(1.0 - lam, _POS_FLOOR)
        x_star = max(x0, 2.0 / beta)
        k1 = math.exp(_target_log(m))
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
        for _ in range(max_iter):
            u_draw = rng.uniform(0.0, 1.0)
            v_draw = rng.uniform(0.0, A)
            if v_draw <= A1:
                x_draw = x0 * v_draw / max(A1, _POS_FLOOR)
                h = k1
            elif v_draw <= A1 + A2:
                vv = v_draw - A1
                if lam == 0.0:
                    x_draw = beta * math.exp(vv * math.exp(beta))
                else:
                    x_draw = (x0**lam + vv * lam / max(k2, _POS_FLOOR)) ** (1.0 / lam)
                h = k2 * x_draw ** (lam - 1.0)
            else:
                vv = v_draw - (A1 + A2)
                inner = math.exp(-x_star * beta / 2.0) - vv * beta / (2.0 * max(k3, _POS_FLOOR))
                if inner <= 0.0:
                    continue
                x_draw = -2.0 / beta * math.log(inner)
                h = k3 * math.exp(-x_draw * beta / 2.0)
            if x_draw <= 0.0:
                continue
            if u_draw * h <= math.exp(_target_log(x_draw)):
                return _clip_positive_scalar(x_draw / alpha, floor=_POS_FLOOR, cap=_POS_CAP)

    # Fallback if sampler did not accept.
    return _POS_FLOOR


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


def _fit_gigg_chain_task(payload: dict) -> dict:
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
        fit_intercept=bool(payload.get("fit_intercept", True)),
        store_lambda=bool(payload["store_lambda"]),
        btrick=bool(payload["btrick"]),
        stable_solve=bool(payload["stable_solve"]),
        lambda_constraint_mode=str(payload.get("lambda_constraint_mode", "hard")),
        lambda_cap=float(payload.get("lambda_cap", _POS_CAP)),
        lambda_soft_cap=float(payload.get("lambda_soft_cap", payload.get("lambda_cap", _POS_CAP))),
    )
    fitted = model.fit(
        np.asarray(payload["X"], dtype=float),
        np.asarray(payload["y"], dtype=float),
        groups=payload["groups"],
        C=np.asarray(payload["C"], dtype=float) if payload["C"] is not None else None,
        alpha_inits=np.asarray(payload["alpha_inits"], dtype=float) if payload["alpha_inits"] is not None else None,
        beta_inits=np.asarray(payload["beta_inits"], dtype=float) if payload["beta_inits"] is not None else None,
        a=np.asarray(payload["a"], dtype=float) if payload["a"] is not None else None,
        b=np.asarray(payload["b"], dtype=float) if payload["b"] is not None else None,
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
    jitter: float = 1e-8
    seed: int = 0
    num_chains: int = 1
    a_value: Optional[float] = None
    a_fixed_default: float = 0.5
    b_init: float = 0.5
    b_floor: float = 1e-3
    b_max: float = 4.0
    tau_sq_init: float = 1.0
    sigma_sq_init: float = 1.0
    tau_scale: Optional[float] = None
    sigma_scale: Optional[float] = None
    mmle_enabled: Optional[bool] = None
    mmle_update: str = "paper_lambda_only"
    mmle_burnin_only: bool = True
    force_a_1_over_n: bool = True
    share_group_hyper: bool = False
    mmle_samp_size: int = 1000
    mmle_tol_scale: float = 1e-4
    mmle_max_iters: int = 50000
    fit_intercept: bool = True
    store_lambda: bool = True
    btrick: bool = False
    stable_solve: bool = True
    lambda_constraint_mode: str = "hard"  # one of {"hard", "soft", "none"}
    lambda_cap: float = _POS_CAP
    lambda_soft_cap: float = _POS_CAP

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
        mode = str(self.lambda_constraint_mode).strip().lower()
        if mode not in {"hard", "soft", "none"}:
            raise ValueError("lambda_constraint_mode must be one of {'hard','soft','none'}.")
        self.lambda_constraint_mode = mode
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
                    "fit_intercept": self.fit_intercept,
                    "store_lambda": self.store_lambda,
                    "btrick": self.btrick,
                    "stable_solve": self.stable_solve,
                    "lambda_constraint_mode": self.lambda_constraint_mode,
                    "lambda_cap": self.lambda_cap,
                    "lambda_soft_cap": self.lambda_soft_cap,
                    "X": np.asarray(X, dtype=float),
                    "y": np.asarray(y, dtype=float),
                    "C": None if C is None else np.asarray(C, dtype=float),
                    "groups": [list(g) for g in groups],
                    "alpha_inits": None if alpha_inits is None else np.asarray(alpha_inits, dtype=float),
                    "beta_inits": None if beta_inits is None else np.asarray(beta_inits, dtype=float),
                    "a": None if a is None else np.asarray(a, dtype=float),
                    "b": None if b is None else np.asarray(b, dtype=float),
                }
            )

        try:
            with ProcessPoolExecutor(max_workers=int(self.num_chains)) as executor:
                chain_results = list(executor.map(_fit_gigg_chain_task, payloads))
        except Exception:
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

        self.rng_ = default_rng(self.seed)
        rng = self.rng_

        lambda_sq = np.ones(p, dtype=float)
        gamma_sq = np.ones(G, dtype=float)
        eta = np.ones(G, dtype=float)
        p_vec = _clip_positive_array(a_vec, floor=self.jitter, cap=_POS_CAP)
        q_vec = _clip_positive_array(b_vec, floor=self.b_floor, cap=self.b_max)
        tau_sq = _clip_positive_scalar(self.tau_sq_init)
        sigma_sq = _clip_positive_scalar(self.sigma_sq_init)
        nu = 1.0

        CtC = C_arr.T @ C_arr if k > 0 else None
        group_arrays = [np.asarray(idxs, dtype=int) for idxs in normalised_groups]
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
                    precision_alpha = CtC + np.eye(k, dtype=float) * max(self.jitter, 0.0)
                    mean_alpha = _chol_solve(precision_alpha, C_arr.T @ resid_alpha)
                    noise_alpha = _chol_solve(precision_alpha, rng.normal(size=k))
                    alpha = mean_alpha + math.sqrt(sigma_sq) * noise_alpha
                else:
                    precision_alpha = CtC + np.eye(k, dtype=float) * max(self.jitter, 0.0)
                    cov_alpha = np.linalg.pinv(precision_alpha)
                    mean_alpha = cov_alpha @ (C_arr.T @ resid_alpha)
                    alpha = mean_alpha + math.sqrt(sigma_sq) * (np.linalg.cholesky(cov_alpha) @ rng.normal(size=k))

            y_tilde = y_arr - (C_arr @ alpha if k > 0 else 0.0)
            local_scale = _clip_positive_array(tau_sq * gamma_sq[group_id] * lambda_sq, floor=self.jitter)
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

            gl_param_expand_diag_inv = _clip_positive_array(1.0 / local_scale, floor=self.jitter)
            tau_rate = _clip_positive_scalar(
                0.5 * tau_sq * float(np.sum(np.minimum(beta**2, _POS_CAP) * gl_param_expand_diag_inv))
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

            for gid, idxs in enumerate(group_arrays):
                psi = _clip_positive_scalar(
                    float(np.sum(np.minimum(beta[idxs] ** 2, _POS_CAP) / _clip_positive_array(lambda_sq[idxs], floor=self.jitter)))
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
                gamma_sq[gid] = _clip_positive_scalar(float(gamma_draw), floor=self.jitter)

                lam_shape = max(float(q_vec[gid]) + 0.5, 1e-6)
                lam_denom = _clip_positive_scalar(2.0 * tau_sq * gamma_sq[gid], floor=self.jitter)
                for j in idxs:
                    lam_scale = _clip_positive_scalar(float(eta[gid]) + min(float(beta[j] ** 2), _POS_CAP) / lam_denom, floor=self.jitter)
                    try:
                        draw = _sample_invgamma_scalar(lam_shape, lam_scale, rng, floor=self.jitter)
                    except Exception:
                        draw = lambda_sq[j]
                    lambda_sq[j] = self._stabilize_lambda_scalar(float(draw))

            eta.fill(1.0)
            nu_scale = _clip_positive_scalar(1.0 / _clip_positive_scalar(tau_sq, floor=self.jitter) + 1.0 / _clip_positive_scalar(sigma_sq, floor=self.jitter), floor=self.jitter)
            try:
                nu = _sample_invgamma_scalar(1.0, nu_scale, rng, floor=self.jitter)
            except Exception:
                nu = nu_scale / 2.0
            nu = _clip_positive_scalar(nu, floor=self.jitter)

            lambda_sq = self._stabilize_lambda_array(lambda_sq)
            gamma_sq = _clip_positive_array(gamma_sq, floor=self.jitter)
            p_vec[:] = _clip_positive_array(p_vec, floor=self.jitter, cap=_POS_CAP)
            q_vec[:] = _clip_positive_array(q_vec, floor=self.b_floor, cap=self.b_max)

        for _ in range(self.n_burn_in):
            _gibbs_step()
            if method_eff == "mmle" and self.mmle_burnin_only:
                if self.share_group_hyper:
                    mean_targets = []
                    for idxs in group_arrays:
                        mean_targets.append(-float(np.mean(np.log(np.maximum(lambda_sq[idxs], self.jitter)))))
                    try:
                        shared_q = _digamma_inv(float(np.mean(mean_targets)))
                    except Exception:
                        shared_q = float(np.mean(q_vec))
                    q_vec[:] = min(max(float(shared_q), self.b_floor), self.b_max)
                else:
                    for gid, idxs in enumerate(group_arrays):
                        target = -float(np.mean(np.log(np.maximum(lambda_sq[idxs], self.jitter))))
                        try:
                            q_est = _digamma_inv(target)
                        except Exception:
                            q_est = float(q_vec[gid])
                        q_vec[gid] = min(max(float(q_est), self.b_floor), self.b_max)
                if self.force_a_1_over_n:
                    p_vec[:] = _clip_positive_array(
                        np.full(G, 1.0 / max(float(n), 1.0), dtype=float),
                        floor=self.jitter,
                        cap=_POS_CAP,
                    )
                else:
                    p_vec[:] = _clip_positive_array(a_vec, floor=self.jitter, cap=_POS_CAP)
                q_vec[:] = _clip_positive_array(q_vec, floor=self.b_floor, cap=self.b_max)

        if method_eff == "mmle" and (not self.mmle_burnin_only):
            mmle_samp_size = max(20, int(self.mmle_samp_size))
            terminate_mmle = float(self.mmle_tol_scale) * float(G)
            delta_mmle = float("inf")
            mmle_cnt = 0
            lambda_mmle_store = np.zeros((mmle_samp_size, p), dtype=float)
            max_mmle_iters = max(1, int(self.mmle_max_iters))
            while delta_mmle >= terminate_mmle and mmle_cnt < max_mmle_iters:
                _gibbs_step()
                lambda_mmle_store[mmle_cnt % mmle_samp_size] = lambda_sq
                mmle_cnt += 1
                if mmle_cnt % mmle_samp_size == 0:
                    q_new = q_vec.copy()
                    if self.force_a_1_over_n:
                        p_new = np.full(G, 1.0 / max(float(n), 1.0), dtype=float)
                    else:
                        p_new = a_vec.copy()
                    if self.share_group_hyper:
                        mean_targets = []
                        for idxs in group_arrays:
                            mean_targets.append(-float(np.mean(np.log(np.maximum(lambda_mmle_store[:, idxs], self.jitter)))))
                        try:
                            shared_q = _digamma_inv(float(np.mean(mean_targets)))
                        except Exception:
                            shared_q = float(np.mean(q_vec))
                        q_new[:] = min(max(float(shared_q), self.b_floor), self.b_max)
                    else:
                        for gid, idxs in enumerate(group_arrays):
                            target = -float(np.mean(np.log(np.maximum(lambda_mmle_store[:, idxs], self.jitter))))
                            try:
                                est = _digamma_inv(target)
                            except Exception:
                                est = float(q_vec[gid])
                            q_new[gid] = min(max(float(est), self.b_floor), self.b_max)
                    delta_mmle = float(np.sum((q_new - q_vec) ** 2 + (p_new - p_vec) ** 2))
                    q_vec[:] = _clip_positive_array(q_new, floor=self.b_floor, cap=self.b_max)
                    p_vec[:] = _clip_positive_array(p_new, floor=self.jitter, cap=_POS_CAP)

        sample_iters = kept * self.n_thin
        for it in range(sample_iters):
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
        expose_b = not (method_eff == "mmle" and (not self.mmle_burnin_only))
        self.b_samples_ = b_draws if (kept and expose_b) else None
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
