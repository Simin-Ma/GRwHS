from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from numpy.random import Generator, default_rng
from scipy.linalg import cho_factor, cho_solve
from scipy.special import digamma, polygamma
from scipy.stats import invgamma

from grrhs.inference.gig import sample_gig

_POS_FLOOR = 1e-8
_POS_CAP = 1e3
_BETA_CAP = 1e3


def _digamma_inv(y: float, tol: float = 1e-8, max_iter: int = 50) -> float:
    """Invert the digamma function via Newton iterations (Minka, 2000)."""
    if y >= -2.22:
        x = math.exp(y) + 0.5
    else:
        x = -1.0 / (y - digamma(1.0))
    for _ in range(max_iter):
        prev = x
        x = prev - (digamma(prev) - y) / polygamma(1, prev)
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
        store_lambda=bool(payload["store_lambda"]),
        btrick=bool(payload["btrick"]),
        stable_solve=bool(payload["stable_solve"]),
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
    share_group_hyper: bool = False
    store_lambda: bool = True
    btrick: bool = False
    stable_solve: bool = True

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
        self.burnin = self.n_burn_in
        self.iters = self.n_burn_in + (self.n_samples * self.n_thin)
        self.thin = self.n_thin

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
                    "store_lambda": self.store_lambda,
                    "btrick": self.btrick,
                    "stable_solve": self.stable_solve,
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
        tau_sq = _clip_positive_scalar(self.tau_sq_init)
        sigma_sq = _clip_positive_scalar(self.sigma_sq_init)
        xi_tau = 1.0
        xi_sigma = 1.0

        XtX = X.T @ X
        CtC = C_arr.T @ C_arr if k > 0 else None

        kept = self.n_samples
        coef_draws = np.zeros((kept, p), dtype=float)
        alpha_draws = np.zeros((kept, k), dtype=float) if k > 0 else None
        tau2_draws = np.zeros(kept, dtype=float)
        sigma2_draws = np.zeros(kept, dtype=float)
        gamma2_draws = np.zeros((kept, G), dtype=float)
        lambda_draws = np.zeros((kept, p), dtype=float) if self.store_lambda else None
        b_draws = np.zeros((kept, G), dtype=float)

        log_lambda_mean = np.zeros(G, dtype=float)
        keep_idx = 0
        total_iters = self.iters

        for it in range(total_iters):
            if k > 0:
                resid_alpha = y_arr - X @ beta
                if self.stable_solve:
                    precision_alpha = CtC + np.eye(k, dtype=float) * max(self.jitter, 0.0)
                    chol_alpha = cho_factor(precision_alpha, lower=True, check_finite=False)
                    mean_alpha = cho_solve(chol_alpha, C_arr.T @ resid_alpha, check_finite=False)
                    noise_alpha = cho_solve(chol_alpha, rng.normal(size=k), check_finite=False)
                    alpha = mean_alpha + math.sqrt(sigma_sq) * noise_alpha
                else:
                    precision_alpha = CtC + np.eye(k, dtype=float) * max(self.jitter, 0.0)
                    cov_alpha = np.linalg.pinv(precision_alpha)
                    mean_alpha = cov_alpha @ (C_arr.T @ resid_alpha)
                    alpha = mean_alpha + math.sqrt(sigma_sq) * (np.linalg.cholesky(cov_alpha) @ rng.normal(size=k))

            y_tilde = y_arr - (C_arr @ alpha if k > 0 else 0.0)
            local_scale = _clip_positive_array(tau_sq * gamma_sq[group_id] * lambda_sq, floor=self.jitter)
            prior_prec = 1.0 / local_scale
            precision_beta = XtX + np.diag(prior_prec)
            if self.jitter > 0.0:
                precision_beta = precision_beta + np.eye(p) * self.jitter
            chol_beta = cho_factor(precision_beta, lower=True, check_finite=False)
            mean_beta = cho_solve(chol_beta, X.T @ y_tilde, check_finite=False)
            noise_beta = cho_solve(chol_beta, rng.normal(size=p), check_finite=False)
            beta = mean_beta + math.sqrt(sigma_sq) * noise_beta
            beta = np.clip(np.nan_to_num(beta, nan=0.0, posinf=_BETA_CAP, neginf=-_BETA_CAP), -_BETA_CAP, _BETA_CAP)

            for gid, idxs in enumerate(normalised_groups):
                denom = _clip_positive_scalar(2.0 * tau_sq * gamma_sq[gid], floor=self.jitter)
                b_shape = max(b_vec[gid] + 0.5, 1e-6)
                for j in idxs:
                    numer = min(float(beta[j] ** 2), _POS_CAP)
                    scale = _clip_positive_scalar(1.0 + numer / denom)
                    try:
                        draw = invgamma.rvs(a=b_shape, scale=scale, random_state=rng)
                    except Exception:
                        draw = lambda_sq[j]
                    lambda_sq[j] = _clip_positive_scalar(draw)

            for gid, idxs in enumerate(normalised_groups):
                lam_param = float(a_vec[gid] - 0.5 * group_sizes[gid])
                denom = _clip_positive_array(lambda_sq[idxs], floor=self.jitter)
                chi = _clip_positive_scalar(np.sum(np.minimum(beta[idxs] ** 2, _POS_CAP) / denom), floor=self.jitter)
                psi = 2.0
                try:
                    theta = sample_gig(
                        lambda_param=lam_param,
                        chi=chi,
                        psi=max(psi, self.jitter),
                        size=1,
                        rng=rng,
                    )[0]
                except Exception:
                    theta = gamma_sq[gid]
                gamma_sq[gid] = _clip_positive_scalar(theta, floor=self.jitter)

            denom_tau = _clip_positive_array(gamma_sq[group_id] * lambda_sq, floor=self.jitter)
            beta_quad = float(np.sum(np.minimum(beta**2, _POS_CAP) / denom_tau))
            shape_tau = 0.5 * (p + 1)
            scale_tau = _clip_positive_scalar(0.5 * beta_quad + 1.0 / _clip_positive_scalar(xi_tau, floor=self.jitter))
            try:
                tau_sq = invgamma.rvs(a=shape_tau, scale=scale_tau, random_state=rng)
            except Exception:
                tau_sq = scale_tau / max(shape_tau + 1.0, 1.0)
            tau_sq = _clip_positive_scalar(tau_sq, floor=self.jitter)
            try:
                xi_tau = invgamma.rvs(a=1.0, scale=1.0 + 1.0 / tau_sq, random_state=rng)
            except Exception:
                xi_tau = 1.0
            xi_tau = _clip_positive_scalar(xi_tau)

            resid = y_arr - X @ beta - (C_arr @ alpha if k > 0 else 0.0)
            rss = float(resid @ resid)
            denom_sigma = _clip_positive_array(tau_sq * gamma_sq[group_id] * lambda_sq, floor=self.jitter)
            prior_quad = float(np.sum(np.minimum(beta**2, _POS_CAP) / denom_sigma))
            shape_sigma = 0.5 * (n + p)
            scale_sigma = _clip_positive_scalar(0.5 * (rss + prior_quad) + 1.0 / _clip_positive_scalar(xi_sigma, floor=self.jitter))
            try:
                sigma_sq = invgamma.rvs(a=shape_sigma, scale=scale_sigma, random_state=rng)
            except Exception:
                sigma_sq = scale_sigma / max(shape_sigma + 1.0, 1.0)
            sigma_sq = _clip_positive_scalar(sigma_sq, floor=self.jitter)
            try:
                xi_sigma = invgamma.rvs(a=1.0, scale=1.0 + 1.0 / sigma_sq, random_state=rng)
            except Exception:
                xi_sigma = 1.0
            xi_sigma = _clip_positive_scalar(xi_sigma)

            do_mmle_update = method_eff == "mmle" and ((not self.mmle_burnin_only) or (it < self.n_burn_in))
            if do_mmle_update:
                targets = np.empty(G, dtype=float)
                for gid, idxs in enumerate(normalised_groups):
                    log_lambda_group = float(np.mean(np.log(np.maximum(lambda_sq[idxs], self.jitter))))
                    log_lambda_mean[gid] = (it * log_lambda_mean[gid] + log_lambda_group) / (it + 1)
                    if self.mmle_update == "paper_lambda_only":
                        targets[gid] = -log_lambda_mean[gid]
                    elif self.mmle_update == "legacy_gamma_minus_lambda":
                        targets[gid] = float(np.log(max(gamma_sq[gid], self.jitter))) - log_lambda_mean[gid]
                    else:
                        raise ValueError(f"Unsupported mmle_update '{self.mmle_update}'.")

                if self.share_group_hyper:
                    aggregate = float(np.mean(targets))
                    try:
                        b_update = _digamma_inv(aggregate)
                    except Exception:
                        b_update = float(np.mean(b_vec))
                    clipped = min(max(float(b_update), self.b_floor), self.b_max)
                    b_vec.fill(clipped)
                else:
                    for gid in range(G):
                        try:
                            b_update = _digamma_inv(targets[gid])
                        except Exception:
                            b_update = b_vec[gid]
                        b_vec[gid] = min(max(float(b_update), self.b_floor), self.b_max)

            lambda_sq = _clip_positive_array(lambda_sq, floor=self.jitter)
            gamma_sq = _clip_positive_array(gamma_sq, floor=self.jitter)
            b_vec = _clip_positive_array(b_vec, floor=self.b_floor, cap=self.b_max)

            if it >= self.n_burn_in and ((it - self.n_burn_in) % self.n_thin == 0):
                coef_draws[keep_idx] = beta
                if alpha_draws is not None:
                    alpha_draws[keep_idx] = alpha
                tau2_draws[keep_idx] = tau_sq
                sigma2_draws[keep_idx] = sigma_sq
                gamma2_draws[keep_idx] = gamma_sq
                if lambda_draws is not None:
                    lambda_draws[keep_idx] = lambda_sq
                b_draws[keep_idx] = b_vec
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
        self.b_mean_ = b_draws.mean(axis=0) if kept else b_vec.copy()
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
