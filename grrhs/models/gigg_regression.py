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


def _slice_sample_1d(
    logpdf,
    x0: float,
    rng: Generator,
    *,
    w: float = 0.5,
    m: int = 100,
    max_steps: int = 500,
) -> float:
    """Univariate slice sampler (Neal, 2003) used for log-scale updates."""

    logy = logpdf(x0) - rng.exponential(1.0)
    u = rng.uniform(0.0, 1.0)
    L = x0 - u * w
    R = L + w
    j = int(rng.integers(0, m))
    k = m - 1 - j

    while j > 0 and logpdf(L) > logy:
        L -= w
        j -= 1
    while k > 0 and logpdf(R) > logy:
        R += w
        k -= 1

    step = 0
    while step < max_steps:
        x1 = rng.uniform(L, R)
        if logpdf(x1) >= logy:
            return x1
        if x1 < x0:
            L = x1
        else:
            R = x1
        step += 1
    return x0


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
        iters=int(payload["iters"]),
        burnin=int(payload["burnin"]),
        thin=int(payload["thin"]),
        jitter=float(payload["jitter"]),
        seed=int(payload["seed"]),
        num_chains=1,
        b_init=float(payload["b_init"]),
        b_floor=float(payload["b_floor"]),
        b_max=float(payload["b_max"]),
        tau_scale=float(payload["tau_scale"]),
        sigma_scale=float(payload["sigma_scale"]),
        store_lambda=bool(payload["store_lambda"]),
        a_value=payload["a_value"],
        share_group_hyper=bool(payload["share_group_hyper"]),
        mmle_enabled=bool(payload["mmle_enabled"]),
        mmle_update=str(payload["mmle_update"]),
    )
    fitted = model.fit(
        np.asarray(payload["X"], dtype=float),
        np.asarray(payload["y"], dtype=float),
        groups=payload["groups"],
    )
    return {
        "coef_samples": None if fitted.coef_samples_ is None else np.asarray(fitted.coef_samples_),
        "tau_samples": None if fitted.tau_samples_ is None else np.asarray(fitted.tau_samples_),
        "sigma_samples": None if fitted.sigma_samples_ is None else np.asarray(fitted.sigma_samples_),
        "gamma_samples": None if fitted.gamma_samples_ is None else np.asarray(fitted.gamma_samples_),
        "tau2_samples": None if fitted.tau2_samples_ is None else np.asarray(fitted.tau2_samples_),
        "sigma2_samples": None if fitted.sigma2_samples_ is None else np.asarray(fitted.sigma2_samples_),
        "gamma2_samples": None if fitted.gamma2_samples_ is None else np.asarray(fitted.gamma2_samples_),
        "lambda_samples": None if fitted.lambda_samples_ is None else np.asarray(fitted.lambda_samples_),
        "b_samples": None if fitted.b_samples_ is None else np.asarray(fitted.b_samples_),
        "coef_mean": None if fitted.coef_mean_ is None else np.asarray(fitted.coef_mean_),
        "b_mean": None if fitted.b_mean_ is None else np.asarray(fitted.b_mean_),
        "intercept": float(fitted.intercept_),
    }


@dataclass
class GIGGRegression:
    """GIGG Gibbs sampler following Boss et al. (2021)."""

    iters: int = 3000
    burnin: int = 1500
    thin: int = 1
    jitter: float = 1e-8
    seed: int = 0
    num_chains: int = 1
    b_init: float = 1.0
    b_floor: float = 1e-3
    b_max: float = 4.0
    tau_scale: float = 1.0
    sigma_scale: float = 1.0
    store_lambda: bool = False
    a_value: Optional[float] = None
    share_group_hyper: bool = False
    mmle_enabled: bool = True
    mmle_update: str = "paper_lambda_only"

    rng_: Generator = field(init=False, repr=False)
    coef_samples_: Optional[np.ndarray] = field(default=None, init=False)
    tau_samples_: Optional[np.ndarray] = field(default=None, init=False)
    sigma_samples_: Optional[np.ndarray] = field(default=None, init=False)
    gamma_samples_: Optional[np.ndarray] = field(default=None, init=False)
    tau2_samples_: Optional[np.ndarray] = field(default=None, init=False)
    sigma2_samples_: Optional[np.ndarray] = field(default=None, init=False)
    gamma2_samples_: Optional[np.ndarray] = field(default=None, init=False)
    lambda_samples_: Optional[np.ndarray] = field(default=None, init=False)
    b_samples_: Optional[np.ndarray] = field(default=None, init=False)
    coef_mean_: Optional[np.ndarray] = field(default=None, init=False)
    b_mean_: Optional[np.ndarray] = field(default=None, init=False)
    intercept_: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if self.num_chains <= 0:
            raise ValueError("num_chains must be a positive integer.")

    @staticmethod
    def _flatten_scalar_draws(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        data = np.asarray(arr, dtype=float)
        return data.reshape(-1)

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

    def _spawn_single_chain(self, *, seed: int) -> "GIGGRegression":
        return type(self)(
            iters=self.iters,
            burnin=self.burnin,
            thin=self.thin,
            jitter=self.jitter,
            seed=seed,
            num_chains=1,
            b_init=self.b_init,
            b_floor=self.b_floor,
            b_max=self.b_max,
            tau_scale=self.tau_scale,
            sigma_scale=self.sigma_scale,
            store_lambda=self.store_lambda,
            a_value=self.a_value,
            share_group_hyper=self.share_group_hyper,
            mmle_enabled=self.mmle_enabled,
            mmle_update=self.mmle_update,
        )

    def _fit_multichain(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        groups: Sequence[Sequence[int]],
    ) -> "GIGGRegression":
        payloads: List[dict] = []
        groups_payload = [list(group) for group in groups]
        for chain_idx in range(int(self.num_chains)):
            payloads.append(
                {
                    "iters": self.iters,
                    "burnin": self.burnin,
                    "thin": self.thin,
                    "jitter": self.jitter,
                    "seed": int(self.seed) + chain_idx,
                    "b_init": self.b_init,
                    "b_floor": self.b_floor,
                    "b_max": self.b_max,
                    "tau_scale": self.tau_scale,
                    "sigma_scale": self.sigma_scale,
                    "store_lambda": self.store_lambda,
                    "a_value": self.a_value,
                    "share_group_hyper": self.share_group_hyper,
                    "mmle_enabled": self.mmle_enabled,
                    "mmle_update": self.mmle_update,
                    "X": np.asarray(X, dtype=float),
                    "y": np.asarray(y, dtype=float),
                    "groups": groups_payload,
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
        self.tau_samples_ = self._stack_chain_draws([item["tau_samples"] for item in chain_results])
        self.sigma_samples_ = self._stack_chain_draws([item["sigma_samples"] for item in chain_results])
        self.gamma_samples_ = self._stack_chain_draws([item["gamma_samples"] for item in chain_results])
        self.tau2_samples_ = self._stack_chain_draws([item["tau2_samples"] for item in chain_results])
        self.sigma2_samples_ = self._stack_chain_draws([item["sigma2_samples"] for item in chain_results])
        self.gamma2_samples_ = self._stack_chain_draws([item["gamma2_samples"] for item in chain_results])
        self.lambda_samples_ = self._stack_chain_draws([item["lambda_samples"] for item in chain_results])
        self.b_samples_ = self._stack_chain_draws([item["b_samples"] for item in chain_results])

        coef_draws = self._flatten_param_draws(self.coef_samples_)
        b_draws = self._flatten_param_draws(self.b_samples_)
        self.coef_mean_ = None if coef_draws is None else coef_draws.mean(axis=0)
        self.b_mean_ = None if b_draws is None else b_draws.mean(axis=0)
        self.intercept_ = float(lead["intercept"])
        return self

    def fit(self, X: np.ndarray, y: np.ndarray, *, groups: Sequence[Sequence[int]]) -> "GIGGRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n, p = X.shape
        if y.shape[0] != n:
            raise ValueError("X and y must have compatible shapes.")
        if not groups:
            raise ValueError("GIGGRegression requires a non-empty group specification.")
        if self.num_chains > 1:
            return self._fit_multichain(X, y, groups=groups)

        normalised_groups = _normalise_groups(groups, p)
        group_id = np.empty(p, dtype=int)
        for gid, idxs in enumerate(normalised_groups):
            group_id[idxs] = gid
        group_sizes = np.array([len(g) for g in normalised_groups], dtype=int)
        G = group_sizes.size

        self.rng_ = default_rng(self.seed)
        rng = self.rng_

        a_const = float(self.a_value) if self.a_value is not None else 1.0 / max(n, 1)
        a_vec = np.full(G, a_const)
        b_vec = np.full(G, max(self.b_init, self.b_floor))

        lambda_sq = np.ones(p, dtype=float)
        gamma_sq = np.ones(G, dtype=float)
        tau_sq = _clip_positive_scalar(self.tau_scale)
        sigma_sq = _clip_positive_scalar(self.sigma_scale)
        xi_tau = 1.0
        xi_sigma = 1.0

        XtX = X.T @ X
        Xty = X.T @ y

        kept = max(0, (self.iters - self.burnin) // max(self.thin, 1))
        coef_draws = np.zeros((kept, p), dtype=float)
        tau2_draws = np.zeros(kept, dtype=float)
        sigma2_draws = np.zeros(kept, dtype=float)
        gamma2_draws = np.zeros((kept, G), dtype=float)
        lambda_draws = np.zeros((kept, p), dtype=float) if self.store_lambda else None
        b_draws = np.zeros((kept, G), dtype=float)

        log_lambda_mean = np.zeros(G, dtype=float)

        keep_idx = 0
        for it in range(self.iters):
            # ---- β | rest
            local_scale = _clip_positive_array(tau_sq * gamma_sq[group_id] * lambda_sq, floor=self.jitter)
            prior_prec = 1.0 / local_scale
            precision = XtX + np.diag(prior_prec)
            if self.jitter > 0.0:
                precision = precision + np.eye(p) * self.jitter
            chol = cho_factor(precision, lower=True, check_finite=False)
            mean = cho_solve(chol, Xty, check_finite=False)
            noise = cho_solve(chol, rng.normal(size=p), check_finite=False)
            beta = mean + math.sqrt(sigma_sq) * noise
            beta = np.clip(np.nan_to_num(beta, nan=0.0, posinf=_BETA_CAP, neginf=-_BETA_CAP), -_BETA_CAP, _BETA_CAP)

            # ---- λ^2 | β, γ
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

            # ---- γ_g^2 | β, λ
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

            # ---- τ^2 | β, λ, γ  (log-space slice sampling for stability)
            denom_tau = _clip_positive_array(gamma_sq[group_id] * lambda_sq, floor=self.jitter)
            beta_quad = float(np.sum(np.minimum(beta ** 2, _POS_CAP) / denom_tau))
            alpha_tau = 0.5 * (p + 1)
            beta_tau = 0.5 * beta_quad + 1.0 / _clip_positive_scalar(xi_tau, floor=self.jitter)

            def _logp_tau(v: float) -> float:
                return -alpha_tau * v - beta_tau * math.exp(-v)

            log_tau = math.log(max(tau_sq, self.jitter))
            log_tau_new = _slice_sample_1d(_logp_tau, log_tau, rng, w=0.5, m=100, max_steps=500)
            tau_sq = _clip_positive_scalar(math.exp(log_tau_new), floor=self.jitter)
            try:
                xi_tau = invgamma.rvs(a=1.0, scale=1.0 + 1.0 / tau_sq, random_state=rng)
            except Exception:
                xi_tau = 1.0
            xi_tau = _clip_positive_scalar(xi_tau)

            # ---- σ^2 | β
            resid = y - X @ beta
            rss = float(resid @ resid)
            denom_sigma = _clip_positive_array(tau_sq * gamma_sq[group_id] * lambda_sq, floor=self.jitter)
            prior_quad = float(np.sum(np.minimum(beta ** 2, _POS_CAP) / denom_sigma))
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

            # Empirical Bayes update for b_g following the paper's b-only MMLE path:
            #   b_g^{l+1} = psi_0^{-1}( - E[ mean_j log(lambda_gj^2) | y ] )
            if self.mmle_enabled:
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

            if it >= self.burnin and ((it - self.burnin) % max(self.thin, 1) == 0):
                coef_draws[keep_idx] = beta
                tau2_draws[keep_idx] = tau_sq
                sigma2_draws[keep_idx] = sigma_sq
                gamma2_draws[keep_idx] = gamma_sq
                if lambda_draws is not None:
                    lambda_draws[keep_idx] = lambda_sq
                b_draws[keep_idx] = b_vec
                keep_idx += 1

        self.coef_samples_ = coef_draws if kept else None
        self.tau2_samples_ = tau2_draws if kept else None
        self.sigma2_samples_ = sigma2_draws if kept else None
        self.gamma2_samples_ = gamma2_draws if kept else None
        self.tau_samples_ = None if self.tau2_samples_ is None else np.sqrt(np.maximum(self.tau2_samples_, self.jitter))
        self.sigma_samples_ = None if self.sigma2_samples_ is None else np.sqrt(np.maximum(self.sigma2_samples_, self.jitter))
        self.gamma_samples_ = None if self.gamma2_samples_ is None else np.sqrt(np.maximum(self.gamma2_samples_, self.jitter))
        self.lambda_samples_ = lambda_draws if lambda_draws is not None else None
        self.b_samples_ = b_draws if kept else None
        self.coef_mean_ = coef_draws.mean(axis=0) if kept else beta
        self.b_mean_ = b_draws.mean(axis=0) if kept else b_vec.copy()
        self.intercept_ = 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_mean_ is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        X_arr = np.asarray(X, dtype=float)
        return X_arr @ self.coef_mean_ + self.intercept_
