from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from numpy.random import Generator, default_rng
from scipy.linalg import cho_factor, cho_solve
from scipy.special import digamma, polygamma
from scipy.stats import invgamma

from grrhs.inference.gig import sample_gig


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


@dataclass
class GIGGRegression:
    """GIGG Gibbs sampler following Boss et al. (2021)."""

    iters: int = 3000
    burnin: int = 1500
    thin: int = 1
    jitter: float = 1e-8
    seed: int = 0
    b_init: float = 1.0
    b_floor: float = 0.25
    b_max: float = 2.0
    tau_scale: float = 1.0
    sigma_scale: float = 1.0
    store_lambda: bool = False
    a_value: Optional[float] = None
    share_group_hyper: bool = False

    rng_: Generator = field(init=False, repr=False)
    coef_samples_: Optional[np.ndarray] = field(default=None, init=False)
    tau_samples_: Optional[np.ndarray] = field(default=None, init=False)
    sigma_samples_: Optional[np.ndarray] = field(default=None, init=False)
    gamma_samples_: Optional[np.ndarray] = field(default=None, init=False)
    lambda_samples_: Optional[np.ndarray] = field(default=None, init=False)
    coef_mean_: Optional[np.ndarray] = field(default=None, init=False)
    intercept_: float = field(default=0.0, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray, *, groups: Sequence[Sequence[int]]) -> "GIGGRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n, p = X.shape
        if y.shape[0] != n:
            raise ValueError("X and y must have compatible shapes.")
        if not groups:
            raise ValueError("GIGGRegression requires a non-empty group specification.")

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
        tau_sq = float(self.tau_scale)
        sigma_sq = float(self.sigma_scale)
        xi_tau = 1.0
        xi_sigma = 1.0

        XtX = X.T @ X
        Xty = X.T @ y

        kept = max(0, (self.iters - self.burnin) // max(self.thin, 1))
        coef_draws = np.zeros((kept, p), dtype=float)
        tau_draws = np.zeros(kept, dtype=float)
        sigma_draws = np.zeros(kept, dtype=float)
        gamma_draws = np.zeros((kept, G), dtype=float)
        lambda_draws = np.zeros((kept, p), dtype=float) if self.store_lambda else None

        log_gamma_mean = np.log(np.maximum(gamma_sq, self.jitter))
        log_lambda_mean = np.zeros(G, dtype=float)

        keep_idx = 0
        for it in range(self.iters):
            # ---- β | rest
            prior_prec = 1.0 / np.maximum(tau_sq * gamma_sq[group_id] * lambda_sq, self.jitter)
            precision = XtX + np.diag(prior_prec)
            if self.jitter > 0.0:
                precision = precision + np.eye(p) * self.jitter
            chol = cho_factor(precision, lower=True, check_finite=False)
            mean = cho_solve(chol, Xty, check_finite=False)
            noise = cho_solve(chol, rng.normal(size=p), check_finite=False)
            beta = mean + math.sqrt(sigma_sq) * noise

            # ---- λ^2 | β, γ
            for gid, idxs in enumerate(normalised_groups):
                denom = 2.0 * max(tau_sq * gamma_sq[gid], self.jitter)
                b_shape = max(b_vec[gid] + 0.5, 1e-6)
                for j in idxs:
                    scale = 1.0 + (beta[j] ** 2) / denom
                    lambda_sq[j] = invgamma.rvs(a=b_shape, scale=scale, random_state=rng)

            # ---- γ_g^2 | β, λ
            for gid, idxs in enumerate(normalised_groups):
                lam_param = float(a_vec[gid] - 0.5 * group_sizes[gid])
                chi = np.sum(beta[idxs] ** 2 / np.maximum(lambda_sq[idxs], self.jitter))
                psi = 2.0 * max(b_vec[gid], self.b_floor)
                theta = sample_gig(
                    lambda_param=lam_param,
                    chi=max(chi, self.jitter),
                    psi=max(psi, self.jitter),
                    size=1,
                    rng=rng,
                )[0]
                gamma_sq[gid] = max(theta, self.jitter)

            # ---- τ^2 | β, λ, γ  (log-space slice sampling for stability)
            beta_quad = np.sum(beta ** 2 / np.maximum(gamma_sq[group_id] * lambda_sq, self.jitter))
            alpha_tau = 0.5 * (p + 1)
            beta_tau = 0.5 * beta_quad + 1.0 / max(xi_tau, self.jitter)

            def _logp_tau(v: float) -> float:
                return -alpha_tau * v - beta_tau * math.exp(-v)

            log_tau = math.log(max(tau_sq, self.jitter))
            log_tau_new = _slice_sample_1d(_logp_tau, log_tau, rng, w=0.5, m=100, max_steps=500)
            tau_sq = math.exp(log_tau_new)
            xi_tau = invgamma.rvs(a=1.0, scale=1.0 + 1.0 / max(tau_sq, self.jitter), random_state=rng)

            # ---- σ^2 | β
            resid = y - X @ beta
            rss = float(resid @ resid)
            prior_quad = np.sum(beta ** 2 / np.maximum(tau_sq * gamma_sq[group_id] * lambda_sq, self.jitter))
            shape_sigma = 0.5 * (n + p)
            scale_sigma = 0.5 * (rss + prior_quad) + 1.0 / max(xi_sigma, self.jitter)
            sigma_sq = invgamma.rvs(a=shape_sigma, scale=scale_sigma, random_state=rng)
            xi_sigma = invgamma.rvs(a=1.0, scale=1.0 + 1.0 / max(sigma_sq, self.jitter), random_state=rng)

            # ---- Empirical Bayes update for b_g
            log_gamma_mean = (it * log_gamma_mean + np.log(np.maximum(gamma_sq, self.jitter))) / (it + 1)
            targets = np.empty(G, dtype=float)
            for gid, idxs in enumerate(normalised_groups):
                log_lambda_group = float(np.mean(np.log(np.maximum(lambda_sq[idxs], self.jitter))))
                log_lambda_mean[gid] = (it * log_lambda_mean[gid] + log_lambda_group) / (it + 1)
                targets[gid] = log_gamma_mean[gid] - log_lambda_mean[gid]

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

            if it >= self.burnin and ((it - self.burnin) % max(self.thin, 1) == 0):
                coef_draws[keep_idx] = beta
                tau_draws[keep_idx] = tau_sq
                sigma_draws[keep_idx] = sigma_sq
                gamma_draws[keep_idx] = gamma_sq
                if lambda_draws is not None:
                    lambda_draws[keep_idx] = lambda_sq
                keep_idx += 1

        self.coef_samples_ = coef_draws if kept else None
        self.tau_samples_ = tau_draws if kept else None
        self.sigma_samples_ = sigma_draws if kept else None
        self.gamma_samples_ = gamma_draws if kept else None
        self.lambda_samples_ = lambda_draws if lambda_draws is not None else None
        self.coef_mean_ = coef_draws.mean(axis=0) if kept else beta
        self.intercept_ = 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_mean_ is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        X_arr = np.asarray(X, dtype=float)
        return X_arr @ self.coef_mean_ + self.intercept_
