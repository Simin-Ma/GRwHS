from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, Optional

import numpy as np
from scipy.linalg import solve_triangular
from numpy.random import Generator, default_rng

from simulation_project.src.core.inference.samplers import slice_sample_1d
from simulation_project.src.core.inference.woodbury import beta_sample_cholesky, beta_sample_woodbury


_MIN_POS = 1e-10


def _flatten_scalar_draws(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    return np.asarray(arr, dtype=float).reshape(-1)


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


def _standardize_design_exact(
    X: np.ndarray,
    *,
    center: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_arr = np.asarray(X, dtype=float)
    x_center = X_arr.mean(axis=0) if center else np.zeros(X_arr.shape[1], dtype=float)
    X_ctr = X_arr - x_center
    x_scale = X_ctr.std(axis=0, ddof=0)
    x_scale = np.where(x_scale < 1e-8, 1.0, x_scale)
    return X_ctr / x_scale, x_center, x_scale


def _log_half_cauchy(value: float, scale: float) -> float:
    v = float(max(value, _MIN_POS))
    s = float(max(scale, _MIN_POS))
    return math.log(2.0 / math.pi) - math.log(s) - math.log1p((v / s) ** 2)


def _ridge_init_beta(X: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    p = int(X.shape[1])
    lhs = X.T @ X + float(ridge) * np.eye(p, dtype=float)
    rhs = X.T @ y
    try:
        beta = np.linalg.solve(lhs, rhs)
    except Exception:
        beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return np.nan_to_num(np.asarray(beta, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def _rhs_single_prior_var(
    *,
    sigma2: float,
    tau2: float,
    lam2: float,
    c2: float,
    jitter: float,
) -> float:
    a2 = max(float(sigma2) * float(tau2) * float(lam2), float(jitter))
    return float(max((float(c2) * a2) / max(float(c2) + a2, float(jitter)), float(jitter)))


def _rhs_single_lambda_tilde(
    *,
    sigma2: float,
    tau2: float,
    lam2: float,
    c2: float,
    jitter: float,
) -> float:
    denom = max(float(c2) + max(float(sigma2) * float(tau2) * float(lam2), float(jitter)), float(jitter))
    return float(math.sqrt(max((float(c2) * float(lam2)) / denom, float(jitter))))


def _log_marginal_gaussian(
    *,
    X: np.ndarray,
    y: np.ndarray,
    sigma2: float,
    prior_var: np.ndarray,
    jitter: float,
) -> float:
    sigma2_use = max(float(sigma2), float(jitter))
    prior_var_use = np.maximum(np.asarray(prior_var, dtype=float).reshape(-1), float(jitter))
    weighted_x = X * prior_var_use
    cov = np.eye(int(X.shape[0]), dtype=float) + (weighted_x @ X.T) / sigma2_use
    cov = 0.5 * (cov + cov.T)
    try:
        chol = np.linalg.cholesky(cov + float(jitter) * np.eye(cov.shape[0], dtype=float))
    except np.linalg.LinAlgError:
        chol = np.linalg.cholesky(cov + (10.0 * float(jitter)) * np.eye(cov.shape[0], dtype=float))
    logdet = 2.0 * float(np.sum(np.log(np.diag(chol))))
    alpha = solve_triangular(chol, np.asarray(y, dtype=float).reshape(-1), lower=True, check_finite=False)
    quad = float(alpha @ alpha) / sigma2_use
    return float(-0.5 * (int(X.shape[0]) * math.log(2.0 * math.pi) + int(X.shape[0]) * math.log(sigma2_use) + logdet + quad))


@dataclass
class RegularizedHorseshoeGibbs:
    """High-dimensional Gaussian RHS sampler using conditional Gaussian beta refreshes.

    This sampler keeps the regularized horseshoe prior structure but avoids
    full-model HMC. It samples beta from its exact Gaussian conditional using
    a Woodbury path when n < p, then updates scalar and local-scale
    hyperparameters with one-dimensional slice sampling.

    The Gaussian variant here is intentionally close to the existing RHS
    baseline but works directly in a sampler-friendly parameterization:
      tau_raw ~ C+(0, scale_global)
      lambda_j ~ C+(0, 1)
      caux ~ InvGamma(slab_df / 2, slab_df / 2)
      c^2 = slab_scale^2 * caux
      beta_j | sigma, tau_raw, lambda_j, c ~ N(0, v_j)

    where
      v_j = a_j^2 c^2 / (c^2 + a_j^2)
      a_j^2 = sigma^2 tau_raw^2 lambda_j^2
    """

    scale_intercept: float = 10.0
    scale_global: float = 0.01
    slab_scale: float = 2.5
    slab_df: float = 4.0
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 1
    thinning: int = 1
    seed: Optional[int] = None
    progress_bar: bool = True
    fit_intercept: bool = True
    jitter: float = 1e-8
    slice_width_log_sigma: float = 0.35
    slice_width_log_tau: float = 0.35
    slice_width_log_lambda: float = 0.45
    slice_width_log_caux: float = 0.45
    slice_max_steps: int = 200
    lambda_active_fraction: float = 0.25
    lambda_active_min: int = 32
    lambda_full_refresh_every: int = 8
    lambda_warmup_full_refresh: bool = True
    tau_refresh_after_local: bool = True
    beta_refresh_after_hyper: bool = True

    coef_samples_: Optional[np.ndarray] = field(default=None, init=False)
    intercept_samples_: Optional[np.ndarray] = field(default=None, init=False)
    sigma_samples_: Optional[np.ndarray] = field(default=None, init=False)
    tau_samples_: Optional[np.ndarray] = field(default=None, init=False)
    lambda_samples_: Optional[np.ndarray] = field(default=None, init=False)
    lambda_tilde_samples_: Optional[np.ndarray] = field(default=None, init=False)
    c_samples_: Optional[np.ndarray] = field(default=None, init=False)
    coef_: Optional[np.ndarray] = field(default=None, init=False)
    coef_mean_: Optional[np.ndarray] = field(default=None, init=False)
    intercept_: Optional[float] = field(default=None, init=False)
    sigma_mean_: Optional[float] = field(default=None, init=False)
    tau_mean_: Optional[float] = field(default=None, init=False)
    lambda_mean_: Optional[np.ndarray] = field(default=None, init=False)
    lambda_tilde_mean_: Optional[np.ndarray] = field(default=None, init=False)
    c_mean_: Optional[float] = field(default=None, init=False)
    sampler_diagnostics_: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.scale_intercept <= 0.0:
            raise ValueError("scale_intercept must be positive.")
        if self.scale_global <= 0.0:
            raise ValueError("scale_global must be positive.")
        if self.slab_scale <= 0.0:
            raise ValueError("slab_scale must be positive.")
        if self.slab_df <= 0.0:
            raise ValueError("slab_df must be positive.")
        if int(self.num_warmup) <= 0 or int(self.num_samples) <= 0:
            raise ValueError("num_warmup and num_samples must be positive.")
        if int(self.num_chains) <= 0:
            raise ValueError("num_chains must be positive.")
        if int(self.thinning) <= 0:
            raise ValueError("thinning must be positive.")
        if float(self.jitter) <= 0.0:
            raise ValueError("jitter must be positive.")
        if int(self.slice_max_steps) <= 0:
            raise ValueError("slice_max_steps must be positive.")
        if not 0.0 < float(self.lambda_active_fraction) <= 1.0:
            raise ValueError("lambda_active_fraction must lie in (0, 1].")
        if int(self.lambda_active_min) <= 0:
            raise ValueError("lambda_active_min must be positive.")
        if int(self.lambda_full_refresh_every) <= 0:
            raise ValueError("lambda_full_refresh_every must be positive.")

    def _prior_var(
        self,
        *,
        sigma: float,
        tau_raw: float,
        lam: np.ndarray,
        caux: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        sigma2 = max(float(sigma) ** 2, self.jitter)
        tau2 = max(float(tau_raw) ** 2, self.jitter)
        lam2 = np.maximum(np.asarray(lam, dtype=float) ** 2, self.jitter)
        c2 = max((float(self.slab_scale) ** 2) * float(max(caux, self.jitter)), self.jitter)
        a2 = sigma2 * tau2 * lam2
        prior_var = c2 * a2 / np.maximum(c2 + a2, self.jitter)
        prior_var = np.maximum(np.asarray(prior_var, dtype=float), self.jitter)
        lambda_tilde = np.sqrt(np.maximum(c2 * lam2 / np.maximum(c2 + a2, self.jitter), self.jitter))
        return prior_var, lambda_tilde, c2

    def _sample_beta(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
        XtX: np.ndarray,
        Xty: np.ndarray,
        sigma: float,
        prior_var: np.ndarray,
        rng: Generator,
    ) -> np.ndarray:
        sigma2 = max(float(sigma) ** 2, self.jitter)
        if int(X.shape[0]) < int(X.shape[1]):
            return beta_sample_woodbury(X, y, sigma2, prior_var, rng, jitter=float(self.jitter))
        return beta_sample_cholesky(XtX, Xty, sigma2, prior_var, rng, jitter=float(self.jitter))

    def _select_lambda_update_indices(self, beta: np.ndarray, *, iteration: int, in_warmup: bool) -> np.ndarray:
        p = int(beta.shape[0])
        if bool(in_warmup) and bool(self.lambda_warmup_full_refresh):
            return np.arange(p, dtype=int)
        if p <= int(self.lambda_active_min) or int(self.lambda_full_refresh_every) <= 1:
            return np.arange(p, dtype=int)
        if int(iteration) % int(self.lambda_full_refresh_every) == 0:
            return np.arange(p, dtype=int)
        active_k = min(p, max(int(self.lambda_active_min), int(math.ceil(float(self.lambda_active_fraction) * p))))
        if active_k >= p:
            return np.arange(p, dtype=int)
        abs_beta = np.abs(np.asarray(beta, dtype=float))
        order = np.argpartition(-abs_beta, kth=active_k - 1)[:active_k]
        return np.asarray(np.sort(order), dtype=int)

    def _sample_single_chain(self, X: np.ndarray, y: np.ndarray, *, seed: int) -> Dict[str, np.ndarray]:
        rng = default_rng(int(seed))
        X_std, x_center, x_scale = _standardize_design_exact(X, center=bool(self.fit_intercept))
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        y_mean = float(y_arr.mean()) if bool(self.fit_intercept) else 0.0
        y_ctr = y_arr - y_mean
        n, p = X_std.shape

        XtX = X_std.T @ X_std
        Xty = X_std.T @ y_ctr
        beta = _ridge_init_beta(X_std, y_ctr)
        resid = y_ctr - X_std @ beta
        sigma = float(max(np.std(resid, ddof=0), 0.1))
        tau_raw = float(max(self.scale_global, 1e-4))
        lam = np.ones(p, dtype=float)
        caux = 1.0

        total_iters = int(self.num_warmup + self.num_samples)
        kept = max(0, (int(self.num_samples) + int(self.thinning) - 1) // int(self.thinning))
        beta_draws = np.zeros((kept, p), dtype=float)
        intercept_draws = np.zeros(kept, dtype=float)
        sigma_draws = np.zeros(kept, dtype=float)
        tau_draws = np.zeros(kept, dtype=float)
        lambda_draws = np.ones((kept, p), dtype=float)
        lambda_tilde_draws = np.ones((kept, p), dtype=float)
        c_draws = np.ones(kept, dtype=float)
        keep_i = 0
        lambda_update_total = 0
        lambda_full_refresh_count = 0
        lambda_active_refresh_count = 0

        iterator = range(total_iters)
        if bool(self.progress_bar):
            try:
                from simulation_project.src.core.utils.logging_utils import progress

                iterator = progress(iterator, total=total_iters, desc="RHS Gibbs")
            except Exception:
                pass

        for it in iterator:
            prior_var, _, _ = self._prior_var(sigma=sigma, tau_raw=tau_raw, lam=lam, caux=caux)
            beta = self._sample_beta(
                X=X_std,
                y=y_ctr,
                XtX=XtX,
                Xty=Xty,
                sigma=sigma,
                prior_var=prior_var,
                rng=rng,
            )
            resid = y_ctr - X_std @ beta

            def _logpost_log_sigma(value: float) -> float:
                sigma_use = math.exp(float(value))
                prior_var_use, _, _ = self._prior_var(
                    sigma=sigma_use,
                    tau_raw=tau_raw,
                    lam=lam,
                    caux=caux,
                )
                rss = float(resid @ resid)
                lp = -float(n) * math.log(max(sigma_use, _MIN_POS))
                lp -= 0.5 * rss / max(sigma_use ** 2, self.jitter)
                lp -= 0.5 * float(np.sum(np.log(np.maximum(prior_var_use, self.jitter))))
                lp -= 0.5 * float(np.sum((beta * beta) / np.maximum(prior_var_use, self.jitter)))
                lp -= 0.5 * sigma_use ** 2
                lp += float(value)
                return float(lp)

            sigma = math.exp(
                slice_sample_1d(
                    _logpost_log_sigma,
                    math.log(max(sigma, _MIN_POS)),
                    rng,
                    width=float(self.slice_width_log_sigma),
                    max_steps=int(self.slice_max_steps),
                )
            )

            sigma2 = max(float(sigma) ** 2, self.jitter)

            def _logpost_log_tau(value: float) -> float:
                tau_use = math.exp(float(value))
                prior_var_use, _, _ = self._prior_var(
                    sigma=sigma,
                    tau_raw=tau_use,
                    lam=lam,
                    caux=caux,
                )
                lp = _log_marginal_gaussian(
                    X=X_std,
                    y=y_ctr,
                    sigma2=sigma2,
                    prior_var=prior_var_use,
                    jitter=float(self.jitter),
                )
                lp += _log_half_cauchy(tau_use, float(self.scale_global))
                lp += float(value)
                return float(lp)

            tau_raw = math.exp(
                slice_sample_1d(
                    _logpost_log_tau,
                    math.log(max(tau_raw, _MIN_POS)),
                    rng,
                    width=float(self.slice_width_log_tau),
                    max_steps=int(self.slice_max_steps),
                )
            )

            tau2 = max(float(tau_raw) ** 2, self.jitter)
            c2 = max((float(self.slab_scale) ** 2) * float(max(caux, self.jitter)), self.jitter)

            lambda_update_idx = self._select_lambda_update_indices(
                beta,
                iteration=it,
                in_warmup=bool(it < int(self.num_warmup)),
            )
            lambda_update_total += int(lambda_update_idx.size)
            if int(lambda_update_idx.size) == p:
                lambda_full_refresh_count += 1
            else:
                lambda_active_refresh_count += 1

            for j in lambda_update_idx:
                beta_j = float(beta[j])

                def _logpost_log_lambda(value: float, beta_j_use: float = beta_j) -> float:
                    lam_use = math.exp(float(value))
                    v_j = _rhs_single_prior_var(
                        sigma2=sigma2,
                        tau2=tau2,
                        lam2=lam_use ** 2,
                        c2=c2,
                        jitter=float(self.jitter),
                    )
                    lp = -0.5 * math.log(v_j) - 0.5 * (beta_j_use ** 2) / v_j
                    lp += _log_half_cauchy(lam_use, 1.0)
                    lp += float(value)
                    return float(lp)

                lam[j] = math.exp(
                    slice_sample_1d(
                        _logpost_log_lambda,
                        math.log(max(float(lam[j]), _MIN_POS)),
                        rng,
                        width=float(self.slice_width_log_lambda),
                        max_steps=int(self.slice_max_steps),
                    )
                )

            def _logpost_log_caux(value: float) -> float:
                caux_use = math.exp(float(value))
                prior_var_use, _, _ = self._prior_var(
                    sigma=sigma,
                    tau_raw=tau_raw,
                    lam=lam,
                    caux=caux_use,
                )
                alpha = 0.5 * float(self.slab_df)
                rate = 0.5 * float(self.slab_df)
                lp = _log_marginal_gaussian(
                    X=X_std,
                    y=y_ctr,
                    sigma2=sigma2,
                    prior_var=prior_var_use,
                    jitter=float(self.jitter),
                )
                lp -= alpha * float(value)
                lp -= rate / max(caux_use, self.jitter)
                return float(lp)

            caux = math.exp(
                slice_sample_1d(
                    _logpost_log_caux,
                    math.log(max(caux, _MIN_POS)),
                    rng,
                    width=float(self.slice_width_log_caux),
                    max_steps=int(self.slice_max_steps),
                )
            )

            if bool(self.tau_refresh_after_local):
                def _logpost_log_tau_local(value: float) -> float:
                    tau_use = math.exp(float(value))
                    prior_var_use, _, _ = self._prior_var(
                        sigma=sigma,
                        tau_raw=tau_use,
                        lam=lam,
                        caux=caux,
                    )
                    lp = _log_marginal_gaussian(
                        X=X_std,
                        y=y_ctr,
                        sigma2=sigma2,
                        prior_var=prior_var_use,
                        jitter=float(self.jitter),
                    )
                    lp += _log_half_cauchy(tau_use, float(self.scale_global))
                    lp += float(value)
                    return float(lp)

                tau_raw = math.exp(
                    slice_sample_1d(
                        _logpost_log_tau_local,
                        math.log(max(tau_raw, _MIN_POS)),
                        rng,
                        width=float(self.slice_width_log_tau),
                        max_steps=int(self.slice_max_steps),
                    )
                )

            if bool(self.beta_refresh_after_hyper):
                prior_var, _, _ = self._prior_var(sigma=sigma, tau_raw=tau_raw, lam=lam, caux=caux)
                beta = self._sample_beta(
                    X=X_std,
                    y=y_ctr,
                    XtX=XtX,
                    Xty=Xty,
                    sigma=sigma,
                    prior_var=prior_var,
                    rng=rng,
                )

            if it >= int(self.num_warmup) and ((it - int(self.num_warmup)) % int(self.thinning) == 0):
                prior_var, lambda_tilde, c2 = self._prior_var(
                    sigma=sigma,
                    tau_raw=tau_raw,
                    lam=lam,
                    caux=caux,
                )
                beta_orig = beta / x_scale
                intercept = float(y_mean - np.dot(x_center, beta_orig)) if bool(self.fit_intercept) else 0.0
                beta_draws[keep_i] = beta_orig
                intercept_draws[keep_i] = intercept
                sigma_draws[keep_i] = sigma
                tau_draws[keep_i] = sigma * tau_raw
                lambda_draws[keep_i] = lam
                lambda_tilde_draws[keep_i] = lambda_tilde
                c_draws[keep_i] = math.sqrt(max(c2, self.jitter))
                keep_i += 1

        return {
            "coef_samples": beta_draws,
            "intercept_samples": intercept_draws,
            "sigma_samples": sigma_draws,
            "tau_samples": tau_draws,
            "lambda_samples": lambda_draws,
            "lambda_tilde_samples": lambda_tilde_draws,
            "c_samples": c_draws,
            "lambda_update_total": np.asarray(lambda_update_total, dtype=int),
            "lambda_full_refresh_count": np.asarray(lambda_full_refresh_count, dtype=int),
            "lambda_active_refresh_count": np.asarray(lambda_active_refresh_count, dtype=int),
            "total_iters": np.asarray(total_iters, dtype=int),
            "p": np.asarray(p, dtype=int),
        }

    def fit(self, X: np.ndarray, y: np.ndarray, *, groups: Any = None) -> "RegularizedHorseshoeGibbs":
        del groups
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("X and y must have compatible first dimensions.")

        start = time.perf_counter()
        chain_results = [
            self._sample_single_chain(X_arr, y_arr, seed=(0 if self.seed is None else int(self.seed)) + chain_idx)
            for chain_idx in range(int(self.num_chains))
        ]
        runtime_sec = max(time.perf_counter() - start, 1e-12)

        if int(self.num_chains) == 1:
            lead = chain_results[0]
            self.coef_samples_ = lead["coef_samples"]
            self.intercept_samples_ = lead["intercept_samples"]
            self.sigma_samples_ = lead["sigma_samples"]
            self.tau_samples_ = lead["tau_samples"]
            self.lambda_samples_ = lead["lambda_samples"]
            self.lambda_tilde_samples_ = lead["lambda_tilde_samples"]
            self.c_samples_ = lead["c_samples"]
            lambda_update_total = int(np.asarray(lead["lambda_update_total"]).reshape(()))
            lambda_full_refresh_count = int(np.asarray(lead["lambda_full_refresh_count"]).reshape(()))
            lambda_active_refresh_count = int(np.asarray(lead["lambda_active_refresh_count"]).reshape(()))
            total_iters = int(np.asarray(lead["total_iters"]).reshape(()))
            p = int(np.asarray(lead["p"]).reshape(()))
        else:
            self.coef_samples_ = np.stack([item["coef_samples"] for item in chain_results], axis=0)
            self.intercept_samples_ = np.stack([item["intercept_samples"] for item in chain_results], axis=0)
            self.sigma_samples_ = np.stack([item["sigma_samples"] for item in chain_results], axis=0)
            self.tau_samples_ = np.stack([item["tau_samples"] for item in chain_results], axis=0)
            self.lambda_samples_ = np.stack([item["lambda_samples"] for item in chain_results], axis=0)
            self.lambda_tilde_samples_ = np.stack([item["lambda_tilde_samples"] for item in chain_results], axis=0)
            self.c_samples_ = np.stack([item["c_samples"] for item in chain_results], axis=0)
            lambda_update_total = int(np.sum([int(np.asarray(item["lambda_update_total"]).reshape(())) for item in chain_results]))
            lambda_full_refresh_count = int(np.sum([int(np.asarray(item["lambda_full_refresh_count"]).reshape(())) for item in chain_results]))
            lambda_active_refresh_count = int(np.sum([int(np.asarray(item["lambda_active_refresh_count"]).reshape(())) for item in chain_results]))
            total_iters = int(np.asarray(chain_results[0]["total_iters"]).reshape(()))
            p = int(np.asarray(chain_results[0]["p"]).reshape(()))

        coef_draws = _flatten_param_draws(self.coef_samples_)
        intercept_draws = _flatten_scalar_draws(self.intercept_samples_)
        sigma_draws = _flatten_scalar_draws(self.sigma_samples_)
        tau_draws = _flatten_scalar_draws(self.tau_samples_)
        lambda_draws = _flatten_param_draws(self.lambda_samples_)
        lambda_tilde_draws = _flatten_param_draws(self.lambda_tilde_samples_)
        c_draws = _flatten_scalar_draws(self.c_samples_)

        self.coef_mean_ = None if coef_draws is None else coef_draws.mean(axis=0)
        self.coef_ = None if self.coef_mean_ is None else self.coef_mean_.copy()
        self.intercept_ = None if intercept_draws is None else float(intercept_draws.mean())
        self.sigma_mean_ = None if sigma_draws is None else float(sigma_draws.mean())
        self.tau_mean_ = None if tau_draws is None else float(tau_draws.mean())
        self.lambda_mean_ = None if lambda_draws is None else lambda_draws.mean(axis=0)
        self.lambda_tilde_mean_ = None if lambda_tilde_draws is None else lambda_tilde_draws.mean(axis=0)
        self.c_mean_ = None if c_draws is None else float(c_draws.mean())

        kept = 0 if self.coef_samples_ is None else int(
            np.asarray(self.coef_samples_).shape[-2]
            if np.asarray(self.coef_samples_).ndim >= 3
            else np.asarray(self.coef_samples_).shape[0]
        )
        self.sampler_diagnostics_ = {
            "backend": "rhs_gibbs_woodbury",
            "runtime_sec": float(runtime_sec),
            "num_chains": int(self.num_chains),
            "kept_draws_per_chain": int(kept),
            "model": "Regularized Horseshoe (Gaussian) Gibbs/Slice",
            "parameterization": {
                "sigma_scaled_global": True,
                "local_prior": "half_cauchy",
                "global_prior": "half_cauchy",
                "slab_prior": "inv_gamma_on_caux",
                "beta_refresh": "woodbury_if_n_lt_p_else_cholesky",
                "lambda_update": "active_subset_plus_periodic_full_slice",
                "tau_refresh_after_local": bool(self.tau_refresh_after_local),
                "beta_refresh_after_hyper": bool(self.beta_refresh_after_hyper),
            },
            "lambda_refresh": {
                "active_fraction": float(self.lambda_active_fraction),
                "active_min": int(self.lambda_active_min),
                "full_refresh_every": int(self.lambda_full_refresh_every),
                "warmup_full_refresh": bool(self.lambda_warmup_full_refresh),
                "total_lambda_updates": int(lambda_update_total),
                "full_refresh_count": int(lambda_full_refresh_count),
                "active_refresh_count": int(lambda_active_refresh_count),
                "mean_lambda_updates_per_iter_per_chain": float(
                    lambda_update_total / max(int(self.num_chains) * max(total_iters, 1), 1)
                ),
                "mean_lambda_update_fraction_per_iter": float(
                    lambda_update_total / max(int(self.num_chains) * max(total_iters, 1) * max(p, 1), 1)
                ),
            },
            "standardization": {
                "x_center": bool(self.fit_intercept),
                "x_scale": "column_std",
                "y_center": bool(self.fit_intercept),
            },
        }
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        X_arr = np.asarray(X, dtype=float)
        return X_arr @ np.asarray(self.coef_, dtype=float) + float(0.0 if self.intercept_ is None else self.intercept_)

    def get_posterior_summaries(self) -> Dict[str, Any]:
        if self.coef_samples_ is None:
            raise RuntimeError("Model must be fitted before requesting summaries.")
        coef_draws = _flatten_param_draws(self.coef_samples_)
        if coef_draws is None:
            raise RuntimeError("Posterior coefficient draws are unavailable.")
        return {
            "beta_mean": coef_draws.mean(axis=0),
            "beta_median": np.median(coef_draws, axis=0),
            "beta_ci95": np.quantile(coef_draws, [0.025, 0.975], axis=0),
            "sigma_mean": self.sigma_mean_,
            "tau_mean": self.tau_mean_,
            "lambda_mean": self.lambda_mean_,
            "lambda_tilde_mean": self.lambda_tilde_mean_,
            "c_mean": self.c_mean_,
        }


__all__ = ["RegularizedHorseshoeGibbs"]
