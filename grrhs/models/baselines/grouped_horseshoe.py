from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.random import Generator, default_rng
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from grrhs.inference.woodbury import beta_sample_woodbury, beta_sample_cholesky


_MIN_POS = 1e-10


def _sample_invgamma(alpha: float, beta: float, rng: Generator) -> float:
    """Sample InvGamma(alpha, beta) under shape-scale parameterization."""
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("InvGamma requires alpha > 0 and beta > 0.")
    z = rng.gamma(shape=float(alpha), scale=1.0 / float(beta))
    return float(1.0 / max(z, _MIN_POS))


def _normalize_groups(groups: Sequence[Sequence[int]], p: int) -> List[List[int]]:
    if not groups:
        raise ValueError("At least one group must be provided.")
    normalised: List[List[int]] = []
    covered = np.zeros(p, dtype=bool)
    for gid, members in enumerate(groups):
        idx = [int(v) for v in members]
        if not idx:
            raise ValueError(f"Group {gid} is empty.")
        arr = np.asarray(idx, dtype=int)
        if np.any(arr < 0) or np.any(arr >= p):
            raise ValueError(f"Group {gid} contains indices outside [0, {p}).")
        covered[arr] = True
        normalised.append(idx)
    if not np.all(covered):
        missing = np.nonzero(~covered)[0].tolist()
        raise ValueError(f"Some features are not assigned to any group: {missing}")
    return normalised


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


def _sample_beta_conditional(
    *,
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    XtX: np.ndarray,
    Xty: np.ndarray,
    sigma2: float,
    prior_var: np.ndarray,
    jitter: float,
    rng: Generator,
) -> np.ndarray:
    """Sample β from its Gaussian full conditional.

    Uses Bhattacharya (2016) Woodbury path (O(n²p)) when X and y are supplied
    and n < p; falls back to Cholesky on the p×p precision otherwise.

    The HBGHS prior is β_j ~ N(0, σ²·prior_var_j), so the posterior is
    N((XᵀX + diag(prior_prec))⁻¹ Xᵀy, σ²·(XᵀX + diag(prior_prec))⁻¹).
    The Woodbury path works in the equivalent scaled space (X' = X, y' = y,
    prior D = prior_var), recovering the same distribution.
    """
    p = int(Xty.shape[0])
    pv = np.asarray(prior_var, dtype=float)
    if X is not None and y is not None and int(y.shape[0]) < p:
        return beta_sample_woodbury(
            np.asarray(X, dtype=float),
            np.asarray(y, dtype=float),
            float(sigma2),
            pv,
            rng,
            jitter=float(jitter),
        )
    # Cholesky on p×p precision: precision = XᵀX + diag(1/prior_var)
    # (σ² factored out; noise scaled by √σ² to match N(μ, σ²·precision⁻¹))
    prior_prec = 1.0 / np.maximum(pv, jitter)
    precision = np.asarray(XtX, dtype=float) + np.diag(prior_prec)
    if jitter > 0.0:
        precision += float(jitter) * np.eye(precision.shape[0], dtype=float)
    chol = cho_factor(precision, lower=True, check_finite=False)
    mean = cho_solve(chol, np.asarray(Xty, dtype=float), check_finite=False)
    z = rng.standard_normal(precision.shape[0])
    noise = solve_triangular(chol[0], z, lower=bool(chol[1]), check_finite=False)
    return np.asarray(mean + math.sqrt(max(float(sigma2), jitter)) * noise, dtype=float)


@dataclass
class GroupedHorseshoePlus:
    """Hierarchical Bayesian Grouped Horseshoe (HBGHS) from Xu et al. (2016).

    Prior structure:
        beta_j | sigma^2, tau^2, lambda_g, delta_j
            ~ N(0, sigma^2 * tau^2 * lambda_{g(j)}^2 * delta_j^2)
        lambda_g ~ C+(0, group_scale_prior),   g = 1,...,G   [group shrinkage]
        delta_j  ~ C+(0, local_scale_prior),   j = 1,...,p   [within-group shrinkage]
        tau      ~ C+(0, tau0)                               [global shrinkage]
        sigma^2  ~ 1/sigma^2 d sigma^2

    Full conditionals follow the Makalic-Schmidt inverse-gamma augmentation.
    """

    fit_intercept: bool = True
    tau0: float = 1.0
    group_scale_prior: float = 1.0
    local_scale_prior: float = 1.0
    iters: int = 3000
    burnin: int = 1500
    thin: int = 1
    seed: int = 42
    num_chains: int = 1
    jitter: float = 1e-8
    progress_bar: bool = False

    coef_samples_: Optional[np.ndarray] = field(default=None, init=False)
    intercept_samples_: Optional[np.ndarray] = field(default=None, init=False)
    sigma2_samples_: Optional[np.ndarray] = field(default=None, init=False)
    sigma_samples_: Optional[np.ndarray] = field(default=None, init=False)
    tau_samples_: Optional[np.ndarray] = field(default=None, init=False)
    lambda_samples_: Optional[np.ndarray] = field(default=None, init=False)
    group_lambda_samples_: Optional[np.ndarray] = field(default=None, init=False)
    coef_: Optional[np.ndarray] = field(default=None, init=False)
    coef_mean_: Optional[np.ndarray] = field(default=None, init=False)
    tau_mean_: Optional[float] = field(default=None, init=False)
    sigma_mean_: Optional[float] = field(default=None, init=False)
    lambda_mean_: Optional[np.ndarray] = field(default=None, init=False)
    group_lambda_mean_: Optional[np.ndarray] = field(default=None, init=False)
    intercept_: float = field(default=0.0, init=False)
    groups_: Optional[List[List[int]]] = field(default=None, init=False)
    group_id_: Optional[np.ndarray] = field(default=None, init=False)
    group_sizes_: Optional[np.ndarray] = field(default=None, init=False)
    sampler_diagnostics_: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.tau0 <= 0.0:
            raise ValueError("tau0 must be positive.")
        if self.group_scale_prior <= 0.0:
            raise ValueError("group_scale_prior must be positive.")
        if self.local_scale_prior <= 0.0:
            raise ValueError("local_scale_prior must be positive.")
        if int(self.iters) <= 0:
            raise ValueError("iters must be positive.")
        if int(self.burnin) < 0 or int(self.burnin) >= int(self.iters):
            raise ValueError("burnin must satisfy 0 <= burnin < iters.")
        if int(self.thin) <= 0:
            raise ValueError("thin must be positive.")
        if int(self.num_chains) <= 0:
            raise ValueError("num_chains must be positive.")
        if float(self.jitter) <= 0.0:
            raise ValueError("jitter must be positive.")

    @staticmethod
    def _prepare_data(
        X: np.ndarray,
        y: np.ndarray,
        *,
        fit_intercept: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("X and y must have compatible first dimensions.")
        if fit_intercept:
            x_mean = X_arr.mean(axis=0)
            y_mean = float(y_arr.mean())
        else:
            x_mean = np.zeros(X_arr.shape[1], dtype=float)
            y_mean = 0.0
        X_centered = X_arr - x_mean
        x_scale = X_centered.std(axis=0, ddof=0)
        x_scale = np.where(x_scale < 1e-8, 1.0, x_scale)
        X_std = X_centered / x_scale
        y_centered = y_arr - y_mean
        return X_std, y_centered, x_mean, x_scale, y_mean

    def _sample_single_chain(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        groups: Sequence[Sequence[int]],
        seed: int,
    ) -> Dict[str, np.ndarray]:
        rng = default_rng(int(seed))
        X_std, y_ctr, x_mean, x_scale, y_mean = self._prepare_data(X, y, fit_intercept=bool(self.fit_intercept))
        n, p = X_std.shape
        groups_norm = _normalize_groups(groups, p)
        G = len(groups_norm)
        group_id = np.empty(p, dtype=int)
        group_sizes = np.zeros(G, dtype=int)
        for gid, members in enumerate(groups_norm):
            idx = np.asarray(members, dtype=int)
            group_id[idx] = gid
            group_sizes[gid] = int(idx.size)

        XtX = X_std.T @ X_std
        Xty = X_std.T @ y_ctr

        try:
            beta = np.linalg.solve(XtX + 1e-3 * np.eye(p), Xty)
        except np.linalg.LinAlgError:
            beta = np.zeros(p, dtype=float)
        sigma2 = max(float(np.var(y_ctr - X_std @ beta)), float(self.jitter))
        tau2 = max(float(self.tau0) ** 2, float(self.jitter))
        tau_aux = 1.0
        group_lambda2 = np.ones(G, dtype=float)
        group_aux = np.ones(G, dtype=float)
        local_delta2 = np.ones(p, dtype=float)
        local_aux = np.ones(p, dtype=float)

        kept = max(0, (int(self.iters) - int(self.burnin) + int(self.thin) - 1) // int(self.thin))
        beta_draws = np.zeros((kept, p), dtype=float)
        intercept_draws = np.zeros(kept, dtype=float)
        sigma2_draws = np.zeros(kept, dtype=float)
        tau_draws = np.zeros(kept, dtype=float)
        group_draws = np.zeros((kept, G), dtype=float)
        local_draws = np.ones((kept, p), dtype=float)

        keep_i = 0
        iterator = range(int(self.iters))
        if bool(self.progress_bar):
            from grrhs.utils.logging_utils import progress
            iterator = progress(iterator, total=int(self.iters), desc="Grouped HS+ Gibbs")

        for it in iterator:
            # beta | sigma^2, tau^2, lambda_g, delta_j
            prior_var = tau2 * group_lambda2[group_id] * local_delta2
            beta = _sample_beta_conditional(
                X=X_std,
                y=y_ctr,
                XtX=XtX,
                Xty=Xty,
                sigma2=sigma2,
                prior_var=prior_var,
                jitter=float(self.jitter),
                rng=rng,
            )

            # sigma^2 | beta, tau^2, lambda_g, delta_j
            resid = y_ctr - X_std @ beta
            prior_quad = float(np.sum((beta * beta) / np.maximum(prior_var, self.jitter)))
            n_eff = max(n - int(bool(self.fit_intercept)), 1)
            sigma2 = _sample_invgamma(
                alpha=0.5 * (n_eff + p),
                beta=0.5 * max(float(resid @ resid) + prior_quad, self.jitter),
                rng=rng,
            )

            # tau^2 | beta, sigma^2, lambda_g, delta_j  [global shrinkage]
            tau_rate = 0.5 * float(np.sum((beta * beta) / np.maximum(group_lambda2[group_id] * local_delta2, self.jitter))) / max(sigma2, self.jitter)
            tau2 = _sample_invgamma(alpha=0.5 * (p + 1), beta=max(tau_rate + (1.0 / max(tau_aux, self.jitter)), self.jitter), rng=rng)
            tau_aux = _sample_invgamma(alpha=1.0, beta=(1.0 / tau2) + (1.0 / (float(self.tau0) ** 2)), rng=rng)

            # lambda_g^2 | beta, sigma^2, tau^2, delta_j  [group shrinkage — vectorized]
            weighted_b2 = beta ** 2 / np.maximum(local_delta2, self.jitter)
            group_beta_sq = np.bincount(group_id, weights=weighted_b2, minlength=G)
            rates_g = (0.5 * group_beta_sq / max(sigma2 * tau2, self.jitter)
                       + 1.0 / np.maximum(group_aux, self.jitter))
            alphas_g = 0.5 * (group_sizes + 1)
            group_lambda2 = 1.0 / rng.gamma(
                shape=np.maximum(alphas_g, self.jitter),
                scale=1.0 / np.maximum(rates_g, self.jitter),
            )
            group_aux = 1.0 / rng.gamma(
                shape=np.ones(G),
                scale=1.0 / np.maximum(
                    1.0 / np.maximum(group_lambda2, self.jitter)
                    + 1.0 / float(self.group_scale_prior) ** 2,
                    self.jitter,
                ),
            )

            # delta_j^2 | beta, sigma^2, tau^2, lambda_g  [within-group — vectorized]
            group_scales_j = group_lambda2[group_id]
            rates_j = (0.5 * beta ** 2 / np.maximum(sigma2 * tau2 * group_scales_j, self.jitter)
                       + 1.0 / np.maximum(local_aux, self.jitter))
            local_delta2 = 1.0 / rng.gamma(
                shape=np.ones(p),
                scale=1.0 / np.maximum(rates_j, self.jitter),
            )
            local_aux = 1.0 / rng.gamma(
                shape=np.ones(p),
                scale=1.0 / np.maximum(
                    1.0 / np.maximum(local_delta2, self.jitter)
                    + 1.0 / float(self.local_scale_prior) ** 2,
                    self.jitter,
                ),
            )

            if it >= int(self.burnin) and ((it - int(self.burnin)) % int(self.thin) == 0):
                beta_orig = beta / x_scale
                intercept = float(y_mean - np.dot(x_mean, beta_orig)) if bool(self.fit_intercept) else 0.0
                beta_draws[keep_i] = beta_orig
                intercept_draws[keep_i] = intercept
                sigma2_draws[keep_i] = sigma2
                tau_draws[keep_i] = math.sqrt(max(tau2, self.jitter))
                group_draws[keep_i] = np.sqrt(np.maximum(group_lambda2, self.jitter))
                local_draws[keep_i] = np.sqrt(np.maximum(local_delta2, self.jitter))
                keep_i += 1

        return {
            "coef_samples": beta_draws,
            "intercept_samples": intercept_draws,
            "sigma2_samples": sigma2_draws,
            "tau_samples": tau_draws,
            "group_lambda_samples": group_draws,
            "local_scale_samples": local_draws,
            "groups": [list(g) for g in groups_norm],
            "group_id": group_id,
            "group_sizes": group_sizes,
        }

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        groups: Optional[Sequence[Sequence[int]]] = None,
    ) -> "GroupedHorseshoePlus":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("X and y must have compatible first dimensions.")
        groups_use = [[j] for j in range(X_arr.shape[1])] if groups is None else [list(map(int, g)) for g in groups]

        start = time.perf_counter()
        chain_results = [
            self._sample_single_chain(X_arr, y_arr, groups=groups_use, seed=int(self.seed) + chain_idx)
            for chain_idx in range(int(self.num_chains))
        ]
        runtime_sec = max(time.perf_counter() - start, 1e-12)

        if int(self.num_chains) == 1:
            lead = chain_results[0]
            self.coef_samples_ = lead["coef_samples"]
            self.intercept_samples_ = lead["intercept_samples"]
            self.sigma2_samples_ = lead["sigma2_samples"]
            self.tau_samples_ = lead["tau_samples"]
            self.group_lambda_samples_ = lead["group_lambda_samples"]
            self.lambda_samples_ = lead["local_scale_samples"]
            self.groups_ = lead["groups"]
            self.group_id_ = np.asarray(lead["group_id"], dtype=int)
            self.group_sizes_ = np.asarray(lead["group_sizes"], dtype=int)
        else:
            self.coef_samples_ = np.stack([item["coef_samples"] for item in chain_results], axis=0)
            self.intercept_samples_ = np.stack([item["intercept_samples"] for item in chain_results], axis=0)
            self.sigma2_samples_ = np.stack([item["sigma2_samples"] for item in chain_results], axis=0)
            self.tau_samples_ = np.stack([item["tau_samples"] for item in chain_results], axis=0)
            self.group_lambda_samples_ = np.stack([item["group_lambda_samples"] for item in chain_results], axis=0)
            self.lambda_samples_ = np.stack([item["local_scale_samples"] for item in chain_results], axis=0)
            self.groups_ = chain_results[0]["groups"]
            self.group_id_ = np.asarray(chain_results[0]["group_id"], dtype=int)
            self.group_sizes_ = np.asarray(chain_results[0]["group_sizes"], dtype=int)

        self.sigma_samples_ = np.sqrt(np.maximum(np.asarray(self.sigma2_samples_, dtype=float), 0.0))
        coef_draws = _flatten_param_draws(self.coef_samples_)
        intercept_draws = _flatten_scalar_draws(self.intercept_samples_)
        tau_draws = _flatten_scalar_draws(self.tau_samples_)
        sigma_draws = _flatten_scalar_draws(self.sigma_samples_)
        local_draws = _flatten_param_draws(self.lambda_samples_)
        group_draws = _flatten_param_draws(self.group_lambda_samples_)

        self.coef_mean_ = None if coef_draws is None else coef_draws.mean(axis=0)
        self.coef_ = None if self.coef_mean_ is None else self.coef_mean_.copy()
        self.intercept_ = 0.0 if intercept_draws is None else float(intercept_draws.mean())
        self.tau_mean_ = None if tau_draws is None else float(tau_draws.mean())
        self.sigma_mean_ = None if sigma_draws is None else float(sigma_draws.mean())
        self.lambda_mean_ = None if local_draws is None else local_draws.mean(axis=0)
        self.group_lambda_mean_ = None if group_draws is None else group_draws.mean(axis=0)

        kept = 0 if self.coef_samples_ is None else int(
            np.asarray(self.coef_samples_).shape[-2]
            if np.asarray(self.coef_samples_).ndim >= 3
            else np.asarray(self.coef_samples_).shape[0]
        )
        self.sampler_diagnostics_ = {
            "backend": "grouped_horseshoe_plus_gibbs",
            "runtime_sec": float(runtime_sec),
            "num_chains": int(self.num_chains),
            "kept_draws_per_chain": int(kept),
            "model": "HBGHS (Xu et al. 2016)",
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
        return X_arr @ np.asarray(self.coef_, dtype=float) + float(self.intercept_)

    def get_posterior_summaries(self) -> Dict[str, Any]:
        if self.coef_samples_ is None:
            raise RuntimeError("Model must be fitted before requesting summaries.")
        coef_draws = _flatten_param_draws(self.coef_samples_)
        if coef_draws is None:
            raise RuntimeError("Posterior coefficient draws are unavailable.")
        return {
            "coef_mean": coef_draws.mean(axis=0),
            "coef_median": np.median(coef_draws, axis=0),
            "coef_ci95": np.quantile(coef_draws, [0.025, 0.975], axis=0),
            "tau_mean": self.tau_mean_,
            "sigma_mean": self.sigma_mean_,
            "group_lambda_mean": self.group_lambda_mean_,
            "local_scale_mean": self.lambda_mean_,
        }


__all__ = ["GroupedHorseshoePlus"]
