"""Python-native Bayesian Group Lasso with Spike-and-Slab (BGLSS).

This implementation is designed as a practical benchmark baseline that mirrors
the high-level `MBSGS::BGLSS` interface while remaining fully Python-native.

Model (univariate response):
    y | beta, sigma2 ~ N(X beta, sigma2 I)
    beta_g | tau_g2, sigma2 ~ N(0, sigma2 * tau_g2 I_{p_g})
    tau_g2 | z_g ~ Gamma(shape=(p_g+1)/2, rate=lambda_{z_g}^2 / 2)
    z_g | pi ~ Bernoulli(pi)
    pi ~ Beta(a, b) (if pi_prior=True)
    sigma2 ~ IG(alpha, gamma)

The group-lasso kernel arises through the Normal-Gamma hierarchy on each group.
Spike-and-slab is encoded via two shrinkage levels:
    lambda_spike2 >> lambda_slab2.

For sampling tau_g2 | beta_g, sigma2, z_g we use the GIG conditional:
    x ~ GIG(p=1/2, chi=||beta_g||^2/sigma2, psi=lambda_z^2),
implemented via scipy.stats.geninvgauss reparameterization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from scipy.stats import geninvgauss


GroupsLike = Sequence[Sequence[int]]


def _normalize_groups(groups: GroupsLike) -> list[np.ndarray]:
    if not groups:
        raise ValueError("At least one group must be specified.")
    out: list[np.ndarray] = []
    for i, g in enumerate(groups):
        idx = np.asarray(list(g), dtype=int).reshape(-1)
        if idx.size == 0:
            raise ValueError(f"Group {i} is empty.")
        out.append(idx)
    return out


def _check_partition(groups: list[np.ndarray], p: int) -> None:
    mask = np.zeros(p, dtype=int)
    for g in groups:
        if np.any(g < 0) or np.any(g >= p):
            raise ValueError("Group indices out of bounds.")
        mask[g] += 1
    if np.any(mask != 1):
        bad = np.where(mask != 1)[0].tolist()
        raise ValueError(f"Groups must form a partition of features; bad indices: {bad[:20]}")


def _sample_mvn_precision(precision: np.ndarray, rhs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw x ~ N(precision^{-1} rhs, precision^{-1}) stably via Cholesky."""
    p = precision.shape[0]
    jitter = 1e-10
    for _ in range(8):
        try:
            L = np.linalg.cholesky(precision + jitter * np.eye(p))
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    else:
        # Fallback via eigen projection.
        vals, vecs = np.linalg.eigh(0.5 * (precision + precision.T))
        vals = np.maximum(vals, 1e-10)
        L = vecs @ np.diag(np.sqrt(vals))
    # Solve for mean: precision * mu = rhs
    mu = np.linalg.solve(precision + jitter * np.eye(p), rhs)
    z = rng.normal(size=p)
    v = np.linalg.solve(L.T, z)  # covariance draw from precision
    return mu + v


@dataclass
class BGLSSPythonRegression:
    groups: GroupsLike
    niter: int = 6000
    burnin: int = 2000
    seed: int = 2025
    fit_intercept: bool = False
    # Beta prior on inclusion probability pi.
    a: float = 1.0
    b: float = 1.0
    pi_prior: bool = True
    pi_init: float = 0.5
    # IG(alpha, gamma) on sigma^2 (shape-scale style via inverse-gamma draw 1/Gamma(shape, rate)).
    alpha: float = 0.1
    gamma: float = 0.1
    # Spike-slab shrinkage intensities.
    lambda_slab2: float = 0.5
    lambda_spike2: float = 25.0
    # Optional simple EM-style lambda updates.
    update_tau: bool = False
    num_update: int = 100
    niter_update: int = 100
    # Storage controls.
    store_beta_samples: bool = True
    verbose: bool = False

    def __post_init__(self) -> None:
        self.groups = _normalize_groups(self.groups)
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.pos_mean_: Optional[np.ndarray] = None
        self.pos_median_: Optional[np.ndarray] = None
        self.coef_samples_: Optional[np.ndarray] = None
        self.pi_samples_: Optional[np.ndarray] = None
        self.sigma2_samples_: Optional[np.ndarray] = None

    def fit(self, X: Any, y: Any, **fit_kwargs: Any) -> "BGLSSPythonRegression":
        del fit_kwargs
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2D.")
        if y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("X and y dimensions are inconsistent.")
        n, p = X_arr.shape
        _check_partition(self.groups, p)

        # Intercept handling by centering, then recovering intercept from posterior mean beta.
        if self.fit_intercept:
            x_mean = X_arr.mean(axis=0)
            y_mean = float(y_arr.mean())
            Xw = X_arr - x_mean
            yw = y_arr - y_mean
        else:
            x_mean = np.zeros(p, dtype=float)
            y_mean = 0.0
            Xw = X_arr
            yw = y_arr

        rng = np.random.default_rng(int(self.seed))
        G = len(self.groups)
        group_sizes = np.array([g.size for g in self.groups], dtype=float)

        # Initialize states.
        beta = np.zeros(p, dtype=float)
        sigma2 = float(np.var(yw) if np.var(yw) > 1e-8 else 1.0)
        pi = float(np.clip(self.pi_init, 1e-4, 1 - 1e-4))
        z = rng.binomial(1, pi, size=G).astype(int)
        tau2 = np.full(G, 1.0, dtype=float)
        XtX = Xw.T @ Xw
        Xty = Xw.T @ yw

        kept = max(0, int(self.niter) - int(self.burnin))
        beta_draws = np.zeros((kept, p), dtype=float) if self.store_beta_samples and kept > 0 else None
        pi_draws = np.zeros(kept, dtype=float) if kept > 0 else None
        sigma2_draws = np.zeros(kept, dtype=float) if kept > 0 else None

        lambda_slab2 = float(self.lambda_slab2)
        lambda_spike2 = float(self.lambda_spike2)
        eps = 1e-10

        for t in range(int(self.niter)):
            # Optional simplistic Monte Carlo-EM style update of slab lambda^2.
            if self.update_tau and t < int(self.num_update) * int(self.niter_update):
                # Calibrate slab shrinkage to current selected groups only.
                sel = np.where(z == 1)[0]
                if sel.size > 0:
                    m_tau = np.mean(tau2[sel])
                    lambda_slab2 = max(eps, np.mean(group_sizes[sel] + 1.0) / max(m_tau, eps))

            # beta | sigma2, tau2, y
            d = np.zeros(p, dtype=float)
            for g_idx, idx in enumerate(self.groups):
                d[idx] = 1.0 / max(sigma2 * tau2[g_idx], eps)
            precision = XtX / max(sigma2, eps) + np.diag(d)
            rhs = Xty / max(sigma2, eps)
            beta = _sample_mvn_precision(precision, rhs, rng)

            # sigma2 | beta, tau2, y  (IG with shape/scale via inverse gamma draw)
            resid = yw - Xw @ beta
            rss = float(resid @ resid)
            quad = 0.0
            for g_idx, idx in enumerate(self.groups):
                quad += float(beta[idx] @ beta[idx]) / max(tau2[g_idx], eps)
            shape = float(self.alpha) + 0.5 * (n + p)
            rate = float(self.gamma) + 0.5 * (rss + quad)
            sigma2 = 1.0 / rng.gamma(shape=shape, scale=1.0 / max(rate, eps))

            # tau2_g | beta_g, sigma2, z_g  ~ GIG(1/2, chi, psi)
            for g_idx, idx in enumerate(self.groups):
                bg2 = float(beta[idx] @ beta[idx])
                chi = max(bg2 / max(sigma2, eps), eps)
                psi = lambda_slab2 if z[g_idx] == 1 else lambda_spike2
                psi = max(float(psi), eps)
                b_gig = np.sqrt(chi * psi)
                scale = np.sqrt(chi / psi)
                # scipy geninvgauss uses x^(p-1) exp(-b*(x+1/x)/2) with optional scale.
                tau2[g_idx] = float(scale * geninvgauss.rvs(0.5, b_gig, random_state=rng))
                tau2[g_idx] = max(tau2[g_idx], eps)

            # z_g | tau2_g, pi
            for g_idx, pg in enumerate(group_sizes):
                shape_g = 0.5 * (pg + 1.0)
                # Prior kernel for tau2 under slab/spike gamma priors.
                # f(tau2|lambda2) ∝ tau2^{shape-1} exp(-lambda2*tau2/2) lambda2^{shape}
                log_f1 = shape_g * np.log(max(lambda_slab2, eps)) - 0.5 * lambda_slab2 * tau2[g_idx]
                log_f0 = shape_g * np.log(max(lambda_spike2, eps)) - 0.5 * lambda_spike2 * tau2[g_idx]
                log_p1 = np.log(max(pi, eps)) + log_f1
                log_p0 = np.log(max(1.0 - pi, eps)) + log_f0
                m = max(log_p0, log_p1)
                p1 = np.exp(log_p1 - m) / (np.exp(log_p1 - m) + np.exp(log_p0 - m))
                z[g_idx] = int(rng.uniform() < p1)

            # pi | z
            if self.pi_prior:
                pi = float(rng.beta(self.a + np.sum(z), self.b + G - np.sum(z)))
            else:
                pi = float(np.clip(self.pi_init, 1e-4, 1 - 1e-4))

            if t >= int(self.burnin):
                k = t - int(self.burnin)
                if beta_draws is not None:
                    beta_draws[k, :] = beta
                if pi_draws is not None:
                    pi_draws[k] = pi
                if sigma2_draws is not None:
                    sigma2_draws[k] = sigma2

            if self.verbose and (t + 1) % 500 == 0:
                print(f"[BGLSS-Python] iter {t+1}/{self.niter} pi={pi:.3f} sigma2={sigma2:.4f}")

        if beta_draws is None:
            beta_draws = beta.reshape(1, -1)
        self.pos_mean_ = beta_draws.mean(axis=0)
        self.pos_median_ = np.median(beta_draws, axis=0)
        self.coef_ = self.pos_mean_.copy()
        self.coef_samples_ = beta_draws.copy()
        self.pi_samples_ = None if pi_draws is None else pi_draws.copy()
        self.sigma2_samples_ = None if sigma2_draws is None else sigma2_draws.copy()

        if self.fit_intercept:
            self.intercept_ = float(y_mean - x_mean @ self.coef_)
        else:
            self.intercept_ = 0.0
        return self

    def predict(self, X: Any, **predict_kwargs: Any) -> np.ndarray:
        del predict_kwargs
        if self.coef_ is None:
            raise RuntimeError("Model must be fitted before prediction.")
        X_arr = np.asarray(X, dtype=float)
        return X_arr @ self.coef_ + float(self.intercept_)

    def get_posterior_summaries(self) -> dict[str, np.ndarray | float]:
        if self.coef_ is None:
            raise RuntimeError("Model must be fitted before requesting summaries.")
        out: dict[str, np.ndarray | float] = {
            "coef": self.coef_.copy(),
            "intercept": float(self.intercept_),
        }
        if self.pos_mean_ is not None:
            out["pos_mean"] = self.pos_mean_.copy()
        if self.pos_median_ is not None:
            out["pos_median"] = self.pos_median_.copy()
        if self.coef_samples_ is not None:
            out["coef_sd"] = self.coef_samples_.std(axis=0)
        return out

