"""Baseline horseshoe regression models implemented with NumPyro."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import jax.numpy as jnp
from jax import random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

ArrayLike = Any


def _ensure_2d(array: ArrayLike, name: str) -> np.ndarray:
    """Coerce input to (n, p) float32 array."""
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got shape {arr.shape}.")
    return arr


def _ensure_1d(array: ArrayLike, name: str) -> np.ndarray:
    """Coerce input to (n,) float32 array."""
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array; got shape {arr.shape}.")
    return arr


def _thin(arr: Optional[np.ndarray], step: int) -> Optional[np.ndarray]:
    if arr is None or step <= 1:
        return arr
    return arr[::step]


@dataclass
class _BaseHorseshoeRegression:
    """Common machinery for horseshoe-style regression baselines."""

    scale_intercept: float = 10.0
    scale_global: float = 1.0
    nu_global: float = 1.0
    nu_local: float = 1.0
    sigma_scale: float = 1.0
    slab_scale: Optional[float] = None
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 1
    thinning: int = 1
    seed: Optional[int] = None
    target_accept_prob: float = 0.99
    progress_bar: bool = False

    coef_samples_: Optional[np.ndarray] = field(default=None, init=False)
    intercept_samples_: Optional[np.ndarray] = field(default=None, init=False)
    sigma_samples_: Optional[np.ndarray] = field(default=None, init=False)
    tau_samples_: Optional[np.ndarray] = field(default=None, init=False)
    lambda_samples_: Optional[np.ndarray] = field(default=None, init=False)
    coef_: Optional[np.ndarray] = field(default=None, init=False)
    intercept_: Optional[float] = field(default=None, init=False)
    _rng_key: Optional[random.PRNGKeyArray] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.scale_intercept <= 0:
            raise ValueError("scale_intercept must be positive.")
        if self.scale_global <= 0:
            raise ValueError("scale_global must be positive.")
        if self.nu_global <= 0 or self.nu_local <= 0:
            raise ValueError("nu_global and nu_local must be positive.")
        if self.sigma_scale <= 0:
            raise ValueError("sigma_scale must be positive.")
        if self.slab_scale is not None and self.slab_scale <= 0:
            raise ValueError("slab_scale must be positive when provided.")
        if self.num_warmup <= 0 or self.num_samples <= 0:
            raise ValueError("num_warmup and num_samples must be positive integers.")
        if self.num_chains <= 0:
            raise ValueError("num_chains must be a positive integer.")
        if self.thinning <= 0:
            raise ValueError("thinning must be a positive integer.")
        if not 0.0 < self.target_accept_prob < 1.0:
            raise ValueError("target_accept_prob must lie in (0, 1).")

    def _numpyro_model(self, X: jnp.ndarray, y: jnp.ndarray) -> None:
        sigma = numpyro.sample("sigma", dist.HalfCauchy(self.sigma_scale))
        beta0 = numpyro.sample("beta0", dist.Normal(0.0, self.scale_intercept))

        r1_global = numpyro.sample("r1_global", dist.HalfNormal(self.scale_global * sigma))
        r2_global = numpyro.sample(
            "r2_global",
            dist.InverseGamma(0.5 * self.nu_global, 0.5 * self.nu_global),
        )
        tau = numpyro.deterministic("tau", r1_global * jnp.sqrt(r2_global))

        p = X.shape[1]
        r1_local = numpyro.sample(
            "r1_local",
            dist.HalfNormal(jnp.ones((p,))).to_event(1),
        )
        inv_gamma_local = dist.InverseGamma(0.5 * self.nu_local, 0.5 * self.nu_local)
        r2_local = numpyro.sample(
            "r2_local",
            inv_gamma_local.expand((p,)).to_event(1),
        )
        lambda_raw = r1_local * jnp.sqrt(r2_local)
        lambda_effective = numpyro.deterministic(
            "lambda",
            self._regularize_lambda(lambda_raw, tau),
        )

        z = numpyro.sample("z", dist.Normal(jnp.zeros((p,)), 1.0).to_event(1))
        beta = numpyro.deterministic("beta", z * lambda_effective * tau)
        mean = beta0 + X @ beta
        numpyro.sample("y", dist.Normal(mean, sigma), obs=y)

    def _regularize_lambda(self, lambda_raw: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
        if self.slab_scale is None:
            return lambda_raw
        c2 = float(self.slab_scale) ** 2
        lam2 = lambda_raw ** 2
        tau2 = tau ** 2
        denom = c2 + tau2 * lam2 + 1e-18
        lambda_tilde_sq = (c2 * lam2) / denom
        return jnp.sqrt(lambda_tilde_sq)

    def fit(self, X: ArrayLike, y: ArrayLike) -> "_BaseHorseshoeRegression":
        X_arr = _ensure_2d(X, "X")
        y_arr = _ensure_1d(y, "y")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have matching number of rows.")

        rng_seed = 0 if self.seed is None else int(self.seed)
        self._rng_key = random.PRNGKey(rng_seed)
        kernel = NUTS(
            self._numpyro_model,
            target_accept_prob=self.target_accept_prob,
            dense_mass=True,
        )
        mcmc = MCMC(
            kernel,
            num_warmup=int(self.num_warmup),
            num_samples=int(self.num_samples),
            num_chains=int(self.num_chains),
            progress_bar=self.progress_bar,
            chain_method="sequential",
        )
        mcmc.run(self._rng_key, jnp.asarray(X_arr), jnp.asarray(y_arr))
        samples = mcmc.get_samples(group_by_chain=False)
        self._store_samples(samples)
        return self

    def _store_samples(self, samples: Dict[str, jnp.ndarray]) -> None:
        def _convert(name: str) -> Optional[np.ndarray]:
            if name not in samples:
                return None
            arr = np.asarray(samples[name], dtype=np.float64)
            return _thin(arr, self.thinning)

        self.coef_samples_ = _convert("beta")
        if self.coef_samples_ is None:
            raise RuntimeError("NumPyro model did not produce beta samples.")
        self.intercept_samples_ = _convert("beta0")
        self.sigma_samples_ = _convert("sigma")
        self.tau_samples_ = _convert("tau")
        self.lambda_samples_ = _convert("lambda")

        self.coef_ = self.coef_samples_.mean(axis=0)
        if self.intercept_samples_ is not None:
            self.intercept_ = float(self.intercept_samples_.mean())
        else:
            self.intercept_ = 0.0

    def predict(self, X: ArrayLike) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        coef = np.asarray(self.coef_, dtype=np.float64)
        intercept = float(self.intercept_ or 0.0)
        X_arr = _ensure_2d(X, "X")
        return X_arr.astype(np.float64) @ coef + intercept

    def get_posterior_summaries(self) -> Dict[str, Any]:
        if self.coef_samples_ is None:
            raise RuntimeError("Model must be fitted before requesting summaries.")
        summaries: Dict[str, Any] = {
            "coef_mean": self.coef_samples_.mean(axis=0),
            "coef_median": np.median(self.coef_samples_, axis=0),
            "coef_ci95": np.quantile(self.coef_samples_, [0.025, 0.975], axis=0),
        }
        if self.sigma_samples_ is not None:
            summaries["sigma_mean"] = float(self.sigma_samples_.mean())
        if self.tau_samples_ is not None:
            summaries["tau_mean"] = float(self.tau_samples_.mean())
        if self.lambda_samples_ is not None:
            summaries["lambda_mean"] = self.lambda_samples_.mean(axis=0)
        return summaries


@dataclass
class HorseshoeRegression(_BaseHorseshoeRegression):
    """Standard horseshoe regression baseline."""

    pass


@dataclass
class RegularizedHorseshoeRegression(_BaseHorseshoeRegression):
    """Regularized horseshoe regression baseline (Piironen & Vehtari, 2017)."""

    slab_scale: float = 1.0
