"""Unified collection of baseline models used across experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from sklearn.linear_model import LogisticRegression as _SklearnLogisticRegression
from grrhs.models.grrhs_gibbs import GRRHS_Gibbs
try:
    from cmdstanpy import CmdStanModel as _CmdStanModel
except Exception:  # pragma: no cover - optional dependency
    _CmdStanModel = None  # type: ignore[assignment]

try:
    from skglm import GeneralizedLinearEstimator as _SkglmGeneralizedLinearEstimator
    from skglm.datafits import QuadraticGroup as _SkglmQuadraticGroup
    from skglm.penalties.block_separable import WeightedL1GroupL2 as _SkglmWeightedL1GroupL2
    from skglm.solvers import GroupBCD as _SkglmGroupBCD
    from skglm.utils.data import grp_converter as _skglm_grp_converter
except ImportError as exc:  # pragma: no cover - handled at runtime
    _SKGLM_IMPORT_ERROR = exc
    _SkglmGeneralizedLinearEstimator = None
    _SkglmQuadraticGroup = None
    _SkglmWeightedL1GroupL2 = None
    _SkglmGroupBCD = None
    _skglm_grp_converter = None
else:  # pragma: no cover - exercised when skglm is present
    _SKGLM_IMPORT_ERROR = None

__all__ = [
    "LogisticRegressionClassifier",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "SparseGroupLasso",
    "HorseshoeRegression",
    "RegularizedHorseshoeRegression",
]

GroupsLike = Sequence[Sequence[int]]
WeightsLike = Optional[Sequence[float]]
FeatureWeightsLike = Optional[Sequence[float]]
ArrayLike = Any
_CMDSTAN_MODEL_CACHE: dict[str, Any] = {}


class _SkglmRegressorBase:
    """Shared utilities for thin wrappers around skglm estimators."""

    def __init__(self) -> None:
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray | float] = None
        self._estimator: Any | None = None

    def _store_fitted_estimator(self, estimator: Any) -> None:
        coef = getattr(estimator, "coef_", None)
        if coef is None:
            raise AttributeError(f"{type(estimator).__name__} does not expose 'coef_'.")
        coef_arr = np.asarray(coef, dtype=float)
        self.coef_ = coef_arr.copy()
        intercept = getattr(estimator, "intercept_", 0.0)
        intercept_arr = np.asarray(intercept, dtype=float)
        if intercept_arr.ndim == 0:
            intercept_value: np.ndarray | float = float(intercept_arr)
        else:
            intercept_value = intercept_arr.copy()
        self.intercept_ = intercept_value
        self._estimator = estimator

    def predict(self, X: Any, **predict_kwargs: Any) -> Any:
        if self._estimator is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        return self._estimator.predict(X, **predict_kwargs)

    def get_posterior_summaries(self) -> dict[str, np.ndarray | float]:
        if self.coef_ is None:
            raise RuntimeError("Model must be fitted before requesting summaries.")
        intercept = self.intercept_
        if intercept is None:
            intercept_summary: np.ndarray | float = 0.0
        elif isinstance(intercept, np.ndarray):
            intercept_summary = intercept.copy()
        else:
            intercept_summary = float(intercept)
        return {"coef": self.coef_.copy(), "intercept": intercept_summary}

    def get_estimator(self) -> Any:
        if self._estimator is None:
            raise RuntimeError("Model must be fitted before accessing the estimator.")
        return self._estimator

    def __getattr__(self, name: str) -> Any:
        estimator = object.__getattribute__(self, "_estimator")
        if estimator is None:
            raise AttributeError(
                f"{type(self).__name__} has no attribute '{name}' (model not fitted yet)."
            )
        return getattr(estimator, name)


def _require_skglm() -> None:
    if _SKGLM_IMPORT_ERROR is not None:
        raise ImportError(
            "skglm-based baselines require the 'skglm' package. "
            "Install it with `pip install skglm`."
        ) from _SKGLM_IMPORT_ERROR


def _normalize_groups(groups: GroupsLike) -> list[list[int]]:
    if not groups:
        raise ValueError("At least one group must be specified.")
    normalized: list[list[int]] = []
    for idx, group in enumerate(groups):
        if isinstance(group, (str, bytes)):
            raise TypeError(f"Group {idx} must be an iterable of indices, not a string.")
        try:
            block = [int(member) for member in group]  # type: ignore[arg-type]
        except TypeError as exc:  # pragma: no cover - defensive
            raise TypeError(
                f"Group {idx} must be an iterable of indices; received {type(group)}."
            ) from exc
        if not block:
            raise ValueError(f"Group {idx} is empty.")
        normalized.append(block)
    return normalized


def _coerce_group_weights(groups: Sequence[Sequence[int]], weights: WeightsLike) -> np.ndarray:
    if weights is None:
        sizes = np.array([len(group) for group in groups], dtype=float)
        return np.sqrt(sizes)
    weights_arr = np.asarray(list(weights), dtype=float)
    if weights_arr.ndim != 1 or weights_arr.shape[0] != len(groups):
        raise ValueError(
            f"Expected {len(groups)} group weights, received shape {weights_arr.shape}."
        )
    if np.any(weights_arr < 0):
        raise ValueError("Group weights must be non-negative.")
    return weights_arr


def _ensure_groups_cover_features(groups: Sequence[Sequence[int]], n_features: int) -> None:
    mask = np.zeros(n_features, dtype=bool)
    for group in groups:
        indices = np.asarray(group, dtype=int)
        if np.any(indices < 0) or np.any(indices >= n_features):
            raise ValueError(f"Group indices {indices.tolist()} out of bounds for {n_features} features.")
        mask[indices] = True
    if not np.all(mask):
        missing = np.nonzero(~mask)[0]
        raise ValueError(f"Some features are not assigned to any group: {missing.tolist()}.")


def _coerce_feature_weights(n_features: int, weights: FeatureWeightsLike) -> np.ndarray:
    if weights is None:
        return np.ones(n_features, dtype=float)
    weights_arr = np.asarray(list(weights), dtype=float)
    if weights_arr.ndim != 1 or weights_arr.shape[0] != n_features:
        raise ValueError(
            f"Expected {n_features} feature weights, received shape {weights_arr.shape}."
        )
    if np.any(weights_arr < 0):
        raise ValueError("Feature weights must be non-negative.")
    return weights_arr


@dataclass
class LogisticRegressionClassifier:
    penalty: str = "l2"
    C: float = 1.0
    fit_intercept: bool = True
    solver: str = "lbfgs"
    max_iter: int = 200
    tol: float = 1e-4
    class_weight: Any = None
    l1_ratio: Optional[float] = None
    multi_class: str = "auto"
    intercept_scaling: float = 1.0
    n_jobs: Optional[int] = None
    warm_start: bool = False
    verbose: int = 0
    random_state: Optional[int] = None

    def __post_init__(self) -> None:
        self._estimator: Optional[_SklearnLogisticRegression] = None
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray | float] = None

    def _estimator_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "penalty": str(self.penalty),
            "C": float(self.C),
            "fit_intercept": bool(self.fit_intercept),
            "solver": str(self.solver),
            "max_iter": int(self.max_iter),
            "tol": float(self.tol),
            "intercept_scaling": float(self.intercept_scaling),
            "warm_start": bool(self.warm_start),
            "verbose": int(self.verbose),
        }
        if self.class_weight is not None:
            params["class_weight"] = self.class_weight
        multi_class_val = None if self.multi_class is None else str(self.multi_class)
        if multi_class_val not in {None, "auto"}:
            params["multi_class"] = multi_class_val
        if self.random_state is not None:
            params["random_state"] = int(self.random_state)
        if self.n_jobs is not None:
            params["n_jobs"] = int(self.n_jobs)
        if self.l1_ratio is not None:
            params["l1_ratio"] = float(self.l1_ratio)
        return params

    def fit(self, X: Any, y: Any, **fit_kwargs: Any) -> "LogisticRegressionClassifier":
        estimator = _SklearnLogisticRegression(**self._estimator_params())
        fitted = estimator.fit(X, y, **fit_kwargs)
        self._estimator = fitted
        coef = np.asarray(fitted.coef_, dtype=float)
        self.coef_ = coef.copy()
        intercept_arr = np.asarray(fitted.intercept_, dtype=float)
        if intercept_arr.ndim == 0:
            self.intercept_ = float(intercept_arr)
        else:
            self.intercept_ = intercept_arr.copy()
        return self

    def predict(self, X: Any, **predict_kwargs: Any) -> Any:
        if self._estimator is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        return self._estimator.predict(X, **predict_kwargs)

    def predict_proba(self, X: Any, **predict_kwargs: Any) -> np.ndarray:
        if self._estimator is None:
            raise RuntimeError("Model must be fitted before calling predict_proba().")
        return self._estimator.predict_proba(X, **predict_kwargs)

    def decision_function(self, X: Any, **predict_kwargs: Any) -> Any:
        if self._estimator is None:
            raise RuntimeError("Model must be fitted before calling decision_function().")
        return self._estimator.decision_function(X, **predict_kwargs)

    def get_estimator(self) -> _SklearnLogisticRegression:
        if self._estimator is None:
            raise RuntimeError("Model must be fitted before accessing the estimator.")
        return self._estimator


@dataclass
class Ridge(_SkglmRegressorBase):
    alpha: float = 1.0
    fit_intercept: bool = False
    max_iter: int = 50
    max_epochs: int = 50_000
    p0: int = 10
    tol: float = 1e-4
    warm_start: bool = True
    ws_strategy: str = "subdiff"
    verbose: int = 0
    positive: bool = False

    def __post_init__(self) -> None:
        super().__init__()

    def fit(self, X: Any, y: Any, **fit_kwargs: Any) -> "Ridge":
        _require_skglm()
        from skglm import ElasticNet as _SkglmElasticNet

        estimator = _SkglmElasticNet(
            alpha=float(self.alpha) * 2.0,
            l1_ratio=0.0,
            max_iter=int(self.max_iter),
            max_epochs=int(self.max_epochs),
            p0=int(self.p0),
            tol=float(self.tol),
            fit_intercept=bool(self.fit_intercept),
            warm_start=bool(self.warm_start),
            ws_strategy=str(self.ws_strategy),
            verbose=self.verbose,
            positive=bool(self.positive),
        )
        fitted = estimator.fit(X, y, **fit_kwargs)
        self._store_fitted_estimator(fitted)
        return self


@dataclass
class Lasso(_SkglmRegressorBase):
    alpha: float = 1.0
    fit_intercept: bool = False
    max_iter: int = 50
    max_epochs: int = 50_000
    p0: int = 10
    tol: float = 1e-4
    warm_start: bool = True
    ws_strategy: str = "subdiff"
    verbose: int = 0
    positive: bool = False

    def __post_init__(self) -> None:
        super().__init__()

    def fit(self, X: Any, y: Any, **fit_kwargs: Any) -> "Lasso":
        _require_skglm()
        from skglm import Lasso as _SkglmLasso

        estimator = _SkglmLasso(
            alpha=float(self.alpha),
            max_iter=int(self.max_iter),
            max_epochs=int(self.max_epochs),
            p0=int(self.p0),
            tol=float(self.tol),
            fit_intercept=bool(self.fit_intercept),
            warm_start=bool(self.warm_start),
            ws_strategy=str(self.ws_strategy),
            verbose=self.verbose,
            positive=bool(self.positive),
        )
        fitted = estimator.fit(X, y, **fit_kwargs)
        self._store_fitted_estimator(fitted)
        return self


@dataclass
class ElasticNet(_SkglmRegressorBase):
    alpha: float = 1.0
    l1_ratio: float = 0.5
    fit_intercept: bool = False
    max_iter: int = 50
    max_epochs: int = 50_000
    p0: int = 10
    tol: float = 1e-4
    warm_start: bool = True
    ws_strategy: str = "subdiff"
    verbose: int = 0
    positive: bool = False

    def __post_init__(self) -> None:
        super().__init__()

    def fit(self, X: Any, y: Any, **fit_kwargs: Any) -> "ElasticNet":
        _require_skglm()
        from skglm import ElasticNet as _SkglmElasticNet

        estimator = _SkglmElasticNet(
            alpha=float(self.alpha),
            l1_ratio=float(self.l1_ratio),
            max_iter=int(self.max_iter),
            max_epochs=int(self.max_epochs),
            p0=int(self.p0),
            tol=float(self.tol),
            fit_intercept=bool(self.fit_intercept),
            warm_start=bool(self.warm_start),
            ws_strategy=str(self.ws_strategy),
            verbose=self.verbose,
            positive=bool(self.positive),
        )
        fitted = estimator.fit(X, y, **fit_kwargs)
        self._store_fitted_estimator(fitted)
        return self


@dataclass
class SparseGroupLasso(_SkglmRegressorBase):
    groups: GroupsLike
    alpha: float = 1.0
    l1_ratio: float = 0.5
    group_weights: WeightsLike = None
    feature_weights: FeatureWeightsLike = None
    fit_intercept: bool = False
    max_iter: int = 2_000
    max_epochs: int = 50_000
    p0: int = 10
    tol: float = 1e-6
    warm_start: bool = True
    ws_strategy: str = "fixpoint"
    verbose: int = 0

    def __post_init__(self) -> None:
        super().__init__()
        if not 0.0 <= float(self.l1_ratio) <= 1.0:
            raise ValueError("SparseGroupLasso requires 0 <= l1_ratio <= 1.")
        self.groups = _normalize_groups(self.groups)
        self._group_weights = _coerce_group_weights(self.groups, self.group_weights)

    def fit(self, X: Any, y: Any, **fit_kwargs: Any) -> "SparseGroupLasso":
        _require_skglm()
        try:
            n_features = int(X.shape[1])
        except Exception as exc:
            raise TypeError("Input design matrix must expose `shape[1]` (number of features).") from exc

        _ensure_groups_cover_features(self.groups, n_features)
        grp_indices, grp_ptr = _skglm_grp_converter(self.groups, n_features)
        group_weights = self._group_weights
        feature_weights = _coerce_feature_weights(n_features, self.feature_weights)

        weights_groups = (1.0 - float(self.l1_ratio)) * group_weights
        weights_features = float(self.l1_ratio) * feature_weights

        penalty = _SkglmWeightedL1GroupL2(
            alpha=float(self.alpha),
            weights_groups=weights_groups,
            weights_features=weights_features,
            grp_ptr=grp_ptr,
            grp_indices=grp_indices,
        )
        datafit = _SkglmQuadraticGroup(grp_ptr=grp_ptr, grp_indices=grp_indices)
        solver = _SkglmGroupBCD(
            max_iter=int(self.max_iter),
            max_epochs=int(self.max_epochs),
            p0=int(self.p0),
            tol=float(self.tol),
            fit_intercept=bool(self.fit_intercept),
            warm_start=bool(self.warm_start),
            ws_strategy=str(self.ws_strategy),
            verbose=self.verbose,
        )

        estimator = _SkglmGeneralizedLinearEstimator(
            datafit=datafit,
            penalty=penalty,
            solver=solver,
        )
        fitted = estimator.fit(X, y, **fit_kwargs)
        self._store_fitted_estimator(fitted)
        return self


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


def _thin_posterior_samples(arr: Optional[np.ndarray], step: int) -> Optional[np.ndarray]:
    """Thin posterior draws while preserving an optional leading chain axis."""

    if arr is None:
        return arr
    if arr.ndim == 0:
        return arr
    if arr.ndim == 1:
        if step <= 1:
            return arr
        return arr[::step]

    thinned = arr if step <= 1 else arr[:, ::step, ...]
    if thinned.shape[0] == 1:
        return thinned[0]
    return thinned


def _flatten_scalar_draws(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    data = np.asarray(arr, dtype=np.float64)
    return data.reshape(-1)


def _flatten_param_draws(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    data = np.asarray(arr, dtype=np.float64)
    if data.ndim == 0:
        return data.reshape(1, 1)
    if data.ndim == 1:
        return data.reshape(1, -1)
    if data.ndim == 2:
        return data
    return data.reshape(-1, *data.shape[2:])


@dataclass
class _BaseHorseshoeRegression:
    """Common machinery for horseshoe-style regression baselines."""

    scale_intercept: float = 10.0
    scale_global: float = 1.0
    nu_global: float = 1.0
    nu_local: float = 1.0
    sigma_scale: float = 1.0
    slab_scale: Optional[float] = None
    slab_df: Optional[float] = None
    likelihood: str = "gaussian"
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 1
    thinning: int = 1
    seed: Optional[int] = None
    target_accept_prob: float = 0.99
    max_tree_depth: int = 10
    progress_bar: bool = False
    stan_file: Optional[str] = None

    coef_samples_: Optional[np.ndarray] = field(default=None, init=False)
    intercept_samples_: Optional[np.ndarray] = field(default=None, init=False)
    sigma_samples_: Optional[np.ndarray] = field(default=None, init=False)
    tau_samples_: Optional[np.ndarray] = field(default=None, init=False)
    lambda_samples_: Optional[np.ndarray] = field(default=None, init=False)
    lambda_tilde_samples_: Optional[np.ndarray] = field(default=None, init=False)
    c_samples_: Optional[np.ndarray] = field(default=None, init=False)
    coef_: Optional[np.ndarray] = field(default=None, init=False)
    intercept_: Optional[float] = field(default=None, init=False)
    sampler_diagnostics_: Dict[str, Any] = field(default_factory=dict, init=False)
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
        if self.slab_df is not None and self.slab_df <= 0:
            raise ValueError("slab_df must be positive when provided.")
        lik = str(self.likelihood).lower()
        if lik in {"classification", "logistic"}:
            self.likelihood = "logistic"
        elif lik in {"gaussian", "regression"}:
            self.likelihood = "gaussian"
        else:
            raise ValueError("likelihood must be either 'gaussian' or 'logistic'.")
        if self.num_warmup <= 0 or self.num_samples <= 0:
            raise ValueError("num_warmup and num_samples must be positive integers.")
        if self.num_chains <= 0:
            raise ValueError("num_chains must be a positive integer.")
        if self.thinning <= 0:
            raise ValueError("thinning must be a positive integer.")
        if not 0.0 < self.target_accept_prob < 1.0:
            raise ValueError("target_accept_prob must lie in (0, 1).")
        if self.max_tree_depth <= 0:
            raise ValueError("max_tree_depth must be a positive integer.")

    def _numpyro_model(self, X: jnp.ndarray, y: jnp.ndarray) -> None:
        sigma = None
        if self.likelihood == "gaussian":
            sigma = numpyro.sample("sigma", dist.HalfCauchy(self.sigma_scale))
            global_scale = self.scale_global * sigma
        else:
            global_scale = self.scale_global

        beta0 = numpyro.sample("beta0", dist.Normal(0.0, self.scale_intercept))

        r1_global = numpyro.sample("r1_global", dist.HalfNormal(global_scale))
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
        slab = self._draw_slab_scale()
        lambda_effective = numpyro.deterministic(
            "lambda",
            self._regularize_lambda(lambda_raw, tau, slab),
        )

        z = numpyro.sample("z", dist.Normal(jnp.zeros((p,)), 1.0).to_event(1))
        beta = numpyro.deterministic("beta", z * lambda_effective * tau)
        mean = beta0 + X @ beta
        if self.likelihood == "gaussian":
            numpyro.sample("y", dist.Normal(mean, sigma), obs=y)
        else:
            numpyro.sample("y", dist.Bernoulli(logits=mean), obs=y)

    def _draw_slab_scale(self) -> Optional[jnp.ndarray]:
        if self.slab_scale is None:
            return None
        if self.slab_df is None:
            return jnp.asarray(float(self.slab_scale))
        caux = numpyro.sample(
            "caux",
            dist.InverseGamma(0.5 * float(self.slab_df), 0.5 * float(self.slab_df)),
        )
        return numpyro.deterministic("c", float(self.slab_scale) * jnp.sqrt(caux))

    def _regularize_lambda(
        self,
        lambda_raw: jnp.ndarray,
        tau: jnp.ndarray,
        slab: Optional[jnp.ndarray],
    ) -> jnp.ndarray:
        if slab is None:
            return lambda_raw
        c2 = slab ** 2
        lam2 = lambda_raw ** 2
        tau2 = tau ** 2
        denom = c2 + tau2 * lam2 + 1e-18
        lambda_tilde_sq = (c2 * lam2) / denom
        return jnp.sqrt(lambda_tilde_sq)

    def _use_gibbs_backend(self) -> bool:
        return self.likelihood == "gaussian"

    def _use_cmdstan_backend(self) -> bool:
        return False

    def _default_stan_file(self) -> Optional[Path]:
        return None

    def _resolve_stan_file(self) -> Path:
        if self.stan_file is not None:
            return Path(self.stan_file).expanduser().resolve()
        default_path = self._default_stan_file()
        if default_path is None:
            raise RuntimeError("No Stan model file configured for this model.")
        return default_path

    def _load_cmdstan_model(self) -> Any:
        if _CmdStanModel is None:
            raise ImportError(
                "cmdstanpy is required for the configured RHS backend. Install with `pip install cmdstanpy` "
                "and ensure CmdStan is installed (e.g., via `python -m cmdstanpy.install_cmdstan`)."
            )
        stan_path = str(self._resolve_stan_file())
        cached = _CMDSTAN_MODEL_CACHE.get(stan_path)
        if cached is not None:
            return cached
        model = _CmdStanModel(stan_file=stan_path)
        _CMDSTAN_MODEL_CACHE[stan_path] = model
        return model

    @staticmethod
    def _extract_cmdstan_array(
        draws_df: "np.ndarray | Any",
        *,
        name: str,
        scalar: bool,
        thinning: int,
    ) -> Optional[np.ndarray]:
        import pandas as pd

        if not isinstance(draws_df, pd.DataFrame):
            return None

        chain_ids = sorted(int(v) for v in draws_df["chain__"].unique())
        if scalar:
            if name not in draws_df.columns:
                return None
            chain_blocks: list[np.ndarray] = []
            for chain_id in chain_ids:
                chain_df = draws_df.loc[draws_df["chain__"] == chain_id].sort_values("draw__")
                chain_blocks.append(chain_df[name].to_numpy(dtype=np.float64))
            arr = np.stack(chain_blocks, axis=0)
            arr = _thin_posterior_samples(arr, thinning)
            if arr is not None and arr.ndim == 2:
                return arr[..., None]
            return arr

        prefix = f"{name}["
        cols = [col for col in draws_df.columns if col.startswith(prefix)]
        if not cols:
            return None
        cols.sort(key=lambda text: int(text.split("[", 1)[1].split("]", 1)[0]))
        chain_blocks_v: list[np.ndarray] = []
        for chain_id in chain_ids:
            chain_df = draws_df.loc[draws_df["chain__"] == chain_id].sort_values("draw__")
            chain_blocks_v.append(chain_df[cols].to_numpy(dtype=np.float64))
        arr_v = np.stack(chain_blocks_v, axis=0)
        return _thin_posterior_samples(arr_v, thinning)

    def _extract_cmdstan_diagnostics(self, draws_df: Any) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {"backend": "cmdstan_hmc"}
        required = {"chain__", "draw__", "divergent__", "energy__"}
        if not required.issubset(set(draws_df.columns)):
            diagnostics["hmc"] = {}
            return diagnostics
        chain_ids = sorted(int(v) for v in draws_df["chain__"].unique())

        divergences = 0
        treedepth_hits = -1
        max_num_steps = -1
        ebfmi_values: list[float] = []
        td_hits = 0
        has_tree = "treedepth__" in draws_df.columns
        has_steps = "n_leapfrog__" in draws_df.columns

        for chain_id in chain_ids:
            chain_df = draws_df.loc[draws_df["chain__"] == chain_id].sort_values("draw__")
            div = chain_df["divergent__"].to_numpy(dtype=float)
            divergences += int(np.sum(div > 0.5))

            if has_tree:
                td = chain_df["treedepth__"].to_numpy(dtype=float)
                td_hits += int(np.sum(td >= float(self.max_tree_depth)))
            if has_steps:
                ns = chain_df["n_leapfrog__"].to_numpy(dtype=float)
                if ns.size:
                    max_num_steps = max(max_num_steps, int(np.max(ns)))

            energy = chain_df["energy__"].to_numpy(dtype=float)
            if energy.size < 3:
                ebfmi_values.append(float("nan"))
            else:
                numer = float(np.mean(np.diff(energy) ** 2))
                denom = float(np.var(energy))
                if denom <= 0.0 or not np.isfinite(denom):
                    ebfmi_values.append(float("nan"))
                else:
                    ebfmi_values.append(numer / denom)

        if has_tree:
            treedepth_hits = int(td_hits)
        ebfmi_min = float(np.nanmin(ebfmi_values)) if ebfmi_values else float("nan")
        diagnostics["hmc"] = {
            "divergences": int(divergences),
            "treedepth_hits": treedepth_hits,
            "max_num_steps": max_num_steps,
            "ebfmi_per_chain": ebfmi_values,
            "ebfmi_min": ebfmi_min,
        }
        return diagnostics

    def _fit_with_cmdstan(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.likelihood != "gaussian":
            raise RuntimeError("CmdStan backend is currently configured for gaussian RHS only.")
        model = self._load_cmdstan_model()
        n, d = X.shape
        data = {
            "n": int(n),
            "d": int(d),
            "y": np.asarray(y, dtype=float).reshape(-1),
            "X": np.asarray(X, dtype=float),
            "scale_icept": float(self.scale_intercept),
            "scale_global": float(self.scale_global),
            "nu_global": float(self.nu_global),
            "nu_local": float(self.nu_local),
            "slab_scale": float(self.slab_scale if self.slab_scale is not None else 2.0),
            "slab_df": float(self.slab_df if self.slab_df is not None else 4.0),
        }
        fit = model.sample(
            data=data,
            seed=0 if self.seed is None else int(self.seed),
            chains=int(self.num_chains),
            parallel_chains=max(1, min(int(self.num_chains), 4)),
            iter_warmup=int(self.num_warmup),
            iter_sampling=int(self.num_samples),
            adapt_delta=float(self.target_accept_prob),
            max_treedepth=int(self.max_tree_depth),
            show_progress=bool(self.progress_bar),
        )
        draws_df = fit.draws_pd()

        self.coef_samples_ = self._extract_cmdstan_array(
            draws_df, name="beta", scalar=False, thinning=int(self.thinning)
        )
        if self.coef_samples_ is None:
            raise RuntimeError("CmdStan RHS model did not produce beta draws.")
        self.intercept_samples_ = self._extract_cmdstan_array(
            draws_df, name="beta0", scalar=True, thinning=int(self.thinning)
        )
        self.sigma_samples_ = self._extract_cmdstan_array(
            draws_df, name="sigma", scalar=True, thinning=int(self.thinning)
        )
        self.tau_samples_ = self._extract_cmdstan_array(
            draws_df, name="tau", scalar=True, thinning=int(self.thinning)
        )
        self.lambda_samples_ = self._extract_cmdstan_array(
            draws_df, name="lambda", scalar=False, thinning=int(self.thinning)
        )
        self.lambda_tilde_samples_ = self._extract_cmdstan_array(
            draws_df, name="lambda_tilde", scalar=False, thinning=int(self.thinning)
        )
        self.c_samples_ = self._extract_cmdstan_array(
            draws_df, name="c", scalar=True, thinning=int(self.thinning)
        )

        coef_draws = _flatten_param_draws(self.coef_samples_)
        intercept_draws = _flatten_scalar_draws(self.intercept_samples_)
        self.coef_ = None if coef_draws is None else coef_draws.mean(axis=0)
        self.intercept_ = 0.0 if intercept_draws is None else float(intercept_draws.mean())
        self.sampler_diagnostics_ = self._extract_cmdstan_diagnostics(draws_df)

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        groups: Optional[list[list[int]]] = None,
    ) -> "_BaseHorseshoeRegression":
        X_arr = _ensure_2d(X, "X")
        y_arr = _ensure_1d(y, "y")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have matching number of rows.")

        if self._use_cmdstan_backend():
            self._fit_with_cmdstan(X_arr.astype(np.float64), y_arr.astype(np.float64))
            return self
        if self._use_gibbs_backend():
            self._fit_with_gibbs(X_arr, y_arr, groups=groups)
            return self

        rng_seed = 0 if self.seed is None else int(self.seed)
        self._rng_key = random.PRNGKey(rng_seed)
        kernel = NUTS(
            self._numpyro_model,
            target_accept_prob=self.target_accept_prob,
            dense_mass=True,
            max_tree_depth=int(self.max_tree_depth),
        )
        mcmc = MCMC(
            kernel,
            num_warmup=int(self.num_warmup),
            num_samples=int(self.num_samples),
            num_chains=int(self.num_chains),
            progress_bar=self.progress_bar,
            chain_method="sequential",
        )
        mcmc.run(
            self._rng_key,
            jnp.asarray(X_arr),
            jnp.asarray(y_arr),
            extra_fields=("diverging", "energy", "num_steps"),
        )
        samples = mcmc.get_samples(group_by_chain=True)
        self.sampler_diagnostics_ = self._extract_hmc_diagnostics(mcmc)
        self._store_samples(samples)
        return self

    def _fit_with_gibbs(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        groups: Optional[list[list[int]]] = None,
    ) -> None:
        use_groups = False
        group_scale = 1.0
        gibbs_groups = None
        total_iters = int(self.num_warmup + self.num_samples)
        sampler = GRRHS_Gibbs(
            c=float(self.slab_scale if self.slab_scale is not None else 1.0),
            tau0=float(self.scale_global),
            eta=group_scale,
            s0=float(self.sigma_scale),
            iters=total_iters,
            burnin=int(self.num_warmup),
            thin=int(self.thinning),
            seed=0 if self.seed is None else int(self.seed),
            use_groups=use_groups,
        )
        fitted = sampler.fit(X, y, groups=gibbs_groups)

        self.coef_samples_ = None if fitted.coef_samples_ is None else fitted.coef_samples_.copy()
        if self.coef_samples_ is None:
            raise RuntimeError("Gibbs sampler did not return coefficient draws.")
        self.coef_ = fitted.coef_mean_.copy() if fitted.coef_mean_ is not None else self.coef_samples_.mean(axis=0)
        self.intercept_ = float(fitted.intercept_)

        if fitted.sigma2_samples_ is not None:
            self.sigma_samples_ = np.sqrt(np.maximum(fitted.sigma2_samples_, 0.0))
        else:
            self.sigma_samples_ = None
        self.tau_samples_ = None if fitted.tau_samples_ is None else fitted.tau_samples_.copy()
        self.lambda_samples_ = None if fitted.lambda_samples_ is None else fitted.lambda_samples_.copy()
        self.intercept_samples_ = None
        self.sampler_diagnostics_ = {
            "backend": "gibbs",
            "hmc": None,
        }

    def _extract_hmc_diagnostics(self, mcmc: MCMC) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {"backend": "hmc"}
        try:
            extra_fields = mcmc.get_extra_fields(group_by_chain=True)
        except Exception:
            diagnostics["hmc"] = {}
            return diagnostics

        diverging_raw = np.asarray(extra_fields.get("diverging", []), dtype=float)
        num_steps_raw = np.asarray(extra_fields.get("num_steps", []), dtype=float)
        energy_raw = np.asarray(
            extra_fields.get("energy", extra_fields.get("potential_energy", [])),
            dtype=float,
        )
        if diverging_raw.ndim == 1 and diverging_raw.size:
            diverging_raw = diverging_raw.reshape(1, -1)
        if num_steps_raw.ndim == 1 and num_steps_raw.size:
            num_steps_raw = num_steps_raw.reshape(1, -1)
        if energy_raw.ndim == 1 and energy_raw.size:
            energy_raw = energy_raw.reshape(1, -1)
        if num_steps_raw.size == 0:
            tree_depth_raw = np.asarray(extra_fields.get("tree_depth", []), dtype=float)
            if tree_depth_raw.ndim == 1 and tree_depth_raw.size:
                tree_depth_raw = tree_depth_raw.reshape(1, -1)
            if tree_depth_raw.size:
                num_steps_raw = np.power(2.0, tree_depth_raw)

        if diverging_raw.size:
            divergences = int(np.sum(diverging_raw > 0.5))
        else:
            divergences = -1
        if num_steps_raw.size:
            treedepth_limit = float(2 ** int(self.max_tree_depth))
            treedepth_hits = int(np.sum(num_steps_raw >= treedepth_limit))
            max_num_steps = int(np.max(num_steps_raw))
        else:
            treedepth_hits = -1
            max_num_steps = -1

        ebfmi_values: list[float] = []
        if energy_raw.size:
            for chain_energy in energy_raw:
                if chain_energy.size < 3:
                    ebfmi_values.append(float("nan"))
                    continue
                centered = np.asarray(chain_energy, dtype=float)
                numer = float(np.mean(np.diff(centered) ** 2))
                denom = float(np.var(centered))
                if denom <= 0.0 or not np.isfinite(denom):
                    ebfmi_values.append(float("nan"))
                else:
                    ebfmi_values.append(numer / denom)
        ebfmi_min = float(np.nanmin(ebfmi_values)) if ebfmi_values else float("nan")

        diagnostics["hmc"] = {
            "divergences": divergences,
            "treedepth_hits": treedepth_hits,
            "max_num_steps": max_num_steps,
            "ebfmi_per_chain": ebfmi_values,
            "ebfmi_min": ebfmi_min,
        }
        return diagnostics

    def _store_samples(self, samples: Dict[str, jnp.ndarray]) -> None:
        def _convert(name: str, *, scalar: bool = False) -> Optional[np.ndarray]:
            if name not in samples:
                return None
            arr = np.asarray(samples[name], dtype=np.float64)
            thinned = _thin_posterior_samples(arr, self.thinning)
            if scalar and thinned is not None and thinned.ndim == 2:
                return thinned[..., None]
            return thinned

        self.coef_samples_ = _convert("beta")
        if self.coef_samples_ is None:
            raise RuntimeError("NumPyro model did not produce beta samples.")
        self.intercept_samples_ = _convert("beta0", scalar=True)
        self.sigma_samples_ = _convert("sigma", scalar=True)
        self.tau_samples_ = _convert("tau", scalar=True)
        self.lambda_samples_ = _convert("lambda")
        self.c_samples_ = _convert("c", scalar=True)

        coef_draws = _flatten_param_draws(self.coef_samples_)
        intercept_draws = _flatten_scalar_draws(self.intercept_samples_)

        self.coef_ = None if coef_draws is None else coef_draws.mean(axis=0)
        if intercept_draws is not None:
            self.intercept_ = float(intercept_draws.mean())
        else:
            self.intercept_ = 0.0

    def predict(self, X: ArrayLike) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        coef = np.asarray(self.coef_, dtype=np.float64)
        intercept = float(self.intercept_ or 0.0)
        X_arr = _ensure_2d(X, "X")
        return X_arr.astype(np.float64) @ coef + intercept

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        if self.likelihood != "logistic":
            raise RuntimeError("predict_proba is only defined for logistic likelihood.")
        logits = self.predict(X)
        logits = np.clip(logits, -60.0, 60.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - probs, probs])

    def get_posterior_summaries(self) -> Dict[str, Any]:
        if self.coef_samples_ is None:
            raise RuntimeError("Model must be fitted before requesting summaries.")
        coef_draws = _flatten_param_draws(self.coef_samples_)
        if coef_draws is None:
            raise RuntimeError("Posterior coefficient draws are unavailable.")
        summaries: Dict[str, Any] = {
            "coef_mean": coef_draws.mean(axis=0),
            "coef_median": np.median(coef_draws, axis=0),
            "coef_ci95": np.quantile(coef_draws, [0.025, 0.975], axis=0),
        }
        if self.sigma_samples_ is not None:
            sigma_draws = _flatten_scalar_draws(self.sigma_samples_)
            if sigma_draws is not None:
                summaries["sigma_mean"] = float(sigma_draws.mean())
        if self.tau_samples_ is not None:
            tau_draws = _flatten_scalar_draws(self.tau_samples_)
            if tau_draws is not None:
                summaries["tau_mean"] = float(tau_draws.mean())
        if self.lambda_samples_ is not None:
            lambda_draws = _flatten_param_draws(self.lambda_samples_)
            if lambda_draws is not None:
                summaries["lambda_mean"] = lambda_draws.mean(axis=0)
        if self.c_samples_ is not None:
            c_draws = _flatten_scalar_draws(self.c_samples_)
            if c_draws is not None:
                summaries["c_mean"] = float(c_draws.mean())
        return summaries


@dataclass
class HorseshoeRegression(_BaseHorseshoeRegression):
    """Standard horseshoe regression baseline."""

    pass


@dataclass
class RegularizedHorseshoeRegression(_BaseHorseshoeRegression):
    """Regularized horseshoe regression baseline (Piironen & Vehtari, 2017)."""

    slab_scale: float = 2.0
    slab_df: float = 4.0

    def _default_stan_file(self) -> Optional[Path]:
        return (Path(__file__).resolve().parent / "stan" / "rhs_gaussian_regression.stan").resolve()

    def _use_cmdstan_backend(self) -> bool:
        # Match paper Appendix C.2 implementation for gaussian RHS.
        return self.likelihood == "gaussian"

    def _use_gibbs_backend(self) -> bool:
        return False


