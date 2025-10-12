"""Unified collection of baseline models used across experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from sklearn.linear_model import LogisticRegression as _SklearnLogisticRegression

try:
    from skglm import GroupLasso as _SkglmGroupLasso
    from skglm import GeneralizedLinearEstimator as _SkglmGeneralizedLinearEstimator
    from skglm.datafits import QuadraticGroup as _SkglmQuadraticGroup
    from skglm.penalties.block_separable import WeightedL1GroupL2 as _SkglmWeightedL1GroupL2
    from skglm.solvers import GroupBCD as _SkglmGroupBCD
    from skglm.utils.data import grp_converter as _skglm_grp_converter
except ImportError as exc:  # pragma: no cover - handled at runtime
    _SKGLM_IMPORT_ERROR = exc
    _SkglmGroupLasso = None
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
    "GroupLasso",
    "SparseGroupLasso",
    "HorseshoeRegression",
    "RegularizedHorseshoeRegression",
]

GroupsLike = Sequence[Sequence[int]]
WeightsLike = Optional[Sequence[float]]
FeatureWeightsLike = Optional[Sequence[float]]
ArrayLike = Any


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
class GroupLasso(_SkglmRegressorBase):
    groups: GroupsLike
    alpha: float = 1.0
    group_weights: WeightsLike = None
    fit_intercept: bool = False
    max_iter: int = 2_000
    max_epochs: int = 50_000
    p0: int = 10
    tol: float = 1e-6
    warm_start: bool = True
    ws_strategy: str = "fixpoint"
    verbose: int = 0
    positive: bool = False

    def __post_init__(self) -> None:
        super().__init__()
        self.groups = _normalize_groups(self.groups)
        self._group_weights = _coerce_group_weights(self.groups, self.group_weights)

    def fit(self, X: Any, y: Any, **fit_kwargs: Any) -> "GroupLasso":
        _require_skglm()
        try:
            n_features = int(X.shape[1])
        except Exception:  # pragma: no cover - X without shape[1]
            n_features = None
        if n_features is not None:
            _ensure_groups_cover_features(self.groups, n_features)
        estimator = _SkglmGroupLasso(
            groups=self.groups,
            alpha=float(self.alpha),
            weights=self._group_weights,
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

