"""Group-lasso style baselines backed by the skglm library."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

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


__all__ = ["GroupLasso", "SparseGroupLasso"]


GroupsLike = Sequence[Sequence[int]]
WeightsLike = Optional[Sequence[float]]
FeatureWeightsLike = Optional[Sequence[float]]


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
    seen = np.zeros(n_features, dtype=bool)
    for group in groups:
        for idx in group:
            if idx < 0 or idx >= n_features:
                raise ValueError(f"Feature index {idx} is out of bounds for p={n_features}.")
            if seen[idx]:
                raise ValueError(f"Feature index {idx} appears in more than one group.")
            seen[idx] = True
    if not np.all(seen):
        missing = np.where(~seen)[0]
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
