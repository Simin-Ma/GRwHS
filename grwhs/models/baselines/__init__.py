"""Baseline linear models used by experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression as _SklearnLogisticRegression

from ..horseshoe_baseline import HorseshoeRegression, RegularizedHorseshoeRegression

from .group_lasso import (
    GroupLasso,
    SparseGroupLasso,
    _SkglmRegressorBase,
    _require_skglm,
)

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
