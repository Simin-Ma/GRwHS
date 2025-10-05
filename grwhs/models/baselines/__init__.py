"""Baseline linear models used by experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..horseshoe_baseline import HorseshoeRegression, RegularizedHorseshoeRegression


from .group_lasso import (
    GroupLasso,
    SparseGroupLasso,
    _SkglmRegressorBase,
    _require_skglm,
)

__all__ = [
    "Ridge",
    "Lasso",
    "ElasticNet",
    "GroupLasso",
    "SparseGroupLasso",
    "HorseshoeRegression",
    "RegularizedHorseshoeRegression",
]


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
