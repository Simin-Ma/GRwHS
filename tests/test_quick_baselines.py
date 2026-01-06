"""Quick regression checks for skglm-backed baselines."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grrhs.models.baselines import ElasticNet, GroupLasso, Lasso, Ridge, SparseGroupLasso


def _make_regression(n: int, p: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    beta = rng.standard_normal(p)
    y = (X @ beta + 0.05 * rng.standard_normal(n)).astype(np.float32)
    return X, y


def test_quick_ridge_runs():
    X, y = _make_regression(16, 4)
    model = Ridge(alpha=0.3, max_iter=15, max_epochs=80, p0=2, tol=5e-3)
    fitted = model.fit(X, y)
    preds = fitted.predict(X)
    assert preds.shape == (16,)


def test_quick_lasso_runs():
    X, y = _make_regression(16, 4, seed=1)
    model = Lasso(alpha=0.1, max_iter=15, max_epochs=80, p0=2, tol=5e-3)
    fitted = model.fit(X, y)
    preds = fitted.predict(X)
    assert preds.shape == (16,)


def test_quick_elastic_net_runs():
    X, y = _make_regression(16, 4, seed=2)
    model = ElasticNet(alpha=0.2, l1_ratio=0.3, max_iter=15, max_epochs=80, p0=2, tol=5e-3)
    fitted = model.fit(X, y)
    preds = fitted.predict(X)
    assert preds.shape == (16,)


def test_quick_group_lasso_runs():
    X, y = _make_regression(20, 4, seed=3)
    groups = [[0, 1], [2, 3]]
    model = GroupLasso(
        groups=groups,
        alpha=0.2,
        max_iter=20,
        max_epochs=80,
        p0=2,
        tol=5e-3,
        warm_start=False,
    )
    fitted = model.fit(X, y)
    preds = fitted.predict(X)
    assert preds.shape == (20,)


def test_quick_sparse_group_lasso_runs():
    X, y = _make_regression(20, 4, seed=4)
    groups = [[0, 1], [2, 3]]
    model = SparseGroupLasso(
        groups=groups,
        alpha=0.2,
        l1_ratio=0.4,
        max_iter=20,
        max_epochs=80,
        p0=2,
        tol=5e-3,
        warm_start=False,
    )
    fitted = model.fit(X, y)
    preds = fitted.predict(X)
    assert preds.shape == (20,)
