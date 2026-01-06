from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grrhs.models.baselines import (
    GroupHorseshoeRegression,
    HorseshoeRegression,
    RegularizedHorseshoeRegression,
)


def _synthetic_regression(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, p = 40, 6
    X = rng.normal(size=(n, p)).astype(np.float32)
    beta = np.array([2.0, 0.0, 0.0, -1.5, 0.0, 0.75], dtype=np.float32)
    noise = rng.normal(scale=0.1, size=n).astype(np.float32)
    y = (X @ beta + noise).astype(np.float32)
    return X, y, beta


def _synthetic_classification(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, p = 60, 5
    X = rng.normal(size=(n, p)).astype(np.float32)
    beta = np.array([1.5, -0.75, 0.5, 0.0, 0.25], dtype=np.float32)
    logits = X @ beta
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -8.0, 8.0)))
    y = rng.binomial(1, probs).astype(np.float32)
    return X, y


def test_horseshoe_regression_prefers_signal_features():
    X, y, _ = _synthetic_regression(seed=123)
    model = HorseshoeRegression(
        num_warmup=150,
        num_samples=150,
        num_chains=1,
        target_accept_prob=0.9,
        progress_bar=False,
        seed=2024,
    )
    fitted = model.fit(X, y)

    assert fitted.coef_samples_ is not None
    assert fitted.coef_samples_.shape == (150, X.shape[1])
    assert fitted.tau_samples_ is not None

    preds = fitted.predict(X[:5])
    assert preds.shape == (5,)

    support = np.array([0, 3, 5])
    inactive = np.array([1, 2, 4])
    active_mean = float(np.mean(np.abs(fitted.coef_[support])))
    inactive_mean = float(np.mean(np.abs(fitted.coef_[inactive])))
    assert active_mean > inactive_mean


def test_regularized_horseshoe_reports_posterior_summaries():
    X, y, _ = _synthetic_regression(seed=321)
    model = RegularizedHorseshoeRegression(
        slab_scale=0.5,
        num_warmup=150,
        num_samples=150,
        num_chains=1,
        target_accept_prob=0.9,
        progress_bar=False,
        seed=2025,
    )
    fitted = model.fit(X, y)

    assert fitted.lambda_samples_ is not None
    summaries = fitted.get_posterior_summaries()
    assert "tau_mean" in summaries
    assert "lambda_mean" in summaries
    assert summaries["lambda_mean"].shape == (X.shape[1],)

    preds = fitted.predict(X)
    mse = float(np.mean((preds - y) ** 2))
    baseline = float(np.var(y))
    assert mse < baseline


def test_group_horseshoe_produces_group_level_shrinkage():
    X, y, _ = _synthetic_regression(seed=4321)
    groups = [[0, 3, 5], [1, 2, 4]]
    model = GroupHorseshoeRegression(
        num_warmup=150,
        num_samples=150,
        num_chains=1,
        target_accept_prob=0.9,
        progress_bar=False,
        seed=10,
        group_scale=1.0,
    )
    fitted = model.fit(X, y, groups=groups)

    assert fitted.lambda_group_samples_ is not None
    assert fitted.lambda_group_samples_.shape == (150, len(groups))
    summaries = fitted.get_posterior_summaries()
    assert "lambda_group_mean" in summaries
    group_means = summaries["lambda_group_mean"]
    assert group_means.shape == (len(groups),)

    assert fitted.lambda_samples_ is not None
    assert fitted.lambda_samples_.shape[1] == X.shape[1]

    assert float(group_means[0]) > float(group_means[1])


def test_horseshoe_logistic_handles_binary_targets():
    X, y = _synthetic_classification(seed=7)
    model = HorseshoeRegression(
        likelihood="logistic",
        num_warmup=120,
        num_samples=120,
        num_chains=1,
        target_accept_prob=0.9,
        progress_bar=False,
        seed=77,
    )
    fitted = model.fit(X, y)

    logits = fitted.predict(X[:5])
    probs = fitted.predict_proba(X[:5])
    assert logits.shape == (5,)
    assert probs.shape == (5, 2)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(5), atol=1e-6)


def test_group_horseshoe_logistic_predicts_probabilities():
    X, y = _synthetic_classification(seed=11)
    groups = [[0, 1, 2], [3, 4]]
    model = GroupHorseshoeRegression(
        groups=groups,
        likelihood="logistic",
        num_warmup=120,
        num_samples=120,
        num_chains=1,
        target_accept_prob=0.9,
        progress_bar=False,
        seed=88,
    )
    fitted = model.fit(X, y, groups=groups)

    probs = fitted.predict_proba(X[:10])
    assert probs.shape == (10, 2)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(10), atol=1e-6)
