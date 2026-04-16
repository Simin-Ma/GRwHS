from __future__ import annotations

import numpy as np

from grrhs.experiments.registry import build_from_config
from grrhs.models.baselines import GroupHorseshoePlusRegression, GroupedHorseshoeRegression


def _toy_grouped_dataset(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(60, 10))
    beta = np.zeros(10, dtype=float)
    beta[0] = 1.5
    beta[1] = 1.0
    beta[7] = 0.8
    y = X @ beta + rng.normal(scale=0.5, size=60)
    groups = [list(range(5)), list(range(5, 10))]
    return X, y, beta, groups


def test_grouped_horseshoe_fit_exposes_group_level_draws():
    X, y, beta_true, groups = _toy_grouped_dataset()
    model = GroupedHorseshoeRegression(iters=80, burnin=30, thin=2, seed=1, num_chains=1)
    fitted = model.fit(X, y, groups=groups)

    assert fitted.coef_.shape == beta_true.shape
    assert fitted.lambda_group_samples_.shape[1] == len(groups)
    assert fitted.lambda_samples_.shape[1] == X.shape[1]
    assert np.isfinite(fitted.tau_mean_)
    assert np.mean((fitted.predict(X) - y) ** 2) < np.var(y)


def test_group_horseshoe_plus_registry_builder_runs():
    X, y, beta_true, groups = _toy_grouped_dataset(seed=7)
    cfg = {
        "model": {
            "name": "group_horseshoe_plus",
            "fit_intercept": True,
            "tau0": 0.5,
            "iters": 80,
            "burnin": 30,
            "thin": 2,
            "num_chains": 1,
            "seed": 7,
        },
        "data": {"groups": groups, "p": X.shape[1]},
    }
    model = build_from_config(cfg)
    assert isinstance(model, GroupHorseshoePlusRegression)
    fitted = model.fit(X, y, groups=groups)

    assert fitted.coef_.shape == beta_true.shape
    assert fitted.group_lambda_mean_.shape[0] == len(groups)
    assert fitted.lambda_mean_.shape[0] == X.shape[1]
    assert fitted.sampler_diagnostics_["hierarchical"] is True
