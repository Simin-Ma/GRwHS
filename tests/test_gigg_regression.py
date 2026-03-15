from __future__ import annotations

import numpy as np

from data.generators import SyntheticConfig, generate_synthetic
from grrhs.diagnostics.convergence import summarize_convergence
from grrhs.models.gigg_regression import GIGGRegression


def test_gigg_regression_smoke():
    cfg = SyntheticConfig(
        n=80,
        p=8,
        G=2,
        group_sizes=[4, 4],
        signal={
            "blueprint": [
                {
                    "groups": [0],
                    "components": [
                        {"distribution": "constant", "count": 2, "value": 1.0, "sign": "positive"},
                    ],
                }
            ]
        },
        noise_sigma=0.2,
        seed=10,
    )
    data = generate_synthetic(cfg)
    model = GIGGRegression(iters=400, burnin=200, thin=5, seed=0, store_lambda=True)
    model.fit(data.X, data.y, groups=data.groups)
    preds = model.predict(data.X[:5])
    assert preds.shape == (5,)
    assert model.coef_samples_ is not None
    assert model.coef_samples_.shape[1] == data.X.shape[1]
    assert model.tau_samples_ is not None
    assert model.gamma_samples_ is not None
    assert model.tau2_samples_ is not None
    assert model.gamma2_samples_ is not None
    assert model.sigma_samples_ is not None
    assert model.sigma2_samples_ is not None
    assert model.lambda_samples_ is not None
    assert model.b_samples_ is not None
    assert model.b_mean_ is not None
    np.testing.assert_allclose(model.tau_samples_**2, model.tau2_samples_, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(model.gamma_samples_**2, model.gamma2_samples_, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(model.sigma_samples_**2, model.sigma2_samples_, rtol=1e-6, atol=1e-8)


def test_gigg_regression_preserves_multichain_draws_for_convergence():
    cfg = SyntheticConfig(
        n=80,
        p=8,
        G=2,
        group_sizes=[4, 4],
        signal={
            "blueprint": [
                {
                    "groups": [0],
                    "components": [
                        {"distribution": "constant", "count": 2, "value": 1.0, "sign": "positive"},
                    ],
                }
            ]
        },
        noise_sigma=0.2,
        seed=12,
    )
    data = generate_synthetic(cfg)
    model = GIGGRegression(
        iters=240,
        burnin=120,
        thin=12,
        seed=9,
        num_chains=2,
        store_lambda=True,
    )
    fitted = model.fit(data.X, data.y, groups=data.groups)

    assert fitted.coef_samples_ is not None and fitted.coef_samples_.ndim == 3
    assert fitted.tau_samples_ is not None and fitted.tau_samples_.ndim == 2
    assert fitted.gamma_samples_ is not None and fitted.gamma_samples_.ndim == 3
    assert fitted.lambda_samples_ is not None and fitted.lambda_samples_.ndim == 3
    assert fitted.b_samples_ is not None and fitted.b_samples_.ndim == 3

    convergence = summarize_convergence(
        {
            "beta": fitted.coef_samples_,
            "tau": fitted.tau_samples_,
            "gamma": fitted.gamma_samples_,
            "lambda": fitted.lambda_samples_,
        }
    )
    assert convergence["beta"]["raw_num_chains"] == 2
    assert convergence["beta"]["diagnostic_valid"] is True
    assert convergence["tau"]["raw_num_chains"] == 2
    assert convergence["tau"]["diagnostic_valid"] is True


def test_gigg_regression_btrick_with_covariates_smoke():
    rng = np.random.default_rng(21)
    n = 60
    p = 8
    k = 2
    X = rng.normal(size=(n, p))
    C = rng.normal(size=(n, k))
    beta_true = np.array([1.0, -0.8, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=float)
    alpha_true = np.array([0.6, -0.3], dtype=float)
    y = X @ beta_true + C @ alpha_true + rng.normal(scale=0.3, size=n)
    groups = [[0, 1, 2, 3], [4, 5, 6, 7]]

    model = GIGGRegression(
        method="mmle",
        n_burn_in=120,
        n_samples=60,
        n_thin=1,
        seed=11,
        btrick=True,
        stable_solve=True,
        store_lambda=True,
    )
    fitted = model.fit(X, y, groups=groups, C=C)
    pred = fitted.predict(X[:10], C=C[:10])

    assert pred.shape == (10,)
    assert fitted.coef_samples_ is not None
    assert fitted.alpha_samples_ is not None
    assert fitted.alpha_samples_.shape[-1] == k
    assert fitted.lambda_samples_ is not None
