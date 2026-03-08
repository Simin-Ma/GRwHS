from __future__ import annotations

import numpy as np

from data.generators import SyntheticConfig, generate_synthetic
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
