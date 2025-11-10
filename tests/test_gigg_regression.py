from __future__ import annotations

from data.generators import SyntheticConfig, generate_synthetic
from grwhs.models.gigg_regression import GIGGRegression


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
    assert model.lambda_samples_ is not None
