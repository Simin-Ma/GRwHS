from __future__ import annotations

import numpy as np

from grrhs.models.gigg_cran import GIGGRegressionCRAN


def test_gigg_cran_wrapper_smoke_multichain():
    rng = np.random.default_rng(0)
    n = 50
    p = 6
    X = rng.normal(size=(n, p))
    beta_true = np.array([1.0, -0.5, 0.0, 0.8, 0.0, 0.0], dtype=float)
    y = X @ beta_true + rng.normal(scale=0.3, size=n)
    groups = [[0, 1, 2], [3, 4, 5]]

    model = GIGGRegressionCRAN(
        method="fixed",
        n_burn_in=40,
        n_samples=30,
        n_thin=1,
        seed=123,
        num_chains=2,
        store_lambda=True,
        fit_intercept=False,
    )
    fitted = model.fit(X, y, groups=groups)
    pred = fitted.predict(X[:7])

    assert pred.shape == (7,)
    assert fitted.coef_samples_ is not None and fitted.coef_samples_.ndim == 3
    assert fitted.coef_samples_.shape[-1] == p
    assert fitted.tau2_samples_ is not None
    assert fitted.sigma2_samples_ is not None
    assert fitted.gamma2_samples_ is not None
    assert fitted.lambda_samples_ is not None and fitted.lambda_samples_.ndim == 3

