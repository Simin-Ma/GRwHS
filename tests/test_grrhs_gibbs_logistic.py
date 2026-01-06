from __future__ import annotations

import numpy as np
import numpy.testing as npt

from grrhs.models.grrhs_gibbs_logistic import GRRHS_Gibbs_Logistic


def _make_synthetic_logistic(seed: int = 0):
    rng = np.random.default_rng(seed)
    n, p = 40, 6
    X = rng.normal(size=(n, p))
    X -= X.mean(axis=0, keepdims=True)
    X /= X.std(axis=0, keepdims=True)

    beta_true = np.array([1.2, -0.6, 0.9, 0.0, -1.1, 0.0])
    logits = X @ beta_true
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probs, size=n)

    groups = [[0, 1, 2], [3, 4, 5]]
    return X.astype(float), y.astype(float), groups


def test_grrhs_gibbs_logistic_runs_and_returns_probabilities():
    X, y, groups = _make_synthetic_logistic()

    model = GRRHS_Gibbs_Logistic(
        c=1.5,
        tau0=0.2,
        eta=0.6,
        s0=1.0,
        iters=60,
        burnin=30,
        thin=5,
        seed=2025,
        slice_w=0.5,
        slice_m=40,
        jitter=1.0e-6,
    )

    fitted = model.fit(X, y, groups=groups)

    samples = fitted.coef_samples_
    assert samples is not None
    expected_draws = (model.iters - model.burnin) // model.thin
    assert samples.shape == (expected_draws, X.shape[1])
    assert fitted.tau_samples_ is not None and fitted.tau_samples_.shape == (expected_draws,)
    assert fitted.phi_samples_ is not None and fitted.phi_samples_.shape == (expected_draws, len(groups))
    assert fitted.lambda_samples_ is not None and fitted.lambda_samples_.shape == (expected_draws, X.shape[1])
    assert fitted.sigma2_samples_ is None

    proba = fitted.predict_proba(X[:5])
    assert proba.shape == (5, 2)
    npt.assert_array_less(proba, 1.0 + 1e-7)
    npt.assert_array_less(-1e-7, proba)
    preds = fitted.predict(X[:5])
    assert preds.shape == (5,)
    assert set(np.unique(preds)).issubset({0, 1})
