from __future__ import annotations

import numpy as np
import numpy.testing as npt

from grrhs.models.grrhs_gibbs import GRRHS_Gibbs


def _synthetic_grouped_regression(seed: int = 0):
    rng = np.random.default_rng(seed)
    n, p = 32, 4
    X = rng.normal(size=(n, p))
    X -= X.mean(axis=0, keepdims=True)
    X /= X.std(axis=0, keepdims=True)

    beta_true = np.array([1.2, 0.0, -0.9, 0.0])
    noise = 0.1 * rng.normal(size=n)
    y = X @ beta_true + noise

    groups = [[0, 1], [2, 3]]
    return X.astype(float), y.astype(float), groups, beta_true


def test_grrhs_gibbs_runs_and_returns_posterior_draws():
    X, y, groups, beta_true = _synthetic_grouped_regression()

    model = GRRHS_Gibbs(
        c=1.5,
        tau0=0.2,
        eta=0.6,
        s0=1.0,
        iters=120,
        burnin=60,
        thin=6,
        seed=2024,
        slice_w=0.5,
        slice_m=50,
    )

    fitted = model.fit(X, y, groups)

    samples = fitted.coef_samples_
    assert samples is not None
    expected_draws = (model.iters - model.burnin) // model.thin
    assert samples.shape == (expected_draws, X.shape[1])
    assert fitted.tau_samples_ is not None and fitted.tau_samples_.shape == (expected_draws,)
    assert fitted.phi_samples_ is not None and fitted.phi_samples_.shape == (expected_draws, len(groups))
    assert fitted.lambda_samples_ is not None and fitted.lambda_samples_.shape == (expected_draws, X.shape[1])
    assert fitted.sigma2_samples_ is not None and fitted.sigma2_samples_.shape == (expected_draws,)

    npt.assert_array_less(0.0, fitted.tau_samples_)
    npt.assert_array_less(0.0, fitted.phi_samples_)
    npt.assert_array_less(0.0, fitted.lambda_samples_)
    npt.assert_array_less(0.0, fitted.sigma2_samples_)

    preds = fitted.predict(X[:5])
    assert preds.shape == (5,)

    summaries = fitted.get_posterior_summaries()
    assert set(["beta_mean", "beta_median", "beta_ci95"]).issubset(summaries.keys())

    beta_mean = summaries["beta_mean"]
    strong_idx = np.array([0, 2])
    weak_idx = np.array([1, 3])
    strong_mean = np.mean(np.abs(beta_mean[strong_idx]))
    weak_mean = np.mean(np.abs(beta_mean[weak_idx]))
    assert strong_mean >= weak_mean
