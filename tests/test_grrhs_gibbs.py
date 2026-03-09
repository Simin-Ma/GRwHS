from __future__ import annotations

import numpy as np
import numpy.testing as npt

from grrhs.diagnostics.convergence import summarize_convergence
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


def test_grrhs_gibbs_preserves_multichain_draws_for_convergence():
    X, y, groups, _ = _synthetic_grouped_regression(seed=11)

    model = GRRHS_Gibbs(
        c=1.5,
        tau0=0.2,
        eta=0.6,
        s0=1.0,
        iters=120,
        burnin=60,
        thin=6,
        seed=2026,
        num_chains=2,
        slice_w=0.5,
        slice_m=50,
    )
    fitted = model.fit(X, y, groups)

    assert fitted.coef_samples_ is not None and fitted.coef_samples_.ndim == 3
    assert fitted.tau_samples_ is not None and fitted.tau_samples_.ndim == 2
    assert fitted.phi_samples_ is not None and fitted.phi_samples_.ndim == 3
    assert fitted.lambda_samples_ is not None and fitted.lambda_samples_.ndim == 3
    assert fitted.sigma2_samples_ is not None and fitted.sigma2_samples_.ndim == 2

    convergence = summarize_convergence(
        {
            "beta": fitted.coef_samples_,
            "tau": fitted.tau_samples_,
            "phi": fitted.phi_samples_,
            "lambda": fitted.lambda_samples_,
        }
    )
    assert convergence["beta"]["raw_num_chains"] == 2
    assert convergence["beta"]["diagnostic_valid"] is True
    assert convergence["tau"]["raw_num_chains"] == 2
    assert convergence["tau"]["diagnostic_valid"] is True


def test_grrhs_gibbs_runs_in_p_small_n_large_regime():
    rng = np.random.default_rng(7)
    n, p = 256, 8
    X = rng.normal(size=(n, p))
    X -= X.mean(axis=0, keepdims=True)
    X /= X.std(axis=0, keepdims=True)
    beta_true = np.array([1.0, -0.8, 0.6, 0.0, 0.0, 0.2, 0.0, 0.0])
    y = X @ beta_true + 0.2 * rng.normal(size=n)
    groups = [[0, 1], [2, 3], [4, 5], [6, 7]]

    model = GRRHS_Gibbs(
        c=1.0,
        tau0=0.15,
        eta=0.5,
        s0=1.0,
        iters=80,
        burnin=40,
        thin=4,
        seed=77,
    )
    fitted = model.fit(X, y, groups)

    assert fitted.coef_samples_ is not None
    assert fitted.coef_samples_.shape == (10, p)
    assert np.all(np.isfinite(fitted.coef_samples_))
    assert fitted.sigma2_samples_ is not None
    assert np.all(fitted.sigma2_samples_ > 0.0)
