from __future__ import annotations

from unittest import mock

import numpy as np

from grrhs.models.grrhs_svi_numpyro import GRRHS_SVI_Numpyro


def _toy_dataset(seed: int = 0):
    rng = np.random.default_rng(seed)
    n, p = 24, 4
    X = rng.normal(size=(n, p)).astype(np.float32)
    X -= X.mean(axis=0, keepdims=True)
    X /= X.std(axis=0, keepdims=True)

    beta_true = np.array([1.0, 0.0, -0.8, 0.0], dtype=np.float32)
    y = (X @ beta_true + 0.1 * rng.normal(size=n)).astype(np.float32)

    groups = [[0, 1], [2, 3]]
    return X, y, groups, beta_true


def test_grrhs_svi_numpyro_runs_and_produces_posterior_samples():
    X, y, groups, _ = _toy_dataset()

    with mock.patch("grrhs.models.grrhs_svi_numpyro.progress", lambda iterable, **_: iterable):
        model = GRRHS_SVI_Numpyro(c=1.5, tau0=0.2, eta=0.6, s0=1.0, num_steps=40, lr=5e-2, seed=2024)
        fitted = model.fit(X, y, groups=groups, num_steps=40, num_samples_export=64)

    samples = fitted.coef_samples_
    assert samples is not None and samples.shape == (64, X.shape[1])
    assert fitted.coef_mean_ is not None
    assert fitted.phi_mean_ is not None and fitted.phi_mean_.shape == (len(groups),)
    assert fitted.lambda_mean_ is not None and fitted.lambda_mean_.shape == (X.shape[1],)
    assert fitted.tau_mean_ is not None
    if fitted.sigma_mean_ is not None:
        assert fitted.sigma_mean_ > 0.0

    preds_mean = fitted.predict(X[:6], use_posterior_mean=True)
    preds_draw = fitted.predict(X[:6], use_posterior_mean=False)
    assert preds_mean.shape == (6,)
    assert preds_draw.shape == (6,)

    summaries = fitted.get_posterior_summaries()
    assert "beta_mean" in summaries
    assert summaries["beta_mean"].shape == (X.shape[1],)

    strong_idx = np.array([0, 2])
    weak_idx = np.array([1, 3])
    strong_mean = float(np.mean(np.abs(fitted.coef_mean_[strong_idx])))
    weak_mean = float(np.mean(np.abs(fitted.coef_mean_[weak_idx])))
    assert strong_mean >= weak_mean

    assert np.all(fitted.lambda_mean_ > 0.0)
    assert np.all(fitted.phi_mean_ > 0.0)
