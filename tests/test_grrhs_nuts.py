from __future__ import annotations

import numpy as np

from grrhs.models.grrhs_nuts import GRRHS_NUTS


def _synthetic_grouped_regression(seed: int = 0):
    rng = np.random.default_rng(seed)
    n, p = 36, 6
    X = rng.normal(size=(n, p)).astype(np.float32)
    beta_true = np.array([1.0, 0.0, -0.8, 0.0, 0.7, 0.0], dtype=np.float32)
    y = (X @ beta_true + 0.2 * rng.normal(size=n)).astype(np.float32)
    groups = [[0, 1], [2, 3], [4, 5]]
    return X, y, groups


def test_grrhs_nuts_runs_and_exports_key_posterior_blocks():
    X, y, groups = _synthetic_grouped_regression(seed=7)
    model = GRRHS_NUTS(
        tau0=None,
        p0=2.0,
        eta=0.5,
        alpha_kappa=2.0,
        beta_kappa=8.0,
        num_warmup=60,
        num_samples=60,
        num_chains=1,
        thinning=1,
        target_accept_prob=0.9,
        max_tree_depth=8,
        dense_mass=False,
        progress_bar=False,
        seed=2026,
    )

    fitted = model.fit(X, y, groups=groups)
    assert fitted.coef_samples_ is not None
    assert fitted.tau_samples_ is not None
    assert fitted.lambda_samples_ is not None
    assert fitted.a_samples_ is not None
    assert fitted.kappa_samples_ is not None
    assert fitted.c2_samples_ is not None

    assert fitted.coef_samples_.shape == (60, X.shape[1])
    assert fitted.a_samples_.shape[1] == len(groups)
    assert fitted.kappa_samples_.shape[1] == len(groups)
    assert np.all(np.asarray(fitted.kappa_samples_) > 0.0)
    assert np.all(np.asarray(fitted.kappa_samples_) < 1.0)

    pred = fitted.predict(X[:5])
    assert pred.shape == (5,)

    summaries = fitted.get_posterior_summaries()
    assert "kappa_mean" in summaries
    assert summaries["kappa_mean"].shape == (len(groups),)

    sampler_diag = getattr(fitted, "sampler_diagnostics_", {})
    assert isinstance(sampler_diag, dict)
    assert "hmc" in sampler_diag
    assert "posterior_quality" in sampler_diag
    assert "parameterization" in sampler_diag
