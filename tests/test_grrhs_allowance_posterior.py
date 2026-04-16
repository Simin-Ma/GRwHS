from __future__ import annotations

import numpy as np

from grrhs.models.grrhs_gibbs import GRRHS_Gibbs


def _make_grouped_signal_dataset(seed: int = 123):
    rng = np.random.default_rng(seed)
    n, p = 120, 8
    X = rng.normal(size=(n, p))
    X -= X.mean(axis=0, keepdims=True)
    X /= np.maximum(X.std(axis=0, keepdims=True), 1e-8)

    # Group 0 has clear signal; group 1 is null.
    beta_true = np.array([1.8, -1.4, 1.2, 0.8, 0.0, 0.0, 0.0, 0.0], dtype=float)
    y = X @ beta_true + 0.35 * rng.normal(size=n)
    groups = [[0, 1, 2, 3], [4, 5, 6, 7]]
    return X.astype(float), y.astype(float), groups


def test_grrhs_allowance_posterior_sanity_and_constraints():
    X, y, groups = _make_grouped_signal_dataset()
    model = GRRHS_Gibbs(
        tau0=0.2,
        eta=0.6,
        s0=1.0,
        alpha_kappa=2.0,
        beta_kappa=2.0,
        iters=180,
        burnin=90,
        thin=3,
        seed=2026,
        use_pcabs_lite=True,
        use_collapsed_scale_updates=False,
    )
    fitted = model.fit(X, y, groups=groups)

    assert fitted.kappa_samples_ is not None
    assert fitted.c2_samples_ is not None
    assert fitted.sigma2_samples_ is not None
    assert fitted.tau_samples_ is not None
    assert fitted.lambda_samples_ is not None
    assert fitted.a_samples_ is not None
    assert fitted.coef_samples_ is not None

    kappa = np.asarray(fitted.kappa_samples_, dtype=float)  # (S, G)
    c2 = np.asarray(fitted.c2_samples_, dtype=float)  # (S, G)
    sigma2 = np.asarray(fitted.sigma2_samples_, dtype=float).reshape(-1, 1)  # (S, 1)
    tau = np.asarray(fitted.tau_samples_, dtype=float).reshape(-1, 1)  # (S, 1)
    lam = np.asarray(fitted.lambda_samples_, dtype=float)  # (S, p)
    a = np.asarray(fitted.a_samples_, dtype=float)  # (S, G)
    beta = np.asarray(fitted.coef_samples_, dtype=float)  # (S, p)

    assert np.all(np.isfinite(kappa))
    assert np.all(np.isfinite(c2))
    assert np.all(np.isfinite(beta))
    assert np.all((kappa > 0.0) & (kappa < 1.0))

    # Core mapping check: kappa = c2 / (sigma2 + c2)
    kappa_from_c2 = c2 / np.maximum(sigma2 + c2, 1e-12)
    np.testing.assert_allclose(kappa, kappa_from_c2, rtol=1e-7, atol=1e-8)

    # Retention bound check under grouped normal-means interpretation:
    # r_{j,g} = v_{j,g} / (sigma2 + v_{j,g}) <= kappa_g
    group_id = np.empty(X.shape[1], dtype=int)
    for g, idxs in enumerate(groups):
        group_id[np.asarray(idxs, dtype=int)] = g
    a_by_j = a[:, group_id]  # (S, p)
    c2_by_j = c2[:, group_id]  # (S, p)
    kappa_by_j = kappa[:, group_id]  # (S, p)
    tau2 = np.maximum(tau * tau, 1e-12)
    lam2 = np.maximum(lam * lam, 1e-12)
    a2 = np.maximum(a_by_j * a_by_j, 1e-12)
    c2_safe = np.maximum(c2_by_j, 1e-12)
    # v = c2 * tau2 * lam2 * a2 / (c2 + tau2 * lam2 * a2)
    num = c2_safe * tau2 * lam2 * a2
    den = c2_safe + tau2 * lam2 * a2
    v = num / np.maximum(den, 1e-12)
    r = v / np.maximum(sigma2 + v, 1e-12)
    assert float(np.max(r - kappa_by_j)) <= 1e-8

    # Posterior reasonableness sanity:
    # signal group should retain more allowance than null group on average.
    kappa_mean = kappa.mean(axis=0)
    assert float(kappa_mean[0]) > float(kappa_mean[1]) + 0.01

    # Signal group posterior effect sizes should dominate null group.
    beta_abs_mean = np.mean(np.abs(beta), axis=0)
    signal_level = float(np.mean(beta_abs_mean[:4]))
    null_level = float(np.mean(beta_abs_mean[4:]))
    assert signal_level > null_level

