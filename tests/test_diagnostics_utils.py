from __future__ import annotations

import numpy as np
import numpy.testing as npt
from scipy.stats import geninvgauss

from grrhs.diagnostics.shrinkage import (
    regularized_lambda,
    prior_precision,
    shrinkage_kappa,
    variance_budget_omegas,
    slab_spike_ratio,
    edf_by_group,
)
from grrhs.inference.gig import sample_gig


def test_regularized_lambda_matches_manual_formula():
    lam = np.array([0.5, 1.25, 0.9])
    tau = 0.8
    c = 1.5
    expected = (c ** 2 * lam ** 2) / (c ** 2 + (tau ** 2) * (lam ** 2))
    out = regularized_lambda(lam, tau, c)
    npt.assert_allclose(out, expected, rtol=1e-10)


def test_prior_precision_and_shrinkage_kappa_consistency():
    lam = np.array([0.4, 1.2])
    tau = 0.7
    c = 1.1
    sigma = 0.9
    tilde_sq = regularized_lambda(lam, tau, c)
    phi = np.array([0.8, 1.05])
    d = prior_precision(phi, tau, tilde_sq, sigma)

    XtX_diag = np.array([6.0, 2.5])
    kappa = shrinkage_kappa(XtX_diag, sigma ** 2, d)

    q = XtX_diag / (sigma ** 2)
    expected_kappa = q / (q + d)
    npt.assert_allclose(kappa, expected_kappa, rtol=1e-10)


def test_variance_budget_omegas_sum_to_one():
    phi = np.array([0.9, 1.3, 0.6])
    tau = 0.75
    lam = np.array([0.4, 1.1, 0.7])
    tilde_sq = regularized_lambda(lam, tau, c=1.2)
    tilde = np.sqrt(tilde_sq)

    omegas = variance_budget_omegas(phi, tau, tilde, eps=1e-8)
    total = omegas["omega_group"] + omegas["omega_tau"] + omegas["omega_lambda"]
    npt.assert_allclose(total, np.ones_like(total), rtol=1e-9, atol=1e-9)


def test_slab_spike_ratio_and_edf_by_group():
    lam = np.array([0.6, 1.0, 0.3, 0.8])
    tau = 0.9
    c = 1.4
    ratio = slab_spike_ratio(tau, lam, c)
    expected_ratio = (tau ** 2 * lam ** 2) / (c ** 2)
    npt.assert_allclose(ratio, expected_ratio, rtol=1e-10)

    kappa = np.array([0.2, 0.5, 0.1, 0.4])
    gidx = np.array([0, 0, 1, 1])
    edf = edf_by_group(kappa, gidx, G=2)
    npt.assert_allclose(edf, np.array([0.7, 0.5]), rtol=1e-10)


def test_sample_gig_matches_scipy_parameterization():
    lam = -0.4
    chi = 0.8
    psi = 1.5
    size = 5

    rng_custom = np.random.default_rng(123)
    samples_custom = sample_gig(lam, chi, psi, size=size, rng=rng_custom)

    rng_expected = np.random.default_rng(123)
    samples_expected = geninvgauss.rvs(
        p=lam,
        b=np.sqrt(chi * psi),
        size=size,
        scale=np.sqrt(chi / psi),
        random_state=rng_expected,
    )

    npt.assert_allclose(samples_custom, samples_expected, rtol=1e-10, atol=1e-12)
