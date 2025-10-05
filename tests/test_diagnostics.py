from __future__ import annotations

import numpy as np
import numpy.testing as npt

from grwhs.diagnostics.postprocess import (
    anchor_log_group_scales,
    compute_diagnostics_from_samples,
)
from grwhs.diagnostics.shrinkage import (
    edf_by_group,
    prior_precision,
    regularized_lambda,
    shrinkage_kappa,
    slab_spike_ratio,
    variance_budget_omegas,
)


def test_anchor_log_group_scales_removes_location_per_draw():
    raw = np.log(
        np.array(
            [
                [0.5, 1.0, 2.0],
                [1.5, 1.5, 1.5],
                [0.8, 1.6, 1.2],
            ],
            dtype=float,
        )
    )
    anchored = anchor_log_group_scales(raw)
    row_means = anchored.mean(axis=1)
    npt.assert_allclose(row_means, np.zeros(raw.shape[0]), atol=1e-12)
    # Relative shape information is preserved
    npt.assert_allclose(anchored[0] - anchored[0, 0], raw[0] - raw[0, 0])


def test_compute_diagnostics_matches_manual_components():
    X = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    group_index = np.array([0, 0, 1, 1], dtype=int)
    G = 2
    T = 4

    lambda_samples = np.array(
        [
            [0.5, 1.0, 0.8, 1.2],
            [0.6, 0.9, 0.7, 1.1],
            [0.55, 0.95, 0.75, 1.05],
            [0.58, 0.92, 0.73, 1.08],
        ],
        dtype=float,
    )
    tau_samples = np.array([0.7, 0.8, 0.75, 0.78], dtype=float)
    phi_samples = np.array(
        [
            [0.9, 1.1],
            [1.0, 1.05],
            [0.95, 1.08],
            [0.98, 1.04],
        ],
        dtype=float,
    )
    sigma_samples = np.array([0.35, 0.4, 0.38, 0.37], dtype=float)

    c = 1.5
    eps = 1e-8

    result = compute_diagnostics_from_samples(
        X=X,
        group_index=group_index,
        c=c,
        eps=eps,
        lambda_=lambda_samples,
        tau=tau_samples,
        phi=phi_samples,
        sigma=sigma_samples,
    )

    assert result.samples_used == T
    assert result.meta["p"] == X.shape[1]
    assert result.meta["G"] == G
    npt.assert_array_equal(result.per_coeff["group_index"], group_index)

    XtX_diag = np.sum(X * X, axis=0)
    kappa_draws = []
    omega_group_draws = []
    omega_tau_draws = []
    omega_lambda_draws = []
    ratio_draws = []
    edf_draws = []

    for t in range(T):
        phi_j = phi_samples[t, group_index]
        tau_t = tau_samples[t]
        sigma_t = sigma_samples[t]
        lam_t = lambda_samples[t]

        tilde_sq = regularized_lambda(lam_t, tau_t, c)
        tilde = np.sqrt(tilde_sq)
        d = prior_precision(phi_j, tau_t, tilde_sq, sigma_t)
        kappa = shrinkage_kappa(XtX_diag, sigma_t ** 2, d)
        omegas = variance_budget_omegas(phi_j, tau_t, tilde, eps=eps)
        ratio = slab_spike_ratio(tau_t, lam_t, c)
        edf = edf_by_group(kappa, group_index, G=G)

        kappa_draws.append(kappa)
        omega_group_draws.append(omegas["omega_group"])
        omega_tau_draws.append(omegas["omega_tau"])
        omega_lambda_draws.append(omegas["omega_lambda"])
        ratio_draws.append(ratio)
        edf_draws.append(edf)

    kappa_draws = np.stack(kappa_draws, axis=0)
    omega_group_draws = np.stack(omega_group_draws, axis=0)
    omega_tau_draws = np.stack(omega_tau_draws, axis=0)
    omega_lambda_draws = np.stack(omega_lambda_draws, axis=0)
    ratio_draws = np.stack(ratio_draws, axis=0)
    edf_draws = np.stack(edf_draws, axis=0)

    kappa_expected = np.median(kappa_draws, axis=0)
    omega_group_expected = np.median(omega_group_draws, axis=0)
    omega_tau_expected = np.median(omega_tau_draws, axis=0)
    omega_lambda_expected = np.median(omega_lambda_draws, axis=0)
    ratio_expected = np.median(ratio_draws, axis=0)
    pr_ratio_gt_one = (ratio_draws > 1.0).mean(axis=0)
    edf_expected = np.median(edf_draws, axis=0)

    npt.assert_allclose(result.per_coeff["kappa"], kappa_expected, rtol=1e-10)
    npt.assert_allclose(result.per_coeff["omega_group"], omega_group_expected, rtol=1e-10)
    npt.assert_allclose(result.per_coeff["omega_tau"], omega_tau_expected, rtol=1e-10)
    npt.assert_allclose(result.per_coeff["omega_lambda"], omega_lambda_expected, rtol=1e-10)
    npt.assert_allclose(result.per_coeff["r"], ratio_expected, rtol=1e-10)
    npt.assert_allclose(result.per_coeff["Pr_r_gt_1"], pr_ratio_gt_one, rtol=1e-10)
    npt.assert_allclose(result.per_group["edf"], edf_expected, rtol=1e-10)
