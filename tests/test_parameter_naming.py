from __future__ import annotations

import numpy as np

from grrhs.models.parameter_naming import unify_parameter_names


class _LegacyModelA:
    def __init__(self) -> None:
        self.coef_samples_ = np.array([[1.0, 2.0], [1.5, 1.8]], dtype=float)
        self.coef_ = np.array([1.2, 1.9], dtype=float)
        self.tau_samples_ = np.array([0.2, 0.3], dtype=float)
        self.lambda_samples_ = np.array([[0.5, 0.6], [0.4, 0.8]], dtype=float)
        self.a_samples_ = np.array([[0.9, 1.1], [1.0, 0.8]], dtype=float)
        self.c2_samples_ = np.array([[1.5, 2.0], [1.2, 1.8]], dtype=float)
        self.sigma2_samples_ = np.array([1.0, 0.8], dtype=float)


class _LegacyModelGIGG:
    def __init__(self) -> None:
        self.coef_samples_ = np.array([[0.1, -0.2], [0.2, -0.1]], dtype=float)
        self.coef_mean_ = np.array([0.15, -0.15], dtype=float)
        self.tau2_samples_ = np.array([0.25, 0.36], dtype=float)
        self.gamma2_samples_ = np.array([[0.49, 0.64], [0.81, 1.00]], dtype=float)
        self.lambda_samples_ = np.array([[0.04, 0.09], [0.16, 0.25]], dtype=float)
        self.sigma_samples_ = np.array([1.0, 1.2], dtype=float)


def test_unify_parameter_names_for_grrhs_like_model() -> None:
    model = _LegacyModelA()
    unify_parameter_names(model)

    assert model.beta_samples_ is not None
    assert model.beta_mean_ is not None
    assert model.global_scale_samples_ is not None
    assert model.global_scale2_samples_ is not None
    assert model.local_scale_samples_ is not None
    assert model.local_scale2_samples_ is not None
    assert model.group_scale_samples_ is not None
    assert model.group_scale2_samples_ is not None
    assert model.slab_scale_samples_ is not None
    assert model.slab_scale2_samples_ is not None
    assert model.noise_scale_samples_ is not None
    assert model.noise_var_samples_ is not None


def test_unify_parameter_names_for_gigg_like_model() -> None:
    model = _LegacyModelGIGG()
    unify_parameter_names(model)

    assert model.global_scale_samples_ is not None
    assert model.global_scale2_samples_ is not None
    assert model.group_scale_samples_ is not None
    assert model.group_scale2_samples_ is not None
    assert model.local_scale_samples_ is not None
    assert model.local_scale2_samples_ is not None
    # GIGG lambda samples are lambda^2 in legacy format; local scale should be sqrt(lambda^2).
    assert np.allclose(model.local_scale_samples_**2, model.local_scale2_samples_, atol=1e-12)

