from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


def _first_available(model: Any, names: Sequence[str]) -> Any:
    for name in names:
        if not hasattr(model, name):
            continue
        value = getattr(model, name)
        if value is not None:
            return value
    return None


def _as_array_or_none(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return None
    return arr


def _set(model: Any, name: str, value: Any) -> None:
    setattr(model, name, value)


def unify_parameter_names(model: Any) -> Any:
    """
    Add canonical parameter names on a fitted model without breaking legacy names.

    Canonical names added:
    - beta_samples_, beta_mean_
    - noise_scale_samples_, noise_var_samples_
    - global_scale_samples_, global_scale2_samples_
    - local_scale_samples_, local_scale2_samples_
    - group_scale_samples_, group_scale2_samples_
    - slab_scale_samples_, slab_scale2_samples_
    """

    # Coefficients
    coef_samples = _first_available(model, ("coef_samples_", "beta_samples_"))
    coef_mean = _first_available(model, ("coef_mean_", "coef_", "pos_mean_", "beta_mean_"))
    _set(model, "beta_samples_", coef_samples)
    _set(model, "beta_mean_", coef_mean)

    # Noise scale / variance
    sigma = _as_array_or_none(_first_available(model, ("sigma_samples_", "noise_scale_samples_")))
    sigma2 = _as_array_or_none(_first_available(model, ("sigma2_samples_", "noise_var_samples_")))
    if sigma2 is None and sigma is not None:
        sigma2 = np.maximum(sigma, 0.0) ** 2
    if sigma is None and sigma2 is not None:
        sigma = np.sqrt(np.maximum(sigma2, 0.0))
    _set(model, "noise_scale_samples_", sigma)
    _set(model, "noise_var_samples_", sigma2)

    # Global shrinkage
    tau = _as_array_or_none(_first_available(model, ("tau_samples_", "global_scale_samples_")))
    tau2 = _as_array_or_none(_first_available(model, ("tau2_samples_", "global_scale2_samples_")))
    if tau2 is None and tau is not None:
        tau2 = np.maximum(tau, 0.0) ** 2
    if tau is None and tau2 is not None:
        tau = np.sqrt(np.maximum(tau2, 0.0))
    _set(model, "global_scale_samples_", tau)
    _set(model, "global_scale2_samples_", tau2)

    # Local shrinkage
    local = _as_array_or_none(_first_available(model, ("local_scale_samples_",)))
    local2 = _as_array_or_none(_first_available(model, ("local_scale2_samples_",)))
    raw_lambda = _as_array_or_none(_first_available(model, ("lambda_samples_", "lambda_tilde_samples_")))
    if local is None and local2 is None and raw_lambda is not None:
        # GIGG stores lambda^2 in lambda_samples_, others mostly store lambda.
        looks_like_gigg = hasattr(model, "gamma2_samples_") or hasattr(model, "tau2_samples_")
        if looks_like_gigg:
            local2 = np.maximum(raw_lambda, 0.0)
            local = np.sqrt(local2)
        else:
            local = np.maximum(raw_lambda, 0.0)
            local2 = local**2
    if local2 is None and local is not None:
        local2 = np.maximum(local, 0.0) ** 2
    if local is None and local2 is not None:
        local = np.sqrt(np.maximum(local2, 0.0))
    _set(model, "local_scale_samples_", local)
    _set(model, "local_scale2_samples_", local2)

    # Group shrinkage
    group = _as_array_or_none(
        _first_available(
            model,
            (
                "group_scale_samples_",
                "phi_samples_",
                "a_samples_",
                "gamma_samples_",
                "lambda_group_samples_",
                "group_lambda_samples_",
            ),
        )
    )
    group2 = _as_array_or_none(_first_available(model, ("group_scale2_samples_", "gamma2_samples_")))
    if group2 is None and group is not None:
        group2 = np.maximum(group, 0.0) ** 2
    if group is None and group2 is not None:
        group = np.sqrt(np.maximum(group2, 0.0))
    _set(model, "group_scale_samples_", group)
    _set(model, "group_scale2_samples_", group2)

    # Slab scale
    slab = _as_array_or_none(_first_available(model, ("slab_scale_samples_", "c_samples_")))
    slab2 = _as_array_or_none(_first_available(model, ("slab_scale2_samples_", "c2_samples_")))
    if slab2 is None and slab is not None:
        slab2 = np.maximum(slab, 0.0) ** 2
    if slab is None and slab2 is not None:
        slab = np.sqrt(np.maximum(slab2, 0.0))
    _set(model, "slab_scale_samples_", slab)
    _set(model, "slab_scale2_samples_", slab2)

    # Extra cross-model convenience aliases
    if coef_samples is not None and not hasattr(model, "coef_samples_"):
        _set(model, "coef_samples_", coef_samples)
    if coef_mean is not None and not hasattr(model, "coef_mean_"):
        _set(model, "coef_mean_", coef_mean)

    return model


__all__ = ["unify_parameter_names"]

