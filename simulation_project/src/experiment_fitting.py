from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .experiment_runtime import (
    METHODS,
    _attach_retry_diagnostics,
    _invalidate_unconverged_result,
    _is_bayesian_method,
    _resolve_method_list,
    _retry_budget_from_limit,
    _sampler_for_bayesian_default,
    _sampler_for_ghs_plus_default,
    _scale_gigg_config_for_retry,
    _scale_sampler_for_retry,
)
from .utils import FitResult, SamplerConfig


def _fit_all_methods(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    task: str,
    seed: int,
    p0: int,
    p0_groups: int | None = None,
    sampler: SamplerConfig,
    grrhs_kwargs: dict[str, Any] | None = None,
    methods: Sequence[str] | None = None,
    gigg_config: dict[str, Any] | None = None,
    bayes_min_chains: int | None = None,
    enforce_bayes_convergence: bool = True,
    max_convergence_retries: int = 2,
) -> dict[str, FitResult]:
    from .fit_classical import fit_lasso_cv, fit_ols
    from .fit_gigg import fit_gigg_fixed, fit_gigg_mmle
    from .fit_ghs_plus import fit_ghs_plus
    from .fit_gr_rhs import fit_gr_rhs
    from .fit_rhs import fit_rhs

    n = X.shape[0]
    grrhs_kwargs = grrhs_kwargs or {}
    tau_target_use = str(grrhs_kwargs.get("tau_target", "coefficients")).strip().lower()
    grrhs_p0 = int(p0)
    if tau_target_use == "groups" and (p0_groups is not None):
        grrhs_p0 = int(p0_groups)
    methods_use = _resolve_method_list(methods, profile="full") if methods is not None else list(METHODS)
    gigg_cfg = dict(gigg_config or {})
    gigg_extra_retry_cfg = max(0, int(gigg_cfg.pop("extra_retry", 0)))
    gigg_retry_cap_raw = gigg_cfg.pop("retry_cap", None)
    gigg_retry_cap_cfg: int | None = None
    if gigg_retry_cap_raw is not None:
        try:
            gigg_retry_cap_cfg = max(0, int(gigg_retry_cap_raw))
        except Exception:
            gigg_retry_cap_cfg = None

    def _fit_once(method: str, attempt: int) -> FitResult:
        sampler_base = sampler
        if _is_bayesian_method(method):
            sampler_base = _sampler_for_bayesian_default(sampler_base, min_chains=bayes_min_chains)
        if method in {"GIGG_MMLE", "GIGG_b_small", "GIGG_GHS", "GIGG_b_large"}:
            # Keep at least two chains for GIGG diagnostics stability.
            min_gigg_chains = max(2, int(bayes_min_chains) if bayes_min_chains is not None else 0)
            sampler_base = _sampler_for_bayesian_default(sampler_base, min_chains=min_gigg_chains)
        if method == "GHS_plus":
            sampler_base = _sampler_for_ghs_plus_default(sampler_base)
        sampler_try = _scale_sampler_for_retry(sampler_base, attempt)
        gigg_try = _scale_gigg_config_for_retry(gigg_cfg, attempt)
        gigg_mmle_try = dict(gigg_try)
        gigg_fixed_try = {k: v for k, v in gigg_try.items() if k not in {"mmle_burnin_only", "mmle_step_size"}}
        if method == "GR_RHS":
            return fit_gr_rhs(
                X,
                y,
                groups,
                task=task,
                seed=seed + 1 + 100 * attempt,
                p0=grrhs_p0,
                sampler=sampler_try,
                **grrhs_kwargs,
            )
        if method == "RHS":
            return fit_rhs(X, y, groups, task=task, seed=seed + 2 + 100 * attempt, p0=p0, sampler=sampler_try)
        if method == "GIGG_MMLE":
            return fit_gigg_mmle(
                X,
                y,
                groups,
                task=task,
                seed=seed + 3 + 100 * attempt,
                sampler=sampler_try,
                p0=p0,
                **gigg_mmle_try,
            )
        if method == "GIGG_b_small":
            return fit_gigg_fixed(
                X,
                y,
                groups,
                task=task,
                seed=seed + 5 + 100 * attempt,
                sampler=sampler_try,
                p0=p0,
                a_val=1.0 / n,
                b_val=1.0 / n,
                method_label="GIGG_b_small",
                **gigg_fixed_try,
            )
        if method == "GIGG_GHS":
            return fit_gigg_fixed(
                X,
                y,
                groups,
                task=task,
                seed=seed + 6 + 100 * attempt,
                sampler=sampler_try,
                p0=p0,
                a_val=0.5,
                b_val=0.5,
                method_label="GIGG_GHS",
                **gigg_fixed_try,
            )
        if method == "GIGG_b_large":
            return fit_gigg_fixed(
                X,
                y,
                groups,
                task=task,
                seed=seed + 7 + 100 * attempt,
                sampler=sampler_try,
                p0=p0,
                a_val=1.0 / n,
                b_val=1.0,
                method_label="GIGG_b_large",
                **gigg_fixed_try,
            )
        if method == "GHS_plus":
            return fit_ghs_plus(X, y, groups, task=task, seed=seed + 4 + 100 * attempt, p0=p0, sampler=sampler_try)
        if method == "OLS":
            return fit_ols(X, y, task=task, seed=seed + 8)
        if method == "LASSO_CV":
            return fit_lasso_cv(X, y, task=task, seed=seed + 9)
        raise ValueError(f"Unsupported method: {method}")

    retry_max, until_mode = _retry_budget_from_limit(int(max_convergence_retries))
    gigg_methods = {"GIGG_MMLE", "GIGG_b_small", "GIGG_GHS", "GIGG_b_large"}
    gigg_no_retry = bool(gigg_cfg.get("no_retry", False))
    gigg_extra_retry = int(gigg_extra_retry_cfg)
    out: dict[str, FitResult] = {}
    for method in methods_use:
        res: FitResult | None = None
        attempts = 1
        # GIGG methods with no_retry=True run exactly once (paper budget = 10k+10k);
        # non-convergence is reported as-is rather than retried with a larger budget.
        if gigg_no_retry and method in gigg_methods:
            method_retry_max = 0
        else:
            method_retry_max = retry_max + (gigg_extra_retry if method in gigg_methods else 0)
        if method in gigg_methods and (gigg_retry_cap_cfg is not None):
            method_retry_max = min(int(method_retry_max), int(gigg_retry_cap_cfg))
        if bool(enforce_bayes_convergence) and _is_bayesian_method(method):
            for attempt in range(method_retry_max + 1):
                attempts = attempt + 1
                res = _fit_once(method, attempt)
                if bool(res.status == "ok" and res.converged and (res.beta_mean is not None)):
                    break
            assert res is not None
            if not bool(res.status == "ok" and res.converged and (res.beta_mean is not None)):
                res = _invalidate_unconverged_result(res, method=method, attempts=attempts)
        else:
            res = _fit_once(method, 0)
        res = _attach_retry_diagnostics(
            res,
            method=str(method),
            attempts=int(attempts),
            retry_max=retry_max,
            until_mode=until_mode,
            enforce_bayes_convergence=bool(enforce_bayes_convergence),
        )
        out[str(method)] = res
    return out


def _fit_with_convergence_retry(
    fit_fn,
    *,
    method: str,
    sampler: SamplerConfig,
    bayes_min_chains: int | None = None,
    max_convergence_retries: int,
    enforce_bayes_convergence: bool,
) -> FitResult:
    retry_max, until_mode = _retry_budget_from_limit(int(max_convergence_retries))
    sampler_base = sampler
    if _is_bayesian_method(method):
        sampler_base = _sampler_for_bayesian_default(sampler_base, min_chains=bayes_min_chains)
    res: FitResult | None = None
    attempts = 1
    for attempt in range(retry_max + 1):
        attempts = attempt + 1
        sampler_try = _scale_sampler_for_retry(sampler_base, attempt)
        res = fit_fn(sampler_try, attempt)
        if not bool(enforce_bayes_convergence):
            break
        if bool(res.status == "ok" and res.converged and (res.beta_mean is not None)):
            break
    assert res is not None
    if bool(enforce_bayes_convergence) and _is_bayesian_method(method):
        if not bool(res.status == "ok" and res.converged and (res.beta_mean is not None)):
            res = _invalidate_unconverged_result(res, method=method, attempts=attempts)
    return _attach_retry_diagnostics(
        res,
        method=method,
        attempts=attempts,
        retry_max=retry_max,
        until_mode=until_mode,
        enforce_bayes_convergence=bool(enforce_bayes_convergence),
    )
