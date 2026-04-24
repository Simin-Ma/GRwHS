from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Sequence

import numpy as np

from .method_registry import MethodContext, build_default_method_registry
from .runtime import (
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
from ..utils import FitResult, SamplerConfig


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
    method_jobs: int = 1,
    ghs_plus_profile: str = "default",
) -> dict[str, FitResult]:
    n = X.shape[0]
    grrhs_kwargs = grrhs_kwargs or {}
    tau_target_use = str(grrhs_kwargs.get("tau_target", "coefficients")).strip().lower()
    grrhs_p0 = int(p0)
    if tau_target_use == "groups" and (p0_groups is not None):
        grrhs_p0 = int(p0_groups)
    methods_use = _resolve_method_list(methods) if methods is not None else list(METHODS)
    gigg_cfg = dict(gigg_config or {})
    gigg_extra_retry_cfg = max(0, int(gigg_cfg.pop("extra_retry", 0)))
    gigg_retry_cap_raw = gigg_cfg.pop("retry_cap", None)
    gigg_retry_cap_cfg: int | None = None
    if gigg_retry_cap_raw is not None:
        try:
            gigg_retry_cap_cfg = max(0, int(gigg_retry_cap_raw))
        except Exception:
            gigg_retry_cap_cfg = None

    registry = build_default_method_registry()

    def _fit_once(method: str, attempt: int) -> FitResult:
        sampler_base = sampler
        if _is_bayesian_method(method):
            sampler_base = _sampler_for_bayesian_default(sampler_base, min_chains=bayes_min_chains)
        if method in {"GIGG_MMLE", "GIGG_b_small", "GIGG_GHS", "GIGG_b_large"}:
            # Keep at least two chains for GIGG diagnostics stability.
            min_gigg_chains = max(2, int(bayes_min_chains) if bayes_min_chains is not None else 0)
            sampler_base = _sampler_for_bayesian_default(sampler_base, min_chains=min_gigg_chains)
        if method == "GHS_plus":
            sampler_base = _sampler_for_ghs_plus_default(sampler_base, profile=ghs_plus_profile)
        sampler_try = _scale_sampler_for_retry(sampler_base, attempt)
        gigg_try = _scale_gigg_config_for_retry(gigg_cfg, attempt)
        gigg_mmle_try = dict(gigg_try)
        gigg_fixed_try = {k: v for k, v in gigg_try.items() if k not in {"mmle_burnin_only", "mmle_step_size"}}
        offset_map = {
            "GR_RHS": 1,
            "RHS": 2,
            "GIGG_MMLE": 3,
            "GHS_plus": 4,
            "GIGG_b_small": 5,
            "GIGG_GHS": 6,
            "GIGG_b_large": 7,
            "OLS": 8,
            "LASSO_CV": 9,
        }
        if method not in offset_map:
            raise ValueError(f"Unsupported method: {method}")
        ctx = MethodContext(
            X=X,
            y=y,
            groups=[list(map(int, g)) for g in groups],
            task=task,
            seed=int(seed + offset_map[method] + 100 * attempt),
            p0=int(p0),
            grrhs_p0=int(grrhs_p0),
            n=int(n),
            sampler=sampler_try,
            grrhs_kwargs={**dict(grrhs_kwargs), "retry_attempt": int(attempt)},
            gigg_mmle_kwargs=gigg_mmle_try,
            gigg_fixed_kwargs=gigg_fixed_try,
        )
        return registry.run(method, ctx)

    retry_max, until_mode = _retry_budget_from_limit(int(max_convergence_retries))
    gigg_methods = {"GIGG_MMLE", "GIGG_b_small", "GIGG_GHS", "GIGG_b_large"}
    gigg_extra_retry = int(gigg_extra_retry_cfg)
    def _run_single_method(method: str) -> tuple[str, FitResult]:
        res: FitResult | None = None
        attempts = 1
        # Keep GIGG aligned to the gigg-master reference path: one run and no
        # experiment-level rescue retries.
        if method in gigg_methods:
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
        return str(method), res

    out: dict[str, FitResult] = {}
    workers = max(1, min(int(method_jobs), len(methods_use)))
    if workers <= 1 or len(methods_use) <= 1:
        for method in methods_use:
            key, res = _run_single_method(method)
            out[key] = res
        return out

    done: dict[str, FitResult] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_map = {ex.submit(_run_single_method, method): str(method) for method in methods_use}
        for fut in as_completed(fut_map):
            key, res = fut.result()
            done[key] = res
    out = {str(method): done[str(method)] for method in methods_use}
    return out


def _fit_with_convergence_retry(
    fit_fn,
    *,
    method: str,
    sampler: SamplerConfig,
    bayes_min_chains: int | None = None,
    max_convergence_retries: int,
    enforce_bayes_convergence: bool,
    continue_on_retry: bool = False,
) -> FitResult:
    retry_max, until_mode = _retry_budget_from_limit(int(max_convergence_retries))
    sampler_base = sampler
    if _is_bayesian_method(method):
        sampler_base = _sampler_for_bayesian_default(sampler_base, min_chains=bayes_min_chains)
    res: FitResult | None = None
    attempts = 1
    resume_payload: dict[str, Any] | None = None
    for attempt in range(retry_max + 1):
        attempts = attempt + 1
        sampler_try = _scale_sampler_for_retry(sampler_base, attempt)
        try:
            res = fit_fn(sampler_try, attempt, resume_payload)
        except TypeError:
            # Backwards compatible path for existing fit_fn signatures.
            res = fit_fn(sampler_try, attempt)
        if bool(continue_on_retry):
            payload = None
            if isinstance(res.diagnostics, dict):
                maybe = res.diagnostics.get("retry_resume_payload")
                if isinstance(maybe, dict):
                    payload = maybe
            resume_payload = payload
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


