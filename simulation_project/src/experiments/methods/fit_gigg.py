from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from simulation_project.src.core.diagnostics.convergence import summarize_convergence
from simulation_project.src.core.models.gigg_regression import GIGGRegression

from .helpers import as_int_groups, fit_error_result, scaled_iteration_budget
from ...utils import FitResult, SamplerConfig, diagnostics_summary_for_method, timed_call

# Iteration counts
# Default now follows the unified SamplerConfig budget rather than pinning
# GIGG to a separate 10k+10k protocol. This keeps the formal experimental
# budget aligned across Bayesian methods while still allowing explicit
# paper-reproduction overrides via iter_floor / iter_cap when needed.
_DEFAULT_GIGG_ITER_MULT = 4
_DEFAULT_GIGG_ITER_FLOOR = 10
_DEFAULT_GIGG_ITER_CAP = 10**9


def _validate_gigg_master_alignment(
    *,
    init_strategy: str,
    init_scale_blend: float,
    randomize_group_order: bool,
    lambda_vectorized_update: bool,
    extra_beta_refresh_prob: float,
    mmle_step_size: float | None = None,
    mmle_update_every: int | None = None,
    mmle_window: int | None = None,
    lambda_constraint_mode: str | None = None,
    q_constraint_mode: str | None = None,
) -> None:
    if str(init_strategy).strip().lower() != "zero":
        raise ValueError("GIGG is locked to gigg-master alignment: init_strategy must be 'zero'.")
    if float(init_scale_blend) != 0.0:
        raise ValueError("GIGG is locked to gigg-master alignment: init_scale_blend must be 0.")
    if bool(randomize_group_order):
        raise ValueError("GIGG is locked to gigg-master alignment: randomize_group_order must be False.")
    if bool(lambda_vectorized_update):
        raise ValueError("GIGG is locked to gigg-master alignment: lambda_vectorized_update must be False.")
    if float(extra_beta_refresh_prob) != 0.0:
        raise ValueError("GIGG is locked to gigg-master alignment: extra_beta_refresh_prob must be 0.")
    if mmle_step_size is not None and float(mmle_step_size) != 1.0:
        raise ValueError("GIGG is locked to gigg-master alignment: mmle_step_size must be 1.")
    if mmle_update_every is not None and int(mmle_update_every) != 1:
        raise ValueError("GIGG is locked to gigg-master alignment: mmle_update_every must be 1.")
    if mmle_window is not None and int(mmle_window) != 1:
        raise ValueError("GIGG is locked to gigg-master alignment: mmle_window must be 1.")
    if lambda_constraint_mode is not None and str(lambda_constraint_mode).strip().lower() != "none":
        raise ValueError("GIGG is locked to gigg-master alignment: lambda_constraint_mode must be 'none'.")
    if q_constraint_mode is not None and str(q_constraint_mode).strip().lower() != "hard":
        raise ValueError("GIGG is locked to gigg-master alignment: q_constraint_mode must be 'hard'.")


def _resolve_exact_highdim_gigg_toggles(
    *,
    X: np.ndarray,
    sampler: SamplerConfig,
    btrick: bool,
    num_chains: int,
    lambda_vectorized_update: bool,
    stable_solve: bool,
    min_highdim_chains: int = 2,
) -> dict[str, object]:
    n, p = map(int, X.shape)
    highdim = bool(p > n and p >= 150)
    return {
        "use_btrick": bool(btrick or highdim),
        "num_chains": int(max(num_chains, min_highdim_chains if highdim else num_chains)),
        "lambda_vectorized_update": bool(lambda_vectorized_update or highdim),
        "stable_solve": bool(stable_solve),
        "mmle_highdim_fastpath": bool(highdim),
    }


def _is_highdim_case(X: np.ndarray) -> bool:
    n, p = map(int, X.shape)
    return bool(p > n and p >= 150)


def _run_gigg_model(
    model: GIGGRegression,
    X: np.ndarray,
    y: np.ndarray,
    *,
    groups: Sequence[Sequence[int]],
    method: str | None = None,
    a: Sequence[float] | None = None,
    b: Sequence[float] | None = None,
    beta_inits: np.ndarray | None = None,
    lambda_sq_inits: np.ndarray | None = None,
    gamma_sq_inits: np.ndarray | None = None,
) -> tuple[GIGGRegression, float]:
    fit_kwargs: dict[str, object] = {
        "groups": as_int_groups(groups),
    }
    if method is not None:
        fit_kwargs["method"] = method
    if a is not None:
        fit_kwargs["a"] = a
    if b is not None:
        fit_kwargs["b"] = b
    if beta_inits is not None:
        fit_kwargs["beta_inits"] = np.asarray(beta_inits, dtype=float)
    if lambda_sq_inits is not None:
        fit_kwargs["lambda_sq_inits"] = np.asarray(lambda_sq_inits, dtype=float)
    if gamma_sq_inits is not None:
        fit_kwargs["gamma_sq_inits"] = np.asarray(gamma_sq_inits, dtype=float)
    return timed_call(model.fit, X, y, **fit_kwargs)


def _select_stage_a_beta_inits(
    beta_draws: np.ndarray | None,
    *,
    num_chains: int,
) -> np.ndarray | None:
    if beta_draws is None:
        return None
    arr = np.asarray(beta_draws, dtype=float)
    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])
    elif arr.ndim != 2:
        return None
    if arr.shape[0] <= 0:
        return None
    if int(num_chains) <= 1:
        return np.asarray(arr[-1], dtype=float)
    grid = np.linspace(0.25, 0.75, int(num_chains))
    idx = np.clip(np.round(grid * max(arr.shape[0] - 1, 0)).astype(int), 0, arr.shape[0] - 1)
    return np.asarray(arr[idx], dtype=float)


def _select_stage_a_group_inits(draws: np.ndarray | None, *, num_chains: int) -> np.ndarray | None:
    if draws is None:
        return None
    arr = np.asarray(draws, dtype=float)
    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])
    elif arr.ndim != 2:
        return None
    if arr.shape[0] <= 0:
        return None
    if int(num_chains) <= 1:
        return np.asarray(arr[-1], dtype=float)
    grid = np.linspace(0.25, 0.75, int(num_chains))
    idx = np.clip(np.round(grid * max(arr.shape[0] - 1, 0)).astype(int), 0, arr.shape[0] - 1)
    return np.asarray(arr[idx], dtype=float)


def _gigg_iters(
    sampler: SamplerConfig,
    *,
    iter_mult: int = _DEFAULT_GIGG_ITER_MULT,
    iter_floor: int = _DEFAULT_GIGG_ITER_FLOOR,
    iter_cap: int = _DEFAULT_GIGG_ITER_CAP,
) -> tuple[int, int]:
    return scaled_iteration_budget(
        sampler,
        iter_mult=iter_mult,
        iter_floor=iter_floor,
        iter_cap=iter_cap,
    )


def _extract_and_diagnose(
    model: GIGGRegression,
    method_label: str,
    sampler: SamplerConfig,
    runtime: float,
) -> FitResult:
    tracked = ["beta"]
    beta_draws = getattr(model, "coef_samples_", None)
    beta_mean = getattr(model, "coef_mean_", None)
    gamma2_draws = getattr(model, "gamma2_samples_", None)

    rhat_max, ess_min, div_ratio, converged, details = diagnostics_summary_for_method(
        model=model,
        tracked_params=tracked,
        beta_draws=beta_draws,
        config=sampler,
    )
    if gamma2_draws is not None:
        gamma_detail = summarize_convergence({"gamma2": np.asarray(gamma2_draws, dtype=float)})
        gamma_diag = dict(gamma_detail.get("gamma2", {}))
        gamma_rhat = float(gamma_diag.get("rhat_max", float("nan")))
        gamma_ess = float(gamma_diag.get("ess_min", float("nan")))
        details = dict(details or {})
        details["gigg_auxiliary_scale_diagnostics"] = {
            "tracked_for_convergence": False,
            "reason": "gamma2 is a nuisance group-scale parameter; convergence is gated on beta, matching the GHS_plus Gibbs diagnostic policy.",
            "rhat_max": float(gamma_rhat),
            "ess_min": float(gamma_ess),
            "detail": dict(gamma_detail or {}),
        }
    return FitResult(
        method=method_label,
        status="ok",
        beta_mean=None if beta_mean is None else np.asarray(beta_mean, dtype=float),
        beta_draws=None if beta_draws is None else np.asarray(beta_draws, dtype=float),
        kappa_draws=None,
        group_scale_draws=None if gamma2_draws is None else np.asarray(gamma2_draws, dtype=float),
        runtime_seconds=float(runtime),
        rhat_max=float(rhat_max),
        bulk_ess_min=float(ess_min),
        divergence_ratio=float(div_ratio),
        converged=bool(converged),
        diagnostics=details,
    )


def _extract_draw_diag(
    beta_chains: np.ndarray,
    *,
    gamma2_chains: np.ndarray | None = None,
    lambda_chains: np.ndarray | None = None,
    tau_chains: np.ndarray | None = None,
    sigma_chains: np.ndarray | None = None,
) -> tuple[float, float, dict[str, Any]]:
    draws: dict[str, np.ndarray] = {"beta": np.asarray(beta_chains, dtype=float)}
    if gamma2_chains is not None:
        draws["gamma2"] = np.asarray(gamma2_chains, dtype=float)
    if lambda_chains is not None:
        draws["lambda"] = np.asarray(lambda_chains, dtype=float)
    if tau_chains is not None:
        draws["tau"] = np.asarray(tau_chains, dtype=float)
    if sigma_chains is not None:
        draws["sigma"] = np.asarray(sigma_chains, dtype=float)
    conv = summarize_convergence(draws)
    beta_diag = dict(conv.get("beta", {}))
    rvals: list[float] = []
    evals: list[float] = []
    for item in conv.values():
        if isinstance(item, dict):
            rv = item.get("rhat_max", float("nan"))
            ev = item.get("ess_min", float("nan"))
            if np.isfinite(rv):
                rvals.append(float(rv))
            if np.isfinite(ev):
                evals.append(float(ev))
    return (
        float(max(rvals)) if rvals else float(beta_diag.get("rhat_max", float("nan"))),
        float(min(evals)) if evals else float(beta_diag.get("ess_min", float("nan"))),
        {"convergence_detail": conv},
    )


# GIGG MMLE

def fit_gigg_mmle(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    task: str,
    seed: int,
    sampler: SamplerConfig,
    p0: int = 0,
    iter_mult: int = _DEFAULT_GIGG_ITER_MULT,
    iter_floor: int = _DEFAULT_GIGG_ITER_FLOOR,
    iter_cap: int = _DEFAULT_GIGG_ITER_CAP,
    btrick: bool = False,
    mmle_burnin_only: bool = False,
    init_strategy: str = "zero",
    init_ridge: float = 1.0,
    init_scale_blend: float = 0.0,
    randomize_group_order: bool = False,
    lambda_vectorized_update: bool = False,
    extra_beta_refresh_prob: float = 0.0,
    mmle_step_size: float = 1.0,
    mmle_update_every: int = 1,
    mmle_window: int = 1,
    mmle_samp_size: int = 1000,
    mmle_tol_scale: float = 1e-4,
    mmle_max_iters: int = 50000,
    lambda_constraint_mode: str = "none",
    q_constraint_mode: str = "hard",
    exact_highdim_fastpath: bool = True,
    method_label: str = "GIGG_MMLE",
    no_retry: bool = False,
    progress_bar: bool = True,
    blas_threads_per_chain: int = 1,
) -> FitResult:
    """GIGG with MMLE hyperparameter estimation aligned to gigg-master defaults.

    Fixes a_g = 1/n for all groups and estimates b_g via MMLE using the
    upstream R package's single-chain, zero-init reference path.

    no_retry is accepted for compatibility with experiment-level retry policies.
    Retry control is handled by the caller; this function always runs one fit.
    """
    _ = bool(no_retry)  # caller-level flag; intentionally unused here
    if str(task).lower() == "logistic":
        return fit_error_result(
            str(method_label),
            f"NotImplementedError: {method_label} is gaussian-only in this repository",
        )
    if not bool(exact_highdim_fastpath):
        _validate_gigg_master_alignment(
            init_strategy=init_strategy,
            init_scale_blend=init_scale_blend,
            randomize_group_order=randomize_group_order,
            lambda_vectorized_update=lambda_vectorized_update,
            extra_beta_refresh_prob=extra_beta_refresh_prob,
            mmle_step_size=mmle_step_size,
            mmle_update_every=mmle_update_every,
            mmle_window=mmle_window,
            lambda_constraint_mode=lambda_constraint_mode,
            q_constraint_mode=q_constraint_mode,
        )

    n, _ = X.shape
    gigg_burnin, gigg_draws = _gigg_iters(
        sampler,
        iter_mult=iter_mult,
        iter_floor=iter_floor,
        iter_cap=iter_cap,
    )
    groups_use = as_int_groups(groups)
    highdim_case = bool(exact_highdim_fastpath) and _is_highdim_case(np.asarray(X, dtype=float))
    toggles = _resolve_exact_highdim_gigg_toggles(
        X=np.asarray(X, dtype=float),
        sampler=sampler,
        btrick=bool(btrick),
        num_chains=max(1, int(getattr(sampler, "chains", 1))),
        lambda_vectorized_update=bool(lambda_vectorized_update),
        stable_solve=True,
        min_highdim_chains=2,
    ) if bool(exact_highdim_fastpath) else {
        "use_btrick": bool(btrick),
        "num_chains": 1,
        "lambda_vectorized_update": bool(lambda_vectorized_update),
        "stable_solve": True,
        "mmle_highdim_fastpath": False,
    }

    try:
        model = GIGGRegression(
            method="mmle",
            n_burn_in=gigg_burnin,
            n_samples=gigg_draws,
            n_thin=1,
            seed=int(seed),
            num_chains=int(toggles["num_chains"]),
            fit_intercept=False,
            store_lambda=False,
            tau_sq_init=1.0,
            btrick=bool(toggles["use_btrick"]),
            stable_solve=bool(toggles["stable_solve"]),
            mmle_burnin_only=bool(mmle_burnin_only),
            mmle_highdim_fastpath=bool(toggles["mmle_highdim_fastpath"]),
            init_strategy=str(init_strategy),
            init_ridge=float(init_ridge),
            init_scale_blend=float(init_scale_blend),
            randomize_group_order=bool(randomize_group_order),
            lambda_vectorized_update=bool(toggles["lambda_vectorized_update"]),
            extra_beta_refresh_prob=float(extra_beta_refresh_prob),
            mmle_step_size=float(mmle_step_size),
            mmle_update_every=int(mmle_update_every),
            mmle_window=int(mmle_window),
            mmle_samp_size=int(mmle_samp_size),
            mmle_tol_scale=float(mmle_tol_scale),
            mmle_max_iters=int(mmle_max_iters),
            lambda_constraint_mode=str(lambda_constraint_mode),
            q_constraint_mode=str(q_constraint_mode),
            progress_bar=bool(progress_bar),
            blas_threads_per_chain=int(blas_threads_per_chain),
        )
        final_model, total_runtime = _run_gigg_model(
            model,
            X,
            y,
            groups=groups_use,
            method="mmle",
        )
        b_hat = getattr(final_model, "b_mean_", None)
        result = _extract_and_diagnose(final_model, str(method_label), sampler, float(total_runtime))
        diag = dict(result.diagnostics or {})
        if b_hat is not None:
            diag["mmle_estimate"] = {
                "q_estimate": np.asarray(b_hat, dtype=float).reshape(-1).tolist(),
                "a_estimate": (np.full(len(groups_use), 1.0 / max(int(n), 1), dtype=float)).tolist(),
            }
        diag["exact_highdim_fastpath"] = bool(exact_highdim_fastpath)
        diag["gigg_sampler_name"] = str(method_label)
        diag["gigg_sampler_strategy"] = "high_dim" if bool(highdim_case) else "low_dim"
        diag["gigg_runtime_toggles"] = {
            "btrick": bool(toggles["use_btrick"]),
            "num_chains": int(toggles["num_chains"]),
            "lambda_vectorized_update": bool(toggles["lambda_vectorized_update"]),
            "stable_solve": bool(toggles["stable_solve"]),
            "blas_threads_per_chain": int(blas_threads_per_chain),
            "mmle_highdim_fastpath": bool(toggles["mmle_highdim_fastpath"]),
            "highdim_original_mmle_btrick": bool(highdim_case),
            "path_name": "GIGG_MMLE_HighDim" if bool(highdim_case) else "GIGG_MMLE_LowDim",
        }
        result.diagnostics = diag
        result.group_scale_draws = None if result.group_scale_draws is None else np.asarray(result.group_scale_draws, dtype=float)
        result.method = str(method_label)
        return result
    except Exception as exc:
        return fit_error_result(
            str(method_label),
            f"{type(exc).__name__}: {exc}",
            diagnostics={
                "exception_type": str(type(exc).__name__),
                "exact_highdim_fastpath": bool(exact_highdim_fastpath),
            },
        )


# GIGG Fixed

def fit_gigg_fixed(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    task: str,
    seed: int,
    sampler: SamplerConfig,
    p0: int = 0,
    a_val: float | None = None,
    b_val: float,
    method_label: str,
    iter_mult: int = _DEFAULT_GIGG_ITER_MULT,
    iter_floor: int = _DEFAULT_GIGG_ITER_FLOOR,
    iter_cap: int = _DEFAULT_GIGG_ITER_CAP,
    btrick: bool = False,
    init_strategy: str = "zero",
    init_ridge: float = 1.0,
    init_scale_blend: float = 0.0,
    randomize_group_order: bool = False,
    lambda_vectorized_update: bool = False,
    extra_beta_refresh_prob: float = 0.0,
    lambda_constraint_mode: str = "none",
    q_constraint_mode: str = "hard",
    exact_highdim_fastpath: bool = False,
    no_retry: bool = False,
    progress_bar: bool = True,
    blas_threads_per_chain: int = 1,
) -> FitResult:
    """GIGG with fixed hyperparameters aligned to gigg-master defaults.

    Key variants that reproduce Table 2 / Table 3 comparisons:

        GIGG_b_small  a=1/n, b=1/n  - near-individualistic; best for concentrated signals
        GIGG_GHS      a=1/2, b=1/2  - group horseshoe (special case, Section 2.1)
        GIGG_b_large  a=1/n, b=1    - best for distributed (dense within-group) signals

    Uses the upstream R package's default single-chain initialization path,
    including `tau_sq_init = 1`.
    """
    _ = bool(no_retry)  # caller-level flag; intentionally unused here
    if str(task).lower() == "logistic":
        return fit_error_result(
            method_label,
            "NotImplementedError: GIGG_fixed is gaussian-only in this repository",
        )
    if not bool(exact_highdim_fastpath):
        _validate_gigg_master_alignment(
            init_strategy=init_strategy,
            init_scale_blend=init_scale_blend,
            randomize_group_order=randomize_group_order,
            lambda_vectorized_update=lambda_vectorized_update,
            extra_beta_refresh_prob=extra_beta_refresh_prob,
            lambda_constraint_mode=lambda_constraint_mode,
            q_constraint_mode=q_constraint_mode,
        )

    n, _ = X.shape
    a_fixed = float(a_val) if a_val is not None else 1.0 / max(n, 1)
    gigg_burnin, gigg_draws = _gigg_iters(
        sampler,
        iter_mult=iter_mult,
        iter_floor=iter_floor,
        iter_cap=iter_cap,
    )
    n_groups = len(list(groups))
    toggles = _resolve_exact_highdim_gigg_toggles(
        X=np.asarray(X, dtype=float),
        sampler=sampler,
        btrick=bool(btrick),
        num_chains=1,
        lambda_vectorized_update=bool(lambda_vectorized_update),
        stable_solve=True,
    ) if bool(exact_highdim_fastpath) else {
        "use_btrick": bool(btrick),
        "num_chains": 1,
        "lambda_vectorized_update": bool(lambda_vectorized_update),
        "stable_solve": True,
        "mmle_highdim_fastpath": False,
    }

    try:
        model = GIGGRegression(
            method="fixed",
            n_burn_in=gigg_burnin,
            n_samples=gigg_draws,
            n_thin=1,
            seed=int(seed),
            num_chains=int(toggles["num_chains"]),
            fit_intercept=False,
            store_lambda=False,
            a_value=a_fixed,
            b_init=float(b_val),
            force_a_1_over_n=False,
            tau_sq_init=1.0,
            btrick=bool(toggles["use_btrick"]),
            stable_solve=bool(toggles["stable_solve"]),
            init_strategy=str(init_strategy),
            init_ridge=float(init_ridge),
            init_scale_blend=float(init_scale_blend),
            randomize_group_order=bool(randomize_group_order),
            lambda_vectorized_update=bool(toggles["lambda_vectorized_update"]),
            extra_beta_refresh_prob=float(extra_beta_refresh_prob),
            lambda_constraint_mode=str(lambda_constraint_mode),
            q_constraint_mode=str(q_constraint_mode),
            progress_bar=bool(progress_bar),
            blas_threads_per_chain=int(blas_threads_per_chain),
        )
        a_arr = [a_fixed] * n_groups
        b_arr = [float(b_val)] * n_groups
        model, runtime = timed_call(
            model.fit,
            X,
            y,
            groups=as_int_groups(groups),
            a=a_arr,
            b=b_arr,
        )
        result = _extract_and_diagnose(model, method_label, sampler, runtime)
        diag = dict(result.diagnostics or {})
        diag["exact_highdim_fastpath"] = bool(exact_highdim_fastpath)
        diag["gigg_runtime_toggles"] = {
            "btrick": bool(toggles["use_btrick"]),
            "num_chains": int(toggles["num_chains"]),
            "lambda_vectorized_update": bool(toggles["lambda_vectorized_update"]),
            "stable_solve": bool(toggles["stable_solve"]),
            "blas_threads_per_chain": int(blas_threads_per_chain),
            "mmle_highdim_fastpath": bool(toggles["mmle_highdim_fastpath"]),
        }
        result.diagnostics = diag
        return result
    except Exception as exc:
        return fit_error_result(
            method_label,
            f"{type(exc).__name__}: {exc}",
            diagnostics={
                "exception_type": str(type(exc).__name__),
                "traceback": traceback.format_exc(),
                "exact_highdim_fastpath": bool(exact_highdim_fastpath),
            },
        )

