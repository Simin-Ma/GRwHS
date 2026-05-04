from __future__ import annotations

from typing import Sequence

import numpy as np

from simulation_project.src.core.models.gigg_regression import GIGGRegression

from .helpers import as_int_groups, fit_error_result, scaled_iteration_budget
from ...utils import FitResult, SamplerConfig, diagnostics_summary_for_method, timed_call

# Iteration counts
# Boss et al. (2024) use 10 000 burn-in + 10 000 draws in all simulations
# (Section 5.2). Defaults here match that budget exactly so direct calls to
# fit_gigg_mmle / fit_gigg_fixed reproduce the paper's computational setting.
# floor=cap=10000 means the sampler always runs exactly 10k+10k regardless
# of the HMC warmup/draw budget passed via SamplerConfig.
_DEFAULT_GIGG_ITER_MULT = 4
_DEFAULT_GIGG_ITER_FLOOR = 10000
_DEFAULT_GIGG_ITER_CAP = 10000


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


def _highdim_mmle_stage_a_iters(gigg_burnin: int, gigg_draws: int) -> tuple[int, int]:
    burnin = int(min(gigg_burnin, max(40, min(120, gigg_burnin // 4 if gigg_burnin > 0 else 40))))
    draws = int(min(gigg_draws, max(40, min(120, gigg_draws // 4 if gigg_draws > 0 else 40))))
    return max(10, burnin), max(10, draws)


def _highdim_stage_b_schedule(gigg_burnin: int, gigg_draws: int) -> tuple[int, int]:
    total = int(max(20, gigg_burnin + gigg_draws))
    burnin = int(min(max(50, gigg_burnin // 12 if gigg_burnin > 0 else 50), max(50, total // 8)))
    draws = int(max(20, total - burnin))
    return burnin, draws


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
    tracked = ["beta", "gamma2"]
    beta_draws = getattr(model, "coef_samples_", None)
    beta_mean = getattr(model, "coef_mean_", None)
    gamma2_draws = getattr(model, "gamma2_samples_", None)

    rhat_max, ess_min, div_ratio, converged, details = diagnostics_summary_for_method(
        model=model,
        tracked_params=tracked,
        beta_draws=beta_draws,
        config=sampler,
    )
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
    lambda_constraint_mode: str = "none",
    q_constraint_mode: str = "hard",
    exact_highdim_fastpath: bool = False,
    method_label: str = "GIGG_MMLE",
    no_retry: bool = False,
    progress_bar: bool = True,
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
        b_hat = None
        stage_info: dict[str, object] = {}
        total_runtime = 0.0
        final_model: GIGGRegression

        if highdim_case:
            stage_a_burnin, stage_a_draws = _highdim_mmle_stage_a_iters(gigg_burnin, gigg_draws)
            stage_a_chain_count = 1
            stage_a_toggles = _resolve_exact_highdim_gigg_toggles(
                X=np.asarray(X, dtype=float),
                sampler=sampler,
                btrick=bool(btrick),
                num_chains=stage_a_chain_count,
                lambda_vectorized_update=bool(lambda_vectorized_update),
                stable_solve=True,
                min_highdim_chains=stage_a_chain_count,
            )
            stage_a_model = GIGGRegression(
                method="mmle",
                n_burn_in=stage_a_burnin,
                n_samples=stage_a_draws,
                n_thin=1,
                seed=int(seed),
                num_chains=stage_a_chain_count,
                fit_intercept=False,
                store_lambda=True,
                tau_sq_init=1.0,
                btrick=bool(stage_a_toggles["use_btrick"]),
                stable_solve=bool(stage_a_toggles["stable_solve"]),
                mmle_burnin_only=bool(mmle_burnin_only),
                mmle_highdim_fastpath=bool(stage_a_toggles["mmle_highdim_fastpath"]),
                init_strategy=str(init_strategy),
                init_ridge=float(init_ridge),
                init_scale_blend=float(init_scale_blend),
                randomize_group_order=bool(randomize_group_order),
                lambda_vectorized_update=bool(stage_a_toggles["lambda_vectorized_update"]),
                extra_beta_refresh_prob=float(extra_beta_refresh_prob),
                mmle_step_size=float(mmle_step_size),
                mmle_update_every=int(mmle_update_every),
                mmle_window=int(mmle_window),
                lambda_constraint_mode=str(lambda_constraint_mode),
                q_constraint_mode=str(q_constraint_mode),
                progress_bar=bool(progress_bar),
            )
            stage_a_model, stage_a_runtime = _run_gigg_model(
                stage_a_model,
                X,
                y,
                groups=groups_use,
                method="mmle",
            )
            total_runtime += float(stage_a_runtime)
            b_hat = getattr(stage_a_model, "b_mean_", None)
            if b_hat is None:
                raise RuntimeError("High-dimensional MMLE stage did not produce a q estimate.")
            a_hat = np.full(len(groups_use), 1.0 / max(int(n), 1), dtype=float)
            b_hat_arr = np.asarray(b_hat, dtype=float).reshape(-1)
            beta_seed = _select_stage_a_beta_inits(
                getattr(stage_a_model, "coef_samples_", None),
                num_chains=int(toggles["num_chains"]),
            )
            lambda_sq_seed = _select_stage_a_group_inits(
                getattr(stage_a_model, "lambda_samples_", None),
                num_chains=int(toggles["num_chains"]),
            )
            gamma_sq_seed = _select_stage_a_group_inits(
                getattr(stage_a_model, "gamma2_samples_", None),
                num_chains=int(toggles["num_chains"]),
            )
            tau2_stage_a = getattr(stage_a_model, "tau2_samples_", None)
            sigma2_stage_a = getattr(stage_a_model, "sigma2_samples_", None)
            tau_sq_seed = float(np.mean(np.asarray(tau2_stage_a, dtype=float))) if tau2_stage_a is not None else 1.0
            sigma_sq_seed = float(np.mean(np.asarray(sigma2_stage_a, dtype=float))) if sigma2_stage_a is not None else 1.0
            stage_b_burnin, stage_b_draws = _highdim_stage_b_schedule(gigg_burnin, gigg_draws)
            stage_b_model = GIGGRegression(
                method="fixed",
                n_burn_in=stage_b_burnin,
                n_samples=stage_b_draws,
                n_thin=1,
                seed=int(seed),
                num_chains=int(toggles["num_chains"]),
                fit_intercept=False,
                store_lambda=True,
                a_value=None,
                b_init=float(np.mean(b_hat_arr)) if b_hat_arr.size else float(max(1.0 / max(int(n), 1), 1e-6)),
                tau_sq_init=float(max(tau_sq_seed, 1e-8)),
                sigma_sq_init=float(max(sigma_sq_seed, 1e-8)),
                btrick=bool(toggles["use_btrick"]),
                stable_solve=bool(toggles["stable_solve"]),
                init_strategy=str(init_strategy),
                init_ridge=float(init_ridge),
                init_scale_blend=float(init_scale_blend),
                randomize_group_order=bool(highdim_case or randomize_group_order),
                locals_first_beta_update=False,
                extra_local_scale_sweeps=0,
                lambda_vectorized_update=bool(toggles["lambda_vectorized_update"]),
                extra_beta_refresh_prob=float(extra_beta_refresh_prob),
                lambda_constraint_mode=str(lambda_constraint_mode),
                q_constraint_mode=str(q_constraint_mode),
                progress_bar=bool(progress_bar),
            )
            final_model, stage_b_runtime = _run_gigg_model(
                stage_b_model,
                X,
                y,
                groups=groups_use,
                a=a_hat.tolist(),
                b=b_hat_arr.tolist(),
                beta_inits=None if beta_seed is None else np.asarray(beta_seed, dtype=float),
                lambda_sq_inits=None if lambda_sq_seed is None else np.asarray(lambda_sq_seed, dtype=float),
                gamma_sq_inits=None if gamma_sq_seed is None else np.asarray(gamma_sq_seed, dtype=float),
            )
            total_runtime += float(stage_b_runtime)
            stage_info = {
                "highdim_two_stage": True,
                "stage_a_mmle": {
                    "num_chains": int(stage_a_chain_count),
                    "burnin": int(stage_a_burnin),
                    "draws": int(stage_a_draws),
                    "runtime_seconds": float(stage_a_runtime),
                },
                "stage_b_fixed": {
                    "num_chains": int(toggles["num_chains"]),
                    "burnin": int(stage_b_burnin),
                    "draws": int(stage_b_draws),
                    "total_iters": int(stage_b_burnin + stage_b_draws),
                    "runtime_seconds": float(stage_b_runtime),
                    "tau_sq_init": float(max(tau_sq_seed, 1e-8)),
                    "sigma_sq_init": float(max(sigma_sq_seed, 1e-8)),
                    "beta_init_mode": "stage_a_draw_quantiles" if beta_seed is not None and np.asarray(beta_seed).ndim == 2 else "stage_a_draw_last",
                    "lambda_sq_seeded": bool(lambda_sq_seed is not None),
                    "gamma_sq_seeded": bool(gamma_sq_seed is not None),
                    "randomize_group_order": bool(highdim_case or randomize_group_order),
                    "locals_first_beta_update": False,
                    "extra_local_scale_sweeps": 0,
                    "extra_beta_refresh_prob": float(extra_beta_refresh_prob),
                },
            }
        else:
            model = GIGGRegression(
                method="mmle",
                n_burn_in=gigg_burnin,
                n_samples=gigg_draws,
                n_thin=1,
                seed=int(seed),
                num_chains=int(toggles["num_chains"]),
                fit_intercept=False,
                store_lambda=True,
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
                lambda_constraint_mode=str(lambda_constraint_mode),
                q_constraint_mode=str(q_constraint_mode),
                progress_bar=bool(progress_bar),
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
        if stage_info:
            diag["staged_runtime"] = stage_info
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
            "mmle_highdim_fastpath": bool(toggles["mmle_highdim_fastpath"]),
            "highdim_two_stage_mmle": bool(highdim_case),
            "path_name": "GIGG_MMLE_HighDim" if bool(highdim_case) else "GIGG_MMLE_LowDim",
        }
        result.diagnostics = diag
        result.group_scale_draws = None if result.group_scale_draws is None else np.asarray(result.group_scale_draws, dtype=float)
        result.method = str(method_label)
        return result
    except Exception as exc:
        return fit_error_result(str(method_label), f"{type(exc).__name__}: {exc}")


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
            store_lambda=True,
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
            "mmle_highdim_fastpath": bool(toggles["mmle_highdim_fastpath"]),
        }
        result.diagnostics = diag
        return result
    except Exception as exc:
        return fit_error_result(method_label, f"{type(exc).__name__}: {exc}")

