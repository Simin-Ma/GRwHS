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
            "GIGG_MMLE",
            "NotImplementedError: GIGG_MMLE is gaussian-only in this repository",
        )
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

    try:
        model = GIGGRegression(
            method="mmle",
            n_burn_in=gigg_burnin,
            n_samples=gigg_draws,
            n_thin=1,
            seed=int(seed),
            num_chains=1,
            fit_intercept=False,
            store_lambda=True,
            tau_sq_init=1.0,
            btrick=bool(btrick),
            stable_solve=True,
            mmle_burnin_only=bool(mmle_burnin_only),
            init_strategy=str(init_strategy),
            init_ridge=float(init_ridge),
            init_scale_blend=float(init_scale_blend),
            randomize_group_order=bool(randomize_group_order),
            lambda_vectorized_update=bool(lambda_vectorized_update),
            extra_beta_refresh_prob=float(extra_beta_refresh_prob),
            mmle_step_size=float(mmle_step_size),
            mmle_update_every=int(mmle_update_every),
            mmle_window=int(mmle_window),
            lambda_constraint_mode=str(lambda_constraint_mode),
            q_constraint_mode=str(q_constraint_mode),
            progress_bar=bool(progress_bar),
        )
        model, runtime = timed_call(model.fit, X, y, groups=groups_use, method="mmle")
        result = _extract_and_diagnose(model, "GIGG_MMLE", sampler, runtime)
        diag = dict(result.diagnostics or {})
        b_hat = getattr(model, "b_mean_", None)
        if b_hat is not None:
            diag["mmle_estimate"] = {
                "q_estimate": np.asarray(b_hat, dtype=float).reshape(-1).tolist(),
                "a_estimate": (np.full(len(groups_use), 1.0 / max(int(n), 1), dtype=float)).tolist(),
            }
        result.diagnostics = diag
        result.group_scale_draws = None if result.group_scale_draws is None else np.asarray(result.group_scale_draws, dtype=float)
        result.method = "GIGG_MMLE"
        return result
    except Exception as exc:
        return fit_error_result("GIGG_MMLE", f"{type(exc).__name__}: {exc}")


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

    try:
        model = GIGGRegression(
            method="fixed",
            n_burn_in=gigg_burnin,
            n_samples=gigg_draws,
            n_thin=1,
            seed=int(seed),
            num_chains=1,
            fit_intercept=False,
            store_lambda=True,
            a_value=a_fixed,
            b_init=float(b_val),
            force_a_1_over_n=False,
            tau_sq_init=1.0,
            btrick=bool(btrick),
            stable_solve=True,
            init_strategy=str(init_strategy),
            init_ridge=float(init_ridge),
            init_scale_blend=float(init_scale_blend),
            randomize_group_order=bool(randomize_group_order),
            lambda_vectorized_update=bool(lambda_vectorized_update),
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
        return _extract_and_diagnose(model, method_label, sampler, runtime)
    except Exception as exc:
        return fit_error_result(method_label, f"{type(exc).__name__}: {exc}")

