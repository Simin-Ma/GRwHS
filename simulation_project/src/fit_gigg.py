from __future__ import annotations

from typing import Sequence

import numpy as np

from simulation_project.src.core.models.gigg_regression import GIGGRegression

from .utils import FitResult, SamplerConfig, diagnostics_summary_for_method, rhs_style_tau0, timed_call

# ── iteration counts ────────────────────────────────────────────────────────
# Boss et al. (2024) use 10 000 burn-in + 10 000 draws in all simulations.
# We use 4× the HMC warmup (floor 2 000, cap 5 000) so MMLE has enough
# steps to converge b_g from b_init = 0.5, keeping wall-time manageable.
_DEFAULT_GIGG_ITER_MULT = 4
_DEFAULT_GIGG_ITER_FLOOR = 2000
_DEFAULT_GIGG_ITER_CAP = 5000


def _gigg_iters(
    sampler: SamplerConfig,
    *,
    iter_mult: int = _DEFAULT_GIGG_ITER_MULT,
    iter_floor: int = _DEFAULT_GIGG_ITER_FLOOR,
    iter_cap: int = _DEFAULT_GIGG_ITER_CAP,
) -> tuple[int, int]:
    mult = max(1, int(iter_mult))
    floor = max(10, int(iter_floor))
    cap = max(floor, int(iter_cap))
    burnin = min(max(int(sampler.warmup) * mult, floor), cap)
    draws = min(max(int(sampler.post_warmup_draws) * mult, floor), cap)
    return burnin, draws


def _make_fit_result_error(method: str, error: str) -> FitResult:
    return FitResult(
        method=method,
        status="error",
        beta_mean=None,
        beta_draws=None,
        kappa_draws=None,
        group_scale_draws=None,
        runtime_seconds=float("nan"),
        rhat_max=float("nan"),
        bulk_ess_min=float("nan"),
        divergence_ratio=float("nan"),
        converged=False,
        error=error,
        diagnostics={},
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


# ── GIGG MMLE ────────────────────────────────────────────────────────────────

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
    btrick: bool = True,
    mmle_burnin_only: bool = True,
    init_strategy: str = "ridge",
    init_ridge: float = 1.0,
    init_scale_blend: float = 0.5,
    randomize_group_order: bool = False,
    lambda_vectorized_update: bool = False,
    extra_beta_refresh_prob: float = 0.0,
) -> FitResult:
    """GIGG with MMLE hyperparameter estimation (Boss et al. 2024, Section 4.2).

    Fixes a_g = 1/n for all groups and estimates b_g via MMLE during burn-in,
    matching the paper's recommended default procedure.

    τ₀ is calibrated using the Carvalho-Polson-Scott formula τ₀ = p₀/((p-p₀)√n),
    consistent with how RHS and GR_RHS initialize their global scale.
    """
    if str(task).lower() == "logistic":
        return _make_fit_result_error(
            "GIGG_MMLE",
            "NotImplementedError: GIGG_MMLE is gaussian-only in this repository",
        )

    n, p = X.shape
    # CPS τ₀ calibration — same formula used for RHS / GR_RHS
    tau0 = rhs_style_tau0(n=n, p=p, p0=max(int(p0), 1))
    gigg_burnin, gigg_draws = _gigg_iters(
        sampler,
        iter_mult=iter_mult,
        iter_floor=iter_floor,
        iter_cap=iter_cap,
    )

    try:
        model = GIGGRegression(
            method="mmle",
            n_burn_in=gigg_burnin,
            n_samples=gigg_draws,
            n_thin=1,
            seed=int(seed),
            num_chains=int(sampler.chains),
            fit_intercept=False,
            store_lambda=True,
            # a_g = 1/n enforced inside GIGGRegression when force_a_1_over_n=True (default)
            tau_sq_init=float(tau0 ** 2),
            btrick=bool(btrick),
            stable_solve=True,
            mmle_burnin_only=bool(mmle_burnin_only),
            init_strategy=str(init_strategy),
            init_ridge=float(init_ridge),
            init_scale_blend=float(init_scale_blend),
            randomize_group_order=bool(randomize_group_order),
            lambda_vectorized_update=bool(lambda_vectorized_update),
            extra_beta_refresh_prob=float(extra_beta_refresh_prob),
        )
        model, runtime = timed_call(model.fit, X, y, groups=[list(map(int, g)) for g in groups])
        return _extract_and_diagnose(model, "GIGG_MMLE", sampler, runtime)
    except Exception as exc:
        return _make_fit_result_error("GIGG_MMLE", f"{type(exc).__name__}: {exc}")


# ── GIGG Fixed ───────────────────────────────────────────────────────────────

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
    btrick: bool = True,
    init_strategy: str = "ridge",
    init_ridge: float = 1.0,
    init_scale_blend: float = 0.5,
    randomize_group_order: bool = False,
    lambda_vectorized_update: bool = False,
    extra_beta_refresh_prob: float = 0.0,
) -> FitResult:
    """GIGG with fixed hyperparameters (Boss et al. 2024, Table 2 ablation variants).

    Key variants that reproduce Table 2 / Table 3 comparisons:

        GIGG_b_small  a=1/n, b=1/n  — near-individualistic; best for concentrated signals
        GIGG_GHS      a=1/2, b=1/2  — group horseshoe (special case, Section 2.1)
        GIGG_b_large  a=1/n, b=1    — best for distributed (dense within-group) signals

    τ₀ is calibrated via CPS formula, matching GR_RHS / RHS / GIGG_MMLE.
    """
    if str(task).lower() == "logistic":
        return _make_fit_result_error(
            method_label,
            "NotImplementedError: GIGG_fixed is gaussian-only in this repository",
        )

    n, p = X.shape
    a_fixed = float(a_val) if a_val is not None else 1.0 / max(n, 1)
    tau0 = rhs_style_tau0(n=n, p=p, p0=max(int(p0), 1))
    gigg_burnin, gigg_draws = _gigg_iters(
        sampler,
        iter_mult=iter_mult,
        iter_floor=iter_floor,
        iter_cap=iter_cap,
    )
    G = len(list(groups))

    try:
        model = GIGGRegression(
            method="fixed",
            n_burn_in=gigg_burnin,
            n_samples=gigg_draws,
            n_thin=1,
            seed=int(seed),
            num_chains=int(sampler.chains),
            fit_intercept=False,
            store_lambda=True,
            a_value=a_fixed,
            b_init=float(b_val),
            force_a_1_over_n=False,
            tau_sq_init=float(tau0 ** 2),
            btrick=bool(btrick),
            stable_solve=True,
            init_strategy=str(init_strategy),
            init_ridge=float(init_ridge),
            init_scale_blend=float(init_scale_blend),
            randomize_group_order=bool(randomize_group_order),
            lambda_vectorized_update=bool(lambda_vectorized_update),
            extra_beta_refresh_prob=float(extra_beta_refresh_prob),
        )
        a_arr = [a_fixed] * G
        b_arr = [float(b_val)] * G
        model, runtime = timed_call(
            model.fit,
            X,
            y,
            groups=[list(map(int, g)) for g in groups],
            a=a_arr,
            b=b_arr,
        )
        return _extract_and_diagnose(model, method_label, sampler, runtime)
    except Exception as exc:
        return _make_fit_result_error(method_label, f"{type(exc).__name__}: {exc}")
