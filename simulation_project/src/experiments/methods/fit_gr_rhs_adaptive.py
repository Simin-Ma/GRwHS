from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .fit_gr_rhs import fit_gr_rhs
from ...utils import FitResult, SamplerConfig


@dataclass(frozen=True)
class AdaptiveBetaCalibration:
    strategy: str
    alpha_kappa: float
    beta_kappa: float
    details: dict[str, Any]


def _sampler_with_budget(
    sampler: SamplerConfig,
    *,
    warmup: int | None,
    draws: int | None,
    adapt_delta: float | None = None,
    max_treedepth: int | None = None,
) -> SamplerConfig:
    return SamplerConfig(
        chains=int(sampler.chains),
        warmup=int(sampler.warmup if warmup is None else warmup),
        post_warmup_draws=int(sampler.post_warmup_draws if draws is None else draws),
        adapt_delta=float(sampler.adapt_delta if adapt_delta is None else adapt_delta),
        max_treedepth=int(sampler.max_treedepth if max_treedepth is None else max_treedepth),
        strict_adapt_delta=float(sampler.strict_adapt_delta),
        strict_max_treedepth=int(sampler.strict_max_treedepth),
        max_divergence_ratio=float(sampler.max_divergence_ratio),
        rhat_threshold=float(sampler.rhat_threshold),
        ess_threshold=float(sampler.ess_threshold),
    )


def _split_train_validation(n: int, *, validation_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n_use = int(n)
    if n_use < 8:
        raise ValueError("Need at least 8 observations for adaptive validation calibration.")
    frac = min(max(float(validation_fraction), 0.05), 0.5)
    n_valid = int(max(4, min(n_use - 4, round(frac * n_use))))
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n_use)
    return np.sort(perm[n_valid:]), np.sort(perm[:n_valid])


def _posterior_predictive_lpd(result: FitResult, X_fit: np.ndarray, y_fit: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray) -> float:
    from ..analysis.metrics import compute_test_lpd, compute_test_lpd_ppd

    if result.beta_mean is None:
        return float("nan")
    resid = np.asarray(y_fit, dtype=float).reshape(-1) - np.asarray(X_fit, dtype=float) @ np.asarray(result.beta_mean, dtype=float).reshape(-1)
    sigma2_hat = float(max(np.mean(resid * resid), 1e-8))
    lpd_ppd = compute_test_lpd_ppd(result.beta_draws, X_valid, y_valid, sigma2_hat=sigma2_hat)
    if np.isfinite(lpd_ppd):
        return float(lpd_ppd)
    return float(compute_test_lpd(result.beta_mean, X_valid, y_valid, sigma2_hat=sigma2_hat))


def _group_scores(X: np.ndarray, y: np.ndarray, groups: Sequence[Sequence[int]]) -> np.ndarray:
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    y_cent = y_arr - float(np.mean(y_arr))
    n = max(int(X_arr.shape[0]), 1)
    out: list[float] = []
    for group in groups:
        idx = np.asarray(group, dtype=int)
        if idx.size == 0:
            out.append(float("nan"))
            continue
        raw = (X_arr[:, idx].T @ y_cent) / float(n)
        out.append(float(np.linalg.norm(raw, ord=2) / math.sqrt(max(int(idx.size), 1))))
    return np.asarray(out, dtype=float)


def _ridge_beta_mean(X: np.ndarray, y: np.ndarray, *, ridge: float) -> np.ndarray:
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    y_cent = y_arr - float(np.mean(y_arr))
    p = int(X_arr.shape[1])
    lhs = X_arr.T @ X_arr + float(max(ridge, 1e-8)) * np.eye(p, dtype=float)
    rhs = X_arr.T @ y_cent
    try:
        beta = np.linalg.solve(lhs, rhs)
    except Exception:
        beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return np.nan_to_num(np.asarray(beta, dtype=float), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)


def _ridge_group_scores(X: np.ndarray, y: np.ndarray, groups: Sequence[Sequence[int]], *, ridge: float) -> np.ndarray:
    beta = _ridge_beta_mean(X, y, ridge=float(ridge))
    out: list[float] = []
    for group in groups:
        idx = np.asarray(group, dtype=int)
        if idx.size == 0:
            out.append(float("nan"))
            continue
        out.append(float(np.linalg.norm(beta[idx], ord=2) / math.sqrt(max(int(idx.size), 1))))
    return np.asarray(out, dtype=float)


def calibrate_grrhs_beta_screening_moment(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    alpha_kappa: float = 0.5,
    null_quantile: float = 0.90,
    n_permutations: int = 500,
    min_beta_kappa: float | None = 1.0,
    max_beta_kappa: float | None = None,
    seed: int = 1,
) -> AdaptiveBetaCalibration:
    group_list = [list(map(int, g)) for g in groups]
    G = max(len(group_list), 1)
    observed = _group_scores(X, y, group_list)
    perm_scores = np.zeros((int(n_permutations), G), dtype=float)
    y_cent = np.asarray(y, dtype=float).reshape(-1)
    y_cent = y_cent - float(np.mean(y_cent))
    rng = np.random.default_rng(int(seed))
    for b in range(int(n_permutations)):
        perm_scores[b, :] = _group_scores(X, rng.permutation(y_cent), group_list)
    thresholds = np.nanquantile(perm_scores, float(null_quantile), axis=0)
    active_mask = np.asarray(observed > thresholds, dtype=bool)
    pi_raw = float(np.sum(active_mask) / float(G))
    pi_hat = float(min(max(pi_raw, 1.0 / float(G)), 0.5))
    beta_raw = float(float(alpha_kappa) * (1.0 - pi_hat) / max(pi_hat, 1e-12))
    beta_hat = float(beta_raw)
    if min_beta_kappa is not None:
        beta_hat = float(max(beta_hat, float(min_beta_kappa)))
    if max_beta_kappa is not None:
        beta_hat = float(min(beta_hat, float(max_beta_kappa)))
    return AdaptiveBetaCalibration(
        strategy="screening_moment",
        alpha_kappa=float(alpha_kappa),
        beta_kappa=float(beta_hat),
        details={
            "null_quantile": float(null_quantile),
            "n_permutations": int(n_permutations),
            "n_groups": int(G),
            "n_screen_active_groups": int(np.sum(active_mask)),
            "pi_raw": float(pi_raw),
            "pi_hat": float(pi_hat),
            "beta_kappa_raw": float(beta_raw),
            "min_beta_kappa": None if min_beta_kappa is None else float(min_beta_kappa),
            "max_beta_kappa": None if max_beta_kappa is None else float(max_beta_kappa),
        },
    )


def calibrate_grrhs_beta_ridge_screening_moment(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    alpha_kappa: float = 0.5,
    null_quantile: float = 0.95,
    n_permutations: int = 300,
    ridge_scale: str = "sqrt_np",
    min_beta_kappa: float | None = 1.0,
    max_beta_kappa: float | None = 16.0,
    seed: int = 1,
) -> AdaptiveBetaCalibration:
    group_list = [list(map(int, g)) for g in groups]
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    n, p = int(X_arr.shape[0]), int(X_arr.shape[1])
    mode = str(ridge_scale).strip().lower()
    if mode == "sqrt_np":
        ridge = float(math.sqrt(max(n * p, 1)))
    elif mode == "n":
        ridge = float(max(n, 1))
    elif mode == "p":
        ridge = float(max(p, 1))
    else:
        ridge = float(max(float(ridge_scale), 1e-8))
    G = max(len(group_list), 1)
    observed = _ridge_group_scores(X_arr, y_arr, group_list, ridge=ridge)
    perm_scores = np.zeros((int(n_permutations), G), dtype=float)
    rng = np.random.default_rng(int(seed))
    for b in range(int(n_permutations)):
        perm_scores[b, :] = _ridge_group_scores(X_arr, rng.permutation(y_arr), group_list, ridge=ridge)
    thresholds = np.nanquantile(perm_scores, float(null_quantile), axis=0)
    active_mask = np.asarray(observed > thresholds, dtype=bool)
    pi_raw = float(np.sum(active_mask) / float(G))
    pi_hat = float(min(max(pi_raw, 1.0 / float(G)), 0.5))
    beta_raw = float(float(alpha_kappa) * (1.0 - pi_hat) / max(pi_hat, 1e-12))
    beta_hat = float(beta_raw)
    if min_beta_kappa is not None:
        beta_hat = float(max(beta_hat, float(min_beta_kappa)))
    if max_beta_kappa is not None:
        beta_hat = float(min(beta_hat, float(max_beta_kappa)))
    return AdaptiveBetaCalibration(
        strategy="ridge_screening_moment",
        alpha_kappa=float(alpha_kappa),
        beta_kappa=float(beta_hat),
        details={
            "null_quantile": float(null_quantile),
            "n_permutations": int(n_permutations),
            "ridge_scale": str(ridge_scale),
            "ridge": float(ridge),
            "n_groups": int(G),
            "n_screen_active_groups": int(np.sum(active_mask)),
            "pi_raw": float(pi_raw),
            "pi_hat": float(pi_hat),
            "beta_kappa_raw": float(beta_raw),
            "min_beta_kappa": None if min_beta_kappa is None else float(min_beta_kappa),
            "max_beta_kappa": None if max_beta_kappa is None else float(max_beta_kappa),
        },
    )


def calibrate_grrhs_beta_validation_opt(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    task: str,
    seed: int,
    p0: int,
    sampler: SamplerConfig,
    grrhs_kwargs: dict[str, Any],
    alpha_kappa: float = 0.5,
    validation_fraction: float = 0.25,
    log_beta_bounds: tuple[float, float] = (math.log(0.5), math.log(16.0)),
    n_initial_points: int = 5,
    n_refine_points: int = 3,
    calibration_warmup: int | None = None,
    calibration_draws: int | None = None,
    min_beta_kappa: float | None = 1.0,
    max_beta_kappa: float | None = None,
) -> AdaptiveBetaCalibration:
    """Continuous validation EB calibration for GR-RHS beta_kappa.

    This is not MMLE. It optimizes a training-internal posterior predictive
    density over a fixed continuous log(beta_kappa) interval and then refits
    the final posterior with the selected value.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    fit_idx, valid_idx = _split_train_validation(X_arr.shape[0], validation_fraction=validation_fraction, seed=int(seed) + 177)
    X_fit = X_arr[fit_idx]
    y_fit = y_arr[fit_idx]
    X_valid = X_arr[valid_idx]
    y_valid = y_arr[valid_idx]

    lo, hi = float(log_beta_bounds[0]), float(log_beta_bounds[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        raise ValueError("log_beta_bounds must be finite increasing values.")
    cal_sampler = _sampler_with_budget(
        sampler,
        warmup=calibration_warmup,
        draws=calibration_draws,
    )

    def eval_beta(beta_value: float, idx: int) -> dict[str, Any]:
        kwargs = dict(grrhs_kwargs)
        kwargs["alpha_kappa"] = float(alpha_kappa)
        kwargs["beta_kappa"] = float(beta_value)
        kwargs.setdefault("progress_bar", False)
        result = fit_gr_rhs(
            X_fit,
            y_fit,
            groups,
            task=str(task),
            seed=int(seed) + 1009 + 31 * int(idx),
            p0=int(p0),
            sampler=cal_sampler,
            method_name="GR_RHS_EB_CAL",
            retry_resume_payload=None,
            **{
                **kwargs,
                "retry_attempt": 0,
                "collapsed_hard_min_warmup": int(cal_sampler.warmup),
                "collapsed_hard_min_draws": int(cal_sampler.post_warmup_draws),
            },
        )
        lpd = _posterior_predictive_lpd(result, X_fit, y_fit, X_valid, y_valid)
        return {
            "beta_kappa": float(beta_value),
            "log_beta_kappa": float(math.log(float(beta_value))),
            "validation_lpd": float(lpd) if np.isfinite(lpd) else float("nan"),
            "converged": bool(result.converged),
            "status": str(result.status),
            "rhat_max": float(result.rhat_max) if np.isfinite(result.rhat_max) else float("nan"),
            "ess_min": float(result.bulk_ess_min) if np.isfinite(result.bulk_ess_min) else float("nan"),
        }

    logs = np.linspace(lo, hi, max(3, int(n_initial_points)))
    candidates: list[dict[str, Any]] = []
    seen: set[float] = set()
    for i, logv in enumerate(logs):
        beta = float(math.exp(float(logv)))
        key = round(beta, 10)
        seen.add(key)
        candidates.append(eval_beta(beta, i))

    finite = [c for c in candidates if np.isfinite(float(c.get("validation_lpd", float("nan"))))]
    converged = [c for c in finite if bool(c.get("converged"))]
    basis = converged if converged else finite
    if not basis:
        raise RuntimeError("GR-RHS adaptive validation calibration produced no finite candidates.")
    best = max(basis, key=lambda c: float(c["validation_lpd"]))

    if int(n_refine_points) > 0:
        best_log = float(best["log_beta_kappa"])
        spacing = float((hi - lo) / max(len(logs) - 1, 1))
        refine_logs = np.linspace(max(lo, best_log - spacing), min(hi, best_log + spacing), int(n_refine_points) + 2)[1:-1]
        for logv in refine_logs:
            beta = float(math.exp(float(logv)))
            key = round(beta, 10)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(eval_beta(beta, len(candidates)))
        finite = [c for c in candidates if np.isfinite(float(c.get("validation_lpd", float("nan"))))]
        converged = [c for c in finite if bool(c.get("converged"))]
        basis = converged if converged else finite
        best = max(basis, key=lambda c: float(c["validation_lpd"]))

    beta_hat = float(best["beta_kappa"])
    if min_beta_kappa is not None:
        beta_hat = float(max(beta_hat, float(min_beta_kappa)))
    if max_beta_kappa is not None:
        beta_hat = float(min(beta_hat, float(max_beta_kappa)))
    return AdaptiveBetaCalibration(
        strategy="validation_opt",
        alpha_kappa=float(alpha_kappa),
        beta_kappa=float(beta_hat),
        details={
            "validation_fraction": float(validation_fraction),
            "log_beta_bounds": [float(lo), float(hi)],
            "n_initial_points": int(n_initial_points),
            "n_refine_points": int(n_refine_points),
            "calibration_warmup": int(cal_sampler.warmup),
            "calibration_draws": int(cal_sampler.post_warmup_draws),
            "selected_before_clipping": float(best["beta_kappa"]),
            "min_beta_kappa": None if min_beta_kappa is None else float(min_beta_kappa),
            "max_beta_kappa": None if max_beta_kappa is None else float(max_beta_kappa),
            "candidates": candidates,
        },
    )


def fit_gr_rhs_adaptive_beta(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    task: str,
    seed: int,
    p0: int,
    sampler: SamplerConfig,
    method_name: str = "GR_RHS_EB",
    adaptive_strategy: str = "validation_opt",
    alpha_kappa: float = 0.5,
    beta_kappa: float = 1.0,
    calibration_warmup: int | None = None,
    calibration_draws: int | None = None,
    validation_fraction: float = 0.25,
    log_beta_min: float = math.log(0.5),
    log_beta_max: float = math.log(16.0),
    n_initial_points: int = 5,
    n_refine_points: int = 3,
    screening_null_quantile: float = 0.90,
    screening_permutations: int = 500,
    ridge_screening_scale: str = "sqrt_np",
    min_beta_kappa: float | None = 1.0,
    max_beta_kappa: float | None = None,
    **grrhs_kwargs: Any,
) -> FitResult:
    total_t0 = time.perf_counter()
    kwargs = dict(grrhs_kwargs)
    kwargs.pop("adaptive_strategy", None)
    kwargs.pop("calibration_warmup", None)
    kwargs.pop("calibration_draws", None)
    kwargs.pop("validation_fraction", None)
    kwargs.pop("log_beta_min", None)
    kwargs.pop("log_beta_max", None)
    kwargs.pop("n_initial_points", None)
    kwargs.pop("n_refine_points", None)
    kwargs.pop("screening_null_quantile", None)
    kwargs.pop("screening_permutations", None)
    kwargs.pop("ridge_screening_scale", None)
    kwargs.pop("min_beta_kappa", None)
    kwargs.pop("max_beta_kappa", None)
    kwargs.setdefault("progress_bar", False)

    strategy = str(adaptive_strategy).strip().lower()
    if strategy in {"screening", "screening_moment"}:
        calib = calibrate_grrhs_beta_screening_moment(
            X,
            y,
            groups,
            alpha_kappa=float(alpha_kappa),
            null_quantile=float(screening_null_quantile),
            n_permutations=int(screening_permutations),
            min_beta_kappa=min_beta_kappa,
            max_beta_kappa=max_beta_kappa,
            seed=int(seed) + 809,
        )
    elif strategy in {"ridge_screening", "ridge_screening_moment", "ridge"}:
        calib = calibrate_grrhs_beta_ridge_screening_moment(
            X,
            y,
            groups,
            alpha_kappa=float(alpha_kappa),
            null_quantile=float(screening_null_quantile),
            n_permutations=int(screening_permutations),
            ridge_scale=str(ridge_screening_scale),
            min_beta_kappa=min_beta_kappa,
            max_beta_kappa=max_beta_kappa,
            seed=int(seed) + 809,
        )
    elif strategy in {"validation", "validation_opt", "val"}:
        calib = calibrate_grrhs_beta_validation_opt(
            X,
            y,
            groups,
            task=str(task),
            seed=int(seed) + 811,
            p0=int(p0),
            sampler=sampler,
            grrhs_kwargs=kwargs,
            alpha_kappa=float(alpha_kappa),
            validation_fraction=float(validation_fraction),
            log_beta_bounds=(float(log_beta_min), float(log_beta_max)),
            n_initial_points=int(n_initial_points),
            n_refine_points=int(n_refine_points),
            calibration_warmup=calibration_warmup,
            calibration_draws=calibration_draws,
            min_beta_kappa=min_beta_kappa,
            max_beta_kappa=max_beta_kappa,
        )
    else:
        raise ValueError(f"Unknown GR-RHS adaptive beta strategy: {adaptive_strategy}")

    final = fit_gr_rhs(
        X,
        y,
        groups,
        task=str(task),
        seed=int(seed) + 919,
        p0=int(p0),
        sampler=sampler,
        method_name=str(method_name),
        alpha_kappa=float(calib.alpha_kappa),
        beta_kappa=float(calib.beta_kappa),
        **kwargs,
    )
    diag = dict(final.diagnostics or {})
    diag["grrhs_adaptive_beta"] = {
        "strategy": str(calib.strategy),
        "alpha_kappa": float(calib.alpha_kappa),
        "beta_kappa": float(calib.beta_kappa),
        "details": calib.details,
        "total_adaptive_wrapper_seconds": float(time.perf_counter() - total_t0),
    }
    final.diagnostics = diag
    final.method = str(method_name)
    return final
