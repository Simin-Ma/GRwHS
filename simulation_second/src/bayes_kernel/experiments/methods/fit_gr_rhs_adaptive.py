from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import digamma

from .fit_gr_rhs import fit_gr_rhs
from ...utils import FitResult, SamplerConfig


@dataclass(frozen=True)
class AdaptiveBetaCalibration:
    strategy: str
    alpha_kappa: float
    beta_kappa: Any
    details: dict[str, Any]


def _sampler_with_budget(
    sampler: SamplerConfig,
    *,
    chains: int | None = None,
    warmup: int | None,
    draws: int | None,
    adapt_delta: float | None = None,
    max_treedepth: int | None = None,
) -> SamplerConfig:
    return SamplerConfig(
        chains=int(sampler.chains if chains is None else chains),
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


def _bh_fdr_mask(p_values: np.ndarray, *, alpha: float) -> np.ndarray:
    p = np.asarray(p_values, dtype=float).reshape(-1)
    m = int(p.size)
    if m == 0:
        return np.asarray([], dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    cut = float("nan")
    for rank, val in enumerate(ranked, start=1):
        if float(val) <= float(alpha) * float(rank) / float(m):
            cut = float(val)
    if not np.isfinite(cut):
        return np.zeros(m, dtype=bool)
    return np.asarray(p <= cut, dtype=bool)


def calibrate_grrhs_beta_ridge_screening_multiplicity(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    alpha_kappa: float = 0.5,
    null_level: float = 0.05,
    n_permutations: int = 500,
    ridge_scale: str = "sqrt_np",
    correction: str = "fwer",
    min_active_groups: float = 1.0,
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
    perms = int(max(1, n_permutations))
    perm_scores = np.zeros((perms, G), dtype=float)
    rng = np.random.default_rng(int(seed))
    for b in range(perms):
        perm_scores[b, :] = _ridge_group_scores(X_arr, rng.permutation(y_arr), group_list, ridge=ridge)

    corr = str(correction).strip().lower()
    level = float(min(max(null_level, 1e-6), 0.5))
    if corr in {"fwer", "max", "maxT".lower()}:
        max_null = np.nanmax(perm_scores, axis=1)
        threshold = float(np.nanquantile(max_null, 1.0 - level))
        active_mask = np.asarray(observed > threshold, dtype=bool)
        p_values = np.asarray([(1.0 + np.sum(max_null >= obs)) / (perms + 1.0) for obs in observed], dtype=float)
        threshold_detail: Any = threshold
    elif corr in {"fdr", "bh", "benjamini-hochberg"}:
        p_values = np.asarray([(1.0 + np.sum(perm_scores[:, g] >= observed[g])) / (perms + 1.0) for g in range(G)], dtype=float)
        active_mask = _bh_fdr_mask(p_values, alpha=level)
        threshold_detail = None
    elif corr in {"pointwise", "per_group", "raw"}:
        thresholds = np.nanquantile(perm_scores, 1.0 - level, axis=0)
        active_mask = np.asarray(observed > thresholds, dtype=bool)
        p_values = np.asarray([(1.0 + np.sum(perm_scores[:, g] >= observed[g])) / (perms + 1.0) for g in range(G)], dtype=float)
        threshold_detail = thresholds.tolist()
    else:
        raise ValueError(f"Unknown multiplicity correction: {correction}")

    active_count = float(np.sum(active_mask))
    s_eff = float(min(max(active_count, float(min_active_groups)), float(G) * 0.5))
    pi_eff = float(s_eff / float(G))
    beta_raw = float(float(alpha_kappa) * (1.0 - pi_eff) / max(pi_eff, 1e-12))
    beta_hat = float(beta_raw)
    if min_beta_kappa is not None:
        beta_hat = float(max(beta_hat, float(min_beta_kappa)))
    if max_beta_kappa is not None:
        beta_hat = float(min(beta_hat, float(max_beta_kappa)))
    return AdaptiveBetaCalibration(
        strategy="ridge_screening_multiplicity",
        alpha_kappa=float(alpha_kappa),
        beta_kappa=float(beta_hat),
        details={
            "correction": corr,
            "null_level": float(level),
            "n_permutations": int(perms),
            "ridge_scale": str(ridge_scale),
            "ridge": float(ridge),
            "n_groups": int(G),
            "n_active_groups": int(active_count),
            "min_active_groups": float(min_active_groups),
            "s_eff": float(s_eff),
            "pi_eff": float(pi_eff),
            "beta_kappa_raw": float(beta_raw),
            "min_beta_kappa": None if min_beta_kappa is None else float(min_beta_kappa),
            "max_beta_kappa": None if max_beta_kappa is None else float(max_beta_kappa),
            "threshold": threshold_detail,
            "p_values": p_values.tolist(),
            "active_mask": active_mask.astype(bool).tolist(),
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


def calibrate_grrhs_beta_group_specific_multiplicity(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    alpha_kappa: float = 0.5,
    null_level: float = 0.05,
    n_permutations: int = 500,
    ridge_scale: str = "sqrt_np",
    correction: str = "fwer",
    min_beta_kappa: float = 1.0,
    max_beta_kappa: float = 16.0,
    min_mean_ceiling: float | None = None,
    max_mean_ceiling: float | None = None,
    seed: int = 1,
) -> AdaptiveBetaCalibration:
    """Group-specific EB calibration via soft mean-ceiling mapping.

    This keeps the GR-RHS ceiling hierarchy intact and only adapts the second
    Beta shape parameter across groups using multiplicity-adjusted screening
    evidence. The calibrated quantity is the prior mean ceiling level
    m_g = E(kappa_g), with beta_g recovered by
        beta_g = alpha_kappa * (1 - m_g) / m_g.
    """
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
    perms = int(max(1, n_permutations))
    perm_scores = np.zeros((perms, G), dtype=float)
    rng = np.random.default_rng(int(seed))
    for b in range(perms):
        perm_scores[b, :] = _ridge_group_scores(X_arr, rng.permutation(y_arr), group_list, ridge=ridge)

    corr = str(correction).strip().lower()
    level = float(min(max(null_level, 1e-6), 0.5))
    if corr in {"fwer", "max", "maxt"}:
        max_null = np.nanmax(perm_scores, axis=1)
        p_values = np.asarray([(1.0 + np.sum(max_null >= obs)) / (perms + 1.0) for obs in observed], dtype=float)
        threshold_detail: Any = float(np.nanquantile(max_null, 1.0 - level))
    elif corr in {"pointwise", "per_group", "raw"}:
        p_values = np.asarray([(1.0 + np.sum(perm_scores[:, g] >= observed[g])) / (perms + 1.0) for g in range(G)], dtype=float)
        threshold_detail = np.nanquantile(perm_scores, 1.0 - level, axis=0).tolist()
    else:
        raise ValueError(f"Group-specific multiplicity supports 'fwer' or 'pointwise', got: {correction}")

    evidence = np.clip(1.0 - p_values / level, 0.0, 1.0)
    if min_mean_ceiling is None:
        min_mean_ceiling = float(alpha_kappa) / (float(alpha_kappa) + float(max_beta_kappa))
    if max_mean_ceiling is None:
        max_mean_ceiling = float(alpha_kappa) / (float(alpha_kappa) + float(min_beta_kappa))
    m_min = float(min(max(float(min_mean_ceiling), 1e-6), 1.0 - 1e-6))
    m_max = float(min(max(float(max_mean_ceiling), m_min + 1e-6), 1.0 - 1e-6))
    mean_ceiling_g = m_min + (m_max - m_min) * evidence
    beta_g = float(alpha_kappa) * (1.0 - mean_ceiling_g) / np.maximum(mean_ceiling_g, 1e-12)
    beta_g = np.clip(beta_g, float(min_beta_kappa), float(max_beta_kappa))

    return AdaptiveBetaCalibration(
        strategy="group_specific_multiplicity",
        alpha_kappa=float(alpha_kappa),
        beta_kappa=np.asarray(beta_g, dtype=float),
        details={
            "correction": corr,
            "null_level": float(level),
            "n_permutations": int(perms),
            "ridge_scale": str(ridge_scale),
            "ridge": float(ridge),
            "n_groups": int(G),
            "threshold": threshold_detail,
            "p_values": p_values.tolist(),
            "evidence": evidence.tolist(),
            "mean_ceiling_min": float(m_min),
            "mean_ceiling_max": float(m_max),
            "mean_ceiling_g": np.asarray(mean_ceiling_g, dtype=float).tolist(),
            "min_beta_kappa": float(min_beta_kappa),
            "max_beta_kappa": float(max_beta_kappa),
            "beta_kappa_g": np.asarray(beta_g, dtype=float).tolist(),
        },
    )


def _flatten_kappa_draws(kappa_draws: Any) -> np.ndarray:
    arr = np.asarray(kappa_draws, dtype=float)
    if arr.size == 0:
        raise ValueError("kappa_draws is empty.")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    else:
        arr = arr.reshape(-1, arr.shape[-1])
    arr = np.nan_to_num(arr, nan=0.5, posinf=1.0 - 1e-8, neginf=1e-8)
    return np.clip(arr, 1e-8, 1.0 - 1e-8)


def _mcem_beta_update_from_kappa(
    kappa_draws: Any,
    *,
    alpha_kappa: float,
    min_beta_kappa: float | None,
    max_beta_kappa: float | None,
) -> dict[str, Any]:
    draws = _flatten_kappa_draws(kappa_draws)
    n_groups = max(int(draws.shape[1]), 1)
    elog1m_by_group = np.mean(np.log1p(-draws), axis=0)
    sum_elog1m = float(np.sum(elog1m_by_group))
    alpha = float(max(alpha_kappa, 1e-8))
    lo = float(1e-6 if min_beta_kappa is None else max(float(min_beta_kappa), 1e-6))
    hi = float(1e6 if max_beta_kappa is None else max(float(max_beta_kappa), lo * 1.0001))

    def score(beta_value: float) -> float:
        beta = float(max(beta_value, 1e-12))
        return float(n_groups * (digamma(alpha + beta) - digamma(beta)) + sum_elog1m)

    score_lo = score(lo)
    score_hi = score(hi)
    boundary = None
    if score_lo <= 0.0:
        beta_hat = lo
        boundary = "lower"
    elif score_hi >= 0.0:
        beta_hat = hi
        boundary = "upper"
    else:
        beta_hat = float(brentq(score, lo, hi, maxiter=100, xtol=1e-8, rtol=1e-8))
    return {
        "beta_kappa_root": float(beta_hat),
        "score_at_min": float(score_lo),
        "score_at_max": float(score_hi),
        "score_at_root": float(score(beta_hat)),
        "boundary": boundary,
        "n_kappa_draws": int(draws.shape[0]),
        "n_groups": int(n_groups),
        "sum_E_log1m_kappa": float(sum_elog1m),
        "mean_E_log1m_kappa": float(sum_elog1m / float(n_groups)),
        "mean_posterior_kappa": float(np.mean(draws)),
    }


def _regularized_beta_update_from_kappa(
    kappa_draws: Any,
    *,
    alpha_kappa: float,
    prior_center: float,
    prior_log_sd: float,
    min_beta_kappa: float | None,
    max_beta_kappa: float | None,
) -> dict[str, Any]:
    draws = _flatten_kappa_draws(kappa_draws)
    n_groups = max(int(draws.shape[1]), 1)
    elog1m_by_group = np.mean(np.log1p(-draws), axis=0)
    sum_elog1m = float(np.sum(elog1m_by_group))
    alpha = float(max(alpha_kappa, 1e-8))
    lo = float(1e-6 if min_beta_kappa is None else max(float(min_beta_kappa), 1e-6))
    hi = float(1e6 if max_beta_kappa is None else max(float(max_beta_kappa), lo * 1.0001))
    center = float(min(max(float(prior_center), lo), hi))
    log_center = math.log(max(center, 1e-12))
    log_sd = float(max(float(prior_log_sd), 1e-6))

    def likelihood_score(beta_value: float) -> float:
        beta = float(max(beta_value, 1e-12))
        return float(n_groups * (digamma(alpha + beta) - digamma(beta)) + sum_elog1m)

    def theta_score(theta: float) -> float:
        beta = float(math.exp(theta))
        return float(beta * likelihood_score(beta) - (float(theta) - log_center) / (log_sd * log_sd))

    def objective(theta: float) -> float:
        beta = float(math.exp(theta))
        log_beta_fn = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
        q_like = float((beta - 1.0) * sum_elog1m - n_groups * log_beta_fn)
        q_prior = -0.5 * ((float(theta) - log_center) / log_sd) ** 2
        return -float(q_like + q_prior)

    update = _mcem_beta_update_from_kappa(
        draws,
        alpha_kappa=float(alpha_kappa),
        min_beta_kappa=min_beta_kappa,
        max_beta_kappa=max_beta_kappa,
    )
    log_lo = math.log(max(lo, 1e-12))
    log_hi = math.log(max(hi, lo * 1.0001))
    score_lo = theta_score(log_lo)
    score_hi = theta_score(log_hi)
    boundary = None
    if np.isfinite(score_lo) and np.isfinite(score_hi) and score_lo * score_hi < 0.0:
        theta_hat = float(brentq(theta_score, log_lo, log_hi, maxiter=100, xtol=1e-8, rtol=1e-8))
    else:
        opt = minimize_scalar(objective, bounds=(log_lo, log_hi), method="bounded", options={"xatol": 1e-8})
        theta_hat = float(opt.x)
        if abs(theta_hat - log_lo) < 1e-5:
            boundary = "lower"
        elif abs(theta_hat - log_hi) < 1e-5:
            boundary = "upper"
    beta_hat = float(min(max(math.exp(theta_hat), lo), hi))
    return {
        **update,
        "beta_kappa_regularized_map": float(beta_hat),
        "regularized_score_at_min": float(score_lo),
        "regularized_score_at_max": float(score_hi),
        "regularized_score_at_map": float(theta_score(theta_hat)),
        "regularized_boundary": boundary,
        "prior_center": float(center),
        "prior_log_sd": float(log_sd),
        "regularized_objective_at_map": float(objective(theta_hat)),
    }


def _clipped_beta_candidate(
    value: float,
    *,
    min_beta_kappa: float | None,
    max_beta_kappa: float | None,
) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out) or out <= 0.0:
        return None
    if min_beta_kappa is not None:
        out = float(max(out, float(min_beta_kappa)))
    if max_beta_kappa is not None:
        out = float(min(out, float(max_beta_kappa)))
    return out if np.isfinite(out) and out > 0.0 else None


def _regularized_posterior_eb_fallback(
    *,
    alpha_kappa: float,
    beta_init: float,
    prior_center: float,
    damping: float,
    pilot: FitResult,
    cal_sampler: SamplerConfig,
    min_beta_kappa: float | None,
    max_beta_kappa: float | None,
    reason: str,
) -> AdaptiveBetaCalibration:
    center = _clipped_beta_candidate(
        float(prior_center),
        min_beta_kappa=min_beta_kappa,
        max_beta_kappa=max_beta_kappa,
    )
    fixed_b04 = _clipped_beta_candidate(
        4.0,
        min_beta_kappa=min_beta_kappa,
        max_beta_kappa=max_beta_kappa,
    )
    initial = _clipped_beta_candidate(
        float(beta_init),
        min_beta_kappa=min_beta_kappa,
        max_beta_kappa=max_beta_kappa,
    )
    if center is not None:
        beta_out = float(center)
        stage = "prior_center"
    elif fixed_b04 is not None:
        beta_out = float(fixed_b04)
        stage = "fixed_b04"
    else:
        beta_out = float(initial if initial is not None else 1.0)
        stage = "initial_beta"
    return AdaptiveBetaCalibration(
        strategy="regularized_posterior_eb",
        alpha_kappa=float(alpha_kappa),
        beta_kappa=float(beta_out),
        details={
            "pilot_beta_kappa": float(beta_init),
            "beta_kappa_out": float(beta_out),
            "damping": float(min(max(float(damping), 0.0), 1.0)),
            "pilot_status": str(pilot.status),
            "pilot_converged": bool(pilot.converged),
            "pilot_rhat_max": float(pilot.rhat_max) if np.isfinite(pilot.rhat_max) else float("nan"),
            "pilot_ess_min": float(pilot.bulk_ess_min) if np.isfinite(pilot.bulk_ess_min) else float("nan"),
            "calibration_chains": int(cal_sampler.chains),
            "calibration_warmup": int(cal_sampler.warmup),
            "calibration_draws": int(cal_sampler.post_warmup_draws),
            "calibration_adapt_delta": float(cal_sampler.adapt_delta),
            "calibration_max_treedepth": int(cal_sampler.max_treedepth),
            "min_beta_kappa": None if min_beta_kappa is None else float(min_beta_kappa),
            "max_beta_kappa": None if max_beta_kappa is None else float(max_beta_kappa),
            "fallback_reason": str(reason),
            "fallback_stage": str(stage),
            "fallback_prior_center": None if center is None else float(center),
            "fallback_fixed_b04": None if fixed_b04 is None else float(fixed_b04),
        },
    )


def calibrate_grrhs_beta_regularized_posterior_eb(
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
    beta_kappa: float = 4.0,
    prior_center: float = 4.0,
    prior_log_sd: float = 0.75,
    damping: float = 0.5,
    calibration_chains: int | None = None,
    calibration_warmup: int | None = None,
    calibration_draws: int | None = None,
    calibration_adapt_delta: float | None = None,
    calibration_max_treedepth: int | None = None,
    min_beta_kappa: float | None = 1.0,
    max_beta_kappa: float | None = 12.0,
) -> AdaptiveBetaCalibration:
    """Regularized one-step posterior EB calibration for common beta_kappa.

    The pilot posterior supplies continuous kappa information. A log-normal
    prior centered at a conservative fixed setting and log-scale damping keep
    small-G low-dimensional cases from chasing an unstable EB boundary.
    """
    beta_init = float(beta_kappa)
    if min_beta_kappa is not None:
        beta_init = float(max(beta_init, float(min_beta_kappa)))
    if max_beta_kappa is not None:
        beta_init = float(min(beta_init, float(max_beta_kappa)))
    cal_sampler = _sampler_with_budget(
        sampler,
        chains=calibration_chains,
        warmup=calibration_warmup,
        draws=calibration_draws,
        adapt_delta=calibration_adapt_delta,
        max_treedepth=calibration_max_treedepth,
    )
    kwargs = dict(grrhs_kwargs)
    kwargs["alpha_kappa"] = float(alpha_kappa)
    kwargs["beta_kappa"] = float(beta_init)
    kwargs.setdefault("progress_bar", False)
    pilot = fit_gr_rhs(
        X,
        y,
        groups,
        task=str(task),
        seed=int(seed) + 2609,
        p0=int(p0),
        sampler=cal_sampler,
        method_name="GR_RHS_REG_POST_EB_CAL",
        retry_resume_payload=None,
        **{
            **kwargs,
            "retry_attempt": 0,
            "collapsed_hard_min_warmup": int(cal_sampler.warmup),
            "collapsed_hard_min_draws": int(cal_sampler.post_warmup_draws),
        },
    )
    if pilot.kappa_draws is None:
        return _regularized_posterior_eb_fallback(
            alpha_kappa=float(alpha_kappa),
            beta_init=float(beta_init),
            prior_center=float(prior_center),
            damping=float(damping),
            pilot=pilot,
            cal_sampler=cal_sampler,
            min_beta_kappa=min_beta_kappa,
            max_beta_kappa=max_beta_kappa,
            reason="pilot_missing_kappa_draws",
        )
    try:
        update = _regularized_beta_update_from_kappa(
            pilot.kappa_draws,
            alpha_kappa=float(alpha_kappa),
            prior_center=float(prior_center),
            prior_log_sd=float(prior_log_sd),
            min_beta_kappa=min_beta_kappa,
            max_beta_kappa=max_beta_kappa,
        )
    except Exception as exc:
        return _regularized_posterior_eb_fallback(
            alpha_kappa=float(alpha_kappa),
            beta_init=float(beta_init),
            prior_center=float(prior_center),
            damping=float(damping),
            pilot=pilot,
            cal_sampler=cal_sampler,
            min_beta_kappa=min_beta_kappa,
            max_beta_kappa=max_beta_kappa,
            reason=f"posterior_eb_update_failed:{type(exc).__name__}",
        )
    beta_map = float(update["beta_kappa_regularized_map"])
    damp = float(min(max(float(damping), 0.0), 1.0))
    beta_out = float(math.exp((1.0 - damp) * math.log(max(beta_init, 1e-12)) + damp * math.log(max(beta_map, 1e-12))))
    if min_beta_kappa is not None:
        beta_out = float(max(beta_out, float(min_beta_kappa)))
    if max_beta_kappa is not None:
        beta_out = float(min(beta_out, float(max_beta_kappa)))
    return AdaptiveBetaCalibration(
        strategy="regularized_posterior_eb",
        alpha_kappa=float(alpha_kappa),
        beta_kappa=float(beta_out),
        details={
            "pilot_beta_kappa": float(beta_init),
            "beta_kappa_unregularized_root": float(update["beta_kappa_root"]),
            "beta_kappa_regularized_map": float(beta_map),
            "beta_kappa_out": float(beta_out),
            "damping": float(damp),
            "pilot_status": str(pilot.status),
            "pilot_converged": bool(pilot.converged),
            "pilot_rhat_max": float(pilot.rhat_max) if np.isfinite(pilot.rhat_max) else float("nan"),
            "pilot_ess_min": float(pilot.bulk_ess_min) if np.isfinite(pilot.bulk_ess_min) else float("nan"),
            "calibration_chains": int(cal_sampler.chains),
            "calibration_warmup": int(cal_sampler.warmup),
            "calibration_draws": int(cal_sampler.post_warmup_draws),
            "calibration_adapt_delta": float(cal_sampler.adapt_delta),
            "calibration_max_treedepth": int(cal_sampler.max_treedepth),
            "min_beta_kappa": None if min_beta_kappa is None else float(min_beta_kappa),
            "max_beta_kappa": None if max_beta_kappa is None else float(max_beta_kappa),
            "update": update,
        },
    )


def calibrate_grrhs_beta_mcem(
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
    beta_kappa: float = 1.0,
    init_strategy: str = "ridge_screening_moment",
    rounds: int = 1,
    calibration_chains: int | None = None,
    calibration_warmup: int | None = None,
    calibration_draws: int | None = None,
    calibration_adapt_delta: float | None = None,
    calibration_max_treedepth: int | None = None,
    step_size: float = 1.0,
    screening_null_quantile: float = 0.95,
    screening_permutations: int = 300,
    ridge_screening_scale: str = "sqrt_np",
    min_beta_kappa: float | None = 1.0,
    max_beta_kappa: float | None = 16.0,
) -> AdaptiveBetaCalibration:
    """MCEM/Type-II EB calibration for beta_kappa.

    With alpha_kappa fixed, this maximizes a Monte Carlo expected complete-data
    log prior for kappa_g under short GR-RHS calibration posteriors.
    """
    strategy0 = str(init_strategy).strip().lower()
    if strategy0 in {"ridge", "ridge_screening", "ridge_screening_moment"}:
        init = calibrate_grrhs_beta_ridge_screening_moment(
            X,
            y,
            groups,
            alpha_kappa=float(alpha_kappa),
            null_quantile=float(screening_null_quantile),
            n_permutations=int(screening_permutations),
            ridge_scale=str(ridge_screening_scale),
            min_beta_kappa=min_beta_kappa,
            max_beta_kappa=max_beta_kappa,
            seed=int(seed) + 421,
        )
        beta_current = float(init.beta_kappa)
        init_details: dict[str, Any] = {
            "init_strategy": str(init.strategy),
            "init_beta_kappa": float(init.beta_kappa),
            "init_details": init.details,
        }
    else:
        beta_current = float(beta_kappa)
        if min_beta_kappa is not None:
            beta_current = float(max(beta_current, float(min_beta_kappa)))
        if max_beta_kappa is not None:
            beta_current = float(min(beta_current, float(max_beta_kappa)))
        init_details = {"init_strategy": "fixed", "init_beta_kappa": float(beta_current)}

    cal_sampler = _sampler_with_budget(
        sampler,
        chains=calibration_chains,
        warmup=calibration_warmup,
        draws=calibration_draws,
        adapt_delta=calibration_adapt_delta,
        max_treedepth=calibration_max_treedepth,
    )
    history: list[dict[str, Any]] = []
    damping = float(min(max(step_size, 0.05), 1.0))
    for round_idx in range(max(1, int(rounds))):
        kwargs = dict(grrhs_kwargs)
        kwargs["alpha_kappa"] = float(alpha_kappa)
        kwargs["beta_kappa"] = float(beta_current)
        kwargs.setdefault("progress_bar", False)
        result = fit_gr_rhs(
            X,
            y,
            groups,
            task=str(task),
            seed=int(seed) + 2003 + 97 * int(round_idx),
            p0=int(p0),
            sampler=cal_sampler,
            method_name="GR_RHS_EB_MCEM_CAL",
            retry_resume_payload=None,
            **{
                **kwargs,
                "retry_attempt": 0,
                "collapsed_hard_min_warmup": int(cal_sampler.warmup),
                "collapsed_hard_min_draws": int(cal_sampler.post_warmup_draws),
            },
        )
        if result.kappa_draws is None:
            raise RuntimeError("GR-RHS MCEM calibration requires kappa_draws.")
        update = _mcem_beta_update_from_kappa(
            result.kappa_draws,
            alpha_kappa=float(alpha_kappa),
            min_beta_kappa=min_beta_kappa,
            max_beta_kappa=max_beta_kappa,
        )
        beta_root = float(update["beta_kappa_root"])
        beta_next = float(math.exp((1.0 - damping) * math.log(max(beta_current, 1e-12)) + damping * math.log(max(beta_root, 1e-12))))
        if min_beta_kappa is not None:
            beta_next = float(max(beta_next, float(min_beta_kappa)))
        if max_beta_kappa is not None:
            beta_next = float(min(beta_next, float(max_beta_kappa)))
        history.append({
            "round": int(round_idx + 1),
            "beta_kappa_in": float(beta_current),
            "beta_kappa_root": float(beta_root),
            "beta_kappa_out": float(beta_next),
            "converged": bool(result.converged),
            "status": str(result.status),
            "rhat_max": float(result.rhat_max) if np.isfinite(result.rhat_max) else float("nan"),
            "ess_min": float(result.bulk_ess_min) if np.isfinite(result.bulk_ess_min) else float("nan"),
            **update,
        })
        beta_current = beta_next

    return AdaptiveBetaCalibration(
        strategy="mcem_beta",
        alpha_kappa=float(alpha_kappa),
        beta_kappa=float(beta_current),
        details={
            **init_details,
            "rounds": int(max(1, int(rounds))),
            "step_size": float(damping),
            "calibration_chains": int(cal_sampler.chains),
            "calibration_warmup": int(cal_sampler.warmup),
            "calibration_draws": int(cal_sampler.post_warmup_draws),
            "calibration_adapt_delta": float(cal_sampler.adapt_delta),
            "calibration_max_treedepth": int(cal_sampler.max_treedepth),
            "min_beta_kappa": None if min_beta_kappa is None else float(min_beta_kappa),
            "max_beta_kappa": None if max_beta_kappa is None else float(max_beta_kappa),
            "history": history,
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
    multiplicity_correction: str = "fwer",
    multiplicity_level: float = 0.05,
    multiplicity_min_active_groups: float = 1.0,
    mcem_rounds: int = 1,
    mcem_step_size: float = 1.0,
    mcem_init_strategy: str = "ridge_screening_moment",
    mcem_calibration_chains: int | None = None,
    mcem_calibration_adapt_delta: float | None = None,
    mcem_calibration_max_treedepth: int | None = None,
    posterior_eb_prior_center: float = 4.0,
    posterior_eb_prior_log_sd: float = 0.75,
    posterior_eb_damping: float = 0.5,
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
    kwargs.pop("multiplicity_correction", None)
    kwargs.pop("multiplicity_level", None)
    kwargs.pop("multiplicity_min_active_groups", None)
    kwargs.pop("mcem_rounds", None)
    kwargs.pop("mcem_step_size", None)
    kwargs.pop("mcem_init_strategy", None)
    kwargs.pop("mcem_calibration_chains", None)
    kwargs.pop("mcem_calibration_adapt_delta", None)
    kwargs.pop("mcem_calibration_max_treedepth", None)
    kwargs.pop("posterior_eb_prior_center", None)
    kwargs.pop("posterior_eb_prior_log_sd", None)
    kwargs.pop("posterior_eb_damping", None)
    kwargs.pop("min_beta_kappa", None)
    kwargs.pop("max_beta_kappa", None)
    kwargs.pop("alpha_kappa", None)
    kwargs.pop("beta_kappa", None)
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
    elif strategy in {"ridge_screening_multiplicity", "multiplicity", "group_multiplicity", "fwer", "fdr"}:
        correction = str(multiplicity_correction)
        if strategy in {"fwer", "fdr"}:
            correction = str(strategy)
        calib = calibrate_grrhs_beta_ridge_screening_multiplicity(
            X,
            y,
            groups,
            alpha_kappa=float(alpha_kappa),
            null_level=float(multiplicity_level),
            n_permutations=int(screening_permutations),
            ridge_scale=str(ridge_screening_scale),
            correction=correction,
            min_active_groups=float(multiplicity_min_active_groups),
            min_beta_kappa=min_beta_kappa,
            max_beta_kappa=max_beta_kappa,
            seed=int(seed) + 809,
        )
    elif strategy in {"mcem", "mcem_beta", "type2", "type_ii", "typeii"}:
        calib = calibrate_grrhs_beta_mcem(
            X,
            y,
            groups,
            task=str(task),
            seed=int(seed) + 813,
            p0=int(p0),
            sampler=sampler,
            grrhs_kwargs=kwargs,
            alpha_kappa=float(alpha_kappa),
            beta_kappa=float(beta_kappa),
            init_strategy=str(mcem_init_strategy),
            rounds=int(mcem_rounds),
            calibration_chains=mcem_calibration_chains,
            calibration_warmup=calibration_warmup,
            calibration_draws=calibration_draws,
            calibration_adapt_delta=mcem_calibration_adapt_delta,
            calibration_max_treedepth=mcem_calibration_max_treedepth,
            step_size=float(mcem_step_size),
            screening_null_quantile=float(screening_null_quantile),
            screening_permutations=int(screening_permutations),
            ridge_screening_scale=str(ridge_screening_scale),
            min_beta_kappa=min_beta_kappa,
            max_beta_kappa=max_beta_kappa,
        )
    elif strategy in {
        "regularized_posterior_eb",
        "posterior_eb_regularized",
        "reg_posterior_eb",
        "regularized_mcem",
        "regularized_type2",
    }:
        calib = calibrate_grrhs_beta_regularized_posterior_eb(
            X,
            y,
            groups,
            task=str(task),
            seed=int(seed) + 817,
            p0=int(p0),
            sampler=sampler,
            grrhs_kwargs=kwargs,
            alpha_kappa=float(alpha_kappa),
            beta_kappa=float(beta_kappa),
            prior_center=float(posterior_eb_prior_center),
            prior_log_sd=float(posterior_eb_prior_log_sd),
            damping=float(posterior_eb_damping),
            calibration_chains=mcem_calibration_chains,
            calibration_warmup=calibration_warmup,
            calibration_draws=calibration_draws,
            calibration_adapt_delta=mcem_calibration_adapt_delta,
            calibration_max_treedepth=mcem_calibration_max_treedepth,
            min_beta_kappa=min_beta_kappa,
            max_beta_kappa=max_beta_kappa,
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
    elif strategy in {"group_specific_multiplicity", "group_specific_eb", "group_eb"}:
        calib = calibrate_grrhs_beta_group_specific_multiplicity(
            X,
            y,
            groups,
            alpha_kappa=float(alpha_kappa),
            null_level=float(multiplicity_level),
            n_permutations=int(screening_permutations),
            ridge_scale=str(ridge_screening_scale),
            correction=str(multiplicity_correction),
            min_beta_kappa=float(1.0 if min_beta_kappa is None else min_beta_kappa),
            max_beta_kappa=float(16.0 if max_beta_kappa is None else max_beta_kappa),
            min_mean_ceiling=None,
            max_mean_ceiling=None,
            seed=int(seed) + 809,
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
        beta_kappa=calib.beta_kappa,
        **kwargs,
    )
    diag = dict(final.diagnostics or {})
    diag["grrhs_adaptive_beta"] = {
        "strategy": str(calib.strategy),
        "alpha_kappa": float(calib.alpha_kappa),
        "beta_kappa": np.asarray(calib.beta_kappa, dtype=float).tolist()
        if np.ndim(np.asarray(calib.beta_kappa, dtype=float)) > 0
        else float(calib.beta_kappa),
        "details": calib.details,
        "total_adaptive_wrapper_seconds": float(time.perf_counter() - total_t0),
    }
    final.diagnostics = diag
    final.method = str(method_name)
    return final

