from __future__ import annotations

import json
import math
from statistics import NormalDist
from typing import Any, Sequence

import numpy as np

from simulation_project.src.experiments.evaluation import (
    _bridge_ratio_diagnostics,
    _kappa_group_means,
    _kappa_group_prob_gt,
)
from simulation_project.src.experiments.runtime import _attempts_used, _is_bayesian_method
from simulation_project.src.experiments.analysis.metrics import compute_test_lpd_ppd
from simulation_project.src.utils import FitResult, method_display_name

from .schemas import PreparedSplit


ZERO_TOL = 1e-8
REL_GROUP_SELECTION_TOL = 1e-2
REL_COEF_SELECTION_TOL = 1e-2
_STANDARD_NORMAL = NormalDist()


def _safe_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _flatten_draws(draws: np.ndarray | None) -> np.ndarray | None:
    if draws is None:
        return None
    arr = np.asarray(draws, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    return arr.reshape(-1, arr.shape[-1])


def _predict_full_from_beta(result: FitResult, split: PreparedSplit, *, on_test: bool) -> np.ndarray | None:
    if result.beta_mean is None:
        return None
    X_used = split.X_test_used if on_test else split.X_train_used
    offset = split.prediction_offset_test if on_test else split.prediction_offset_train
    if X_used is None or offset is None:
        return None
    beta = np.asarray(result.beta_mean, dtype=float).reshape(-1)
    mu_model = float(split.y_offset) + float(split.y_scale) * (np.asarray(X_used, dtype=float) @ beta)
    return np.asarray(offset, dtype=float).reshape(-1) + np.asarray(mu_model, dtype=float).reshape(-1)


def _predictive_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    resid = yt - yp
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    denom = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - np.sum(resid ** 2) / denom) if denom > 0.0 else float("nan")
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def _plugin_lpd(y_true: np.ndarray, y_pred: np.ndarray, sigma2_hat: float) -> float:
    s2 = max(float(sigma2_hat), 1e-8)
    resid = np.asarray(y_true, dtype=float).reshape(-1) - np.asarray(y_pred, dtype=float).reshape(-1)
    return float(-0.5 * math.log(2.0 * math.pi * s2) - 0.5 * float(np.mean(resid ** 2)) / s2)


def _posterior_predictive_lpd_full_scale(
    result: FitResult,
    split: PreparedSplit,
    *,
    sigma2_hat: float,
) -> float:
    if result.beta_draws is None or split.X_test_used is None or split.prediction_offset_test is None:
        return float("nan")
    draws = _flatten_draws(result.beta_draws)
    if draws is None:
        return float("nan")
    mu_model = float(split.y_offset) + float(split.y_scale) * (np.asarray(split.X_test_used, dtype=float) @ draws.T)
    mu_full = np.asarray(split.prediction_offset_test, dtype=float).reshape(-1, 1) + mu_model
    y_test = np.asarray(split.y_test, dtype=float).reshape(-1)
    s2 = max(float(sigma2_hat), 1e-8)
    loglik = -0.5 * math.log(2.0 * math.pi * s2) - 0.5 * ((y_test[:, None] - mu_full) ** 2) / s2
    m = np.max(loglik, axis=1, keepdims=True)
    lme = (m + np.log(np.mean(np.exp(loglik - m), axis=1, keepdims=True))).reshape(-1)
    return float(np.mean(lme))


def _predictive_variance_full_scale(
    result: FitResult,
    split: PreparedSplit,
    *,
    sigma2_hat: float,
    on_test: bool,
) -> np.ndarray | None:
    X_used = split.X_test_used if on_test else split.X_train_used
    offset = split.prediction_offset_test if on_test else split.prediction_offset_train
    if X_used is None or offset is None:
        return None
    base = np.full(int(np.asarray(X_used).shape[0]), max(float(sigma2_hat), 1e-8), dtype=float)
    if result.beta_draws is None:
        return base
    draws = _flatten_draws(result.beta_draws)
    if draws is None or draws.shape[0] <= 1:
        return base
    mu_model = float(split.y_offset) + float(split.y_scale) * (np.asarray(X_used, dtype=float) @ draws.T)
    mu_full = np.asarray(offset, dtype=float).reshape(-1, 1) + mu_model
    mu_var = np.var(mu_full, axis=1, ddof=1)
    return np.maximum(mu_var + max(float(sigma2_hat), 1e-8), 1e-8)


def _predictive_interval_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pred_var: np.ndarray,
    *,
    level: float,
) -> tuple[float, float]:
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    vv = np.asarray(pred_var, dtype=float).reshape(-1)
    if yt.size == 0 or yp.size != yt.size or vv.size != yt.size:
        return float("nan"), float("nan")
    if not (0.0 < float(level) < 1.0):
        return float("nan"), float("nan")
    z = float(_STANDARD_NORMAL.inv_cdf(0.5 + 0.5 * float(level)))
    sd = np.sqrt(np.maximum(vv, 1e-8))
    lo = yp - z * sd
    hi = yp + z * sd
    return float(np.mean(hi - lo)), float(np.mean((yt >= lo) & (yt <= hi)))


def _group_scores(beta: np.ndarray, groups: Sequence[Sequence[int]]) -> np.ndarray:
    beta_arr = np.asarray(beta, dtype=float).reshape(-1)
    return np.asarray(
        [float(np.linalg.norm(beta_arr[np.asarray(group, dtype=int)], ord=2)) for group in groups],
        dtype=float,
    )


def _normalized_entropy(values: np.ndarray) -> float:
    vec = np.asarray(values, dtype=float).reshape(-1)
    vec = vec[np.isfinite(vec) & (vec >= 0.0)]
    if vec.size == 0:
        return float("nan")
    total = float(np.sum(vec))
    if total <= 0.0:
        return 0.0
    probs = vec / total
    probs = probs[probs > 0.0]
    if probs.size <= 1:
        return 0.0
    ent = -float(np.sum(probs * np.log(probs)))
    return float(ent / math.log(float(len(vec))))


def _top_group_labels(group_scores: np.ndarray, group_labels: Sequence[str], *, k: int = 3) -> list[str]:
    scores = np.asarray(group_scores, dtype=float).reshape(-1)
    if scores.size == 0:
        return []
    order = np.argsort(-scores, kind="stable")
    labels = [str(group_labels[int(idx)]) for idx in order[: min(int(k), len(order))]]
    return labels


def _relative_selection_mask(values: np.ndarray, *, tol: float) -> np.ndarray:
    vec = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(vec)
    if not np.any(finite):
        return np.zeros_like(vec, dtype=bool)
    vmax = float(np.max(np.abs(vec[finite])))
    if vmax <= 0.0:
        return np.zeros_like(vec, dtype=bool)
    return finite & (np.abs(vec) >= float(tol) * vmax)


def _overall_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _diagnostic_float(result: FitResult, section: str, key: str) -> float:
    diag = result.diagnostics if isinstance(result.diagnostics, dict) else {}
    payload = diag.get(section) if isinstance(diag, dict) else None
    if not isinstance(payload, dict):
        return float("nan")
    try:
        value = float(payload.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def evaluate_method_result(
    result: FitResult,
    split: PreparedSplit,
) -> dict[str, Any]:
    nan = float("nan")
    base = {
        "method_label": method_display_name(result.method),
        "method_type": "bayesian" if _is_bayesian_method(result.method) else "classical",
        "fit_attempts": int(_attempts_used(result)),
        "predictive_rhat_max": _diagnostic_float(result, "convergence_partition", "predictive_rhat_max"),
        "predictive_ess_min": _diagnostic_float(result, "convergence_partition", "predictive_ess_min"),
        "global_scale_rhat_max": _diagnostic_float(result, "convergence_partition", "global_scale_rhat_max"),
        "global_scale_ess_min": _diagnostic_float(result, "convergence_partition", "global_scale_ess_min"),
    }
    if result.beta_mean is None:
        return {
            **base,
            "rmse_train": nan,
            "mae_train": nan,
            "r2_train": nan,
            "rmse_test": nan,
            "mae_test": nan,
            "r2_test": nan,
            "lpd_test": nan,
            "lpd_test_ppd": nan,
            "lpd_test_plugin": nan,
            "avg_pred_interval_length_90": nan,
            "pred_coverage_90": nan,
            "avg_pred_interval_length_95": nan,
            "pred_coverage_95": nan,
            "sigma2_hat_train": nan,
            "coef_l1_norm": nan,
            "coef_l2_norm": nan,
            "coef_nonzero_count": nan,
            "coef_rel_1pct_count": nan,
            "group_selected_count": nan,
            "group_selected_fraction": nan,
            "group_norm_entropy": nan,
            "top_group_label": "",
            "top_group_score": nan,
            "top_groups_json": "[]",
            "group_scores_json": "[]",
            "group_selected_json": "[]",
            "group_selected_labels_json": "[]",
            "kappa_group_mean_json": "[]",
            "kappa_group_prob_gt_0_5_json": "[]",
            "kappa_mean_overall": nan,
            "kappa_prob_gt_0_5_overall": nan,
            "bridge_ratio_mean": nan,
            "bridge_ratio_min": nan,
            "bridge_ratio_max": nan,
            "bridge_ratio_p95": nan,
            "bridge_ratio_violations": nan,
            "bridge_ratio_by_group": "",
        }

    yhat_train = _predict_full_from_beta(result, split, on_test=False)
    yhat_test = _predict_full_from_beta(result, split, on_test=True)
    if yhat_train is None or yhat_test is None:
        return {
            **base,
            "rmse_train": nan,
            "mae_train": nan,
            "r2_train": nan,
            "rmse_test": nan,
            "mae_test": nan,
            "r2_test": nan,
            "lpd_test": nan,
            "lpd_test_ppd": nan,
            "lpd_test_plugin": nan,
            "avg_pred_interval_length_90": nan,
            "pred_coverage_90": nan,
            "avg_pred_interval_length_95": nan,
            "pred_coverage_95": nan,
            "sigma2_hat_train": nan,
            "coef_l1_norm": nan,
            "coef_l2_norm": nan,
            "coef_nonzero_count": nan,
            "coef_rel_1pct_count": nan,
            "group_selected_count": nan,
            "group_selected_fraction": nan,
            "group_norm_entropy": nan,
            "top_group_label": "",
            "top_group_score": nan,
            "top_groups_json": "[]",
            "group_scores_json": "[]",
            "group_selected_json": "[]",
            "group_selected_labels_json": "[]",
            "kappa_group_mean_json": "[]",
            "kappa_group_prob_gt_0_5_json": "[]",
            "kappa_mean_overall": nan,
            "kappa_prob_gt_0_5_overall": nan,
            "bridge_ratio_mean": nan,
            "bridge_ratio_min": nan,
            "bridge_ratio_max": nan,
            "bridge_ratio_p95": nan,
            "bridge_ratio_violations": nan,
            "bridge_ratio_by_group": "",
        }

    train_metrics = _predictive_metrics(split.y_train, yhat_train)
    test_metrics = _predictive_metrics(split.y_test, yhat_test)
    sigma2_hat = float(np.mean((np.asarray(split.y_train, dtype=float).reshape(-1) - yhat_train) ** 2))
    lpd_plugin = _plugin_lpd(split.y_test, yhat_test, sigma2_hat)
    lpd_ppd = _posterior_predictive_lpd_full_scale(result, split, sigma2_hat=sigma2_hat)
    lpd = lpd_ppd if np.isfinite(lpd_ppd) else lpd_plugin
    pred_var_test = _predictive_variance_full_scale(result, split, sigma2_hat=sigma2_hat, on_test=True)
    if pred_var_test is None:
        avg_pred_interval_length_90 = float("nan")
        pred_coverage_90 = float("nan")
        avg_pred_interval_length_95 = float("nan")
        pred_coverage_95 = float("nan")
    else:
        avg_pred_interval_length_90, pred_coverage_90 = _predictive_interval_metrics(
            split.y_test,
            yhat_test,
            pred_var_test,
            level=0.90,
        )
        avg_pred_interval_length_95, pred_coverage_95 = _predictive_interval_metrics(
            split.y_test,
            yhat_test,
            pred_var_test,
            level=0.95,
        )

    beta = np.asarray(result.beta_mean, dtype=float).reshape(-1)
    coef_abs = np.abs(beta)
    group_scores = _group_scores(beta, split.groups)
    coef_rel_mask = _relative_selection_mask(coef_abs, tol=REL_COEF_SELECTION_TOL)
    group_rel_mask = _relative_selection_mask(group_scores, tol=REL_GROUP_SELECTION_TOL)
    top_idx = int(np.argmax(group_scores)) if group_scores.size else -1
    top_label = str(split.dataset.group_labels[top_idx]) if top_idx >= 0 else ""
    top_score = float(group_scores[top_idx]) if top_idx >= 0 else nan
    top_groups = _top_group_labels(group_scores, split.dataset.group_labels, k=3)

    kappa_means = _kappa_group_means(result, len(split.groups))
    kappa_prob_gt = _kappa_group_prob_gt(result, len(split.groups), threshold=0.5)
    bridge = _bridge_ratio_diagnostics(
        result,
        groups=split.groups,
        X=np.asarray(split.X_train_used, dtype=float),
        y=np.asarray(split.y_train_used, dtype=float),
        signal_group_mask=None,
    )

    return {
        **base,
        "rmse_train": float(train_metrics["rmse"]),
        "mae_train": float(train_metrics["mae"]),
        "r2_train": float(train_metrics["r2"]),
        "rmse_test": float(test_metrics["rmse"]),
        "mae_test": float(test_metrics["mae"]),
        "r2_test": float(test_metrics["r2"]),
        "lpd_test": float(lpd),
        "lpd_test_ppd": float(lpd_ppd),
        "lpd_test_plugin": float(lpd_plugin),
        "avg_pred_interval_length_90": float(avg_pred_interval_length_90),
        "pred_coverage_90": float(pred_coverage_90),
        "avg_pred_interval_length_95": float(avg_pred_interval_length_95),
        "pred_coverage_95": float(pred_coverage_95),
        "sigma2_hat_train": float(sigma2_hat),
        "coef_l1_norm": float(np.sum(coef_abs)),
        "coef_l2_norm": float(np.linalg.norm(beta, ord=2)),
        "coef_nonzero_count": int(np.sum(coef_abs > ZERO_TOL)),
        "coef_rel_1pct_count": int(np.sum(coef_rel_mask)),
        "group_selected_count": int(np.sum(group_rel_mask)),
        "group_selected_fraction": float(np.mean(group_rel_mask)) if group_rel_mask.size else nan,
        "group_norm_entropy": float(_normalized_entropy(group_scores)),
        "top_group_label": top_label,
        "top_group_score": top_score,
        "top_groups_json": _safe_json(top_groups),
        "group_scores_json": _safe_json([float(v) for v in group_scores.tolist()]),
        "group_selected_json": _safe_json([bool(v) for v in group_rel_mask.tolist()]),
        "group_selected_labels_json": _safe_json(
            [str(label) for label, keep in zip(split.dataset.group_labels, group_rel_mask.tolist()) if bool(keep)]
        ),
        "kappa_group_mean_json": _safe_json([float(v) for v in kappa_means]),
        "kappa_group_prob_gt_0_5_json": _safe_json([float(v) for v in kappa_prob_gt]),
        "kappa_mean_overall": float(_overall_mean(kappa_means)),
        "kappa_prob_gt_0_5_overall": float(_overall_mean(kappa_prob_gt)),
        **bridge,
    }
