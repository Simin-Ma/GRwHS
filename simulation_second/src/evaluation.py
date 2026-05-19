from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np

from simulation_second.src.bayes_kernel.experiments.evaluation import (
    _bridge_ratio_diagnostics,
    _evaluate_row as real_evaluate_row,
    _kappa_group_means,
    _kappa_group_prob_gt,
)
from simulation_second.src.bayes_kernel.experiments.runtime import _attempts_used, _is_bayesian_method
from simulation_second.src.bayes_kernel.utils import FitResult

from .schemas import GroupedRegressionDataset
from .utils import method_display_name


def _safe_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _active_group_mask(dataset: GroupedRegressionDataset) -> np.ndarray:
    beta = np.asarray(dataset.beta, dtype=float).reshape(-1)
    mask = []
    for group in dataset.groups:
        idx = np.asarray(group, dtype=int)
        mask.append(bool(np.any(np.abs(beta[idx]) > 1e-12)))
    return np.asarray(mask, dtype=bool)


def _masked_mean(values: list[float], mask: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or mask.size != arr.size:
        return float("nan")
    keep = np.isfinite(arr) & np.asarray(mask, dtype=bool)
    if not np.any(keep):
        return float("nan")
    return float(np.mean(arr[keep]))


def _group_ids(p: int, groups: Sequence[Sequence[int]]) -> np.ndarray:
    group_id = np.full(int(p), -1, dtype=int)
    for gid, members in enumerate(groups):
        idx = np.asarray(members, dtype=int)
        group_id[idx] = int(gid)
    return group_id


def _beta_estimate(result: FitResult, p: int) -> np.ndarray:
    beta = np.full(int(p), np.nan, dtype=float)
    if result.beta_mean is None:
        return beta
    arr = np.asarray(result.beta_mean, dtype=float).reshape(-1)
    take = min(int(p), int(arr.shape[0]))
    if take > 0:
        beta[:take] = arr[:take]
    return beta


def _gaussian_lpd(y: np.ndarray, pred: np.ndarray) -> float:
    resid = np.asarray(y, dtype=float).reshape(-1) - np.asarray(pred, dtype=float).reshape(-1)
    sigma2 = float(np.mean(resid**2))
    sigma2 = max(sigma2, 1e-10)
    return float(np.mean(-0.5 * (np.log(2.0 * np.pi * sigma2) + resid**2 / sigma2)))


def _kappa_group_means(result: FitResult, n_groups: int) -> list[float]:
    if result.kappa_draws is None:
        return [float("nan")] * int(n_groups)
    arr = np.asarray(result.kappa_draws, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    out = []
    for gid in range(int(n_groups)):
        if gid < arr.shape[1]:
            out.append(float(np.nanmean(arr[:, gid])))
        else:
            out.append(float("nan"))
    return out


def _kappa_group_prob_gt(result: FitResult, n_groups: int, threshold: float = 0.5) -> list[float]:
    if result.kappa_draws is None:
        return [float("nan")] * int(n_groups)
    arr = np.asarray(result.kappa_draws, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    out = []
    for gid in range(int(n_groups)):
        if gid < arr.shape[1]:
            out.append(float(np.nanmean(arr[:, gid] > float(threshold))))
        else:
            out.append(float("nan"))
    return out


def _is_bayesian_method(method: str) -> bool:
    return str(method).upper() not in {"OLS", "LASSO_CV"}


def _attempts_used(result: FitResult) -> int:
    attempts = result.diagnostics.get("attempts_used") if isinstance(result.diagnostics, dict) else None
    try:
        return max(1, int(attempts))
    except (TypeError, ValueError):
        return 1


def evaluate_method_result(
    result: FitResult,
    dataset: GroupedRegressionDataset,
) -> dict[str, Any]:
    metrics = real_evaluate_row(
        result,
        dataset.beta,
        X_train=dataset.X_train,
        y_train=dataset.y_train,
        X_test=dataset.X_test,
        y_test=dataset.y_test,
    )
    kappa_means = _kappa_group_means(result, len(dataset.groups))
    kappa_prob_gt = _kappa_group_prob_gt(result, len(dataset.groups), threshold=0.5)
    active_group_mask = _active_group_mask(dataset)
    bridge = _bridge_ratio_diagnostics(
        result,
        groups=dataset.groups,
        X=dataset.X_train,
        y=dataset.y_train,
        signal_group_mask=active_group_mask.tolist(),
    )
    return {
        **metrics,
        **bridge,
        "method_label": method_display_name(result.method),
        "method_type": "bayesian" if _is_bayesian_method(result.method) else "classical",
        "fit_attempts": int(_attempts_used(result)),
        "kappa_group_mean_json": _safe_json(kappa_means),
        "kappa_group_prob_gt_0_5_json": _safe_json(kappa_prob_gt),
        "kappa_signal_mean": _masked_mean(kappa_means, active_group_mask),
        "kappa_null_mean": _masked_mean(kappa_means, ~active_group_mask),
        "kappa_signal_prob_gt_0_5": _masked_mean(kappa_prob_gt, active_group_mask),
        "kappa_null_prob_gt_0_5": _masked_mean(kappa_prob_gt, ~active_group_mask),
        "bridge_ratio_signal_mean": float("nan"),
        "bridge_ratio_null_mean": float("nan"),
        "bridge_ratio_signal_over_null": float("nan"),
    }
