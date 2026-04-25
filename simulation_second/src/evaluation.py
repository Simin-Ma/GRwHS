from __future__ import annotations

import json
from typing import Any

import numpy as np

from simulation_project.src.experiments.evaluation import (
    _bridge_ratio_diagnostics,
    _evaluate_row as legacy_evaluate_row,
    _kappa_group_means,
    _kappa_group_prob_gt,
)
from simulation_project.src.experiments.runtime import _attempts_used, _is_bayesian_method
from simulation_project.src.utils import FitResult, method_display_name

from .schemas import GroupedRegressionDataset


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


def evaluate_method_result(
    result: FitResult,
    dataset: GroupedRegressionDataset,
) -> dict[str, Any]:
    metrics = legacy_evaluate_row(
        result,
        dataset.beta,
        X_train=dataset.X_train,
        y_train=dataset.y_train,
        X_test=dataset.X_test,
        y_test=dataset.y_test,
    )
    active_mask = _active_group_mask(dataset)
    kappa_means = _kappa_group_means(result, len(dataset.groups))
    kappa_prob_gt = _kappa_group_prob_gt(result, len(dataset.groups), threshold=0.5)
    bridge = _bridge_ratio_diagnostics(
        result,
        groups=dataset.groups,
        X=dataset.X_train,
        y=dataset.y_train,
        signal_group_mask=active_mask.tolist(),
    )
    return {
        **metrics,
        **bridge,
        "method_label": method_display_name(result.method),
        "method_type": "bayesian" if _is_bayesian_method(result.method) else "classical",
        "fit_attempts": int(_attempts_used(result)),
        "kappa_group_mean_json": _safe_json(kappa_means),
        "kappa_group_prob_gt_0_5_json": _safe_json(kappa_prob_gt),
        "kappa_signal_mean": _masked_mean(kappa_means, active_mask),
        "kappa_null_mean": _masked_mean(kappa_means, ~active_mask),
        "kappa_signal_prob_gt_0_5": _masked_mean(kappa_prob_gt, active_mask),
        "kappa_null_prob_gt_0_5": _masked_mean(kappa_prob_gt, ~active_mask),
    }

