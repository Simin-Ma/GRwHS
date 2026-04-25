from __future__ import annotations

import json
from typing import Any

import numpy as np

from simulation_project.src.experiments.evaluation import _kappa_group_means, _kappa_group_prob_gt
from simulation_project.src.experiments.group_aware_v2_common import summarize_method_row
from simulation_project.src.experiments.runtime import _is_bayesian_method
from simulation_project.src.utils import FitResult

from .schemas import MechanismDataset, active_group_mask
from .utils import mechanism_method_family, mechanism_method_label


def _safe_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _masked_mean(values: list[float], mask: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    keep = np.asarray(mask, dtype=bool)
    if arr.size == 0 or keep.size != arr.size:
        return float("nan")
    sub = arr[np.isfinite(arr) & keep]
    if sub.size == 0:
        return float("nan")
    return float(np.mean(sub))


def evaluate_method_result(
    *,
    method_name: str,
    result: FitResult,
    dataset: MechanismDataset,
) -> dict[str, Any]:
    active_mask = active_group_mask(dataset.beta, dataset.groups)
    row = summarize_method_row(
        result=result,
        method=str(method_name),
        beta_true=dataset.beta,
        groups=dataset.groups,
        X_train=dataset.X_train,
        y_train=dataset.y_train,
        X_test=dataset.X_test,
        y_test=dataset.y_test,
        group_has_signal=active_mask.tolist(),
    )
    kappa_means = _kappa_group_means(result, len(dataset.groups))
    kappa_prob_gt = _kappa_group_prob_gt(result, len(dataset.groups), threshold=0.5)
    decoy_group = int(dataset.metadata.get("decoy_group", -1))
    row.update(
        {
            "method_label": mechanism_method_label(method_name),
            "method_family": mechanism_method_family(method_name),
            "method_type": "bayesian"
            if _is_bayesian_method(mechanism_method_family(method_name))
            else "classical",
            "kappa_group_mean_json": _safe_json(kappa_means),
            "kappa_group_prob_gt_0_5_json": _safe_json(kappa_prob_gt),
            "kappa_signal_prob_gt_0_5": _masked_mean(kappa_prob_gt, active_mask),
            "kappa_null_prob_gt_0_5": _masked_mean(kappa_prob_gt, ~active_mask),
            "kappa_decoy_mean": (
                float(kappa_means[decoy_group])
                if decoy_group >= 0 and decoy_group < len(kappa_means)
                else float("nan")
            ),
            "kappa_decoy_prob_gt_0_5": (
                float(kappa_prob_gt[decoy_group])
                if decoy_group >= 0 and decoy_group < len(kappa_prob_gt)
                else float("nan")
            ),
        }
    )
    return row


def build_group_kappa_rows(
    *,
    method_name: str,
    result: FitResult,
    dataset: MechanismDataset,
    base_fields: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    active_mask = active_group_mask(dataset.beta, dataset.groups)
    kappa_means = _kappa_group_means(result, len(dataset.groups))
    kappa_prob_gt = _kappa_group_prob_gt(result, len(dataset.groups), threshold=0.5)
    decoy_group = int(dataset.metadata.get("decoy_group", -1))
    beta_arr = np.asarray(dataset.beta, dtype=float)
    for group_id, group in enumerate(dataset.groups):
        idx = np.asarray(group, dtype=int)
        is_active = bool(active_mask[group_id]) if group_id < active_mask.size else False
        is_decoy = int(group_id) == int(decoy_group) and not is_active
        group_role = "active" if is_active else ("decoy_null" if is_decoy else "other_null")
        rows.append(
            {
                **base_fields,
                "method": str(method_name),
                "method_label": mechanism_method_label(method_name),
                "group_id": int(group_id),
                "group_size": int(idx.size),
                "is_active_group": bool(is_active),
                "is_decoy_group": bool(is_decoy),
                "group_role": group_role,
                "true_group_nonzero": int(np.count_nonzero(beta_arr[idx])),
                "true_group_l2_norm": float(np.linalg.norm(beta_arr[idx])),
                "kappa_group_mean": (
                    float(kappa_means[group_id]) if group_id < len(kappa_means) else float("nan")
                ),
                "kappa_group_prob_gt_0_5": (
                    float(kappa_prob_gt[group_id]) if group_id < len(kappa_prob_gt) else float("nan")
                ),
                "status": str(result.status),
                "converged": bool(result.converged),
            }
        )
    return rows
