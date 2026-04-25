from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .evaluation import _bridge_ratio_diagnostics, _evaluate_row, _kappa_group_means
from .runtime import _attempts_used, _result_diag_fields


def _masked_nanmean(values: np.ndarray, mask: np.ndarray) -> float:
    if values.shape != mask.shape:
        return float("nan")
    sub = np.asarray(values[mask], dtype=float)
    if sub.size == 0:
        return float("nan")
    sub = sub[np.isfinite(sub)]
    if sub.size == 0:
        return float("nan")
    return float(np.mean(sub))


def active_group_mask_from_beta(
    beta: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    threshold: float = 1e-10,
) -> np.ndarray:
    beta_arr = np.asarray(beta, dtype=float).reshape(-1)
    out = []
    thr = float(max(threshold, 0.0))
    for g in groups:
        idx = np.asarray(list(g), dtype=int)
        out.append(bool(np.any(np.abs(beta_arr[idx]) > thr)) if idx.size else False)
    return np.asarray(out, dtype=bool)


def summarize_method_row(
    *,
    result: Any,
    method: str,
    beta_true: np.ndarray,
    groups: Sequence[Sequence[int]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    group_has_signal: Sequence[bool] | None = None,
) -> dict[str, Any]:
    from .analysis.metrics import group_auroc, group_l2_error, group_l2_score

    metrics = _evaluate_row(
        result,
        beta_true,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    row: dict[str, Any] = {
        "method": str(method),
        "status": str(getattr(result, "status", "")),
        "converged": bool(getattr(result, "converged", False)),
        "fit_attempts": _attempts_used(result),
        **_result_diag_fields(result),
        **metrics,
    }

    beta_mean = getattr(result, "beta_mean", None)
    if beta_mean is not None:
        labels = np.asarray(group_has_signal, dtype=bool) if group_has_signal is not None else active_group_mask_from_beta(beta_true, groups)
        err = group_l2_error(beta_mean, beta_true, groups)
        score = group_l2_score(beta_mean, groups)
        row["group_auroc"] = group_auroc(score, labels.astype(int))
        row["null_group_mse"] = float(np.mean(err[~labels])) if np.any(~labels) else float("nan")
        row["signal_group_mse"] = float(np.mean(err[labels])) if np.any(labels) else float("nan")
    else:
        row["group_auroc"] = float("nan")
        row["null_group_mse"] = float("nan")
        row["signal_group_mse"] = float("nan")

    n_groups = int(len(groups))
    km = _kappa_group_means(result, n_groups)
    if group_has_signal is None:
        mask = active_group_mask_from_beta(beta_true, groups)
    else:
        mask = np.asarray(group_has_signal, dtype=bool).reshape(-1)
    if len(km) == n_groups:
        kms = np.asarray(km, dtype=float)
        row["kappa_null_mean"] = _masked_nanmean(kms, ~mask)
        row["kappa_signal_mean"] = _masked_nanmean(kms, mask)
        row["kappa_gap"] = float(row["kappa_signal_mean"] - row["kappa_null_mean"]) if np.isfinite(row["kappa_signal_mean"]) and np.isfinite(row["kappa_null_mean"]) else float("nan")
    else:
        row["kappa_null_mean"] = float("nan")
        row["kappa_signal_mean"] = float("nan")
        row["kappa_gap"] = float("nan")

    row.update(
        _bridge_ratio_diagnostics(
            result,
            groups=groups,
            X=X_train,
            y=y_train,
            signal_group_mask=mask,
        )
    )
    return row
