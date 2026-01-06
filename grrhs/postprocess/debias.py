from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge


@dataclass
class DebiasConfig:
    enabled: bool
    selector: str
    k_grid: Sequence[int]
    ridge_lam_grid: Sequence[float]

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DebiasConfig":
        enabled = bool(payload.get("enabled", False))
        selector = str(payload.get("selector", "absbeta")).lower()
        k_grid_raw = payload.get("k_grid", [10, 20, 30])
        lam_grid_raw = payload.get("ridge_lam_grid", [0.0, 0.01, 0.1, 1.0])
        k_grid = [int(k) for k in k_grid_raw if int(k) > 0]
        ridge_lam_grid = [float(lam) for lam in lam_grid_raw if float(lam) >= 0.0]
        if not k_grid:
            raise ValueError("Debias k_grid must contain at least one positive integer.")
        if not ridge_lam_grid:
            raise ValueError("Debias ridge_lam_grid must contain at least one non-negative value.")
        return cls(enabled=enabled, selector=selector, k_grid=k_grid, ridge_lam_grid=ridge_lam_grid)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(np.square(diff))))


def _select_indices(beta_mean: np.ndarray, selector: str, k: int) -> np.ndarray:
    if selector not in {"absbeta", "magnitude"}:
        raise ValueError(f"Unsupported selector '{selector}'. Supported: 'absbeta'.")
    scores = np.abs(beta_mean)
    order = np.argsort(-scores)
    return order[: min(k, beta_mean.size)]


def _fit_model(X: np.ndarray, y: np.ndarray, lam: float) -> Any:
    if lam <= 0.0:
        return LinearRegression(fit_intercept=False)
    return Ridge(alpha=lam, fit_intercept=False)


def apply_debias_refit(
    run_dir: Path,
    config: Dict[str, Any],
    *,
    dataset: Dict[str, np.ndarray],
    posterior_arrays: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    debias_cfg = DebiasConfig.from_dict(config)
    if not debias_cfg.enabled:
        return {}

    beta_samples = posterior_arrays.get("beta")
    if beta_samples is None or beta_samples.size == 0:
        raise ValueError("Posterior samples missing 'beta'; cannot perform debias refit.")

    beta_mean = np.asarray(beta_samples).mean(axis=0)

    X_train = dataset["X_train"]
    y_train = dataset["y_train"]
    X_val = dataset["X_val"]
    y_val = dataset["y_val"]
    X_test = dataset["X_test"]
    y_test = dataset["y_test"]

    baseline_val_rmse = _rmse(y_val, X_val @ beta_mean)
    baseline_test_rmse = _rmse(y_test, X_test @ beta_mean)

    results: List[Tuple[int, float, float]] = []
    best_pred: Tuple[int, float, float] | None = None
    best_model = None
    best_indices = None

    for k in debias_cfg.k_grid:
        idx = _select_indices(beta_mean, debias_cfg.selector, k)
        if idx.size == 0:
            continue
        X_train_subset = X_train[:, idx]
        X_val_subset = X_val[:, idx]
        X_test_subset = X_test[:, idx]
        for lam in debias_cfg.ridge_lam_grid:
            model = _fit_model(X_train_subset, y_train, lam)
            model.fit(X_train_subset, y_train)
            val_pred = model.predict(X_val_subset)
            val_rmse = _rmse(y_val, val_pred)
            results.append((k, lam, val_rmse))
            if best_pred is None or val_rmse < best_pred[2]:
                best_pred = (k, lam, val_rmse)
                best_model = model
                best_indices = idx

    if best_pred is None or best_model is None or best_indices is None:
        raise RuntimeError("Failed to compute debias refit; no valid (k, lambda) combination.")

    k_best, lam_best, val_best = best_pred
    test_rmse = _rmse(y_test, best_model.predict(X_test[:, best_indices]))

    summary = {
        "enabled": True,
        "selector": debias_cfg.selector,
        "k_grid": list(debias_cfg.k_grid),
        "ridge_lam_grid": list(debias_cfg.ridge_lam_grid),
        "k_best": int(k_best),
        "lambda_best": float(lam_best),
        "rmse_val_debiased": float(val_best),
        "rmse_test_debiased": float(test_rmse),
        "rmse_val_original": float(baseline_val_rmse),
        "rmse_test_original": float(baseline_test_rmse),
        "rmse_val_gain_pct": float((baseline_val_rmse - val_best) / baseline_val_rmse * 100.0 if baseline_val_rmse > 0 else 0.0),
        "rmse_test_gain_pct": float((baseline_test_rmse - test_rmse) / baseline_test_rmse * 100.0 if baseline_test_rmse > 0 else 0.0),
    }

    post_dir = run_dir / "postprocess"
    post_dir.mkdir(exist_ok=True)
    (post_dir / "debias_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
