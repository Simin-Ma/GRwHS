from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grrhs.cli.run_sweep import _build_comparison_rows
from grrhs.metrics.evaluation import evaluate_model_metrics


class _PosteriorDummyModel:
    def __init__(self, coef_samples: np.ndarray, sigma_samples: np.ndarray) -> None:
        self.coef_samples_ = coef_samples
        self.coef_ = np.mean(coef_samples, axis=0)
        self.coef_mean_ = self.coef_
        self.sigma_samples_ = sigma_samples
        self.sigma_mean_ = float(np.mean(sigma_samples))
        self.intercept_ = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_


def test_synthetic_recovery_metrics_are_reported():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(24, 4))
    X_test = rng.normal(size=(12, 4))
    beta_true = np.array([1.2, 0.0, -0.8, 0.0])
    y_train = X_train @ beta_true + rng.normal(scale=0.15, size=24)
    y_test = X_test @ beta_true + rng.normal(scale=0.15, size=12)

    coef_samples = np.stack([beta_true + rng.normal(scale=0.08, size=4) for _ in range(80)], axis=0)
    sigma_samples = np.full(80, 0.15, dtype=float)
    model = _PosteriorDummyModel(coef_samples, sigma_samples)

    metrics = evaluate_model_metrics(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        beta_truth=beta_true,
        group_index=np.array([0, 0, 1, 1]),
        coverage_level=0.9,
        task="regression",
    )

    assert metrics["BetaRMSE"] is not None
    assert metrics["BetaPearson"] is not None
    assert metrics["GroupNormRMSE"] is not None
    assert metrics["BetaCoverage90"] is not None
    assert metrics["ActiveBetaIntervalWidth90"] is not None
    assert metrics["PredictiveCoverage90"] is not None


def test_sweep_comparison_prefers_configured_metric_order():
    summary = [
        {
            "name": "var_a",
            "model": "ridge",
            "status": "OK",
            "run_dir": "run_a",
            "comparison_metrics": ["RMSE", "BetaRMSE", "AUC-PR"],
            "metrics": {"RMSE": 0.4, "AUC-PR": 0.8, "BetaRMSE": 0.2, "ExtraMetric": 1.0},
        },
        {
            "name": "var_b",
            "model": "grrhs_gibbs",
            "status": "OK",
            "run_dir": "run_b",
            "comparison_metrics": ["RMSE", "BetaRMSE", "AUC-PR"],
            "metrics": {"RMSE": 0.3, "AUC-PR": 0.7, "BetaRMSE": 0.1},
        },
    ]

    _rows, metric_keys = _build_comparison_rows(summary)
    assert metric_keys == ["RMSE", "BetaRMSE", "AUC-PR"]
