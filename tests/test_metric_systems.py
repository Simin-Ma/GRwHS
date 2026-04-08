from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grrhs.cli.run_sweep import _build_comparison_rows
from grrhs.diagnostics.postprocess import compute_diagnostics_from_samples
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


class _PosteriorWrongPredictModel(_PosteriorDummyModel):
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], 99.0, dtype=float)


class _GroupedPosteriorDummyModel(_PosteriorDummyModel):
    def __init__(
        self,
        coef_samples: np.ndarray,
        sigma_samples: np.ndarray,
        tau_samples: np.ndarray,
        lambda_samples: np.ndarray,
        group_scale_samples: np.ndarray,
        *,
        attr_name: str,
    ) -> None:
        super().__init__(coef_samples, sigma_samples)
        self.tau_samples_ = tau_samples
        self.lambda_samples_ = lambda_samples
        setattr(self, attr_name, group_scale_samples)


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
        group_index=np.array([0, 0, 1, 2]),
        coverage_level=0.9,
        task="regression",
    )

    assert metrics["BetaRMSE"] is not None
    assert metrics["BetaPearson"] is not None
    assert metrics["GroupNormRMSE"] is not None
    assert metrics["GroupNormMSE"] is not None
    assert metrics["NullGroupMeanNorm"] is not None
    assert metrics["GroupTPR"] is not None
    assert metrics["GroupFPR"] is not None
    assert metrics["GroupF1"] is not None
    assert metrics["ActiveGroupSignalRMSE"] is not None
    assert metrics["ActiveGroupNoiseAbsMean"] is not None
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


def test_sweep_comparison_uses_common_valid_fold_intersection():
    summary = [
        {
            "name": "var_a",
            "model": "grrhs_gibbs",
            "status": "PARTIAL",
            "run_dir": "run_a",
            "comparison_metrics": ["RMSE"],
            "metrics": {
                "metrics": {"RMSE": 5.0},
                "valid_fold_count": 1,
                "invalid_fold_count": 1,
                "repeat_summaries": [
                    {
                        "repeat_index": 1,
                        "folds": [
                            {"hash": "fold_a", "status": "OK", "metrics": {"RMSE": 1.0}},
                            {"hash": "fold_b", "status": "INVALID_CONVERGENCE", "metrics": {"RMSE": 9.0}},
                        ],
                    }
                ],
            },
        },
        {
            "name": "var_b",
            "model": "ridge",
            "status": "OK",
            "run_dir": "run_b",
            "comparison_metrics": ["RMSE"],
            "metrics": {
                "metrics": {"RMSE": 4.0},
                "valid_fold_count": 2,
                "invalid_fold_count": 0,
                "repeat_summaries": [
                    {
                        "repeat_index": 1,
                        "folds": [
                            {"hash": "fold_a", "status": "OK", "metrics": {"RMSE": 2.0}},
                            {"hash": "fold_b", "status": "OK", "metrics": {"RMSE": 6.0}},
                        ],
                    }
                ],
            },
        },
    ]

    rows, metric_keys = _build_comparison_rows(summary)
    assert metric_keys == ["RMSE"]
    row_map = {row["variation"]: row for row in rows}
    assert row_map["var_a"]["metrics"]["RMSE"] == 1.0
    assert row_map["var_b"]["metrics"]["RMSE"] == 2.0
    assert row_map["var_a"]["comparison_basis"] == "common_valid_folds"
    assert row_map["var_b"]["comparison_fold_count"] == 1


def test_strict_predictive_density_disables_proxy_for_deterministic_models():
    rng = np.random.default_rng(1)
    X_train = rng.normal(size=(20, 3))
    X_test = rng.normal(size=(8, 3))
    beta_true = np.array([0.8, -0.3, 0.0])
    y_train = X_train @ beta_true + rng.normal(scale=0.2, size=20)
    y_test = X_test @ beta_true + rng.normal(scale=0.2, size=8)

    class _PointOnlyModel:
        def __init__(self, coef: np.ndarray) -> None:
            self.coef_ = coef
            self.intercept_ = 0.0

        def predict(self, X: np.ndarray) -> np.ndarray:
            return X @ self.coef_

    model = _PointOnlyModel(beta_true)
    metrics = evaluate_model_metrics(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        task="regression",
        predictive_density_mode="strict",
    )

    assert metrics["MLPD"] is None
    assert metrics["PredictiveLogLikelihood"] is None
    assert metrics["MLPD_source"] == "disabled"


def test_regression_predictive_density_uses_exact_gaussian_loglik_from_posterior_draws():
    X_train = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    X_test = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    y_train = np.array([1.0, 2.0], dtype=float)
    y_test = np.array([1.1, 1.9], dtype=float)
    coef_samples = np.array([[1.0, 2.0], [1.2, 1.8]], dtype=float)
    sigma_samples = np.array([0.5, 0.25], dtype=float)
    model = _PosteriorDummyModel(coef_samples, sigma_samples)

    metrics = evaluate_model_metrics(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        task="regression",
        predictive_density_mode="strict",
    )

    mean_draws = np.array(
        [
            [1.0, 2.0],
            [1.2, 1.8],
        ],
        dtype=float,
    )
    sigma = sigma_samples[:, np.newaxis]
    residual = y_test[np.newaxis, :] - mean_draws
    expected_loglik = -0.5 * ((residual**2) / (sigma**2)) - np.log(np.sqrt(2 * np.pi)) - np.log(sigma)
    expected_mlpd = float(np.mean(logsumexp(expected_loglik, axis=0) - np.log(expected_loglik.shape[0])))

    assert np.isclose(metrics["PredictiveLogLikelihood"], float(expected_loglik.mean()))
    assert np.isclose(metrics["MLPD"], expected_mlpd)
    assert metrics["MLPD_source"] == "posterior_draws"


def test_selection_metrics_use_shared_absolute_coefficient_ranking():
    rng = np.random.default_rng(11)
    X_train = rng.normal(size=(20, 4))
    X_test = rng.normal(size=(10, 4))
    beta_true = np.array([1.0, 0.0, -0.7, 0.0], dtype=float)
    y_train = X_train @ beta_true + rng.normal(scale=0.15, size=20)
    y_test = X_test @ beta_true + rng.normal(scale=0.15, size=10)

    coef_samples = np.stack([beta_true + rng.normal(scale=0.05, size=4) for _ in range(40)], axis=0)
    sigma_samples = np.full(40, 0.15, dtype=float)
    posterior_model = _PosteriorDummyModel(coef_samples, sigma_samples)

    class _PointOnlyModel:
        def __init__(self, coef: np.ndarray) -> None:
            self.coef_ = coef
            self.intercept_ = 0.0

        def predict(self, X: np.ndarray) -> np.ndarray:
            return X @ self.coef_

    point_model = _PointOnlyModel(np.mean(coef_samples, axis=0))

    posterior_metrics = evaluate_model_metrics(
        model=posterior_model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        beta_truth=beta_true,
        task="regression",
    )
    point_metrics = evaluate_model_metrics(
        model=point_model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        beta_truth=beta_true,
        task="regression",
    )

    assert np.isclose(posterior_metrics["AUC-PR"], point_metrics["AUC-PR"])
    assert np.isclose(posterior_metrics["F1"], point_metrics["F1"])


def test_bayesian_regression_predictions_use_posterior_mean_not_model_predict():
    X_train = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    X_test = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    y_train = np.array([1.0, 2.0], dtype=float)
    y_test = np.array([1.0, 2.0], dtype=float)
    coef_samples = np.array([[1.0, 2.0], [1.0, 2.0]], dtype=float)
    sigma_samples = np.array([0.1, 0.1], dtype=float)
    model = _PosteriorWrongPredictModel(coef_samples, sigma_samples)

    metrics = evaluate_model_metrics(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        task="regression",
        predictive_density_mode="strict",
    )

    assert np.isclose(metrics["RMSE"], 0.0)


def test_gigg_gamma_samples_feed_group_shrinkage_diagnostics():
    rng = np.random.default_rng(2)
    X_train = rng.normal(size=(18, 4))
    X_test = rng.normal(size=(8, 4))
    beta_true = np.array([1.1, 0.0, -0.7, 0.0])
    y_train = X_train @ beta_true + rng.normal(scale=0.2, size=18)
    y_test = X_test @ beta_true + rng.normal(scale=0.2, size=8)

    coef_samples = np.stack([beta_true + rng.normal(scale=0.05, size=4) for _ in range(32)], axis=0)
    sigma_samples = np.full(32, 0.2, dtype=float)
    tau_samples = np.linspace(0.3, 0.6, 32)
    lambda_samples = np.abs(rng.normal(loc=0.9, scale=0.1, size=(32, 4))) + 0.1
    gamma_samples = np.tile(np.array([[0.4, 1.6]], dtype=float), (32, 1))
    group_index = np.array([0, 0, 1, 1], dtype=int)
    model = _GroupedPosteriorDummyModel(
        coef_samples,
        sigma_samples,
        tau_samples,
        lambda_samples,
        gamma_samples,
        attr_name="gamma_samples_",
    )

    metrics = evaluate_model_metrics(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        beta_truth=beta_true,
        group_index=group_index,
        slab_width=2.0,
        task="regression",
    )
    diag = compute_diagnostics_from_samples(
        X=X_train,
        group_index=group_index,
        c=2.0,
        eps=1e-8,
        lambda_=lambda_samples,
        tau=tau_samples,
        phi=gamma_samples,
        sigma=sigma_samples,
    )

    assert np.isclose(metrics["MeanKappa"], float(np.mean(diag.per_coeff["kappa"])))
    assert np.isclose(metrics["EffectiveDoF"], float(np.sum(diag.per_group["edf"])))
    assert np.isclose(metrics["MeanEffectiveNonzeros"], float(diag.meta["effective_nonzeros_mean"]))
