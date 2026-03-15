import numpy as np

import grrhs.experiments.runner as runner
from data.preprocess import StandardizationConfig
from data.splits import OuterFold
from grrhs.experiments.runner import _aggregate_metrics, _perform_inner_cv, _run_fold_nested, _set_nested_config_value


def test_nested_search_key_assignment():
    cfg = {"tau": {"mode": "calibrated", "p0": {"value": 10}}}
    _set_nested_config_value(cfg, "tau.p0.value", 18)
    _set_nested_config_value(cfg, "tau.target", "coefficients")

    assert cfg["tau"]["p0"]["value"] == 18
    assert cfg["tau"]["target"] == "coefficients"


def test_inner_cv_supports_bayesian_optimization_for_regression():
    base_config = {
        "task": "regression",
        "seed": 7,
        "model": {
            "name": "ridge",
            "fit_intercept": False,
            "search": {
                "strategy": "bayes",
                "budget": 4,
                "init_points": 2,
                "random_candidates": 64,
                "seed": 7,
                "space": {
                    "alpha": {
                        "mode": "logspace",
                        "low": 1e-4,
                        "high": 10.0,
                    }
                },
            },
        },
        "splits": {
            "inner": {
                "n_splits": 3,
                "shuffle": True,
                "seed": 11,
            }
        },
    }

    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 0.0],
            [4.0, -1.0],
            [5.0, -2.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0.2, 1.1, 1.8, 2.9, 4.2, 5.1], dtype=np.float32)
    groups = [[0], [1]]
    std_cfg = StandardizationConfig(X="unit_variance", y_center=True)

    params, history = _perform_inner_cv(
        base_config,
        X,
        y,
        groups,
        task="regression",
        std_cfg=std_cfg,
    )

    assert "alpha" in params
    assert isinstance(params["alpha"], float)
    assert history is not None
    assert len(history) == 4
    assert {entry["stage"] for entry in history}.issubset({"init", "bayes"})


def test_run_fold_retries_until_convergence(monkeypatch, tmp_path):
    class DummyModel:
        def __init__(self):
            self.coef_mean_ = np.zeros(2, dtype=np.float32)
            self.coef_samples_ = np.zeros((8, 2), dtype=np.float32)

        def fit(self, X, y, groups=None):
            return self

        def predict(self, X):
            return np.zeros(X.shape[0], dtype=np.float32)

    attempts = {"count": 0}

    def fake_instantiate_model(config, groups, p):
        attempts["count"] += 1
        return DummyModel()

    summaries = [
        {
            "beta": {
                "rhat_max": 1.20,
                "ess_min": 20.0,
                "mcse_over_sd_max": 0.22,
                "diagnostic_valid": True,
            }
        },
        {
            "beta": {
                "rhat_max": 1.01,
                "ess_min": 120.0,
                "mcse_over_sd_max": 0.09,
                "diagnostic_valid": True,
            }
        },
    ]

    monkeypatch.setattr(runner, "_instantiate_model", fake_instantiate_model)
    monkeypatch.setattr(runner, "_perform_inner_cv", lambda *args, **kwargs: ({}, None))
    monkeypatch.setattr(runner, "evaluate_model_metrics", lambda **kwargs: {"RMSE": 1.0})
    monkeypatch.setattr(runner, "summarize_convergence", lambda arrays: summaries.pop(0))

    base_config = {
        "task": "regression",
        "model": {"name": "grrhs_gibbs", "c": 1.0, "eta": 0.5, "tau0": 0.1, "iters": 20},
        "inference": {"gibbs": {"burn_in": 10, "thin": 1, "seed": 0}},
        "experiments": {
            "save_posterior": True,
            "metrics": {"regression": ["RMSE"]},
            "convergence": {
                "enabled": True,
                "max_rhat": 1.05,
                "min_ess_by_block": {"beta": 100},
                "max_mcse_over_sd": 0.10,
                "max_retries": 1,
                "retry_scale": 2.0,
            },
            "bayesian_fairness": {"disable_budget_retry": False},
        },
        "splits": {"inner": {"n_splits": 2, "shuffle": True, "seed": 0}},
    }
    dataset = {
        "X": np.array([[0.0, 1.0], [1.0, 0.0], [2.0, -1.0], [3.0, -2.0]], dtype=np.float32),
        "y": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        "groups": [[0], [1]],
    }
    fold = OuterFold(
        repeat=1,
        fold=1,
        train=np.array([0, 1, 2], dtype=int),
        test=np.array([3], dtype=int),
        hash="retry-fold",
    )
    std_cfg = StandardizationConfig(X="unit_variance", y_center=True)

    result = _run_fold_nested(
        base_config,
        dataset,
        fold,
        fold_dir=tmp_path / "fold",
        task="regression",
        std_cfg=std_cfg,
    )

    assert result["status"] == "OK"
    assert attempts["count"] == 2
    assert len(result["convergence_attempts"]) == 2
    assert result["convergence_attempts"][0]["converged"] is False
    assert result["convergence_attempts"][1]["converged"] is True


def test_inner_cv_is_disabled_for_bayesian_models():
    base_config = {
        "task": "regression",
        "model": {
            "name": "grrhs_gibbs",
            "search": {"strategy": "grid", "space": {"tau0": [0.1, 0.2]}},
        },
        "experiments": {
            "bayesian_fairness": {
                "enabled": True,
                "disable_inner_cv": True,
            }
        },
        "splits": {
            "inner": {
                "n_splits": 3,
                "shuffle": True,
                "seed": 11,
            }
        },
    }

    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 0.0],
            [4.0, -1.0],
            [5.0, -2.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0.2, 1.1, 1.8, 2.9, 4.2, 5.1], dtype=np.float32)
    groups = [[0], [1]]
    std_cfg = StandardizationConfig(X="unit_variance", y_center=True)

    params, history = _perform_inner_cv(
        base_config,
        X,
        y,
        groups,
        task="regression",
        std_cfg=std_cfg,
    )

    assert params == {}
    assert history == [{"reason": "disabled_for_bayesian_fairness"}]


def test_run_fold_rejects_invalid_diagnostics_when_required(monkeypatch, tmp_path):
    class DummyModel:
        def __init__(self):
            self.coef_mean_ = np.zeros(2, dtype=np.float32)
            self.coef_samples_ = np.zeros((8, 2), dtype=np.float32)

        def fit(self, X, y, groups=None):
            return self

        def predict(self, X):
            return np.zeros(X.shape[0], dtype=np.float32)

    monkeypatch.setattr(runner, "_instantiate_model", lambda *args, **kwargs: DummyModel())
    monkeypatch.setattr(runner, "_perform_inner_cv", lambda *args, **kwargs: ({}, None))
    monkeypatch.setattr(runner, "evaluate_model_metrics", lambda **kwargs: {"RMSE": 1.0})
    monkeypatch.setattr(
        runner,
        "summarize_convergence",
        lambda arrays: {
            "beta": {
                "rhat_max": 1.01,
                "ess_min": 120.0,
                "diagnostic_valid": False,
            }
        },
    )

    base_config = {
        "task": "regression",
        "model": {"name": "grrhs_gibbs", "c": 1.0, "eta": 0.5, "tau0": 0.1, "iters": 20},
        "inference": {"gibbs": {"burn_in": 10, "thin": 1, "seed": 0}},
        "experiments": {
            "save_posterior": True,
            "metrics": {"regression": ["RMSE"]},
            "convergence": {
                "enabled": True,
                "max_rhat": 1.05,
                "max_retries": 0,
                "require_valid_diagnostics": True,
            },
        },
        "splits": {"inner": {"n_splits": 2, "shuffle": True, "seed": 0}},
    }
    dataset = {
        "X": np.array([[0.0, 1.0], [1.0, 0.0], [2.0, -1.0], [3.0, -2.0]], dtype=np.float32),
        "y": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        "groups": [[0], [1]],
    }
    fold = OuterFold(
        repeat=1,
        fold=1,
        train=np.array([0, 1, 2], dtype=int),
        test=np.array([3], dtype=int),
        hash="invalid-diag-fold",
    )
    std_cfg = StandardizationConfig(X="unit_variance", y_center=True)

    result = _run_fold_nested(
        base_config,
        dataset,
        fold,
        fold_dir=tmp_path / "fold",
        task="regression",
        std_cfg=std_cfg,
    )

    assert result["status"] == "INVALID_CONVERGENCE"
    assert len(result["convergence_attempts"]) == 1
    assert result["convergence_attempts"][0]["converged"] is False
    assert "beta.diagnostic_valid=false" in result["convergence_attempts"][0]["failures"]


def test_aggregate_metrics_skips_invalid_convergence():
    mean_metrics, summary = _aggregate_metrics(
        [
            {"status": "OK", "metrics": {"RMSE": 1.0}},
            {"status": "INVALID_CONVERGENCE", "metrics": {"RMSE": 9.0}},
        ]
    )

    assert mean_metrics["RMSE"] == 1.0
    assert summary["RMSE"]["count"] == 1.0


def test_run_fold_marks_invalid_when_posterior_validation_fails(monkeypatch, tmp_path):
    class DummyModel:
        def __init__(self):
            self.coef_mean_ = np.zeros(2, dtype=np.float32)
            self.coef_samples_ = np.zeros((12, 2), dtype=np.float32)

        def fit(self, X, y, groups=None):
            return self

        def predict(self, X):
            return np.zeros(X.shape[0], dtype=np.float32)

    monkeypatch.setattr(runner, "_instantiate_model", lambda *args, **kwargs: DummyModel())
    monkeypatch.setattr(runner, "_perform_inner_cv", lambda *args, **kwargs: ({}, None))
    monkeypatch.setattr(runner, "evaluate_model_metrics", lambda **kwargs: {"RMSE": 1.0})
    monkeypatch.setattr(
        runner,
        "summarize_convergence",
        lambda arrays: {
            "beta": {
                "rhat_max": 1.0,
                "ess_min": 500.0,
                "mcse_over_sd_max": 0.02,
                "diagnostic_valid": True,
            }
        },
    )
    monkeypatch.setattr(
        runner,
        "_run_posterior_validation",
        lambda **kwargs: {
            "enabled": True,
            "status": "fail",
            "failures": ["ppc.p_mean=0.0000 outside [0.0250,0.9750]"],
            "ppc": {"status": "fail", "reasons": ["p_mean_tail"]},
        },
    )

    base_config = {
        "task": "regression",
        "model": {"name": "grrhs_gibbs", "c": 1.0, "eta": 0.5, "tau0": 0.1, "iters": 20},
        "inference": {"gibbs": {"burn_in": 10, "thin": 1, "seed": 0}},
        "experiments": {
            "save_posterior": True,
            "metrics": {"regression": ["RMSE"]},
            "convergence": {
                "enabled": True,
                "max_rhat": 1.01,
                "min_ess_by_block": {"beta": 100},
                "max_mcse_over_sd": 0.10,
                "max_retries": 0,
                "require_valid_diagnostics": True,
            },
            "posterior_validation": {"enabled": True},
        },
        "splits": {"inner": {"n_splits": 2, "shuffle": True, "seed": 0}},
    }
    dataset = {
        "X": np.array([[0.0, 1.0], [1.0, 0.0], [2.0, -1.0], [3.0, -2.0]], dtype=np.float32),
        "y": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        "groups": [[0], [1]],
        "beta": np.array([1.0, 0.0], dtype=np.float32),
    }
    fold = OuterFold(
        repeat=1,
        fold=1,
        train=np.array([0, 1, 2], dtype=int),
        test=np.array([3], dtype=int),
        hash="invalid-posterior-validation-fold",
    )
    std_cfg = StandardizationConfig(X="unit_variance", y_center=True)

    result = _run_fold_nested(
        base_config,
        dataset,
        fold,
        fold_dir=tmp_path / "fold",
        task="regression",
        std_cfg=std_cfg,
    )

    assert result["status"] == "INVALID_POSTERIOR_VALIDATION"
    assert "posterior_validation" in result
    assert result["posterior_validation"]["status"] == "fail"
