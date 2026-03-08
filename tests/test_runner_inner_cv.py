import numpy as np

from data.preprocess import StandardizationConfig
from data.splits import OuterFold
from grrhs.experiments.runner import _perform_inner_cv, _run_fold_nested, _set_nested_config_value


def test_inner_cv_skips_single_class_folds():
    base_config = {
        "task": "classification",
        "model": {
            "name": "logistic_regression",
            "search": {"C": [1.0]},
        },
        "splits": {
            "inner": {
                "n_splits": 3,
                "shuffle": True,
                "seed": 0,
            }
        },
    }

    X = np.array(
        [
            [0.0, 0.0],
            [1.0, -1.0],
            [2.0, -2.0],
            [3.0, -3.0],
            [4.0, -4.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    groups = [[0], [1]]
    std_cfg = StandardizationConfig(X="unit_variance", y_center=False)

    params, history = _perform_inner_cv(
        base_config,
        X,
        y,
        groups,
        task="classification",
        std_cfg=std_cfg,
    )

    assert params == {"C": 1.0}
    assert history is not None
    assert history[0].get("skipped_folds") == 1
    assert np.isfinite(history[0]["score"])


def test_run_fold_handles_single_class_outer_train(tmp_path):
    base_config = {
        "task": "classification",
        "model": {"name": "logistic_regression"},
        "splits": {"inner": {"n_splits": 2, "shuffle": True, "seed": 0}},
    }
    dataset = {
        "X": np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
            ],
            dtype=np.float32,
        ),
        "y": np.zeros(5, dtype=np.float32),
        "groups": [[0], [1]],
    }
    fold = OuterFold(
        repeat=1,
        fold=1,
        train=np.array([0, 1, 2], dtype=int),
        test=np.array([3, 4], dtype=int),
        hash="degenerate-fold",
    )
    std_cfg = StandardizationConfig(X="unit_variance", y_center=False)

    result = _run_fold_nested(
        base_config,
        dataset,
        fold,
        fold_dir=tmp_path / "fold",
        task="classification",
        std_cfg=std_cfg,
    )

    assert result["status"] == "DEGENERATE_LABELS"
    assert result.get("degenerate_label") == 0.0
    assert result["metrics"]["ClassAccuracy"] is not None


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
