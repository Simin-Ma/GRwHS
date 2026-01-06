
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

from data.loaders import load_real_dataset
from grrhs.experiments import registry as registry_module
from grrhs.experiments.aggregator import aggregate_runs
from grrhs.experiments.runner import run_experiment
from grrhs.inference.samplers import slice_sample_1d
from grrhs.cli import make_report as make_report_cli
from grrhs.cli import run_sweep as run_sweep_cli


def _invoke_cli(main_fn, arguments):
    original = sys.argv
    sys.argv = ["prog", *arguments]
    try:
        main_fn()
    finally:
        sys.argv = original


def _tiny_nested_splits(stratify: str | bool = "auto") -> dict:
    return {
        "outer": {
            "n_splits": 2,
            "shuffle": True,
            "stratify": stratify,
            "seed": 333,
        },
        "inner": {
            "n_splits": 2,
            "shuffle": True,
            "stratify": stratify,
            "seed": 444,
        },
    }


def _make_basic_config(tmp_path: Path) -> dict:
    return {
        "seed": 123,
        "task": "regression",
        "data": {
            "type": "synthetic",
            "n": 32,
            "p": 8,
            "G": 2,
            "group_sizes": "equal",
            "signal": {
                "sparsity": 0.25,
                "strong_frac": 0.5,
                "beta_scale_strong": 1.5,
                "beta_scale_weak": 0.3,
                "sign_mix": "random",
            },
            "noise_sigma": 0.5,
            "val_ratio": 0.1,
            "test_ratio": 0.2,
        },
        "standardization": {"X": "unit_variance", "y_center": True},
        "splits": _tiny_nested_splits(),
        "model": {"name": "ridge", "alpha": 0.5, "fit_intercept": False},
        "experiments": {
            "metrics": {
                "regression": ["RMSE", "PredictiveLogLikelihood"],
                "classification": ["ClassAccuracy", "ClassLogLoss"],
            },
            "coverage_level": 0.9,
            "classification_threshold": 0.5,
        },
    }


def test_group_weight_mode_size():
    cfg = {
        "model": {
            "name": "group_lasso",
            "group_weight_mode": "size",
        },
        "data": {
            "groups": [[0, 1], [2, 3, 4]],
        },
    }
    model = registry_module._build_group_lasso(cfg)
    assert np.allclose(model._group_weights, np.array([2.0, 3.0]))


def test_ridge_reports_pseudo_mlpd(tmp_path):
    config = _make_basic_config(tmp_path)
    config["name"] = "ridge_pseudo"
    config["model"] = {
        "name": "ridge",
        "alpha": 1.0,
        "fit_intercept": False,
    }
    run_dir = tmp_path / "ridge_pseudo_run"
    run_experiment(config, run_dir)
    fold_metrics_path = run_dir / "repeat_001" / "fold_01" / "metrics.json"
    metrics_payload = json.loads(fold_metrics_path.read_text(encoding="utf-8"))
    assert metrics_payload["MLPD"] is not None
    assert metrics_payload.get("MLPD_source") == "gaussian_residual_proxy"


def test_load_real_dataset_and_runner(tmp_path):
    n, p = 12, 4
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, p)).astype(np.float32)
    beta = np.array([1.0, -1.0, 0.5, 0.0], dtype=np.float32)
    y = (X @ beta).astype(np.float32)

    path_X = tmp_path / "X.npy"
    path_y = tmp_path / "y.npy"
    np.save(path_X, X)
    np.save(path_y, y)

    feature_names_path = tmp_path / "features.txt"
    feature_names_path.write_text("f0\nf1\nf2\nf3\n", encoding="utf-8")
    group_map_path = tmp_path / "groups.json"
    group_map_path.write_text(json.dumps({"f0": 0, "f1": 0, "f2": 1, "f3": 1}), encoding="utf-8")

    loader_cfg = {
        "path_X": str(path_X),
        "path_y": str(path_y),
        "path_feature_names": str(feature_names_path),
        "path_group_map": str(group_map_path),
    }

    dataset = load_real_dataset(loader_cfg, base_dir=tmp_path)
    assert dataset.X.shape == (n, p)
    assert dataset.y is not None and dataset.y.shape[0] == n
    assert dataset.groups == [[0, 1], [2, 3]]

    config = {
        "seed": 123,
        "name": "loader_run",
        "task": "regression",
        "data": {
            "type": "loader",
            "loader": loader_cfg,
            "val_ratio": 0.1,
            "test_ratio": 0.2,
        },
        "standardization": {"X": "unit_variance", "y_center": True},
        "splits": _tiny_nested_splits(),
        "model": {"name": "ridge", "alpha": 1.0, "fit_intercept": False},
        "experiments": {
            "metrics": {"regression": ["RMSE", "PredictiveLogLikelihood"]},
            "coverage_level": 0.9,
            "classification_threshold": 0.5,
        },
    }

    out_dir = tmp_path / "run_single"
    result = run_experiment(config, out_dir)
    assert result["status"] == "OK"
    metrics_payload = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "RMSE" in metrics_payload

    run_root = tmp_path / "multi_runs"
    run_root.mkdir()
    shutil.copytree(out_dir, run_root / "run_0")
    shutil.copytree(out_dir, run_root / "run_1")

    summary = aggregate_runs(run_root)
    assert summary["total_runs"] == 2
    assert "RMSE" in summary["metrics"]
    assert summary["metrics"]["RMSE"]["count"] == 2
    assert (run_root / "aggregate_summary.json").exists()


def test_slice_sampler_perturbs_state():
    rng = np.random.default_rng(1)
    log_density = lambda x: -0.5 * (x ** 2)  # standard normal up to constant
    sample = slice_sample_1d(log_density, 0.0, rng)
    assert isinstance(sample, float)
    assert abs(sample) > 0  # with overwhelming probability leaves the mode


def test_make_report_aggregates(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    config = _make_basic_config(tmp_path)
    config["name"] = "report_a"
    run_dir_a = runs_root / "report_a"
    run_experiment(config, run_dir_a)

    config["name"] = "report_b"
    run_dir_b = runs_root / "report_b"
    run_experiment(config, run_dir_b)

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    _invoke_cli(
        make_report_cli.main,
        [
            "--run",
            str(run_dir_a),
            "--run",
            str(run_dir_b),
            "--dest",
            str(reports_dir),
        ],
    )

    summary_index = json.loads((reports_dir / "summary_index.json").read_text(encoding="utf-8"))
    assert "runs" in summary_index and len(summary_index["runs"]) == 2
    assert summary_index.get("aggregates")
    assert (reports_dir / "aggregates_summary.json").exists()


def test_run_experiment_classification(tmp_path):
    config = {
        "seed": 7,
        "task": "classification",
        "data": {
            "type": "synthetic",
            "n": 40,
            "p": 6,
            "G": 3,
            "group_sizes": "equal",
            "signal": {
                "sparsity": 0.5,
                "strong_frac": 0.5,
                "beta_scale_strong": 1.2,
                "beta_scale_weak": 0.3,
            },
            "classification": {
                "scale": 0.9,
                "bias": 0.0,
                "noise_std": 0.1,
            },
            "test_ratio": 0.25,
            "val_ratio": 0.1,
        },
        "standardization": {"X": "unit_variance", "y_center": False},
        "splits": _tiny_nested_splits(stratify=True),
        "model": {
            "name": "logistic_regression",
            "logistic": {
                "solver": "lbfgs",
                "max_iter": 300,
                "penalty": "l2",
                "C": 1.0,
            },
        },
        "experiments": {
            "metrics": {
                "classification": ["ClassAccuracy", "ClassLogLoss", "ClassF1"],
            },
            "coverage_level": 0.9,
            "classification_threshold": 0.5,
        },
    }

    out_dir = tmp_path / "classification_run"
    result = run_experiment(config, out_dir)
    assert result["status"] == "OK"
    metrics_payload = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "ClassAccuracy" in metrics_payload
    assert metrics_payload["ClassAccuracy"] is not None


def test_run_sweep_parallel(tmp_path):
    base_cfg = _make_basic_config(tmp_path)
    base_cfg_path = tmp_path / "base.yaml"
    base_cfg_path.write_text(yaml.safe_dump(base_cfg), encoding="utf-8")

    sweep_spec = {
        "name": "parallel_demo",
        "output_dir": str(tmp_path / "sweep_runs"),
        "variations": [
            {"name": "var0", "overrides": {"model": {"alpha": 0.1}}},
            {"name": "var1", "overrides": {"model": {"alpha": 0.9}}},
        ],
    }
    sweep_cfg_path = tmp_path / "sweep.yaml"
    sweep_cfg_path.write_text(yaml.safe_dump(sweep_spec), encoding="utf-8")

    outdir = tmp_path / "sweep_outputs"

    _invoke_cli(
        run_sweep_cli.main,
        [
            "--base-config",
            str(base_cfg_path),
            "--sweep-config",
            str(sweep_cfg_path),
            "--outdir",
            str(outdir),
            "--jobs",
            "2",
        ],
    )

    run_dirs = [p for p in outdir.iterdir() if p.is_dir()]
    assert len(run_dirs) >= 2
    summary_files = list(outdir.glob("sweep_summary_*.json"))
    assert summary_files
    comparison_csv = list(outdir.glob("sweep_comparison_*.csv"))
    assert comparison_csv
    comparison_json = list(outdir.glob("sweep_comparison_*.json"))
    assert comparison_json
    comparison_payload = json.loads(comparison_json[0].read_text(encoding="utf-8"))
    assert len(comparison_payload.get("rows", [])) >= 2
    assert comparison_payload.get("metric_extrema")
