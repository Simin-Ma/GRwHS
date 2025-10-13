"""Smoke test for synthetic experiment runner."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grwhs.experiments.runner import run_experiment


def test_run_experiment_creates_artifacts(tmp_path):
    config = {
        "seed": 123,
        "name": "smoke",
        "task": "regression",
        "data": {
            "type": "synthetic",
            "n": 64,
            "p": 16,
            "G": 4,
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
        "model": {"name": "ridge", "alpha": 0.5, "fit_intercept": False},
        "experiments": {
            "metrics": {
                "regression": ["RMSE", "PredictiveLogLikelihood"],
            },
            "coverage_level": 0.9,
            "classification_threshold": 0.5,
        },
    }

    out_dir = tmp_path / "run"
    result = run_experiment(config, out_dir)

    assert result["status"] == "OK"
    assert (out_dir / "dataset.npz").exists()
    assert (out_dir / "dataset_meta.json").exists()

    assert result.get("repeats", 1) == 1
    repeat_metrics = result.get("repeat_metrics", [])
    assert len(repeat_metrics) == 1

    metrics = result.get("metrics", {})
    assert "RMSE" in metrics and "PredictiveLogLikelihood" in metrics
    assert metrics["RMSE"] is not None

    data = np.load(out_dir / "dataset.npz")
    assert data["X_train"].shape[0] > 0
    assert data["beta_true"].shape[0] == config["data"]["p"]

    meta = json.loads((out_dir / "dataset_meta.json").read_text(encoding="utf-8"))
    assert meta["model"] == "ridge"
    assert meta["standardization"]["X"] == "unit_variance"

def test_run_experiment_honours_repeats(tmp_path):
    base_config = {
        "seed": 321,
        "task": "regression",
        "data": {
            "type": "synthetic",
            "n": 48,
            "p": 12,
            "G": 3,
            "group_sizes": "equal",
            "signal": {
                "sparsity": 0.25,
                "strong_frac": 0.6,
                "beta_scale_strong": 1.2,
                "beta_scale_weak": 0.25,
                "sign_mix": "random",
            },
            "noise_sigma": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.2,
        },
        "standardization": {"X": "unit_variance", "y_center": True},
        "model": {"name": "ridge", "alpha": 1.0, "fit_intercept": False},
        "inference": {
            "gibbs": {"seed": 500},
            "svi": {"seed": 600},
        },
        "experiments": {
            "repeats": 3,
            "metrics": {
                "regression": ["RMSE", "PredictiveLogLikelihood"],
            },
            "coverage_level": 0.9,
            "classification_threshold": 0.5,
        },
    }

    out_dir = tmp_path / "multi_repeats"
    result = run_experiment(base_config, out_dir)

    assert result.get("repeats") == 3
    repeat_metrics = result.get("repeat_metrics", [])
    assert len(repeat_metrics) == 3

    gibbs_seeds = [entry.get("seeds", {}).get("inference", {}).get("gibbs") for entry in repeat_metrics]
    assert gibbs_seeds == [500, 501, 502]
    svi_seeds = [entry.get("seeds", {}).get("inference", {}).get("svi") for entry in repeat_metrics]
    assert svi_seeds == [600, 601, 602]

    repeat_dirs = result.get("artifacts", {}).get("repeat_dirs", [])
    assert len(repeat_dirs) == 3
    for idx, repeat_path in enumerate(repeat_dirs, start=1):
        repeat_dir = Path(repeat_path)
        assert repeat_dir.exists(), repeat_dir
        assert (repeat_dir / "dataset.npz").exists()
        assert (repeat_dir / "dataset_meta.json").exists()
        if repeat_metrics:
            assert repeat_metrics[idx - 1]["repeat"] == idx

    # Ensure root directory only stores summary (no dataset files at top-level)
    assert not (out_dir / "dataset.npz").exists()
    assert not (out_dir / "dataset_meta.json").exists()

