"""Tests for convergence diagnostics utilities."""
from __future__ import annotations

import json

import numpy as np

from grwhs.diagnostics.convergence import (
    split_rhat,
    effective_sample_size,
    summarize_convergence,
)
from grwhs.experiments.runner import run_experiment


def test_split_rhat_single_chain():
    rng = np.random.default_rng(0)
    draws = rng.normal(size=(200, 3))
    rhat = split_rhat(draws)
    assert rhat.shape == (3,)
    assert np.all(rhat < 1.1)


def test_effective_sample_size_reasonable():
    rng = np.random.default_rng(1)
    draws = rng.normal(size=(200,))
    ess = effective_sample_size(draws)
    assert ess > 100


def test_summarize_convergence_keys():
    rng = np.random.default_rng(2)
    draws = rng.normal(size=(200, 2))
    summary = summarize_convergence({"beta": draws})
    assert "beta" in summary
    beta_summary = summary["beta"]
    assert "rhat_max" in beta_summary
    assert "ess_min" in beta_summary


def test_run_experiment_outputs_convergence(tmp_path):
    config = {
        "seed": 123,
        "task": "regression",
        "data": {
            "type": "synthetic",
            "n": 40,
            "p": 10,
            "G": 5,
            "group_sizes": "equal",
            "signal": {
                "sparsity": 0.3,
                "strong_frac": 0.5,
                "beta_scale_strong": 1.0,
                "beta_scale_weak": 0.2,
                "sign_mix": "random",
            },
            "noise_sigma": 0.5,
            "val_ratio": 0.1,
            "test_ratio": 0.2,
        },
        "standardization": {"X": "unit_variance", "y_center": True},
        "splits": {
            "outer": {"n_splits": 2, "shuffle": True, "stratify": "auto", "seed": 555},
            "inner": {"n_splits": 2, "shuffle": True, "stratify": "auto", "seed": 666},
        },
        "model": {"name": "grwhs_gibbs", "c": 1.5, "eta": 0.5, "tau0": 0.1},
        "inference": {
            "gibbs": {"iters": 30, "burn_in": 10, "thin": 1, "seed": 321}
        },
        "experiments": {
            "metrics": {"regression": ["RMSE"]},
            "save_posterior": True,
            "classification_threshold": 0.5,
        },
    }

    run_experiment(config, tmp_path)

    posterior_path = tmp_path / "posterior_samples.npz"
    assert posterior_path.exists()

    aggregate_convergence = tmp_path / "convergence.json"
    assert not aggregate_convergence.exists()

    fold_convergence_files = sorted(tmp_path.glob("repeat_*/fold_*/convergence.json"))
    assert fold_convergence_files, "expected per-fold convergence diagnostics"

    convergence = json.loads(fold_convergence_files[0].read_text())
    assert "beta" in convergence
    assert "rhat_max" in convergence["beta"]
