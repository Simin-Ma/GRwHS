from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from simulation_project.src.utils import FitResult

from simulation_second.src.blueprint import sample_signal_blueprint
from simulation_second.src.config import load_benchmark_config
from simulation_second.src.dataset import generate_grouped_dataset
from simulation_second.src.runner import run_benchmark
from simulation_second.src.suite import build_main_suite, get_setting_by_id


def test_main_suite_has_six_settings() -> None:
    settings = build_main_suite()
    assert len(settings) == 6
    assert settings[0].family == "classical_reference"
    assert settings[-1].family == "multimode_heterogeneous"


def test_classical_reference_shares_hyperparameters_across_active_groups() -> None:
    setting = get_setting_by_id("setting_1_classical_equal_medium")
    draw = sample_signal_blueprint(setting, np.random.default_rng(123))
    support_vals = list(draw.support_fractions.values())
    alpha_vals = list(draw.concentrations.values())

    assert len(set(round(v, 12) for v in support_vals)) == 1
    assert len(set(round(v, 12) for v in alpha_vals)) == 1


def test_multimode_acceptance_rule_is_enforced() -> None:
    setting = get_setting_by_id("setting_5_multimode_equal")
    draw = sample_signal_blueprint(setting, np.random.default_rng(456))
    alpha_vals = np.asarray(list(draw.concentrations.values()), dtype=float)
    support_vals = np.asarray(list(draw.support_fractions.values()), dtype=float)

    alpha_ratio = float(np.max(alpha_vals) / np.min(alpha_vals))
    support_gap = float(np.max(support_vals) - np.min(support_vals))
    assert alpha_ratio >= 3.0 or support_gap >= 0.25


def test_dataset_generation_has_expected_shapes_and_target_r2() -> None:
    setting = get_setting_by_id("setting_3_single_mode_equal", n_test=40)
    dataset = generate_grouped_dataset(setting, replicate_id=2, master_seed=20260425)

    assert dataset.X_train.shape == (500, 50)
    assert dataset.X_test.shape == (40, 50)
    assert dataset.y_train.shape == (500,)
    assert dataset.y_test.shape == (40,)
    assert dataset.sigma2 > 0.0
    assert abs(float(dataset.metadata["implied_population_r2"]) - setting.target_r2) < 1e-10


def test_default_benchmark_yaml_loads_with_six_methods() -> None:
    config = load_benchmark_config()
    assert config.package == "simulation_second"
    assert len(config.settings) == 6
    assert config.methods.roster == ("GR_RHS", "RHS", "GHS_plus", "GIGG_MMLE", "LASSO_CV", "OLS")
    assert config.methods.grrhs_kwargs["tau_target"] == "groups"


def test_run_benchmark_pipeline_smoke(monkeypatch, tmp_path) -> None:
    config = load_benchmark_config()
    config = replace(
        config,
        settings=(config.setting_map()["setting_1_classical_equal_medium"],),
        runner=replace(
            config.runner,
            repeats=2,
            n_jobs=1,
            method_jobs=1,
            output_dir=str(tmp_path),
            build_tables=True,
        ),
    )

    offsets = {
        "GR_RHS": 0.00,
        "RHS": 0.01,
        "GHS_plus": 0.02,
        "GIGG_MMLE": 0.03,
        "LASSO_CV": 0.04,
        "OLS": 0.05,
    }

    def fake_fit_benchmark_methods(
        X,
        y,
        groups,
        *,
        task,
        seed,
        p0,
        p0_groups,
        methods,
        gate,
        grrhs_kwargs,
        gigg_config,
        method_jobs,
    ):
        out = {}
        p = int(X.shape[1])
        n_groups = int(len(groups))
        for idx, method in enumerate(methods):
            center = np.full(p, offsets[method], dtype=float)
            draws = None
            kappa_draws = None
            if method in {"GR_RHS", "RHS", "GHS_plus", "GIGG_MMLE"}:
                draws = np.vstack([center - 0.01, center, center + 0.01])
            if method == "GR_RHS":
                kappa_draws = np.full((3, n_groups), 0.6, dtype=float)
            out[method] = FitResult(
                method=method,
                status="ok",
                beta_mean=center,
                beta_draws=draws,
                kappa_draws=kappa_draws,
                group_scale_draws=None,
                runtime_seconds=float(idx + 1) / 10.0,
                rhat_max=1.0,
                bulk_ess_min=500.0,
                divergence_ratio=0.0,
                converged=True,
                diagnostics={},
            )
        return out

    monkeypatch.setattr("simulation_second.src.runner.fit_benchmark_methods", fake_fit_benchmark_methods)

    result_paths = run_benchmark(config)
    assert (tmp_path / "raw_results.csv").exists()
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "summary_paired.csv").exists()
    assert (tmp_path / "summary_paired_deltas.csv").exists()
    assert (tmp_path / "paper_tables" / "paper_table_main.md").exists()
    assert (tmp_path / "paper_tables" / "paper_table_appendix_full.md").exists()
    assert Path(result_paths["summary_paired"]).exists()

    raw = pd.read_csv(tmp_path / "raw_results.csv")
    summary_paired = pd.read_csv(tmp_path / "summary_paired.csv")
    appendix = pd.read_csv(tmp_path / "paper_tables" / "paper_table_appendix_full.csv")

    assert raw.shape[0] == 12
    assert set(raw["method"]) == set(config.methods.roster)
    assert summary_paired.shape[0] == 6
    assert appendix.shape[0] == 6
