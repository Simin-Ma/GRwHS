from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from simulation_project.src.utils import FitResult

from simulation_second.src.blueprint import sample_signal_blueprint
from simulation_second.src.config import benchmark_config_from_payload, load_benchmark_config
from simulation_second.src.dataset import generate_grouped_dataset
from simulation_second.src.plotting import build_benchmark_figures_from_results_dir
from simulation_second.src.runner import run_benchmark
from simulation_second.src.suite import build_main_suite, get_setting_by_id
from simulation_second.src.cli import run_blueprint_cli


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
    assert config.convergence_gate.enforce_bayes_convergence is True
    assert config.convergence_gate.max_convergence_retries == -1


def test_benchmark_config_forces_until_convergence_even_if_payload_disables_it() -> None:
    payload = {
        "convergence_gate": {
            "enforce_bayes_convergence": False,
            "max_convergence_retries": 0,
            "chains": 3,
        },
        "settings": [],
    }
    config = benchmark_config_from_payload(payload)
    assert config.convergence_gate.enforce_bayes_convergence is True
    assert config.convergence_gate.max_convergence_retries == -1
    assert config.convergence_gate.chains == 3


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
    run_dir = Path(result_paths["run_dir"])
    assert (run_dir / "raw_results.csv").exists()
    assert (run_dir / "summary.csv").exists()
    assert (run_dir / "summary_paired.csv").exists()
    assert (run_dir / "summary_paired_deltas.csv").exists()
    assert (run_dir / "coefficient_estimates.csv").exists()
    assert (run_dir / "coefficient_estimates_paired.csv").exists()
    assert (run_dir / "paper_tables" / "paper_table_main.md").exists()
    assert (run_dir / "paper_tables" / "paper_table_appendix_full.md").exists()
    assert (run_dir / "paper_tables" / "figure_data" / "figure1_coefficient_recovery_profile.csv").exists()
    assert (run_dir / "figures" / "figure1_coefficient_recovery_profile.png").exists()
    assert (tmp_path / "latest_run.json").exists()
    assert Path(result_paths["summary_paired"]).exists()
    assert Path(result_paths["run_dir"]).exists()
    assert Path(result_paths["run_manifest"]).exists()

    raw = pd.read_csv(run_dir / "raw_results.csv")
    summary_paired = pd.read_csv(run_dir / "summary_paired.csv")
    appendix = pd.read_csv(run_dir / "paper_tables" / "paper_table_appendix_full.csv")
    coef = pd.read_csv(run_dir / "coefficient_estimates.csv")
    fig1 = pd.read_csv(run_dir / "paper_tables" / "figure_data" / "figure1_coefficient_recovery_profile.csv")
    appendix_md = (run_dir / "paper_tables" / "paper_table_appendix_full.md").read_text(encoding="utf-8")

    assert raw.shape[0] == 12
    assert set(raw["method"]) == set(config.methods.roster)
    assert summary_paired.shape[0] == 6
    assert appendix.shape[0] == 6
    assert set(coef["method"]) == set(config.methods.roster)
    assert "true_beta" in coef.columns
    assert not fig1.empty
    assert "plot_order" in fig1.columns
    assert "rhat_max_mean" in appendix.columns
    assert "**" in appendix_md

    rebuilt = build_benchmark_figures_from_results_dir(tmp_path)
    assert Path(rebuilt["figure1_coefficient_recovery_profile"]).exists()


def test_build_figures_cli_avoids_loading_heavy_runner_modules(tmp_path) -> None:
    fig_data_dir = tmp_path / "paper_tables" / "figure_data"
    fig_data_dir.mkdir(parents=True)
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "setting_id": "setting_5_multimode_equal",
                "setting_label": "Setting 5: Main Showcase, Multi-Mode Heterogeneous Active Groups",
                "replicate_id": 3,
                "method": "GR_RHS",
                "method_label": "GR-RHS",
                "method_type": "bayesian",
                "method_order": 0,
                "coefficient_index": 0,
                "group_id": 0,
                "group_size": 10,
                "within_group_index": 0,
                "group_rank_within_plot": 0,
                "plot_order": 0,
                "group_plot_lo": 0,
                "group_plot_hi": 1,
                "group_plot_center": 0.5,
                "is_active_group": True,
                "is_active_coefficient": True,
                "true_beta": 1.0,
                "estimated_beta": 0.9,
                "error": -0.1,
                "sq_error": 0.01,
                "abs_error": 0.1,
                "method_signal_rmse": 0.1,
                "method_overall_rmse": 0.1,
                "representative_setting_id": "setting_5_multimode_equal",
                "representative_replicate_id": 3,
                "representative_selector_method": "GR_RHS",
                "representative_selector_signal_mse": 0.01,
                "representative_selector_distance_to_median": 0.0,
            },
            {
                "setting_id": "setting_5_multimode_equal",
                "setting_label": "Setting 5: Main Showcase, Multi-Mode Heterogeneous Active Groups",
                "replicate_id": 3,
                "method": "GR_RHS",
                "method_label": "GR-RHS",
                "method_type": "bayesian",
                "method_order": 0,
                "coefficient_index": 1,
                "group_id": 0,
                "group_size": 10,
                "within_group_index": 1,
                "group_rank_within_plot": 1,
                "plot_order": 1,
                "group_plot_lo": 0,
                "group_plot_hi": 1,
                "group_plot_center": 0.5,
                "is_active_group": True,
                "is_active_coefficient": False,
                "true_beta": 0.0,
                "estimated_beta": 0.05,
                "error": 0.05,
                "sq_error": 0.0025,
                "abs_error": 0.05,
                "method_signal_rmse": 0.1,
                "method_overall_rmse": 0.1,
                "representative_setting_id": "setting_5_multimode_equal",
                "representative_replicate_id": 3,
                "representative_selector_method": "GR_RHS",
                "representative_selector_signal_mse": 0.01,
                "representative_selector_distance_to_median": 0.0,
            },
            {
                "setting_id": "setting_5_multimode_equal",
                "setting_label": "Setting 5: Main Showcase, Multi-Mode Heterogeneous Active Groups",
                "replicate_id": 3,
                "method": "RHS",
                "method_label": "RHS",
                "method_type": "bayesian",
                "method_order": 1,
                "coefficient_index": 0,
                "group_id": 0,
                "group_size": 10,
                "within_group_index": 0,
                "group_rank_within_plot": 0,
                "plot_order": 0,
                "group_plot_lo": 0,
                "group_plot_hi": 1,
                "group_plot_center": 0.5,
                "is_active_group": True,
                "is_active_coefficient": True,
                "true_beta": 1.0,
                "estimated_beta": 0.8,
                "error": -0.2,
                "sq_error": 0.04,
                "abs_error": 0.2,
                "method_signal_rmse": 0.2,
                "method_overall_rmse": 0.2,
                "representative_setting_id": "setting_5_multimode_equal",
                "representative_replicate_id": 3,
                "representative_selector_method": "GR_RHS",
                "representative_selector_signal_mse": 0.01,
                "representative_selector_distance_to_median": 0.0,
            },
            {
                "setting_id": "setting_5_multimode_equal",
                "setting_label": "Setting 5: Main Showcase, Multi-Mode Heterogeneous Active Groups",
                "replicate_id": 3,
                "method": "RHS",
                "method_label": "RHS",
                "method_type": "bayesian",
                "method_order": 1,
                "coefficient_index": 1,
                "group_id": 0,
                "group_size": 10,
                "within_group_index": 1,
                "group_rank_within_plot": 1,
                "plot_order": 1,
                "group_plot_lo": 0,
                "group_plot_hi": 1,
                "group_plot_center": 0.5,
                "is_active_group": True,
                "is_active_coefficient": False,
                "true_beta": 0.0,
                "estimated_beta": 0.10,
                "error": 0.10,
                "sq_error": 0.01,
                "abs_error": 0.10,
                "method_signal_rmse": 0.2,
                "method_overall_rmse": 0.2,
                "representative_setting_id": "setting_5_multimode_equal",
                "representative_replicate_id": 3,
                "representative_selector_method": "GR_RHS",
                "representative_selector_signal_mse": 0.01,
                "representative_selector_distance_to_median": 0.0,
            },
        ]
    ).to_csv(fig_data_dir / "figure1_coefficient_recovery_profile.csv", index=False)

    Path(tmp_path / "latest_run.txt").write_text(str(tmp_path), encoding="utf-8")

    for mod in [
        "simulation_second.src.runner",
        "simulation_second.src.fitting",
        "simulation_second.src.dataset",
    ]:
        sys.modules.pop(mod, None)

    exit_code = run_blueprint_cli.main(["build-figures", "--results-dir", str(tmp_path)])
    assert exit_code == 0
    assert (figures_dir / "figure1_coefficient_recovery_profile.png").exists()
    assert "simulation_second.src.runner" not in sys.modules
    assert "simulation_second.src.fitting" not in sys.modules


def test_build_tables_recreates_coefficient_figure_data(tmp_path) -> None:
    raw = pd.DataFrame(
        [
            {
                "setting_id": "setting_5_multimode_equal",
                "setting_label": "Setting 5: Main Showcase, Multi-Mode Heterogeneous Active Groups",
                "family": "multimode_heterogeneous",
                "suite": "main",
                "role": "main showcase setting",
                "notes": "demo",
                "group_config": "G10x5",
                "group_sizes": "[10,10,10,10,10]",
                "active_groups": "[0,1,2]",
                "n_train": 500,
                "n_test": 100,
                "rho_within": 0.8,
                "rho_between": 0.2,
                "target_r2": 0.7,
                "replicate_id": 1,
                "seed": 20260425,
                "method": "GR_RHS",
                "method_label": "GR-RHS",
                "status": "ok",
                "converged": True,
                "mse_null": 0.01,
                "mse_signal": 0.02,
                "mse_overall": 0.015,
                "lpd_test": -1.0,
                "runtime_seconds": 1.0,
            },
            {
                "setting_id": "setting_5_multimode_equal",
                "setting_label": "Setting 5: Main Showcase, Multi-Mode Heterogeneous Active Groups",
                "family": "multimode_heterogeneous",
                "suite": "main",
                "role": "main showcase setting",
                "notes": "demo",
                "group_config": "G10x5",
                "group_sizes": "[10,10,10,10,10]",
                "active_groups": "[0,1,2]",
                "n_train": 500,
                "n_test": 100,
                "rho_within": 0.8,
                "rho_between": 0.2,
                "target_r2": 0.7,
                "replicate_id": 1,
                "seed": 20260425,
                "method": "RHS",
                "method_label": "RHS",
                "status": "ok",
                "converged": True,
                "mse_null": 0.02,
                "mse_signal": 0.03,
                "mse_overall": 0.025,
                "lpd_test": -1.1,
                "runtime_seconds": 2.0,
            },
        ]
    )
    raw.to_csv(tmp_path / "raw_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "setting_id": "setting_5_multimode_equal",
                "replicate_id": 1,
                "method": "GR_RHS",
                "method_label": "GR-RHS",
                "method_type": "bayesian",
                "method_order": 0,
                "setting_label": "Setting 5: Main Showcase, Multi-Mode Heterogeneous Active Groups",
                "family": "multimode_heterogeneous",
                "suite": "main",
                "role": "main showcase setting",
                "coefficient_index": 0,
                "group_id": 0,
                "group_size": 10,
                "within_group_index": 0,
                "is_active_group": True,
                "is_active_coefficient": True,
                "true_beta": 1.0,
                "estimated_beta": 0.9,
                "abs_true_beta": 1.0,
                "abs_estimated_beta": 0.9,
                "error": -0.1,
                "sq_error": 0.01,
                "abs_error": 0.1,
                "paired_common_converged": True,
            },
            {
                "setting_id": "setting_5_multimode_equal",
                "replicate_id": 1,
                "method": "RHS",
                "method_label": "RHS",
                "method_type": "bayesian",
                "method_order": 1,
                "setting_label": "Setting 5: Main Showcase, Multi-Mode Heterogeneous Active Groups",
                "family": "multimode_heterogeneous",
                "suite": "main",
                "role": "main showcase setting",
                "coefficient_index": 0,
                "group_id": 0,
                "group_size": 10,
                "within_group_index": 0,
                "is_active_group": True,
                "is_active_coefficient": True,
                "true_beta": 1.0,
                "estimated_beta": 0.8,
                "abs_true_beta": 1.0,
                "abs_estimated_beta": 0.8,
                "error": -0.2,
                "sq_error": 0.04,
                "abs_error": 0.2,
                "paired_common_converged": True,
            },
        ]
    ).to_csv(tmp_path / "coefficient_estimates.csv", index=False)

    result = run_blueprint_cli.main(["build-tables", "--results-dir", str(tmp_path)])
    assert result == 0
    assert (tmp_path / "paper_tables" / "figure_data" / "figure1_coefficient_recovery_profile.csv").exists()
