from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from simulation_project.src.utils import FitResult

from simulation_mechanism.src.config import load_mechanism_config, mechanism_config_from_payload
from simulation_mechanism.src.dgp import generate_mechanism_dataset
from simulation_mechanism.src.plotting import build_mechanism_figures_from_results_dir
from simulation_mechanism.src.runner import run_mechanism
from simulation_mechanism.src.schemas import active_group_mask
from simulation_mechanism.src.suite import build_mechanism_suite, get_setting_by_id
from simulation_mechanism.src.utils import mechanism_method_family


def test_mechanism_suite_has_twelve_settings() -> None:
    settings = build_mechanism_suite()
    assert len(settings) == 12
    assert {setting.experiment_id for setting in settings} == {"M1", "M2", "M3", "M4"}


def test_mixed_decoy_dataset_marks_decoy_group() -> None:
    setting = get_setting_by_id("m2_mixed_decoy_rw080")
    dataset = generate_mechanism_dataset(setting, replicate_id=2, master_seed=20260425)
    assert dataset.X_train.shape == (100, 50)
    assert dataset.X_test.shape == (30, 50)
    assert int(dataset.metadata["decoy_group"]) >= 0
    assert int(dataset.metadata["p0_groups_true"]) == 2


def test_default_mechanism_yaml_loads() -> None:
    config = load_mechanism_config()
    assert config.package == "simulation_mechanism"
    assert len(config.settings) == 12
    assert config.methods.standard_methods == ("GR_RHS", "RHS")
    assert config.methods.grrhs_kwargs["tau_target"] == "groups"
    assert config.convergence_gate.enforce_bayes_convergence is True
    assert config.convergence_gate.max_convergence_retries == -1


def test_mechanism_config_forces_until_convergence_even_if_payload_disables_it() -> None:
    payload = {
        "convergence_gate": {
            "enforce_bayes_convergence": False,
            "max_convergence_retries": 1,
            "chains": 3,
        },
        "settings": [],
    }
    config = mechanism_config_from_payload(payload)
    assert config.convergence_gate.enforce_bayes_convergence is True
    assert config.convergence_gate.max_convergence_retries == -1
    assert config.convergence_gate.chains == 3


def test_run_mechanism_pipeline_smoke(monkeypatch, tmp_path) -> None:
    config = load_mechanism_config()
    config = replace(
        config,
        settings=(
            config.setting_map()["m2_mixed_decoy_rw080"],
            config.setting_map()["m4_ablation_p0_05"],
        ),
        runner=replace(
            config.runner,
            repeats=2,
            n_jobs=1,
            method_jobs=1,
            output_dir=str(tmp_path),
            build_tables=True,
        ),
    )

    beta_scales = {
        "GR_RHS": 0.98,
        "RHS": 0.85,
        "RHS_oracle": 0.88,
        "GR_RHS_fixed_10x": 0.90,
        "GR_RHS_oracle": 0.96,
        "GR_RHS_no_local_scales": 0.92,
        "GR_RHS_shared_kappa": 0.89,
        "GR_RHS_no_kappa": 0.87,
    }
    kappa_profiles = {
        "GR_RHS": (0.82, 0.18),
        "GR_RHS_fixed_10x": (0.76, 0.28),
        "GR_RHS_oracle": (0.80, 0.20),
        "GR_RHS_no_local_scales": (0.70, 0.34),
        "GR_RHS_shared_kappa": (0.56, 0.46),
        "GR_RHS_no_kappa": (0.50, 0.50),
    }

    def fake_fit_setting_methods(
        dataset,
        setting,
        *,
        task,
        gate,
        grrhs_kwargs,
        gigg_config,
        ablation_variant_specs,
        method_jobs,
    ):
        out = {}
        p = int(dataset.beta.shape[0])
        active_mask = active_group_mask(dataset.beta, dataset.groups)
        for method in setting.methods:
            family = mechanism_method_family(method)
            scale = beta_scales.get(str(method), 0.9)
            center = np.asarray(dataset.beta, dtype=float) * scale
            draws = np.vstack([center - 0.01, center, center + 0.01])
            kappa_draws = None
            if family == "GR_RHS":
                active_val, null_val = kappa_profiles.get(str(method), (0.75, 0.25))
                base = np.where(active_mask, active_val, null_val).astype(float)
                kappa_draws = np.vstack([np.clip(base - 0.02, 0.01, 0.99), base, np.clip(base + 0.02, 0.01, 0.99)])
            out[str(method)] = FitResult(
                method=family,
                status="ok",
                beta_mean=center.reshape(p),
                beta_draws=draws,
                kappa_draws=kappa_draws,
                group_scale_draws=None,
                runtime_seconds=0.15,
                rhat_max=1.0,
                bulk_ess_min=500.0,
                divergence_ratio=0.0,
                converged=True,
                tau_draws=np.asarray([0.20, 0.22, 0.24], dtype=float),
                diagnostics={},
            )
        return out

    monkeypatch.setattr("simulation_mechanism.src.runner.fit_setting_methods", fake_fit_setting_methods)

    result_paths = run_mechanism(config)
    assert Path(result_paths["summary_paired"]).exists()
    assert Path(result_paths["per_group_kappa"]).exists()
    assert (tmp_path / "paper_tables" / "paper_table_mechanism.md").exists()
    assert (tmp_path / "paper_tables" / "figure_data" / "figure4_representative_profile.csv").exists()
    assert (tmp_path / "paper_tables" / "figure_data" / "figure6_ablation_deltas.csv").exists()
    assert (tmp_path / "figures" / "figure1_mechanism_schematic.png").exists()
    assert (tmp_path / "figures" / "figure6_ablation.png").exists()

    raw = pd.read_csv(tmp_path / "raw_results.csv")
    paired_deltas = pd.read_csv(tmp_path / "summary_paired_deltas.csv")
    per_group = pd.read_csv(tmp_path / "per_group_kappa.csv")
    fig4 = pd.read_csv(tmp_path / "paper_tables" / "figure_data" / "figure4_representative_profile.csv")

    assert set(raw["experiment_id"]) == {"M2", "M4"}
    assert "paired_common_converged" in per_group.columns
    assert not fig4.empty
    m4_deltas = paired_deltas.loc[paired_deltas["experiment_id"] == "M4"]
    assert not m4_deltas.empty
    assert set(m4_deltas["baseline_method"]) == {"GR_RHS"}

    rebuilt = build_mechanism_figures_from_results_dir(tmp_path)
    assert Path(rebuilt["figure3_correlation_ambiguity"]).exists()
    assert Path(rebuilt["figure6_ablation"]).exists()
