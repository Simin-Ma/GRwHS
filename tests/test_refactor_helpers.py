from __future__ import annotations

from simulation_project.src.cli.run_experiment_cli import main as cli_main
from simulation_project.src.experiment_aliases import cli_choice_to_key, normalize_sweep_experiment
from simulation_project.src.experiments import (
    run_exp1_kappa_profile_regimes,
    run_exp2_group_separation,
    run_exp3_linear_benchmark,
    run_exp3a_main_benchmark,
    run_exp3b_boundary_stress,
    run_exp4_variant_ablation,
    run_exp5_prior_sensitivity,
)
from simulation_project.src.experiments.evaluation import _evaluate_row, _kappa_group_means, _kappa_group_prob_gt
from simulation_project.src.experiments.fitting import _fit_all_methods, _fit_with_convergence_retry
from simulation_project.src.experiments.method_registry import MethodRegistry, build_default_method_registry
from simulation_project.src.experiments.orchestration import run_all_experiments
from simulation_project.src.experiments.methods.helpers import as_int_groups, fit_error_result, scaled_iteration_budget
from simulation_project.src.experiments.schemas import RunCommonConfig, RunManifest
from simulation_project.src.output_layout import resolve_analysis_dir, resolve_run_save_dir, resolve_workspace_dir
from simulation_project.src.utils import SamplerConfig


def test_normalize_sweep_experiment_aliases() -> None:
    assert normalize_sweep_experiment("exp1_to_exp5") == "all"
    assert normalize_sweep_experiment("EXP3B") == "exp3b"
    assert normalize_sweep_experiment("4") == "exp4"


def test_cli_choice_to_key() -> None:
    assert cli_choice_to_key("1") == "exp1"
    assert cli_choice_to_key("analysis") == "analysis"


def test_scaled_iteration_budget_clamps_floor_and_cap() -> None:
    sampler = SamplerConfig(warmup=100, post_warmup_draws=150)
    burnin, draws = scaled_iteration_budget(
        sampler,
        iter_mult=2,
        iter_floor=500,
        iter_cap=600,
    )
    assert burnin == 500
    assert draws == 500

    burnin2, draws2 = scaled_iteration_budget(
        sampler,
        iter_mult=10,
        iter_floor=100,
        iter_cap=700,
    )
    assert burnin2 == 700
    assert draws2 == 700


def test_fit_error_result_shape() -> None:
    res = fit_error_result("RHS", "boom")
    assert res.method == "RHS"
    assert res.status == "error"
    assert res.error == "boom"
    assert res.beta_mean is None
    assert res.converged is False


def test_as_int_groups() -> None:
    groups = as_int_groups([(0, 1), [2, 3]])
    assert groups == [[0, 1], [2, 3]]


def test_new_refactor_modules_importable() -> None:
    assert _fit_all_methods is not None
    assert _fit_with_convergence_retry is not None
    assert _evaluate_row is not None
    assert _kappa_group_means is not None
    assert _kappa_group_prob_gt is not None
    assert run_exp1_kappa_profile_regimes is not None
    assert run_exp2_group_separation is not None
    assert run_exp3_linear_benchmark is not None
    assert run_exp3a_main_benchmark is not None
    assert run_exp3b_boundary_stress is not None
    assert run_exp4_variant_ablation is not None
    assert run_exp5_prior_sensitivity is not None
    assert run_all_experiments is not None


def test_experiments_layer_importable() -> None:
    assert run_exp1_kappa_profile_regimes is not None
    assert run_exp2_group_separation is not None
    assert run_exp3_linear_benchmark is not None
    assert run_exp3a_main_benchmark is not None
    assert run_exp3b_boundary_stress is not None
    assert run_exp4_variant_ablation is not None
    assert run_exp5_prior_sensitivity is not None


def test_entrypoint_importable() -> None:
    assert cli_main is not None


def test_architecture_models_and_registry() -> None:
    cfg = RunCommonConfig(
        n_jobs=2,
        seed=123,
        save_dir="simulation_project",
        profile="full",
        enforce_bayes_convergence=True,
        max_convergence_retries=2,
        until_bayes_converged=True,
        sampler_backend="nuts",
    )
    cfg_kwargs = cfg.as_kwargs()
    assert cfg_kwargs["n_jobs"] == 2
    assert cfg_kwargs["sampler_backend"] == "nuts"

    manifest = RunManifest(
        exp_key="expX",
        timestamp="20260101_000000",
        run_dir="d:/tmp/run",
        result_paths={"summary": "d:/tmp/run/summary.csv"},
        run_summary_table="d:/tmp/run/table.csv",
        run_summary_md="d:/tmp/run/summary.md",
        run_analysis_json="d:/tmp/run/analysis.json",
        archived_artifacts=["d:/tmp/run/artifacts/a.csv"],
    )
    manifest_dict = manifest.to_dict()
    assert manifest_dict["exp_key"] == "expX"
    assert "archived_artifacts" in manifest_dict

    registry = build_default_method_registry()
    assert isinstance(registry, MethodRegistry)
    names = registry.names()
    assert "GR_RHS" in names
    assert "RHS" in names


def test_output_layout_auto_session_and_analysis_resolution(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    auto_dir = resolve_run_save_dir(None, workspace=str(workspace), run_label="cli_exp3")
    assert auto_dir.exists()
    assert "sessions" in auto_dir.as_posix()

    latest_dir = resolve_analysis_dir(None, workspace=str(workspace))
    assert latest_dir == auto_dir


def test_output_layout_relative_explicit_path_is_centralized(tmp_path) -> None:
    workspace = resolve_workspace_dir(str(tmp_path / "workspace"), create=True)
    explicit = resolve_run_save_dir("my_custom/output", workspace=str(workspace), run_label="cli_exp1")
    assert explicit.exists()
    assert (workspace / "adhoc") in explicit.parents
