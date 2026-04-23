from __future__ import annotations

import numpy as np
from pathlib import Path

from simulation_project.src.cli.run_experiment_cli import main as cli_main
from simulation_project.src.experiment_aliases import cli_choice_to_key, normalize_sweep_experiment
from simulation_project.src.experiments import (
    run_exp1_kappa_profile_regimes,
    run_exp2_group_separation,
    run_exp3_linear_benchmark,
    run_exp3a_main_benchmark,
    run_exp3b_boundary_stress,
    run_exp3c_highdim_stress,
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
from simulation_project.src.utils import FitResult, SamplerConfig


def test_normalize_sweep_experiment_aliases() -> None:
    assert normalize_sweep_experiment("exp1_to_exp5") == "all"
    assert normalize_sweep_experiment("EXP3B") == "exp3b"
    assert normalize_sweep_experiment("exp3c") == "exp3c"
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
    assert run_exp3c_highdim_stress is not None
    assert run_exp4_variant_ablation is not None
    assert run_exp5_prior_sensitivity is not None
    assert run_all_experiments is not None


def test_experiments_layer_importable() -> None:
    assert run_exp1_kappa_profile_regimes is not None
    assert run_exp2_group_separation is not None
    assert run_exp3_linear_benchmark is not None
    assert run_exp3a_main_benchmark is not None
    assert run_exp3b_boundary_stress is not None
    assert run_exp3c_highdim_stress is not None
    assert run_exp4_variant_ablation is not None
    assert run_exp5_prior_sensitivity is not None


def test_entrypoint_importable() -> None:
    assert cli_main is not None


def test_architecture_models_and_registry() -> None:
    cfg = RunCommonConfig(
        n_jobs=2,
        method_jobs=3,
        seed=123,
        save_dir="simulation_project",
        enforce_bayes_convergence=True,
        max_convergence_retries=2,
        until_bayes_converged=True,
        sampler_backend="nuts",
    )
    cfg_kwargs = cfg.as_kwargs()
    assert cfg_kwargs["n_jobs"] == 2
    assert cfg_kwargs["method_jobs"] == 3
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


def test_fit_with_convergence_retry_passes_resume_payload_when_enabled() -> None:
    seen_payloads: list[dict | None] = []

    def _fit_stub(sampler_try, attempt: int, resume_payload=None):
        seen_payloads.append(resume_payload)
        diag = {
            "retry_resume_payload": {
                "backend": "gibbs",
                "chain_states": [{"log_sigma": float(attempt)}],
            }
        }
        return FitResult(
            method="GR_RHS",
            status="ok",
            beta_mean=np.asarray([0.0], dtype=float),
            beta_draws=np.asarray([[0.0], [0.0], [0.0], [0.0]], dtype=float),
            kappa_draws=None,
            group_scale_draws=None,
            runtime_seconds=1.0,
            rhat_max=1.02,
            bulk_ess_min=500.0,
            divergence_ratio=0.0,
            converged=bool(attempt >= 1),
            diagnostics=diag,
        )

    out = _fit_with_convergence_retry(
        _fit_stub,
        method="GR_RHS",
        sampler=SamplerConfig(),
        bayes_min_chains=1,
        max_convergence_retries=3,
        enforce_bayes_convergence=True,
        continue_on_retry=True,
    )

    assert len(seen_payloads) == 2
    assert seen_payloads[0] is None
    assert isinstance(seen_payloads[1], dict)
    assert seen_payloads[1].get("backend") == "gibbs"
    assert bool(out.converged)


def test_exp5_defaults_to_full_sensitivity_and_retry_budget_5(monkeypatch) -> None:
    import simulation_project.src.experiments.exp5 as exp5_mod

    captured: dict[str, object] = {}

    def _fake_parallel_rows(tasks, worker, n_jobs, **kwargs):
        captured.setdefault("tasks_by_worker", {})[getattr(worker, "__name__", "worker")] = list(tasks)
        captured["tasks"] = list(tasks)
        if worker is exp5_mod._exp5_screen_prior_worker:
            rows = []
            for task in tasks:
                sid, _group_sizes, _mu, _seed, _sampler, alpha_k, beta_k, _bayes_min_chains, _enforce, _max_retries, _backend = task
                rows.append(
                    {
                        "setting_id": int(sid),
                        "prior_key": f"{float(alpha_k):.6g}|{float(beta_k):.6g}",
                        "alpha_kappa": float(alpha_k),
                        "beta_kappa": float(beta_k),
                        "status": "ok",
                        "converged": (float(alpha_k), float(beta_k)) == (1.0, 3.0),
                        "fit_attempts": 1,
                        "runtime_seconds": 1.0,
                        "error": "",
                    }
                )
            return rows
        return [worker(t) for t in tasks]

    def _fake_exp5_worker(task):
        sid, r, _group_sizes, _mu, _seed, _sampler, prior_grid, _bayes_min_chains, _method_jobs, _enforce, _max_retries, _backend = task
        rows = []
        for pid, (alpha_k, beta_k) in enumerate(prior_grid, start=1):
            rows.append(
                {
                    "setting_id": int(sid),
                    "replicate_id": int(r),
                    "prior_id": int(pid),
                    "alpha_kappa": float(alpha_k),
                    "beta_kappa": float(beta_k),
                    "p0_signal_groups": 3,
                    "tau_target": "groups",
                    "status": "ok",
                    "converged": True,
                    "fit_attempts": 1,
                    "mse_null": 1.0,
                    "mse_signal": 1.0,
                    "group_auroc": 0.75,
                    "kappa_null_mean": 0.1,
                    "kappa_signal_mean": 0.5,
                    "kappa_null_prob_gt_0_1": 0.2,
                    "runtime_seconds": 1.0,
                    "rhat_max": 1.01,
                    "bulk_ess_min": 500.0,
                    "divergence_ratio": 0.0,
                    "error": "",
                    "bridge_ratio_mean": 1.0,
                    "bridge_ratio_min": 1.0,
                    "bridge_ratio_max": 1.0,
                    "bridge_ratio_p95": 1.0,
                    "bridge_ratio_violations": 0,
                    "bridge_ratio_null_mean": 1.0,
                    "bridge_ratio_signal_mean": 1.0,
                    "bridge_ratio_by_group": "{}",
                }
            )
        return rows

    monkeypatch.setattr(exp5_mod, "_parallel_rows", _fake_parallel_rows)
    monkeypatch.setattr(exp5_mod, "_exp5_worker", _fake_exp5_worker)

    save_dir = Path("outputs") / "exp5_default_probe_test"
    save_dir.mkdir(parents=True, exist_ok=True)

    exp5_mod.run_exp5_prior_sensitivity(
        n_jobs=1,
        repeats=1,
        seed=20260415,
        save_dir=str(save_dir),
        max_convergence_retries=None,
    )

    tasks = captured.get("tasks")
    assert isinstance(tasks, list) and tasks
    screen_tasks = captured["tasks_by_worker"]["_exp5_screen_prior_worker"]
    full_tasks = captured["tasks_by_worker"]["_fake_exp5_worker"]
    assert screen_tasks and full_tasks
    for task in full_tasks:
        assert int(task[10]) == 5  # retry budget
        assert len(task[6]) == 2  # default prior + screened-in prior

    summary_partial = save_dir / "results" / "exp5_prior_sensitivity" / "summary_partial.csv"
    assert summary_partial.exists()


def test_exp4_forces_collapsed_backend_only_for_p0_5(monkeypatch, tmp_path) -> None:
    import simulation_project.src.experiments.exp4 as exp4_mod

    captured_tasks: list[tuple] = []

    def _fake_parallel_rows(tasks, worker, n_jobs, **kwargs):
        captured_tasks.extend(list(tasks))
        out = []
        for task in tasks:
            p0_true = int(task[0])
            out.append(
                [
                    {
                        "p0_true": p0_true,
                        "variant": "RHS_oracle",
                        "method_type": "RHS",
                        "status": "ok",
                        "converged": True,
                        "fit_attempts": 1,
                        "tau0_oracle": 0.01,
                        "tau_post_mean": 0.01,
                        "tau_ratio_to_oracle": 1.0,
                        "kappa_null_mean": 0.1,
                        "kappa_signal_mean": 0.2,
                        "g_true_active": 1,
                        "runtime_seconds": 1.0,
                        "rhat_max": 1.0,
                        "bulk_ess_min": 500.0,
                        "divergence_ratio": 0.0,
                        "error": "",
                        "bridge_ratio_mean": 1.0,
                        "bridge_ratio_min": 1.0,
                        "bridge_ratio_max": 1.0,
                        "bridge_ratio_p95": 1.0,
                        "bridge_ratio_violations": 0,
                        "bridge_ratio_null_mean": 1.0,
                        "bridge_ratio_signal_mean": 1.0,
                        "bridge_ratio_by_group": "{}",
                        "mse_null": 1.0,
                        "mse_signal": 1.0,
                        "mse_overall": 1.0,
                    }
                ]
            )
        return out

    monkeypatch.setattr(exp4_mod, "_parallel_rows", _fake_parallel_rows)

    save_dir = tmp_path / "exp4_backend_route_probe"
    exp4_mod.run_exp4_variant_ablation(
        n_jobs=1,
        repeats=1,
        seed=20260415,
        save_dir=str(save_dir),
        p0_list=[5, 15, 30],
        sampler_backend="nuts",
        include_oracle=False,
        enforce_bayes_convergence=False,
        until_bayes_converged=False,
    )

    assert captured_tasks
    backends_by_p0: dict[int, set[str]] = {}
    for task in captured_tasks:
        p0_true = int(task[0])
        backend = str(task[11])
        backends_by_p0.setdefault(p0_true, set()).add(backend)

    assert backends_by_p0.get(5) == {"collapsed"}
    assert backends_by_p0.get(15) == {"nuts"}
    assert backends_by_p0.get(30) == {"nuts"}


def test_exp4_default_correlation_uses_0_8_and_0_2(monkeypatch) -> None:
    import simulation_project.src.experiments.exp4 as exp4_mod
    from simulation_project.src.utils import SamplerConfig

    captured: dict[str, float] = {}

    def _fake_sample_correlated_design(*, n, group_sizes, rho_within, rho_between, seed):
        captured["rho_within"] = float(rho_within)
        captured["rho_between"] = float(rho_between)
        p = int(sum(group_sizes))
        return np.zeros((int(n), p), dtype=float), np.eye(p, dtype=float)

    def _fake_fit_with_convergence_retry(*args, **kwargs):
        return FitResult(
            method="RHS",
            status="ok",
            beta_mean=np.zeros(50, dtype=float),
            beta_draws=None,
            kappa_draws=None,
            group_scale_draws=None,
            runtime_seconds=0.0,
            rhat_max=1.0,
            bulk_ess_min=500.0,
            divergence_ratio=0.0,
            converged=True,
            tau_draws=None,
            error="",
            diagnostics={},
        )

    monkeypatch.setattr("simulation_project.src.utils.sample_correlated_design", _fake_sample_correlated_design)
    monkeypatch.setattr(exp4_mod, "_fit_with_convergence_retry", _fake_fit_with_convergence_retry)

    variants = {"RHS_oracle": {"method": "RHS"}}
    task = (5, 1, 20260415, [10, 10, 10, 10, 10], SamplerConfig(), variants, 4, 1, True, 1, 100, "collapsed")
    rows = exp4_mod._exp4_worker(task)

    assert rows
    assert captured["rho_within"] == 0.8
    assert captured["rho_between"] == 0.2
