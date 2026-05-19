from __future__ import annotations

import numpy as np

from simulation_second.src.bayes_kernel.experiments.evaluation import (
    _evaluate_row,
    _kappa_group_means,
    _kappa_group_prob_gt,
)
from simulation_second.src.bayes_kernel.experiments.fitting import (
    _fit_all_methods,
    _fit_with_convergence_retry,
)
from simulation_second.src.bayes_kernel.experiments.method_registry import (
    MethodRegistry,
    build_default_method_registry,
)
from simulation_second.src.bayes_kernel.experiments.methods.helpers import (
    as_int_groups,
    fit_error_result,
    scaled_iteration_budget,
)
from simulation_second.src.bayes_kernel.experiments.schemas import RunCommonConfig, RunManifest
from simulation_second.src.bayes_kernel.utils import FitResult, SamplerConfig
from simulation_second.src.cli.run_blueprint_cli import main as blueprint_cli_main
from Simulation_highdimension.src.cli.run_highdimension_cli import main as highdim_cli_main
from simulation_mechanism.src.cli.run_mechanism_cli import main as mechanism_cli_main
from real_data_experiment.src.cli.run_real_data_cli import main as real_data_cli_main


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


def test_current_refactor_modules_importable() -> None:
    assert _fit_all_methods is not None
    assert _fit_with_convergence_retry is not None
    assert _evaluate_row is not None
    assert _kappa_group_means is not None
    assert _kappa_group_prob_gt is not None
    assert blueprint_cli_main is not None
    assert highdim_cli_main is not None
    assert mechanism_cli_main is not None
    assert real_data_cli_main is not None


def test_architecture_models_and_registry() -> None:
    cfg = RunCommonConfig(
        n_jobs=2,
        method_jobs=3,
        seed=123,
        save_dir="simulation_second",
        skip_run_analysis=True,
        archive_artifacts=False,
        enforce_bayes_convergence=True,
        max_convergence_retries=2,
        until_bayes_converged=True,
    )
    cfg_kwargs = cfg.as_kwargs()
    assert cfg_kwargs["n_jobs"] == 2
    assert cfg_kwargs["method_jobs"] == 3
    assert cfg_kwargs["skip_run_analysis"] is True
    assert cfg_kwargs["archive_artifacts"] is False
    assert cfg_kwargs["until_bayes_converged"] is True

    manifest = RunManifest(
        exp_key="current",
        timestamp="20260101_000000",
        run_dir="d:/tmp/run",
        result_paths={"summary": "d:/tmp/run/summary.csv"},
        run_summary_table="d:/tmp/run/table.csv",
        run_summary_md="d:/tmp/run/summary.md",
        run_analysis_json="d:/tmp/run/analysis.json",
        archived_artifacts=["d:/tmp/run/artifacts/a.csv"],
    )
    manifest_dict = manifest.to_dict()
    assert manifest_dict["exp_key"] == "current"
    assert "archived_artifacts" in manifest_dict

    registry = build_default_method_registry()
    assert isinstance(registry, MethodRegistry)
    names = registry.names()
    for method in ("GR_RHS", "GR_RHS_B01", "GR_RHS_B04", "GR_RHS_B08", "GR_RHS_Adaptive", "RHS", "GIGG_MMLE"):
        assert method in names


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
