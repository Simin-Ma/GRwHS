from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from simulation_project.src.experiments.runtime import _parallel_rows
from simulation_project.src.utils import FitResult, load_pandas

from .config import (
    BenchmarkConfig,
    family_spec_from_dict,
    force_until_converged_gate,
    setting_spec_from_dict,
)
from .dataset import generate_grouped_dataset, save_grouped_dataset
from .evaluation import evaluate_method_result
from .fitting import fit_benchmark_methods
from .reporting import (
    DEFAULT_REQUIRED_METRICS,
    build_paired_deltas,
    build_paired_summary,
    build_summary,
    default_setting_group_cols,
    write_json_manifest,
)
from .table_builder import build_paper_tables
from .utils import ensure_dir, save_json


def _group_config_name(group_sizes: Sequence[int]) -> str:
    sizes = [int(x) for x in group_sizes]
    if len(set(sizes)) == 1:
        return f"G{sizes[0]}x{len(sizes)}"
    return "G" + "_".join(str(x) for x in sizes)


def _stringify_groups(group_sizes: Sequence[int]) -> str:
    return "[" + ",".join(str(int(x)) for x in group_sizes) + "]"


def _stringify_active_groups(groups: Sequence[int]) -> str:
    return "[" + ",".join(str(int(x)) for x in groups) + "]"


def _error_fit_result(method: str, message: str) -> FitResult:
    return FitResult(
        method=str(method),
        status="error",
        beta_mean=None,
        beta_draws=None,
        kappa_draws=None,
        group_scale_draws=None,
        runtime_seconds=0.0,
        rhat_max=float("nan"),
        bulk_ess_min=float("nan"),
        divergence_ratio=float("nan"),
        converged=False,
        tau_draws=None,
        error=str(message),
        diagnostics={},
    )


def _gate_from_payload(payload: Mapping[str, Any]):
    from .schemas import ConvergenceGateSpec

    gate_dict = dict(payload)
    return ConvergenceGateSpec(**gate_dict)


def _task_payloads(config: BenchmarkConfig) -> list[dict[str, Any]]:
    family_payload = {name: spec.to_dict() for name, spec in config.families.items()}
    tasks: list[dict[str, Any]] = []
    for setting in config.settings:
        setting_methods = [method for method in config.methods.roster if method in set(setting.methods)]
        for replicate_id in range(1, int(config.runner.repeats) + 1):
            tasks.append(
                {
                    "setting": setting.to_dict(),
                    "family_specs": family_payload,
                    "replicate_id": int(replicate_id),
                    "master_seed": int(config.runner.seed),
                    "task": str(config.runner.task),
                    "methods": list(setting_methods),
                    "gate": config.convergence_gate.to_dict(),
                    "grrhs_kwargs": dict(config.methods.grrhs_kwargs),
                    "gigg_config": dict(config.methods.gigg_config),
                    "method_jobs": int(config.runner.method_jobs),
                    "save_datasets": bool(config.runner.save_datasets),
                    "dataset_dir": str(Path(config.runner.output_dir) / "datasets"),
                }
            )
    return tasks


def _run_replicate_task(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    setting = setting_spec_from_dict(task["setting"], default_methods=tuple(task.get("methods", [])))
    family_specs = {
        str(name): family_spec_from_dict(str(name), payload)
        for name, payload in dict(task["family_specs"]).items()
    }
    dataset = generate_grouped_dataset(
        setting,
        replicate_id=int(task["replicate_id"]),
        master_seed=int(task["master_seed"]),
        family_specs=family_specs,
    )
    if bool(task.get("save_datasets", False)):
        save_grouped_dataset(dataset, Path(str(task["dataset_dir"])) / setting.setting_id)

    p0 = int(np.count_nonzero(np.asarray(dataset.beta, dtype=float)))
    p0_groups = int(sum(np.any(np.abs(np.asarray(dataset.beta)[np.asarray(group, dtype=int)]) > 1e-12) for group in dataset.groups))
    methods = [str(method) for method in task["methods"]]

    try:
        results = fit_benchmark_methods(
            dataset.X_train,
            dataset.y_train,
            dataset.groups,
            task=str(task["task"]),
            seed=int(dataset.metadata.get("seed", 0)) + 17,
            p0=p0,
            p0_groups=p0_groups,
            methods=methods,
            gate=_gate_from_payload(task["gate"]),
            grrhs_kwargs=dict(task.get("grrhs_kwargs", {})),
            gigg_config=dict(task.get("gigg_config", {})),
            method_jobs=int(task.get("method_jobs", 1)),
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        results = {method: _error_fit_result(method, message) for method in methods}

    signal_meta = dataset.signal_draw.metadata()
    setting_meta = dataset.setting.to_dict()
    out_rows: list[dict[str, Any]] = []
    for method in methods:
        result = results.get(method, _error_fit_result(method, "Missing result from fit_benchmark_methods"))
        eval_row = evaluate_method_result(result, dataset)
        out_rows.append(
            {
                "setting_id": setting.setting_id,
                "setting_label": setting.label,
                "family": setting.family,
                "suite": setting.suite,
                "role": setting.role,
                "notes": setting.notes,
                "group_config": _group_config_name(setting.group_sizes),
                "group_sizes": _stringify_groups(setting.group_sizes),
                "active_groups": _stringify_active_groups(setting.active_groups),
                "n_train": int(setting.n_train),
                "n_test": int(setting.n_test),
                "rho_within": float(setting.rho_within),
                "rho_between": float(setting.rho_between),
                "target_r2": float(setting.target_r2),
                "replicate_id": int(dataset.metadata["replicate_id"]),
                "seed": int(dataset.metadata["seed"]),
                "sigma2": float(dataset.sigma2),
                "signal_variance_population": float(dataset.metadata["signal_variance_population"]),
                "implied_population_r2": float(dataset.metadata["implied_population_r2"]),
                "beta_nonzero": int(np.count_nonzero(dataset.beta)),
                "p0_true": int(p0),
                "p0_groups_true": int(p0_groups),
                "signal_family": signal_meta["family"],
                "signal_energy_shares_json": json.dumps(signal_meta["energy_shares"], ensure_ascii=True, sort_keys=True),
                "signal_support_fractions_json": json.dumps(signal_meta["support_fractions"], ensure_ascii=True, sort_keys=True),
                "signal_concentrations_json": json.dumps(signal_meta["concentrations"], ensure_ascii=True, sort_keys=True),
                "signal_group_signs_json": json.dumps(signal_meta["group_signs"], ensure_ascii=True, sort_keys=True),
                "signal_support_indices_json": json.dumps(signal_meta["support_indices"], ensure_ascii=True, sort_keys=True),
                "signal_acceptance_restarts": int(signal_meta["acceptance_restarts"]),
                "method": str(method),
                "method_label": eval_row["method_label"],
                "method_type": eval_row["method_type"],
                "status": str(result.status),
                "converged": bool(result.converged),
                "error": str(result.error),
                "runtime_seconds": float(result.runtime_seconds),
                "rhat_max": float(result.rhat_max),
                "bulk_ess_min": float(result.bulk_ess_min),
                "divergence_ratio": float(result.divergence_ratio),
                **{key: value for key, value in eval_row.items() if key not in {"method_label", "method_type"}},
                "setting_json": json.dumps(setting_meta, ensure_ascii=True, sort_keys=True),
            }
        )
    return out_rows


def run_benchmark(config: BenchmarkConfig) -> dict[str, str]:
    pd = load_pandas()
    config = replace(config, convergence_gate=force_until_converged_gate(config.convergence_gate))
    out_dir = ensure_dir(config.runner.output_dir)
    paper_dir = ensure_dir(out_dir / "paper_tables")

    spec_path = save_json(config.to_manifest(), out_dir / "benchmark_spec.json")
    tasks = _task_payloads(config)
    task_chunks = _parallel_rows(
        tasks,
        _run_replicate_task,
        n_jobs=int(config.runner.n_jobs),
        prefer_process=bool(int(config.runner.n_jobs) > 1),
        process_fallback="thread",
        progress_desc="Simulation Second Benchmark",
    )

    rows: list[dict[str, Any]] = []
    for chunk in task_chunks:
        rows.extend(chunk)

    raw = pd.DataFrame(rows)
    if not raw.empty:
        raw = raw.sort_values(["setting_id", "replicate_id", "method"], kind="stable").reset_index(drop=True)

    group_cols = default_setting_group_cols(raw)
    summary = build_summary(
        raw,
        group_cols=group_cols,
        method_order=config.methods.roster,
        required_metric_cols=DEFAULT_REQUIRED_METRICS,
    )
    paired_raw, paired_stats, summary_paired = build_paired_summary(
        raw,
        group_cols=group_cols,
        method_levels=config.methods.roster,
        required_metric_cols=config.runner.required_metrics_for_pairing,
        method_order=config.methods.roster,
    )
    paired_deltas = build_paired_deltas(
        paired_raw,
        group_cols=group_cols,
        baseline_method=config.runner.baseline_method,
    )

    raw_path = out_dir / "raw_results.csv"
    summary_path = out_dir / "summary.csv"
    paired_raw_path = out_dir / "raw_results_paired.csv"
    paired_stats_path = out_dir / "paired_replicate_stats.csv"
    summary_paired_path = out_dir / "summary_paired.csv"
    paired_deltas_path = out_dir / "summary_paired_deltas.csv"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    paired_raw.to_csv(paired_raw_path, index=False)
    paired_stats.to_csv(paired_stats_path, index=False)
    summary_paired.to_csv(summary_paired_path, index=False)
    paired_deltas.to_csv(paired_deltas_path, index=False)

    result_paths: dict[str, str] = {
        "benchmark_spec": str(spec_path),
        "raw_results": str(raw_path),
        "summary": str(summary_path),
        "raw_results_paired": str(paired_raw_path),
        "paired_replicate_stats": str(paired_stats_path),
        "summary_paired": str(summary_paired_path),
        "summary_paired_deltas": str(paired_deltas_path),
    }

    if bool(config.runner.build_tables):
        result_paths.update(
            build_paper_tables(
                raw,
                out_dir=paper_dir,
                method_order=config.methods.roster,
                group_cols=group_cols,
                required_metric_cols=config.runner.required_metrics_for_pairing,
            )
        )

    manifest = {
        "package": "simulation_second",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_rows": int(raw.shape[0]),
        "n_settings": int(len(config.settings)),
        "repeats": int(config.runner.repeats),
        "methods": list(config.methods.roster),
        "group_cols": list(group_cols),
        "result_paths": dict(result_paths),
    }
    manifest_path = write_json_manifest(manifest, out_dir / "run_manifest.json")
    result_paths["run_manifest"] = str(manifest_path)
    return result_paths
