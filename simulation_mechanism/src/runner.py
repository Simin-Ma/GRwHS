from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from simulation_project.src.experiments.runtime import _parallel_rows
from simulation_project.src.utils import FitResult, load_pandas

from .config import MechanismConfig, force_until_converged_gate, setting_spec_from_dict
from .dgp import generate_mechanism_dataset, save_mechanism_dataset
from .evaluation import build_group_kappa_rows, evaluate_method_result
from .fitting import fit_setting_methods, oracle_tau0_for_method
from .reporting import (
    DEFAULT_REQUIRED_METRICS,
    build_paired_deltas,
    build_paired_summary_by_setting,
    build_summary,
    default_setting_group_cols,
)
from .plotting import build_mechanism_figures_from_results_dir
from .table_builder import build_paper_tables
from .utils import ensure_dir, run_timestamp_tag, save_json, snapshot_result_files, stringify_groups


def _group_config_name(group_sizes: Sequence[int]) -> str:
    sizes = [int(x) for x in group_sizes]
    if len(set(sizes)) == 1:
        return f"G{sizes[0]}x{len(sizes)}"
    return "G" + "_".join(str(x) for x in sizes)


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

    return ConvergenceGateSpec(**dict(payload))


def _task_payloads(config: MechanismConfig) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for setting in config.settings:
        for replicate_id in range(1, int(config.runner.repeats) + 1):
            tasks.append(
                {
                    "setting": setting.to_dict(),
                    "replicate_id": int(replicate_id),
                    "master_seed": int(config.runner.seed),
                    "task": str(config.runner.task),
                    "gate": config.convergence_gate.to_dict(),
                    "grrhs_kwargs": dict(config.methods.grrhs_kwargs),
                    "gigg_config": dict(config.methods.gigg_config),
                    "ablation_variant_specs": {
                        str(name): dict(spec)
                        for name, spec in config.methods.ablation_variant_specs.items()
                    },
                    "standard_methods": list(config.methods.standard_methods),
                    "ablation_variants": list(config.methods.ablation_variants),
                    "method_jobs": int(config.runner.method_jobs),
                    "save_datasets": bool(config.runner.save_datasets),
                    "dataset_dir": str(Path(config.runner.output_dir) / "datasets"),
                }
            )
    return tasks


def _run_replicate_task(task: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    setting = setting_spec_from_dict(
        task["setting"],
        default_methods=tuple(str(item) for item in task.get("standard_methods", [])),
        default_ablation_variants=tuple(str(item) for item in task.get("ablation_variants", [])),
    )
    dataset = generate_mechanism_dataset(
        setting,
        replicate_id=int(task["replicate_id"]),
        master_seed=int(task["master_seed"]),
    )
    if bool(task.get("save_datasets", False)):
        save_mechanism_dataset(dataset, Path(str(task["dataset_dir"])) / setting.setting_id)

    methods = [str(method) for method in setting.methods]
    try:
        results = fit_setting_methods(
            dataset,
            setting,
            task=str(task["task"]),
            gate=_gate_from_payload(task["gate"]),
            grrhs_kwargs=dict(task.get("grrhs_kwargs", {})),
            gigg_config=dict(task.get("gigg_config", {})),
            ablation_variant_specs=dict(task.get("ablation_variant_specs", {})),
            method_jobs=int(task.get("method_jobs", 1)),
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        results = {method: _error_fit_result(method, message) for method in methods}

    raw_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for method in methods:
        result = results.get(method, _error_fit_result(method, "Missing result from fit_setting_methods"))
        eval_row = evaluate_method_result(method_name=method, result=result, dataset=dataset)
        tau_post_mean = float("nan")
        if result.tau_draws is not None:
            tau_post_mean = float(np.mean(np.asarray(result.tau_draws, dtype=float)))
        tau0_oracle = float("nan")
        tau_ratio_to_oracle = float("nan")
        if str(setting.experiment_kind).strip().lower() == "ablation":
            tau0_oracle = oracle_tau0_for_method(
                method,
                dataset=dataset,
                grrhs_kwargs=dict(task.get("grrhs_kwargs", {})),
                ablation_variant_specs=dict(task.get("ablation_variant_specs", {})),
            )
            if np.isfinite(tau_post_mean) and np.isfinite(tau0_oracle) and abs(tau0_oracle) > 1e-12:
                tau_ratio_to_oracle = float(tau_post_mean / tau0_oracle)

        row = {
            "experiment_id": setting.experiment_id,
            "experiment_label": setting.experiment_label,
            "experiment_kind": setting.experiment_kind,
            "line_id": setting.line_id,
            "line_label": setting.line_label,
            "scientific_question": setting.scientific_question,
            "primary_metric": setting.primary_metric,
            "setting_id": setting.setting_id,
            "setting_label": setting.setting_label,
            "suite": setting.suite,
            "role": setting.role,
            "notes": setting.notes,
            "group_config": _group_config_name(setting.group_sizes),
            "group_sizes": stringify_groups(setting.group_sizes),
            "active_groups": _stringify_active_groups(setting.active_groups),
            "n_train": int(setting.n_train),
            "n_test": int(setting.n_test),
            "rho_within": float(setting.rho_within),
            "rho_between": float(setting.rho_between),
            "target_snr": float(setting.target_snr),
            "within_group_pattern": str(setting.within_group_pattern),
            "complexity_pattern": str(setting.complexity_pattern),
            "total_active_coeff": int(setting.total_active_coeff),
            "replicate_id": int(dataset.metadata["replicate_id"]),
            "seed": int(dataset.metadata["seed"]),
            "sigma2_true": float(dataset.sigma2),
            "signal_variance_population": float(dataset.metadata["signal_variance_population"]),
            "implied_population_snr": float(dataset.metadata["implied_population_snr"]),
            "p0_true": int(dataset.metadata["p0_true"]),
            "p0_groups_true": int(dataset.metadata["p0_groups_true"]),
            "decoy_group": int(dataset.metadata.get("decoy_group", -1)),
            "method": str(method),
            "error": str(result.error),
            "tau_post_mean": float(tau_post_mean),
            "tau0_oracle": float(tau0_oracle),
            "tau_ratio_to_oracle": float(tau_ratio_to_oracle),
            **eval_row,
            "setting_json": json.dumps(setting.to_dict(), ensure_ascii=True, sort_keys=True),
        }
        raw_rows.append(row)
        group_rows.extend(
            build_group_kappa_rows(
                method_name=method,
                result=result,
                dataset=dataset,
                base_fields={
                    "experiment_id": setting.experiment_id,
                    "experiment_label": setting.experiment_label,
                    "experiment_kind": setting.experiment_kind,
                    "setting_id": setting.setting_id,
                    "setting_label": setting.setting_label,
                    "line_id": setting.line_id,
                    "line_label": setting.line_label,
                    "replicate_id": int(dataset.metadata["replicate_id"]),
                    "seed": int(dataset.metadata["seed"]),
                    "rho_within": float(setting.rho_within),
                    "rho_between": float(setting.rho_between),
                    "within_group_pattern": str(setting.within_group_pattern),
                    "complexity_pattern": str(setting.complexity_pattern),
                    "total_active_coeff": int(setting.total_active_coeff),
                    "decoy_group": int(dataset.metadata.get("decoy_group", -1)),
                },
            )
        )
    return {"raw_rows": raw_rows, "group_rows": group_rows}


def _dedup_method_order(settings: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    order: list[str] = []
    for setting in settings:
        for method in setting.methods:
            name = str(method)
            if name not in seen:
                seen.add(name)
                order.append(name)
    return order


def run_mechanism(config: MechanismConfig) -> dict[str, str]:
    pd = load_pandas()
    config = replace(config, convergence_gate=force_until_converged_gate(config.convergence_gate))
    out_dir = ensure_dir(config.runner.output_dir)
    paper_dir = ensure_dir(out_dir / "paper_tables")
    run_timestamp = run_timestamp_tag()

    spec_path = save_json(config.to_manifest(), out_dir / "mechanism_spec.json")
    task_chunks = _parallel_rows(
        _task_payloads(config),
        _run_replicate_task,
        n_jobs=int(config.runner.n_jobs),
        prefer_process=bool(int(config.runner.n_jobs) > 1),
        process_fallback="thread",
        progress_desc="Simulation Mechanism",
    )

    raw_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for chunk in task_chunks:
        raw_rows.extend(chunk.get("raw_rows", []))
        group_rows.extend(chunk.get("group_rows", []))

    raw = pd.DataFrame(raw_rows)
    per_group = pd.DataFrame(group_rows)
    if not raw.empty:
        raw = raw.sort_values(["experiment_id", "setting_id", "replicate_id", "method"], kind="stable").reset_index(drop=True)
    if not per_group.empty:
        per_group = per_group.sort_values(["experiment_id", "setting_id", "replicate_id", "method", "group_id"], kind="stable").reset_index(drop=True)

    group_cols = default_setting_group_cols(raw)
    method_order = _dedup_method_order(config.settings)
    summary = build_summary(
        raw,
        group_cols=group_cols,
        method_order=method_order,
        required_metric_cols=DEFAULT_REQUIRED_METRICS,
    )
    paired_raw, paired_stats, summary_paired = build_paired_summary_by_setting(
        raw,
        group_cols=group_cols,
        required_metric_cols=config.runner.required_metrics_for_pairing,
        method_order=method_order,
    )
    paired_deltas = build_paired_deltas(
        paired_raw,
        group_cols=group_cols,
        baseline_method=config.runner.baseline_method,
        baseline_by_experiment_kind={"ablation": config.runner.ablation_baseline_method},
    )

    if not per_group.empty:
        pair_keys = paired_raw.loc[:, [col for col in ["setting_id", "replicate_id", "method"] if col in paired_raw.columns]].drop_duplicates()
        if not pair_keys.empty:
            pair_keys = pair_keys.assign(paired_common_converged=True)
            per_group = per_group.merge(
                pair_keys,
                on=[col for col in ["setting_id", "replicate_id", "method"] if col in pair_keys.columns],
                how="left",
            )
            per_group["paired_common_converged"] = per_group["paired_common_converged"].fillna(False).astype(bool)
        else:
            per_group["paired_common_converged"] = False
    per_group_paired = per_group.loc[
        per_group["paired_common_converged"].fillna(False).astype(bool)
    ].copy() if not per_group.empty else per_group.copy()

    raw_path = out_dir / "raw_results.csv"
    summary_path = out_dir / "summary.csv"
    paired_raw_path = out_dir / "raw_results_paired.csv"
    paired_stats_path = out_dir / "paired_replicate_stats.csv"
    summary_paired_path = out_dir / "summary_paired.csv"
    paired_deltas_path = out_dir / "summary_paired_deltas.csv"
    per_group_path = out_dir / "per_group_kappa.csv"
    per_group_paired_path = out_dir / "per_group_kappa_paired.csv"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    paired_raw.to_csv(paired_raw_path, index=False)
    paired_stats.to_csv(paired_stats_path, index=False)
    summary_paired.to_csv(summary_paired_path, index=False)
    paired_deltas.to_csv(paired_deltas_path, index=False)
    per_group.to_csv(per_group_path, index=False)
    per_group_paired.to_csv(per_group_paired_path, index=False)

    result_paths: dict[str, str] = {
        "mechanism_spec": str(spec_path),
        "raw_results": str(raw_path),
        "summary": str(summary_path),
        "raw_results_paired": str(paired_raw_path),
        "paired_replicate_stats": str(paired_stats_path),
        "summary_paired": str(summary_paired_path),
        "summary_paired_deltas": str(paired_deltas_path),
        "per_group_kappa": str(per_group_path),
        "per_group_kappa_paired": str(per_group_paired_path),
    }

    if bool(config.runner.build_tables):
        result_paths.update(
            build_paper_tables(
                summary_paired=summary_paired,
                paired_deltas=paired_deltas,
                paired_raw=paired_raw,
                per_group_kappa=per_group,
                out_dir=paper_dir,
            )
        )
        result_paths.update(build_mechanism_figures_from_results_dir(out_dir))
    run_manifest_path = save_json(
        {
            "package": "simulation_mechanism",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "run_timestamp": str(run_timestamp),
            "n_rows": int(raw.shape[0]),
            "n_settings": int(len(config.settings)),
            "repeats": int(config.runner.repeats),
            "methods": list(method_order),
            "group_cols": list(group_cols),
            "result_paths": dict(result_paths),
        },
        out_dir / "run_manifest.json",
    )
    result_paths["run_manifest"] = str(run_manifest_path)
    result_paths.update(
        {
            key: str(value)
            for key, value in snapshot_result_files(
                out_dir,
                result_paths,
                timestamp=run_timestamp,
            ).items()
            if key != "archived_paths"
        }
    )
    return result_paths
