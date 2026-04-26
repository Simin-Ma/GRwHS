from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from simulation_project.src.experiments.runtime import _parallel_rows
from simulation_project.src.utils import FitResult, load_pandas

from .config import dataset_spec_from_dict, force_until_converged_gate
from .dataset import load_prepared_real_dataset, prepare_split, save_prepared_split
from .evaluation import evaluate_method_result
from .fitting import fit_real_data_methods
from .reporting import (
    DEFAULT_REQUIRED_METRICS,
    build_group_selection_frequency,
    build_paired_deltas,
    build_paired_summary,
    build_selection_stability,
    build_summary,
    default_dataset_group_cols,
    write_json_manifest,
)
from .schemas import DatasetSpec, RealDataConfig
from .table_builder import build_paper_tables
from .utils import ensure_dir, prepare_history_run_dir, save_json, write_history_run_index


def _safe_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


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


def _heuristic_p0(spec: DatasetSpec, p: int) -> int:
    if spec.p0_override is not None:
        return max(1, min(int(spec.p0_override), int(p)))
    strategy = str(spec.p0_strategy).strip().lower()
    if strategy == "sqrt_p":
        value = int(np.ceil(np.sqrt(max(int(p), 1))))
    elif strategy == "half_p":
        value = int(np.ceil(max(int(p), 1) / 2.0))
    elif strategy == "quarter_p":
        value = int(np.ceil(max(int(p), 1) / 4.0))
    elif strategy == "log2_p":
        value = int(np.ceil(np.log2(max(int(p), 2))))
    elif strategy == "all":
        value = int(p)
    else:
        raise ValueError(f"Unsupported p0_strategy for dataset '{spec.dataset_id}': {spec.p0_strategy!r}")
    return max(1, min(value, int(p)))


def _heuristic_p0_groups(spec: DatasetSpec, n_groups: int) -> int:
    if spec.p0_groups_override is not None:
        return max(1, min(int(spec.p0_groups_override), int(n_groups)))
    strategy = str(spec.p0_groups_strategy).strip().lower()
    if strategy == "half_groups":
        value = int(np.ceil(max(int(n_groups), 1) / 2.0))
    elif strategy == "sqrt_groups":
        value = int(np.ceil(np.sqrt(max(int(n_groups), 1))))
    elif strategy == "all":
        value = int(n_groups)
    elif strategy == "one":
        value = 1
    else:
        raise ValueError(
            f"Unsupported p0_groups_strategy for dataset '{spec.dataset_id}': {spec.p0_groups_strategy!r}"
        )
    return max(1, min(value, int(n_groups)))


def _task_payloads(config: RealDataConfig) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for dataset in config.datasets:
        dataset_methods = [method for method in config.methods.roster if method in set(dataset.methods)]
        for replicate_id in range(1, int(dataset.repeats) + 1):
            tasks.append(
                {
                    "dataset": dataset.to_dict(),
                    "replicate_id": int(replicate_id),
                    "master_seed": int(config.runner.seed),
                    "methods": list(dataset_methods),
                    "gate": config.convergence_gate.to_dict(),
                    "grrhs_kwargs": dict(config.methods.grrhs_kwargs),
                    "gigg_config": dict(config.methods.gigg_config),
                    "method_jobs": int(config.runner.method_jobs),
                    "save_splits": bool(config.runner.save_splits),
                    "split_dir": str(Path(config.runner.output_dir) / "splits" / dataset.dataset_id / f"rep_{replicate_id:03d}"),
                }
            )
    return tasks


def _gate_from_payload(payload: Mapping[str, Any]):
    from .schemas import ConvergenceGateSpec

    return ConvergenceGateSpec(**dict(payload))


def _run_replicate_task(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    spec = dataset_spec_from_dict(task["dataset"], default_methods=tuple(task.get("methods", [])))
    dataset = load_prepared_real_dataset(spec)
    split = prepare_split(
        dataset,
        replicate_id=int(task["replicate_id"]),
        master_seed=int(task["master_seed"]),
    )
    if bool(task.get("save_splits", False)):
        save_prepared_split(split, Path(str(task["split_dir"])))

    if split.X_train_used is None or split.y_train_used is None:
        raise RuntimeError(f"Prepared split for dataset '{dataset.dataset_id}' is missing model-ready train arrays.")

    p0 = _heuristic_p0(spec, split.X_train_used.shape[1])
    p0_groups = _heuristic_p0_groups(spec, len(split.groups))
    methods = [str(method) for method in task["methods"]]

    try:
        results = fit_real_data_methods(
            split.X_train_used,
            split.y_train_used,
            split.groups,
            task=str(spec.task),
            seed=int(split.seed) + 17,
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

    group_sizes = [int(len(group)) for group in split.groups]
    rows: list[dict[str, Any]] = []
    for method in methods:
        result = results.get(method, _error_fit_result(method, "Missing result from fit_real_data_methods"))
        eval_row = evaluate_method_result(result, split)
        rows.append(
            {
                "dataset_id": str(dataset.dataset_id),
                "dataset_label": str(dataset.label),
                "description": str(spec.description),
                "task": str(spec.task),
                "target_label": str(spec.target_label),
                "target_transform": str(spec.target_transform),
                "covariate_mode": str(spec.covariate_mode),
                "response_standardization": str(spec.response_standardization),
                "notes": str(spec.notes),
                "replicate_id": int(split.replicate_id),
                "seed": int(split.seed),
                "split_hash": str(split.split_hash),
                "sample_count": int(dataset.X.shape[0]),
                "feature_count": int(dataset.X.shape[1]),
                "covariate_count": int(dataset.covariates.shape[1]) if dataset.covariates is not None else 0,
                "group_count": int(len(split.groups)),
                "group_sizes_json": _safe_json(group_sizes),
                "group_labels_json": _safe_json(list(dataset.group_labels)),
                "n_train": int(split.train_idx.size),
                "n_test": int(split.test_idx.size),
                "p0_estimated": int(p0),
                "p0_groups_estimated": int(p0_groups),
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
            }
        )
    return rows


def run_real_data_experiment(config: RealDataConfig) -> dict[str, str]:
    pd = load_pandas()
    config = replace(config, convergence_gate=force_until_converged_gate(config.convergence_gate))
    requested_output_dir = str(config.runner.output_dir)
    history_root, out_dir, run_timestamp = prepare_history_run_dir(requested_output_dir)
    config = replace(config, runner=replace(config.runner, output_dir=str(out_dir)))
    paper_dir = ensure_dir(out_dir / "paper_tables")

    spec_path = save_json(config.to_manifest(), out_dir / "real_data_spec.json")

    dataset_manifest_rows: list[dict[str, Any]] = []
    for dataset_spec in config.datasets:
        prepared = load_prepared_real_dataset(dataset_spec)
        summary = prepared.to_summary()
        summary["notes"] = str(dataset_spec.notes)
        summary["target_label"] = str(dataset_spec.target_label)
        summary["covariate_mode"] = str(dataset_spec.covariate_mode)
        dataset_manifest_rows.append(summary)
        save_json(summary, out_dir / "datasets" / dataset_spec.dataset_id / "dataset_summary.json")

    tasks = _task_payloads(config)
    task_chunks = _parallel_rows(
        tasks,
        _run_replicate_task,
        n_jobs=int(config.runner.n_jobs),
        prefer_process=bool(int(config.runner.n_jobs) > 1),
        process_fallback="thread",
        progress_desc="Real Data Experiment",
    )

    rows: list[dict[str, Any]] = []
    for chunk in task_chunks:
        rows.extend(chunk)

    raw = pd.DataFrame(rows)
    if not raw.empty:
        raw = raw.sort_values(["dataset_id", "replicate_id", "method"], kind="stable").reset_index(drop=True)

    group_cols = default_dataset_group_cols(raw)
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
    selection_stability = build_selection_stability(
        raw,
        group_cols=group_cols,
        required_metric_cols=config.runner.required_metrics_for_pairing,
    )
    group_selection_frequency = build_group_selection_frequency(
        raw,
        group_cols=group_cols,
        required_metric_cols=config.runner.required_metrics_for_pairing,
    )

    raw_path = out_dir / "raw_results.csv"
    summary_path = out_dir / "summary.csv"
    paired_raw_path = out_dir / "raw_results_paired.csv"
    paired_stats_path = out_dir / "paired_replicate_stats.csv"
    summary_paired_path = out_dir / "summary_paired.csv"
    paired_deltas_path = out_dir / "summary_paired_deltas.csv"
    stability_path = out_dir / "selection_stability.csv"
    group_freq_path = out_dir / "group_selection_frequency.csv"
    dataset_catalog_path = out_dir / "dataset_catalog.json"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    paired_raw.to_csv(paired_raw_path, index=False)
    paired_stats.to_csv(paired_stats_path, index=False)
    summary_paired.to_csv(summary_paired_path, index=False)
    paired_deltas.to_csv(paired_deltas_path, index=False)
    selection_stability.to_csv(stability_path, index=False)
    group_selection_frequency.to_csv(group_freq_path, index=False)
    save_json(dataset_manifest_rows, dataset_catalog_path)

    result_paths: dict[str, str] = {
        "real_data_spec": str(spec_path),
        "dataset_catalog": str(dataset_catalog_path),
        "raw_results": str(raw_path),
        "summary": str(summary_path),
        "raw_results_paired": str(paired_raw_path),
        "paired_replicate_stats": str(paired_stats_path),
        "summary_paired": str(summary_paired_path),
        "summary_paired_deltas": str(paired_deltas_path),
        "selection_stability": str(stability_path),
        "group_selection_frequency": str(group_freq_path),
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
        "package": "real_data_experiment",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_timestamp": str(run_timestamp),
        "history_root": str(history_root),
        "requested_output_dir": requested_output_dir,
        "run_dir": str(out_dir),
        "n_rows": int(raw.shape[0]),
        "n_datasets": int(len(config.datasets)),
        "methods": list(config.methods.roster),
        "group_cols": list(group_cols),
        "result_paths": dict(result_paths),
    }
    manifest_path = write_json_manifest(manifest, out_dir / "run_manifest.json")
    result_paths["history_root"] = str(history_root)
    result_paths["requested_output_dir"] = requested_output_dir
    result_paths["run_dir"] = str(out_dir)
    result_paths["run_timestamp"] = str(run_timestamp)
    result_paths["run_manifest"] = str(manifest_path)
    result_paths.update(
        write_history_run_index(
            history_root,
            run_timestamp=run_timestamp,
            run_dir=out_dir,
            result_paths=result_paths,
        )
    )
    return result_paths
