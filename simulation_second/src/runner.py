from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from simulation_project.src.experiments.runtime import _parallel_rows
from simulation_project.src.utils import FitResult, append_jsonl_records, load_pandas, save_fit_result_artifacts

from .config import (
    BenchmarkConfig,
    family_spec_from_dict,
    force_until_converged_gate,
    setting_spec_from_dict,
)
from .dataset import generate_grouped_dataset, save_grouped_dataset
from .evaluation import evaluate_method_result
from .fitting import fit_benchmark_methods
from .plotting import build_benchmark_figures_from_results_dir
from .reporting import (
    DEFAULT_REQUIRED_METRICS,
    build_paired_deltas,
    build_paired_summary,
    build_summary,
    default_setting_group_cols,
    write_json_manifest,
)
from .table_builder import build_paper_tables
from .utils import ensure_dir, prepare_history_run_dir, save_json, write_history_run_index


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


def _coefficient_recovery_rows(
    *,
    dataset,
    result: FitResult,
    method: str,
    method_label: str,
    method_type: str,
    method_order: int,
    base_fields: Mapping[str, Any],
) -> list[dict[str, Any]]:
    beta_true = np.asarray(dataset.beta, dtype=float).reshape(-1)
    p = int(beta_true.shape[0])
    beta_est = np.full(p, np.nan, dtype=float)
    if result.beta_mean is not None:
        beta_mean = np.asarray(result.beta_mean, dtype=float).reshape(-1)
        take = min(p, int(beta_mean.shape[0]))
        if take > 0:
            beta_est[:take] = beta_mean[:take]

    groups = dataset.groups
    group_id = np.full(p, -1, dtype=int)
    within_group_index = np.full(p, -1, dtype=int)
    group_size = np.zeros(p, dtype=int)
    group_active = np.zeros(len(groups), dtype=bool)
    for gid, members in enumerate(groups):
        idx = np.asarray(members, dtype=int)
        if idx.size == 0:
            continue
        group_id[idx] = int(gid)
        within_group_index[idx] = np.arange(idx.size, dtype=int)
        group_size[idx] = int(idx.size)
        group_active[gid] = bool(np.any(np.abs(beta_true[idx]) > 1e-12))

    rows: list[dict[str, Any]] = []
    for coef_idx in range(p):
        true_beta = float(beta_true[coef_idx])
        est_beta = float(beta_est[coef_idx]) if np.isfinite(beta_est[coef_idx]) else float("nan")
        error = float(est_beta - true_beta) if np.isfinite(est_beta) else float("nan")
        sq_error = float(error ** 2) if np.isfinite(error) else float("nan")
        abs_error = float(abs(error)) if np.isfinite(error) else float("nan")
        active_coef = bool(abs(true_beta) > 1e-12)
        gid = int(group_id[coef_idx])
        sign_match_active = float("nan")
        if active_coef and np.isfinite(est_beta):
            sign_match_active = float(np.sign(est_beta) == np.sign(true_beta))
        rows.append(
            {
                **dict(base_fields),
                "method": str(method),
                "method_label": str(method_label),
                "method_type": str(method_type),
                "method_order": int(method_order),
                "status": str(result.status),
                "converged": bool(result.converged),
                "coefficient_index": int(coef_idx),
                "group_id": gid,
                "group_size": int(group_size[coef_idx]),
                "within_group_index": int(within_group_index[coef_idx]),
                "is_active_group": bool(group_active[gid]) if 0 <= gid < len(group_active) else False,
                "is_active_coefficient": active_coef,
                "true_beta": true_beta,
                "estimated_beta": est_beta,
                "abs_true_beta": float(abs(true_beta)),
                "abs_estimated_beta": float(abs(est_beta)) if np.isfinite(est_beta) else float("nan"),
                "error": error,
                "sq_error": sq_error,
                "abs_error": abs_error,
                "sign_match_active": sign_match_active,
            }
        )
    return rows


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
                    "save_datasets": True,
                    "dataset_dir": str(Path(config.runner.output_dir) / "datasets"),
                    "fit_detail_dir": str(Path(config.runner.output_dir) / "fit_details"),
                }
            )
    return tasks


def _run_replicate_task(task: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
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
    dataset_artifacts: dict[str, str] = {}
    if bool(task.get("save_datasets", True)):
        dataset_artifacts = save_grouped_dataset(dataset, Path(str(task["dataset_dir"])) / setting.setting_id)

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
    coefficient_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    if dataset_artifacts:
        artifact_rows.append(
            {
                "artifact_type": "dataset",
                "setting_id": setting.setting_id,
                "replicate_id": int(dataset.metadata["replicate_id"]),
                "seed": int(dataset.metadata["seed"]),
                **dataset_artifacts,
            }
        )
    for method_order, method in enumerate(methods):
        result = results.get(method, _error_fit_result(method, "Missing result from fit_benchmark_methods"))
        eval_row = evaluate_method_result(result, dataset)
        row_base = {
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
        }
        fit_artifacts = save_fit_result_artifacts(
            Path(str(task["fit_detail_dir"])) / setting.setting_id / f"rep_{int(dataset.metadata['replicate_id']):03d}" / str(method),
            result=result,
            run_context={
                **row_base,
                "method": str(method),
                "method_order": int(method_order),
            },
            coefficient_truth=dataset.beta,
            extra_json={
                "evaluation": eval_row,
                "setting": setting_meta,
                "signal_draw": signal_meta,
            },
        )
        artifact_rows.append(
            {
                "artifact_type": "fit_result",
                "setting_id": setting.setting_id,
                "replicate_id": int(dataset.metadata["replicate_id"]),
                "method": str(method),
                "seed": int(dataset.metadata["seed"]),
                **fit_artifacts,
            }
        )
        out_rows.append(
            {
                **row_base,
                "method": str(method),
                "method_label": eval_row["method_label"],
                "method_type": eval_row["method_type"],
                "fit_artifact_dir": str(fit_artifacts.get("fit_dir", "")),
                "dataset_arrays_path": str(dataset_artifacts.get("arrays", "")),
                "dataset_metadata_path": str(dataset_artifacts.get("metadata", "")),
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
        coefficient_rows.extend(
            _coefficient_recovery_rows(
                dataset=dataset,
                result=result,
                method=method,
                method_label=str(eval_row["method_label"]),
                method_type=str(eval_row["method_type"]),
                method_order=int(method_order),
                base_fields=row_base,
            )
        )
    return {
        "raw_rows": out_rows,
        "coefficient_rows": coefficient_rows,
        "artifact_rows": artifact_rows,
    }


def _format_progress_metric(value: Any, *, ndigits: int = 6) -> str:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return "na"
    if not np.isfinite(value_float):
        return "na"
    return f"{value_float:.{ndigits}f}"


def _single_line_error(value: Any, *, limit: int = 80) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= int(limit):
        return text
    return text[: max(0, int(limit) - 3)] + "..."


def _print_task_result_line(
    task: Mapping[str, Any],
    result_rows: Sequence[Mapping[str, Any]] | None,
    *,
    completed: int,
    total: int,
) -> None:
    rows = [dict(row) for row in (result_rows or [])]
    setting_payload = dict(task.get("setting", {}))
    setting_id = str(setting_payload.get("setting_id", "?"))
    replicate_id = int(task.get("replicate_id", -1))
    method_order = {str(name): idx for idx, name in enumerate(task.get("methods", []))}

    if rows:
        setting_id = str(rows[0].get("setting_id", setting_id))
        replicate_id = int(rows[0].get("replicate_id", replicate_id))
        rows = sorted(
            rows,
            key=lambda row: (
                method_order.get(str(row.get("method", "")), len(method_order)),
                str(row.get("method", "")),
            ),
        )

    ok_count = sum(str(row.get("status", "")).lower() == "ok" for row in rows)
    converged_count = sum(bool(row.get("converged", False)) for row in rows)

    best_row = None
    best_mse = float("inf")
    for row in rows:
        try:
            mse_value = float(row.get("mse_overall", float("nan")))
        except (TypeError, ValueError):
            continue
        if np.isfinite(mse_value) and mse_value < best_mse:
            best_mse = mse_value
            best_row = row

    best_text = "best=na"
    if best_row is not None:
        best_text = f"best={best_row.get('method', '?')}:{_format_progress_metric(best_mse)}"

    method_parts: list[str] = []
    for row in rows:
        method = str(row.get("method", "?"))
        status = str(row.get("status", "?"))
        conv = "conv" if bool(row.get("converged", False)) else "no-conv"
        mse_text = _format_progress_metric(row.get("mse_overall"))
        lpd_text = _format_progress_metric(row.get("lpd_test"))
        runtime_text = _format_progress_metric(row.get("runtime_seconds"), ndigits=2)
        part = f"{method}[{status},{conv},mse={mse_text},lpd={lpd_text},rt={runtime_text}s"
        if str(status).lower() != "ok":
            error_text = _single_line_error(row.get("error", ""))
            if error_text:
                part += f",err={error_text}"
        part += "]"
        method_parts.append(part)

    methods_text = " | ".join(method_parts) if method_parts else "no rows returned"
    line = (
        f"[task {int(completed)}/{int(total)}] "
        f"setting={setting_id} rep={replicate_id} "
        f"ok={ok_count}/{len(rows)} conv={converged_count}/{len(rows)} {best_text} :: {methods_text}"
    )
    print(line, flush=True)


def run_benchmark(config: BenchmarkConfig) -> dict[str, str]:
    pd = load_pandas()
    config = replace(config, convergence_gate=force_until_converged_gate(config.convergence_gate))
    requested_output_dir = str(config.runner.output_dir)
    history_root, out_dir, run_timestamp = prepare_history_run_dir(requested_output_dir)
    config = replace(config, runner=replace(config.runner, output_dir=str(out_dir), save_datasets=True))
    paper_dir = ensure_dir(out_dir / "paper_tables")

    spec_path = save_json(config.to_manifest(), out_dir / "benchmark_spec.json")
    incremental_raw_path = out_dir / "raw_results_incremental.jsonl"
    incremental_coefficients_path = out_dir / "coefficient_estimates_incremental.jsonl"
    incremental_artifacts_path = out_dir / "artifact_catalog_incremental.jsonl"
    for checkpoint_path in (
        incremental_raw_path,
        incremental_coefficients_path,
        incremental_artifacts_path,
    ):
        ensure_dir(checkpoint_path.parent)
        checkpoint_path.write_text("", encoding="utf-8")

    tasks = _task_payloads(config)
    task_progress = {"completed": 0, "total": len(tasks)}

    def _on_task_done(task: Mapping[str, Any], result_rows: Any) -> None:
        task_progress["completed"] += 1
        if isinstance(result_rows, Mapping):
            append_jsonl_records(
                incremental_raw_path,
                [dict(row) for row in result_rows.get("raw_rows", [])],
            )
            append_jsonl_records(
                incremental_coefficients_path,
                [dict(row) for row in result_rows.get("coefficient_rows", [])],
            )
            append_jsonl_records(
                incremental_artifacts_path,
                [dict(row) for row in result_rows.get("artifact_rows", [])],
            )
        _print_task_result_line(
            task,
            (result_rows or {}).get("raw_rows") if isinstance(result_rows, Mapping) else result_rows,
            completed=int(task_progress["completed"]),
            total=int(task_progress["total"]),
        )

    task_chunks = _parallel_rows(
        tasks,
        _run_replicate_task,
        n_jobs=int(config.runner.n_jobs),
        prefer_process=bool(int(config.runner.n_jobs) > 1),
        process_fallback="thread",
        progress_desc="Simulation Second Benchmark",
        on_task_done=_on_task_done,
    )

    rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    for chunk in task_chunks:
        if isinstance(chunk, Mapping):
            rows.extend(list(chunk.get("raw_rows", [])))
            coefficient_rows.extend(list(chunk.get("coefficient_rows", [])))
            artifact_rows.extend(list(chunk.get("artifact_rows", [])))
        else:
            rows.extend(chunk)

    raw = pd.DataFrame(rows)
    coefficient_estimates = pd.DataFrame(coefficient_rows)
    if not raw.empty:
        raw = raw.sort_values(["setting_id", "replicate_id", "method"], kind="stable").reset_index(drop=True)
    if not coefficient_estimates.empty:
        coefficient_estimates = coefficient_estimates.sort_values(
            ["setting_id", "replicate_id", "method_order", "method", "coefficient_index"],
            kind="stable",
        ).reset_index(drop=True)

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

    if coefficient_estimates.empty:
        coefficient_estimates["paired_common_converged"] = pd.Series(dtype=bool)
        coefficient_estimates_paired = coefficient_estimates.copy()
    else:
        pair_keys = paired_raw.loc[
            :,
            [col for col in ["setting_id", "replicate_id", "method"] if col in paired_raw.columns],
        ].drop_duplicates()
        if not pair_keys.empty:
            pair_keys = pair_keys.assign(paired_common_converged=True)
            coefficient_estimates = coefficient_estimates.merge(
                pair_keys,
                on=[col for col in ["setting_id", "replicate_id", "method"] if col in pair_keys.columns],
                how="left",
            )
            coefficient_estimates["paired_common_converged"] = (
                coefficient_estimates["paired_common_converged"].fillna(False).astype(bool)
            )
        else:
            coefficient_estimates["paired_common_converged"] = False
        coefficient_estimates_paired = coefficient_estimates.loc[
            coefficient_estimates["paired_common_converged"].fillna(False).astype(bool)
        ].copy()

    raw_path = out_dir / "raw_results.csv"
    summary_path = out_dir / "summary.csv"
    paired_raw_path = out_dir / "raw_results_paired.csv"
    paired_stats_path = out_dir / "paired_replicate_stats.csv"
    summary_paired_path = out_dir / "summary_paired.csv"
    paired_deltas_path = out_dir / "summary_paired_deltas.csv"
    coefficient_estimates_path = out_dir / "coefficient_estimates.csv"
    coefficient_estimates_paired_path = out_dir / "coefficient_estimates_paired.csv"
    artifact_catalog_path = out_dir / "artifact_catalog.json"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    paired_raw.to_csv(paired_raw_path, index=False)
    paired_stats.to_csv(paired_stats_path, index=False)
    summary_paired.to_csv(summary_paired_path, index=False)
    paired_deltas.to_csv(paired_deltas_path, index=False)
    coefficient_estimates.to_csv(coefficient_estimates_path, index=False)
    coefficient_estimates_paired.to_csv(coefficient_estimates_paired_path, index=False)
    save_json(artifact_rows, artifact_catalog_path)

    result_paths: dict[str, str] = {
        "benchmark_spec": str(spec_path),
        "raw_results": str(raw_path),
        "summary": str(summary_path),
        "raw_results_paired": str(paired_raw_path),
        "paired_replicate_stats": str(paired_stats_path),
        "summary_paired": str(summary_paired_path),
        "summary_paired_deltas": str(paired_deltas_path),
        "coefficient_estimates": str(coefficient_estimates_path),
        "coefficient_estimates_paired": str(coefficient_estimates_paired_path),
        "artifact_catalog": str(artifact_catalog_path),
        "raw_results_incremental": str(incremental_raw_path),
        "coefficient_estimates_incremental": str(incremental_coefficients_path),
        "artifact_catalog_incremental": str(incremental_artifacts_path),
        "fit_details_dir": str(out_dir / "fit_details"),
        "datasets_dir": str(out_dir / "datasets"),
    }

    if bool(config.runner.build_tables):
        result_paths.update(
            build_paper_tables(
                raw,
                out_dir=paper_dir,
                method_order=config.methods.roster,
                group_cols=group_cols,
                required_metric_cols=config.runner.required_metrics_for_pairing,
                coefficient_estimates=coefficient_estimates,
            )
        )
        result_paths.update(build_benchmark_figures_from_results_dir(out_dir))

    manifest = {
        "package": "simulation_second",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_timestamp": str(run_timestamp),
        "history_root": str(history_root),
        "requested_output_dir": requested_output_dir,
        "run_dir": str(out_dir),
        "n_rows": int(raw.shape[0]),
        "n_settings": int(len(config.settings)),
        "repeats": int(config.runner.repeats),
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
