from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from .fitting import _fit_all_methods
from .group_aware_v2_common import summarize_method_row
from .reporting import _finalize_experiment_run, _paired_converged_subset, _record_produced_paths
from .runtime import (
    _BAYESIAN_DEFAULT_CHAINS,
    _gigg_config_default,
    _parallel_rows,
    _resolve_convergence_retry_limit,
    _sampler_for_exp4,
    _sampler_for_standard,
)
from ..utils import (
    MASTER_SEED,
    canonical_groups,
    ensure_dir,
    experiment_seed,
    load_pandas,
    method_result_label,
    print_experiment_result,
    sample_correlated_design,
    save_fit_result_artifacts,
    save_dataframe,
    save_json,
    setup_logger,
)


def _build_complexity_mismatch_beta(
    *,
    group_sizes: Sequence[int],
    pattern: str,
    within_group_pattern: str,
    total_active_coeff: int,
) -> np.ndarray:
    groups = canonical_groups(group_sizes)
    p = int(sum(group_sizes))
    beta = np.zeros(p, dtype=float)
    total_active = max(1, int(total_active_coeff))
    pat = str(pattern).strip().lower()
    within = str(within_group_pattern).strip().lower()

    if pat == "few_groups":
        active_group_ids = [0]
    elif pat == "many_groups":
        # Use many, but not nearly-all, active groups so the contrast against
        # few_groups is meaningful while null-group diagnostics remain stable.
        max_active_groups = max(2, len(groups) - 2)
        active_count = min(max_active_groups, max(2, total_active))
        active_group_ids = list(range(active_count))
    else:
        raise ValueError(f"unknown complexity pattern: {pattern!r}")

    if within == "concentrated":
        remaining = int(total_active)
        for gid in active_group_ids:
            idx = np.asarray(groups[gid], dtype=int)
            if idx.size == 0:
                continue
            take = min(int(idx.size), max(1, remaining if gid == active_group_ids[-1] else int(np.ceil(total_active / max(len(active_group_ids), 1)))))
            beta[idx[:take]] = 1.0
            remaining = max(0, remaining - take)
            if remaining <= 0:
                break
    elif within == "distributed":
        weights = np.zeros(len(groups), dtype=int)
        base = max(1, int(total_active // max(len(active_group_ids), 1)))
        rem = int(total_active)
        for gid in active_group_ids:
            weights[gid] = min(int(len(groups[gid])), base)
            rem -= weights[gid]
        ptr = 0
        while rem > 0 and active_group_ids:
            gid = active_group_ids[ptr % len(active_group_ids)]
            if weights[gid] < len(groups[gid]):
                weights[gid] += 1
                rem -= 1
            ptr += 1
            if ptr > 10000:
                break
        for gid in active_group_ids:
            idx = np.asarray(groups[gid], dtype=int)
            k = int(max(0, min(weights[gid], idx.size)))
            if k > 0:
                beta[idx[:k]] = 1.0 / np.sqrt(k)
    else:
        raise ValueError(f"unknown within_group_pattern: {within!r}")
    return beta

def _ga_v2_complexity_worker(
    task: tuple[
        int,
        int,
        list[int],
        str,
        str,
        int,
        int,
        int,
        float,
        float,
        float,
        Any,
        list[str],
        dict[str, Any],
        int,
        bool,
        int,
        dict[str, Any],
        int,
        str,
    ]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    (
        replicate_id,
        seed,
        group_sizes,
        pattern,
        within_group_pattern,
        total_active_coeff,
        n_train,
        n_test,
        rho_within,
        rho_between,
        target_snr,
        sampler,
        methods,
        gigg_config,
        bayes_min_chains,
        enforce_convergence,
        max_retries,
        grrhs_kwargs,
        method_jobs,
        log_path,
    ) = task
    from .dgp.grouped_linear import generate_grouped_linear_dataset

    s = experiment_seed(62, 1, int(replicate_id), master_seed=int(seed))
    beta_true = _build_complexity_mismatch_beta(
        group_sizes=group_sizes,
        pattern=pattern,
        within_group_pattern=within_group_pattern,
        total_active_coeff=int(total_active_coeff),
    )
    ds = generate_grouped_linear_dataset(
        n=int(n_train),
        group_sizes=group_sizes,
        rho_within=float(rho_within),
        rho_between=float(rho_between),
        beta_shape=beta_true,
        seed=s,
        target_snr=float(target_snr),
        design_type="correlated",
    )
    X_test, cov_x = sample_correlated_design(
        n=int(n_test),
        group_sizes=group_sizes,
        rho_within=float(rho_within),
        rho_between=float(rho_between),
        seed=s + 7171,
    )
    rng = np.random.default_rng(s + 7272)
    sigma2 = float(ds["sigma2"])
    y_test = X_test @ beta_true + rng.normal(0.0, np.sqrt(sigma2), int(n_test))
    group_has_signal = np.array([np.any(np.abs(beta_true[g]) > 1e-10) for g in ds["groups"]], dtype=bool)
    p0_groups = int(np.sum(group_has_signal))

    fits = _fit_all_methods(
        ds["X"],
        ds["y"],
        ds["groups"],
        task="gaussian",
        seed=s,
        p0=int(total_active_coeff),
        p0_groups=p0_groups,
        sampler=sampler,
        grrhs_kwargs=grrhs_kwargs or {},
        methods=methods,
        gigg_config=gigg_config,
        bayes_min_chains=int(bayes_min_chains),
        enforce_bayes_convergence=bool(enforce_convergence),
        max_convergence_retries=int(max_retries),
        method_jobs=int(method_jobs),
    )

    out: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    base_dir = Path(log_path).resolve().parents[1]
    dataset_dir = ensure_dir(base_dir / "results" / "group_aware_v2" / "ga_v2_complexity_mismatch" / "datasets" / f"rep_{int(replicate_id):03d}")
    dataset_arrays_path = dataset_dir / "dataset_arrays.npz"
    dataset_metadata_path = dataset_dir / "dataset_metadata.json"
    np.savez_compressed(
        dataset_arrays_path,
        X_train=np.asarray(ds["X"], dtype=float),
        y_train=np.asarray(ds["y"], dtype=float),
        X_test=np.asarray(X_test, dtype=float),
        y_test=np.asarray(y_test, dtype=float),
        beta_true=np.asarray(beta_true, dtype=float),
        cov_x=np.asarray(cov_x, dtype=float),
    )
    save_json(
        {
            "suite": "group_aware_v2",
            "experiment_family": "ga_v2_complexity_mismatch",
            "replicate_id": int(replicate_id),
            "seed": int(s),
            "group_sizes": [int(v) for v in group_sizes],
            "groups": [[int(idx) for idx in group] for group in ds["groups"]],
            "pattern": str(pattern),
            "within_group_pattern": str(within_group_pattern),
            "total_active_coeff": int(total_active_coeff),
            "rho_within": float(rho_within),
            "rho_between": float(rho_between),
            "target_snr": float(target_snr),
            "sigma2": float(sigma2),
            "n_train": int(n_train),
            "n_test": int(n_test),
        },
        dataset_metadata_path,
    )
    artifact_rows.append(
        {
            "artifact_type": "dataset",
            "experiment_family": "ga_v2_complexity_mismatch",
            "replicate_id": int(replicate_id),
            "seed": int(s),
            "dataset_arrays": str(dataset_arrays_path),
            "dataset_metadata": str(dataset_metadata_path),
        }
    )
    for method, res in fits.items():
        row = summarize_method_row(
            result=res,
            method=method,
            beta_true=beta_true,
            groups=ds["groups"],
            X_train=ds["X"],
            y_train=ds["y"],
            X_test=X_test,
            y_test=y_test,
            group_has_signal=group_has_signal,
        )
        row.update(
            {
                "replicate_id": int(replicate_id),
                "suite": "group_aware_v2",
                "experiment_family": "ga_v2_complexity_mismatch",
                "complexity_pattern": str(pattern),
                "within_group_pattern": str(within_group_pattern),
                "total_active_coeff": int(total_active_coeff),
                "true_active_groups": int(p0_groups),
                "rho_within": float(rho_within),
                "rho_between": float(rho_between),
                "n_train": int(n_train),
                "n_test": int(n_test),
                "target_snr": float(target_snr),
                "sigma2_true": float(sigma2),
                "group_sizes": "|".join(str(int(v)) for v in group_sizes),
            }
        )
        fit_artifacts = save_fit_result_artifacts(
            dataset_dir.parent.parent / "fit_details" / f"rep_{int(replicate_id):03d}" / str(method),
            result=res,
            run_context={
                "suite": "group_aware_v2",
                "experiment_family": "ga_v2_complexity_mismatch",
                "replicate_id": int(replicate_id),
                "seed": int(s),
                "method": str(method),
                "pattern": str(pattern),
                "within_group_pattern": str(within_group_pattern),
                "target_snr": float(target_snr),
            },
            coefficient_truth=beta_true,
            extra_json={
                "metrics": row,
                "dataset_context": {
                    "group_sizes": [int(v) for v in group_sizes],
                    "groups": [[int(idx) for idx in group] for group in ds["groups"]],
                    "total_active_coeff": int(total_active_coeff),
                },
            },
        )
        row["fit_artifact_dir"] = str(fit_artifacts.get("fit_dir", ""))
        row["dataset_arrays_path"] = str(dataset_arrays_path)
        row["dataset_metadata_path"] = str(dataset_metadata_path)
        artifact_rows.append(
            {
                "artifact_type": "fit_result",
                "experiment_family": "ga_v2_complexity_mismatch",
                "replicate_id": int(replicate_id),
                "seed": int(s),
                "method": str(method),
                **fit_artifacts,
            }
        )
        print_experiment_result(
            "GA-V2-B",
            row,
            context_keys=["replicate_id", "complexity_pattern", "within_group_pattern", "method"],
            metric_keys=["group_auroc", "kappa_gap", "mse_overall"],
            log_path=log_path,
        )
        out.append(row)
    return out, artifact_rows


def run_ga_v2_complexity_mismatch(
    n_jobs: int = 1,
    method_jobs: int = 1,
    seed: int = MASTER_SEED,
    repeats: int = 40,
    save_dir: str = "outputs/simulation_project",
    *,
    skip_run_analysis: bool = False,
    archive_artifacts: bool = True,
    bayes_min_chains: int | None = None,
    methods: Sequence[str] | None = None,
    enforce_bayes_convergence: bool = True,
    max_convergence_retries: int | None = None,
    until_bayes_converged: bool = True,
    group_sizes: Sequence[int] | None = None,
    patterns: Sequence[str] | None = None,
    within_group_patterns: Sequence[str] | None = None,
    total_active_coeff: int = 10,
    n_train: int = 100,
    n_test: int = 30,
    rho_within: float = 0.8,
    rho_between: float = 0.2,
    target_snr: float = 1.0,
) -> Dict[str, str]:
    """
    Group-aware validation suite V2, Experiment B.

    Hold total active coefficient count fixed while changing how activity is
    distributed across groups.
    """
    pd = load_pandas()
    produced: set[Path] = set()

    base = Path(save_dir)
    out_dir = ensure_dir(base / "results" / "group_aware_v2" / "ga_v2_complexity_mismatch")
    tab_dir = ensure_dir(base / "tables" / "group_aware_v2")
    log = setup_logger("ga_v2_complexity_mismatch", base / "logs" / "ga_v2_complexity_mismatch.log")
    log_path = str(base / "logs" / "ga_v2_complexity_mismatch.log")

    sampler = _sampler_for_exp4(_sampler_for_standard(experiment="ga_v2_complexity_mismatch"))
    bayes_min_chains_use = int(bayes_min_chains) if bayes_min_chains is not None else int(_BAYESIAN_DEFAULT_CHAINS)
    bayes_min_chains_use = max(1, int(bayes_min_chains_use))
    retry_limit = _resolve_convergence_retry_limit(max_convergence_retries, until_bayes_converged=bool(until_bayes_converged))
    methods_use = [str(m) for m in (methods or ["GR_RHS", "RHS"])]
    group_sizes_use = [int(v) for v in (group_sizes or [10, 10, 10, 10, 10])]
    patterns_use = [str(v) for v in (patterns or ["few_groups", "many_groups"])]
    within_use = [str(v) for v in (within_group_patterns or ["concentrated", "distributed"])]

    grrhs_kwargs = {"tau_target": "groups", "progress_bar": False}
    tasks: list[tuple[Any, ...]] = []
    rid = 0
    for pattern in patterns_use:
        for within in within_use:
            for _ in range(int(repeats)):
                rid += 1
                tasks.append(
                    (
                        int(rid),
                        int(seed),
                        group_sizes_use,
                        str(pattern),
                        str(within),
                        int(total_active_coeff),
                        int(n_train),
                        int(n_test),
                        float(rho_within),
                        float(rho_between),
                        float(target_snr),
                        sampler,
                        methods_use,
                        _gigg_config_default(),
                        int(bayes_min_chains_use),
                        bool(enforce_bayes_convergence),
                        int(retry_limit),
                        grrhs_kwargs,
                        int(method_jobs),
                        log_path,
                    )
                )

    rows_nested = _parallel_rows(
        tasks,
        _ga_v2_complexity_worker,
        n_jobs=n_jobs,
        prefer_process=True,
        process_fallback="serial",
        progress_desc="GA-V2 Complexity Mismatch",
    )
    rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    for row_chunk, artifact_chunk in rows_nested:
        rows.extend(row_chunk)
        artifact_rows.extend(artifact_chunk)

    raw = pd.DataFrame(rows)
    if not raw.empty and "method" in raw.columns:
        raw["method_label"] = raw["method"].map(method_result_label)

    paired_raw, paired_stats = _paired_converged_subset(
        raw,
        group_cols=["complexity_pattern", "within_group_pattern"],
        method_col="method",
        replicate_col="replicate_id",
        converged_col="converged",
        required_cols=["group_auroc", "mse_overall"],
        method_levels=methods_use,
    )
    summary = paired_raw.groupby(["complexity_pattern", "within_group_pattern", "method"], as_index=False).agg(
        group_auroc=("group_auroc", "mean"),
        kappa_gap=("kappa_gap", "mean"),
        kappa_null_mean=("kappa_null_mean", "mean"),
        kappa_signal_mean=("kappa_signal_mean", "mean"),
        null_group_mse=("null_group_mse", "mean"),
        signal_group_mse=("signal_group_mse", "mean"),
        mse_null=("mse_null", "mean"),
        mse_signal=("mse_signal", "mean"),
        mse_overall=("mse_overall", "mean"),
        lpd_test=("lpd_test", "mean"),
        runtime_seconds=("runtime_seconds", "median"),
        true_active_groups=("true_active_groups", "mean"),
        n_effective=("converged", "sum"),
    )
    if not summary.empty:
        summary["method_label"] = summary["method"].map(method_result_label)

    save_dataframe(raw, out_dir / "raw_results.csv")
    save_dataframe(paired_raw, out_dir / "summary_paired_raw.csv")
    save_dataframe(summary, out_dir / "summary.csv")
    save_dataframe(summary, out_dir / "summary_paired.csv")
    save_dataframe(paired_stats, out_dir / "paired_stats.csv")
    save_dataframe(summary, tab_dir / "ga_v2_complexity_mismatch_summary.csv")
    save_json(artifact_rows, out_dir / "artifact_catalog.json")
    save_json(
        {
            "suite": "group_aware_v2",
            "experiment": "ga_v2_complexity_mismatch",
            "patterns": patterns_use,
            "within_group_patterns": within_use,
            "group_sizes": group_sizes_use,
            "total_active_coeff": int(total_active_coeff),
            "n_train": int(n_train),
            "n_test": int(n_test),
            "rho_within": float(rho_within),
            "rho_between": float(rho_between),
            "target_snr": float(target_snr),
            "methods": methods_use,
            "repeats_per_cell": int(repeats),
        },
        out_dir / "meta.json",
    )
    _record_produced_paths(
        produced,
        out_dir / "raw_results.csv",
        out_dir / "summary_paired_raw.csv",
        out_dir / "summary.csv",
        out_dir / "summary_paired.csv",
        out_dir / "paired_stats.csv",
        out_dir / "artifact_catalog.json",
        out_dir / "meta.json",
        tab_dir / "ga_v2_complexity_mismatch_summary.csv",
    )

    result_paths = {
        "raw": str(out_dir / "raw_results.csv"),
        "summary_paired_raw": str(out_dir / "summary_paired_raw.csv"),
        "summary": str(out_dir / "summary.csv"),
        "summary_paired": str(out_dir / "summary_paired.csv"),
        "paired_stats": str(out_dir / "paired_stats.csv"),
        "artifact_catalog": str(out_dir / "artifact_catalog.json"),
        "fit_details_dir": str(out_dir / "fit_details"),
        "datasets_dir": str(out_dir / "datasets"),
        "table": str(tab_dir / "ga_v2_complexity_mismatch_summary.csv"),
        "meta": str(out_dir / "meta.json"),
    }
    return _finalize_experiment_run(
        exp_key="ga_v2_complexity_mismatch",
        save_dir=save_dir,
        results_dir=out_dir,
        produced_paths=produced,
        result_paths=result_paths,
        skip_run_analysis=bool(skip_run_analysis),
        archive_artifacts=bool(archive_artifacts),
    )
