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
    _sampler_for_standard,
)
from ..utils import (
    MASTER_SEED,
    ensure_dir,
    experiment_seed,
    load_pandas,
    method_result_label,
    print_experiment_result,
    save_fit_result_artifacts,
    save_dataframe,
    save_json,
    sample_correlated_design,
    setup_logger,
)


def _ga_v2_group_sep_worker(
    task: tuple[
        int,
        int,
        list[int],
        list[float],
        float,
        float,
        float,
        int,
        int,
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
    from .dgp.grouped_linear import generate_heterogeneity_dataset

    (
        replicate_id,
        seed,
        group_sizes,
        mu,
        rho_within,
        rho_between,
        sigma2,
        n_train,
        n_test,
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
    labels = (np.asarray(mu, dtype=float) > 0.0).astype(bool)
    p0_signal_groups = int(np.sum(labels))
    s = experiment_seed(61, 1, int(replicate_id), master_seed=int(seed))

    ds = generate_heterogeneity_dataset(
        n=int(n_train),
        group_sizes=group_sizes,
        rho_within=float(rho_within),
        rho_between=float(rho_between),
        sigma2=float(sigma2),
        mu=mu,
        seed=s,
    )
    X_test, _ = sample_correlated_design(
        n=int(n_test),
        group_sizes=group_sizes,
        rho_within=float(rho_within),
        rho_between=float(rho_between),
        seed=s + 9191,
    )
    rng = np.random.default_rng(s + 9292)
    y_test = X_test @ ds["beta0"] + rng.normal(0.0, np.sqrt(float(sigma2)), int(n_test))

    fits = _fit_all_methods(
        ds["X"],
        ds["y"],
        ds["groups"],
        task="gaussian",
        seed=s,
        p0=p0_signal_groups,
        p0_groups=p0_signal_groups,
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
    dataset_dir = ensure_dir(base_dir / "results" / "group_aware_v2" / "ga_v2_group_separation" / "datasets" / f"rep_{int(replicate_id):03d}")
    dataset_arrays_path = dataset_dir / "dataset_arrays.npz"
    dataset_metadata_path = dataset_dir / "dataset_metadata.json"
    np.savez_compressed(
        dataset_arrays_path,
        X_train=np.asarray(ds["X"], dtype=float),
        y_train=np.asarray(ds["y"], dtype=float),
        X_test=np.asarray(X_test, dtype=float),
        y_test=np.asarray(y_test, dtype=float),
        beta_true=np.asarray(ds["beta0"], dtype=float),
        cov_x=np.asarray(ds["cov_x"], dtype=float),
    )
    save_json(
        {
            "suite": "group_aware_v2",
            "experiment_family": "ga_v2_group_separation",
            "replicate_id": int(replicate_id),
            "seed": int(s),
            "group_sizes": [int(v) for v in group_sizes],
            "groups": [[int(idx) for idx in group] for group in ds["groups"]],
            "mu": [float(v) for v in np.asarray(ds["mu"], dtype=float).tolist()],
            "rho_within": float(rho_within),
            "rho_between": float(rho_between),
            "sigma2": float(sigma2),
            "n_train": int(n_train),
            "n_test": int(n_test),
        },
        dataset_metadata_path,
    )
    artifact_rows.append(
        {
            "artifact_type": "dataset",
            "experiment_family": "ga_v2_group_separation",
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
            beta_true=ds["beta0"],
            groups=ds["groups"],
            X_train=ds["X"],
            y_train=ds["y"],
            X_test=X_test,
            y_test=y_test,
            group_has_signal=labels,
        )
        row.update(
            {
                "replicate_id": int(replicate_id),
                "suite": "group_aware_v2",
                "experiment_family": "ga_v2_group_separation",
                "rho_within": float(rho_within),
                "rho_between": float(rho_between),
                "n_train": int(n_train),
                "n_test": int(n_test),
                "group_sizes": "|".join(str(int(v)) for v in group_sizes),
            }
        )
        fit_artifacts = save_fit_result_artifacts(
            dataset_dir.parent.parent / "fit_details" / f"rep_{int(replicate_id):03d}" / str(method),
            result=res,
            run_context={
                "suite": "group_aware_v2",
                "experiment_family": "ga_v2_group_separation",
                "replicate_id": int(replicate_id),
                "seed": int(s),
                "method": str(method),
                "rho_within": float(rho_within),
                "rho_between": float(rho_between),
                "sigma2": float(sigma2),
            },
            coefficient_truth=ds["beta0"],
            extra_json={
                "metrics": row,
                "dataset_context": {
                    "group_sizes": [int(v) for v in group_sizes],
                    "mu": [float(v) for v in np.asarray(ds["mu"], dtype=float).tolist()],
                    "groups": [[int(idx) for idx in group] for group in ds["groups"]],
                },
            },
        )
        row["fit_artifact_dir"] = str(fit_artifacts.get("fit_dir", ""))
        row["dataset_arrays_path"] = str(dataset_arrays_path)
        row["dataset_metadata_path"] = str(dataset_metadata_path)
        artifact_rows.append(
            {
                "artifact_type": "fit_result",
                "experiment_family": "ga_v2_group_separation",
                "replicate_id": int(replicate_id),
                "seed": int(s),
                "method": str(method),
                **fit_artifacts,
            }
        )
        print_experiment_result(
            "GA-V2-A",
            row,
            context_keys=["replicate_id", "method"],
            metric_keys=["group_auroc", "kappa_gap", "mse_overall"],
            log_path=log_path,
        )
        out.append(row)
    return out, artifact_rows


def run_ga_v2_group_separation(
    n_jobs: int = 1,
    method_jobs: int = 1,
    seed: int = MASTER_SEED,
    repeats: int = 100,
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
    mu: Sequence[float] | None = None,
    n_train: int = 100,
    n_test: int = 30,
    rho_within: float = 0.8,
    rho_between: float = 0.2,
    sigma2: float = 1.0,
) -> Dict[str, str]:
    """
    Group-aware validation suite V2, Experiment A.

    A clean group-separation experiment focused on mechanism metrics rather than
    only global coefficient error.
    """
    pd = load_pandas()
    produced: set[Path] = set()

    base = Path(save_dir)
    out_dir = ensure_dir(base / "results" / "group_aware_v2" / "ga_v2_group_separation")
    tab_dir = ensure_dir(base / "tables" / "group_aware_v2")
    log = setup_logger("ga_v2_group_separation", base / "logs" / "ga_v2_group_separation.log")
    log_path = str(base / "logs" / "ga_v2_group_separation.log")

    sampler = _sampler_for_standard(experiment="ga_v2_group_separation")
    bayes_min_chains_use = int(bayes_min_chains) if bayes_min_chains is not None else int(_BAYESIAN_DEFAULT_CHAINS)
    bayes_min_chains_use = max(1, int(bayes_min_chains_use))
    retry_limit = _resolve_convergence_retry_limit(max_convergence_retries, until_bayes_converged=bool(until_bayes_converged))
    methods_use = [str(m) for m in (methods or ["GR_RHS", "RHS"])]
    group_sizes_use = [int(v) for v in (group_sizes or [10, 10, 10, 10, 10])]
    mu_use = [float(v) for v in (mu or [0.0, 0.0, 1.5, 4.0, 10.0])]
    if len(group_sizes_use) != len(mu_use):
        raise ValueError("group_sizes and mu must have the same length.")

    log.info(
        "GA-V2-A: repeats=%d, rho_within=%.3f, rho_between=%.3f, n_train=%d, n_test=%d",
        int(repeats),
        float(rho_within),
        float(rho_between),
        int(n_train),
        int(n_test),
    )

    tasks: list[tuple[Any, ...]] = []
    grrhs_kwargs = {"tau_target": "groups", "progress_bar": False}
    for r in range(1, int(repeats) + 1):
        tasks.append(
            (
                int(r),
                int(seed),
                group_sizes_use,
                mu_use,
                float(rho_within),
                float(rho_between),
                float(sigma2),
                int(n_train),
                int(n_test),
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
        _ga_v2_group_sep_worker,
        n_jobs=n_jobs,
        prefer_process=True,
        process_fallback="serial",
        progress_desc="GA-V2 Group Separation",
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
        group_cols=[],
        method_col="method",
        replicate_col="replicate_id",
        converged_col="converged",
        required_cols=["group_auroc", "mse_overall"],
        method_levels=methods_use,
    )

    summary = paired_raw.groupby("method", as_index=False).agg(
        group_auroc=("group_auroc", "mean"),
        group_auroc_std=("group_auroc", "std"),
        kappa_gap=("kappa_gap", "mean"),
        kappa_gap_std=("kappa_gap", "std"),
        kappa_null_mean=("kappa_null_mean", "mean"),
        kappa_signal_mean=("kappa_signal_mean", "mean"),
        null_group_mse=("null_group_mse", "mean"),
        signal_group_mse=("signal_group_mse", "mean"),
        mse_null=("mse_null", "mean"),
        mse_signal=("mse_signal", "mean"),
        mse_overall=("mse_overall", "mean"),
        lpd_test=("lpd_test", "mean"),
        runtime_seconds=("runtime_seconds", "median"),
        n_effective=("converged", "sum"),
    )
    if not summary.empty:
        summary["method_label"] = summary["method"].map(method_result_label)

    save_dataframe(raw, out_dir / "raw_results.csv")
    save_dataframe(paired_raw, out_dir / "summary_paired_raw.csv")
    save_dataframe(summary, out_dir / "summary.csv")
    save_dataframe(summary, out_dir / "summary_paired.csv")
    save_dataframe(paired_stats, out_dir / "paired_stats.csv")
    save_dataframe(summary, tab_dir / "ga_v2_group_separation_summary.csv")
    save_json(artifact_rows, out_dir / "artifact_catalog.json")
    save_json(
        {
            "suite": "group_aware_v2",
            "experiment": "ga_v2_group_separation",
            "repeats": int(repeats),
            "methods": methods_use,
            "group_sizes": group_sizes_use,
            "mu": mu_use,
            "rho_within": float(rho_within),
            "rho_between": float(rho_between),
            "n_train": int(n_train),
            "n_test": int(n_test),
            "sigma2": float(sigma2),
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
        tab_dir / "ga_v2_group_separation_summary.csv",
    )

    log.info("GA-V2-A done: %d rows", len(rows))
    result_paths = {
        "raw": str(out_dir / "raw_results.csv"),
        "summary_paired_raw": str(out_dir / "summary_paired_raw.csv"),
        "summary": str(out_dir / "summary.csv"),
        "summary_paired": str(out_dir / "summary_paired.csv"),
        "paired_stats": str(out_dir / "paired_stats.csv"),
        "artifact_catalog": str(out_dir / "artifact_catalog.json"),
        "fit_details_dir": str(out_dir / "fit_details"),
        "datasets_dir": str(out_dir / "datasets"),
        "table": str(tab_dir / "ga_v2_group_separation_summary.csv"),
        "meta": str(out_dir / "meta.json"),
    }
    return _finalize_experiment_run(
        exp_key="ga_v2_group_separation",
        save_dir=save_dir,
        results_dir=out_dir,
        produced_paths=produced,
        result_paths=result_paths,
        skip_run_analysis=bool(skip_run_analysis),
        archive_artifacts=bool(archive_artifacts),
    )
