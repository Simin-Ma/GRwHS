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
    block_correlation,
    canonical_groups,
    ensure_dir,
    experiment_seed,
    load_pandas,
    method_result_label,
    nearest_positive_definite,
    print_experiment_result,
    sample_correlated_design,
    save_dataframe,
    save_json,
    standardize_columns,
    setup_logger,
)


def _build_correlation_stress_beta(
    *,
    group_sizes: Sequence[int],
    active_groups: Sequence[int],
    within_group_pattern: str,
) -> np.ndarray:
    groups = canonical_groups(group_sizes)
    beta = np.zeros(int(sum(group_sizes)), dtype=float)
    active = sorted({int(g) for g in active_groups if 0 <= int(g) < len(groups)})
    if not active:
        raise ValueError("active_groups must contain at least one valid group index.")

    pattern = str(within_group_pattern).strip().lower()
    for gid in active:
        idx = np.asarray(groups[gid], dtype=int)
        if idx.size == 0:
            continue
        if pattern == "concentrated":
            beta[idx[0]] = 1.0
        elif pattern == "distributed":
            beta[idx] = 1.0 / np.sqrt(float(idx.size))
        elif pattern == "mixed_decoy":
            beta[idx[0]] = 1.0
            if idx.size > 1:
                beta[idx[1:]] = 0.25
        else:
            raise ValueError(f"unknown within_group_pattern: {within_group_pattern!r}")
    return beta


def _sample_mixed_decoy_design(
    *,
    n: int,
    group_sizes: Sequence[int],
    active_groups: Sequence[int],
    rho_within: float,
    rho_between: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    groups = canonical_groups(group_sizes)
    active = sorted({int(g) for g in active_groups if 0 <= int(g) < len(groups)})
    null_candidates = [gid for gid in range(len(groups)) if gid not in set(active)]
    decoy_group = int(null_candidates[0]) if null_candidates else None

    cov = block_correlation(group_sizes, rho_within=float(rho_within), rho_between=float(rho_between))
    if active and decoy_group is not None:
        primary_idx = np.asarray(groups[int(active[0])], dtype=int)
        decoy_idx = np.asarray(groups[int(decoy_group)], dtype=int)
        if primary_idx.size and decoy_idx.size:
            # Extra latent coupling makes one null group look structurally similar
            # to the primary active group, creating a genuine group-level decoy.
            load = np.zeros(int(sum(group_sizes)), dtype=float)
            load[primary_idx] = 1.0 / np.sqrt(float(primary_idx.size))
            load[decoy_idx] = 0.85 / np.sqrt(float(decoy_idx.size))
            cov = cov + float(max(rho_within - rho_between, 0.05)) * 0.35 * np.outer(load, load)
            cov = nearest_positive_definite(cov)

    rng = np.random.default_rng(int(seed))
    X = rng.multivariate_normal(mean=np.zeros(cov.shape[0]), cov=cov, size=int(n))
    X = standardize_columns(X)
    return X, cov, decoy_group


def _ga_v2_correlation_worker(
    task: tuple[
        int,
        int,
        list[int],
        list[int],
        str,
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
) -> list[dict[str, Any]]:
    (
        replicate_id,
        seed,
        group_sizes,
        active_groups,
        within_group_pattern,
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

    s = experiment_seed(63, 1, int(replicate_id), master_seed=int(seed))
    groups = canonical_groups(group_sizes)
    pattern_name = str(within_group_pattern).strip().lower()
    beta_true = _build_correlation_stress_beta(
        group_sizes=group_sizes,
        active_groups=active_groups,
        within_group_pattern=pattern_name,
    )
    decoy_group: int | None = None
    if pattern_name == "mixed_decoy":
        from .dgp.grouped_linear import sigma2_for_target_snr

        X_train, cov_x, decoy_group = _sample_mixed_decoy_design(
            n=int(n_train),
            group_sizes=group_sizes,
            active_groups=active_groups,
            rho_within=float(rho_within),
            rho_between=float(rho_between),
            seed=s,
        )
        sigma2 = float(sigma2_for_target_snr(beta=beta_true, cov_x=cov_x, target_snr=float(target_snr)))
        rng_train = np.random.default_rng(s + 17)
        y_train = X_train @ beta_true + rng_train.normal(0.0, np.sqrt(sigma2), int(n_train))
        X_test, _, _ = _sample_mixed_decoy_design(
            n=int(n_test),
            group_sizes=group_sizes,
            active_groups=active_groups,
            rho_within=float(rho_within),
            rho_between=float(rho_between),
            seed=s + 6161,
        )
        rng_test = np.random.default_rng(s + 6262)
        y_test = X_test @ beta_true + rng_test.normal(0.0, np.sqrt(sigma2), int(n_test))
        ds = {
            "X": X_train,
            "y": y_train,
            "groups": groups,
            "sigma2": float(sigma2),
        }
    else:
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
        X_test, _ = sample_correlated_design(
            n=int(n_test),
            group_sizes=group_sizes,
            rho_within=float(rho_within),
            rho_between=float(rho_between),
            seed=s + 6161,
        )
        sigma2 = float(ds["sigma2"])
        rng = np.random.default_rng(s + 6262)
        y_test = X_test @ beta_true + rng.normal(0.0, np.sqrt(sigma2), int(n_test))
    group_has_signal = np.array([np.any(np.abs(beta_true[g]) > 1e-10) for g in ds["groups"]], dtype=bool)
    p0 = int(np.sum(np.abs(beta_true) > 1e-10))
    p0_groups = int(np.sum(group_has_signal))

    fits = _fit_all_methods(
        ds["X"],
        ds["y"],
        ds["groups"],
        task="gaussian",
        seed=s,
        p0=p0,
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
                "experiment_family": "ga_v2_correlation_stress",
                "within_group_pattern": str(within_group_pattern),
                "rho_within": float(rho_within),
                "rho_between": float(rho_between),
                "n_train": int(n_train),
                "n_test": int(n_test),
                "target_snr": float(target_snr),
                "sigma2_true": float(sigma2),
                "group_sizes": "|".join(str(int(v)) for v in group_sizes),
                "active_groups": "|".join(str(int(v)) for v in active_groups),
                "true_active_groups": int(p0_groups),
                "true_active_coeff": int(p0),
                "decoy_group": int(decoy_group) if decoy_group is not None else -1,
            }
        )
        print_experiment_result(
            "GA-V2-C",
            row,
            context_keys=["replicate_id", "rho_within", "within_group_pattern", "method"],
            metric_keys=["group_auroc", "kappa_gap", "mse_overall"],
            log_path=log_path,
        )
        out.append(row)
    return out


def run_ga_v2_correlation_stress(
    n_jobs: int = 1,
    method_jobs: int = 1,
    seed: int = MASTER_SEED,
    repeats: int = 30,
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
    active_groups: Sequence[int] | None = None,
    within_group_patterns: Sequence[str] | None = None,
    rho_within_list: Sequence[float] | None = None,
    n_train: int = 100,
    n_test: int = 30,
    rho_between: float = 0.2,
    target_snr: float = 1.0,
) -> Dict[str, str]:
    """
    Group-aware validation suite V2, Experiment C.

    Fix the group sparsity pattern and vary within-group correlation to test
    whether group-aware shrinkage becomes more valuable as grouped designs
    become more collinear.
    """
    pd = load_pandas()
    produced: set[Path] = set()

    base = Path(save_dir)
    out_dir = ensure_dir(base / "results" / "group_aware_v2" / "ga_v2_correlation_stress")
    tab_dir = ensure_dir(base / "tables" / "group_aware_v2")
    log = setup_logger("ga_v2_correlation_stress", base / "logs" / "ga_v2_correlation_stress.log")
    log_path = str(base / "logs" / "ga_v2_correlation_stress.log")

    sampler = _sampler_for_standard(experiment="ga_v2_correlation_stress")
    bayes_min_chains_use = int(bayes_min_chains) if bayes_min_chains is not None else int(_BAYESIAN_DEFAULT_CHAINS)
    bayes_min_chains_use = max(1, int(bayes_min_chains_use))
    retry_limit = _resolve_convergence_retry_limit(max_convergence_retries, until_bayes_converged=bool(until_bayes_converged))
    methods_use = [str(m) for m in (methods or ["GR_RHS", "RHS"])]
    group_sizes_use = [int(v) for v in (group_sizes or [10, 10, 10, 10, 10])]
    active_groups_use = [int(v) for v in (active_groups or [0, 1])]
    within_use = [str(v) for v in (within_group_patterns or ["mixed_decoy", "concentrated"])]
    rho_within_values = [float(v) for v in (rho_within_list or [0.4, 0.6, 0.8, 0.95])]
    if not rho_within_values:
        raise ValueError("rho_within_list must be non-empty.")
    if any(v <= float(rho_between) for v in rho_within_values):
        raise ValueError("All rho_within_list values must be strictly greater than rho_between.")

    grrhs_kwargs = {"tau_target": "groups", "progress_bar": False}
    tasks: list[tuple[Any, ...]] = []
    rid = 0
    for rho_within in rho_within_values:
        for within in within_use:
            for _ in range(int(repeats)):
                rid += 1
                tasks.append(
                    (
                        int(rid),
                        int(seed),
                        group_sizes_use,
                        active_groups_use,
                        str(within),
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
        _ga_v2_correlation_worker,
        n_jobs=n_jobs,
        prefer_process=True,
        process_fallback="serial",
        progress_desc="GA-V2 Correlation Stress",
    )
    rows: list[dict[str, Any]] = []
    for chunk in rows_nested:
        rows.extend(chunk)

    raw = pd.DataFrame(rows)
    if not raw.empty and "method" in raw.columns:
        raw["method_label"] = raw["method"].map(method_result_label)

    paired_raw, paired_stats = _paired_converged_subset(
        raw,
        group_cols=["rho_within", "within_group_pattern"],
        method_col="method",
        replicate_col="replicate_id",
        converged_col="converged",
        required_cols=["group_auroc", "mse_overall"],
        method_levels=methods_use,
    )
    summary = paired_raw.groupby(["rho_within", "within_group_pattern", "method"], as_index=False).agg(
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
        true_active_groups=("true_active_groups", "mean"),
        true_active_coeff=("true_active_coeff", "mean"),
        n_effective=("converged", "sum"),
    )
    if not summary.empty:
        summary["method_label"] = summary["method"].map(method_result_label)

    save_dataframe(raw, out_dir / "raw_results.csv")
    save_dataframe(paired_raw, out_dir / "summary_paired_raw.csv")
    save_dataframe(summary, out_dir / "summary.csv")
    save_dataframe(summary, out_dir / "summary_paired.csv")
    save_dataframe(paired_stats, out_dir / "paired_stats.csv")
    save_dataframe(summary, tab_dir / "ga_v2_correlation_stress_summary.csv")
    save_json(
        {
            "suite": "group_aware_v2",
            "experiment": "ga_v2_correlation_stress",
            "rho_within_list": rho_within_values,
            "within_group_patterns": within_use,
            "group_sizes": group_sizes_use,
            "active_groups": active_groups_use,
            "n_train": int(n_train),
            "n_test": int(n_test),
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
        out_dir / "meta.json",
        tab_dir / "ga_v2_correlation_stress_summary.csv",
    )

    result_paths = {
        "raw": str(out_dir / "raw_results.csv"),
        "summary_paired_raw": str(out_dir / "summary_paired_raw.csv"),
        "summary": str(out_dir / "summary.csv"),
        "summary_paired": str(out_dir / "summary_paired.csv"),
        "paired_stats": str(out_dir / "paired_stats.csv"),
        "table": str(tab_dir / "ga_v2_correlation_stress_summary.csv"),
        "meta": str(out_dir / "meta.json"),
    }
    return _finalize_experiment_run(
        exp_key="ga_v2_correlation_stress",
        save_dir=save_dir,
        results_dir=out_dir,
        produced_paths=produced,
        result_paths=result_paths,
        skip_run_analysis=bool(skip_run_analysis),
        archive_artifacts=bool(archive_artifacts),
    )
