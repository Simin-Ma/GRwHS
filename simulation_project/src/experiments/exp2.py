from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from .evaluation import _bridge_ratio_diagnostics, _evaluate_row, _kappa_group_means, _kappa_group_prob_gt
from .fitting import _fit_all_methods
from .reporting import _finalize_experiment_run, _paired_converged_subset, _record_produced_paths
from .runtime import (
    _BAYESIAN_DEFAULT_CHAINS,
    _attempts_used,
    _gigg_config_default,
    _parallel_rows,
    _resolve_convergence_retry_limit,
    _result_diag_fields,
    _sampler_for_standard,
    xi_crit_u0_rho,
)
from ..utils import (
    MASTER_SEED,
    SamplerConfig,
    ensure_dir,
    experiment_seed,
    load_pandas,
    method_result_label,
    print_experiment_result,
    save_dataframe,
    save_json,
    setup_logger,
)

def _exp2_worker(
    task: tuple[
        int,
        int,
        list[int],
        list[float],
        list[float],
        float,
        float,
        float,
        int,
        SamplerConfig,
        list[str],
        dict[str, Any],
        int,
        bool,
        int,
        int,
        dict,
        int,
        str,
    ]
) -> tuple[list[dict], list[dict]]:
    from .dgp.grouped_linear import generate_heterogeneity_dataset
    from .analysis.metrics import group_auroc, group_l2_error, group_l2_score
    from ..utils import sample_correlated_design

    (
        r,
        seed,
        group_sizes,
        mu,
        xi_ratios,
        rho_within,
        rho_between,
        sigma2,
        n_train,
        sampler,
        methods,
        gigg_config,
        bayes_min_chains,
        enforce_convergence,
        max_retries,
        n_test,
        grrhs_kwargs,
        method_jobs,
        log_path,
    ) = task
    labels = (np.asarray(mu) > 0.0).astype(int)
    p0_signal_groups = int(np.sum(labels))
    s = experiment_seed(2, 1, r, master_seed=seed)

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
        seed=s + 77777,
    )
    rng_test = np.random.default_rng(s + 88888)
    y_test = X_test @ ds["beta0"] + rng_test.normal(0.0, math.sqrt(float(sigma2)), int(n_test))

    fits = _fit_all_methods(
        ds["X"], ds["y"], ds["groups"],
        task="gaussian", seed=s, p0=p0_signal_groups,
        sampler=sampler, methods=methods, gigg_config=gigg_config,
        bayes_min_chains=int(bayes_min_chains),
        grrhs_kwargs=grrhs_kwargs or {},
        enforce_bayes_convergence=bool(enforce_convergence),
        max_convergence_retries=int(max_retries),
        method_jobs=int(method_jobs),
    )

    rep_rows: list[dict] = []
    kappa_rows: list[dict] = []
    n_groups = len(group_sizes)

    for method, res in fits.items():
        is_valid = bool(res.beta_mean is not None)
        metrics = _evaluate_row(res, ds["beta0"], X_train=ds["X"], y_train=ds["y"], X_test=X_test, y_test=y_test)
        bridge_diag = _bridge_ratio_diagnostics(
            res,
            groups=ds["groups"],
            X=ds["X"],
            y=ds["y"],
            signal_group_mask=(labels == 1),
        )
        null_mse_group = float("nan")
        sig_mse_group  = float("nan")
        auroc          = float("nan")
        if is_valid:
            err  = group_l2_error(res.beta_mean, ds["beta0"], ds["groups"])
            score = group_l2_score(res.beta_mean, ds["groups"])
            null_mse_group = float(np.mean(err[labels == 0]))
            sig_mse_group  = float(np.mean(err[labels == 1]))
            auroc = group_auroc(score, labels)
        rep_rows.append({
            "replicate_id": r, "method": method,
            "status": res.status, "converged": bool(res.converged), "fit_attempts": _attempts_used(res),
            "null_group_mse": null_mse_group, "signal_group_mse": sig_mse_group,
            "group_auroc": auroc,
            **_result_diag_fields(res),
            **bridge_diag,
            **metrics,
        })
        print_experiment_result(
            "Exp2",
            rep_rows[-1],
            context_keys=["replicate_id", "method"],
            metric_keys=["mse_overall", "mse_null", "mse_signal", "group_auroc"],
            log_path=log_path,
        )
        if method == "GR_RHS" and res.beta_mean is not None:
            kmeans = _kappa_group_means(res, n_groups)
            kprobs = _kappa_group_prob_gt(res, n_groups, threshold=0.5)
            for gid in range(n_groups):
                kappa_rows.append({
                    "replicate_id": r, "group_id": gid,
                    "mu_g": float(ds["mu"][gid]),
                    "xi_ratio": float(xi_ratios[gid]) if gid < len(xi_ratios) else float("nan"),
                    "signal_label": int(labels[gid]),
                    "post_mean_kappa_g": kmeans[gid],
                    "post_prob_kappa_g_gt_0_5": kprobs[gid],
                })
    return rep_rows, kappa_rows


def run_exp2_group_separation(
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
    rho_ref: float = 0.8,
    group_sizes: Sequence[int] | None = None,
    xi_ratios: Sequence[float] | None = None,
    n_train: int = 100,
    n_test: int = 30,
    rho_within: float = 0.8,
    rho_between: float = 0.2,
    sigma2: float = 1.0,
) -> Dict[str, str]:
    """
    Exp2: Toy-example group separation (Theorem 3.34), single-default protocol.

    Default design aligned to Exp3 scale (p=50; G10x5), calibrated at
    xi_crit(u0=0.5, rho=rho_ref=0.8):
      group_sizes = [10, 10, 10, 10, 10]
      xi_ratios   = [0.0, 1.0, 2.0, 5.0, 10.0]
      n_train=100, n_test=30, rho_within=0.8, rho_between=0.2, sigma2=1.0

    Methods: GR_RHS vs RHS only; summary uses paired-converged subset.

    Key claims:
      - null groups exhibit strong contraction (Thm 3.22)
      - boundary-near groups show threshold-sensitive activation behavior
      - strongest signal groups retain large posterior activity (Thm 3.32)
      - GR_RHS lower null MSE and higher signal retention than RHS
    """
    pd = load_pandas()
    produced: set[Path] = set()

    base = Path(save_dir)
    out_dir = ensure_dir(base / "results" / "exp2_group_separation")
    fig_dir = ensure_dir(base / "figures")
    tab_dir = ensure_dir(base / "tables")
    log = setup_logger("exp2", base / "logs" / "exp2_group_separation.log")
    log_path = str(base / "logs" / "exp2_group_separation.log")

    sampler = _sampler_for_standard()
    bayes_min_chains_use = int(bayes_min_chains) if bayes_min_chains is not None else int(_BAYESIAN_DEFAULT_CHAINS)
    bayes_min_chains_use = max(1, int(bayes_min_chains_use))
    # Only GR_RHS and RHS are part of Exp2's default protocol.
    methods_use = [m for m in (methods or ["GR_RHS", "RHS"]) if m in ("GR_RHS", "RHS")]
    if not methods_use:
        methods_use = ["GR_RHS", "RHS"]
    gigg_cfg = _gigg_config_default()
    retry_limit = _resolve_convergence_retry_limit(max_convergence_retries, until_bayes_converged=bool(until_bayes_converged))

    if int(n_train) <= 0 or int(n_test) <= 0:
        raise ValueError("n_train and n_test must be positive integers.")
    if float(sigma2) <= 0.0:
        raise ValueError("sigma2 must be > 0.")
    if float(rho_within) <= float(rho_between):
        raise ValueError("Expected rho_within > rho_between for grouped design.")

    group_sizes_use = [int(v) for v in (group_sizes or [10, 10, 10, 10, 10])]
    if not group_sizes_use or any(v <= 0 for v in group_sizes_use):
        raise ValueError("group_sizes must be a non-empty sequence of positive integers.")
    xi_ratios_use = [float(v) for v in (xi_ratios or [0.0, 1.0, 2.0, 5.0, 10.0])]
    if len(xi_ratios_use) != len(group_sizes_use):
        raise ValueError("xi_ratios length must equal group_sizes length.")

    sigma2_use = float(sigma2)
    xi_c = xi_crit_u0_rho(u0=0.5, rho=float(rho_ref) / math.sqrt(sigma2_use))
    mu = [xi_ratios_use[i] * xi_c * group_sizes_use[i] for i in range(len(group_sizes_use))]

    log.info(
        "Exp2 toy: rho_ref=%.2f, xi_crit=%.4f, rho_within=%.2f, rho_between=%.2f, n_train=%d, sigma2=%.2f, xi_ratios=%s, mu=%s",
        rho_ref,
        xi_c,
        float(rho_within),
        float(rho_between),
        int(n_train),
        sigma2_use,
        xi_ratios_use,
        [round(v, 3) for v in mu],
    )

    grrhs_kw = {"tau_target": "groups", "progress_bar": False}
    tasks: list[tuple] = []
    for r in range(1, int(repeats) + 1):
        tasks.append(
            (
                r,
                seed,
                group_sizes_use,
                mu,
                xi_ratios_use,
                float(rho_within),
                float(rho_between),
                sigma2_use,
                int(n_train),
                sampler,
                methods_use,
                gigg_cfg,
                int(bayes_min_chains_use),
                bool(enforce_bayes_convergence),
                int(retry_limit),
                int(n_test),
                grrhs_kw,
                int(method_jobs),
                log_path,
            )
        )
    results = _parallel_rows(tasks, _exp2_worker, n_jobs=n_jobs, prefer_process=True, process_fallback="serial", progress_desc="Exp2 Group Separation")

    rep_rows: list[dict] = []
    kappa_rows: list[dict] = []
    for rep_chunk, kappa_chunk in results:
        rep_rows.extend(rep_chunk)
        kappa_rows.extend(kappa_chunk)

    raw = pd.DataFrame(rep_rows)
    if not raw.empty and "method" in raw.columns:
        raw["method_label"] = raw["method"].map(method_result_label)
    kappa_df = pd.DataFrame(kappa_rows)

    # Summary: paired-converged across methods
    paired_raw, paired_stats = _paired_converged_subset(
        raw,
        group_cols=[],
        method_col="method",
        replicate_col="replicate_id",
        converged_col="converged",
        required_cols=[
            "null_group_mse",
            "signal_group_mse",
            "mse_null",
            "mse_signal",
            "mse_overall",
            "group_auroc",
        ],
        method_levels=methods_use,
    )

    # Paper-aligned reporting (GIGG / GRASP):
    # For Exp2 summaries, report coefficient-level MSE contributions so that
    #   mse_overall = mse_null + mse_signal
    # where null/signal are the aggregate contributions over all coordinates.
    if not paired_raw.empty and {"mse_null", "mse_signal", "mse_overall"}.issubset(set(paired_raw.columns)):
        m0 = pd.to_numeric(paired_raw["mse_null"], errors="coerce")
        m1 = pd.to_numeric(paired_raw["mse_signal"], errors="coerce")
        total_dim = int(sum(group_sizes_use))
        signal_dim = int(sum(group_sizes_use[i] for i, mg in enumerate(mu) if float(mg) > 0.0))
        null_dim = max(total_dim - signal_dim, 0)
        w_null = float(null_dim / total_dim) if total_dim > 0 else 0.0
        w_signal = float(signal_dim / total_dim) if total_dim > 0 else 0.0
        paired_raw["mse_null"] = m0 * w_null
        paired_raw["mse_signal"] = m1 * w_signal

    summary_df = paired_raw.groupby("method", as_index=False).agg(
        null_group_mse=("null_group_mse", "mean"),
        null_group_mse_std=("null_group_mse", "std"),
        signal_group_mse=("signal_group_mse", "mean"),
        signal_group_mse_std=("signal_group_mse", "std"),
        mse_null=("mse_null", "mean"),
        mse_null_std=("mse_null", "std"),
        mse_signal=("mse_signal", "mean"),
        mse_signal_std=("mse_signal", "std"),
        mse_overall=("mse_overall", "mean"),
        mse_overall_std=("mse_overall", "std"),
        group_auroc=("group_auroc", "mean"),
        group_auroc_std=("group_auroc", "std"),
        lpd_test=("lpd_test", "mean"),
        lpd_test_std=("lpd_test", "std"),
        n_effective=("converged", "sum"),
    )
    if "lpd_test_ppd" in paired_raw.columns:
        lpd_ppd = paired_raw.groupby("method", as_index=False).agg(
            lpd_test_ppd=("lpd_test_ppd", "mean"),
            lpd_test_ppd_std=("lpd_test_ppd", "std"),
        )
        summary_df = summary_df.merge(lpd_ppd, on="method", how="left")
    if "lpd_test_plugin" in paired_raw.columns:
        lpd_plugin = paired_raw.groupby("method", as_index=False).agg(
            lpd_test_plugin=("lpd_test_plugin", "mean"),
            lpd_test_plugin_std=("lpd_test_plugin", "std"),
        )
        summary_df = summary_df.merge(lpd_plugin, on="method", how="left")
    summary_df["method_label"] = summary_df["method"].map(method_result_label)

    paired_delta_rows: list[dict[str, Any]] = []
    for metric in [
        "mse_null",
        "mse_signal",
        "mse_overall",
        "null_group_mse",
        "signal_group_mse",
        "group_auroc",
        "lpd_test",
    ]:
        if metric not in paired_raw.columns:
            continue
        wide = paired_raw.pivot_table(index="replicate_id", columns="method", values=metric, aggfunc="mean")
        if "GR_RHS" not in wide.columns or "RHS" not in wide.columns:
            continue
        diff = (wide["GR_RHS"] - wide["RHS"]).dropna()
        n_eff = int(diff.shape[0])
        if n_eff == 0:
            continue
        mean_v = float(diff.mean())
        sd_v = float(diff.std(ddof=1)) if n_eff > 1 else float("nan")
        se_v = float(sd_v / np.sqrt(n_eff)) if n_eff > 1 else float("nan")
        ci_lo = float(mean_v - 1.96 * se_v) if np.isfinite(se_v) else float("nan")
        ci_hi = float(mean_v + 1.96 * se_v) if np.isfinite(se_v) else float("nan")
        paired_delta_rows.append(
            {
                "metric": metric,
                "contrast": "GR_RHS - RHS",
                "contrast_label": "GR_RHS - RHS [stan_rstanarm_hs]",
                "mean_diff": mean_v,
                "std_diff": sd_v,
                "se_diff": se_v,
                "ci95_lo": ci_lo,
                "ci95_hi": ci_hi,
                "n_effective_pairs": n_eff,
            }
        )
    paired_delta_df = pd.DataFrame(paired_delta_rows)
    # kappa summary by group (GR_RHS)
    if not kappa_df.empty:
        kappa_summary = kappa_df.groupby(["group_id", "signal_label"], as_index=False).agg(
            mu_g=("mu_g", "first"),
            mean_kappa=("post_mean_kappa_g", "mean"),
            mean_prob_gt_0_5=("post_prob_kappa_g_gt_0_5", "mean"),
            n_reps=("replicate_id", "nunique"),
        )
    else:
        kappa_summary = pd.DataFrame()

    save_dataframe(raw, out_dir / "raw_results.csv")
    _record_produced_paths(produced, out_dir / "raw_results.csv")
    save_dataframe(summary_df, out_dir / "summary.csv")
    _record_produced_paths(produced, out_dir / "summary.csv")
    save_dataframe(summary_df, out_dir / "summary_paired.csv")
    _record_produced_paths(produced, out_dir / "summary_paired.csv")
    save_dataframe(paired_delta_df, out_dir / "paired_deltas.csv")
    _record_produced_paths(produced, out_dir / "paired_deltas.csv")
    save_dataframe(kappa_df, out_dir / "kappa_realizations.csv")
    _record_produced_paths(produced, out_dir / "kappa_realizations.csv")
    if not kappa_summary.empty:
        save_dataframe(kappa_summary, out_dir / "kappa_summary_by_group.csv")
        _record_produced_paths(produced, out_dir / "kappa_summary_by_group.csv")
        save_dataframe(kappa_summary, tab_dir / "table_kappa_group_separation.csv")
        _record_produced_paths(produced, tab_dir / "table_kappa_group_separation.csv")
    save_dataframe(summary_df, tab_dir / "table_group_separation.csv")
    _record_produced_paths(produced, tab_dir / "table_group_separation.csv")
    save_json(
        {
            "rho_ref": float(rho_ref),
            "rho_within": float(rho_within),
            "rho_between": float(rho_between),
            "sigma2": sigma2_use,
            "xi_crit": float(xi_c),
            "xi_ratios": xi_ratios_use,
            "mu": [round(v, 4) for v in mu],
            "group_sizes": group_sizes_use,
            "n_train": int(n_train),
            "n_test": int(n_test),
            "methods": methods_use,
            "bayes_min_chains": int(bayes_min_chains_use),
            "method_jobs": int(method_jobs),
        },
        out_dir / "exp2_meta.json",
    )
    _record_produced_paths(produced, out_dir / "exp2_meta.json")
    save_json(
        {
            "paired_stats": paired_stats.to_dict(orient="records"),
            "pairing_note": "summary.csv is computed from paired-converged subset only",
        },
        out_dir / "paired_stats.json",
    )
    _record_produced_paths(produced, out_dir / "paired_stats.json")

    try:
        from .analysis.plotting import plot_exp2_separation
        plot_exp2_separation(summary_df, kappa_df, out_dir=fig_dir)
        _record_produced_paths(produced, fig_dir / "fig2a_method_comparison.png", fig_dir / "fig2b_kappa_by_group.png")
    except Exception as exc:
        log.warning("Plot exp2 failed: %s", exc)

    log.info("Exp2 done: %d replicates, %d kappa rows", len(rep_rows), len(kappa_rows))
    result_paths = {
        "raw": str(out_dir / "raw_results.csv"),
        "summary": str(out_dir / "summary.csv"),
        "summary_paired": str(out_dir / "summary_paired.csv"),
        "paired_deltas": str(out_dir / "paired_deltas.csv"),
        "table": str(tab_dir / "table_group_separation.csv"),
        "kappa_realizations": str(out_dir / "kappa_realizations.csv"),
        "meta": str(out_dir / "exp2_meta.json"),
        "fig2a_method_comparison": str(fig_dir / "fig2a_method_comparison.png"),
        "fig2b_kappa_by_group": str(fig_dir / "fig2b_kappa_by_group.png"),
    }
    if (out_dir / "kappa_summary_by_group.csv").exists():
        result_paths["kappa_summary_by_group"] = str(out_dir / "kappa_summary_by_group.csv")
    if (tab_dir / "table_kappa_group_separation.csv").exists():
        result_paths["table_kappa_group_separation"] = str(tab_dir / "table_kappa_group_separation.csv")
    return _finalize_experiment_run(
        exp_key="exp2",
        save_dir=save_dir,
        results_dir=out_dir,
        produced_paths=produced,
        result_paths=result_paths,
        skip_run_analysis=bool(skip_run_analysis),
        archive_artifacts=bool(archive_artifacts),
    )
