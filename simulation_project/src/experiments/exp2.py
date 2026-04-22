from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from .evaluation import _evaluate_row, _kappa_group_means, _kappa_group_prob_gt
from .fitting import _fit_all_methods
from .reporting import _finalize_experiment_run, _paired_converged_subset, _record_produced_paths
from .runtime import (
    _BAYESIAN_DEFAULT_CHAINS,
    _attempts_used,
    _gigg_config_for_profile,
    _normalize_compute_profile,
    _parallel_rows,
    _resolve_convergence_retry_limit,
    _result_diag_fields,
    _sampler_for_profile,
    xi_crit_u0_rho,
)
from ..utils import (
    MASTER_SEED,
    SamplerConfig,
    ensure_dir,
    experiment_seed,
    load_pandas,
    save_dataframe,
    save_json,
    setup_logger,
)

def _exp2_worker(
    task: tuple[int, int, list[int], list[float], list[float], SamplerConfig, list[str], dict[str, Any], int, bool, int, int, dict]
) -> tuple[list[dict], list[dict]]:
    from .dgp.grouped_linear import generate_heterogeneity_dataset
    from .analysis.metrics import group_auroc, group_l2_error, group_l2_score
    from ..utils import sample_correlated_design

    r, seed, group_sizes, mu, xi_ratios, sampler, methods, gigg_config, bayes_min_chains, enforce_convergence, max_retries, n_test, grrhs_kwargs = task
    labels = (np.asarray(mu) > 0.0).astype(int)
    p0_signal_groups = int(np.sum(labels))
    n_train = 200
    s = experiment_seed(2, 1, r, master_seed=seed)

    ds = generate_heterogeneity_dataset(
        n=n_train, group_sizes=group_sizes, rho_within=0.3, rho_between=0.05,
        sigma2=1.0, mu=mu, seed=s,
    )
    X_test, _ = sample_correlated_design(n=n_test, group_sizes=group_sizes, rho_within=0.3, rho_between=0.05, seed=s + 77777)
    rng_test = np.random.default_rng(s + 88888)
    y_test = X_test @ ds["beta0"] + rng_test.normal(0.0, 1.0, n_test)

    fits = _fit_all_methods(
        ds["X"], ds["y"], ds["groups"],
        task="gaussian", seed=s, p0=p0_signal_groups,
        sampler=sampler, methods=methods, gigg_config=gigg_config,
        bayes_min_chains=int(bayes_min_chains),
        grrhs_kwargs=grrhs_kwargs or {},
        enforce_bayes_convergence=bool(enforce_convergence),
        max_convergence_retries=int(max_retries),
    )

    rep_rows: list[dict] = []
    kappa_rows: list[dict] = []
    n_groups = len(group_sizes)

    for method, res in fits.items():
        is_valid = bool(res.beta_mean is not None)
        metrics = _evaluate_row(res, ds["beta0"], X_train=ds["X"], y_train=ds["y"], X_test=X_test, y_test=y_test)
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
            **metrics,
        })
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
    seed: int = MASTER_SEED,
    repeats: int = 30,
    save_dir: str = "outputs/simulation_project",
    *,
    profile: str = "full",
    bayes_min_chains: int | None = None,
    methods: Sequence[str] | None = None,
    enforce_bayes_convergence: bool = True,
    max_convergence_retries: int | None = None,
    until_bayes_converged: bool = True,
    rho_ref: float = 0.5,
    n_test: int = 50,
    sampler_backend: str = "nuts",
) -> Dict[str, str]:
    """
    Exp2: Toy-example group separation (Theorem 3.34).

    4-group design calibrated at xi_crit(u0=0.5, rho=rho_ref=0.5):
      xi_crit ~= 0.1  (20x stronger signals than the old rho_ref=0.1 design)

      G0 (null):           p_g=30, xi_ratio=0.0  mu=0       beta_j=0.00
      G1 (below thresh):   p_g=20, xi_ratio=1.0  mu=2.0     beta_j=0.10
      G2 (above thresh):   p_g=15, xi_ratio=4.0  mu=6.0     beta_j=0.40
      G3 (strong signal):  p_g=10, xi_ratio=8.0  mu=8.0     beta_j=0.80

    Smaller group sizes for G2/G3 increase per-coefficient SNR so that the
    full-model kappa values approach the profile-specialization theory.

    Methods: GR_RHS vs RHS only.

    Key claims:
      - kappa_G0 ~ 0  (null contraction, Thm 3.22)
      - kappa_G1 < 0.5  (below-threshold suppression)
      - kappa_G2 > 0.5  (above-threshold activation, phase transition)
      - kappa_G3 ~ 1   (strong-signal retention, Thm 3.32)
      - GR_RHS lower null MSE and higher signal retention than RHS
    """
    pd = load_pandas()
    produced: set[Path] = set()

    base = Path(save_dir)
    out_dir = ensure_dir(base / "results" / "exp2_group_separation")
    fig_dir = ensure_dir(base / "figures")
    tab_dir = ensure_dir(base / "tables")
    log = setup_logger("exp2", base / "logs" / "exp2_group_separation.log")

    profile_name = _normalize_compute_profile(profile)
    sampler = _sampler_for_profile(profile_name)
    bayes_min_chains_use = int(bayes_min_chains) if bayes_min_chains is not None else (2 if profile_name == "laptop" else int(_BAYESIAN_DEFAULT_CHAINS))
    bayes_min_chains_use = max(1, int(bayes_min_chains_use))
    # Only GR_RHS and RHS; ignore any wider method list from profile resolver
    methods_use = [m for m in (methods or ["GR_RHS", "RHS"]) if m in ("GR_RHS", "RHS")]
    if not methods_use:
        methods_use = ["GR_RHS", "RHS"]
    gigg_cfg = _gigg_config_for_profile(profile_name)
    retry_limit = _resolve_convergence_retry_limit(profile_name, max_convergence_retries, until_bayes_converged=bool(until_bayes_converged))

    sigma2 = 1.0
    xi_c = xi_crit_u0_rho(u0=0.5, rho=float(rho_ref) / math.sqrt(sigma2))

    group_sizes = [30, 20, 15, 10]
    xi_ratios   = [0.0, 1.0, 4.0, 8.0]
    mu = [xi_ratios[i] * xi_c * group_sizes[i] for i in range(len(group_sizes))]

    log.info("Exp2 toy: rho_ref=%.2f, xi_crit=%.4f, xi_ratios=%s, mu=%s",
             rho_ref, xi_c, xi_ratios, [round(v, 3) for v in mu])

    grrhs_kw = {"backend": str(sampler_backend), "tau_target": "groups"}
    # Method-level task granularity gives smoother progress updates and better
    # load balancing when one method is much slower than others.
    tasks: list[tuple] = []
    for r in range(1, int(repeats) + 1):
        for method in methods_use:
            tasks.append(
                (
                    r,
                    seed,
                    group_sizes,
                    mu,
                    xi_ratios,
                    sampler,
                    [method],
                    gigg_cfg,
                    int(bayes_min_chains_use),
                    bool(enforce_bayes_convergence),
                    int(retry_limit),
                    int(n_test),
                    grrhs_kw,
                )
            )
    results = _parallel_rows(tasks, _exp2_worker, n_jobs=n_jobs, prefer_process=True, process_fallback="serial", progress_desc="Exp2 Group Separation")

    rep_rows: list[dict] = []
    kappa_rows: list[dict] = []
    for rep_chunk, kappa_chunk in results:
        rep_rows.extend(rep_chunk)
        kappa_rows.extend(kappa_chunk)

    raw = pd.DataFrame(rep_rows)
    kappa_df = pd.DataFrame(kappa_rows)

    # Summary: paired-converged across methods
    paired_raw, paired_stats = _paired_converged_subset(
        raw,
        group_cols=[],
        method_col="method",
        replicate_col="replicate_id",
        converged_col="converged",
        required_cols=["null_group_mse", "signal_group_mse", "group_auroc", "mse_overall"],
        method_levels=methods_use,
    )
    summary_df = raw.loc[raw["converged"]].groupby("method", as_index=False).agg(
        null_group_mse=("null_group_mse", "mean"),
        null_group_mse_std=("null_group_mse", "std"),
        signal_group_mse=("signal_group_mse", "mean"),
        signal_group_mse_std=("signal_group_mse", "std"),
        mse_overall=("mse_overall", "mean"),
        group_auroc=("group_auroc", "mean"),
        group_auroc_std=("group_auroc", "std"),
        lpd_test=("lpd_test", "mean"),
        lpd_test_std=("lpd_test", "std"),
        n_effective=("converged", "sum"),
    )
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
    save_dataframe(kappa_df, out_dir / "kappa_realizations.csv")
    _record_produced_paths(produced, out_dir / "kappa_realizations.csv")
    if not kappa_summary.empty:
        save_dataframe(kappa_summary, out_dir / "kappa_summary_by_group.csv")
        _record_produced_paths(produced, out_dir / "kappa_summary_by_group.csv")
        save_dataframe(kappa_summary, tab_dir / "table_kappa_group_separation.csv")
        _record_produced_paths(produced, tab_dir / "table_kappa_group_separation.csv")
    save_dataframe(summary_df, tab_dir / "table_group_separation.csv")
    _record_produced_paths(produced, tab_dir / "table_group_separation.csv")
    save_json({"rho_ref": float(rho_ref), "xi_crit": float(xi_c), "xi_ratios": xi_ratios, "mu": [round(v, 4) for v in mu], "group_sizes": group_sizes, "methods": methods_use, "bayes_min_chains": int(bayes_min_chains_use)}, out_dir / "exp2_meta.json")
    _record_produced_paths(produced, out_dir / "exp2_meta.json")

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
    )




