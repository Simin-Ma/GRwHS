from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from .evaluation import _bridge_ratio_diagnostics, _kappa_group_means
from .fitting import _fit_with_convergence_retry
from .reporting import _finalize_experiment_run, _paired_converged_subset, _record_produced_paths
from .runtime import (
    _BAYESIAN_DEFAULT_CHAINS,
    _EXP5_DEFAULT_MAX_CONV_RETRIES,
    _attempts_used,
    _parallel_rows,
    _resolve_convergence_retry_limit,
    _result_diag_fields,
    _sampler_for_exp5,
    _sampler_for_standard,
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

_DEFAULT_PRIOR_GRID: list[tuple[float, float]] = [
    (0.5, 1.0),  # default (slight null preference)
    (1.0, 1.0),  # Uniform on (0,1)
    (0.5, 0.5),  # U-shape (mass near boundaries)
    (2.0, 5.0),  # concentrated near 0 (aggressive null shrinkage)
    (1.0, 3.0),  # moderate null preference
]


def _prior_key(alpha_kappa: float, beta_kappa: float) -> str:
    return f"{float(alpha_kappa):.6g}|{float(beta_kappa):.6g}"


def _screen_sampler_for_exp5(base: SamplerConfig) -> SamplerConfig:
    return SamplerConfig(
        chains=max(2, min(int(base.chains), 2)),
        warmup=max(200, min(int(base.warmup), 300)),
        post_warmup_draws=max(200, min(int(base.post_warmup_draws), 300)),
        adapt_delta=max(0.9, min(float(base.adapt_delta), 0.95)),
        max_treedepth=max(10, min(int(base.max_treedepth), 12)),
        strict_adapt_delta=max(0.95, min(float(base.strict_adapt_delta), 0.99)),
        strict_max_treedepth=max(12, min(int(base.strict_max_treedepth), 14)),
        max_divergence_ratio=float(base.max_divergence_ratio),
        rhat_threshold=max(1.02, float(base.rhat_threshold)),
        ess_threshold=min(250.0, float(base.ess_threshold)),
    )


def _exp5_screen_prior_worker(
    task: tuple[int, list[int], list[float], int, SamplerConfig, float, float, int, bool, int, str]
) -> dict[str, Any]:
    from .dgp.grouped_linear import generate_heterogeneity_dataset
    from .methods.fit_gr_rhs import fit_gr_rhs

    sid, group_sizes, mu, seed, sampler, alpha_k, beta_k, bayes_min_chains, enforce_conv, max_retries, backend = task
    labels = (np.asarray(mu) > 0.0).astype(int)
    p0_signal_groups = int(np.sum(labels))
    s = experiment_seed(5, int(sid), 1, master_seed=seed)
    ds = generate_heterogeneity_dataset(
        n=100,
        group_sizes=group_sizes,
        rho_within=0.3,
        rho_between=0.05,
        sigma2=1.0,
        mu=mu,
        seed=s,
    )
    res = _fit_with_convergence_retry(
        lambda st, att, _resume=None, _a=alpha_k, _b=beta_k, _s=s, _be=backend: fit_gr_rhs(
            ds["X"],
            ds["y"],
            ds["groups"],
            task="gaussian",
            seed=_s + 10_000 + 100 * att,
            p0=p0_signal_groups,
            sampler=st,
            alpha_kappa=float(_a),
            beta_kappa=float(_b),
            use_group_scale=True,
            use_local_scale=True,
            shared_kappa=False,
            tau_target="groups",
            backend=_be,
            retry_resume_payload=_resume,
        ),
        method="GR_RHS",
        sampler=sampler,
        bayes_min_chains=int(bayes_min_chains),
        max_convergence_retries=max_retries,
        enforce_bayes_convergence=bool(enforce_conv),
        continue_on_retry=True,
    )
    return {
        "setting_id": int(sid),
        "prior_key": _prior_key(alpha_k, beta_k),
        "alpha_kappa": float(alpha_k),
        "beta_kappa": float(beta_k),
        "status": str(res.status),
        "converged": bool(res.converged),
        "fit_attempts": _attempts_used(res),
        "runtime_seconds": float(res.runtime_seconds),
        "error": str(res.error),
    }


def _exp5_worker(
    task: tuple[int, int, list[int], list[float], int, SamplerConfig, list[tuple[float, float]], int, int, bool, int, str]
) -> list[dict[str, Any]]:
    from .dgp.grouped_linear import generate_heterogeneity_dataset
    from .methods.fit_gr_rhs import fit_gr_rhs
    from .analysis.metrics import group_auroc, group_l2_score

    sid, r, group_sizes, mu, seed, sampler, prior_grid, bayes_min_chains, method_jobs, enforce_conv, max_retries, backend = task
    labels = (np.asarray(mu) > 0.0).astype(int)
    p0_signal_groups = int(np.sum(labels))
    s = experiment_seed(5, int(sid), r, master_seed=seed)
    # All priors evaluated on THE SAME dataset (paired comparison).
    ds = generate_heterogeneity_dataset(
        n=100,
        group_sizes=group_sizes,
        rho_within=0.3,
        rho_between=0.05,
        sigma2=1.0,
        mu=mu,
        seed=s,
    )
    n_groups = len(group_sizes)
    def _fit_prior(item: tuple[int, tuple[float, float]]) -> dict[str, Any]:
        pid, prior_vals = item
        alpha_k, beta_k = prior_vals
        res = _fit_with_convergence_retry(
            lambda st, att, _resume=None, _a=alpha_k, _b=beta_k, _s=s, _pid=pid, _be=backend: fit_gr_rhs(
                ds["X"],
                ds["y"],
                ds["groups"],
                task="gaussian",
                seed=_s + 100 + _pid + 100 * att,
                p0=p0_signal_groups,
                sampler=st,
                alpha_kappa=float(_a),
                beta_kappa=float(_b),
                use_group_scale=True,
                use_local_scale=True,
                shared_kappa=False,
                tau_target="groups",
                backend=_be,
                retry_resume_payload=_resume,
            ),
            method="GR_RHS",
            sampler=sampler,
            bayes_min_chains=int(bayes_min_chains),
            max_convergence_retries=max_retries,
            enforce_bayes_convergence=bool(enforce_conv),
            continue_on_retry=True,
        )
        is_valid = bool(res.beta_mean is not None)
        mse_null = float("nan")
        mse_signal = float("nan")
        auroc = float("nan")
        kappa_null_mean = float("nan")
        kappa_signal_mean = float("nan")
        kappa_null_prob_gt_0_1 = float("nan")
        if is_valid:
            from .analysis.metrics import group_auroc, group_l2_score, mse_null_signal_overall

            m = mse_null_signal_overall(res.beta_mean, ds["beta0"])
            mse_null = m["mse_null"]
            mse_signal = m["mse_signal"]
            score = group_l2_score(res.beta_mean, ds["groups"])
            auroc = group_auroc(score, labels)
            km = _kappa_group_means(res, n_groups)
            kms = np.array(km)
            kappa_null_mean = float(np.nanmean(kms[labels == 0])) if np.any(labels == 0) else float("nan")
            kappa_signal_mean = float(np.nanmean(kms[labels == 1])) if np.any(labels == 1) else float("nan")
            if res.kappa_draws is not None:
                kd = np.asarray(res.kappa_draws, dtype=float)
                if kd.ndim > 2:
                    kd = kd.reshape(-1, kd.shape[-1])
                null_idx = np.where(labels == 0)[0]
                if null_idx.size > 0 and kd.shape[-1] >= n_groups:
                    kappa_null_prob_gt_0_1 = float(np.mean(kd[:, null_idx] > 0.1))
        return {
            "setting_id": int(sid),
            "replicate_id": int(r),
            "prior_id": pid,
            "alpha_kappa": float(alpha_k),
            "beta_kappa": float(beta_k),
            "p0_signal_groups": p0_signal_groups,
            "tau_target": "groups",
            "status": res.status,
            "converged": bool(res.converged),
            "fit_attempts": _attempts_used(res),
            "mse_null": mse_null,
            "mse_signal": mse_signal,
            "group_auroc": auroc,
            "kappa_null_mean": kappa_null_mean,
            "kappa_signal_mean": kappa_signal_mean,
            "kappa_null_prob_gt_0_1": kappa_null_prob_gt_0_1,
            **_result_diag_fields(res),
            **_bridge_ratio_diagnostics(
                res,
                groups=ds["groups"],
                X=ds["X"],
                y=ds["y"],
                signal_group_mask=(labels == 1),
            ),
        }

    prior_items = list(enumerate(prior_grid, start=1))
    workers = max(1, min(int(method_jobs), len(prior_items)))
    if workers <= 1 or len(prior_items) <= 1:
        return [_fit_prior(item) for item in prior_items]

    done: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_map = {ex.submit(_fit_prior, item): int(item[0]) for item in prior_items}
        for fut in as_completed(fut_map):
            row = fut.result()
            done[int(row["prior_id"])] = row
    return [done[int(pid)] for pid, _ in prior_items]


def run_exp5_prior_sensitivity(
    n_jobs: int = 1,
    method_jobs: int = 1,
    seed: int = MASTER_SEED,
    repeats: int = 30,
    save_dir: str = "outputs/simulation_project",
    *,
    skip_run_analysis: bool = False,
    archive_artifacts: bool = True,
    prior_grid: Sequence[tuple[float, float]] | None = None,
    bayes_min_chains: int | None = None,
    enforce_bayes_convergence: bool = True,
    max_convergence_retries: int | None = None,
    until_bayes_converged: bool = True,
    sampler_backend: str = "collapsed",
    screen_priors: bool = True,
    screen_min_successes: int | None = None,
    screen_max_retries: int = 1,
) -> Dict[str, str]:
    """
    Exp5: Prior sensitivity - (alpha_kappa, beta_kappa) grid.

    All prior configurations run on the SAME DGP replicate (paired evaluation),
    so differences in output reflect ONLY the prior choice, not data variation.

    DGP aligned with Exp3 scale (n=100, rho_within=0.3):
      S1 (equal groups):   group_sizes=[10]*5  (p=50, G10x5), mu=[0,0,1.5,4.0,10.0] (2 null, 3 signal)
      S2 (unequal groups): group_sizes=[30,10,5,3,2] (p=50, CL),  mu=[0,0,1.5,4.0,10.0] (2 null, 3 signal)

    Prior grid (alpha_kappa, beta_kappa):
      (0.5, 1.0): default - slight null preference
      (1.0, 1.0): Uniform(0,1) - flat
      (0.5, 0.5): U-shape - mass near 0 and 1
      (2.0, 5.0): concentrated near 0 - aggressive null shrinkage
      (1.0, 3.0): moderate null preference
    """
    pd = load_pandas()
    produced: set[Path] = set()

    base = Path(save_dir)
    out_dir = ensure_dir(base / "results" / "exp5_prior_sensitivity")
    tab_dir = ensure_dir(base / "tables")
    log = setup_logger("exp5", base / "logs" / "exp5_prior_sensitivity.log")

    sampler = _sampler_for_exp5(_sampler_for_standard())
    screen_sampler = _screen_sampler_for_exp5(sampler)
    bayes_min_chains_use = int(bayes_min_chains) if bayes_min_chains is not None else int(_BAYESIAN_DEFAULT_CHAINS)
    bayes_min_chains_use = max(1, int(bayes_min_chains_use))
    if max_convergence_retries is None:
        retry_limit = int(_EXP5_DEFAULT_MAX_CONV_RETRIES)
    else:
        retry_limit = _resolve_convergence_retry_limit(max_convergence_retries, until_bayes_converged=bool(until_bayes_converged))
    if retry_limit < 0:
        # Exp5 is intentionally heavy; cap unlimited mode to a practical retry budget.
        retry_limit = int(_EXP5_DEFAULT_MAX_CONV_RETRIES)
    priors = list(prior_grid or _DEFAULT_PRIOR_GRID)

    scenarios: list[tuple[int, list[int], list[float]]] = [
        (1, [10, 10, 10, 10, 10], [0.0, 0.0, 1.5, 4.0, 10.0]),  # G10x5: p=50, 2 null + 3 signal
        (2, [30, 10, 5, 3, 2], [0.0, 0.0, 1.5, 4.0, 10.0]),  # CL:    p=50, 2 null + 3 signal
    ]

    screen_df = pd.DataFrame()
    retained_priors = list(priors)
    default_prior = tuple(float(v) for v in priors[0]) if priors else None
    if bool(screen_priors) and len(priors) > 1:
        screen_tasks: list[tuple[int, list[int], list[float], int, SamplerConfig, float, float, int, bool, int, str]] = []
        for scen_id, grp_sizes, mu in scenarios:
            for alpha_k, beta_k in priors:
                screen_tasks.append(
                    (
                        int(scen_id),
                        list(grp_sizes),
                        list(mu),
                        int(seed),
                        screen_sampler,
                        float(alpha_k),
                        float(beta_k),
                        int(bayes_min_chains_use),
                        bool(enforce_bayes_convergence),
                        int(max(0, screen_max_retries)),
                        str(sampler_backend),
                    )
                )
        screen_rows = _parallel_rows(
            screen_tasks,
            _exp5_screen_prior_worker,
            n_jobs=n_jobs,
            prefer_process=True,
            process_fallback="serial",
            progress_desc="Exp5 Prior Screen",
        )
        screen_df = pd.DataFrame(screen_rows)
        if not screen_df.empty:
            success_df = screen_df.groupby(["prior_key", "alpha_kappa", "beta_kappa"], as_index=False).agg(
                n_screen_runs=("setting_id", "count"),
                n_screen_success=("converged", lambda s: int(s.fillna(False).astype(bool).sum())),
                mean_screen_runtime=("runtime_seconds", "mean"),
            )
            threshold = int(screen_min_successes) if screen_min_successes is not None else max(1, len(scenarios) // 2)
            keep_keys = {
                str(row["prior_key"])
                for row in success_df.to_dict(orient="records")
                if int(row["n_screen_success"]) >= threshold
            }
            if default_prior is not None:
                keep_keys.add(_prior_key(*default_prior))
            if len(keep_keys) < min(2, len(priors)):
                ranked = success_df.sort_values(
                    by=["n_screen_success", "mean_screen_runtime"],
                    ascending=[False, True],
                )
                for row in ranked.to_dict(orient="records"):
                    keep_keys.add(str(row["prior_key"]))
                    if len(keep_keys) >= min(2, len(priors)):
                        break
            retained_priors = [
                (float(a), float(b))
                for (a, b) in priors
                if _prior_key(a, b) in keep_keys
            ]
            if not retained_priors:
                retained_priors = list(priors)
            save_dataframe(success_df, out_dir / "screening_summary.csv")
            _record_produced_paths(produced, out_dir / "screening_summary.csv")
        log.info(
            "Exp5 prior screen kept %d/%d priors: %s",
            len(retained_priors),
            len(priors),
            [_prior_key(a, b) for (a, b) in retained_priors],
        )

    tasks: list[tuple] = []
    for scen_id, grp_sizes, mu in scenarios:
        for r in range(1, int(repeats) + 1):
            tasks.append((scen_id, r, grp_sizes, mu, seed, sampler, retained_priors, int(bayes_min_chains_use), int(method_jobs), bool(enforce_bayes_convergence), int(retry_limit), str(sampler_backend)))

    log.info("Exp5: %d scenarios x %d repeats x %d retained priors = %d task-rows", len(scenarios), repeats, len(retained_priors), len(tasks) * len(retained_priors))
    all_chunks = _parallel_rows(tasks, _exp5_worker, n_jobs=n_jobs, prefer_process=True, process_fallback="serial", progress_desc="Exp5 Prior Sensitivity")
    rows: list[dict] = []
    for chunk in all_chunks:
        rows.extend(chunk)

    raw = pd.DataFrame(rows)
    raw["prior_key"] = raw.apply(lambda r: _prior_key(r["alpha_kappa"], r["beta_kappa"]), axis=1)
    prior_levels = [_prior_key(a, b) for (a, b) in retained_priors]
    paired_raw, paired_stats = _paired_converged_subset(
        raw,
        group_cols=["setting_id"],
        method_col="prior_key",
        replicate_col="replicate_id",
        converged_col="converged",
        required_cols=["mse_null", "mse_signal", "group_auroc"],
        method_levels=prior_levels,
    )

    partial_raw = raw.loc[
        raw["converged"].fillna(False).astype(bool)
        & raw["status"].astype(str).str.lower().eq("ok")
    ].copy()

    summary = paired_raw.groupby(["setting_id", "alpha_kappa", "beta_kappa"], as_index=False).agg(
        mse_null=("mse_null", "mean"),
        mse_signal=("mse_signal", "mean"),
        group_auroc=("group_auroc", "mean"),
        kappa_null_mean=("kappa_null_mean", "mean"),
        kappa_signal_mean=("kappa_signal_mean", "mean"),
        kappa_null_prob_gt_0_1=("kappa_null_prob_gt_0_1", "mean"),
        n_effective=("converged", "sum"),
    )
    partial_summary = partial_raw.groupby(["setting_id", "alpha_kappa", "beta_kappa"], as_index=False).agg(
        mse_null=("mse_null", "mean"),
        mse_signal=("mse_signal", "mean"),
        group_auroc=("group_auroc", "mean"),
        kappa_null_mean=("kappa_null_mean", "mean"),
        kappa_signal_mean=("kappa_signal_mean", "mean"),
        kappa_null_prob_gt_0_1=("kappa_null_prob_gt_0_1", "mean"),
        n_effective=("converged", "sum"),
    ) if not partial_raw.empty else pd.DataFrame()

    default_key = _prior_key(*(default_prior or (0.5, 1.0)))
    paired_delta_rows: list[dict[str, Any]] = []
    for sid in sorted(set(int(v) for v in paired_raw["setting_id"].tolist())):
        sub = paired_raw.loc[paired_raw["setting_id"].astype(int) == int(sid)].copy()
        for metric in ["mse_null", "mse_signal", "group_auroc"]:
            wide = sub.pivot_table(index="replicate_id", columns="prior_key", values=metric, aggfunc="mean")
            if default_key not in wide.columns:
                continue
            for pk in [c for c in wide.columns if str(c) != default_key]:
                diff = (wide[pk] - wide[default_key]).dropna()
                n_eff = int(diff.shape[0])
                if n_eff == 0:
                    continue
                mean_v = float(diff.mean())
                sd_v = float(diff.std(ddof=1)) if n_eff > 1 else float("nan")
                se_v = float(sd_v / np.sqrt(n_eff)) if n_eff > 1 else float("nan")
                ci_lo = float(mean_v - 1.96 * se_v) if np.isfinite(se_v) else float("nan")
                ci_hi = float(mean_v + 1.96 * se_v) if np.isfinite(se_v) else float("nan")
                a_v, b_v = [float(x) for x in str(pk).split("|")]
                paired_delta_rows.append(
                    {
                        "setting_id": int(sid),
                        "metric": metric,
                        "prior_key": str(pk),
                        "alpha_kappa": float(a_v),
                        "beta_kappa": float(b_v),
                        "contrast": "prior - default(0.5,1.0)",
                        "mean_diff": mean_v,
                        "std_diff": sd_v,
                        "se_diff": se_v,
                        "ci95_lo": ci_lo,
                        "ci95_hi": ci_hi,
                        "n_effective_pairs": n_eff,
                    }
                )
    delta_df = pd.DataFrame(paired_delta_rows)

    save_dataframe(raw, out_dir / "raw_results.csv")
    _record_produced_paths(produced, out_dir / "raw_results.csv")
    save_dataframe(summary, out_dir / "summary.csv")
    _record_produced_paths(produced, out_dir / "summary.csv")
    save_dataframe(summary, out_dir / "summary_paired.csv")
    _record_produced_paths(produced, out_dir / "summary_paired.csv")
    save_dataframe(partial_summary, out_dir / "summary_partial.csv")
    _record_produced_paths(produced, out_dir / "summary_partial.csv")
    save_dataframe(delta_df, out_dir / "prior_pairwise_delta.csv")
    _record_produced_paths(produced, out_dir / "prior_pairwise_delta.csv")
    table_summary = summary if not summary.empty else partial_summary
    save_dataframe(table_summary, tab_dir / "table_prior_sensitivity.csv")
    _record_produced_paths(produced, tab_dir / "table_prior_sensitivity.csv")
    save_json(
        {
            "prior_grid": [list(p) for p in priors],
            "retained_prior_grid": [list(p) for p in retained_priors],
            "scenarios": [[s, g, m] for s, g, m in scenarios],
            "bayes_min_chains": int(bayes_min_chains_use),
            "method_jobs": int(method_jobs),
            "paired_stats": paired_stats.to_dict(orient="records"),
            "pairing_note": "summary.csv uses paired-converged subset across all priors per setting",
            "partial_pairing_note": "summary_partial.csv uses all converged rows without requiring common replicate support across priors",
            "screen_priors": bool(screen_priors),
            "screen_max_retries": int(max(0, screen_max_retries)),
        },
        out_dir / "exp5_meta.json",
    )
    _record_produced_paths(produced, out_dir / "exp5_meta.json")

    try:
        from .analysis.plotting import plot_exp5_prior_sensitivity

        plot_source = summary if not summary.empty else partial_summary
        plot_exp5_prior_sensitivity(plot_source, out_dir=base / "figures")
        _record_produced_paths(produced, base / "figures" / "fig5_prior_sensitivity.png", base / "figures" / "fig5b_kappa_separation.png")
    except Exception as exc:
        log.warning("Plot exp5 failed: %s", exc)

    log.info("Exp5 done: %d rows", len(rows))
    result_paths = {
        "raw": str(out_dir / "raw_results.csv"),
        "summary": str(out_dir / "summary.csv"),
        "summary_paired": str(out_dir / "summary_paired.csv"),
        "summary_partial": str(out_dir / "summary_partial.csv"),
        "prior_pairwise_delta": str(out_dir / "prior_pairwise_delta.csv"),
        "table": str(tab_dir / "table_prior_sensitivity.csv"),
        "meta": str(out_dir / "exp5_meta.json"),
        "fig5_prior_sensitivity": str(base / "figures" / "fig5_prior_sensitivity.png"),
        "fig5b_kappa_separation": str(base / "figures" / "fig5b_kappa_separation.png"),
    }
    if (out_dir / "screening_summary.csv").exists():
        result_paths["screening_summary"] = str(out_dir / "screening_summary.csv")
    return _finalize_experiment_run(
        exp_key="exp5",
        save_dir=save_dir,
        results_dir=out_dir,
        produced_paths=produced,
        result_paths=result_paths,
        skip_run_analysis=bool(skip_run_analysis),
        archive_artifacts=bool(archive_artifacts),
    )




