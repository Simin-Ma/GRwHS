from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import math
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from .evaluation import _bridge_ratio_diagnostics, _kappa_group_means
from .fitting import _fit_with_convergence_retry
from .reporting import _finalize_experiment_run, _record_produced_paths, _stable_name_seed
from .runtime import (
    _attempts_used,
    bayes_rhat_threshold_default,
    _parallel_rows,
    _resolve_convergence_retry_limit,
    _result_diag_fields,
    _sampler_for_exp4,
    _sampler_for_standard,
)
from ..utils import (
    MASTER_SEED,
    SamplerConfig,
    ensure_dir,
    experiment_seed,
    load_pandas,
    method_result_label,
    print_experiment_result,
    rhs_style_tau0,
    save_dataframe,
    save_json,
    setup_logger,
)


def _smoke_sampler_for_exp4(base: SamplerConfig) -> SamplerConfig:
    return SamplerConfig(
        chains=1,
        warmup=min(int(base.warmup), 200),
        post_warmup_draws=min(int(base.post_warmup_draws), 200),
        adapt_delta=min(float(base.adapt_delta), 0.90),
        max_treedepth=min(int(base.max_treedepth), 10),
        strict_adapt_delta=min(float(base.strict_adapt_delta), 0.95),
        strict_max_treedepth=min(int(base.strict_max_treedepth), 12),
        max_divergence_ratio=float(base.max_divergence_ratio),
        rhat_threshold=max(1.02, bayes_rhat_threshold_default(), float(base.rhat_threshold)),
        ess_threshold=min(150.0, float(base.ess_threshold)),
    )


def _exp4_worker(
    task: tuple[int, int, int, list[int], SamplerConfig, dict[str, dict], int, int, bool, int, int, str]
) -> list[dict[str, Any]]:
    from .methods.fit_gr_rhs import fit_gr_rhs
    from .methods.fit_rhs import fit_rhs
    from ..utils import canonical_groups, sample_correlated_design

    p0_true, r, seed, group_sizes, sampler, variants, bayes_min_chains, method_jobs, enforce_conv, max_retries, n, log_path = task
    p = int(sum(group_sizes))
    s = experiment_seed(4, int(p0_true), r, master_seed=seed)

    # DGP: mixed strong/weak signals, randomly placed.
    # Here p0_true is the true active COEFFICIENT count.
    X, _cov_x = sample_correlated_design(n=n, group_sizes=group_sizes, rho_within=0.8, rho_between=0.2, seed=s)
    groups = canonical_groups(group_sizes)
    rng = np.random.default_rng(s + 19)
    beta = np.zeros(p, dtype=float)
    active = rng.choice(np.arange(p), size=int(p0_true), replace=False)
    n_strong = max(1, int(math.ceil(0.5 * int(p0_true))))
    beta[active[:n_strong]] = 2.0
    if active[n_strong:].size > 0:
        beta[active[n_strong:]] = 0.5
    y = X @ beta + np.random.default_rng(s + 23).normal(0.0, 1.0, n)
    group_has_signal = np.array([np.any(np.abs(beta[g]) > 0.1) for g in groups], dtype=bool)
    g_true_active = int(np.sum(group_has_signal))

    # Oracle tau for reference.
    tau0_oracle = rhs_style_tau0(n=n, p=p, p0=int(p0_true))

    def _fit_variant(item: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        vname, spec = item
        method = str(spec["method"])
        if method == "GR_RHS":
            res = _fit_with_convergence_retry(
                lambda st, att, _resume=None, _s=spec, _s_val=s, _vn=vname: fit_gr_rhs(
                    X,
                    y,
                    groups,
                    task="gaussian",
                    seed=_s_val + 31 + _stable_name_seed(_vn, mod=1000) + 100 * att,
                    p0=int(_s.get("p0_for_fit", p0_true)),
                    sampler=st,
                    auto_calibrate_tau=bool(_s.get("auto_calibrate_tau", False)),
                    tau0=_s.get("tau0"),
                    alpha_kappa=float(_s.get("alpha_kappa", 0.5)),
                    beta_kappa=float(_s.get("beta_kappa", 1.0)),
                    use_local_scale=bool(_s.get("use_local_scale", True)),
                    shared_kappa=bool(_s.get("shared_kappa", False)),
                    retry_resume_payload=_resume,
                    retry_attempt=int(att),
                ),
                method="GR_RHS",
                sampler=sampler,
                bayes_min_chains=int(bayes_min_chains),
                max_convergence_retries=max_retries,
                enforce_bayes_convergence=bool(enforce_conv),
                continue_on_retry=True,
            )
        else:  # RHS baseline, now unified as the single Stan/HMC rstanarm-style implementation
            res = _fit_with_convergence_retry(
                lambda st, att, _resume=None, _s_val=s: fit_rhs(
                    X,
                    y,
                    groups,
                    task="gaussian",
                    seed=_s_val + 32 + 100 * att,
                    p0=int(p0_true),
                    sampler=st,
                ),
                method="RHS",
                sampler=sampler,
                bayes_min_chains=int(bayes_min_chains),
                max_convergence_retries=max_retries,
                enforce_bayes_convergence=bool(enforce_conv),
            )
        is_valid = bool(res.beta_mean is not None)
        from .analysis.metrics import mse_null_signal_overall

        mse_metrics: dict[str, float] = {"mse_null": float("nan"), "mse_signal": float("nan"), "mse_overall": float("nan")}
        tau_post_mean = float("nan")
        kappa_null_mean = float("nan")
        kappa_signal_mean = float("nan")
        if is_valid:
            mse_metrics = mse_null_signal_overall(res.beta_mean, beta)
            if res.tau_draws is not None:
                tau_post_mean = float(np.mean(np.asarray(res.tau_draws, dtype=float)))
            if res.kappa_draws is not None:
                n_groups = len(group_sizes)
                km = _kappa_group_means(res, n_groups)
                kms = np.array(km)
                kappa_null_mean = float(np.nanmean(kms[~group_has_signal])) if np.any(~group_has_signal) else float("nan")
                kappa_signal_mean = float(np.nanmean(kms[group_has_signal])) if np.any(group_has_signal) else float("nan")
        row = {
            "p0_true": int(p0_true),
            "s_true_active_coeff": int(p0_true),
            "g_true_active": int(g_true_active),
            "p": p,
            "n": n,
            "replicate_id": r,
            "variant": vname,
            "method_type": method,
            "variant_label": method_result_label(vname),
            "method_type_label": method_result_label(method),
            "status": res.status,
            "converged": bool(res.converged),
            "fit_attempts": _attempts_used(res),
            "tau0_oracle": float(tau0_oracle),
            "tau_post_mean": tau_post_mean,
            "tau_ratio_to_oracle": float(tau_post_mean / max(tau0_oracle, 1e-12)) if np.isfinite(tau_post_mean) else float("nan"),
            "kappa_null_mean": kappa_null_mean,
            "kappa_signal_mean": kappa_signal_mean,
            **_result_diag_fields(res),
            **_bridge_ratio_diagnostics(
                res,
                groups=groups,
                X=X,
                y=y,
                signal_group_mask=group_has_signal,
            ),
            **mse_metrics,
        }
        print_experiment_result(
            "Exp4",
            row,
            context_keys=["p0_true", "replicate_id", "variant", "method_type"],
            metric_keys=["mse_overall", "mse_null", "mse_signal", "tau_ratio_to_oracle"],
            log_path=log_path,
        )
        return row

    variant_items = list(variants.items())
    workers = max(1, min(int(method_jobs), len(variant_items)))
    if workers <= 1 or len(variant_items) <= 1:
        return [_fit_variant(item) for item in variant_items]

    done: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_map = {ex.submit(_fit_variant, item): str(item[0]) for item in variant_items}
        for fut in as_completed(fut_map):
            row = fut.result()
            done[str(row["variant"])] = row
    return [done[str(vname)] for vname, _ in variant_items]


def run_exp4_variant_ablation(
    n_jobs: int = 1,
    method_jobs: int = 1,
    seed: int = MASTER_SEED,
    repeats: int = 10,
    save_dir: str = "outputs/simulation_project",
    *,
    skip_run_analysis: bool = False,
    archive_artifacts: bool = True,
    p0_list: Sequence[int] | None = None,
    include_oracle: bool = True,
    bayes_min_chains: int | None = None,
    enforce_bayes_convergence: bool = True,
    max_convergence_retries: int | None = None,
    until_bayes_converged: bool = True,
) -> Dict[str, str]:
    """
    Exp4: GR-RHS variant ablation - tau calibration strategies.

    Tests the 0415 tau calibration formula: tau0 = p0/(p-p0)/sqrt(n).
    Primary goal is predictive/mechanistic robustness of the calibrated variant
    against a misspecified tau (x10) and the RHS oracle baseline.

    Variants:
      calibrated: auto_calibrate=True  (0415 formula with estimated p0)
      fixed_10x:  tau0 * 10  (over-permissive, nulls not shrunk enough)
      RHS_oracle: unified Stan/HMC RHS with oracle p0 (baseline without kappa_g layer)
      oracle:     optional; enable with include_oracle=True for full ablation

    Note: p0 here denotes active coefficients (sparsity in coefficients).
    DGP defaults: p=50 (5 groups of 10), n=100, rho_within=0.8, rho_between=0.2.
    Default: p0 in {5, 15, 30}, include_oracle=True, repeats=10, retries=1.
    Sampler protocol:
      GR-RHS uses the default routing from fit_gr_rhs.
      RHS_oracle uses the single Stan/HMC RHS baseline from fit_rhs.
    """
    pd = load_pandas()
    produced: set[Path] = set()

    base = Path(save_dir)
    out_dir = ensure_dir(base / "results" / "exp4_variant_ablation")
    tab_dir = ensure_dir(base / "tables")
    log = setup_logger("exp4", base / "logs" / "exp4_variant_ablation.log")
    log_path = str(base / "logs" / "exp4_variant_ablation.log")

    sampler = _sampler_for_exp4(_sampler_for_standard())
    bayes_min_chains_use = int(bayes_min_chains) if bayes_min_chains is not None else 2
    if not bool(enforce_bayes_convergence):
        sampler = _smoke_sampler_for_exp4(sampler)
        if bayes_min_chains is None:
            bayes_min_chains_use = 1
    bayes_min_chains_use = max(1, int(bayes_min_chains_use))
    retry_limit = _resolve_convergence_retry_limit(
        max_convergence_retries,
        until_bayes_converged=bool(until_bayes_converged),
    )
    group_sizes = [10, 10, 10, 10, 10]
    p = int(sum(group_sizes))
    n = 100
    p0_vals = list(p0_list or [5, 15, 30])

    def _variants_for_p0(p0_true: int) -> dict[str, dict]:
        tau0_oracle = rhs_style_tau0(n=n, p=p, p0=p0_true)
        variants: dict[str, dict] = {
            "calibrated": {"method": "GR_RHS", "auto_calibrate_tau": True, "tau0": None, "p0_for_fit": p0_true, "use_local_scale": True},
            "fixed_10x": {"method": "GR_RHS", "auto_calibrate_tau": False, "tau0": tau0_oracle * 10.0, "p0_for_fit": p0_true, "use_local_scale": True},
            "RHS_oracle": {"method": "RHS"},
        }
        if bool(include_oracle):
            variants["oracle"] = {
                "method": "GR_RHS",
                "auto_calibrate_tau": False,
                "tau0": tau0_oracle,
                "p0_for_fit": p0_true,
                "use_local_scale": True,
            }
        return variants

    tasks: list[tuple] = []
    for p0_v in p0_vals:
        variants = _variants_for_p0(int(p0_v))
        for r in range(1, int(repeats) + 1):
            tasks.append(
                (
                    int(p0_v),
                    r,
                    seed,
                    group_sizes,
                    sampler,
                    variants,
                    int(bayes_min_chains_use),
                    int(method_jobs),
                    bool(enforce_bayes_convergence),
                    int(retry_limit),
                    n,
                    log_path,
                )
            )

    n_variants = len(_variants_for_p0(int(p0_vals[0]))) if p0_vals else 0
    log.info(
        "Exp4: %d p0 levels x %d repeats = %d tasks; variants per task=%d; retries=%d",
        len(p0_vals),
        repeats,
        len(tasks),
        n_variants,
        int(retry_limit),
    )
    all_chunks = _parallel_rows(tasks, _exp4_worker, n_jobs=n_jobs, prefer_process=True, process_fallback="serial", progress_desc="Exp4 Variant Ablation")
    rows: list[dict] = []
    for chunk in all_chunks:
        rows.extend(chunk)

    raw = pd.DataFrame(rows)
    if not raw.empty:
        if "variant" in raw.columns:
            raw["variant_label"] = raw["variant"].map(method_result_label)
        if "method_type" in raw.columns:
            raw["method_type_label"] = raw["method_type"].map(method_result_label)
    valid_mask = raw["converged"].fillna(False).astype(bool) & raw["status"].astype(str).str.lower().eq("ok")
    summary = raw.loc[valid_mask].groupby(["p0_true", "variant"], as_index=False).agg(
        mse_null=("mse_null", "mean"),
        mse_signal=("mse_signal", "mean"),
        mse_overall=("mse_overall", "mean"),
        mse_overall_std=("mse_overall", "std"),
        tau0_oracle=("tau0_oracle", "first"),
        tau_post_mean=("tau_post_mean", "mean"),
        tau_ratio_to_oracle=("tau_ratio_to_oracle", "mean"),
        kappa_null_mean=("kappa_null_mean", "mean"),
        kappa_signal_mean=("kappa_signal_mean", "mean"),
        g_true_active_mean=("g_true_active", "mean"),
        n_effective=("converged", "sum"),
    )
    summary["variant_label"] = summary["variant"].map(method_result_label)
    summary["mse_overall_sem"] = summary["mse_overall_std"] / np.sqrt(np.maximum(summary["n_effective"], 1))
    summary["kappa_gap"] = summary["kappa_signal_mean"] - summary["kappa_null_mean"]

    rhs_ref = summary.loc[summary["variant"] == "RHS_oracle", ["p0_true", "mse_overall"]].rename(columns={"mse_overall": "mse_rhs_oracle"})
    summary = summary.merge(rhs_ref, on="p0_true", how="left")
    denom = summary["mse_rhs_oracle"].astype(float)
    summary["mse_rel_rhs_oracle"] = np.where(np.isfinite(denom) & (denom > 0), summary["mse_overall"] / denom, np.nan)
    summary["mse_delta_rhs_oracle_pct"] = (summary["mse_rel_rhs_oracle"] - 1.0) * 100.0

    save_dataframe(raw, out_dir / "raw_results.csv")
    _record_produced_paths(produced, out_dir / "raw_results.csv")
    save_dataframe(summary, out_dir / "summary.csv")
    _record_produced_paths(produced, out_dir / "summary.csv")
    save_dataframe(summary, tab_dir / "table_variant_ablation.csv")
    _record_produced_paths(produced, tab_dir / "table_variant_ablation.csv")
    save_json(
        {
            "p0_vals": p0_vals,
            "group_sizes": group_sizes,
            "n": n,
            "include_oracle": bool(include_oracle),
            "bayes_min_chains": int(bayes_min_chains_use),
            "method_jobs": int(method_jobs),
            "max_convergence_retries": int(retry_limit),
        },
        out_dir / "exp4_meta.json",
    )
    _record_produced_paths(produced, out_dir / "exp4_meta.json")

    try:
        from .analysis.plotting import plot_exp4_ablation

        plot_exp4_ablation(summary, out_dir=base / "figures")
        _record_produced_paths(produced, base / "figures" / "fig4a_tau_scatter.png", base / "figures" / "fig4b_mse_normalized.png")
    except Exception as exc:
        log.warning("Plot exp4 failed: %s", exc)

    log.info("Exp4 done: %d rows", len(rows))
    result_paths = {
        "raw": str(out_dir / "raw_results.csv"),
        "summary": str(out_dir / "summary.csv"),
        "table": str(tab_dir / "table_variant_ablation.csv"),
        "meta": str(out_dir / "exp4_meta.json"),
        "fig4a_tau_scatter": str(base / "figures" / "fig4a_tau_scatter.png"),
        "fig4b_mse_normalized": str(base / "figures" / "fig4b_mse_normalized.png"),
    }
    return _finalize_experiment_run(
        exp_key="exp4",
        save_dir=save_dir,
        results_dir=out_dir,
        produced_paths=produced,
        result_paths=result_paths,
        skip_run_analysis=bool(skip_run_analysis),
        archive_artifacts=bool(archive_artifacts),
    )
