from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from ...infrastructure.evaluation import _kappa_group_means
from ...infrastructure.fitting import _fit_with_convergence_retry
from ...infrastructure.reporting import _finalize_experiment_run, _record_produced_paths
from ...infrastructure.runtime import (
    _BAYESIAN_DEFAULT_CHAINS,
    _attempts_used,
    _normalize_compute_profile,
    _parallel_rows,
    _resolve_convergence_retry_limit,
    _result_diag_fields,
    _sampler_for_exp5,
    _sampler_for_profile,
)
from ...utils import (
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


def _exp5_worker(
    task: tuple[int, int, list[int], list[float], int, SamplerConfig, list[tuple[float, float]], int, bool, int, str]
) -> list[dict[str, Any]]:
    from ...dgp_grouped_linear import generate_heterogeneity_dataset
    from ...fit_gr_rhs import fit_gr_rhs
    from ...metrics import group_auroc, group_l2_score

    sid, r, group_sizes, mu, seed, sampler, prior_grid, bayes_min_chains, enforce_conv, max_retries, backend = task
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
    rows: list[dict[str, Any]] = []
    for pid, (alpha_k, beta_k) in enumerate(prior_grid, start=1):
        res = _fit_with_convergence_retry(
            lambda st, att, _a=alpha_k, _b=beta_k, _s=s, _pid=pid, _be=backend: fit_gr_rhs(
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
            ),
            method="GR_RHS",
            sampler=sampler,
            bayes_min_chains=int(bayes_min_chains),
            max_convergence_retries=max_retries,
            enforce_bayes_convergence=bool(enforce_conv),
        )
        is_valid = bool(res.beta_mean is not None)
        mse_null = float("nan")
        mse_signal = float("nan")
        auroc = float("nan")
        kappa_null_mean = float("nan")
        kappa_signal_mean = float("nan")
        kappa_null_prob_gt_0_1 = float("nan")
        if is_valid:
            from ...metrics import group_auroc, group_l2_score, mse_null_signal_overall

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
        rows.append(
            {
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
            }
        )
    return rows


def run_exp5_prior_sensitivity(
    n_jobs: int = 1,
    seed: int = MASTER_SEED,
    repeats: int = 30,
    save_dir: str = "simulation_project",
    *,
    prior_grid: Sequence[tuple[float, float]] | None = None,
    profile: str = "full",
    bayes_min_chains: int | None = None,
    enforce_bayes_convergence: bool = True,
    max_convergence_retries: int | None = None,
    until_bayes_converged: bool = True,
    sampler_backend: str = "nuts",
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

    profile_name = _normalize_compute_profile(profile)
    sampler = _sampler_for_exp5(_sampler_for_profile(profile_name), profile=profile_name)
    bayes_min_chains_use = int(bayes_min_chains) if bayes_min_chains is not None else (2 if profile_name == "laptop" else int(_BAYESIAN_DEFAULT_CHAINS))
    bayes_min_chains_use = max(1, int(bayes_min_chains_use))
    retry_limit = _resolve_convergence_retry_limit(profile_name, max_convergence_retries, until_bayes_converged=bool(until_bayes_converged))
    if max_convergence_retries is None and retry_limit < 0:
        # Exp5 is intentionally heavy; cap unlimited mode to a practical retry budget.
        retry_limit = 3
    priors = list(prior_grid or _DEFAULT_PRIOR_GRID)

    scenarios: list[tuple[int, list[int], list[float]]] = [
        (1, [10, 10, 10, 10, 10], [0.0, 0.0, 1.5, 4.0, 10.0]),  # G10x5: p=50, 2 null + 3 signal
        (2, [30, 10, 5, 3, 2], [0.0, 0.0, 1.5, 4.0, 10.0]),  # CL:    p=50, 2 null + 3 signal
    ]

    tasks: list[tuple] = []
    for scen_id, grp_sizes, mu in scenarios:
        for r in range(1, int(repeats) + 1):
            tasks.append((scen_id, r, grp_sizes, mu, seed, sampler, priors, int(bayes_min_chains_use), bool(enforce_bayes_convergence), int(retry_limit), str(sampler_backend)))

    log.info("Exp5: %d scenarios x %d repeats x %d priors = %d task-rows", len(scenarios), repeats, len(priors), len(tasks) * len(priors))
    all_chunks = _parallel_rows(tasks, _exp5_worker, n_jobs=n_jobs, prefer_process=True, process_fallback="serial", progress_desc="Exp5 Prior Sensitivity")
    rows: list[dict] = []
    for chunk in all_chunks:
        rows.extend(chunk)

    raw = pd.DataFrame(rows)
    summary = raw.loc[raw["converged"]].groupby(["setting_id", "alpha_kappa", "beta_kappa"], as_index=False).agg(
        mse_null=("mse_null", "mean"),
        mse_signal=("mse_signal", "mean"),
        group_auroc=("group_auroc", "mean"),
        kappa_null_mean=("kappa_null_mean", "mean"),
        kappa_signal_mean=("kappa_signal_mean", "mean"),
        kappa_null_prob_gt_0_1=("kappa_null_prob_gt_0_1", "mean"),
        n_effective=("converged", "sum"),
    )

    save_dataframe(raw, out_dir / "raw_results.csv")
    save_dataframe(summary, out_dir / "summary.csv")
    save_dataframe(summary, tab_dir / "table_prior_sensitivity.csv")
    _record_produced_paths(produced, tab_dir / "table_prior_sensitivity.csv")
    save_json({"profile": profile_name, "prior_grid": [list(p) for p in priors], "scenarios": [[s, g, m] for s, g, m in scenarios], "bayes_min_chains": int(bayes_min_chains_use)}, out_dir / "exp5_meta.json")
    _record_produced_paths(produced, out_dir / "exp5_meta.json")

    try:
        from ...plotting import plot_exp5_prior_sensitivity

        plot_exp5_prior_sensitivity(summary, out_dir=base / "figures")
        _record_produced_paths(produced, base / "figures" / "fig5_prior_sensitivity.png", base / "figures" / "fig5b_kappa_separation.png")
    except Exception as exc:
        log.warning("Plot exp5 failed: %s", exc)

    log.info("Exp5 done: %d rows", len(rows))
    result_paths = {
        "raw": str(out_dir / "raw_results.csv"),
        "summary": str(out_dir / "summary.csv"),
        "table": str(tab_dir / "table_prior_sensitivity.csv"),
        "meta": str(out_dir / "exp5_meta.json"),
        "fig5_prior_sensitivity": str(base / "figures" / "fig5_prior_sensitivity.png"),
        "fig5b_kappa_separation": str(base / "figures" / "fig5b_kappa_separation.png"),
    }
    return _finalize_experiment_run(
        exp_key="exp5",
        save_dir=save_dir,
        results_dir=out_dir,
        produced_paths=produced,
        result_paths=result_paths,
    )

