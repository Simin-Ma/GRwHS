from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
from tqdm.auto import tqdm

from .dgp_normal_means import (
    generate_null_group,
    generate_signal_group_distributed,
    kappa_posterior_grid,
    posterior_summary_from_grid,
)
from .experiment_aliases import CLI_EXPERIMENT_CHOICES, cli_choice_to_key
from .experiment_eval import _evaluate_row, _kappa_group_means, _kappa_group_prob_gt
from .experiment_exp4 import run_exp4_variant_ablation
from .experiment_exp5 import run_exp5_prior_sensitivity
from .experiment_fitting import _fit_all_methods
from .utils import (
    MASTER_SEED,
    SamplerConfig,
    ensure_dir,
    experiment_seed,
    load_pandas,
    save_dataframe,
    save_json,
    setup_logger,
)

from .experiment_runtime import (
    COMPUTE_PROFILES,
    EXP3_GIGG_MODES,
    _BAYESIAN_DEFAULT_CHAINS,
    _attempts_used,
    _default_repeats,
    _exp3_gigg_config_for_mode,
    _gigg_config_for_profile,
    _is_bayesian_method,
    _normalize_compute_profile,
    _normalize_exp3_gigg_mode,
    _parallel_rows,
    _resolve_convergence_retry_limit,
    _resolve_sampler_backend_for_experiment,
    _result_diag_fields,
    _sampler_for_profile,
    xi_crit_u0_rho,
)
from .experiment_reporting import (
    _finalize_experiment_run,
    _paired_converged_subset,
    _record_produced_paths,
)

def _linreg_slope_ci(x: np.ndarray, y: np.ndarray) -> tuple[float, tuple[float, float]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    x0 = x - x.mean()
    beta1 = float(np.sum(x0 * (y - y.mean())) / np.sum(x0 * x0))
    beta0 = float(y.mean() - beta1 * x.mean())
    resid = y - (beta0 + beta1 * x)
    s2 = float(np.sum(resid * resid) / max(n - 2, 1))
    se = math.sqrt(s2 / max(np.sum(x0 * x0), 1e-12))
    return beta1, (beta1 - 1.96 * se, beta1 + 1.96 * se)

# ---------------------------------------------------------------------------
# EXP1 - kappa_g Profile Regimes
# ---------------------------------------------------------------------------
# Two panels from the 1-D profile posterior (profile specialization: lambda=1,
# a_g=1, tau fixed):
#   Panel A (null): sweep p_g, verify E[kappa_g | Y_null] = O(p_g^{-1/2})
#                   (Theorem 3.22)
#   Panel B (phase): sweep xi/xi_crit, verify P(kappa_g > u0) -> 1 for xi > xi_crit
#                   (Corollary 3.33, Theorem 3.32)
# ---------------------------------------------------------------------------

def _exp1_null_worker(task: tuple) -> dict[str, Any]:
    sid, pg, r, seed, tau, alpha_kappa, beta_kappa, tail_eps = task
    s = experiment_seed(1, sid, r, master_seed=seed)
    y = generate_null_group(pg=int(pg), sigma2=1.0, seed=s)
    grid = kappa_posterior_grid(y, tau=float(tau), sigma2=1.0, alpha_kappa=alpha_kappa, beta_kappa=beta_kappa)
    sm = posterior_summary_from_grid(grid["kappa"], grid["density"], tail_threshold=float(tail_eps))
    return {
        "panel": "null",
        "p_g": int(pg),
        "setting_id": int(sid),
        "replicate_id": int(r),
        "tau": float(tau),
        "alpha_kappa": float(alpha_kappa),
        "beta_kappa": float(beta_kappa),
        "post_mean_kappa": sm["post_mean_kappa"],
        "post_median_kappa": sm["post_median_kappa"],
        "tail_prob_kappa_gt_eps": sm.get("tail_prob", float("nan")),
    }


def _exp1_null_setting_worker(task: tuple) -> list[dict[str, Any]]:
    sid, pg, repeats, seed, tau, alpha_kappa, beta_kappa, tail_eps = task
    rows = []
    for r in range(1, int(repeats) + 1):
        s = experiment_seed(1, int(sid), r, master_seed=seed)
        y = generate_null_group(pg=int(pg), sigma2=1.0, seed=s)
        grid = kappa_posterior_grid(y, tau=float(tau), sigma2=1.0, alpha_kappa=alpha_kappa, beta_kappa=beta_kappa)
        sm = posterior_summary_from_grid(grid["kappa"], grid["density"], tail_threshold=float(tail_eps))
        rows.append({
            "panel": "null",
            "p_g": int(pg),
            "setting_id": int(sid),
            "replicate_id": int(r),
            "tau": float(tau),
            "alpha_kappa": float(alpha_kappa),
            "beta_kappa": float(beta_kappa),
            "post_mean_kappa": sm["post_mean_kappa"],
            "post_median_kappa": sm["post_median_kappa"],
            "tail_prob_kappa_gt_eps": sm.get("tail_prob", float("nan")),
        })
    return rows


def _exp1_phase_setting_worker(task: tuple) -> list[dict[str, Any]]:
    sid, pg, xid, xi, repeats, seed, tau, sigma2, u0, alpha_kappa, beta_kappa = task
    rows = []
    mu_g = float(xi) * int(pg)
    xi_c = xi_crit_u0_rho(u0=float(u0), rho=float(tau) / math.sqrt(max(float(sigma2), 1e-12)))
    for r in range(1, int(repeats) + 1):
        s = experiment_seed(1, int(sid) * 100 + int(xid), r, master_seed=seed)
        y, _ = generate_signal_group_distributed(pg=int(pg), mu_g=float(mu_g), sigma2=float(sigma2), seed=s)
        grid = kappa_posterior_grid(y, tau=float(tau), sigma2=float(sigma2), alpha_kappa=alpha_kappa, beta_kappa=beta_kappa)
        sm = posterior_summary_from_grid(grid["kappa"], grid["density"], tail_threshold=float(u0))
        rows.append({
            "panel": "phase",
            "p_g": int(pg),
            "setting_id": int(sid),
            "replicate_id": int(r),
            "tau": float(tau),
            "xi": float(xi),
            "xi_crit": float(xi_c),
            "xi_ratio": float(xi) / max(xi_c, 1e-12),
            "u0": float(u0),
            "alpha_kappa": float(alpha_kappa),
            "beta_kappa": float(beta_kappa),
            "post_mean_kappa": sm["post_mean_kappa"],
            "post_prob_kappa_gt_u0": sm.get("tail_prob", float("nan")),
        })
    return rows


def _exp1_phase_worker(task: tuple) -> dict[str, Any]:
    sid, pg, xid, xi, r, seed, tau, sigma2, u0, alpha_kappa, beta_kappa = task
    mu_g = float(xi) * int(pg)
    xi_c = xi_crit_u0_rho(u0=float(u0), rho=float(tau) / math.sqrt(max(float(sigma2), 1e-12)))
    s = experiment_seed(1, int(sid) * 100 + int(xid), r, master_seed=seed)
    y, _ = generate_signal_group_distributed(pg=int(pg), mu_g=float(mu_g), sigma2=float(sigma2), seed=s)
    grid = kappa_posterior_grid(y, tau=float(tau), sigma2=float(sigma2), alpha_kappa=alpha_kappa, beta_kappa=beta_kappa)
    sm = posterior_summary_from_grid(grid["kappa"], grid["density"], tail_threshold=float(u0))
    return {
        "panel": "phase",
        "p_g": int(pg),
        "setting_id": int(sid),
        "replicate_id": int(r),
        "tau": float(tau),
        "xi": float(xi),
        "xi_crit": float(xi_c),
        "xi_ratio": float(xi) / max(xi_c, 1e-12),
        "u0": float(u0),
        "alpha_kappa": float(alpha_kappa),
        "beta_kappa": float(beta_kappa),
        "post_mean_kappa": sm["post_mean_kappa"],
        "post_prob_kappa_gt_u0": sm.get("tail_prob", float("nan")),
    }


def run_exp1_kappa_profile_regimes(
    n_jobs: int = 1,
    seed: int = MASTER_SEED,
    repeats: int = 500,
    save_dir: str = "simulation_project",
    *,
    # Panel A - null contraction
    pg_null_list: Sequence[int] | None = None,
    tau_null: float = 0.5,
    tail_eps: float = 0.1,
    # Panel B - phase diagram
    pg_phase_list: Sequence[int] | None = None,
    tau_phase_list: Sequence[float] | None = None,
    xi_multiplier_list: Sequence[float] | None = None,
    u0: float = 0.5,
    sigma2_phase: float = 1.0,
    # Shared prior
    alpha_kappa: float = 0.5,
    beta_kappa: float = 1.0,
) -> Dict[str, str]:
    """
    Exp1: kappa_g profile regimes (Theorems 3.22, 3.32, Corollary 3.33).

    Panel A - null contraction
      DGP: Y_j ~ N(0, sigma2), profile posterior under lambda=1, a_g=1, tau fixed.
      Validates E[kappa_g | Y_null] = O(p_g^{-1/2}): log-log slope should be -1/2.

    Panel B - phase diagram
      DGP: distributed signal Y_j ~ N(beta_val, 1), mu_g = xi * p_g.
      Sweeps xi/xi_crit across [0.3, 2.0]; P(kappa_g > u0 | Y) -> 1 iff xi > xi_crit.
      xi_crit = u0 * rho^2 / (2*(u0 + (1-u0)*rho^2)), eq. 104 of 0415 paper.
    """
    from .plotting import plot_exp1, plot_exp1_phase
    produced: set[Path] = set()

    base = Path(save_dir)
    out_dir = ensure_dir(base / "results" / "exp1_kappa_profile_regimes")
    fig_dir = ensure_dir(base / "figures")
    log = setup_logger("exp1", base / "logs" / "exp1_kappa_profile_regimes.log")

    pg_null = list(pg_null_list or [10, 20, 50, 100, 200, 500, 1000, 2000])
    pg_phase = list(pg_phase_list or [30, 60, 120, 240, 480])
    # tau=0.1 produces xi_crit around 0.005; signal is undetectable at finite p_g,
    # leaving flat P(kappa > u0) close to prior for all xi_ratio values.
    # Use [0.5, 0.7, 1.0, 1.5] so every tau produces a visible phase transition.
    tau_phase = list(tau_phase_list or [0.5, 0.7, 1.0, 1.5])
    xi_mults = list(xi_multiplier_list or [0.3, 0.5, 0.7, 0.85, 0.95, 1.05, 1.15, 1.3, 1.5, 2.0])

    # --- Panel A: null contraction ---
    log.info("Exp1 Panel A: null contraction, pg=%s, tau=%.2f", pg_null, tau_null)
    null_tasks: list[tuple] = []
    for sid, pg in enumerate(pg_null, start=1):
        for r in range(1, int(repeats) + 1):
            null_tasks.append((sid, pg, r, seed, tau_null, alpha_kappa, beta_kappa, tail_eps))
    null_rows = _parallel_rows(null_tasks, _exp1_null_worker, n_jobs=n_jobs, prefer_process=False, progress_desc="Exp1A Null")

    # Summary: median E[kappa_g|Y] per p_g
    null_agg: list[dict] = []
    pg_vals_seen = sorted({int(r["p_g"]) for r in null_rows})
    for pg in pg_vals_seen:
        sub = [r for r in null_rows if int(r["p_g"]) == pg]
        means = np.array([float(r["post_mean_kappa"]) for r in sub])
        tails = np.array([float(r.get("tail_prob_kappa_gt_eps", float("nan"))) for r in sub])
        null_agg.append({
            "p_g": pg,
            "median_post_mean_kappa": float(np.median(means)),
            "mean_post_mean_kappa": float(np.mean(means)),
            "q25_post_mean_kappa": float(np.quantile(means, 0.25)),
            "q75_post_mean_kappa": float(np.quantile(means, 0.75)),
            "std_post_mean_kappa": float(np.std(means, ddof=1)) if len(means) > 1 else float("nan"),
            "mean_tail_prob_kappa_gt_eps": float(np.nanmean(tails)),
            "n_replicates": len(means),
        })
    log_pg = np.log(np.array([r["p_g"] for r in null_agg], dtype=float))
    log_kappa = np.log(np.maximum(np.array([r["median_post_mean_kappa"] for r in null_agg], dtype=float), 1e-12))
    # Fit slope on the asymptotic regime p_g in [20, 500] only; p_g=10 is pre-asymptotic
    # and p_g>=1000 approaches the numerical floor, both bias the slope estimate.
    fit_mask = np.array([20 <= r["p_g"] <= 500 for r in null_agg])
    slope, slope_ci = _linreg_slope_ci(log_pg[fit_mask], log_kappa[fit_mask])
    log.info("Panel A log-log slope (p_g 20-500): %.4f (95%% CI [%.4f, %.4f]), expected -0.5", slope, slope_ci[0], slope_ci[1])

    # --- Panel B: phase diagram ---
    log.info("Exp1 Panel B: phase diagram, pg=%s, tau=%s, xi_mults=%s", pg_phase, tau_phase, xi_mults)
    phase_tasks: list[tuple] = []
    sid = 100  # offset from panel A
    for tau in tau_phase:
        xi_crit_ref = xi_crit_u0_rho(u0=float(u0), rho=float(tau) / math.sqrt(max(float(sigma2_phase), 1e-12)))
        for pg in pg_phase:
            sid += 1
            for xid, mult in enumerate(xi_mults, start=1):
                xi_val = xi_crit_ref * float(mult)
                for r in range(1, int(repeats) + 1):
                    phase_tasks.append((sid, pg, xid, xi_val, r, seed, tau, sigma2_phase, u0, alpha_kappa, beta_kappa))
    phase_rows = _parallel_rows(phase_tasks, _exp1_phase_worker, n_jobs=n_jobs, prefer_process=False, progress_desc="Exp1B Phase")

    # Phase summary: mean P(kappa > u0) by (tau, p_g, xi_ratio)
    phase_agg: list[dict] = []
    keys_seen = sorted({(float(r["tau"]), int(r["p_g"]), float(r["xi_ratio"])) for r in phase_rows})
    for tau_v, pg_v, xi_r in keys_seen:
        sub = [r for r in phase_rows if float(r["tau"]) == tau_v and int(r["p_g"]) == pg_v and abs(float(r["xi_ratio"]) - xi_r) < 1e-8]
        probs = np.array([float(r["post_prob_kappa_gt_u0"]) for r in sub])
        phase_agg.append({
            "tau": tau_v, "p_g": pg_v, "xi_ratio": xi_r,
            "xi_crit": float(sub[0]["xi_crit"]),
            "xi": float(sub[0]["xi"]),
            "mean_prob_kappa_gt_u0": float(np.mean(probs)),
            "n_replicates": len(probs),
        })

    # --- Save ---
    pd = load_pandas()
    all_rows = null_rows + phase_rows
    save_dataframe(pd.DataFrame(all_rows), out_dir / "raw_results.csv")
    _record_produced_paths(produced, out_dir / "raw_results.csv")
    save_dataframe(pd.DataFrame(null_agg), out_dir / "summary_null.csv")
    _record_produced_paths(produced, out_dir / "summary_null.csv")
    save_dataframe(pd.DataFrame(phase_agg), out_dir / "summary_phase.csv")
    _record_produced_paths(produced, out_dir / "summary_phase.csv")
    # Statistically correct criterion: does the 95% CI for slope contain the theoretical -0.5?
    # Also require the point estimate to be in a plausible range [-0.8, -0.25] to reject degenerate fits.
    _pass_ci_contains = slope_ci[0] < -0.5 < slope_ci[1]
    _pass_estimate = -0.8 < slope < -0.25
    save_json({"slope": slope, "slope_ci": list(slope_ci), "expected_slope": -0.5, "fit_range_pg": [20, 500], "ci_contains_theory": _pass_ci_contains, "pass": bool(_pass_ci_contains and _pass_estimate)}, out_dir / "null_slope_check.json")
    _record_produced_paths(produced, out_dir / "null_slope_check.json")

    try:
        plot_exp1(pd.DataFrame(null_agg), slope=slope, slope_ci=slope_ci, out_path=fig_dir / "fig1a_null_contraction.png")
        _record_produced_paths(produced, fig_dir / "fig1a_null_contraction.png")
    except Exception as exc:
        log.warning("Plot exp1A failed: %s", exc)
    try:
        plot_exp1_phase(pd.DataFrame(phase_agg), out_path=fig_dir / "fig1b_phase_diagram.png")
        _record_produced_paths(produced, fig_dir / "fig1b_phase_diagram.png")
    except Exception as exc:
        log.warning("Plot exp1B failed: %s", exc)

    log.info("Exp1 done: %d null rows, %d phase rows", len(null_rows), len(phase_rows))
    result_paths = {
        "null_raw": str(out_dir / "raw_results.csv"),
        "null_summary": str(out_dir / "summary_null.csv"),
        "phase_summary": str(out_dir / "summary_phase.csv"),
        "null_slope_check": str(out_dir / "null_slope_check.json"),
        "fig1a_null_contraction": str(fig_dir / "fig1a_null_contraction.png"),
        "fig1b_phase_diagram": str(fig_dir / "fig1b_phase_diagram.png"),
    }
    return _finalize_experiment_run(
        exp_key="exp1",
        save_dir=save_dir,
        results_dir=out_dir,
        produced_paths=produced,
        result_paths=result_paths,
    )

# ---------------------------------------------------------------------------
# EXP2 - Full-Model Group Separation
# ---------------------------------------------------------------------------
# Tests Theorem 3.34: simultaneous null contraction + signal retention in the
# full grouped horseshoe model.
#
# DGP (xi_crit-calibrated):
#   4 groups: [30, 20, 15, 10]
#   xi_ratios = [0.0, 1.0, 4.0, 8.0]
#   mu_g = xi_ratio * xi_crit * p_g
#   rho_ref controls xi_crit calibration (sigma2 = 1.0)
#   n_train = 200, n_test default = 50
#
# Key outputs:
#   - group-level null/signal MSE for all methods
#   - group AUROC for all methods
#   - MLPD_test for all methods
#   - posterior mean kappa_g per group for GR_RHS (direct mechanism check)
# ---------------------------------------------------------------------------

def _exp2_worker(
    task: tuple[int, int, list[int], list[float], list[float], SamplerConfig, list[str], dict[str, Any], int, bool, int, int, dict]
) -> tuple[list[dict], list[dict]]:
    from .dgp_grouped_linear import generate_heterogeneity_dataset
    from .metrics import group_auroc, group_l2_error, group_l2_score
    from .utils import sample_correlated_design

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
    save_dir: str = "simulation_project",
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
    _record_produced_paths(produced, out_dir / "raw_results.csv")
    _record_produced_paths(produced, out_dir / "raw_results.csv")
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
        from .plotting import plot_exp2_separation
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

# ---------------------------------------------------------------------------
# EXP3 - Linear Benchmark: Concentrated vs. Distributed vs. Boundary
# ---------------------------------------------------------------------------
# Factor design directly testing the core GR-RHS hypothesis:
#   "kappa_g mechanism is most beneficial when signals are group-concentrated"
#
# Factors (default core30 design):
#   signal_structure: concentrated / distributed / boundary
#     concentrated: 2/5 groups fully active, beta_j = 1/sqrt(p_g) (dense in group)
#     distributed:  2/5 groups with one active variable each, beta_j = 1
#     boundary:     2/5 groups active, beta calibrated at xi_ratio * xi_crit(u0, rho_profile),
#                   where rho_profile = rho_within / sqrt(sigma2_boundary)
#   env points:
#     E0        : (rw=0.3, rb=0.1, snr=1.0) [all signals]
#     RW_PLUS   : (rw=0.8, rb=0.1, snr=1.0) [all signals]
#
# Prediction:
#   concentrated + moderate/high rho: GR-RHS wins on null_group_mse
#   distributed: RHS matches GR-RHS (individual-level shrinkage is sufficient)
#   boundary: GR-RHS separates null/signal groups; competitors may fail
# ---------------------------------------------------------------------------

_BOUNDARY_U0 = 0.5
_BOUNDARY_XI_RATIO = 1.2
_SIGMA2_BOUNDARY = 1.0

# ---------------------------------------------------------------------------
# Default group configurations for Exp3 - mirrors GIGG paper Table 1 coverage,
# plus GR-RHS-favorable scenarios (large null blocks, rho_between > 0).
#
# Each entry:
#   name          - short label used in output CSV / meta JSON
#   group_sizes   - list of per-group sizes (sum = p)
#   active_groups - group indices containing signal (rest are null)
#
# G10x5 : 5 equal groups of size 10 (p=50)  - GIGG paper C10H/D10H baseline
# CL    : [30,10,5,3,2], signal in large groups - GIGG Table 3 CL/DL
# CS    : [30,10,5,3,2], signal in small groups - GR-RHS kappa_g advantage
#         (large null blocks enable strong collective contraction)
# ---------------------------------------------------------------------------
_DEFAULT_EXP3_GROUP_CONFIGS: list[dict[str, Any]] = [
    {"name": "G10x5", "group_sizes": [10, 10, 10, 10, 10],  "active_groups": [0, 1]},
    {"name": "CL",    "group_sizes": [30, 10, 5, 3, 2],     "active_groups": [0, 1]},
    {"name": "CS",    "group_sizes": [30, 10, 5, 3, 2],     "active_groups": [3, 4]},
]

_DEFAULT_EXP3_ENV_POINTS_CORE30: list[dict[str, Any]] = [
    {
        "env_id": "E0",
        "setting_block": "anchor",
        "rho_within": 0.3,
        "rho_between": 0.1,
        "target_snr": 1.0,
        "signals": ["concentrated", "distributed", "boundary"],
    },
    {
        "env_id": "RW_PLUS",
        "setting_block": "rw_axis",
        "rho_within": 0.8,
        "rho_between": 0.1,
        "target_snr": 1.0,
        "signals": ["concentrated", "distributed", "boundary"],
    },
]


def _exp3_keep_env_point_rw_gt_rb(ep: dict[str, Any]) -> bool:
    """Keep Exp3 environment point iff within-group correlation is strictly larger."""
    rw = float(ep.get("rho_within", float("nan")))
    rb = float(ep.get("rho_between", float("nan")))
    return bool(np.isfinite(rw) and np.isfinite(rb) and (rw > rb))


def _exp3_filter_env_points_rw_gt_rb(points: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for ep in points:
        epi = dict(ep)
        if _exp3_keep_env_point_rw_gt_rb(epi):
            kept.append(epi)
        else:
            dropped.append(epi)
    return kept, dropped


def _build_benchmark_beta(
    signal: str,
    group_sizes: Sequence[int],
    *,
    active_groups: Sequence[int] | None = None,
    sigma2: float = 1.0,
    p: int | None = None,
    boundary_u0: float = _BOUNDARY_U0,
    boundary_xi_ratio: float = _BOUNDARY_XI_RATIO,
    boundary_rho_profile: float | None = None,
) -> np.ndarray:
    """Construct beta for each benchmark signal structure.

    concentrated: all variables in active groups with equal weight (||beta_g||=1 per group)
    distributed:  first variable only in each active group (beta_j=1)
    boundary:     all vars in active groups, calibrated at xi_ratio * xi_crit(u0, rho_profile)
    """
    from .utils import canonical_groups
    groups = canonical_groups(group_sizes)
    total_p = int(p or sum(group_sizes))
    beta = np.zeros(total_p, dtype=float)
    _active = list(active_groups) if active_groups is not None else [0, 1]

    if signal == "concentrated":
        for gid in _active:
            idx = np.asarray(groups[gid], dtype=int)
            beta[idx] = 1.0 / math.sqrt(len(idx))
    elif signal == "distributed":
        for gid in _active:
            beta[groups[gid][0]] = 1.0
    elif signal == "boundary":
        if boundary_rho_profile is None:
            raise ValueError("boundary_rho_profile must be provided for boundary signal calibration.")
        xi_c = xi_crit_u0_rho(u0=float(boundary_u0), rho=float(boundary_rho_profile))
        for gid in _active:
            idx = np.asarray(groups[gid], dtype=int)
            pg = len(idx)
            mu_g = float(boundary_xi_ratio) * xi_c * pg
            beta_val = math.sqrt(2.0 * float(sigma2) * mu_g / pg)
            beta[idx] = beta_val
    else:
        raise ValueError(f"unknown signal structure: {signal!r}")
    return beta


def _exp3_worker(
    task: dict[str, Any] | tuple,
) -> list[dict[str, Any]]:
    from .dgp_grouped_linear import generate_orthonormal_block_design
    from .utils import canonical_groups, sample_correlated_design

    if isinstance(task, dict):
        sid = int(task["setting_id"])
        signal = str(task["signal"])
        group_cfg = dict(task["group_cfg"])
        setting_block = str(task["setting_block"])
        env_id = str(task["env_id"])
        design_type = str(task["design_type"])
        rho_within = float(task["rho_within"])
        rho_between = float(task["rho_between"])
        target_snr = float(task["target_snr"])
        r = int(task["replicate_id"])
        seed_base = int(task["seed_base"])
        n_test = int(task["n_test"])
        sampler = task["sampler"]
        method_name = str(task["method"])
        gigg_config = dict(task["gigg_config"])
        gigg_mode = str(task.get("gigg_mode", "stable"))
        bayes_min_chains = task.get("bayes_min_chains")
        enforce_conv = bool(task["enforce_bayes_convergence"])
        max_retries = int(task["max_convergence_retries"])
        grrhs_kwargs = dict(task["grrhs_kwargs"])
        methods = [method_name]
    else:
        sid, signal, group_cfg, setting_block, env_id, design_type, rho_within, rho_between, target_snr, r, seed_base, n_test, sampler, methods, gigg_config, bayes_min_chains, enforce_conv, max_retries, grrhs_kwargs = task
        gigg_mode = "stable"
    gigg_config = dict(gigg_config)
    gigg_mode_name = _normalize_exp3_gigg_mode(gigg_mode)
    if str(method_name if isinstance(task, dict) else (methods[0] if methods else "")).upper() == "GIGG_MMLE":
        if gigg_mode_name == "paper_ref":
            gigg_config["extra_retry"] = 0
            gigg_config.pop("retry_cap", None)
        else:
            hard_setting = (group_cfg_name := str(group_cfg["name"])) in {"CL", "G10x5"} and signal in {"concentrated", "distributed"}
            if hard_setting:
                gigg_config["extra_retry"] = max(1, int(gigg_config.get("extra_retry", 0)))
                # Keep rescue behavior efficient while preserving robustness.
                gigg_config["retry_cap"] = 2
                # Full profile defaults to paper-locked no_retry; for hard Exp3 settings
                # we allow one bounded rescue attempt to improve benchmark completeness.
                if bool(gigg_config.get("no_retry", False)):
                    gigg_config["no_retry"] = False
                gigg_config["progress_bar"] = bool(gigg_config.get("progress_bar", False))
                # For difficult Exp3 settings, prefer stronger mixing from the first attempt.
                gigg_config["randomize_group_order"] = bool(gigg_config.get("randomize_group_order", True))
                gigg_config["lambda_vectorized_update"] = bool(gigg_config.get("lambda_vectorized_update", True))
                gigg_config["extra_beta_refresh_prob"] = max(float(gigg_config.get("extra_beta_refresh_prob", 0.0)), 0.08)
                gigg_config["init_scale_blend"] = max(float(gigg_config.get("init_scale_blend", 0.5)), 0.65)
                # Damped MMLE updates reduce q_g oscillation in difficult correlated regimes.
                gigg_config["mmle_step_size"] = min(max(float(gigg_config.get("mmle_step_size", 0.6)), 0.0), 1.0)
    s = experiment_seed(3, int(sid), r, master_seed=int(seed_base))

    group_sizes: list[int] = list(group_cfg["group_sizes"])
    active_groups: list[int] = list(group_cfg["active_groups"])
    group_cfg_name: str = str(group_cfg["name"])
    n_train = 100

    sigma2_boundary = float(_SIGMA2_BOUNDARY)
    boundary_rho_profile = float(rho_within) / math.sqrt(max(sigma2_boundary, 1e-12))
    boundary_xi_crit = float("nan")
    boundary_xi = float("nan")
    if signal == "boundary":
        boundary_xi_crit = xi_crit_u0_rho(u0=float(_BOUNDARY_U0), rho=boundary_rho_profile)
        boundary_xi = float(_BOUNDARY_XI_RATIO) * boundary_xi_crit

    beta0 = _build_benchmark_beta(
        signal,
        group_sizes,
        active_groups=active_groups,
        sigma2=sigma2_boundary if signal == "boundary" else 1.0,
        boundary_u0=float(_BOUNDARY_U0),
        boundary_xi_ratio=float(_BOUNDARY_XI_RATIO),
        boundary_rho_profile=boundary_rho_profile if signal == "boundary" else None,
    )
    p = int(sum(group_sizes))

    # Construct training dataset
    if str(design_type) == "orthonormal":
        from .dgp_grouped_linear import generate_orthonormal_block_design
        X_train = generate_orthonormal_block_design(n=n_train, group_sizes=group_sizes, seed=s)
        cov_x = np.eye(p, dtype=float)
    else:
        X_train, cov_x = sample_correlated_design(n=n_train, group_sizes=group_sizes, rho_within=rho_within, rho_between=rho_between, seed=s)

    # sigma2: SNR-calibrated for concentrated/distributed; fixed for boundary
    if signal == "boundary":
        sigma2 = sigma2_boundary
    else:
        from .dgp_grouped_linear import sigma2_for_target_snr as _s2
        sigma2 = _s2(beta=beta0, cov_x=cov_x, target_snr=float(target_snr))

    rng_y = np.random.default_rng(s + 17)
    y_train = X_train @ beta0 + rng_y.normal(0.0, math.sqrt(sigma2), n_train)

    # Test set: fresh X and noise, same DGP parameters
    if str(design_type) == "orthonormal":
        X_test = generate_orthonormal_block_design(n=n_test, group_sizes=group_sizes, seed=s + 77777)
    else:
        X_test, _ = sample_correlated_design(n=n_test, group_sizes=group_sizes, rho_within=rho_within, rho_between=rho_between, seed=s + 77777)
    rng_yt = np.random.default_rng(s + 88888)
    y_test = X_test @ beta0 + rng_yt.normal(0.0, math.sqrt(sigma2), n_test)

    groups = canonical_groups(group_sizes)
    p0 = int(np.sum(np.abs(beta0) > 1e-12))
    p0_signal_groups = int(
        np.sum(
            [
                int(np.any(np.abs(beta0[np.asarray(g, dtype=int)]) > 1e-12))
                for g in groups
            ]
        )
    )
    n_groups = len(group_sizes)

    fits = _fit_all_methods(
        X_train, y_train, groups,
        task="gaussian", seed=s, p0=p0,
        p0_groups=p0_signal_groups,
        sampler=sampler, methods=methods, gigg_config=gigg_config,
        bayes_min_chains=int(bayes_min_chains) if bayes_min_chains is not None else None,
        grrhs_kwargs=grrhs_kwargs or {},
        enforce_bayes_convergence=bool(enforce_conv),
        max_convergence_retries=int(max_retries),
    )

    out_rows: list[dict[str, Any]] = []
    for method, res in fits.items():
        metrics = _evaluate_row(res, beta0, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        kappa_null_mean = float("nan")
        kappa_signal_mean = float("nan")
        if method == "GR_RHS" and res.beta_mean is not None:
            km = _kappa_group_means(res, n_groups)
            null_groups = [g for g in range(n_groups) if g not in set(active_groups)]
            _sig_vals = [km[g] for g in active_groups if not np.isnan(km[g])]
            _null_vals = [km[g] for g in null_groups if not np.isnan(km[g])]
            kappa_signal_mean = float(np.mean(_sig_vals)) if _sig_vals else float("nan")
            kappa_null_mean = float(np.mean(_null_vals)) if _null_vals else float("nan")
        out_rows.append({
            "setting_id": int(sid),
            "gigg_mode": str(gigg_mode_name),
            "group_config": group_cfg_name,
            "signal": signal,
            "setting_block": str(setting_block),
            "env_id": str(env_id),
            "design_type": str(design_type),
            "rho_within": float(rho_within),
            "rho_between": float(rho_between),
            "target_snr": float(target_snr),
            "sigma2": float(sigma2),
            "boundary_u0": float(_BOUNDARY_U0) if signal == "boundary" else float("nan"),
            "boundary_xi_ratio": float(_BOUNDARY_XI_RATIO) if signal == "boundary" else float("nan"),
            "boundary_rho_profile": boundary_rho_profile if signal == "boundary" else float("nan"),
            "boundary_xi_crit": boundary_xi_crit,
            "boundary_xi": boundary_xi,
            "replicate_id": int(r),
            "method": method,
            "status": res.status,
            "converged": bool(res.converged),
            "fit_attempts": _attempts_used(res),
            "kappa_null_mean": kappa_null_mean,
            "kappa_signal_mean": kappa_signal_mean,
            **_result_diag_fields(res),
            **metrics,
        })
    return out_rows


def run_exp3_linear_benchmark(
    n_jobs: int = 1,
    seed: int = MASTER_SEED,
    repeats: int = 20,
    save_dir: str = "simulation_project",
    *,
    signal_types: Sequence[str] | None = None,
    rho_within_values: Sequence[float] | None = None,
    snr_values: Sequence[float] | None = None,
    rho_between: float = 0.1,
    exp3_design: str = "core30",
    env_points: Sequence[dict[str, Any]] | None = None,
    bayes_min_chains: int | None = None,
    group_configs: list[dict[str, Any]] | None = None,
    profile: str = "full",
    methods: Sequence[str] | None = None,
    enforce_bayes_convergence: bool = True,
    max_convergence_retries: int | None = None,
    until_bayes_converged: bool = True,
    n_test: int = 30,
    sampler_backend: str = "nuts",
    grrhs_extra_kwargs: dict | None = None,
    gigg_mode: str = "stable",
    result_dir_name: str = "exp3_linear_benchmark",
    exp_key: str = "exp3",
) -> Dict[str, str]:
    """
    Exp3 benchmark with two design modes:

      core30 (default):
        compact, theory-aligned design with rw>rb constraint and no SNR axis.
        Under current defaults this yields 6 settings per group config
        (2 concentrated + 2 distributed + 2 boundary) = 18 total settings
        across [G10x5, CL, CS].

      legacy_factorial:
        factor design without SNR axis:
          signal_structure x group_config x rho_within x rho_between
        with automatic filtering of combinations that violate rw>rb.
        target_snr is fixed at 1.0 in this mode.

    Signal types (default ["concentrated", "distributed", "boundary"]):
      concentrated: all nonzero beta in G0 and G1, G2/G3/G4 are null.
      distributed: one nonzero beta in each of G0 and G1.
      boundary: signal set to xi_ratio * xi_crit(u0=0.5, rho_profile),
                with rho_profile = rho_within / sqrt(sigma2_boundary).

    bayes_min_chains:
      Minimum number of chains for Bayesian methods in Exp3.
      Default: 2 for laptop profile, 4 for full profile.

    Methods:
      GR_RHS, GHS_plus, GIGG_MMLE, RHS, LASSO_CV, OLS.

    gigg_mode:
      paper_ref: strict baseline mode (no Exp3 hard-setting rescue/stabilization).
      stable:    enhanced mode with bounded rescue/stabilization in hard settings.
    """
    pd = load_pandas()
    produced: set[Path] = set()

    base = Path(save_dir)
    out_name = str(result_dir_name).strip() or "exp3_linear_benchmark"
    exp_key_name = str(exp_key).strip().lower() or "exp3"
    out_dir = ensure_dir(base / "results" / out_name)
    fig_dir = ensure_dir(base / "figures" / out_name)
    tab_dir = ensure_dir(base / "tables" / out_name)
    log = setup_logger(str(exp_key_name), base / "logs" / f"{out_name}.log")

    profile_name = _normalize_compute_profile(profile)
    sampler = _sampler_for_profile(profile_name)
    bayes_min_chains_use = int(bayes_min_chains) if bayes_min_chains is not None else (2 if profile_name == "laptop" else int(_BAYESIAN_DEFAULT_CHAINS))
    bayes_min_chains_use = max(1, int(bayes_min_chains_use))
    _exp3_methods = ["GR_RHS", "GHS_plus", "GIGG_MMLE", "RHS", "LASSO_CV", "OLS"]
    methods_use = [m for m in (methods or _exp3_methods) if m in set(_exp3_methods)]
    if not methods_use:
        methods_use = list(_exp3_methods)
    gigg_mode_name = _normalize_exp3_gigg_mode(gigg_mode)
    gigg_cfg = _exp3_gigg_config_for_mode(_gigg_config_for_profile(profile_name), gigg_mode=gigg_mode_name)
    retry_limit = _resolve_convergence_retry_limit(
        profile_name,
        max_convergence_retries,
        until_bayes_converged=bool(until_bayes_converged),
    )

    design_mode = str(exp3_design).strip().lower()
    if design_mode not in {"core30", "legacy_factorial"}:
        raise ValueError(f"unknown exp3_design: {exp3_design!r}. Use 'core30' or 'legacy_factorial'.")

    signals = list(signal_types or ["concentrated", "distributed", "boundary"])
    gc_list: list[dict[str, Any]] = list(group_configs) if group_configs is not None else list(_DEFAULT_EXP3_GROUP_CONFIGS)

    # settings:
    #   sid, signal, group_cfg, setting_block, env_id, design_type, rho_within, rho_between, target_snr
    settings: list[tuple[int, str, dict, str, str, str, float, float, float]] = []
    sid = 0
    env_points_used: list[dict[str, Any]] = []

    if design_mode == "core30":
        points_raw = list(env_points) if env_points is not None else list(_DEFAULT_EXP3_ENV_POINTS_CORE30)
        points, dropped_points = _exp3_filter_env_points_rw_gt_rb(points_raw)
        if dropped_points:
            log.warning(
                "Exp3: dropped %d env point(s) with rw<=rb: %s",
                len(dropped_points),
                [
                    (str(ep.get("env_id", "?")), float(ep.get("rho_within", float("nan"))), float(ep.get("rho_between", float("nan"))))
                    for ep in dropped_points
                ],
            )
        if not points:
            raise ValueError("No valid Exp3 env points remain after enforcing rw>rb.")
        for ep in points:
            env_points_used.append(
                {
                    "env_id": str(ep["env_id"]),
                    "setting_block": str(ep.get("setting_block", "custom")),
                    "rho_within": float(ep["rho_within"]),
                    "rho_between": float(ep["rho_between"]),
                    "target_snr": float(ep["target_snr"]),
                    "signals": [str(s) for s in ep.get("signals", signals)],
                }
            )
        for gc in gc_list:
            for ep in env_points_used:
                sig_set = set(ep.get("signals", signals))
                for signal in signals:
                    if signal not in sig_set:
                        continue
                    sid += 1
                    rho = float(ep["rho_within"])
                    rhob = float(ep["rho_between"])
                    snr = float(ep["target_snr"])
                    design = "orthonormal" if rho == 0.0 and rhob == 0.0 else "correlated"
                    settings.append((sid, signal, gc, str(ep["setting_block"]), str(ep["env_id"]), design, rho, rhob, snr))
    else:
        rho_values = list(rho_within_values if rho_within_values is not None else [0.3, 0.8])
        if snr_values is not None:
            log.warning("Exp3 legacy_factorial ignores snr_values (SNR axis removed); using fixed target_snr=1.0.")
        snr_list = [1.0]
        rhob = float(rho_between)
        rho_values_valid = [float(rho) for rho in rho_values if float(rho) > rhob]
        rho_values_dropped = [float(rho) for rho in rho_values if float(rho) <= rhob]
        if rho_values_dropped:
            log.warning(
                "Exp3 legacy_factorial: dropped rho_within values that violate rw>rb (rb=%.3f): %s",
                rhob,
                rho_values_dropped,
            )
        if not rho_values_valid:
            raise ValueError(f"No valid rho_within values remain after enforcing rw>rb with rho_between={rhob}.")
        env_points_used = [
            {
                "env_id": f"LEGACY_RW{float(rho):.1f}_SNR{float(snr):.1f}",
                "setting_block": "legacy_factorial",
                "rho_within": float(rho),
                "rho_between": rhob,
                "target_snr": float(snr),
                "signals": list(signals),
            }
            for rho in rho_values_valid
            for snr in snr_list
        ]
        for gc in gc_list:
            for signal in signals:
                for rho in rho_values_valid:
                    for snr in snr_list:
                        sid += 1
                        design = "orthonormal" if float(rho) == 0.0 and rhob == 0.0 else "correlated"
                        settings.append(
                            (
                                sid,
                                signal,
                                gc,
                                "legacy_factorial",
                                f"LEGACY_RW{float(rho):.1f}_SNR{float(snr):.1f}",
                                design,
                                float(rho),
                                rhob,
                                float(snr),
                            )
                        )

    grrhs_kw: dict = {"backend": str(sampler_backend), "tau_target": "groups"}
    if grrhs_extra_kwargs:
        grrhs_kw.update(grrhs_extra_kwargs)
    tasks: list[dict[str, Any]] = []
    for (sid_v, signal_v, gc_v, block_v, env_v, dt_v, rho_v, rhob_v, snr_v) in settings:
        for r in range(1, int(repeats) + 1):
            for method in methods_use:
                tasks.append(
                    {
                        "setting_id": int(sid_v),
                        "signal": str(signal_v),
                        "group_cfg": dict(gc_v),
                        "setting_block": str(block_v),
                        "env_id": str(env_v),
                        "design_type": str(dt_v),
                        "rho_within": float(rho_v),
                        "rho_between": float(rhob_v),
                        "target_snr": float(snr_v),
                        "replicate_id": int(r),
                        "seed_base": int(seed),
                        "n_test": int(n_test),
                        "sampler": sampler,
                        "method": str(method),
                        "gigg_config": dict(gigg_cfg),
                        "gigg_mode": str(gigg_mode_name),
                        "bayes_min_chains": int(bayes_min_chains_use),
                        "enforce_bayes_convergence": bool(enforce_bayes_convergence),
                        "max_convergence_retries": int(retry_limit),
                        "grrhs_kwargs": dict(grrhs_kw),
                    }
                )

    log.info(
        "Exp3[%s]: %d settings x %d repeats x %d methods = %d tasks "
        "(group_configs=%s, signals=%s, env_points=%s), methods=%s, bayes_min_chains=%d, enforce=%s, retry_limit=%d, gigg_mode=%s",
        design_mode,
        len(settings), repeats, len(methods_use), len(tasks),
        [gc["name"] for gc in gc_list], signals,
        [ep["env_id"] for ep in env_points_used],
        methods_use, int(bayes_min_chains_use), bool(enforce_bayes_convergence), int(retry_limit), str(gigg_mode_name),
    )
    bayes_tasks: list[dict[str, Any]] = []
    classical_tasks: list[dict[str, Any]] = []
    for t in tasks:
        method_name = str(t.get("method", ""))
        if _is_bayesian_method(method_name):
            bayes_tasks.append(t)
        else:
            classical_tasks.append(t)

    chunks_bayes: list[Any] = []
    chunks_classic: list[Any] = []
    if bayes_tasks:
        chunks_bayes = _parallel_rows(
            bayes_tasks,
            _exp3_worker,
            n_jobs=n_jobs,
            prefer_process=True,
            process_fallback="serial",
            progress_desc="Exp3 Linear Benchmark (Bayes)",
        )
    if classical_tasks:
        chunks_classic = _parallel_rows(
            classical_tasks,
            _exp3_worker,
            n_jobs=n_jobs,
            prefer_process=False,
            progress_desc="Exp3 Linear Benchmark (Classical)",
        )

    rows: list[dict] = []
    for chunk in list(chunks_bayes) + list(chunks_classic):
        rows.extend(chunk)

    raw = pd.DataFrame(rows)

    ok_raw = raw.loc[raw["status"] == "ok"].copy()
    conv_raw = ok_raw.loc[ok_raw["converged"].fillna(False).astype(bool)].copy()
    group_keys = ["gigg_mode", "group_config", "signal", "setting_block", "env_id", "design_type", "rho_within", "rho_between", "target_snr", "method"]

    counts_df = raw.groupby(group_keys, as_index=False).agg(
        n_reps_total=("replicate_id", "count"),
        n_reps_ok=("status", lambda s: int((s == "ok").sum())),
        n_reps_converged=("converged", lambda s: int(s.fillna(False).astype(bool).sum())),
    )

    metric_df = conv_raw.groupby(group_keys, as_index=False).agg(
        mse_null=("mse_null", "mean"),
        mse_signal=("mse_signal", "mean"),
        mse_overall=("mse_overall", "mean"),
        lpd_test=("lpd_test", "mean"),
        coverage_95=("coverage_95", "mean"),
        avg_ci_length=("avg_ci_length", "mean"),
        kappa_null_mean=("kappa_null_mean", "mean"),
        kappa_signal_mean=("kappa_signal_mean", "mean"),
    )

    agg_df = counts_df.merge(metric_df, on=group_keys, how="left")
    agg_df["n_reps"] = agg_df["n_reps_converged"]

    save_dataframe(raw, out_dir / "raw_results.csv")
    _record_produced_paths(produced, out_dir / "raw_results.csv")
    save_dataframe(agg_df, out_dir / "summary.csv")
    _record_produced_paths(produced, out_dir / "summary.csv")
    table_df = metric_df.merge(counts_df[group_keys + ["n_reps_converged"]], on=group_keys, how="left")
    table_df = table_df.rename(columns={"n_reps_converged": "n_reps"})
    save_dataframe(table_df, tab_dir / "table_linear_benchmark.csv")
    _record_produced_paths(produced, tab_dir / "table_linear_benchmark.csv")
    save_json({
        "exp3_design": design_mode,
        "profile": profile_name,
        "gigg_mode": str(gigg_mode_name),
        "group_configs": [{"name": gc["name"], "group_sizes": gc["group_sizes"], "active_groups": gc["active_groups"]} for gc in gc_list],
        "signals": signals,
        "env_points": env_points_used,
        "boundary_calibration": {
            "u0": float(_BOUNDARY_U0),
            "xi_ratio": float(_BOUNDARY_XI_RATIO),
            "sigma2_boundary": float(_SIGMA2_BOUNDARY),
            "rho_profile_formula": "rho_within / sqrt(sigma2_boundary)",
        },
        "n_train": 100,
        "n_test": int(n_test),
        "methods": methods_use,
        "bayes_min_chains": int(bayes_min_chains_use),
        "n_settings": len(settings),
        "repeats": int(repeats),
        "enforce_bayes_convergence": bool(enforce_bayes_convergence),
        "max_convergence_retries": int(retry_limit),
        "until_bayes_converged": bool(until_bayes_converged),
    }, out_dir / "exp3_meta.json")
    _record_produced_paths(produced, out_dir / "exp3_meta.json")

    try:
        from .plotting import plot_exp3_benchmark
        if not table_df.empty:
            plot_exp3_benchmark(table_df, out_dir=fig_dir)
            _record_produced_paths(produced, fig_dir / "fig3a_mse_by_signal.png", fig_dir / "fig3b_lpd_by_signal.png", fig_dir / "fig3c_null_signal_scatter.png")
        else:
            log.warning("Plot exp3 skipped: no converged rows available.")
    except Exception as exc:
        log.warning("Plot exp3 failed: %s", exc)

    log.info(
        "Exp3 done: %d rows, %d settings, %d converged rows used for metrics",
        len(rows),
        len(settings),
        int(conv_raw.shape[0]),
    )
    result_paths = {
        "raw": str(out_dir / "raw_results.csv"),
        "summary": str(out_dir / "summary.csv"),
        "table": str(tab_dir / "table_linear_benchmark.csv"),
        "meta": str(out_dir / "exp3_meta.json"),
        "fig3a_mse_by_signal": str(fig_dir / "fig3a_mse_by_signal.png"),
        "fig3b_lpd_by_signal": str(fig_dir / "fig3b_lpd_by_signal.png"),
        "fig3c_null_signal_scatter": str(fig_dir / "fig3c_null_signal_scatter.png"),
    }
    return _finalize_experiment_run(
        exp_key=str(exp_key_name),
        save_dir=save_dir,
        results_dir=out_dir,
        produced_paths=produced,
        result_paths=result_paths,
    )


def run_exp3a_main_benchmark(
    n_jobs: int = 1,
    seed: int = MASTER_SEED,
    repeats: int = 20,
    save_dir: str = "simulation_project",
    **kwargs,
) -> Dict[str, str]:
    """Exp3a: main benchmark (concentrated + distributed only)."""
    return run_exp3_linear_benchmark(
        n_jobs=n_jobs,
        seed=seed,
        repeats=repeats,
        save_dir=save_dir,
        signal_types=["concentrated", "distributed"],
        result_dir_name="exp3a_main_benchmark",
        exp_key="exp3a",
        **kwargs,
    )


def run_exp3b_boundary_stress(
    n_jobs: int = 1,
    seed: int = MASTER_SEED,
    repeats: int = 20,
    save_dir: str = "simulation_project",
    **kwargs,
) -> Dict[str, str]:
    """Exp3b: boundary-only stress benchmark."""
    return run_exp3_linear_benchmark(
        n_jobs=n_jobs,
        seed=seed,
        repeats=repeats,
        save_dir=save_dir,
        signal_types=["boundary"],
        result_dir_name="exp3b_boundary_stress",
        exp_key="exp3b",
        **kwargs,
    )

# ---------------------------------------------------------------------------
# EXP4 - GR-RHS Variant Ablation: tau Calibration
# ---------------------------------------------------------------------------
# Mechanism-oriented ablation around tau-prior calibration:
#   tau0 = p0 / (p - p0) / sqrt(n)  (CPS-style scaling)
#
# IMPORTANT SEMANTICS:
#   p0_true in this experiment denotes the number of active coefficients
#   (not the number of active groups).
#
# Compared variants:
#   calibrated: auto-calibrated tau0 from p0 estimate
#   fixed_10x:  intentionally over-permissive tau0 = 10 * tau0_oracle
#   RHS_oracle: standard RHS baseline with oracle p0
#   oracle:     optional GR-RHS oracle-tau variant (disabled by default)
#
# Evaluation focus:
#   - predictive error (MSE overall/null/signal)
#   - shrinkage separation (kappa_null vs kappa_signal)
#   - tau ratio retained as a diagnostic signal only
#
# Lightweight default: p0_true in {5, 30}
# DGP aligned with Exp3 scale: n=100, p=50 (5 groups x 10 vars)
# Default retries: 0 (fixed budget, predictable wall-time)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Run-all orchestrator
# ---------------------------------------------------------------------------

def run_all_experiments(
    n_jobs: int = 1,
    seed: int = MASTER_SEED,
    save_dir: str = "simulation_project",
    *,
    profile: str = "full",
    enforce_bayes_convergence: bool = True,
    max_convergence_retries: int | None = None,
    until_bayes_converged: bool = True,
    sampler_backend: str = "nuts",
    exp3_gigg_mode: str = "stable",
    skip_analysis: bool = False,
) -> Dict[str, Any]:
    profile_name = _normalize_compute_profile(profile)
    exp3_gigg_mode_name = _normalize_exp3_gigg_mode(exp3_gigg_mode)
    retry_limit = max_convergence_retries
    common = dict(
        n_jobs=n_jobs, seed=seed, save_dir=save_dir, profile=profile_name,
        enforce_bayes_convergence=bool(enforce_bayes_convergence),
        max_convergence_retries=retry_limit,
        until_bayes_converged=bool(until_bayes_converged),
        sampler_backend=str(sampler_backend),
    )
    out: Dict[str, Any] = {}
    jobs: list[tuple[str, Any]] = [
        ("exp1", lambda: run_exp1_kappa_profile_regimes(n_jobs=n_jobs, seed=seed, save_dir=save_dir, repeats=_default_repeats("exp1", profile_name))),
        ("exp2", lambda: run_exp2_group_separation(repeats=_default_repeats("exp2", profile_name), **common)),
        ("exp3a", lambda: run_exp3a_main_benchmark(repeats=_default_repeats("exp3", profile_name), gigg_mode=exp3_gigg_mode_name, **common)),
        ("exp3b", lambda: run_exp3b_boundary_stress(repeats=_default_repeats("exp3", profile_name), gigg_mode=exp3_gigg_mode_name, **common)),
        ("exp4", lambda: run_exp4_variant_ablation(repeats=_default_repeats("exp4", profile_name), **common)),
        ("exp5", lambda: run_exp5_prior_sensitivity(repeats=_default_repeats("exp5", profile_name), **common)),
    ]
    for name, runner in tqdm(jobs, total=len(jobs), desc="All Experiments", leave=True):
        out[name] = runner()
    save_json(
        {"profile": profile_name, "enforce_bayes_convergence": bool(enforce_bayes_convergence), "max_convergence_retries": retry_limit, "until_bayes_converged": bool(until_bayes_converged), "exp3_gigg_mode": str(exp3_gigg_mode_name), "results": out},
        Path(save_dir) / "results" / "run_manifest.json",
    )
    if not skip_analysis:
        from .analysis import run_analysis
        run_analysis(save_dir=save_dir)
    return out

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the unified simulation pipeline (Exp1, Exp2, Exp3a, Exp3b, Exp4, Exp5). "
            "On Windows, process-pool parallelism is disabled by default; "
            "set SIM_ALLOW_WINDOWS_PROCESS_POOL=1 to force-enable from a spawn-safe script entrypoint."
        )
    )
    parser.add_argument("--experiment", default="all", choices=list(CLI_EXPERIMENT_CHOICES))
    parser.add_argument("--save-dir", default="simulation_project")
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--profile", type=str, default="full", choices=list(COMPUTE_PROFILES))
    parser.add_argument("--no-enforce-bayes-convergence", action="store_true")
    parser.add_argument("--max-convergence-retries", type=int, default=None)
    parser.add_argument("--until-bayes-converged", action="store_true")
    parser.add_argument("--exp3-gigg-mode", type=str, default="stable", choices=list(EXP3_GIGG_MODES),
                        help="Exp3 GIGG mode: stable (enhanced, default) or paper_ref (strict baseline).")
    parser.add_argument("--sampler", type=str, default="nuts", choices=["nuts", "collapsed", "gibbs"],
                        help="GR-RHS posterior sampler: nuts (global default), collapsed (beta marginalized, Gaussian only), gibbs (Gibbs+slice, Gaussian only); Exp4 defaults to gibbs when --sampler is omitted")
    args = parser.parse_args()
    exp_key = cli_choice_to_key(args.experiment)
    profile_name = _normalize_compute_profile(args.profile)
    exp3_gigg_mode_name = _normalize_exp3_gigg_mode(args.exp3_gigg_mode)
    enforce_conv = not bool(args.no_enforce_bayes_convergence)
    until_conv = bool(args.until_bayes_converged) or (enforce_conv and args.max_convergence_retries is None)
    sampler_backend_cli = _resolve_sampler_backend_for_experiment(exp_key, args.sampler)
    common = dict(
        n_jobs=args.n_jobs, seed=args.seed, save_dir=args.save_dir,
        profile=profile_name,
        enforce_bayes_convergence=enforce_conv,
        max_convergence_retries=args.max_convergence_retries,
        until_bayes_converged=until_conv,
        sampler_backend=sampler_backend_cli,
    )
    reps = args.repeats

    from .analysis import analyze_exp1, analyze_exp2, analyze_exp3, analyze_exp4, analyze_exp5, _safe_print, run_analysis
    _base = Path(args.save_dir)

    def _print_exp_analysis(label: str, result: dict) -> None:
        sep = "=" * 68
        lines = [sep, f"ANALYSIS: {label}", "=" * 68]
        for finding in result.get("findings", []):
            lines.append(finding)
        lines.append(sep)
        _safe_print("\n".join(lines))
        out_path = _base / "results" / f"analysis_{label.lower().replace(' ', '_').replace(':', '')}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as _f:
            json.dump(result.get("metrics", {}), _f, indent=2)

    if exp_key == "all":
        run_all_experiments(**common, exp3_gigg_mode=exp3_gigg_mode_name)
    elif exp_key == "analysis":
        run_analysis(save_dir=args.save_dir)
    else:
        dispatch: dict[str, dict[str, Any]] = {
            "exp1": {
                "run": lambda: run_exp1_kappa_profile_regimes(
                    n_jobs=args.n_jobs,
                    seed=args.seed,
                    save_dir=args.save_dir,
                    repeats=reps or _default_repeats("exp1", profile_name),
                ),
                "analyze": analyze_exp1,
                "label": "Exp1: kappa_g Profile Regimes",
                "results_subdir": "exp1_kappa_profile_regimes",
            },
            "exp2": {
                "run": lambda: run_exp2_group_separation(
                    repeats=reps or _default_repeats("exp2", profile_name),
                    **common,
                ),
                "analyze": analyze_exp2,
                "label": "Exp2: Group Separation",
                "results_subdir": "exp2_group_separation",
            },
            "exp3": {
                "run": lambda: run_exp3_linear_benchmark(
                    repeats=reps or _default_repeats("exp3", profile_name),
                    gigg_mode=exp3_gigg_mode_name,
                    **common,
                ),
                "analyze": analyze_exp3,
                "label": "Exp3: Linear Benchmark",
                "results_subdir": "exp3_linear_benchmark",
            },
            "exp3a": {
                "run": lambda: run_exp3a_main_benchmark(
                    repeats=reps or _default_repeats("exp3", profile_name),
                    gigg_mode=exp3_gigg_mode_name,
                    **common,
                ),
                "analyze": analyze_exp3,
                "label": "Exp3a: Main Benchmark",
                "results_subdir": "exp3a_main_benchmark",
            },
            "exp3b": {
                "run": lambda: run_exp3b_boundary_stress(
                    repeats=reps or _default_repeats("exp3", profile_name),
                    gigg_mode=exp3_gigg_mode_name,
                    **common,
                ),
                "analyze": analyze_exp3,
                "label": "Exp3b: Boundary Stress",
                "results_subdir": "exp3b_boundary_stress",
            },
            "exp4": {
                "run": lambda: run_exp4_variant_ablation(
                    repeats=reps or _default_repeats("exp4", profile_name),
                    **common,
                ),
                "analyze": analyze_exp4,
                "label": "Exp4: Variant Ablation",
                "results_subdir": "exp4_variant_ablation",
            },
            "exp5": {
                "run": lambda: run_exp5_prior_sensitivity(
                    repeats=reps or _default_repeats("exp5", profile_name),
                    **common,
                ),
                "analyze": analyze_exp5,
                "label": "Exp5: Prior Sensitivity",
                "results_subdir": "exp5_prior_sensitivity",
            },
        }
        spec = dispatch[exp_key]
        spec["run"]()
        analyzer = spec["analyze"]
        label = str(spec["label"])
        results_subdir = str(spec["results_subdir"])
        _print_exp_analysis(label, analyzer(_base / "results" / results_subdir))


if __name__ == "__main__":
    _cli()
