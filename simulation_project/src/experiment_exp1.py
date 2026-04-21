from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from .dgp_normal_means import (
    generate_null_group,
    generate_signal_group_distributed,
    kappa_posterior_grid,
    posterior_summary_from_grid,
)
from .experiment_reporting import _finalize_experiment_run, _record_produced_paths
from .experiment_runtime import _parallel_rows, xi_crit_u0_rho
from .utils import (
    MASTER_SEED,
    ensure_dir,
    experiment_seed,
    load_pandas,
    save_dataframe,
    save_json,
    setup_logger,
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