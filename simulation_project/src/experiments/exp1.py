from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from .dgp.normal_means import (
    generate_null_group,
    generate_signal_group_distributed,
    kappa_posterior_grid,
    posterior_summary_from_grid,
)
from .fitting import _fit_with_convergence_retry
from .reporting import _finalize_experiment_run, _record_produced_paths
from .runtime import (
    _normalize_compute_profile,
    _parallel_rows,
    _sampler_for_profile,
    kappa_star_xi_ratio_u0_rho,
    xi_crit_u0_rho,
)
from ..utils import (
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
    rho_profile = float(tau) / math.sqrt(max(float(sigma2), 1e-12))
    xi_c = xi_crit_u0_rho(u0=float(u0), rho=rho_profile)
    xi_ratio = float(xi) / max(xi_c, 1e-12)
    kappa_star = kappa_star_xi_ratio_u0_rho(xi_ratio=xi_ratio, u0=float(u0), rho=rho_profile)
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
            "xi_ratio": xi_ratio,
            "rho_profile": rho_profile,
            "kappa_star_theory": float(kappa_star) if np.isfinite(kappa_star) else float("nan"),
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
    rho_profile = float(tau) / math.sqrt(max(float(sigma2), 1e-12))
    xi_c = xi_crit_u0_rho(u0=float(u0), rho=rho_profile)
    xi_ratio = float(xi) / max(xi_c, 1e-12)
    kappa_star = kappa_star_xi_ratio_u0_rho(xi_ratio=xi_ratio, u0=float(u0), rho=rho_profile)
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
        "xi_ratio": xi_ratio,
        "rho_profile": rho_profile,
        "kappa_star_theory": float(kappa_star) if np.isfinite(kappa_star) else float("nan"),
        "u0": float(u0),
        "alpha_kappa": float(alpha_kappa),
        "beta_kappa": float(beta_kappa),
        "post_mean_kappa": sm["post_mean_kappa"],
        "post_prob_kappa_gt_u0": sm.get("tail_prob", float("nan")),
    }


def _summarize_null_panel(null_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float, tuple[float, float]]:
    null_agg: list[dict[str, Any]] = []
    pg_vals_seen = sorted({int(r["p_g"]) for r in null_rows})
    for pg in pg_vals_seen:
        sub = [r for r in null_rows if int(r["p_g"]) == pg]
        means = np.asarray([float(r.get("post_mean_kappa", float("nan"))) for r in sub], dtype=float)
        tails = np.asarray([float(r.get("tail_prob_kappa_gt_eps", float("nan"))) for r in sub], dtype=float)
        means = means[np.isfinite(means)]
        tails_finite = tails[np.isfinite(tails)]
        null_agg.append(
            {
                "p_g": int(pg),
                "median_post_mean_kappa": float(np.median(means)) if means.size else float("nan"),
                "mean_post_mean_kappa": float(np.mean(means)) if means.size else float("nan"),
                "q25_post_mean_kappa": float(np.quantile(means, 0.25)) if means.size else float("nan"),
                "q75_post_mean_kappa": float(np.quantile(means, 0.75)) if means.size else float("nan"),
                "std_post_mean_kappa": float(np.std(means, ddof=1)) if means.size > 1 else float("nan"),
                "mean_tail_prob_kappa_gt_eps": float(np.mean(tails_finite)) if tails_finite.size else float("nan"),
                "n_replicates": int(len(sub)),
            }
        )

    if not null_agg:
        return null_agg, float("nan"), (float("nan"), float("nan"))
    log_pg = np.log(np.asarray([r["p_g"] for r in null_agg], dtype=float))
    log_kappa = np.log(np.maximum(np.asarray([r["median_post_mean_kappa"] for r in null_agg], dtype=float), 1e-12))
    fit_mask = np.asarray([20 <= int(r["p_g"]) <= 500 for r in null_agg], dtype=bool)
    fit_mask &= np.isfinite(log_pg) & np.isfinite(log_kappa)
    if int(np.sum(fit_mask)) < 2:
        return null_agg, float("nan"), (float("nan"), float("nan"))
    slope, slope_ci = _linreg_slope_ci(log_pg[fit_mask], log_kappa[fit_mask])
    return null_agg, float(slope), (float(slope_ci[0]), float(slope_ci[1]))


def _exp1_full_null_worker(task: tuple) -> dict[str, Any]:
    sid, pg, r, seed, alpha_kappa, beta_kappa, tail_eps, sampler, backend, retry_limit, enforce_convergence = task
    from .methods.fit_gr_rhs import fit_gr_rhs

    s = experiment_seed(1, int(sid), int(r), master_seed=int(seed) + 7_000_000)
    y = generate_null_group(pg=int(pg), sigma2=1.0, seed=s)
    X = np.eye(int(pg), dtype=float)
    groups = [list(range(int(pg)))]

    res = _fit_with_convergence_retry(
        lambda st, att, _s=s, _be=str(backend), _ak=alpha_kappa, _bk=beta_kappa: fit_gr_rhs(
            X,
            y,
            groups,
            task="gaussian",
            seed=int(_s + 31 + 100 * int(att)),
            p0=1,
            sampler=st,
            alpha_kappa=float(_ak),
            beta_kappa=float(_bk),
            use_group_scale=True,
            use_local_scale=True,
            shared_kappa=False,
            tau_target="groups",
            backend=str(_be),
            progress_bar=False,
        ),
        method="GR_RHS",
        sampler=sampler,
        bayes_min_chains=int(getattr(sampler, "chains", 2)),
        max_convergence_retries=int(retry_limit),
        enforce_bayes_convergence=bool(enforce_convergence),
    )

    post_mean = float("nan")
    post_median = float("nan")
    tail_prob = float("nan")
    if res.kappa_draws is not None:
        kd = np.asarray(res.kappa_draws, dtype=float)
        if kd.ndim > 2:
            kd = kd.reshape(-1, kd.shape[-1])
        if kd.ndim == 1:
            kd = kd.reshape(-1, 1)
        if kd.shape[-1] >= 1:
            kg = kd[:, 0]
            post_mean = float(np.mean(kg))
            post_median = float(np.median(kg))
            tail_prob = float(np.mean(kg > float(tail_eps)))

    return {
        "panel": "null_full",
        "p_g": int(pg),
        "setting_id": int(sid),
        "replicate_id": int(r),
        "alpha_kappa": float(alpha_kappa),
        "beta_kappa": float(beta_kappa),
        "post_mean_kappa": post_mean,
        "post_median_kappa": post_median,
        "tail_prob_kappa_gt_eps": tail_prob,
        "status": str(res.status),
        "converged": bool(res.converged),
    }


def run_exp1_kappa_profile_regimes(
    n_jobs: int = 1,
    seed: int = MASTER_SEED,
    repeats: int = 500,
    save_dir: str = "outputs/simulation_project",
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
    # Optional full-model null-curve overlay (for rebuttal diagnostics)
    include_full_null_curve: bool = False,
    full_null_repeats: int | None = None,
    full_null_pg_list: Sequence[int] | None = None,
    full_null_profile: str = "laptop",
    full_null_backend: str = "gibbs",
    full_null_max_convergence_retries: int = 1,
    full_null_enforce_convergence: bool = True,
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
    from .analysis.plotting import plot_exp1, plot_exp1_phase, plot_exp1_phase_kappa_overlay
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
    null_agg, slope, slope_ci = _summarize_null_panel(null_rows)
    log.info("Panel A log-log slope (p_g 20-500): %.4f (95%% CI [%.4f, %.4f]), expected -0.5", slope, slope_ci[0], slope_ci[1])

    full_rows: list[dict[str, Any]] = []
    full_agg: list[dict[str, Any]] = []
    full_slope = float("nan")
    full_slope_ci = (float("nan"), float("nan"))
    if bool(include_full_null_curve):
        full_profile = _normalize_compute_profile(full_null_profile)
        full_sampler = _sampler_for_profile(full_profile)
        full_reps = int(full_null_repeats) if full_null_repeats is not None else int(min(20, int(repeats)))
        full_pg = list(full_null_pg_list or [20, 50, 100, 200, 500])
        log.info(
            "Exp1 Panel A full overlay: pg=%s, repeats=%d, backend=%s, profile=%s",
            full_pg,
            full_reps,
            str(full_null_backend),
            full_profile,
        )
        full_tasks: list[tuple] = []
        for sid, pg in enumerate(full_pg, start=10_001):
            for r in range(1, full_reps + 1):
                full_tasks.append(
                    (
                        sid,
                        int(pg),
                        int(r),
                        int(seed),
                        float(alpha_kappa),
                        float(beta_kappa),
                        float(tail_eps),
                        full_sampler,
                        str(full_null_backend),
                        int(full_null_max_convergence_retries),
                        bool(full_null_enforce_convergence),
                    )
                )
        full_rows = _parallel_rows(
            full_tasks,
            _exp1_full_null_worker,
            n_jobs=n_jobs,
            prefer_process=False,
            progress_desc="Exp1A Null (full overlay)",
        )
        full_agg, full_slope, full_slope_ci = _summarize_null_panel(full_rows)
        log.info(
            "Panel A full-overlay slope (p_g 20-500): %.4f (95%% CI [%.4f, %.4f])",
            full_slope,
            full_slope_ci[0],
            full_slope_ci[1],
        )

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
        post_kappa = np.array([float(r["post_mean_kappa"]) for r in sub], dtype=float)
        kappa_star = np.array([float(r.get("kappa_star_theory", float("nan"))) for r in sub], dtype=float)
        phase_agg.append({
            "tau": tau_v, "p_g": pg_v, "xi_ratio": xi_r,
            "xi_crit": float(sub[0]["xi_crit"]),
            "xi": float(sub[0]["xi"]),
            "mean_prob_kappa_gt_u0": float(np.mean(probs)),
            "mean_post_mean_kappa": float(np.mean(post_kappa[np.isfinite(post_kappa)])) if np.any(np.isfinite(post_kappa)) else float("nan"),
            "mean_kappa_star_theory": float(np.mean(kappa_star[np.isfinite(kappa_star)])) if np.any(np.isfinite(kappa_star)) else float("nan"),
            "n_replicates": len(probs),
        })

    # --- Save ---
    pd = load_pandas()
    all_rows = null_rows + phase_rows + full_rows
    save_dataframe(pd.DataFrame(all_rows), out_dir / "raw_results.csv")
    _record_produced_paths(produced, out_dir / "raw_results.csv")
    save_dataframe(pd.DataFrame(null_agg), out_dir / "summary_null.csv")
    _record_produced_paths(produced, out_dir / "summary_null.csv")
    save_dataframe(pd.DataFrame(phase_agg), out_dir / "summary_phase.csv")
    _record_produced_paths(produced, out_dir / "summary_phase.csv")
    if full_agg:
        save_dataframe(pd.DataFrame(full_agg), out_dir / "summary_null_full.csv")
        _record_produced_paths(produced, out_dir / "summary_null_full.csv")
    # Statistically correct criterion: does the 95% CI for slope contain the theoretical -0.5?
    # Also require the point estimate to be in a plausible range [-0.8, -0.25] to reject degenerate fits.
    _pass_ci_contains = slope_ci[0] < -0.5 < slope_ci[1]
    _pass_estimate = -0.8 < slope < -0.25
    save_json({"slope": slope, "slope_ci": list(slope_ci), "expected_slope": -0.5, "fit_range_pg": [20, 500], "ci_contains_theory": _pass_ci_contains, "pass": bool(_pass_ci_contains and _pass_estimate)}, out_dir / "null_slope_check.json")
    _record_produced_paths(produced, out_dir / "null_slope_check.json")
    if full_agg:
        _full_ci_contains = full_slope_ci[0] < -0.5 < full_slope_ci[1]
        _full_pass_est = -0.8 < full_slope < -0.25
        save_json(
            {
                "slope": float(full_slope),
                "slope_ci": [float(full_slope_ci[0]), float(full_slope_ci[1])],
                "expected_slope": -0.5,
                "fit_range_pg": [20, 500],
                "ci_contains_theory": bool(_full_ci_contains),
                "pass": bool(_full_ci_contains and _full_pass_est),
                "backend": str(full_null_backend),
                "profile": str(full_profile),
                "repeats": int(full_reps),
            },
            out_dir / "null_slope_check_full.json",
        )
        _record_produced_paths(produced, out_dir / "null_slope_check_full.json")

    try:
        plot_exp1(
            pd.DataFrame(null_agg),
            slope=slope,
            slope_ci=slope_ci,
            out_path=fig_dir / "fig1a_null_contraction.png",
            full_df=(pd.DataFrame(full_agg) if full_agg else None),
            full_slope=(float(full_slope) if np.isfinite(full_slope) else None),
            full_slope_ci=(
                (float(full_slope_ci[0]), float(full_slope_ci[1]))
                if (np.isfinite(full_slope_ci[0]) and np.isfinite(full_slope_ci[1]))
                else None
            ),
        )
        _record_produced_paths(produced, fig_dir / "fig1a_null_contraction.png")
    except Exception as exc:
        log.warning("Plot exp1A failed: %s", exc)
    try:
        plot_exp1_phase(pd.DataFrame(phase_agg), out_path=fig_dir / "fig1b_phase_diagram.png")
        _record_produced_paths(produced, fig_dir / "fig1b_phase_diagram.png")
    except Exception as exc:
        log.warning("Plot exp1B failed: %s", exc)
    try:
        plot_exp1_phase_kappa_overlay(pd.DataFrame(phase_agg), out_path=fig_dir / "fig1c_kappa_phase_overlay.png")
        _record_produced_paths(produced, fig_dir / "fig1c_kappa_phase_overlay.png")
    except Exception as exc:
        log.warning("Plot exp1C (kappa overlay) failed: %s", exc)

    log.info("Exp1 done: %d null rows, %d phase rows", len(null_rows), len(phase_rows))
    result_paths = {
        "null_raw": str(out_dir / "raw_results.csv"),
        "null_summary": str(out_dir / "summary_null.csv"),
        "phase_summary": str(out_dir / "summary_phase.csv"),
        "null_slope_check": str(out_dir / "null_slope_check.json"),
        "fig1a_null_contraction": str(fig_dir / "fig1a_null_contraction.png"),
        "fig1b_phase_diagram": str(fig_dir / "fig1b_phase_diagram.png"),
        "fig1c_kappa_phase_overlay": str(fig_dir / "fig1c_kappa_phase_overlay.png"),
    }
    if full_agg:
        result_paths["null_summary_full"] = str(out_dir / "summary_null_full.csv")
        result_paths["null_slope_check_full"] = str(out_dir / "null_slope_check_full.json")
    return _finalize_experiment_run(
        exp_key="exp1",
        save_dir=save_dir,
        results_dir=out_dir,
        produced_paths=produced,
        result_paths=result_paths,
    )




