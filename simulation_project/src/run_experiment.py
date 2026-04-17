from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import csv
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
from .utils import MASTER_SEED, FitResult, SamplerConfig, ensure_dir, experiment_seed, save_dataframe, save_json, setup_logger


METHODS = ["GR_RHS", "RHS", "GIGG_MMLE", "GHS_plus"]


def _save_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _resolve_workers(n_jobs: int, n_tasks: int) -> int:
    return max(1, min(int(n_jobs), int(n_tasks)))


def _parallel_rows(
    tasks: list[Any],
    worker,
    n_jobs: int,
    *,
    prefer_process: bool = False,
    progress_desc: str | None = None,
) -> list[Any]:
    if len(tasks) == 0:
        return []
    workers = _resolve_workers(n_jobs=n_jobs, n_tasks=len(tasks))
    if workers <= 1:
        return [worker(t) for t in tqdm(tasks, total=len(tasks), desc=progress_desc or "Running", leave=True)]
    out: list[Any] = [None] * len(tasks)
    executor_cls = ProcessPoolExecutor if prefer_process else ThreadPoolExecutor
    try:
        with executor_cls(max_workers=workers) as ex:
            fut_map = {ex.submit(worker, tasks[i]): i for i in range(len(tasks))}
            iterator = tqdm(as_completed(fut_map), total=len(tasks), desc=progress_desc or "Running", leave=True)
            for fut in iterator:
                out[fut_map[fut]] = fut.result()
    except Exception as exc:
        if prefer_process:
            print(f"[WARN] Process pool failed ({type(exc).__name__}: {exc}). Falling back to thread pool.")
            with ThreadPoolExecutor(max_workers=workers) as ex:
                fut_map = {ex.submit(worker, tasks[i]): i for i in range(len(tasks))}
                iterator = tqdm(as_completed(fut_map), total=len(tasks), desc=(progress_desc or "Running") + " [thread-fallback]", leave=True)
                for fut in iterator:
                    out[fut_map[fut]] = fut.result()
        else:
            raise
    return out


def theta_u0_rho(u0: float, rho: float) -> float:
    u = float(u0)
    rho2 = float(rho) ** 2
    den = u + (1.0 - u) * rho2
    return float((u * rho2) / max(den, 1e-12))


def xi_crit_u0_rho(u0: float, rho: float) -> float:
    return 0.5 * theta_u0_rho(u0=u0, rho=rho)


def _exp1_worker(task: tuple[int, int, int, int, float, float]) -> dict[str, Any]:
    sid, pg, r, seed, tau_eval, tail_M = task
    s = experiment_seed(1, sid, r, master_seed=seed)
    y = generate_null_group(pg=pg, sigma2=1.0, seed=s)
    grid = kappa_posterior_grid(y, tau=tau_eval, sigma2=1.0, alpha_kappa=1.0, beta_kappa=1.0)
    summary = posterior_summary_from_grid(grid["kappa"], grid["density"], tail_threshold=tail_M / math.sqrt(pg))
    return {
        "p_g": pg,
        "setting_id": sid,
        "replicate_id": r,
        "post_mean_kappa": summary["post_mean_kappa"],
        "post_median_kappa": summary["post_median_kappa"],
        "post_sd_kappa": summary["post_sd_kappa"],
        "tail_prob_kappa_gt_2_over_sqrt_pg": summary.get("tail_prob", float("nan")),
    }


def _exp1_setting_worker(task: tuple[int, int, int, int, float, float, int]) -> list[dict[str, Any]]:
    sid, pg, repeats, seed, tau_eval, tail_M, grid_size = task
    rows: list[dict[str, Any]] = []
    for r in range(1, int(repeats) + 1):
        s = experiment_seed(1, sid, r, master_seed=seed)
        y = generate_null_group(pg=pg, sigma2=1.0, seed=s)
        grid = kappa_posterior_grid(y, tau=tau_eval, sigma2=1.0, alpha_kappa=1.0, beta_kappa=1.0, grid_size=grid_size)
        summary = posterior_summary_from_grid(grid["kappa"], grid["density"], tail_threshold=tail_M / math.sqrt(pg))
        rows.append(
            {
                "p_g": pg,
                "setting_id": sid,
                "replicate_id": r,
                "post_mean_kappa": summary["post_mean_kappa"],
                "post_median_kappa": summary["post_median_kappa"],
                "post_sd_kappa": summary["post_sd_kappa"],
                "tail_prob_kappa_gt_2_over_sqrt_pg": summary.get("tail_prob", float("nan")),
            }
        )
    return rows


def _exp2_worker(task: tuple[int, int, int, float, float, int, float, float, float]) -> dict[str, Any]:
    sid, pg, r, mu_g, scale, seed, tau_eval, win_lo, win_hi = task
    s = experiment_seed(2, sid, r, master_seed=seed)
    y, _ = generate_signal_group_distributed(pg=pg, mu_g=mu_g, sigma2=1.0, seed=s)
    grid = kappa_posterior_grid(y, tau=tau_eval, sigma2=1.0, alpha_kappa=1.0, beta_kappa=1.0)
    summary = posterior_summary_from_grid(grid["kappa"], grid["density"], window_lower=win_lo, window_upper=win_hi)
    return {
        "p_g": pg,
        "mu_g": mu_g,
        "tau_eval": float(tau_eval),
        "window_lo": float(win_lo),
        "window_hi": float(win_hi),
        "replicate_id": r,
        "post_mean_kappa": summary["post_mean_kappa"],
        "ratio_R": summary["post_mean_kappa"] / max(scale, 1e-12),
        "post_prob_kappa_in_scale_window": summary.get("window_prob", float("nan")),
    }


def _exp2_setting_worker(task: tuple[int, int, int, float, float, int, float, float, float, int]) -> list[dict[str, Any]]:
    sid, pg, repeats, mu_g, scale, seed, tau_eval, win_lo, win_hi, grid_size = task
    rows: list[dict[str, Any]] = []
    for r in range(1, int(repeats) + 1):
        s = experiment_seed(2, sid, r, master_seed=seed)
        y, _ = generate_signal_group_distributed(pg=pg, mu_g=mu_g, sigma2=1.0, seed=s)
        grid = kappa_posterior_grid(y, tau=tau_eval, sigma2=1.0, alpha_kappa=1.0, beta_kappa=1.0, grid_size=grid_size)
        summary = posterior_summary_from_grid(grid["kappa"], grid["density"], window_lower=win_lo, window_upper=win_hi)
        rows.append(
            {
                "p_g": pg,
                "mu_g": mu_g,
                "tau_eval": float(tau_eval),
                "window_lo": float(win_lo),
                "window_hi": float(win_hi),
                "replicate_id": r,
                "post_mean_kappa": summary["post_mean_kappa"],
                "ratio_R": summary["post_mean_kappa"] / max(scale, 1e-12),
                "post_prob_kappa_in_scale_window": summary.get("window_prob", float("nan")),
            }
        )
    return rows


def _exp3_worker(task: tuple[int, int, int, float, int, int, float, float, float]) -> dict[str, Any]:
    sid, pg, xid, xi, r, seed, tau, sigma2, u0 = task
    mu_g = xi * pg
    s = experiment_seed(3, sid * 100 + xid, r, master_seed=seed)
    y, _ = generate_signal_group_distributed(pg=pg, mu_g=mu_g, sigma2=sigma2, seed=s)
    grid = kappa_posterior_grid(y, tau=tau, sigma2=sigma2, alpha_kappa=1.0, beta_kappa=1.0)
    summary = posterior_summary_from_grid(grid["kappa"], grid["density"], tail_threshold=u0)
    return {
        "tau": float(tau),
        "xi": xi,
        "p_g": pg,
        "replicate_id": r,
        "post_prob_kappa_gt_u0": summary.get("tail_prob", float("nan")),
        "post_mean_kappa": summary["post_mean_kappa"],
    }


def _exp3_setting_worker(task: tuple[int, int, int, float, int, int, float, float, float, int]) -> list[dict[str, Any]]:
    sid, pg, xid, xi, repeats, seed, tau, sigma2, u0, grid_size = task
    rows: list[dict[str, Any]] = []
    mu_g = xi * pg
    for r in range(1, int(repeats) + 1):
        s = experiment_seed(3, sid * 100 + xid, r, master_seed=seed)
        y, _ = generate_signal_group_distributed(pg=pg, mu_g=mu_g, sigma2=sigma2, seed=s)
        grid = kappa_posterior_grid(y, tau=tau, sigma2=sigma2, alpha_kappa=1.0, beta_kappa=1.0, grid_size=grid_size)
        summary = posterior_summary_from_grid(grid["kappa"], grid["density"], tail_threshold=u0)
        rows.append(
            {
                "tau": float(tau),
                "xi": xi,
                "p_g": pg,
                "replicate_id": r,
                "post_prob_kappa_gt_u0": summary.get("tail_prob", float("nan")),
                "post_mean_kappa": summary["post_mean_kappa"],
            }
        )
    return rows


def _exp4_worker(task: tuple[int, str, dict[str, Any], int, int, SamplerConfig]) -> list[dict[str, Any]]:
    from .dgp_grouped_linear import build_linear_beta, generate_grouped_linear_dataset

    sid, setting, spec, r, seed, sampler = task
    s = experiment_seed(4, sid, r, master_seed=seed)
    beta_shape = build_linear_beta(setting, spec["group_sizes"])
    ds = generate_grouped_linear_dataset(
        n=500,
        group_sizes=spec["group_sizes"],
        rho_within=spec["rho_within"],
        rho_between=spec["rho_between"],
        beta_shape=beta_shape,
        seed=s,
        target_snr=0.70,
        design_type=str(spec.get("design_type", "correlated")),
    )
    fits = _fit_all_methods(ds["X"], ds["y"], ds["groups"], task="gaussian", seed=s, p0=int(np.sum(np.abs(ds["beta0"]) > 0.0)), sampler=sampler)
    out_rows: list[dict[str, Any]] = []
    for method, result in fits.items():
        metrics = _evaluate_method_row(result, ds["beta0"])
        out_rows.append({"setting": setting, "replicate_id": r, "method": method, "status": result.status, "converged": result.converged, "runtime_seconds": result.runtime_seconds, "rhat_max": result.rhat_max, "bulk_ess_min": result.bulk_ess_min, "divergence_ratio": result.divergence_ratio, "error": result.error, **metrics})
    return out_rows


def _exp5_worker(task: tuple[int, int, list[int], list[float], SamplerConfig]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from .dgp_grouped_linear import generate_heterogeneity_dataset
    from .metrics import group_auroc, group_l2_error, group_l2_score

    r, seed, group_sizes, mu, sampler = task
    labels = (np.asarray(mu) > 0.0).astype(int)
    s = experiment_seed(5, 1, r, master_seed=seed)
    ds = generate_heterogeneity_dataset(n=300, group_sizes=group_sizes, rho_within=0.3, rho_between=0.05, sigma2=1.0, mu=mu, seed=s)
    fits = _fit_all_methods(ds["X"], ds["y"], ds["groups"], task="gaussian", seed=s, p0=int(np.sum(labels)), sampler=sampler)
    rep_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    kappa_rows: list[dict[str, Any]] = []
    for method, res in fits.items():
        is_valid = bool(res.converged and (res.beta_mean is not None))
        if not is_valid:
            rep_rows.append(
                {
                    "replicate_id": r,
                    "method": method,
                    "status": res.status,
                    "converged": bool(res.converged),
                    "null_group_mse_avg": float("nan"),
                    "signal_group_mse_avg": float("nan"),
                    "overall_mse": float("nan"),
                    "group_auroc_score": float("nan"),
                }
            )
            continue
        score = group_l2_score(res.beta_mean, ds["groups"])
        err = group_l2_error(res.beta_mean, ds["beta0"], ds["groups"])
        rep_rows.append(
            {
                "replicate_id": r,
                "method": method,
                "status": res.status,
                "converged": bool(res.converged),
                "null_group_mse_avg": float(np.mean(err[labels == 0])),
                "signal_group_mse_avg": float(np.mean(err[labels == 1])),
                "overall_mse": float(np.mean((res.beta_mean - ds["beta0"]) ** 2)),
                "group_auroc_score": group_auroc(score, labels),
            }
        )
        for gid in range(len(ds["groups"])):
            group_rows.append({"replicate_id": r, "method": method, "group_id": gid, "mu_g": float(ds["mu"][gid]), "group_score_g": float(score[gid]), "group_mse_g": float(err[gid]), "group_signal_label_g": int(labels[gid])})
        if method == "GR_RHS" and res.kappa_draws is not None:
            kd = np.asarray(res.kappa_draws, dtype=float)
            if kd.ndim > 2:
                kd = kd.reshape(-1, kd.shape[-1])
            for gid in range(kd.shape[-1]):
                kappa_rows.append({"replicate_id": r, "group_id": gid, "mu_g": float(ds["mu"][gid]), "post_mean_kappa_g": float(np.mean(kd[:, gid])), "post_median_kappa_g": float(np.median(kd[:, gid])), "post_sd_kappa_g": float(np.std(kd[:, gid])), "post_prob_kappa_gt_0_5_g": float(np.mean(kd[:, gid] > 0.5))})
    return rep_rows, group_rows, kappa_rows


def _exp6_worker(task: tuple[int, int, np.ndarray, SamplerConfig]) -> list[dict[str, Any]]:
    from .dgp_grouped_logistic import generate_grouped_logistic_dataset

    r, seed, beta0, sampler = task
    s = experiment_seed(6, 1, r, master_seed=seed)
    ds = generate_grouped_logistic_dataset(n=200, group_sizes=[5, 5, 5], rho_within=0.5, rho_between=0.05, beta0=beta0, seed=s, min_separator_auc=0.8)
    fits = _fit_all_methods(ds["X"], ds["y"], ds["groups"], task="logistic", seed=s, p0=4, sampler=sampler)
    out_rows: list[dict[str, Any]] = []
    for method, res in fits.items():
        is_valid = bool(res.converged and (res.beta_mean is not None))
        b = np.asarray(res.beta_mean, dtype=float) if res.beta_mean is not None else None
        g1 = np.asarray(ds["groups"][0], dtype=int)
        g2 = np.asarray(ds["groups"][1], dtype=int)
        k1, k2 = float("nan"), float("nan")
        if is_valid and method == "GR_RHS" and res.kappa_draws is not None:
            kd = np.asarray(res.kappa_draws, dtype=float)
            if kd.ndim > 2:
                kd = kd.reshape(-1, kd.shape[-1])
            k1 = float(np.mean(kd[:, 0] > 0.5))
            k2 = float(np.mean(kd[:, 1] > 0.5))
        out_rows.append(
            {
                "replicate_id": r,
                "method": method,
                "status": res.status,
                "converged": bool(res.converged),
                "error": res.error,
                "beta11_post_mean": float(b[0]) if is_valid and b is not None else float("nan"),
                "beta12_post_mean": float(b[1]) if is_valid and b is not None else float("nan"),
                "beta_group1_l2_norm": float(np.linalg.norm(b[g1])) if is_valid and b is not None else float("nan"),
                "beta_group2_l2_norm": float(np.linalg.norm(b[g2])) if is_valid and b is not None else float("nan"),
                "overall_runtime": res.runtime_seconds,
                "divergence_ratio": res.divergence_ratio,
                "bulk_ess_min": res.bulk_ess_min,
                "post_prob_kappa_group1_gt_0_5": k1,
                "post_prob_kappa_group2_gt_0_5": k2,
            }
        )
    return out_rows


def _exp7_worker(task: tuple[int, int, list[int], list[float], str, dict[str, Any], SamplerConfig]) -> list[dict[str, Any]]:
    from .dgp_grouped_linear import generate_heterogeneity_dataset, generate_sparse_within_group_dataset
    from .fit_gr_rhs import fit_gr_rhs
    from .fit_rhs import fit_rhs
    from .metrics import group_auroc, group_l2_error, group_l2_score

    r, seed, group_sizes, mu, dgp_type, variants, sampler = task
    labels = (np.asarray(mu) > 0).astype(int)
    s = experiment_seed(7, 1, r, master_seed=seed)
    if str(dgp_type).lower() == "sparse_within_group":
        ds = generate_sparse_within_group_dataset(
            n=300,
            group_sizes=group_sizes,
            rho_within=0.7,
            rho_between=0.05,
            sigma2=1.0,
            mu=mu,
            seed=s,
            sparsity=0.2,
        )
    else:
        ds = generate_heterogeneity_dataset(
            n=300,
            group_sizes=group_sizes,
            rho_within=0.7,
            rho_between=0.05,
            sigma2=1.0,
            mu=mu,
            seed=s,
        )
    out_rows: list[dict[str, Any]] = []
    for vname, spec in variants.items():
        if spec["method"] == "GR_RHS":
            res = fit_gr_rhs(ds["X"], ds["y"], ds["groups"], task="gaussian", seed=s + 11, p0=int(np.sum(labels)), sampler=sampler, **spec["grrhs_kwargs"])
        else:
            res = fit_rhs(ds["X"], ds["y"], ds["groups"], task="gaussian", seed=s + 12, p0=int(np.sum(labels)), sampler=sampler)
        is_valid = bool(res.converged and (res.beta_mean is not None))
        if not is_valid:
            out_rows.append(
                {
                    "replicate_id": r,
                    "dgp_type": str(dgp_type),
                    "variant": vname,
                    "status": res.status,
                    "converged": bool(res.converged),
                    "null_group_mse_avg": float("nan"),
                    "signal_group_mse_avg": float("nan"),
                    "overall_mse": float("nan"),
                    "group_auroc": float("nan"),
                }
            )
            continue
        score = group_l2_score(res.beta_mean, ds["groups"])
        err = group_l2_error(res.beta_mean, ds["beta0"], ds["groups"])
        out_rows.append(
            {
                "replicate_id": r,
                "dgp_type": str(dgp_type),
                "variant": vname,
                "status": res.status,
                "converged": bool(res.converged),
                "null_group_mse_avg": float(np.mean(err[labels == 0])),
                "signal_group_mse_avg": float(np.mean(err[labels == 1])),
                "overall_mse": float(np.mean((res.beta_mean - ds["beta0"]) ** 2)),
                "group_auroc": group_auroc(score, labels),
            }
        )
    return out_rows


def _exp9_worker(task: tuple[int, float, float, int, int, list[float], np.ndarray, list[int], str, SamplerConfig]) -> dict[str, Any]:
    from .dgp_grouped_linear import generate_heterogeneity_dataset
    from .fit_gr_rhs import fit_gr_rhs
    from .metrics import group_auroc, group_l2_error, group_l2_score

    pid, a, b, r, seed, mu, labels, group_sizes, scenario, sampler = task
    s = experiment_seed(9, pid, r, master_seed=seed)
    ds = generate_heterogeneity_dataset(n=300, group_sizes=group_sizes, rho_within=0.3, rho_between=0.05, sigma2=1.0, mu=mu, seed=s)
    res = fit_gr_rhs(ds["X"], ds["y"], ds["groups"], task="gaussian", seed=s + 7, p0=int(np.sum(labels)), sampler=sampler, alpha_kappa=a, beta_kappa=b, use_group_scale=True, shared_kappa=False)
    is_valid = bool(res.converged and (res.beta_mean is not None))
    if not is_valid:
        return {
            "alpha_kappa": a,
            "beta_kappa": b,
            "scenario": str(scenario),
            "replicate_id": r,
            "status": res.status,
            "converged": bool(res.converged),
            "null_group_mse_avg": float("nan"),
            "signal_group_mse_avg": float("nan"),
            "group_auroc": float("nan"),
        }
    score = group_l2_score(res.beta_mean, ds["groups"])
    err = group_l2_error(res.beta_mean, ds["beta0"], ds["groups"])
    return {
        "alpha_kappa": a,
        "beta_kappa": b,
        "scenario": str(scenario),
        "replicate_id": r,
        "status": res.status,
        "converged": bool(res.converged),
        "null_group_mse_avg": float(np.mean(err[labels == 0])),
        "signal_group_mse_avg": float(np.mean(err[labels == 1])),
        "group_auroc": group_auroc(score, labels),
    }


def _exp8_worker(task: tuple[int, int, int, list[int], SamplerConfig, float, float, bool]) -> dict[str, Any]:
    from .fit_gr_rhs import fit_gr_rhs
    from .utils import canonical_groups, sample_correlated_design

    p0, r, seed, group_sizes, sampler, tau_target, tau_prior_scale, use_auto = task
    n = 500
    p = int(sum(group_sizes))
    s = experiment_seed(8, int(p0), int(r), master_seed=seed)

    X, cov_x = sample_correlated_design(
        n=n,
        group_sizes=group_sizes,
        rho_within=0.3,
        rho_between=0.05,
        seed=s,
    )
    groups = canonical_groups(group_sizes)
    rng = np.random.default_rng(s + 19)
    beta = np.zeros(p, dtype=float)
    active = rng.choice(np.arange(p), size=int(p0), replace=False)
    beta[active] = 2.0
    y = X @ beta + np.random.default_rng(s + 23).normal(0.0, 1.0, size=n)
    p0_fit = int(np.sum(np.abs(beta) > 0.0))
    tau0_use = float(tau_prior_scale) * float(tau_target)

    res = fit_gr_rhs(
        X,
        y,
        groups,
        task="gaussian",
        seed=s + 31,
        p0=p0_fit,
        sampler=sampler,
        alpha_kappa=0.5,
        beta_kappa=1.0,
        use_group_scale=True,
        use_local_scale=True,
        shared_kappa=False,
        auto_calibrate_tau=bool(use_auto),
        tau0=None if use_auto else tau0_use,
    )

    valid_tau = bool(res.converged and (res.tau_draws is not None))
    tau_mean = float(np.mean(np.asarray(res.tau_draws, dtype=float))) if valid_tau else float("nan")
    tau_sd = float(np.std(np.asarray(res.tau_draws, dtype=float))) if valid_tau else float("nan")
    kappa_eff = float("nan")
    if bool(res.converged and (res.kappa_draws is not None)):
        kd = np.asarray(res.kappa_draws, dtype=float)
        if kd.ndim > 2:
            kd = kd.reshape(-1, kd.shape[-1])
        kappa_eff = float(np.mean(np.sum(kd, axis=1)))

    return {
        "p0": int(p0),
        "p": int(p),
        "n": int(n),
        "replicate_id": int(r),
        "tau_target": float(tau_target),
        "tau_prior_scale": float(tau_prior_scale),
        "tau_mode": "auto_calibrated" if bool(use_auto) else f"fixed_{tau_prior_scale:.2f}x",
        "tau_post_mean": tau_mean,
        "tau_post_sd": tau_sd,
        "kappa_eff_sum_post_mean": kappa_eff,
        "converged": bool(res.converged),
        "status": str(res.status),
        "signal_var_true": float(beta.T @ cov_x @ beta),
    }


def _fit_all_methods(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    task: str,
    seed: int,
    p0: int,
    sampler: SamplerConfig,
    grrhs_kwargs: dict[str, Any] | None = None,
) -> Dict[str, FitResult]:
    from .fit_gigg import fit_gigg_mmle
    from .fit_ghs_plus import fit_ghs_plus
    from .fit_gr_rhs import fit_gr_rhs
    from .fit_rhs import fit_rhs

    grrhs_kwargs = grrhs_kwargs or {}
    out: Dict[str, FitResult] = {}
    out["GR_RHS"] = fit_gr_rhs(X, y, groups, task=task, seed=seed + 1, p0=p0, sampler=sampler, **grrhs_kwargs)
    out["RHS"] = fit_rhs(X, y, groups, task=task, seed=seed + 2, p0=p0, sampler=sampler)
    out["GIGG_MMLE"] = fit_gigg_mmle(X, y, groups, task=task, seed=seed + 3, sampler=sampler)
    out["GHS_plus"] = fit_ghs_plus(X, y, groups, task=task, seed=seed + 4, p0=p0, sampler=sampler)
    return out


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
    ci = (beta1 - 1.96 * se, beta1 + 1.96 * se)
    return beta1, ci


def run_exp1_null_contraction(
    n_jobs: int = 1,
    seed: int = MASTER_SEED,
    repeats: int = 500,
    save_dir: str = "simulation_project",
    *,
    pg_list: Sequence[int] | None = None,
    tau_eval: float = 0.5,
    tail_M: float = 2.0,
    grid_size: int = 801,
) -> Dict[str, str]:
    from .plotting import plot_exp1

    base = Path(save_dir)
    out = ensure_dir(base / "results" / "exp1_null_contraction")
    fig_dir = ensure_dir(base / "figures")
    tab_dir = ensure_dir(base / "tables")
    log = setup_logger("exp1", base / "logs" / "exp1_null_contraction.log")

    pg_vals = list(pg_list or [10, 20, 50, 100, 200, 500])
    tasks: list[tuple[int, int, int, int, float, float, int]] = []
    for sid, pg in enumerate(pg_vals, start=1):
        tasks.append((sid, pg, int(repeats), seed, float(tau_eval), float(tail_M), int(grid_size)))
    rows_nested = _parallel_rows(tasks, _exp1_setting_worker, n_jobs=n_jobs, prefer_process=False, progress_desc="Exp1 Null Contraction")
    rows: list[dict[str, Any]] = []
    for chunk in rows_nested:
        rows.extend(chunk)

    agg_rows: list[dict[str, Any]] = []
    for pg in sorted({int(r["p_g"]) for r in rows}):
        sub = [r for r in rows if int(r["p_g"]) == pg]
        post = np.asarray([float(r["post_mean_kappa"]) for r in sub], dtype=float)
        tail = np.asarray([float(r["tail_prob_kappa_gt_2_over_sqrt_pg"]) for r in sub], dtype=float)
        agg_rows.append(
            {
                "p_g": pg,
                "median_post_mean_kappa": float(np.median(post)),
                "iqr_post_mean_kappa_low": float(np.quantile(post, 0.25)),
                "iqr_post_mean_kappa_high": float(np.quantile(post, 0.75)),
                "mean_tail_prob": float(np.mean(tail)),
            }
        )
    x = np.log(np.asarray([float(r["p_g"]) for r in agg_rows], dtype=float))
    y = np.log(np.asarray([float(r["median_post_mean_kappa"]) for r in agg_rows], dtype=float))
    slope, slope_ci = _linreg_slope_ci(x, y)
    _save_rows_csv(rows, out / "raw_results.csv")
    _save_rows_csv(agg_rows, out / "summary.csv")
    agg_tau_rows: list[dict[str, Any]] = []
    for tau_eval in sorted({float(r["tau_eval"]) for r in rows}):
        for pg in sorted({int(r["p_g"]) for r in rows}):
            sub = [r for r in rows if int(r["p_g"]) == pg and float(r["tau_eval"]) == tau_eval]
            if not sub:
                continue
            ratio = np.asarray([float(r["ratio_R"]) for r in sub], dtype=float)
            win = np.asarray([float(r["post_prob_kappa_in_scale_window"]) for r in sub], dtype=float)
            agg_tau_rows.append(
                {
                    "tau_eval": tau_eval,
                    "p_g": pg,
                    "median_ratio_R": float(np.median(ratio)),
                    "iqr_ratio_R_low": float(np.quantile(ratio, 0.25)),
                    "iqr_ratio_R_high": float(np.quantile(ratio, 0.75)),
                    "mean_window_prob": float(np.mean(win)),
                }
            )
    _save_rows_csv(agg_tau_rows, out / "summary_tau_sweep.csv")
    save_json({"tau_eval": float(tau_eval), "tail_M": float(tail_M), "tail_threshold_formula": "M/sqrt(p_g)", "p_g_list": pg_vals, "grid_size": int(grid_size)}, out / "exp1_meta.json")
    (tab_dir / "table_exp1_slope.txt").write_text(f"slope={slope:.6f}, ci95=({slope_ci[0]:.6f},{slope_ci[1]:.6f})\n", encoding="utf-8")
    plot_exp1(agg_rows, slope=slope, slope_ci=slope_ci, out_path=fig_dir / "fig1_null_contraction.png")
    log.info("Completed exp1 with repeats=%d", repeats)
    return {"raw": str(out / "raw_results.csv"), "summary": str(out / "summary.csv"), "figure": str(fig_dir / "fig1_null_contraction.png")}


def run_exp2_adaptive_localization(
    n_jobs: int = 1,
    seed: int = MASTER_SEED,
    repeats: int = 500,
    save_dir: str = "simulation_project",
    *,
    pg_list: Sequence[int] | None = None,
    tau_list: Sequence[float] | None = None,
    c_window: float = 3.0,
    grid_size: int = 801,
) -> Dict[str, str]:
    from .plotting import plot_exp2

    base = Path(save_dir)
    out = ensure_dir(base / "results" / "exp2_adaptive_localization")
    fig_dir = ensure_dir(base / "figures")
    log = setup_logger("exp2", base / "logs" / "exp2_adaptive_localization.log")
    pg_vals = list(pg_list or [20, 50, 100, 200, 500])
    tau_vals = list(tau_list or [0.5, 1.0, 2.0])
    tasks: list[tuple[int, int, int, float, float, int, float, float, float, int]] = []
    sid = 0
    for tau_eval in tau_vals:
        for pg in pg_vals:
            sid += 1
            mu_g = 2.0 * (pg ** 0.75)
            scale = mu_g / pg
            win_lo = max(scale * (1.0 - float(c_window) / math.sqrt(pg)), 1e-4)
            win_hi = min(scale * (1.0 + float(c_window) / math.sqrt(pg)), 1.0 - 1e-4)
            tasks.append((sid, pg, int(repeats), mu_g, scale, seed, float(tau_eval), float(win_lo), float(win_hi), int(grid_size)))
    rows_nested = _parallel_rows(tasks, _exp2_setting_worker, n_jobs=n_jobs, prefer_process=False, progress_desc="Exp2 Adaptive Localization")
    rows: list[dict[str, Any]] = []
    for chunk in rows_nested:
        rows.extend(chunk)
    agg_rows: list[dict[str, Any]] = []
    for pg in sorted({int(r["p_g"]) for r in rows}):
        sub = [r for r in rows if int(r["p_g"]) == pg]
        ratio = np.asarray([float(r["ratio_R"]) for r in sub], dtype=float)
        win = np.asarray([float(r["post_prob_kappa_in_scale_window"]) for r in sub], dtype=float)
        agg_rows.append(
            {
                "p_g": pg,
                "median_ratio_R": float(np.median(ratio)),
                "iqr_ratio_R_low": float(np.quantile(ratio, 0.25)),
                "iqr_ratio_R_high": float(np.quantile(ratio, 0.75)),
                "mean_window_prob": float(np.mean(win)),
            }
        )
    _save_rows_csv(rows, out / "raw_results.csv")
    _save_rows_csv(agg_rows, out / "summary.csv")
    save_json({"tau_list": [float(t) for t in tau_vals], "window_rule": "s_g * (1 ± C/sqrt(p_g))", "c_window": float(c_window), "p_g_list": pg_vals, "grid_size": int(grid_size)}, out / "exp2_meta.json")
    plot_exp2(agg_rows, out_path=fig_dir / "fig2_adaptive_localization.png")
    log.info("Completed exp2 with repeats=%d", repeats)
    return {"raw": str(out / "raw_results.csv"), "summary": str(out / "summary.csv"), "figure": str(fig_dir / "fig2_adaptive_localization.png")}


def run_exp3_phase_diagram(
    n_jobs: int = 1,
    seed: int = MASTER_SEED,
    repeats: int = 200,
    save_dir: str = "simulation_project",
    *,
    p_g_list: Sequence[int] | None = None,
    tau_list: Sequence[float] | None = None,
    xi_multiplier_list: Sequence[float] | None = None,
    u0: float = 0.5,
    sigma2: float = 1.0,
    theory_check_tau: float = 0.3,
    grid_size: int = 801,
) -> Dict[str, str]:
    from .plotting import plot_exp3_curves, plot_exp3_heatmap

    base = Path(save_dir)
    out = ensure_dir(base / "results" / "exp3_phase_diagram")
    fig_dir = ensure_dir(base / "figures")
    log = setup_logger("exp3", base / "logs" / "exp3_phase_diagram.log")
    tau_vals = list(tau_list or [0.1, 0.2, 0.3, 0.5, 1.0])
    xi_mults = list(xi_multiplier_list or [0.3, 0.5, 0.7, 0.85, 0.95, 1.05, 1.15, 1.3, 1.5, 2.0])
    pg_vals = list(p_g_list or [30, 60, 120])
    tasks: list[tuple[int, int, int, float, int, int, float, float, float, int]] = []
    xi_by_tau: dict[float, list[float]] = {}
    sid = 0
    for tau in tau_vals:
        xi_crit_tau = xi_crit_u0_rho(u0=u0, rho=tau / math.sqrt(sigma2))
        xi_vals = [float(xi_crit_tau * m) for m in xi_mults]
        xi_by_tau[float(tau)] = xi_vals
        for pg in pg_vals:
            sid += 1
            for xid, xi in enumerate(xi_vals, start=1):
                tasks.append((sid, pg, xid, xi, int(repeats), seed, float(tau), sigma2, u0, int(grid_size)))
    rows_nested = _parallel_rows(tasks, _exp3_setting_worker, n_jobs=n_jobs, prefer_process=False, progress_desc="Exp3 Phase Diagram")
    rows: list[dict[str, Any]] = []
    for chunk in rows_nested:
        rows.extend(chunk)
    agg_rows: list[dict[str, Any]] = []
    keys = sorted({(float(r["xi"]), int(r["p_g"]), float(r["tau"])) for r in rows}, key=lambda z: (z[2], z[1], z[0]))
    for xi, pg, tau in keys:
        sub = [r for r in rows if float(r["xi"]) == xi and int(r["p_g"]) == pg and float(r["tau"]) == tau]
        prob = np.asarray([float(r["post_prob_kappa_gt_u0"]) for r in sub], dtype=float)
        agg_rows.append({"tau": tau, "xi": xi, "p_g": pg, "mean_prob_gt_u0": float(np.mean(prob)), "sd_prob_gt_u0": float(np.std(prob, ddof=1))})
    _save_rows_csv(rows, out / "raw_results.csv")
    _save_rows_csv(agg_rows, out / "summary_tau_sweep.csv")
    tau_ref = float(theory_check_tau if theory_check_tau in tau_vals else tau_vals[0])
    ref_rows = [r for r in rows if float(r["tau"]) == tau_ref]
    agg_ref: list[dict[str, Any]] = []
    keys_ref = sorted({(float(r["xi"]), int(r["p_g"])) for r in ref_rows}, key=lambda z: (z[1], z[0]))
    for xi, pg in keys_ref:
        sub = [r for r in ref_rows if float(r["xi"]) == xi and int(r["p_g"]) == pg]
        prob = np.asarray([float(r["post_prob_kappa_gt_u0"]) for r in sub], dtype=float)
        agg_ref.append({"xi": xi, "p_g": pg, "mean_prob_gt_u0": float(np.mean(prob)), "sd_prob_gt_u0": float(np.std(prob, ddof=1))})
    _save_rows_csv(agg_ref, out / "summary.csv")
    xi_crit_ref = xi_crit_u0_rho(u0=u0, rho=tau_ref / math.sqrt(sigma2))
    theta_ref = theta_u0_rho(u0=u0, rho=tau_ref / math.sqrt(sigma2))
    plot_exp3_heatmap(agg_ref, out_path=fig_dir / "fig3_phase_heatmap.png")
    plot_exp3_curves(agg_ref, xi_crit=xi_crit_ref, out_path=fig_dir / "fig3_phase_curves.png")
    save_json(
        {
            "u0": u0,
            "sigma2": sigma2,
            "tau": tau_ref,
            "rho": tau_ref / math.sqrt(sigma2),
            "theta_u0_rho": theta_ref,
            "xi_crit": xi_crit_ref,
            "p_g_list": pg_vals,
            "tau_list": [float(t) for t in tau_vals],
            "xi_by_tau": {f"{k:.6g}": v for k, v in xi_by_tau.items()},
            "grid_size": int(grid_size),
        },
        out / "phase_threshold_meta.json",
    )
    log.info("Completed exp3 with repeats=%d", repeats)
    return {"raw": str(out / "raw_results.csv"), "summary": str(out / "summary.csv")}


def _evaluate_method_row(result: FitResult, beta0: np.ndarray) -> dict[str, float]:
    from .metrics import ci_length_and_coverage, mse_null_signal_overall

    if (result.beta_mean is None) or (not result.converged):
        return {"mse_null": float("nan"), "mse_signal": float("nan"), "mse_overall": float("nan"), "avg_ci_length": float("nan"), "coverage_95": float("nan")}
    m = mse_null_signal_overall(result.beta_mean, beta0)
    ci_len, cov = ci_length_and_coverage(beta0, result.beta_draws)
    return {"mse_null": m["mse_null"], "mse_signal": m["mse_signal"], "mse_overall": m["mse_overall"], "avg_ci_length": ci_len, "coverage_95": cov}


def run_exp4_benchmark_linear(n_jobs: int = 1, seed: int = MASTER_SEED, repeats: int = 50, save_dir: str = "simulation_project") -> Dict[str, str]:
    import pandas as pd
    from .plotting import plot_exp4_overall_mse
    base = Path(save_dir)
    out = ensure_dir(base / "results" / "exp4_benchmark_linear")
    fig_dir = ensure_dir(base / "figures")
    tab_dir = ensure_dir(base / "tables")
    log = setup_logger("exp4", base / "logs" / "exp4_benchmark_linear.log")
    sampler = SamplerConfig()
    settings = {
        "L0": {"group_sizes": [10, 10, 10, 10, 10], "rho_within": 0.0, "rho_between": 0.0, "design_type": "orthonormal"},
        "L1": {"group_sizes": [10, 10, 10, 10, 10], "rho_within": 0.3, "rho_between": 0.10, "design_type": "correlated"},
        "L2": {"group_sizes": [10, 10, 10, 10, 10], "rho_within": 0.3, "rho_between": 0.10, "design_type": "correlated"},
        "L3": {"group_sizes": [10, 10, 10, 10, 10], "rho_within": 0.8, "rho_between": 0.10, "design_type": "correlated"},
        "L4": {"group_sizes": [10, 10, 10, 10, 10], "rho_within": 0.8, "rho_between": 0.10, "design_type": "correlated"},
        "L5": {"group_sizes": [30, 10, 5, 3, 2], "rho_within": 0.3, "rho_between": 0.10, "design_type": "correlated"},
    }
    tasks: list[tuple[int, str, dict[str, Any], int, int, SamplerConfig]] = []
    for sid, (setting, spec) in enumerate(settings.items(), start=1):
        for r in range(1, int(repeats) + 1):
            tasks.append((sid, setting, spec, r, seed, sampler))

    rows = []
    for chunk in _parallel_rows(tasks, _exp4_worker, n_jobs=n_jobs, prefer_process=True, progress_desc="Exp4 Benchmark Linear"):
        rows.extend(chunk)
    raw = pd.DataFrame(rows)
    summary = raw.groupby(["setting", "method"], as_index=False).agg(mse_null=("mse_null", "mean"), mse_signal=("mse_signal", "mean"), mse_overall=("mse_overall", "mean"), avg_ci_length=("avg_ci_length", "mean"), coverage_95=("coverage_95", "mean"), n_effective=("converged", "sum"))
    save_dataframe(raw, out / "raw_results.csv")
    save_dataframe(summary, out / "summary.csv")
    save_dataframe(summary, tab_dir / "table_benchmark_linear.csv")
    plot_exp4_overall_mse(summary, out_path=fig_dir / "fig4_benchmark_overall_mse.png")
    log.info("Completed exp4 with repeats=%d", repeats)
    return {"raw": str(out / "raw_results.csv"), "table": str(tab_dir / "table_benchmark_linear.csv")}


def run_exp5_heterogeneity(n_jobs: int = 1, seed: int = MASTER_SEED, repeats: int = 100, save_dir: str = "simulation_project") -> Dict[str, str]:
    import pandas as pd
    from .plotting import plot_exp5_group_ranking, plot_exp5_kappa_stratification, plot_exp5_null_signal_mse
    base = Path(save_dir)
    out = ensure_dir(base / "results" / "exp5_heterogeneity")
    fig_dir = ensure_dir(base / "figures")
    tab_dir = ensure_dir(base / "tables")
    log = setup_logger("exp5", base / "logs" / "exp5_heterogeneity.log")
    sampler = SamplerConfig()
    sigma2 = 1.0
    tau_ref = 0.1
    group_sizes = [50, 50, 20, 10, 10, 10]
    xi_boundary = xi_crit_u0_rho(u0=0.5, rho=tau_ref / math.sqrt(sigma2))
    mu_boundary = 1.2 * xi_boundary * group_sizes[2]
    mu = [0.0, 0.0, float(mu_boundary), 2.0, 8.0, 25.0]
    labels = (np.asarray(mu) > 0.0).astype(int)
    tasks = [(r, seed, group_sizes, mu, sampler) for r in range(1, int(repeats) + 1)]

    row_rep, row_group, row_grrhs_kappa = [], [], []
    for rep_rows, group_rows, kappa_rows in _parallel_rows(tasks, _exp5_worker, n_jobs=n_jobs, prefer_process=True, progress_desc="Exp5 Heterogeneity"):
        row_rep.extend(rep_rows)
        row_group.extend(group_rows)
        row_grrhs_kappa.extend(kappa_rows)
    raw_rep = pd.DataFrame(row_rep)
    raw_group = pd.DataFrame(row_group)
    raw_kappa = pd.DataFrame(row_grrhs_kappa)
    raw = raw_rep.merge(raw_group, on=["replicate_id", "method"], how="left")
    auroc_table = raw_rep.groupby("method", as_index=False).agg(
        group_auroc=("group_auroc_score", "mean"),
        avg_null_group_mse=("null_group_mse_avg", "mean"),
        avg_signal_group_mse=("signal_group_mse_avg", "mean"),
        n_effective=("converged", "sum"),
    )
    save_dataframe(raw, out / "raw_results.csv")
    save_dataframe(raw_rep, out / "summary_replicate.csv")
    save_dataframe(raw_kappa, out / "summary_kappa.csv")
    save_dataframe(auroc_table, tab_dir / "table_heterogeneity_auroc.csv")
    if not raw_kappa.empty:
        plot_exp5_kappa_stratification(raw_kappa, out_path=fig_dir / "fig5_kappa_stratification.png")
        plot_exp5_group_ranking(raw_kappa, auroc_table.loc[auroc_table["method"].isin(METHODS)], out_path=fig_dir / "fig5_group_ranking.png")
    plot_exp5_null_signal_mse(auroc_table.loc[auroc_table["method"].isin(METHODS)], out_path=fig_dir / "fig5_null_signal_mse.png")
    log.info("Completed exp5 with repeats=%d", repeats)
    return {"raw": str(out / "raw_results.csv"), "table": str(tab_dir / "table_heterogeneity_auroc.csv")}


def run_exp6_grouped_logistic(n_jobs: int = 1, seed: int = MASTER_SEED, repeats: int = 50, save_dir: str = "simulation_project") -> Dict[str, str]:
    import pandas as pd
    from .plotting import plot_exp6_coefficients, plot_exp6_diagnostics, plot_exp6_kappa, plot_exp6_null_group
    base = Path(save_dir)
    out = ensure_dir(base / "results" / "exp6_grouped_logistic")
    fig_dir = ensure_dir(base / "figures")
    log = setup_logger("exp6", base / "logs" / "exp6_grouped_logistic.log")
    sampler = SamplerConfig()
    beta0 = np.array([1.5, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0], dtype=float)
    tasks = [(r, seed, beta0, sampler) for r in range(1, int(repeats) + 1)]

    rows = []
    for chunk in _parallel_rows(tasks, _exp6_worker, n_jobs=n_jobs, prefer_process=True, progress_desc="Exp6 Grouped Logistic"):
        rows.extend(chunk)
    raw = pd.DataFrame(rows)
    save_dataframe(raw, out / "raw_results.csv")
    summary = raw.groupby("method", as_index=False).agg(
        beta11_post_mean=("beta11_post_mean", "mean"),
        beta12_post_mean=("beta12_post_mean", "mean"),
        beta_group1_l2_norm=("beta_group1_l2_norm", "mean"),
        beta_group2_l2_norm=("beta_group2_l2_norm", "mean"),
        overall_runtime=("overall_runtime", "mean"),
        divergence_ratio=("divergence_ratio", "mean"),
        bulk_ess_min=("bulk_ess_min", "mean"),
        post_prob_kappa_group1_gt_0_5=("post_prob_kappa_group1_gt_0_5", "mean"),
        post_prob_kappa_group2_gt_0_5=("post_prob_kappa_group2_gt_0_5", "mean"),
        n_effective=("converged", "sum"),
    )
    save_dataframe(summary, out / "summary.csv")
    ok = raw.loc[raw["converged"] == True].copy()
    if not ok.empty:
        plot_exp6_coefficients(ok, out_path=fig_dir / "fig6_logistic_coefficients.png")
        plot_exp6_null_group(ok, out_path=fig_dir / "fig6_logistic_null_group.png")
        plot_exp6_diagnostics(ok, out_path=fig_dir / "fig6_logistic_diagnostics.png")
        plot_exp6_kappa(ok, out_path=fig_dir / "fig6_kappa_logistic.png")
    log.info("Completed exp6 with repeats=%d", repeats)
    return {"raw": str(out / "raw_results.csv")}


def run_exp7_ablation(n_jobs: int = 1, seed: int = MASTER_SEED, repeats: int = 50, save_dir: str = "simulation_project") -> Dict[str, str]:
    import pandas as pd
    base = Path(save_dir)
    out = ensure_dir(base / "results" / "exp7_ablation")
    tab_dir = ensure_dir(base / "tables")
    log = setup_logger("exp7", base / "logs" / "exp7_ablation.log")
    sampler = SamplerConfig()
    mu = [0.0, 0.0, 2.0, 8.0, 25.0, 80.0]
    variants = {
        "GR_RHS_full": {"grrhs_kwargs": {"alpha_kappa": 0.5, "beta_kappa": 1.0, "use_group_scale": True, "shared_kappa": False}, "method": "GR_RHS"},
        "GR_RHS_no_ag": {"grrhs_kwargs": {"alpha_kappa": 0.5, "beta_kappa": 1.0, "use_group_scale": False, "shared_kappa": False}, "method": "GR_RHS"},
        "GR_RHS_no_local_scales": {"grrhs_kwargs": {"alpha_kappa": 0.5, "beta_kappa": 1.0, "use_group_scale": True, "use_local_scale": False, "shared_kappa": False}, "method": "GR_RHS"},
        "GR_RHS_shared_kappa": {"grrhs_kwargs": {"alpha_kappa": 0.5, "beta_kappa": 1.0, "use_group_scale": True, "shared_kappa": True}, "method": "GR_RHS"},
        "RHS": {"grrhs_kwargs": {}, "method": "RHS"},
    }
    dgp_types = ["dense_uniform", "sparse_within_group"]
    tasks = [
        (r, seed, [10, 10, 10, 10, 10, 10], mu, dgp_type, variants, sampler)
        for dgp_type in dgp_types
        for r in range(1, int(repeats) + 1)
    ]

    rows = []
    for chunk in _parallel_rows(tasks, _exp7_worker, n_jobs=n_jobs, prefer_process=True, progress_desc="Exp7 Ablation"):
        rows.extend(chunk)
    raw = pd.DataFrame(rows)
    table = raw.groupby(["dgp_type", "variant"], as_index=False).agg(
        null_group_mse_avg=("null_group_mse_avg", "mean"),
        signal_group_mse_avg=("signal_group_mse_avg", "mean"),
        overall_mse=("overall_mse", "mean"),
        group_auroc=("group_auroc", "mean"),
        n_effective=("converged", "sum"),
    )
    save_dataframe(raw, out / "raw_results.csv")
    save_dataframe(table, tab_dir / "table_ablation.csv")
    save_dataframe(table, out / "summary.csv")
    log.info("Completed exp7 with repeats=%d", repeats)
    return {"raw": str(out / "raw_results.csv"), "table": str(tab_dir / "table_ablation.csv")}


def run_exp8_tau_calibration(n_jobs: int = 1, seed: int = MASTER_SEED, repeats: int = 100, save_dir: str = "simulation_project") -> Dict[str, str]:
    import pandas as pd
    from .plotting import plot_exp8_tau
    base = Path(save_dir)
    out = ensure_dir(base / "results" / "exp8_tau_calibration")
    fig_dir = ensure_dir(base / "figures")
    log = setup_logger("exp8", base / "logs" / "exp8_tau_calibration.log")
    sampler = SamplerConfig(chains=2, warmup=300, post_warmup_draws=300)

    n = 500
    group_sizes = [10, 10, 10, 10, 10, 10]
    p = int(sum(group_sizes))
    p0_list = [2, 6, 12, 30]
    tau_scales = [0.5, 1.0, 2.0]
    tasks: list[tuple[int, int, int, list[int], SamplerConfig, float, float, bool]] = []
    for p0 in p0_list:
        tau_target = p0 / ((p - p0) * math.sqrt(n))
        for r in range(1, int(repeats) + 1):
            tasks.append((int(p0), int(r), seed, group_sizes, sampler, float(tau_target), 1.0, True))
            for sc in tau_scales:
                tasks.append((int(p0), int(r), seed, group_sizes, sampler, float(tau_target), float(sc), False))

    rows = _parallel_rows(tasks, _exp8_worker, n_jobs=n_jobs, prefer_process=True, progress_desc="Exp8 Tau Calibration")
    raw = pd.DataFrame(rows)
    raw["tau_abs_error"] = (raw["tau_post_mean"] - raw["tau_target"]).abs()
    raw["tau_rel_error"] = raw["tau_abs_error"] / raw["tau_target"].clip(lower=1e-12)
    summary = raw.groupby(["p0", "tau_mode"], as_index=False).agg(
        tau_target=("tau_target", "mean"),
        tau_post_mean=("tau_post_mean", "mean"),
        tau_post_sd=("tau_post_sd", "mean"),
        tau_abs_error=("tau_abs_error", "mean"),
        tau_rel_error=("tau_rel_error", "mean"),
        kappa_eff_sum_post_mean=("kappa_eff_sum_post_mean", "mean"),
        n_effective=("converged", "sum"),
    )
    save_dataframe(raw, out / "raw_results.csv")
    save_dataframe(summary, out / "summary.csv")
    plot_exp8_tau(raw, out_path=fig_dir / "fig7_tau_calibration.png")
    save_json({"n": int(n), "p": int(p), "group_sizes": list(group_sizes), "p0_list": list(p0_list), "tau_scales_fixed": list(tau_scales)}, out / "exp8_meta.json")
    log.info("Completed exp8 with repeats=%d", repeats)
    return {"raw": str(out / "raw_results.csv"), "summary": str(out / "summary.csv"), "figure": str(fig_dir / "fig7_tau_calibration.png")}


def run_exp9_beta_prior_sensitivity(n_jobs: int = 1, seed: int = MASTER_SEED, repeats: int = 30, save_dir: str = "simulation_project") -> Dict[str, str]:
    import pandas as pd
    base = Path(save_dir)
    out = ensure_dir(base / "results" / "exp9_beta_prior_sensitivity")
    tab_dir = ensure_dir(base / "tables")
    log = setup_logger("exp9", base / "logs" / "exp9_beta_prior_sensitivity.log")
    sampler = SamplerConfig()
    scenarios = [
        ("baseline", [0.0, 0.0, 2.0, 8.0, 25.0, 80.0], [4, 4, 4, 4, 4, 4]),
        ("tail_extreme", [0.0, 0.0, 0.0, 0.0, 0.0, 200.0], [4, 4, 4, 4, 4, 4]),
    ]
    priors = [(0.5, 0.5), (1.0, 1.0), (1.0, 2.0), (0.5, 1.0), (2.5, 1.0)]
    tasks: list[tuple[int, float, float, int, int, list[float], np.ndarray, list[int], str, SamplerConfig]] = []
    for sid, (scenario, mu, group_sizes) in enumerate(scenarios, start=1):
        labels = (np.asarray(mu) > 0).astype(int)
        for pid, (a, b) in enumerate(priors, start=1):
            for r in range(1, int(repeats) + 1):
                task_id = sid * 100 + pid
                tasks.append((task_id, a, b, r, seed, mu, labels, group_sizes, scenario, sampler))
    rows = _parallel_rows(tasks, _exp9_worker, n_jobs=n_jobs, prefer_process=True, progress_desc="Exp9 Beta Prior Sensitivity")
    raw = pd.DataFrame(rows)
    table = raw.groupby(["scenario", "alpha_kappa", "beta_kappa"], as_index=False).agg(
        null_group_mse_avg=("null_group_mse_avg", "mean"),
        signal_group_mse_avg=("signal_group_mse_avg", "mean"),
        group_auroc=("group_auroc", "mean"),
        n_effective=("converged", "sum"),
    )
    save_dataframe(raw, out / "raw_results.csv")
    save_dataframe(table, out / "summary.csv")
    save_dataframe(table, tab_dir / "table_beta_prior_sensitivity.csv")
    log.info("Completed exp9 with repeats=%d", repeats)
    return {"raw": str(out / "raw_results.csv"), "table": str(tab_dir / "table_beta_prior_sensitivity.csv")}


def run_all(save_dir: str = "simulation_project", seed: int = MASTER_SEED, n_jobs: int = 1) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    jobs = [
        ("exp1", run_exp1_null_contraction),
        ("exp2", run_exp2_adaptive_localization),
        ("exp3", run_exp3_phase_diagram),
        ("exp4", run_exp4_benchmark_linear),
        ("exp5", run_exp5_heterogeneity),
        ("exp6", run_exp6_grouped_logistic),
        ("exp7", run_exp7_ablation),
        ("exp8", run_exp8_tau_calibration),
        ("exp9", run_exp9_beta_prior_sensitivity),
    ]
    for name, fn in tqdm(jobs, total=len(jobs), desc="All Experiments", leave=True):
        out[name] = fn(save_dir=save_dir, seed=seed, n_jobs=n_jobs)
    save_json(out, Path(save_dir) / "results" / "run_manifest.json")
    return out


def run_theory_check(save_dir: str = "simulation_project") -> Dict[str, str]:
    from .theory_validation import validate_theory_results, write_theory_report

    result = validate_theory_results(save_dir=save_dir)
    paths = write_theory_report(result, save_dir=save_dir)
    return {"overall": "PASS" if result.pass_all else "FAIL", **paths}


def run_theory_auto_debug(
    save_dir: str = "simulation_project",
    seed: int = MASTER_SEED,
    n_jobs: int = 1,
    repeats1: int = 500,
    repeats2: int = 500,
    repeats3: int = 200,
    max_attempts: int = 8,
) -> Dict[str, Any]:
    from .theory_validation import validate_theory_results, write_theory_report

    base = Path(save_dir)
    log = setup_logger("theory_auto_debug", base / "logs" / "theory_auto_debug.log")
    attempts = []
    exp1_pg_candidates = [
        [10, 20, 50, 100, 200, 500],
        [10, 20, 50, 100, 200, 500, 1000, 2000],
        [10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
    ]
    exp1_tau_candidates = [0.5, 1.0]
    exp1_m_candidates = [2.0, 3.0, 4.0, 6.0]
    exp2_tau_candidates = [1.5, 2.0, 3.0]
    exp3_pg_candidates = [
        [30, 60, 120],
        [30, 60, 120, 240, 480],
        [30, 60, 120, 240, 480, 960],
    ]
    aid = 0
    for pg1 in exp1_pg_candidates:
        for tau1 in exp1_tau_candidates:
            for m1 in exp1_m_candidates:
                for tau2 in exp2_tau_candidates:
                    for pg3 in exp3_pg_candidates:
                        aid += 1
                        cfg = {
                            "attempt": aid,
                            "exp1": {"pg_list": pg1, "tau_eval": tau1, "tail_M": m1},
                            "exp2": {"tau_list": [tau2]},
                            "exp3": {"p_g_list": pg3},
                        }
                        log.info("Auto-debug attempt %d | cfg=%s", aid, cfg)
                        run_exp1_null_contraction(
                            n_jobs=n_jobs,
                            seed=seed,
                            repeats=repeats1,
                            save_dir=save_dir,
                            pg_list=pg1,
                            tau_eval=tau1,
                            tail_M=m1,
                        )
                        run_exp2_adaptive_localization(
                            n_jobs=n_jobs,
                            seed=seed,
                            repeats=repeats2,
                            save_dir=save_dir,
                            tau_list=[tau2],
                        )
                        run_exp3_phase_diagram(
                            n_jobs=n_jobs,
                            seed=seed,
                            repeats=repeats3,
                            save_dir=save_dir,
                            p_g_list=pg3,
                        )
                        chk = validate_theory_results(save_dir=save_dir)
                        attempts.append({"config": cfg, "pass_all": chk.pass_all, "checks": chk.checks, "metrics": chk.metrics})
                        if chk.pass_all:
                            rpt = write_theory_report(chk, save_dir=save_dir)
                            save_json({"selected": cfg, "attempts": attempts, "status": "PASS", "report": rpt}, base / "results" / "theory_validation" / "auto_debug_trace.json")
                            return {"status": "PASS", "attempt": aid, **rpt}
                        if aid >= int(max_attempts):
                            rpt = write_theory_report(chk, save_dir=save_dir)
                            save_json({"selected": cfg, "attempts": attempts, "status": "FAIL_MAX_ATTEMPTS", "report": rpt}, base / "results" / "theory_validation" / "auto_debug_trace.json")
                            return {"status": "FAIL_MAX_ATTEMPTS", "attempt": aid, **rpt}
    chk = validate_theory_results(save_dir=save_dir)
    rpt = write_theory_report(chk, save_dir=save_dir)
    save_json({"attempts": attempts, "status": "FAIL_SPACE_EXHAUSTED", "report": rpt}, base / "results" / "theory_validation" / "auto_debug_trace.json")
    return {"status": "FAIL_SPACE_EXHAUSTED", **rpt}


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run GR-RHS simulation pipeline")
    parser.add_argument("--experiment", default="all", choices=["all", "1", "2", "3", "4", "5", "6", "7", "8", "9", "theory-check", "theory-auto"])
    parser.add_argument("--save-dir", default="simulation_project")
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=8)
    args = parser.parse_args()

    if args.experiment == "all":
        run_all(save_dir=args.save_dir, seed=args.seed, n_jobs=args.n_jobs)
    elif args.experiment == "1":
        run_exp1_null_contraction(repeats=args.repeats or 500, save_dir=args.save_dir, seed=args.seed, n_jobs=args.n_jobs)
    elif args.experiment == "2":
        run_exp2_adaptive_localization(repeats=args.repeats or 500, save_dir=args.save_dir, seed=args.seed, n_jobs=args.n_jobs)
    elif args.experiment == "3":
        run_exp3_phase_diagram(repeats=args.repeats or 200, save_dir=args.save_dir, seed=args.seed, n_jobs=args.n_jobs)
    elif args.experiment == "4":
        run_exp4_benchmark_linear(repeats=args.repeats or 50, save_dir=args.save_dir, seed=args.seed, n_jobs=args.n_jobs)
    elif args.experiment == "5":
        run_exp5_heterogeneity(repeats=args.repeats or 100, save_dir=args.save_dir, seed=args.seed, n_jobs=args.n_jobs)
    elif args.experiment == "6":
        run_exp6_grouped_logistic(repeats=args.repeats or 50, save_dir=args.save_dir, seed=args.seed, n_jobs=args.n_jobs)
    elif args.experiment == "7":
        run_exp7_ablation(repeats=args.repeats or 50, save_dir=args.save_dir, seed=args.seed, n_jobs=args.n_jobs)
    elif args.experiment == "8":
        run_exp8_tau_calibration(repeats=args.repeats or 100, save_dir=args.save_dir, seed=args.seed, n_jobs=args.n_jobs)
    elif args.experiment == "9":
        run_exp9_beta_prior_sensitivity(repeats=args.repeats or 30, save_dir=args.save_dir, seed=args.seed, n_jobs=args.n_jobs)
    elif args.experiment == "theory-check":
        run_theory_check(save_dir=args.save_dir)
    elif args.experiment == "theory-auto":
        run_theory_auto_debug(
            save_dir=args.save_dir,
            seed=args.seed,
            n_jobs=args.n_jobs,
            repeats1=args.repeats or 500,
            repeats2=args.repeats or 500,
            repeats3=200,
            max_attempts=args.max_attempts,
        )


if __name__ == "__main__":
    _cli()
