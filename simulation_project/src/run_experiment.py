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
    # θ(u₀, ρ) = u₀ρ² / (u₀ + (1−u₀)ρ²)  ——  定理 3.32 的关键量
    u = float(u0)
    rho2 = float(rho) ** 2
    den = u + (1.0 - u) * rho2
    return float((u * rho2) / max(den, 1e-12))


def xi_crit_u0_rho(u0: float, rho: float) -> float:
    # ξ_crit = θ(u₀,ρ)/2  ——  相变临界值（定理 3.32/Corollary 3.33）
    # ξ > ξ_crit → P(κ > u₀|Y) → 1；ξ < ξ_crit → P(κ > u₀|Y) → 0
    return 0.5 * theta_u0_rho(u0=u0, rho=rho)


def _eps_tag(eps: float) -> str:
    return format(float(eps), ".6g").replace("-", "m").replace(".", "_")


def _tail_prob_col_name(eps: float) -> str:
    return f"tail_prob_kappa_gt_{_eps_tag(float(eps))}"


def _tail_prob_from_grid(kappa_grid: np.ndarray, density: np.ndarray, threshold: float) -> float:
    g = np.asarray(kappa_grid, dtype=float)
    d = np.asarray(density, dtype=float)
    mask = g > float(threshold)
    return float(np.trapezoid(d[mask], g[mask])) if np.any(mask) else 0.0


def _exp1_worker(task: tuple[int, int, int, int, float, tuple[float, ...], float, float]) -> dict[str, Any]:
    sid, pg, r, seed, tau_eval, tail_eps, alpha_kappa, beta_kappa = task
    s = experiment_seed(1, sid, r, master_seed=seed)
    y = generate_null_group(pg=pg, sigma2=1.0, seed=s)
    grid = kappa_posterior_grid(y, tau=tau_eval, sigma2=1.0, alpha_kappa=alpha_kappa, beta_kappa=beta_kappa)
    summary = posterior_summary_from_grid(grid["kappa"], grid["density"])
    row: dict[str, Any] = {
        "p_g": pg,
        "setting_id": sid,
        "replicate_id": r,
        "tau_eval": float(tau_eval),
        "alpha_kappa": float(alpha_kappa),
        "beta_kappa": float(beta_kappa),
        "post_mean_kappa": summary["post_mean_kappa"],
        "post_median_kappa": summary["post_median_kappa"],
        "post_sd_kappa": summary["post_sd_kappa"],
    }
    for eps in tail_eps:
        row[_tail_prob_col_name(eps)] = _tail_prob_from_grid(grid["kappa"], grid["density"], threshold=float(eps))
    return row


def _exp1_setting_worker(task: tuple[int, int, int, int, float, tuple[float, ...], int, float, float]) -> list[dict[str, Any]]:
    sid, pg, repeats, seed, tau_eval, tail_eps, grid_size, alpha_kappa, beta_kappa = task
    rows: list[dict[str, Any]] = []
    for r in range(1, int(repeats) + 1):
        s = experiment_seed(1, sid, r, master_seed=seed)
        y = generate_null_group(pg=pg, sigma2=1.0, seed=s)
        grid = kappa_posterior_grid(
            y,
            tau=tau_eval,
            sigma2=1.0,
            alpha_kappa=alpha_kappa,
            beta_kappa=beta_kappa,
            grid_size=grid_size,
        )
        summary = posterior_summary_from_grid(grid["kappa"], grid["density"])
        row: dict[str, Any] = {
            "p_g": pg,
            "setting_id": sid,
            "replicate_id": r,
            "tau_eval": float(tau_eval),
            "alpha_kappa": float(alpha_kappa),
            "beta_kappa": float(beta_kappa),
            "post_mean_kappa": summary["post_mean_kappa"],
            "post_median_kappa": summary["post_median_kappa"],
            "post_sd_kappa": summary["post_sd_kappa"],
        }
        for eps in tail_eps:
            row[_tail_prob_col_name(eps)] = _tail_prob_from_grid(grid["kappa"], grid["density"], threshold=float(eps))
        rows.append(row)
    return rows


def _exp2_worker(task: tuple[int, int, int, float, float, int, float, float, float, float, float]) -> dict[str, Any]:
    sid, pg, r, mu_g, scale, seed, tau_eval, win_lo, win_hi, alpha_kappa, beta_kappa = task
    s = experiment_seed(2, sid, r, master_seed=seed)
    y, _ = generate_signal_group_distributed(pg=pg, mu_g=mu_g, sigma2=1.0, seed=s)
    grid = kappa_posterior_grid(y, tau=tau_eval, sigma2=1.0, alpha_kappa=alpha_kappa, beta_kappa=beta_kappa)
    summary = posterior_summary_from_grid(grid["kappa"], grid["density"], window_lower=win_lo, window_upper=win_hi)
    return {
        "p_g": pg,
        "mu_g": mu_g,
        "tau_eval": float(tau_eval),
        "alpha_kappa": float(alpha_kappa),
        "beta_kappa": float(beta_kappa),
        "window_lo": float(win_lo),
        "window_hi": float(win_hi),
        "replicate_id": r,
        "post_mean_kappa": summary["post_mean_kappa"],
        "ratio_R": summary["post_mean_kappa"] / max(scale, 1e-12),
        "post_prob_kappa_in_scale_window": summary.get("window_prob", float("nan")),
    }


def _exp2_setting_worker(task: tuple[int, int, int, float, float, int, float, float, float, int, float, float]) -> list[dict[str, Any]]:
    sid, pg, repeats, mu_g, scale, seed, tau_eval, win_lo, win_hi, grid_size, alpha_kappa, beta_kappa = task
    rows: list[dict[str, Any]] = []
    for r in range(1, int(repeats) + 1):
        s = experiment_seed(2, sid, r, master_seed=seed)
        y, _ = generate_signal_group_distributed(pg=pg, mu_g=mu_g, sigma2=1.0, seed=s)
        grid = kappa_posterior_grid(
            y,
            tau=tau_eval,
            sigma2=1.0,
            alpha_kappa=alpha_kappa,
            beta_kappa=beta_kappa,
            grid_size=grid_size,
        )
        summary = posterior_summary_from_grid(grid["kappa"], grid["density"], window_lower=win_lo, window_upper=win_hi)
        rows.append(
            {
                "p_g": pg,
                "mu_g": mu_g,
                "tau_eval": float(tau_eval),
                "alpha_kappa": float(alpha_kappa),
                "beta_kappa": float(beta_kappa),
                "window_lo": float(win_lo),
                "window_hi": float(win_hi),
                "replicate_id": r,
                "post_mean_kappa": summary["post_mean_kappa"],
                "ratio_R": summary["post_mean_kappa"] / max(scale, 1e-12),
                "post_prob_kappa_in_scale_window": summary.get("window_prob", float("nan")),
            }
        )
    return rows


def _exp3_worker(task: tuple[int, int, int, float, int, int, float, float, float, float, float]) -> dict[str, Any]:
    sid, pg, xid, xi, r, seed, tau, sigma2, u0, alpha_kappa, beta_kappa = task
    mu_g = xi * pg
    s = experiment_seed(3, sid * 100 + xid, r, master_seed=seed)
    y, _ = generate_signal_group_distributed(pg=pg, mu_g=mu_g, sigma2=sigma2, seed=s)
    grid = kappa_posterior_grid(y, tau=tau, sigma2=sigma2, alpha_kappa=alpha_kappa, beta_kappa=beta_kappa)
    summary = posterior_summary_from_grid(grid["kappa"], grid["density"], tail_threshold=u0)
    return {
        "tau": float(tau),
        "xi": xi,
        "p_g": pg,
        "replicate_id": r,
        "post_prob_kappa_gt_u0": summary.get("tail_prob", float("nan")),
        "post_mean_kappa": summary["post_mean_kappa"],
    }


def _exp3_setting_worker(task: tuple[int, int, int, float, int, int, float, float, float, int, float, float]) -> list[dict[str, Any]]:
    sid, pg, xid, xi, repeats, seed, tau, sigma2, u0, grid_size, alpha_kappa, beta_kappa = task
    rows: list[dict[str, Any]] = []
    mu_g = xi * pg
    for r in range(1, int(repeats) + 1):
        s = experiment_seed(3, sid * 100 + xid, r, master_seed=seed)
        y, _ = generate_signal_group_distributed(pg=pg, mu_g=mu_g, sigma2=sigma2, seed=s)
        grid = kappa_posterior_grid(
            y,
            tau=tau,
            sigma2=sigma2,
            alpha_kappa=alpha_kappa,
            beta_kappa=beta_kappa,
            grid_size=grid_size,
        )
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
    ds = generate_grouped_logistic_dataset(n=200, group_sizes=[5, 5, 5], rho_within=0.5, rho_between=0.05, beta0=beta0, seed=s, min_separator_auc=0.0)
    fits = _fit_all_methods(ds["X"], ds["y"], ds["groups"], task="logistic", seed=s, p0=4, sampler=sampler)
    out_rows: list[dict[str, Any]] = []
    for method, res in fits.items():
        is_valid = bool(res.converged and (res.beta_mean is not None))
        b = np.asarray(res.beta_mean, dtype=float) if res.beta_mean is not None else None
        g1 = np.asarray(ds["groups"][0], dtype=int)
        g2 = np.asarray(ds["groups"][1], dtype=int)
        g3 = np.asarray(ds["groups"][2], dtype=int)
        k1, k2, k3 = float("nan"), float("nan"), float("nan")
        km1, km2, km3 = float("nan"), float("nan"), float("nan")
        if is_valid and method == "GR_RHS" and res.kappa_draws is not None:
            kd = np.asarray(res.kappa_draws, dtype=float)
            if kd.ndim > 2:
                kd = kd.reshape(-1, kd.shape[-1])
            k1 = float(np.mean(kd[:, 0] > 0.5))
            k2 = float(np.mean(kd[:, 1] > 0.5))
            k3 = float(np.mean(kd[:, 2] > 0.5))
            km1 = float(np.mean(kd[:, 0]))
            km2 = float(np.mean(kd[:, 1]))
            km3 = float(np.mean(kd[:, 2]))
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
                "beta_group3_l2_norm": float(np.linalg.norm(b[g3])) if is_valid and b is not None else float("nan"),
                "overall_runtime": res.runtime_seconds,
                "divergence_ratio": res.divergence_ratio,
                "bulk_ess_min": res.bulk_ess_min,
                "post_prob_kappa_group1_gt_0_5": k1,
                "post_prob_kappa_group2_gt_0_5": k2,
                "post_prob_kappa_group3_gt_0_5": k3,
                "post_mean_kappa_group1": km1,
                "post_mean_kappa_group2": km2,
                "post_mean_kappa_group3": km3,
                "separator_auc": float(ds.get("separator_auc", float("nan"))),
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
            rho_within=0.3,
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


def _exp9_worker(task: tuple[int, float, float, int, int, list[float], np.ndarray, list[int], str, str, int, SamplerConfig]) -> dict[str, Any]:
    from .dgp_grouped_linear import generate_heterogeneity_dataset
    from .fit_gr_rhs import fit_gr_rhs
    from .metrics import group_auroc, group_l2_error, group_l2_score

    pid, a, b, r, seed, mu, labels, group_sizes, scenario, scenario_base, p_g, sampler = task
    s = experiment_seed(9, pid, r, master_seed=seed)
    ds = generate_heterogeneity_dataset(n=300, group_sizes=group_sizes, rho_within=0.3, rho_between=0.05, sigma2=1.0, mu=mu, seed=s)
    res = fit_gr_rhs(ds["X"], ds["y"], ds["groups"], task="gaussian", seed=s + 7, p0=int(np.sum(labels)), sampler=sampler, alpha_kappa=a, beta_kappa=b, use_group_scale=True, shared_kappa=False)
    null_kappa_mean = float("nan")
    signal_kappa_mean = float("nan")
    null_prob_kappa_gt_0_1 = float("nan")
    if res.kappa_draws is not None:
        kd = np.asarray(res.kappa_draws, dtype=float)
        if kd.ndim > 2:
            kd = kd.reshape(-1, kd.shape[-1])
        null_idx = np.where(labels == 0)[0]
        sig_idx = np.where(labels == 1)[0]
        if null_idx.size > 0:
            null_kappa_mean = float(np.mean(kd[:, null_idx]))
            null_prob_kappa_gt_0_1 = float(np.mean(kd[:, null_idx] > 0.1))
        if sig_idx.size > 0:
            signal_kappa_mean = float(np.mean(kd[:, sig_idx]))
    is_valid = bool(res.converged and (res.beta_mean is not None))
    if not is_valid:
        return {
            "alpha_kappa": a,
            "beta_kappa": b,
            "scenario": str(scenario),
            "scenario_base": str(scenario_base),
            "p_g": int(p_g),
            "replicate_id": r,
            "status": res.status,
            "converged": bool(res.converged),
            "null_group_mse_avg": float("nan"),
            "signal_group_mse_avg": float("nan"),
            "group_auroc": float("nan"),
            "null_group_kappa_mean": null_kappa_mean,
            "signal_group_kappa_mean": signal_kappa_mean,
            "null_group_prob_kappa_gt_0_1": null_prob_kappa_gt_0_1,
        }
    score = group_l2_score(res.beta_mean, ds["groups"])
    err = group_l2_error(res.beta_mean, ds["beta0"], ds["groups"])
    return {
        "alpha_kappa": a,
        "beta_kappa": b,
        "scenario": str(scenario),
        "scenario_base": str(scenario_base),
        "p_g": int(p_g),
        "replicate_id": r,
        "status": res.status,
        "converged": bool(res.converged),
        "null_group_mse_avg": float(np.mean(err[labels == 0])),
        "signal_group_mse_avg": float(np.mean(err[labels == 1])),
        "group_auroc": group_auroc(score, labels),
        "null_group_kappa_mean": null_kappa_mean,
        "signal_group_kappa_mean": signal_kappa_mean,
        "null_group_prob_kappa_gt_0_1": null_prob_kappa_gt_0_1,
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
    tail_eps_list: Sequence[float] | None = None,
    alpha_kappa: float = 0.5,
    beta_kappa: float = 1.0,
    tail_M: float | None = None,
    grid_size: int = 801,
) -> Dict[str, str]:
    # ============================================================
    # EXP1 — 零收缩验证（Theorem 3.22）
    #
    # 【理论预测】
    #   E[κ_g | Y_null] = O_{P₀}(p_g^{-1/2})
    #   → log-log 斜率目标：-0.5
    #   → 固定阈值尾概率 P(κ > ε) → 0（任意固定 ε > 0）
    #
    # 【参数选择说明】
    #   pg_list  = [10,...,2000]：覆盖 pre-asymptotic 到 asymptotic 区域
    #   tau_eval = 0.5：ρ = τ/√σ² = 0.5，居中选取以避免 ρ→0/∞ 的退化
    #   alpha_kappa=0.5, beta_kappa=1.0：先验对 κ=0 有轻微倾向（比 Beta(1,1) 收缩快）
    #   tail_eps_list=[0.1, 0.2]：固定阈值（不随 p_g 缩放），直接测零收缩
    #
    # 【若 log-log 斜率未达到 -0.5 的调整方向】
    #   症状 A — 斜率 < -0.5（过快下降）：
    #     → tau_eval 偏小（ρ 太小，先验对 κ 的约束很强），尝试 tau_eval=1.0
    #     → alpha_kappa 偏小，先验对 κ 施加过强的下压，尝试 alpha_kappa=1.0
    #
    #   症状 B — 斜率 > -0.5（下降太慢，如 -0.30）：
    #     → p_g 范围不够大，渐近区域未到达；扩展 pg_list 到 [500,1000,2000,5000]
    #     → beta_kappa 偏小（先验均值 α/(α+β) 过大），尝试 beta_kappa=2.0
    #     → repeats 不足导致 median 估计噪声大，增加到 1000
    #
    #   症状 C — 尾概率 P(κ > 0.1) 非单调（出现驼峰）：
    #     → 检查 tau_eval：对于极小 τ，后验在小 p_g 时就很集中，大 p_g 时变化不大
    #     → 增加 tail_eps_list 中的 ε 值（如 0.3, 0.5）以观察不同尾部行为
    # ============================================================
    from .plotting import plot_exp1

    base = Path(save_dir)
    out = ensure_dir(base / "results" / "exp1_null_contraction")
    fig_dir = ensure_dir(base / "figures")
    tab_dir = ensure_dir(base / "tables")
    log = setup_logger("exp1", base / "logs" / "exp1_null_contraction.log")

    pg_vals = list(pg_list or [10, 20, 50, 100, 200, 500, 1000, 2000])
    input_tau_eval = float(tau_eval)
    tail_eps_vals = sorted({float(v) for v in (tail_eps_list or [0.1, 0.2])})
    tasks: list[tuple[int, int, int, int, float, tuple[float, ...], int, float, float]] = []
    for sid, pg in enumerate(pg_vals, start=1):
        tasks.append(
            (
                sid,
                pg,
                int(repeats),
                seed,
                input_tau_eval,
                tuple(tail_eps_vals),
                int(grid_size),
                float(alpha_kappa),
                float(beta_kappa),
            )
        )
    rows_nested = _parallel_rows(tasks, _exp1_setting_worker, n_jobs=n_jobs, prefer_process=False, progress_desc="Exp1 Null Contraction")
    rows: list[dict[str, Any]] = []
    for chunk in rows_nested:
        rows.extend(chunk)

    agg_rows: list[dict[str, Any]] = []
    for pg in sorted({int(r["p_g"]) for r in rows}):
        sub = [r for r in rows if int(r["p_g"]) == pg]
        post = np.asarray([float(r["post_mean_kappa"]) for r in sub], dtype=float)
        row = {
            "p_g": pg,
            "tau_eval": float(input_tau_eval),
            "alpha_kappa": float(alpha_kappa),
            "beta_kappa": float(beta_kappa),
            "median_post_mean_kappa": float(np.median(post)),
            "iqr_post_mean_kappa_low": float(np.quantile(post, 0.25)),
            "iqr_post_mean_kappa_high": float(np.quantile(post, 0.75)),
        }
        for eps in tail_eps_vals:
            col = _tail_prob_col_name(eps)
            out_col = f"mean_tail_prob_eps_{_eps_tag(eps)}"
            vals = np.asarray([float(r[col]) for r in sub], dtype=float)
            row[out_col] = float(np.mean(vals))
        # Backward-compatible alias uses the primary epsilon.
        row["mean_tail_prob"] = float(row[f"mean_tail_prob_eps_{_eps_tag(tail_eps_vals[0])}"])
        agg_rows.append(row)
    x = np.log(np.asarray([float(r["p_g"]) for r in agg_rows], dtype=float))
    y = np.log(np.asarray([float(r["median_post_mean_kappa"]) for r in agg_rows], dtype=float))
    # 理论目标：slope ≈ -0.5（95% CI 覆盖 -0.5）
    # 若斜率显著偏离，见函数头部注释的【症状 A/B】调整方向
    slope, slope_ci = _linreg_slope_ci(x, y)
    _save_rows_csv(rows, out / "raw_results.csv")
    _save_rows_csv(agg_rows, out / "summary.csv")
    _save_rows_csv(agg_rows, out / "summary_tau_sweep.csv")
    save_json(
        {
            "tau_eval": input_tau_eval,
            "alpha_kappa": float(alpha_kappa),
            "beta_kappa": float(beta_kappa),
            "tail_eps_list": [float(v) for v in tail_eps_vals],
            "tail_threshold_formula": "fixed_epsilon",
            "tail_M_deprecated": None if tail_M is None else float(tail_M),
            "p_g_list": pg_vals,
            "grid_size": int(grid_size),
        },
        out / "exp1_meta.json",
    )
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
    mu_coef: float = 3.0,
    x_lo: float = 0.5,
    x_hi: float = 2.0,
    alpha_kappa: float = 0.5,
    beta_kappa: float = 1.0,
    c_window: float | None = None,
    grid_size: int = 801,
) -> Dict[str, str]:
    # ============================================================
    # EXP2 — 自适应局部化验证（Theorem 3.30）
    #
    # 【理论预测】
    #   中间信号下（r_g = μ_g²/p_g → ∞），后验集中于 [x_- s_g, x_+ s_g]
    #   其中 s_g = μ_g/p_g，x_-, x_+ 为固定常数（不依赖 p_g）
    #   → ratio_R = E[κ|Y] / s_g → 1（后验均值收敛到 s_g）
    #   → P(κ ∈ [x_lo·s_g, x_hi·s_g]) → 1
    #
    # 【参数选择说明】
    #   mu_coef=3.0：μ_g = 3·p_g^{0.75}，使 r_g = 9·p_g^{0.5} → ∞（满足 Cond. 3.25）
    #                 同时 s_g = 3·p_g^{-0.25} 使 s_g ∈ (0,1) 对所有设定的 p_g 成立
    #   x_lo=0.5, x_hi=2.0：定理的"固定常数窗口"。窗口宽度 = 1.5·s_g（固定比例）
    #                         注意：旧版曾错用 (1±C/√p_g) 的缩窄窗口，已修正为此
    #   tau_list=[0.5,1.0,2.0]：跨越 ρ<1/ρ=1/ρ>1 三种区域测试普适性
    #
    # 【若 ratio_R 未收敛到 1 的调整方向】
    #   症状 A — ratio_R 随 p_g 单调增加（> 1）：
    #     → s_g 对所有大 p_g 都偏小（< 0.5），Beta 先验均值=α/(α+β) 将 κ 上推
    #     → 尝试增大 mu_coef（如 mu_coef=5.0）使 s_g 更大，远离先验均值
    #     → 或减小 beta_kappa（如 beta_kappa=0.5）使先验均值靠近 0
    #
    #   症状 B — ratio_R 随 p_g 单调减少（< 1）：
    #     → s_g 偏大（接近 1），κ 空间边界效应（κ ∈ (0,1)）导致均值偏低
    #     → 尝试减小 mu_coef（如 mu_coef=1.5）
    #
    #   症状 C — window_prob 不随 p_g 增大（局部化不成立）：
    #     → 检查 r_g = mu_g²/p_g 是否足够大：打印 mu_g²/p_g 对各 p_g 的值
    #     → 若 r_g 增长慢，尝试 mu_coef^2 > p_g^{0.5} 更快增长（如 mu_coef * p_g^{0.6}）
    #     → 扩展 pg_list 到 [20,50,100,200,500,1000]
    #
    #   症状 D — 不同 tau 的 ratio_R 曲线差异很大（tau 依赖性强）：
    #     → 这表明 pre-asymptotic 效应主导；增大 p_g 或检验 ρ 是否过于极端
    # ============================================================
    from .plotting import plot_exp2

    base = Path(save_dir)
    out = ensure_dir(base / "results" / "exp2_adaptive_localization")
    fig_dir = ensure_dir(base / "figures")
    log = setup_logger("exp2", base / "logs" / "exp2_adaptive_localization.log")
    pg_vals = list(pg_list or [20, 50, 100, 200, 500])
    tau_vals = list(tau_list or [0.5, 1.0, 2.0])
    tasks: list[tuple[int, int, int, float, float, int, float, float, float, int, float, float]] = []
    sid = 0
    for tau_eval in tau_vals:
        for pg in pg_vals:
            sid += 1
            mu_g = float(mu_coef) * (pg ** 0.75)  # r_g = mu_g²/p_g = mu_coef²·p_g^0.5 → ∞
            scale = mu_g / pg                       # s_g = μ_g/p_g，定理 3.30 的局部化中心
            # 窗口为 [x_lo·s_g, x_hi·s_g]：x_lo/x_hi 是固定常数（不依赖 p_g）
            # 旧设计错误：(1 ± C/√p_g)·s_g 是缩窄窗口，窗口概率必然趋向 0
            # 正确：固定比例窗口使得 Theorem 3.30 的"集中于 [x_-·s_g, x_+·s_g]"可以被验证
            win_lo = max(float(x_lo) * scale, 1e-4)
            win_hi = min(float(x_hi) * scale, 1.0 - 1e-4)
            tasks.append(
                (
                    sid,
                    pg,
                    int(repeats),
                    mu_g,
                    scale,
                    seed,
                    float(tau_eval),
                    float(win_lo),
                    float(win_hi),
                    int(grid_size),
                    float(alpha_kappa),
                    float(beta_kappa),
                )
            )
    rows_nested = _parallel_rows(tasks, _exp2_setting_worker, n_jobs=n_jobs, prefer_process=False, progress_desc="Exp2 Adaptive Localization")
    rows: list[dict[str, Any]] = []
    for chunk in rows_nested:
        rows.extend(chunk)
    agg_rows: list[dict[str, Any]] = []
    for tau_eval in sorted({float(r["tau_eval"]) for r in rows}):
        for pg in sorted({int(r["p_g"]) for r in rows}):
            sub = [r for r in rows if float(r["tau_eval"]) == tau_eval and int(r["p_g"]) == pg]
            if not sub:
                continue
            ratio = np.asarray([float(r["ratio_R"]) for r in sub], dtype=float)
            win = np.asarray([float(r["post_prob_kappa_in_scale_window"]) for r in sub], dtype=float)
            agg_rows.append(
                {
                    "tau_eval": float(tau_eval),
                    "p_g": pg,
                    "median_ratio_R": float(np.median(ratio)),
                    "iqr_ratio_R_low": float(np.quantile(ratio, 0.25)),
                    "iqr_ratio_R_high": float(np.quantile(ratio, 0.75)),
                    "mean_window_prob": float(np.mean(win)),
                }
            )
    _save_rows_csv(rows, out / "raw_results.csv")
    _save_rows_csv(agg_rows, out / "summary.csv")
    _save_rows_csv(agg_rows, out / "summary_tau_sweep.csv")
    save_json(
        {
            "tau_list": [float(t) for t in tau_vals],
            "alpha_kappa": float(alpha_kappa),
            "beta_kappa": float(beta_kappa),
            "window_rule": "[x_lo * s_g, x_hi * s_g]",
            "mu_rule": "mu_g = mu_coef * p_g^0.75",
            "mu_coef": float(mu_coef),
            "x_lo": float(x_lo),
            "x_hi": float(x_hi),
            "c_window_deprecated": None if c_window is None else float(c_window),
            "p_g_list": pg_vals,
            "grid_size": int(grid_size),
        },
        out / "exp2_meta.json",
    )
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
    alpha_kappa: float = 0.5,
    beta_kappa: float = 1.0,
    grid_size: int = 801,
) -> Dict[str, str]:
    # ============================================================
    # EXP3 — 相变图（Theorem 3.32 / Corollary 3.33）
    #
    # 【理论预测】
    #   ξ_crit = θ(u₀,ρ)/2 = u₀ρ²/(2(u₀+(1−u₀)ρ²))
    #   当 ξ = μ_g/p_g > ξ_crit → P(κ_g > u₀ | Y) → 1（p_g → ∞）
    #   当 ξ < ξ_crit            → P(κ_g > u₀ | Y) → 0
    #   x 轴为 ξ/ξ_crit（已标准化），相变点固定在 1.0
    #
    # 【参数选择说明】
    #   p_g_list=[30,60,120,240,480]：覆盖渐近趋势；480 是观察锐利跳跃的最小 p_g
    #   xi_multiplier_list=[0.3,...,2.0]×ξ_crit：阈值两侧均匀采样
    #   u0=0.5：κ_g 的判断阈值（可改变以测试不同 u₀ 下 ξ_crit 公式的正确性）
    #   theory_check_tau=0.3：主图使用的 τ 参考值（不同 τ 由 τ-sweep 图展示）
    #
    # 【若相变曲线未表现出锐利跳跃的调整方向】
    #   症状 A — 曲线平坦（无明显跳跃，各 p_g 差异小）：
    #     → p_g 太小；将 p_g_list 最大值扩展到 960 或 1920
    #     → 检查 xi_crit 计算是否正确：打印 xi_crit_u0_rho(u0, tau/sqrt(sigma2))
    #     → 确认 mu_g = xi * pg（在 _exp3_worker 中），不要用 mu_g = xi
    #
    #   症状 B — 跳跃点不在 ξ/ξ_crit = 1.0 处（系统偏移）：
    #     → tau 值极端（τ≈0.1 时 ξ_crit 极小，数值精度会影响），检查 ρ = τ/√σ²
    #     → sigma2 参数与数据生成中的实际 σ² 不一致，确保两者相同
    #     → u0 需与分析的阈值一致（更改 u0 会同时移动 ξ_crit 和测试统计量）
    #
    #   症状 C — 不同 τ 的 ξ/ξ_crit 曲线相互不重叠（标准化失败）：
    #     → 标准化 x 轴的 ξ_crit 在代码中正确按 tau 计算（见 xi_by_tau 字典），
    #       若曲线不重叠，说明有限样本下 ξ_crit 的有效值与理论值有偏差
    #     → 增大 repeats 到 500 以减少噪声
    #
    #   症状 D — P(κ > u₀) 在 ξ >> ξ_crit 时无法达到 1：
    #     → alpha_kappa/beta_kappa 的先验均值过低，先验压制 κ；尝试 alpha_kappa=1.0
    #     → grid_size 太小导致后验积分精度低；增加到 1601
    # ============================================================
    from .plotting import plot_exp3_curves, plot_exp3_heatmap, plot_exp3_tau_sweep

    base = Path(save_dir)
    out = ensure_dir(base / "results" / "exp3_phase_diagram")
    fig_dir = ensure_dir(base / "figures")
    log = setup_logger("exp3", base / "logs" / "exp3_phase_diagram.log")
    tau_vals = list(tau_list or [0.1, 0.2, 0.3, 0.5, 1.0])
    xi_mults = list(xi_multiplier_list or [0.3, 0.5, 0.7, 0.85, 0.95, 1.05, 1.15, 1.3, 1.5, 2.0])
    pg_vals = list(p_g_list or [30, 60, 120, 240, 480])
    tasks: list[tuple[int, int, int, float, int, int, float, float, float, int, float, float]] = []
    xi_by_tau: dict[float, list[float]] = {}
    sid = 0
    for tau in tau_vals:
        xi_crit_tau = xi_crit_u0_rho(u0=u0, rho=tau / math.sqrt(sigma2))
        xi_vals = [float(xi_crit_tau * m) for m in xi_mults]
        xi_by_tau[float(tau)] = xi_vals
        for pg in pg_vals:
            sid += 1
            for xid, xi in enumerate(xi_vals, start=1):
                tasks.append(
                    (
                        sid,
                        pg,
                        xid,
                        xi,
                        int(repeats),
                        seed,
                        float(tau),
                        sigma2,
                        u0,
                        int(grid_size),
                        float(alpha_kappa),
                        float(beta_kappa),
                    )
                )
    rows_nested = _parallel_rows(tasks, _exp3_setting_worker, n_jobs=n_jobs, prefer_process=False, progress_desc="Exp3 Phase Diagram")
    rows: list[dict[str, Any]] = []
    for chunk in rows_nested:
        rows.extend(chunk)
    for r in rows:
        tau_now = float(r["tau"])
        xi_crit_now = xi_crit_u0_rho(u0=u0, rho=tau_now / math.sqrt(sigma2))
        # xi_ratio = ξ/ξ_crit：标准化 x 轴，使不同 τ 的相变点均落在 1.0 处
        # 若 xi_ratio=1.0 处 P(κ>u₀) ≠ 0.5，说明 finite-p_g 偏差或 ξ_crit 公式偏差
        r["xi_ratio"] = float(r["xi"]) / max(float(xi_crit_now), 1e-12)
    agg_rows: list[dict[str, Any]] = []
    keys = sorted({(float(r["xi"]), float(r["xi_ratio"]), int(r["p_g"]), float(r["tau"])) for r in rows}, key=lambda z: (z[3], z[2], z[1]))
    for xi, xi_ratio, pg, tau in keys:
        sub = [r for r in rows if float(r["xi"]) == xi and int(r["p_g"]) == pg and float(r["tau"]) == tau]
        prob = np.asarray([float(r["post_prob_kappa_gt_u0"]) for r in sub], dtype=float)
        agg_rows.append(
            {
                "tau": tau,
                "xi": xi,
                "xi_ratio": float(xi_ratio),
                "p_g": pg,
                "mean_prob_gt_u0": float(np.mean(prob)),
                "sd_prob_gt_u0": float(np.std(prob, ddof=1)),
            }
        )
    _save_rows_csv(rows, out / "raw_results.csv")
    _save_rows_csv(agg_rows, out / "summary_tau_sweep.csv")
    tau_ref = float(theory_check_tau if theory_check_tau in tau_vals else tau_vals[0])
    ref_rows = [r for r in rows if float(r["tau"]) == tau_ref]
    agg_ref: list[dict[str, Any]] = []
    keys_ref = sorted({(float(r["xi"]), float(r["xi_ratio"]), int(r["p_g"])) for r in ref_rows}, key=lambda z: (z[2], z[1]))
    for xi, xi_ratio, pg in keys_ref:
        sub = [r for r in ref_rows if float(r["xi"]) == xi and int(r["p_g"]) == pg]
        prob = np.asarray([float(r["post_prob_kappa_gt_u0"]) for r in sub], dtype=float)
        agg_ref.append(
            {
                "xi": xi,
                "xi_ratio": float(xi_ratio),
                "p_g": pg,
                "mean_prob_gt_u0": float(np.mean(prob)),
                "sd_prob_gt_u0": float(np.std(prob, ddof=1)),
            }
        )
    _save_rows_csv(agg_ref, out / "summary.csv")
    xi_crit_ref = xi_crit_u0_rho(u0=u0, rho=tau_ref / math.sqrt(sigma2))
    theta_ref = theta_u0_rho(u0=u0, rho=tau_ref / math.sqrt(sigma2))
    plot_exp3_heatmap(agg_ref, out_path=fig_dir / "fig3_phase_heatmap.png")
    plot_exp3_curves(agg_ref, xi_crit=1.0, out_path=fig_dir / "fig3_phase_curves.png")
    plot_exp3_tau_sweep(agg_rows, out_path=fig_dir / "fig3_phase_curves_tau_sweep.png")
    save_json(
        {
            "u0": u0,
            "sigma2": sigma2,
            "tau": tau_ref,
            "rho": tau_ref / math.sqrt(sigma2),
            "theta_u0_rho": theta_ref,
            "xi_crit": xi_crit_ref,
            "x_axis": "xi_over_xi_crit",
            "alpha_kappa": float(alpha_kappa),
            "beta_kappa": float(beta_kappa),
            "p_g_list": pg_vals,
            "tau_list": [float(t) for t in tau_vals],
            "xi_by_tau": {f"{k:.6g}": v for k, v in xi_by_tau.items()},
            "grid_size": int(grid_size),
        },
        out / "phase_threshold_meta.json",
    )
    log.info("Completed exp3 with repeats=%d", repeats)
    return {"raw": str(out / "raw_results.csv"), "summary": str(out / "summary.csv"), "summary_tau": str(out / "summary_tau_sweep.csv")}


def _evaluate_method_row(result: FitResult, beta0: np.ndarray) -> dict[str, float]:
    from .metrics import ci_length_and_coverage, mse_null_signal_overall

    if (result.beta_mean is None) or (not result.converged):
        return {"mse_null": float("nan"), "mse_signal": float("nan"), "mse_overall": float("nan"), "avg_ci_length": float("nan"), "coverage_95": float("nan")}
    m = mse_null_signal_overall(result.beta_mean, beta0)
    ci_len, cov = ci_length_and_coverage(beta0, result.beta_draws)
    return {"mse_null": m["mse_null"], "mse_signal": m["mse_signal"], "mse_overall": m["mse_overall"], "avg_ci_length": ci_len, "coverage_95": cov}


def run_exp4_benchmark_linear(n_jobs: int = 1, seed: int = MASTER_SEED, repeats: int = 100, save_dir: str = "simulation_project") -> Dict[str, str]:
    # ============================================================
    # EXP4 — 线性回归基准比较（GR_RHS vs RHS / GIGG_MMLE / GHS_plus）
    #
    # 【理想结果】
    #   GR_RHS 在 mse_null（零组）上显著优于 RHS（强组间收缩）
    #   GR_RHS 在 mse_signal（信号组）上与 RHS 相当或更好
    #   GR_RHS coverage_95 ≈ 0.95（区间校准良好）
    #   高相关场景（L3,L4）GR_RHS 的优势应更明显
    #
    # 【若 GR_RHS 不优于 RHS 的调整方向】
    #   症状 A — mse_null 无改善（GR_RHS ≈ RHS）：
    #     → 检查 alpha_kappa/beta_kappa 设置；增大 beta_kappa 使先验更倾向 κ→0
    #     → 确认 GR_RHS 使用了 use_group_scale=True（组间尺度 a_g 是分离的关键）
    #
    #   症状 B — mse_signal 显著变差（GR_RHS 过度收缩信号组）：
    #     → tau 设置可能过小；检查 grrhs_kwargs 中的 tau0
    #     → 信号组的 μ_g 接近相变阈值，增大信号强度或改变 build_linear_beta 的设置
    #
    #   症状 C — coverage_95 远低于 0.95：
    #     → MCMC 未收敛（检查 rhat_max < 1.05, bulk_ess_min > 100）
    #     → 增加 chains/warmup 在 SamplerConfig 中
    #
    #   症状 D — n_effective 偏低（大量不收敛）：
    #     → sampler 默认配置不足，需显式传入更多链数/步数
    #     → repeats=100 勉强够用，若结果噪声大考虑 repeats=200
    # ============================================================
    import pandas as pd
    from .dgp_grouped_linear import build_linear_beta
    from .plotting import plot_exp4_mse_partition, plot_exp4_overall_mse
    from .utils import canonical_groups
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
    plot_exp4_mse_partition(summary, out_path=fig_dir / "fig4_benchmark_mse_partition.png")
    design_rows: list[dict[str, Any]] = []
    for setting, spec in settings.items():
        groups = canonical_groups(spec["group_sizes"])
        beta0 = build_linear_beta(setting, spec["group_sizes"])
        active_groups = []
        group_nonzero_counts = []
        for gid, g in enumerate(groups):
            idx = np.asarray(g, dtype=int)
            nnz = int(np.sum(np.abs(beta0[idx]) > 0.0))
            group_nonzero_counts.append(nnz)
            if nnz > 0:
                active_groups.append(gid + 1)
        design_rows.append(
            {
                "setting": str(setting),
                "group_sizes": [int(v) for v in spec["group_sizes"]],
                "rho_within": float(spec["rho_within"]),
                "rho_between": float(spec["rho_between"]),
                "design_type": str(spec.get("design_type", "correlated")),
                "active_groups_1_based": active_groups,
                "group_nonzero_counts": group_nonzero_counts,
                "nonzero_total": int(np.sum(np.abs(beta0) > 0.0)),
            }
        )
    save_json({"settings": design_rows, "repeats": int(repeats)}, out / "exp4_design_meta.json")
    log.info("Completed exp4 with repeats=%d", repeats)
    return {"raw": str(out / "raw_results.csv"), "table": str(tab_dir / "table_benchmark_linear.csv")}


def run_exp5_heterogeneity(n_jobs: int = 1, seed: int = MASTER_SEED, repeats: int = 100, save_dir: str = "simulation_project") -> Dict[str, str]:
    # ============================================================
    # EXP5 — 异质性组结构下的分组辨识（连接定理 3.34 的 simultaneous separation）
    #
    # 【数据生成设计说明】
    #   group_sizes = [50, 50, 20, 10, 10, 10]，sigma2=1.0，tau_ref=0.1
    #   mu = [0, 0,  mu_boundary,  2,  8,  25]
    #         零  零  阈值附近       弱  中  强
    #   mu_boundary = 1.2 × ξ_crit(0.5, 0.1) × 20  （比相变阈值高 20%）
    #   → Group 1,2 (p_g=50)：应被强力收缩（κ≈0）
    #   → Group 3   (p_g=20)：信号轻微超过阈值，κ 应处于过渡区（0.3~0.7）
    #   → Group 4,5,6 (p_g=10)：强信号，κ≈1
    #
    # 【理想结果】
    #   GR_RHS AUROC > 0.90（显著优于 RHS 的 ~0.60~0.70）
    #   零组 κ 均值 < 0.05，强信号组 κ 均值 > 0.80
    #   Group 3 的 κ 分布宽且居中（体现相变边界的不确定性）
    #
    # 【若辨识失败（AUROC 低）的调整方向】
    #   症状 A — GR_RHS AUROC ≈ RHS（组结构未被利用）：
    #     → 检查 use_group_scale=True 在 fit_gr_rhs 调用中是否有效
    #     → tau_ref=0.1 使得 ξ_crit 极小，Group 3 的信号已超阈值较多；
    #       若仍不收敛，尝试 tau_ref=0.3 使阈值更大，分离更难
    #
    #   症状 B — Group 3 的 κ 无差异（要么全 0 要么全 1）：
    #     → mu_boundary 倍数从 1.2 调整为 1.0（恰好在阈值上）
    #     → 增大 p_g=20 组到 30 或 50 以使相变更锐利
    #
    #   症状 C — 大零组（p_g=50）的 κ 不趋向 0：
    #     → 与 EXP1 一致的问题：p_g=50 在 tau_ref=0.1 下可能仍未充分收缩
    #     → 增大 n（当前 n=300），使数据对先验的支配力更强
    # ============================================================
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
    save_json(
        {
            "group_sizes": [int(v) for v in group_sizes],
            "mu": [float(v) for v in mu],
            "tau_ref": float(tau_ref),
            "xi_boundary": float(xi_boundary),
            "mu_boundary": float(mu_boundary),
            "boundary_multiplier_vs_xi_crit": 1.2,
            "repeats": int(repeats),
        },
        out / "exp5_meta.json",
    )
    if not raw_kappa.empty:
        plot_exp5_kappa_stratification(raw_kappa, out_path=fig_dir / "fig5_kappa_stratification.png")
        plot_exp5_group_ranking(raw_kappa, auroc_table.loc[auroc_table["method"].isin(METHODS)], out_path=fig_dir / "fig5_group_ranking.png")
    plot_exp5_null_signal_mse(auroc_table.loc[auroc_table["method"].isin(METHODS)], out_path=fig_dir / "fig5_null_signal_mse.png")
    log.info("Completed exp5 with repeats=%d", repeats)
    return {"raw": str(out / "raw_results.csv"), "table": str(tab_dir / "table_heterogeneity_auroc.csv")}


def run_exp6_grouped_logistic(n_jobs: int = 1, seed: int = MASTER_SEED, repeats: int = 50, save_dir: str = "simulation_project") -> Dict[str, str]:
    # ============================================================
    # EXP6 — 分组 Logistic 回归（二元结果的适用性验证）
    #
    # 【数据生成设计说明】
    #   3 组各 5 个预测变量（p_g=5，注意 p_g 很小，不处于理论渐近区域）
    #   beta0 = [1.5,1.5,0,0,0 | 0,0,0,0,0 | 0.5,0.5,0,0,0]
    #            强信号组          零组          弱信号组
    #   →  Group 1: ||β||²=4.5（强），Group 2: ||β||²=0（零），Group 3: ||β||²=0.5（弱）
    #   n=200，min_separator_auc=0.8 过滤极难样本
    #
    # 【理想结果】
    #   P(κ₁ > 0.5) ≈ 1.0（强信号组明确检测）
    #   P(κ₂ > 0.5) ≈ 0.0（零组明确抑制）
    #   P(κ₃ > 0.5) ∈ (0.3, 0.7)（弱信号组不确定，视为成功展示灵敏度）
    #   β_{11}, β_{12} 后验均值接近 1.5
    #
    # 【若 κ 分组无法区分的调整方向】
    #   症状 A — P(κ₁ > 0.5) 偏低（强信号也未被检测）：
    #     → 检查 logistic 回归的 MCMC 收敛（divergence_ratio < 0.01）
    #     → n=200 可能不足以支持 p_g=5 的信号；尝试 n=500
    #     → beta0 中的 1.5 对 logistic 尺度是否合适（log-odds 为 1.5，AUC 约 0.85）
    #
    #   症状 B — Group 3（弱信号）与 Group 2（零组）的 κ 完全相同：
    #     → 弱信号 ||β||²=0.5 在 n=200 下可能与零组无法区分（这本身是合理现象）
    #     → 若希望区分，增大 beta0[10:12] 到 [0.8, 0.8] 或增大 n
    #
    #   症状 C — 收敛率（n_effective）低（< 0.8）：
    #     → Logistic 模型容易出现发散；使用 SamplerConfig(adapt_delta=0.95)
    #     → 降低 min_separator_auc 阈值（过滤太严导致样本量不足）
    # ============================================================
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
        beta_group3_l2_norm=("beta_group3_l2_norm", "mean"),
        overall_runtime=("overall_runtime", "mean"),
        divergence_ratio=("divergence_ratio", "mean"),
        bulk_ess_min=("bulk_ess_min", "mean"),
        post_prob_kappa_group1_gt_0_5=("post_prob_kappa_group1_gt_0_5", "mean"),
        post_prob_kappa_group2_gt_0_5=("post_prob_kappa_group2_gt_0_5", "mean"),
        post_prob_kappa_group3_gt_0_5=("post_prob_kappa_group3_gt_0_5", "mean"),
        post_mean_kappa_group1=("post_mean_kappa_group1", "mean"),
        post_mean_kappa_group2=("post_mean_kappa_group2", "mean"),
        post_mean_kappa_group3=("post_mean_kappa_group3", "mean"),
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


def run_exp7_ablation(n_jobs: int = 1, seed: int = MASTER_SEED, repeats: int = 100, save_dir: str = "simulation_project") -> Dict[str, str]:
    # ============================================================
    # EXP7 — 消融研究（组件价值分析）
    #
    # 【消融变体说明】
    #   GR_RHS_full         ：完整模型（基准，应最优）
    #   GR_RHS_no_ag        ：去除组尺度 a_g（无组间异质性校正）
    #   GR_RHS_no_local_scales：去除组内局部尺度 λ_j（全组均匀收缩）
    #   GR_RHS_shared_kappa ：所有组共享一个 κ（无组特异性收缩）
    #   GR_RHS_no_kappa     ：退化为标准 RHS（无 κ 机制）
    #   RHS                 ：纯 horseshoe 基准
    #
    # 【DGP 类型与预期差异】
    #   dense_uniform：组内所有变量均有信号
    #     → no_local_scales 影响应较小（组内同质）
    #     → no_ag 影响应较大（组间强度差异需要 a_g 捕获）
    #   sparse_within_group：组内仅 20% 变量有信号
    #     → no_local_scales 应显著变差（λ_j 是组内稀疏性检测的关键）
    #     → GR_RHS_full 优势体现在 mse_signal 上（精准定位活跃变量）
    #
    # 【理想结果】
    #   GR_RHS_full 在两种 DGP 下都具有最低 null_group_mse 和最高 AUROC
    #   GR_RHS_no_local_scales 在 sparse_within_group 下 signal_group_mse 明显更高
    #   GR_RHS_shared_kappa 的 AUROC < GR_RHS_full（失去组特异性）
    #
    # 【若消融差异不显著的调整方向】
    #   症状 A — 所有变体性能相近（消融无效果）：
    #     → mu 中弱信号组（mu=2）可能已超出 ξ_crit，信号过强导致所有方法都能检测
    #     → 将 mu[2] 降低至接近相变阈值（见 EXP5 的 mu_boundary 计算方法）
    #     → rho_within=0.7 下组内变量高度相关，局部尺度作用可能被相关性掩盖
    #
    #   症状 B — GR_RHS_no_ag 与 GR_RHS_full 无差异：
    #     → 组间信号强度差异不够大（mu=[0,0,2,8,25,80] 相差悬殊，但 a_g 的作用
    #       体现在中等差异场景），考虑改为 [0,0,5,10,20,40]
    #
    #   症状 C — repeats=100 下置信区间过宽：
    #     → 增加到 repeats=200，或使用 paired t-test 而非均值比较
    # ============================================================
    import pandas as pd
    from .plotting import plot_exp7_ablation_bars
    base = Path(save_dir)
    out = ensure_dir(base / "results" / "exp7_ablation")
    fig_dir = ensure_dir(base / "figures")
    tab_dir = ensure_dir(base / "tables")
    log = setup_logger("exp7", base / "logs" / "exp7_ablation.log")
    sampler = SamplerConfig()
    mu = [0.0, 0.0, 2.0, 8.0, 25.0, 80.0]
    variants = {
        "GR_RHS_full": {"grrhs_kwargs": {"alpha_kappa": 0.5, "beta_kappa": 1.0, "use_group_scale": True, "shared_kappa": False}, "method": "GR_RHS"},
        "GR_RHS_no_ag": {"grrhs_kwargs": {"alpha_kappa": 0.5, "beta_kappa": 1.0, "use_group_scale": False, "shared_kappa": False}, "method": "GR_RHS"},
        "GR_RHS_no_local_scales": {"grrhs_kwargs": {"alpha_kappa": 0.5, "beta_kappa": 1.0, "use_group_scale": True, "use_local_scale": False, "shared_kappa": False}, "method": "GR_RHS"},
        "GR_RHS_shared_kappa": {"grrhs_kwargs": {"alpha_kappa": 0.5, "beta_kappa": 1.0, "use_group_scale": True, "shared_kappa": True}, "method": "GR_RHS"},
        "GR_RHS_no_kappa": {"grrhs_kwargs": {}, "method": "RHS"},
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
    plot_exp7_ablation_bars(table, out_path=fig_dir / "fig7_ablation_metrics.png")
    save_json(
        {
            "mu": [float(v) for v in mu],
            "dgp_types": dgp_types,
            "variants": list(variants.keys()),
            "repeats": int(repeats),
            "sparse_within_group_rho_within": 0.3,
        },
        out / "exp7_meta.json",
    )
    log.info("Completed exp7 with repeats=%d", repeats)
    return {"raw": str(out / "raw_results.csv"), "table": str(tab_dir / "table_ablation.csv")}


def run_exp8_tau_calibration(n_jobs: int = 1, seed: int = MASTER_SEED, repeats: int = 100, save_dir: str = "simulation_project") -> Dict[str, str]:
    # ============================================================
    # EXP8 — τ 自动校准验证
    #
    # 【理论基准】
    #   Carvalho-Polson-Scott 推荐：τ_target = p₀ / ((p−p₀)√n)
    #   自动校准后验均值应接近 τ_target（tau_rel_error < 0.20 视为成功）
    #
    # 【参数选择说明】
    #   p0_list=[2,6,12,30]：覆盖稀疏（p₀/p=3%）到稠密（p₀/p=50%）
    #   tau_scales=[0.5,1.0,2.0]：固定 τ 分别使用 0.5x/1x/2x τ_target
    #   SamplerConfig(chains=4, warmup=600)：τ 是全局参数，混合较慢，需要足够预热
    #
    # 【若自动校准未能达到 tau_target 的调整方向】
    #   症状 A — tau_post_mean 系统低于 tau_target（自动校准欠估计）：
    #     → 检查 auto_calibrate_tau 的实现：是否用了正确的 p₀ 作为先验信息
    #     → tau_target 本身计算：p0/((p-p0)*sqrt(n))，确认 p₀ 是活跃变量数而非组数
    #     → 尝试 tau_prior_scale=2.0 给自动校准更宽的搜索范围
    #
    #   症状 B — tau_post_sd 很大（后验发散）：
    #     → chains=4 但 warmup=600 可能对 tau 不足；增加到 warmup=1000
    #     → 大 p₀（如 p₀=30/p=60）时先验信息弱，tau 后验天然更宽，属正常现象
    #
    #   症状 C — Fixed 1x tau 与 auto 表现相近但 0.5x/2x 差异不明显：
    #     → beta0 全部设为 2.0（恒等信号），信号过强导致 tau 变化对估计影响小
    #     → 考虑混合信号强度（部分 beta=0.5，部分 beta=2.0）以增强 tau 的影响力
    #
    #   症状 D — n_effective（收敛率）偏低：
    #     → Logistic 版本或高相关设计会导致 tau 混合困难
    #     → 确认 use_auto=True 时 tau0=None（不能同时指定起始值）
    # ============================================================
    import pandas as pd
    from .plotting import plot_exp8_tau
    base = Path(save_dir)
    out = ensure_dir(base / "results" / "exp8_tau_calibration")
    fig_dir = ensure_dir(base / "figures")
    log = setup_logger("exp8", base / "logs" / "exp8_tau_calibration.log")
    sampler = SamplerConfig(chains=4, warmup=600, post_warmup_draws=600)

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
    plot_exp8_tau(raw, out_path=fig_dir / "fig8_tau_calibration.png")
    # Keep a legacy filename for downstream references.
    plot_exp8_tau(raw, out_path=fig_dir / "fig7_tau_calibration.png")
    save_json(
        {
            "n": int(n),
            "p": int(p),
            "group_sizes": list(group_sizes),
            "p0_list": list(p0_list),
            "tau_scales_fixed": list(tau_scales),
            "sampler": {
                "chains": int(sampler.chains),
                "warmup": int(sampler.warmup),
                "post_warmup_draws": int(sampler.post_warmup_draws),
            },
            "primary_error_metric": "tau_rel_error",
        },
        out / "exp8_meta.json",
    )
    log.info("Completed exp8 with repeats=%d", repeats)
    return {"raw": str(out / "raw_results.csv"), "summary": str(out / "summary.csv"), "figure": str(fig_dir / "fig8_tau_calibration.png")}


def run_exp9_beta_prior_sensitivity(n_jobs: int = 1, seed: int = MASTER_SEED, repeats: int = 120, save_dir: str = "simulation_project") -> Dict[str, str]:
    # ============================================================
    # EXP9 — Beta(α_κ, β_κ) 先验敏感性（Theorem 2.8）
    #
    # 【理论连接】
    #   Theorem 2.8：β_κ 控制边际先验 π(β_j) 的尾部指数。
    #   β_κ 越大 → 先验越保守（对大 |β_j| 惩罚更重） → 零组收缩更快
    #   α_κ 越大 → 先验均值更高（κ 更倾向 1） → 信号检测更积极
    #
    # 【参数选择说明】
    #   priors = [(0.5,0.5),(1,1),(1,2),(0.5,1),(2.5,1)]
    #     (0.5,1.0)：默认推荐，轻微倾向 κ→0
    #     (2.5,1.0)：更积极的信号检测（α 大）
    #     (1.0,2.0)：更保守的收缩（β 大）
    #   pg_levels=[20,50]：有限样本下先验影响应在小 p_g 时更大（大 p_g 似然主导）
    #   scenarios：baseline（梯度信号）+ tail_extreme（单极端信号）
    #
    # 【理想结果】
    #   结果对先验"稳健"：5 种先验的 AUROC 差异 < 0.05（鲁棒性）
    #   null_group_kappa_mean：β_κ 增大时应更小（零组更好收缩）
    #   null_group_prob_kappa_gt_0_1（κ_null > 0.1 的概率）：β_κ 增大时应降低
    #   tail_extreme 场景：所有先验都应正确识别极强信号组（AUROC≈1）
    #
    # 【若先验敏感性过大的调整方向】
    #   症状 A — AUROC 随先验剧烈变化（差距 > 0.1）：
    #     → pg 太小（p_g=20,50 处于先验主导区域），增大至 pg_levels=[50,100]
    #     → 或这是合理的结果，说明先验选择对小样本有实质影响，在论文中讨论
    #
    #   症状 B — (0.5,0.5) 先验的零组 κ 无法收缩（null_kappa_mean 偏大）：
    #     → Beta(0.5,0.5) 是双模 U 形分布，大量质量在 0 和 1 附近，但也允许中间值
    #     → 在这种先验下 p_g=20 的似然可能不足以压制先验，这是预期行为
    #
    #   症状 C — kappa_curve（按 p_g 分层的 κ 曲线）无分层差异：
    #     → p_g=20 vs 50 的差距不够；增加 pg_levels=[20,50,100,200]
    #     → 或者先验效应在所有 p_g 下都小，说明似然已主导 → 鲁棒性结论
    # ============================================================
    import pandas as pd
    from .plotting import plot_exp9_prior_sensitivity
    base = Path(save_dir)
    out = ensure_dir(base / "results" / "exp9_beta_prior_sensitivity")
    fig_dir = ensure_dir(base / "figures")
    tab_dir = ensure_dir(base / "tables")
    log = setup_logger("exp9", base / "logs" / "exp9_beta_prior_sensitivity.log")
    sampler = SamplerConfig()
    scenario_templates = [
        ("baseline", [0.0, 0.0, 2.0, 8.0, 25.0, 80.0]),
        ("tail_extreme", [0.0, 0.0, 0.0, 0.0, 0.0, 200.0]),
    ]
    pg_levels = [20, 50]
    scenarios: list[tuple[str, str, list[float], list[int], int]] = []
    for base_name, mu in scenario_templates:
        for pg in pg_levels:
            scenarios.append((f"{base_name}_pg{pg}", base_name, list(mu), [int(pg)] * len(mu), int(pg)))
    priors = [(0.5, 0.5), (1.0, 1.0), (1.0, 2.0), (0.5, 1.0), (2.5, 1.0)]
    tasks: list[tuple[int, float, float, int, int, list[float], np.ndarray, list[int], str, str, int, SamplerConfig]] = []
    for sid, (scenario, scenario_base, mu, group_sizes, p_g) in enumerate(scenarios, start=1):
        labels = (np.asarray(mu) > 0).astype(int)
        for pid, (a, b) in enumerate(priors, start=1):
            for r in range(1, int(repeats) + 1):
                task_id = sid * 100 + pid
                tasks.append((task_id, a, b, r, seed, mu, labels, group_sizes, scenario, scenario_base, int(p_g), sampler))
    rows = _parallel_rows(tasks, _exp9_worker, n_jobs=n_jobs, prefer_process=True, progress_desc="Exp9 Beta Prior Sensitivity")
    raw = pd.DataFrame(rows)
    table = raw.groupby(["scenario", "scenario_base", "p_g", "alpha_kappa", "beta_kappa"], as_index=False).agg(
        null_group_mse_avg=("null_group_mse_avg", "mean"),
        signal_group_mse_avg=("signal_group_mse_avg", "mean"),
        group_auroc=("group_auroc", "mean"),
        null_group_kappa_mean=("null_group_kappa_mean", "mean"),
        signal_group_kappa_mean=("signal_group_kappa_mean", "mean"),
        null_group_prob_kappa_gt_0_1=("null_group_prob_kappa_gt_0_1", "mean"),
        n_effective=("converged", "sum"),
    )
    kappa_curve = raw.groupby(["scenario_base", "p_g", "alpha_kappa", "beta_kappa"], as_index=False).agg(
        null_group_kappa_mean=("null_group_kappa_mean", "mean"),
        null_group_prob_kappa_gt_0_1=("null_group_prob_kappa_gt_0_1", "mean"),
        n_effective=("converged", "sum"),
    )
    save_dataframe(raw, out / "raw_results.csv")
    save_dataframe(table, out / "summary.csv")
    save_dataframe(kappa_curve, out / "summary_kappa_curve.csv")
    save_dataframe(table, tab_dir / "table_beta_prior_sensitivity.csv")
    plot_exp9_prior_sensitivity(table, kappa_curve, out_path=fig_dir / "fig9_beta_prior_sensitivity.png")
    save_json(
        {
            "repeats": int(repeats),
            "pg_levels": pg_levels,
            "scenario_templates": [{"name": name, "mu": mu} for name, mu in scenario_templates],
            "priors": [{"alpha_kappa": float(a), "beta_kappa": float(b)} for a, b in priors],
        },
        out / "exp9_meta.json",
    )
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
        [10, 20, 50, 100, 200, 500, 1000, 2000],
        [10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
    ]
    exp1_tau_candidates = [0.5, 1.0, 1.5]
    exp1_eps_candidates = [0.1, 0.2]
    exp2_tau_candidates = [0.5, 1.0, 2.0]
    exp3_pg_candidates = [
        [30, 60, 120, 240, 480],
        [30, 60, 120, 240, 480, 960],
    ]
    aid = 0
    for pg1 in exp1_pg_candidates:
        for tau1 in exp1_tau_candidates:
            for eps1 in exp1_eps_candidates:
                for tau2 in exp2_tau_candidates:
                    for pg3 in exp3_pg_candidates:
                        aid += 1
                        cfg = {
                            "attempt": aid,
                            "exp1": {"pg_list": pg1, "tau_eval": tau1, "tail_eps_list": [eps1]},
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
                            tail_eps_list=[eps1],
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
        run_exp4_benchmark_linear(repeats=args.repeats or 100, save_dir=args.save_dir, seed=args.seed, n_jobs=args.n_jobs)
    elif args.experiment == "5":
        run_exp5_heterogeneity(repeats=args.repeats or 100, save_dir=args.save_dir, seed=args.seed, n_jobs=args.n_jobs)
    elif args.experiment == "6":
        run_exp6_grouped_logistic(repeats=args.repeats or 50, save_dir=args.save_dir, seed=args.seed, n_jobs=args.n_jobs)
    elif args.experiment == "7":
        run_exp7_ablation(repeats=args.repeats or 100, save_dir=args.save_dir, seed=args.seed, n_jobs=args.n_jobs)
    elif args.experiment == "8":
        run_exp8_tau_calibration(repeats=args.repeats or 100, save_dir=args.save_dir, seed=args.seed, n_jobs=args.n_jobs)
    elif args.experiment == "9":
        run_exp9_beta_prior_sensitivity(repeats=args.repeats or 120, save_dir=args.save_dir, seed=args.seed, n_jobs=args.n_jobs)
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
