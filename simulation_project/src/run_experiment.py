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


METHODS = ["GR_RHS", "RHS", "GIGG_MMLE", "GIGG_b_small", "GIGG_GHS", "GIGG_b_large", "GHS_plus", "OLS", "LASSO_CV"]
LAPTOP_METHODS = ["GR_RHS", "RHS", "GIGG_MMLE", "GHS_plus", "OLS", "LASSO_CV"]
COMPUTE_PROFILES = ("full", "laptop")


def _normalize_compute_profile(profile: str) -> str:
    p = str(profile).strip().lower()
    if p not in COMPUTE_PROFILES:
        raise ValueError(f"unknown compute profile: {profile}; expected one of {COMPUTE_PROFILES}")
    return p


def _resolve_method_list(methods: Sequence[str] | None, *, profile: str) -> list[str]:
    if methods is None:
        base = METHODS if _normalize_compute_profile(profile) == "full" else LAPTOP_METHODS
        return list(base)
    requested = [str(m).strip() for m in methods]
    unknown = sorted(set(requested) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown methods requested: {unknown}")
    # Keep canonical ordering for stable tables/plots.
    return [m for m in METHODS if m in set(requested)]


def _sampler_for_profile(profile: str, *, experiment: str) -> SamplerConfig:
    p = _normalize_compute_profile(profile)
    exp = str(experiment).strip().lower()
    if p == "full":
        if exp == "exp8":
            return SamplerConfig(chains=4, warmup=600, post_warmup_draws=600)
        return SamplerConfig()
    if exp == "exp8":
        # tau is the global calibration target; keep this experiment slightly heavier.
        return SamplerConfig(
            chains=2,
            warmup=300,
            post_warmup_draws=300,
            adapt_delta=0.92,
            max_treedepth=10,
            strict_adapt_delta=0.97,
            strict_max_treedepth=12,
            max_divergence_ratio=0.01,
            rhat_threshold=1.03,
            ess_threshold=120.0,
        )
    return SamplerConfig(
        chains=1,
        warmup=250,
        post_warmup_draws=250,
        adapt_delta=0.92,
        max_treedepth=10,
        strict_adapt_delta=0.97,
        strict_max_treedepth=12,
        max_divergence_ratio=0.01,
        rhat_threshold=1.03,
        ess_threshold=120.0,
    )


def _gigg_config_for_profile(profile: str) -> dict[str, Any]:
    p = _normalize_compute_profile(profile)
    if p == "full":
        return {
            "iter_mult": 4,
            "iter_floor": 2000,
            "iter_cap": 5000,
            "btrick": True,
            "mmle_burnin_only": True,
        }
    return {
        "iter_mult": 2,
        "iter_floor": 500,
        "iter_cap": 1500,
        "btrick": True,
        "mmle_burnin_only": True,
    }


def _default_repeats(exp: str, profile: str) -> int:
    p = _normalize_compute_profile(profile)
    exp_key = str(exp).strip().lower()
    full = {
        "exp1": 500,
        "exp2": 500,
        "exp3": 200,
        "exp4": 100,
        "exp5": 100,
        "exp6": 50,
        "exp7": 100,
        "exp8": 100,
        "exp9": 120,
    }
    laptop = {
        "exp1": 300,
        "exp2": 300,
        "exp3": 120,
        "exp4": 15,
        "exp5": 20,
        "exp6": 20,
        "exp7": 20,
        "exp8": 30,
        "exp9": 15,
    }
    table = full if p == "full" else laptop
    if exp_key not in table:
        raise ValueError(f"unknown experiment key: {exp}")
    return int(table[exp_key])


_BAYESIAN_METHODS = {"GR_RHS", "RHS", "GIGG_MMLE", "GIGG_b_small", "GIGG_GHS", "GIGG_b_large", "GHS_plus"}
_UNTIL_CONVERGED_RETRY_HARD_CAP = 12
_RETRY_MAX_WARMUP = 8000
_RETRY_MAX_POST_DRAWS = 8000
_RETRY_MAX_GIGG_ITER = 50000


def _is_bayesian_method(method: str) -> bool:
    return str(method) in _BAYESIAN_METHODS


def _default_convergence_retries(profile: str) -> int:
    return 2 if _normalize_compute_profile(profile) == "full" else 1


def _resolve_convergence_retry_limit(
    profile: str,
    max_convergence_retries: int | None,
    *,
    until_bayes_converged: bool,
) -> int:
    if max_convergence_retries is not None:
        return int(max_convergence_retries)
    if bool(until_bayes_converged):
        # Negative value means "until converged" mode with an internal hard cap.
        return -1
    return _default_convergence_retries(profile)


def _retry_budget_from_limit(max_convergence_retries: int) -> tuple[int, bool]:
    retry_raw = int(max_convergence_retries)
    if retry_raw >= 0:
        return retry_raw, False
    return int(_UNTIL_CONVERGED_RETRY_HARD_CAP), True


def _scale_sampler_for_retry(base: SamplerConfig, attempt: int) -> SamplerConfig:
    k = max(0, int(attempt))
    if k == 0:
        return base
    mul = int(2 ** k)
    return SamplerConfig(
        chains=max(1, int(base.chains)),
        warmup=min(_RETRY_MAX_WARMUP, max(50, int(base.warmup) * mul)),
        post_warmup_draws=min(_RETRY_MAX_POST_DRAWS, max(50, int(base.post_warmup_draws) * mul)),
        adapt_delta=min(0.995, float(base.adapt_delta) + 0.02 * k),
        max_treedepth=min(15, int(base.max_treedepth) + k),
        strict_adapt_delta=min(0.999, float(base.strict_adapt_delta) + 0.01 * k),
        strict_max_treedepth=min(16, int(base.strict_max_treedepth) + k),
        max_divergence_ratio=float(base.max_divergence_ratio),
        rhat_threshold=float(base.rhat_threshold),
        ess_threshold=float(base.ess_threshold),
    )


def _scale_gigg_config_for_retry(cfg: dict[str, Any], attempt: int) -> dict[str, Any]:
    k = max(0, int(attempt))
    if k == 0:
        return dict(cfg)
    out = dict(cfg)
    mul = int(2 ** k)
    iter_mult = int(out.get("iter_mult", 1))
    iter_floor = int(out.get("iter_floor", 500))
    iter_cap = int(out.get("iter_cap", 1500))
    out["iter_mult"] = max(1, iter_mult * mul)
    out["iter_floor"] = min(_RETRY_MAX_GIGG_ITER, max(10, iter_floor * mul))
    out["iter_cap"] = min(_RETRY_MAX_GIGG_ITER, max(out["iter_floor"], iter_cap * mul))
    return out


def _invalidate_unconverged_result(res: FitResult, *, method: str, attempts: int) -> FitResult:
    # Enforce posterior-trust policy: non-converged Bayesian fits are kept as failed.
    msg = f"ConvergenceError: {method} did not converge after {attempts} attempt(s)"
    if str(res.error).strip():
        msg = f"{msg}; last_error={res.error}"
    res.status = "error"
    res.error = msg
    res.converged = False
    res.beta_mean = None
    res.beta_draws = None
    res.kappa_draws = None
    res.group_scale_draws = None
    res.tau_draws = None
    return res


def _fit_with_convergence_retry(
    fit_fn,
    *,
    method: str,
    sampler: SamplerConfig,
    max_convergence_retries: int,
    enforce_bayes_convergence: bool,
) -> FitResult:
    retry_max, until_mode = _retry_budget_from_limit(int(max_convergence_retries))
    res: FitResult | None = None
    attempts = 1
    for attempt in range(retry_max + 1):
        attempts = attempt + 1
        sampler_try = _scale_sampler_for_retry(sampler, attempt)
        res = fit_fn(sampler_try, attempt)
        if not bool(enforce_bayes_convergence):
            break
        if bool(res.status == "ok" and res.converged and (res.beta_mean is not None)):
            break
    assert res is not None
    if bool(enforce_bayes_convergence) and _is_bayesian_method(method):
        if not bool(res.status == "ok" and res.converged and (res.beta_mean is not None)):
            res = _invalidate_unconverged_result(res, method=method, attempts=attempts)
    res = _attach_retry_diagnostics(
        res,
        method=method,
        attempts=attempts,
        retry_max=retry_max,
        until_mode=until_mode,
        enforce_bayes_convergence=bool(enforce_bayes_convergence),
    )
    return res


def _attach_retry_diagnostics(
    res: FitResult,
    *,
    method: str,
    attempts: int,
    retry_max: int,
    until_mode: bool,
    enforce_bayes_convergence: bool,
) -> FitResult:
    diag = dict(res.diagnostics or {})
    diag["convergence_retry"] = {
        "method": str(method),
        "attempts_used": int(max(1, attempts)),
        "max_attempts": int(max(1, retry_max + 1)),
        "until_converged_mode": bool(until_mode),
        "enforce_bayes_convergence": bool(enforce_bayes_convergence),
        "status": str(res.status),
        "converged": bool(res.converged),
    }
    res.diagnostics = diag
    return res


def _attempts_used(res: FitResult) -> int:
    diag = res.diagnostics if isinstance(res.diagnostics, dict) else {}
    retry = diag.get("convergence_retry", {}) if isinstance(diag, dict) else {}
    try:
        return int(retry.get("attempts_used", 1))
    except Exception:
        return 1


def _paired_converged_subset(
    raw,
    *,
    group_cols: Sequence[str],
    method_col: str,
    replicate_col: str,
    converged_col: str,
    required_cols: Sequence[str],
    method_levels: Sequence[str] | None = None,
):
    import pandas as pd

    group_cols_use = [str(c) for c in group_cols]
    if raw.empty:
        stats_cols = list(group_cols_use) + [
            "n_total_replicates",
            "n_common_replicates",
            "common_rate",
            "methods_required",
            "methods_list",
        ]
        return raw.copy(), pd.DataFrame(columns=stats_cols)

    work = raw.copy()
    work[method_col] = work[method_col].astype(str)
    methods_present = sorted(set(work[method_col].tolist()))
    if method_levels is not None:
        methods_target = [str(m) for m in method_levels if str(m) in set(methods_present)]
    else:
        methods_target = methods_present
    if not methods_target:
        stats_cols = list(group_cols_use) + [
            "n_total_replicates",
            "n_common_replicates",
            "common_rate",
            "methods_required",
            "methods_list",
        ]
        return work.iloc[0:0].copy(), pd.DataFrame(columns=stats_cols)

    work = work.loc[work[method_col].isin(methods_target)].copy()
    valid = work[converged_col].fillna(False).astype(bool)
    for c in required_cols:
        valid &= work[c].notna()
    work["_pair_valid"] = valid

    key_cols = list(group_cols_use) + [str(replicate_col)]
    pivot = work.pivot_table(index=key_cols, columns=method_col, values="_pair_valid", aggfunc="max")
    for m in methods_target:
        if m not in pivot.columns:
            pivot[m] = False
    common_idx = pivot[methods_target].fillna(False).all(axis=1)
    common_keys = pivot.loc[common_idx].reset_index()[key_cols]

    paired = work.merge(common_keys, on=key_cols, how="inner")
    paired = paired.drop(columns=["_pair_valid"], errors="ignore")

    if group_cols_use:
        total = work.groupby(group_cols_use, as_index=False).agg(n_total_replicates=(replicate_col, "nunique"))
        if common_keys.empty:
            common = total[group_cols_use].copy()
            common["n_common_replicates"] = 0
        else:
            common = common_keys.groupby(group_cols_use, as_index=False).agg(n_common_replicates=(replicate_col, "nunique"))
        stats = total.merge(common, on=group_cols_use, how="left")
    else:
        stats = pd.DataFrame(
            [
                {
                    "n_total_replicates": int(work[replicate_col].nunique()),
                    "n_common_replicates": int(common_keys[replicate_col].nunique()) if not common_keys.empty else 0,
                }
            ]
        )

    stats["n_common_replicates"] = stats["n_common_replicates"].fillna(0).astype(int)
    stats["n_total_replicates"] = stats["n_total_replicates"].fillna(0).astype(int)
    stats["common_rate"] = stats["n_common_replicates"] / stats["n_total_replicates"].clip(lower=1)
    stats["methods_required"] = int(len(methods_target))
    stats["methods_list"] = "|".join(methods_target)
    return paired, stats


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
    # theta(u0, rho) = u0 * rho^2 / (u0 + (1-u0) * rho^2): key quantity in Theorem 3.32.
    u = float(u0)
    rho2 = float(rho) ** 2
    den = u + (1.0 - u) * rho2
    return float((u * rho2) / max(den, 1e-12))


def xi_crit_u0_rho(u0: float, rho: float) -> float:
    # xi_crit = theta(u0, rho) / 2. If xi > xi_crit then P(kappa > u0 | Y) increases.
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


def _exp4_worker(
    task: tuple[int, str, dict[str, Any], float, int, int, SamplerConfig, list[str], dict[str, Any], bool, int]
) -> list[dict[str, Any]]:
    from .dgp_grouped_linear import build_linear_beta, generate_grouped_linear_dataset

    sid, setting, spec, target_snr, r, seed, sampler, methods, gigg_config, enforce_bayes_convergence, max_convergence_retries = task
    s = experiment_seed(4, sid, r, master_seed=seed)
    beta_setting = str(spec.get("beta_setting", setting))
    beta_shape = build_linear_beta(beta_setting, spec["group_sizes"])
    ds = generate_grouped_linear_dataset(
        n=500,
        group_sizes=spec["group_sizes"],
        rho_within=spec["rho_within"],
        rho_between=spec["rho_between"],
        beta_shape=beta_shape,
        seed=s,
        target_snr=float(target_snr),
        design_type=str(spec.get("design_type", "correlated")),
    )
    fits = _fit_all_methods(
        ds["X"],
        ds["y"],
        ds["groups"],
        task="gaussian",
        seed=s,
        p0=int(np.sum(np.abs(ds["beta0"]) > 0.0)),
        sampler=sampler,
        methods=methods,
        gigg_config=gigg_config,
        enforce_bayes_convergence=bool(enforce_bayes_convergence),
        max_convergence_retries=int(max_convergence_retries),
    )
    out_rows: list[dict[str, Any]] = []
    for method, result in fits.items():
        metrics = _evaluate_method_row(result, ds["beta0"])
        out_rows.append(
            {
                "setting": setting,
                "beta_setting": beta_setting,
                "target_snr": float(target_snr),
                "replicate_id": r,
                "method": method,
                "status": result.status,
                "converged": result.converged,
                "runtime_seconds": result.runtime_seconds,
                "rhat_max": result.rhat_max,
                "bulk_ess_min": result.bulk_ess_min,
                "divergence_ratio": result.divergence_ratio,
                "error": result.error,
                "fit_attempts": _attempts_used(result),
                **metrics,
            }
        )
    return out_rows


def _exp5_worker(
    task: tuple[int, int, list[int], list[float], SamplerConfig, list[str], dict[str, Any], bool, int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from .dgp_grouped_linear import generate_heterogeneity_dataset
    from .metrics import group_auroc, group_l2_error, group_l2_score

    r, seed, group_sizes, mu, sampler, methods, gigg_config, enforce_bayes_convergence, max_convergence_retries = task
    labels = (np.asarray(mu) > 0.0).astype(int)
    s = experiment_seed(5, 1, r, master_seed=seed)
    ds = generate_heterogeneity_dataset(n=300, group_sizes=group_sizes, rho_within=0.3, rho_between=0.05, sigma2=1.0, mu=mu, seed=s)
    fits = _fit_all_methods(
        ds["X"],
        ds["y"],
        ds["groups"],
        task="gaussian",
        seed=s,
        p0=int(np.sum(labels)),
        sampler=sampler,
        methods=methods,
        gigg_config=gigg_config,
        enforce_bayes_convergence=bool(enforce_bayes_convergence),
        max_convergence_retries=int(max_convergence_retries),
    )
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
                    "fit_attempts": _attempts_used(res),
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
                "fit_attempts": _attempts_used(res),
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


def _exp6_worker(
    task: tuple[int, int, np.ndarray, SamplerConfig, float, list[str], dict[str, Any], bool, int]
) -> list[dict[str, Any]]:
    from .dgp_grouped_logistic import generate_grouped_logistic_dataset

    r, seed, beta0, sampler, min_separator_auc, methods, gigg_config, enforce_bayes_convergence, max_convergence_retries = task
    s = experiment_seed(6, 1, r, master_seed=seed)
    ds = generate_grouped_logistic_dataset(
        n=200,
        group_sizes=[5, 5, 5],
        rho_within=0.5,
        rho_between=0.05,
        beta0=beta0,
        seed=s,
        min_separator_auc=float(min_separator_auc),
    )
    fits = _fit_all_methods(
        ds["X"],
        ds["y"],
        ds["groups"],
        task="logistic",
        seed=s,
        p0=4,
        sampler=sampler,
        methods=methods,
        gigg_config=gigg_config,
        enforce_bayes_convergence=bool(enforce_bayes_convergence),
        max_convergence_retries=int(max_convergence_retries),
    )
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
                "fit_attempts": _attempts_used(res),
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


def _exp7_worker(
    task: tuple[int, int, list[int], list[float], str, dict[str, Any], SamplerConfig, bool, int]
) -> list[dict[str, Any]]:
    from .dgp_grouped_linear import generate_heterogeneity_dataset, generate_sparse_within_group_dataset
    from .fit_gr_rhs import fit_gr_rhs
    from .fit_rhs import fit_rhs
    from .metrics import group_auroc, group_l2_error, group_l2_score

    r, seed, group_sizes, mu, dgp_type, variants, sampler, enforce_bayes_convergence, max_convergence_retries = task
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
            res = _fit_with_convergence_retry(
                lambda sampler_try, attempt: fit_gr_rhs(
                    ds["X"],
                    ds["y"],
                    ds["groups"],
                    task="gaussian",
                    seed=s + 11 + 100 * attempt,
                    p0=int(np.sum(labels)),
                    sampler=sampler_try,
                    **spec["grrhs_kwargs"],
                ),
                method="GR_RHS",
                sampler=sampler,
                max_convergence_retries=int(max_convergence_retries),
                enforce_bayes_convergence=bool(enforce_bayes_convergence),
            )
        else:
            res = _fit_with_convergence_retry(
                lambda sampler_try, attempt: fit_rhs(
                    ds["X"],
                    ds["y"],
                    ds["groups"],
                    task="gaussian",
                    seed=s + 12 + 100 * attempt,
                    p0=int(np.sum(labels)),
                    sampler=sampler_try,
                ),
                method="RHS",
                sampler=sampler,
                max_convergence_retries=int(max_convergence_retries),
                enforce_bayes_convergence=bool(enforce_bayes_convergence),
            )
        is_valid = bool(res.converged and (res.beta_mean is not None))
        if not is_valid:
            out_rows.append(
                {
                    "replicate_id": r,
                    "dgp_type": str(dgp_type),
                    "variant": vname,
                    "status": res.status,
                    "converged": bool(res.converged),
                    "fit_attempts": _attempts_used(res),
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
                "fit_attempts": _attempts_used(res),
                "null_group_mse_avg": float(np.mean(err[labels == 0])),
                "signal_group_mse_avg": float(np.mean(err[labels == 1])),
                "overall_mse": float(np.mean((res.beta_mean - ds["beta0"]) ** 2)),
                "group_auroc": group_auroc(score, labels),
            }
        )
    return out_rows


def _exp9_worker(
    task: tuple[int, int, str, str, list[float], list[int], int, SamplerConfig, list[tuple[float, float]], bool, int]
) -> list[dict[str, Any]]:
    from .dgp_grouped_linear import generate_heterogeneity_dataset
    from .fit_gr_rhs import fit_gr_rhs
    from .metrics import group_auroc, group_l2_error, group_l2_score

    sid, r, scenario, scenario_base, mu, group_sizes, seed, sampler, priors, enforce_bayes_convergence, max_convergence_retries = task
    labels = (np.asarray(mu) > 0).astype(int)
    p_g = int(group_sizes[0]) if group_sizes else 0
    s = experiment_seed(9, sid, r, master_seed=seed)
    ds = generate_heterogeneity_dataset(
        n=300,
        group_sizes=group_sizes,
        rho_within=0.3,
        rho_between=0.05,
        sigma2=1.0,
        mu=mu,
        seed=s,
    )

    rows: list[dict[str, Any]] = []
    for pid, (a, b) in enumerate(priors, start=1):
        res = _fit_with_convergence_retry(
            lambda sampler_try, attempt: fit_gr_rhs(
                ds["X"],
                ds["y"],
                ds["groups"],
                task="gaussian",
                seed=s + 100 + pid + 100 * attempt,
                p0=int(np.sum(labels)),
                sampler=sampler_try,
                alpha_kappa=float(a),
                beta_kappa=float(b),
                use_group_scale=True,
                shared_kappa=False,
            ),
            method="GR_RHS",
            sampler=sampler,
            max_convergence_retries=int(max_convergence_retries),
            enforce_bayes_convergence=bool(enforce_bayes_convergence),
        )
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
            rows.append(
                {
                    "alpha_kappa": float(a),
                    "beta_kappa": float(b),
                    "scenario": str(scenario),
                    "scenario_base": str(scenario_base),
                    "p_g": int(p_g),
                    "replicate_id": int(r),
                    "status": res.status,
                    "converged": bool(res.converged),
                    "fit_attempts": _attempts_used(res),
                    "null_group_mse_avg": float("nan"),
                    "signal_group_mse_avg": float("nan"),
                    "group_auroc": float("nan"),
                    "null_group_kappa_mean": null_kappa_mean,
                    "signal_group_kappa_mean": signal_kappa_mean,
                    "null_group_prob_kappa_gt_0_1": null_prob_kappa_gt_0_1,
                }
            )
            continue
        score = group_l2_score(res.beta_mean, ds["groups"])
        err = group_l2_error(res.beta_mean, ds["beta0"], ds["groups"])
        rows.append(
            {
                "alpha_kappa": float(a),
                "beta_kappa": float(b),
                "scenario": str(scenario),
                "scenario_base": str(scenario_base),
                "p_g": int(p_g),
                "replicate_id": int(r),
                "status": res.status,
                "converged": bool(res.converged),
                "fit_attempts": _attempts_used(res),
                "null_group_mse_avg": float(np.mean(err[labels == 0])),
                "signal_group_mse_avg": float(np.mean(err[labels == 1])),
                "group_auroc": group_auroc(score, labels),
                "null_group_kappa_mean": null_kappa_mean,
                "signal_group_kappa_mean": signal_kappa_mean,
                "null_group_prob_kappa_gt_0_1": null_prob_kappa_gt_0_1,
            }
        )
    return rows


def _exp8_worker(
    task: tuple[int, int, int, list[int], SamplerConfig, float, list[float], bool, int]
) -> list[dict[str, Any]]:
    from .fit_gr_rhs import fit_gr_rhs
    from .utils import canonical_groups, sample_correlated_design

    p0, r, seed, group_sizes, sampler, tau_target, tau_scales, enforce_bayes_convergence, max_convergence_retries = task
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
    n_strong = max(1, int(math.ceil(0.5 * int(p0))))
    strong_idx = active[:n_strong]
    weak_idx = active[n_strong:]
    beta[strong_idx] = 2.0
    if weak_idx.size > 0:
        beta[weak_idx] = 0.5
    y = X @ beta + np.random.default_rng(s + 23).normal(0.0, 1.0, size=n)
    p0_fit = int(np.sum(np.abs(beta) > 0.0))
    rows: list[dict[str, Any]] = []
    modes: list[tuple[str, float, bool]] = [("auto_calibrated", 1.0, True)]
    for sc in tau_scales:
        modes.append((f"fixed_{float(sc):.2f}x", float(sc), False))

    for mode_idx, (mode_name, tau_prior_scale, use_auto) in enumerate(modes, start=1):
        tau0_use = float(tau_prior_scale) * float(tau_target)
        res = _fit_with_convergence_retry(
            lambda sampler_try, attempt: fit_gr_rhs(
                X,
                y,
                groups,
                task="gaussian",
                seed=s + 31 + mode_idx + 100 * attempt,
                p0=p0_fit,
                sampler=sampler_try,
                alpha_kappa=0.5,
                beta_kappa=1.0,
                use_group_scale=True,
                use_local_scale=True,
                shared_kappa=False,
                auto_calibrate_tau=bool(use_auto),
                tau0=None if use_auto else tau0_use,
            ),
            method="GR_RHS",
            sampler=sampler,
            max_convergence_retries=int(max_convergence_retries),
            enforce_bayes_convergence=bool(enforce_bayes_convergence),
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

        rows.append(
            {
                "p0": int(p0),
                "p": int(p),
                "n": int(n),
                "replicate_id": int(r),
                "tau_target": float(tau_target),
                "tau_prior_scale": float(tau_prior_scale),
                "tau_mode": mode_name,
                "tau_post_mean": tau_mean,
                "tau_post_sd": tau_sd,
                "kappa_eff_sum_post_mean": kappa_eff,
                "converged": bool(res.converged),
                "status": str(res.status),
                "fit_attempts": _attempts_used(res),
                "signal_var_true": float(beta.T @ cov_x @ beta),
                "n_strong_active": int(strong_idx.size),
                "n_weak_active": int(weak_idx.size),
            }
        )
    return rows


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
    methods: Sequence[str] | None = None,
    gigg_config: dict[str, Any] | None = None,
    enforce_bayes_convergence: bool = True,
    max_convergence_retries: int = 2,
) -> Dict[str, FitResult]:
    from .fit_classical import fit_lasso_cv, fit_ols
    from .fit_gigg import fit_gigg_fixed, fit_gigg_mmle
    from .fit_ghs_plus import fit_ghs_plus
    from .fit_gr_rhs import fit_gr_rhs
    from .fit_rhs import fit_rhs

    n = X.shape[0]
    grrhs_kwargs = grrhs_kwargs or {}
    methods_use = _resolve_method_list(methods, profile="full") if methods is not None else list(METHODS)
    gigg_cfg = dict(gigg_config or {})
    gigg_mmle_cfg = dict(gigg_cfg)
    gigg_fixed_cfg = {k: v for k, v in gigg_cfg.items() if k != "mmle_burnin_only"}
    out: Dict[str, FitResult] = {}

    def _fit_once(method: str, attempt: int) -> FitResult:
        sampler_try = _scale_sampler_for_retry(sampler, attempt)
        gigg_try = _scale_gigg_config_for_retry(gigg_cfg, attempt)
        gigg_mmle_try = dict(gigg_try)
        gigg_fixed_try = {k: v for k, v in gigg_try.items() if k != "mmle_burnin_only"}

        if method == "GR_RHS":
            return fit_gr_rhs(X, y, groups, task=task, seed=seed + 1 + 100 * attempt, p0=p0, sampler=sampler_try, **grrhs_kwargs)
        if method == "RHS":
            return fit_rhs(X, y, groups, task=task, seed=seed + 2 + 100 * attempt, p0=p0, sampler=sampler_try)
        if method == "GIGG_MMLE":
            return fit_gigg_mmle(X, y, groups, task=task, seed=seed + 3 + 100 * attempt, sampler=sampler_try, p0=p0, **gigg_mmle_try)
        if method == "GIGG_b_small":
            return fit_gigg_fixed(
                X,
                y,
                groups,
                task=task,
                seed=seed + 5 + 100 * attempt,
                sampler=sampler_try,
                p0=p0,
                a_val=1.0 / n,
                b_val=1.0 / n,
                method_label="GIGG_b_small",
                **gigg_fixed_try,
            )
        if method == "GIGG_GHS":
            return fit_gigg_fixed(
                X,
                y,
                groups,
                task=task,
                seed=seed + 6 + 100 * attempt,
                sampler=sampler_try,
                p0=p0,
                a_val=0.5,
                b_val=0.5,
                method_label="GIGG_GHS",
                **gigg_fixed_try,
            )
        if method == "GIGG_b_large":
            return fit_gigg_fixed(
                X,
                y,
                groups,
                task=task,
                seed=seed + 7 + 100 * attempt,
                sampler=sampler_try,
                p0=p0,
                a_val=1.0 / n,
                b_val=1.0,
                method_label="GIGG_b_large",
                **gigg_fixed_try,
            )
        if method == "GHS_plus":
            return fit_ghs_plus(X, y, groups, task=task, seed=seed + 4 + 100 * attempt, p0=p0, sampler=sampler_try)
        if method == "OLS":
            return fit_ols(X, y, task=task, seed=seed + 8)
        if method == "LASSO_CV":
            return fit_lasso_cv(X, y, task=task, seed=seed + 9)
        raise ValueError(f"Unsupported method: {method}")

    retry_max, until_mode = _retry_budget_from_limit(int(max_convergence_retries))
    for method in methods_use:
        res: FitResult | None = None
        attempts = 1
        if bool(enforce_bayes_convergence) and _is_bayesian_method(method):
            for attempt in range(retry_max + 1):
                attempts = attempt + 1
                res = _fit_once(method, attempt)
                if bool(res.status == "ok" and res.converged and (res.beta_mean is not None)):
                    break
            assert res is not None
            if not bool(res.status == "ok" and res.converged and (res.beta_mean is not None)):
                res = _invalidate_unconverged_result(res, method=method, attempts=attempts)
        else:
            res = _fit_once(method, 0)
        res = _attach_retry_diagnostics(
            res,
            method=str(method),
            attempts=int(attempts),
            retry_max=retry_max,
            until_mode=until_mode,
            enforce_bayes_convergence=bool(enforce_bayes_convergence),
        )
        out[str(method)] = res
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
    # EXP1 鈥?闆舵敹缂╅獙璇侊紙Theorem 3.22锛?
    #
    # 銆愮悊璁洪娴嬨€?
    #   E[魏_g | Y_null] = O_{P鈧€}(p_g^{-1/2})
    #   鈫?log-log 鏂滅巼鐩爣锛?0.5
    #   鈫?鍥哄畾闃堝€煎熬姒傜巼 P(魏 > 蔚) 鈫?0锛堜换鎰忓浐瀹?蔚 > 0锛?
    #
    # 銆愬弬鏁伴€夋嫨璇存槑銆?
    #   pg_list  = [10,...,2000]锛氳鐩?pre-asymptotic 鍒?asymptotic 鍖哄煙
    #   tau_eval = 0.5锛毾?= 蟿/鈭毾兟?= 0.5锛屽眳涓€夊彇浠ラ伩鍏?蟻鈫?/鈭?鐨勯€€鍖?
    #   alpha_kappa=0.5, beta_kappa=1.0锛氬厛楠屽 魏=0 鏈夎交寰€惧悜锛堟瘮 Beta(1,1) 鏀剁缉蹇級
    #   tail_eps_list=[0.1, 0.2]锛氬浐瀹氶槇鍊硷紙涓嶉殢 p_g 缂╂斁锛夛紝鐩存帴娴嬮浂鏀剁缉
    #
    # 銆愯嫢 log-log 鏂滅巼鏈揪鍒?-0.5 鐨勮皟鏁存柟鍚戙€?
    #   鐥囩姸 A 鈥?鏂滅巼 < -0.5锛堣繃蹇笅闄嶏級锛?
    #     鈫?tau_eval 鍋忓皬锛埾?澶皬锛屽厛楠屽 魏 鐨勭害鏉熷緢寮猴級锛屽皾璇?tau_eval=1.0
    #     鈫?alpha_kappa 鍋忓皬锛屽厛楠屽 魏 鏂藉姞杩囧己鐨勪笅鍘嬶紝灏濊瘯 alpha_kappa=1.0
    #
    #   鐥囩姸 B 鈥?鏂滅巼 > -0.5锛堜笅闄嶅お鎱紝濡?-0.30锛夛細
    #     鈫?p_g 鑼冨洿涓嶅澶э紝娓愯繎鍖哄煙鏈埌杈撅紱鎵╁睍 pg_list 鍒?[500,1000,2000,5000]
    #     鈫?beta_kappa 鍋忓皬锛堝厛楠屽潎鍊?伪/(伪+尾) 杩囧ぇ锛夛紝灏濊瘯 beta_kappa=2.0
    #     鈫?repeats 涓嶈冻瀵艰嚧 median 浼拌鍣０澶э紝澧炲姞鍒?1000
    #
    #   鐥囩姸 C 鈥?灏炬鐜?P(魏 > 0.1) 闈炲崟璋冿紙鍑虹幇椹煎嘲锛夛細
    #     鈫?妫€鏌?tau_eval锛氬浜庢瀬灏?蟿锛屽悗楠屽湪灏?p_g 鏃跺氨寰堥泦涓紝澶?p_g 鏃跺彉鍖栦笉澶?
    #     鈫?澧炲姞 tail_eps_list 涓殑 蔚 鍊硷紙濡?0.3, 0.5锛変互瑙傚療涓嶅悓灏鹃儴琛屼负
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
    # 鐞嗚鐩爣锛歴lope 鈮?-0.5锛?5% CI 瑕嗙洊 -0.5锛?
    # 鑻ユ枩鐜囨樉钁楀亸绂伙紝瑙佸嚱鏁板ご閮ㄦ敞閲婄殑銆愮棁鐘?A/B銆戣皟鏁存柟鍚?
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
    # EXP2 鈥?鑷€傚簲灞€閮ㄥ寲楠岃瘉锛圱heorem 3.30锛?
    #
    # 銆愮悊璁洪娴嬨€?
    #   涓棿淇″彿涓嬶紙r_g = 渭_g虏/p_g 鈫?鈭烇級锛屽悗楠岄泦涓簬 [x_- s_g, x_+ s_g]
    #   鍏朵腑 s_g = 渭_g/p_g锛寈_-, x_+ 涓哄浐瀹氬父鏁帮紙涓嶄緷璧?p_g锛?
    #   鈫?ratio_R = E[魏|Y] / s_g 鈫?1锛堝悗楠屽潎鍊兼敹鏁涘埌 s_g锛?
    #   鈫?P(魏 鈭?[x_lo路s_g, x_hi路s_g]) 鈫?1
    #
    # 銆愬弬鏁伴€夋嫨璇存槑銆?
    #   mu_coef=3.0锛毼糭g = 3路p_g^{0.75}锛屼娇 r_g = 9路p_g^{0.5} 鈫?鈭烇紙婊¤冻 Cond. 3.25锛?
    #                 鍚屾椂 s_g = 3路p_g^{-0.25} 浣?s_g 鈭?(0,1) 瀵规墍鏈夎瀹氱殑 p_g 鎴愮珛
    #   x_lo=0.5, x_hi=2.0锛氬畾鐞嗙殑"鍥哄畾甯告暟绐楀彛"銆傜獥鍙ｅ搴?= 1.5路s_g锛堝浐瀹氭瘮渚嬶級
    #                         娉ㄦ剰锛氭棫鐗堟浘閿欑敤 (1卤C/鈭歱_g) 鐨勭缉绐勭獥鍙ｏ紝宸蹭慨姝ｄ负姝?
    #   tau_list=[0.5,1.0,2.0]锛氳法瓒?蟻<1/蟻=1/蟻>1 涓夌鍖哄煙娴嬭瘯鏅€傛€?
    #
    # 銆愯嫢 ratio_R 鏈敹鏁涘埌 1 鐨勮皟鏁存柟鍚戙€?
    #   鐥囩姸 A 鈥?ratio_R 闅?p_g 鍗曡皟澧炲姞锛? 1锛夛細
    #     鈫?s_g 瀵规墍鏈夊ぇ p_g 閮藉亸灏忥紙< 0.5锛夛紝Beta 鍏堥獙鍧囧€?伪/(伪+尾) 灏?魏 涓婃帹
    #     鈫?灏濊瘯澧炲ぇ mu_coef锛堝 mu_coef=5.0锛変娇 s_g 鏇村ぇ锛岃繙绂诲厛楠屽潎鍊?
    #     鈫?鎴栧噺灏?beta_kappa锛堝 beta_kappa=0.5锛変娇鍏堥獙鍧囧€奸潬杩?0
    #
    #   鐥囩姸 B 鈥?ratio_R 闅?p_g 鍗曡皟鍑忓皯锛? 1锛夛細
    #     鈫?s_g 鍋忓ぇ锛堟帴杩?1锛夛紝魏 绌洪棿杈圭晫鏁堝簲锛埼?鈭?(0,1)锛夊鑷村潎鍊煎亸浣?
    #     鈫?灏濊瘯鍑忓皬 mu_coef锛堝 mu_coef=1.5锛?
    #
    #   鐥囩姸 C 鈥?window_prob 涓嶉殢 p_g 澧炲ぇ锛堝眬閮ㄥ寲涓嶆垚绔嬶級锛?
    #     鈫?妫€鏌?r_g = mu_g虏/p_g 鏄惁瓒冲澶э細鎵撳嵃 mu_g虏/p_g 瀵瑰悇 p_g 鐨勫€?
    #     鈫?鑻?r_g 澧為暱鎱紝灏濊瘯 mu_coef^2 > p_g^{0.5} 鏇村揩澧為暱锛堝 mu_coef * p_g^{0.6}锛?
    #     鈫?鎵╁睍 pg_list 鍒?[20,50,100,200,500,1000]
    #
    #   鐥囩姸 D 鈥?涓嶅悓 tau 鐨?ratio_R 鏇茬嚎宸紓寰堝ぇ锛坱au 渚濊禆鎬у己锛夛細
    #     鈫?杩欒〃鏄?pre-asymptotic 鏁堝簲涓诲锛涘澶?p_g 鎴栨楠?蟻 鏄惁杩囦簬鏋佺
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
            mu_g = float(mu_coef) * (pg ** 0.75)  # r_g = mu_g虏/p_g = mu_coef虏路p_g^0.5 鈫?鈭?
            scale = mu_g / pg                       # s_g = 渭_g/p_g锛屽畾鐞?3.30 鐨勫眬閮ㄥ寲涓績
            # 绐楀彛涓?[x_lo路s_g, x_hi路s_g]锛歺_lo/x_hi 鏄浐瀹氬父鏁帮紙涓嶄緷璧?p_g锛?
            # 鏃ц璁￠敊璇細(1 卤 C/鈭歱_g)路s_g 鏄缉绐勭獥鍙ｏ紝绐楀彛姒傜巼蹇呯劧瓒嬪悜 0
            # 姝ｇ‘锛氬浐瀹氭瘮渚嬬獥鍙ｄ娇寰?Theorem 3.30 鐨?闆嗕腑浜?[x_-路s_g, x_+路s_g]"鍙互琚獙璇?
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
    # EXP3 鈥?鐩稿彉鍥撅紙Theorem 3.32 / Corollary 3.33锛?
    #
    # 銆愮悊璁洪娴嬨€?
    #   尉_crit = 胃(u鈧€,蟻)/2 = u鈧€蟻虏/(2(u鈧€+(1鈭抲鈧€)蟻虏))
    #   褰?尉 = 渭_g/p_g > 尉_crit 鈫?P(魏_g > u鈧€ | Y) 鈫?1锛坧_g 鈫?鈭烇級
    #   褰?尉 < 尉_crit            鈫?P(魏_g > u鈧€ | Y) 鈫?0
    #   x 杞翠负 尉/尉_crit锛堝凡鏍囧噯鍖栵級锛岀浉鍙樼偣鍥哄畾鍦?1.0
    #
    # 銆愬弬鏁伴€夋嫨璇存槑銆?
    #   p_g_list=[30,60,120,240,480]锛氳鐩栨笎杩戣秼鍔匡紱480 鏄瀵熼攼鍒╄烦璺冪殑鏈€灏?p_g
    #   xi_multiplier_list=[0.3,...,2.0]脳尉_crit锛氶槇鍊间袱渚у潎鍖€閲囨牱
    #   u0=0.5锛毼篲g 鐨勫垽鏂槇鍊硷紙鍙敼鍙樹互娴嬭瘯涓嶅悓 u鈧€ 涓?尉_crit 鍏紡鐨勬纭€э級
    #   theory_check_tau=0.3锛氫富鍥句娇鐢ㄧ殑 蟿 鍙傝€冨€硷紙涓嶅悓 蟿 鐢?蟿-sweep 鍥惧睍绀猴級
    #
    # 銆愯嫢鐩稿彉鏇茬嚎鏈〃鐜板嚭閿愬埄璺宠穬鐨勮皟鏁存柟鍚戙€?
    #   鐥囩姸 A 鈥?鏇茬嚎骞冲潶锛堟棤鏄庢樉璺宠穬锛屽悇 p_g 宸紓灏忥級锛?
    #     鈫?p_g 澶皬锛涘皢 p_g_list 鏈€澶у€兼墿灞曞埌 960 鎴?1920
    #     鈫?妫€鏌?xi_crit 璁＄畻鏄惁姝ｇ‘锛氭墦鍗?xi_crit_u0_rho(u0, tau/sqrt(sigma2))
    #     鈫?纭 mu_g = xi * pg锛堝湪 _exp3_worker 涓級锛屼笉瑕佺敤 mu_g = xi
    #
    #   鐥囩姸 B 鈥?璺宠穬鐐逛笉鍦?尉/尉_crit = 1.0 澶勶紙绯荤粺鍋忕Щ锛夛細
    #     鈫?tau 鍊兼瀬绔紙蟿鈮?.1 鏃?尉_crit 鏋佸皬锛屾暟鍊肩簿搴︿細褰卞搷锛夛紝妫€鏌?蟻 = 蟿/鈭毾兟?
    #     鈫?sigma2 鍙傛暟涓庢暟鎹敓鎴愪腑鐨勫疄闄?蟽虏 涓嶄竴鑷达紝纭繚涓よ€呯浉鍚?
    #     鈫?u0 闇€涓庡垎鏋愮殑闃堝€间竴鑷达紙鏇存敼 u0 浼氬悓鏃剁Щ鍔?尉_crit 鍜屾祴璇曠粺璁￠噺锛?
    #
    #   鐥囩姸 C 鈥?涓嶅悓 蟿 鐨?尉/尉_crit 鏇茬嚎鐩镐簰涓嶉噸鍙狅紙鏍囧噯鍖栧け璐ワ級锛?
    #     鈫?鏍囧噯鍖?x 杞寸殑 尉_crit 鍦ㄤ唬鐮佷腑姝ｇ‘鎸?tau 璁＄畻锛堣 xi_by_tau 瀛楀吀锛夛紝
    #       鑻ユ洸绾夸笉閲嶅彔锛岃鏄庢湁闄愭牱鏈笅 尉_crit 鐨勬湁鏁堝€间笌鐞嗚鍊兼湁鍋忓樊
    #     鈫?澧炲ぇ repeats 鍒?500 浠ュ噺灏戝櫔澹?
    #
    #   鐥囩姸 D 鈥?P(魏 > u鈧€) 鍦?尉 >> 尉_crit 鏃舵棤娉曡揪鍒?1锛?
    #     鈫?alpha_kappa/beta_kappa 鐨勫厛楠屽潎鍊艰繃浣庯紝鍏堥獙鍘嬪埗 魏锛涘皾璇?alpha_kappa=1.0
    #     鈫?grid_size 澶皬瀵艰嚧鍚庨獙绉垎绮惧害浣庯紱澧炲姞鍒?1601
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
        # xi_ratio = 尉/尉_crit锛氭爣鍑嗗寲 x 杞达紝浣夸笉鍚?蟿 鐨勭浉鍙樼偣鍧囪惤鍦?1.0 澶?
        # 鑻?xi_ratio=1.0 澶?P(魏>u鈧€) 鈮?0.5锛岃鏄?finite-p_g 鍋忓樊鎴?尉_crit 鍏紡鍋忓樊
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

    # Convergence gating is enforced upstream in fit wrappers when enabled.
    # Here we still guard on beta_mean availability for robustness.
    if result.beta_mean is None:
        return {"mse_null": float("nan"), "mse_signal": float("nan"), "mse_overall": float("nan"), "avg_ci_length": float("nan"), "coverage_95": float("nan")}
    m = mse_null_signal_overall(result.beta_mean, beta0)
    ci_len, cov = ci_length_and_coverage(beta0, result.beta_draws)
    return {"mse_null": m["mse_null"], "mse_signal": m["mse_signal"], "mse_overall": m["mse_overall"], "avg_ci_length": ci_len, "coverage_95": cov}


def run_exp4_benchmark_linear(
    n_jobs: int = 1,
    seed: int = MASTER_SEED,
    repeats: int = 100,
    save_dir: str = "simulation_project",
    *,
    snr_list: Sequence[float] | None = None,
    profile: str = "full",
    methods: Sequence[str] | None = None,
    enforce_bayes_convergence: bool = True,
    max_convergence_retries: int | None = None,
    until_bayes_converged: bool = True,
) -> Dict[str, str]:
    # ============================================================
    # EXP4 鈥?绾挎€у洖褰掑熀鍑嗘瘮杈冿紙GR_RHS vs RHS / GIGG_MMLE / GHS_plus锛?
    #
    # 銆愮悊鎯崇粨鏋溿€?
    #   GR_RHS 鍦?mse_null锛堥浂缁勶級涓婃樉钁椾紭浜?RHS锛堝己缁勯棿鏀剁缉锛?
    #   GR_RHS 鍦?mse_signal锛堜俊鍙风粍锛変笂涓?RHS 鐩稿綋鎴栨洿濂?
    #   GR_RHS coverage_95 鈮?0.95锛堝尯闂存牎鍑嗚壇濂斤級
    #   楂樼浉鍏冲満鏅紙L3,L4锛塆R_RHS 鐨勪紭鍔垮簲鏇存槑鏄?
    #
    # 銆愯嫢 GR_RHS 涓嶄紭浜?RHS 鐨勮皟鏁存柟鍚戙€?
    #   鐥囩姸 A 鈥?mse_null 鏃犳敼鍠勶紙GR_RHS 鈮?RHS锛夛細
    #     鈫?妫€鏌?alpha_kappa/beta_kappa 璁剧疆锛涘澶?beta_kappa 浣垮厛楠屾洿鍊惧悜 魏鈫?
    #     鈫?纭 GR_RHS 浣跨敤浜?use_group_scale=True锛堢粍闂村昂搴?a_g 鏄垎绂荤殑鍏抽敭锛?
    #
    #   鐥囩姸 B 鈥?mse_signal 鏄捐憲鍙樺樊锛圙R_RHS 杩囧害鏀剁缉淇″彿缁勶級锛?
    #     鈫?tau 璁剧疆鍙兘杩囧皬锛涙鏌?grrhs_kwargs 涓殑 tau0
    #     鈫?淇″彿缁勭殑 渭_g 鎺ヨ繎鐩稿彉闃堝€硷紝澧炲ぇ淇″彿寮哄害鎴栨敼鍙?build_linear_beta 鐨勮缃?
    #
    #   鐥囩姸 C 鈥?coverage_95 杩滀綆浜?0.95锛?
    #     鈫?MCMC 鏈敹鏁涳紙妫€鏌?rhat_max < 1.05, bulk_ess_min > 100锛?
    #     鈫?澧炲姞 chains/warmup 鍦?SamplerConfig 涓?
    #
    #   鐥囩姸 D 鈥?n_effective 鍋忎綆锛堝ぇ閲忎笉鏀舵暃锛夛細
    #     鈫?sampler 榛樿閰嶇疆涓嶈冻锛岄渶鏄惧紡浼犲叆鏇村閾炬暟/姝ユ暟
    #     鈫?repeats=100 鍕夊己澶熺敤锛岃嫢缁撴灉鍣０澶ц€冭檻 repeats=200
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
    profile_name = _normalize_compute_profile(profile)
    sampler = _sampler_for_profile(profile_name, experiment="exp4")
    methods_use = _resolve_method_list(methods, profile=profile_name)
    gigg_cfg = _gigg_config_for_profile(profile_name)
    retry_limit = _resolve_convergence_retry_limit(
        profile_name,
        max_convergence_retries,
        until_bayes_converged=bool(until_bayes_converged),
    )
    snr_vals = [float(v) for v in (snr_list or [0.70])]
    settings = {
        "L0": {"group_sizes": [10, 10, 10, 10, 10], "rho_within": 0.0, "rho_between": 0.0, "design_type": "orthonormal"},
        "L1": {"group_sizes": [10, 10, 10, 10, 10], "rho_within": 0.3, "rho_between": 0.10, "design_type": "correlated"},
        "L2": {"group_sizes": [10, 10, 10, 10, 10], "rho_within": 0.3, "rho_between": 0.10, "design_type": "correlated"},
        "L3": {"group_sizes": [10, 10, 10, 10, 10], "rho_within": 0.8, "rho_between": 0.10, "design_type": "correlated"},
        "L4": {"group_sizes": [10, 10, 10, 10, 10], "rho_within": 0.8, "rho_between": 0.10, "design_type": "correlated"},
        "L5": {"group_sizes": [30, 10, 5, 3, 2], "rho_within": 0.3, "rho_between": 0.10, "design_type": "correlated"},
        "L6": {"group_sizes": [30, 10, 5, 3, 2], "rho_within": 0.3, "rho_between": 0.10, "design_type": "correlated"},
        "L4B20": {"group_sizes": [10, 10, 10, 10, 10], "rho_within": 0.8, "rho_between": 0.20, "design_type": "correlated", "beta_setting": "L4"},
    }
    tasks: list[tuple[int, str, dict[str, Any], float, int, int, SamplerConfig, list[str], dict[str, Any], bool, int]] = []
    sid = 0
    for target_snr in snr_vals:
        for setting, spec in settings.items():
            sid += 1
            for r in range(1, int(repeats) + 1):
                tasks.append((sid, setting, spec, float(target_snr), r, seed, sampler, methods_use, gigg_cfg, bool(enforce_bayes_convergence), int(retry_limit)))

    rows = []
    for chunk in _parallel_rows(tasks, _exp4_worker, n_jobs=n_jobs, prefer_process=True, progress_desc="Exp4 Benchmark Linear"):
        rows.extend(chunk)
    raw = pd.DataFrame(rows)
    raw["estimate_available"] = raw["mse_overall"].notna()

    run_counts = raw.groupby(["target_snr", "setting", "method"], as_index=False).agg(n_total_runs=("replicate_id", "count"))
    convergence_audit = raw.groupby(["target_snr", "setting", "method"], as_index=False).agg(
        n_total_runs=("replicate_id", "count"),
        n_converged=("converged", "sum"),
        convergence_rate=("converged", "mean"),
        fit_attempts_mean=("fit_attempts", "mean"),
        fit_attempts_max=("fit_attempts", "max"),
    )

    summary_all = raw.loc[raw["estimate_available"]].groupby(["target_snr", "setting", "method"], as_index=False).agg(
        mse_null=("mse_null", "mean"),
        mse_signal=("mse_signal", "mean"),
        mse_overall=("mse_overall", "mean"),
        avg_ci_length=("avg_ci_length", "mean"),
        coverage_95=("coverage_95", "mean"),
        n_effective=("converged", "sum"),
        n_estimate_available=("replicate_id", "count"),
    )
    summary_all = summary_all.merge(run_counts, on=["target_snr", "setting", "method"], how="left")
    summary_all["valid_rate"] = summary_all["n_effective"] / summary_all["n_total_runs"].clip(lower=1)

    summary_conv = raw.loc[raw["estimate_available"] & raw["converged"]].groupby(["target_snr", "setting", "method"], as_index=False).agg(
        mse_null=("mse_null", "mean"),
        mse_signal=("mse_signal", "mean"),
        mse_overall=("mse_overall", "mean"),
        avg_ci_length=("avg_ci_length", "mean"),
        coverage_95=("coverage_95", "mean"),
        n_effective=("converged", "sum"),
        n_estimate_available=("replicate_id", "count"),
    )
    summary_conv = summary_conv.merge(run_counts, on=["target_snr", "setting", "method"], how="left")
    summary_conv["valid_rate"] = summary_conv["n_effective"] / summary_conv["n_total_runs"].clip(lower=1)

    paired_raw, paired_stats = _paired_converged_subset(
        raw,
        group_cols=["target_snr", "setting"],
        method_col="method",
        replicate_col="replicate_id",
        converged_col="converged",
        required_cols=["mse_null", "mse_signal", "mse_overall", "avg_ci_length", "coverage_95"],
        method_levels=methods_use,
    )
    summary_paired = paired_raw.groupby(["target_snr", "setting", "method"], as_index=False).agg(
        mse_null=("mse_null", "mean"),
        mse_signal=("mse_signal", "mean"),
        mse_overall=("mse_overall", "mean"),
        avg_ci_length=("avg_ci_length", "mean"),
        coverage_95=("coverage_95", "mean"),
        n_effective=("converged", "sum"),
        n_paired=("replicate_id", "nunique"),
    )
    if not summary_paired.empty:
        summary_paired = summary_paired.merge(paired_stats, on=["target_snr", "setting"], how="left")
        summary_paired["paired_rate"] = summary_paired["n_paired"] / summary_paired["n_total_replicates"].clip(lower=1)

    summary = summary_paired if not summary_paired.empty else (summary_conv if not summary_conv.empty else summary_all)
    snr_ref = 0.70 if any(abs(v - 0.70) < 1e-12 for v in snr_vals) else snr_vals[0]
    summary_plot = summary.loc[np.isclose(summary["target_snr"], snr_ref)].copy()
    if summary_plot.empty:
        summary_plot = summary_conv.loc[np.isclose(summary_conv["target_snr"], snr_ref)].copy()
    if summary_plot.empty:
        summary_plot = summary_all.loc[np.isclose(summary_all["target_snr"], snr_ref)].copy()
    save_dataframe(raw, out / "raw_results.csv")
    save_dataframe(summary, out / "summary.csv")
    save_dataframe(summary_all, out / "summary_all.csv")
    save_dataframe(summary_conv, out / "summary_converged.csv")
    save_dataframe(summary_paired, out / "summary_paired_converged.csv")
    save_dataframe(convergence_audit, out / "convergence_audit.csv")
    save_dataframe(paired_stats, out / "paired_convergence_audit.csv")
    save_dataframe(summary, tab_dir / "table_benchmark_linear.csv")
    save_dataframe(summary_all, tab_dir / "table_benchmark_linear_all.csv")
    save_dataframe(summary_conv, tab_dir / "table_benchmark_linear_converged.csv")
    save_dataframe(summary_paired, tab_dir / "table_benchmark_linear_paired_converged.csv")
    plot_exp4_overall_mse(summary_plot, out_path=fig_dir / "fig4_benchmark_overall_mse.png")
    plot_exp4_mse_partition(summary_plot, out_path=fig_dir / "fig4_benchmark_mse_partition.png")
    design_rows: list[dict[str, Any]] = []
    for setting, spec in settings.items():
        groups = canonical_groups(spec["group_sizes"])
        beta_setting = str(spec.get("beta_setting", setting))
        beta0 = build_linear_beta(beta_setting, spec["group_sizes"])
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
                "beta_setting": beta_setting,
                "active_groups_1_based": active_groups,
                "group_nonzero_counts": group_nonzero_counts,
                "nonzero_total": int(np.sum(np.abs(beta0) > 0.0)),
            }
        )
    save_json(
        {
            "settings": design_rows,
            "repeats": int(repeats),
            "snr_list": snr_vals,
            "plot_snr_reference": float(snr_ref),
            "profile": profile_name,
            "methods": methods_use,
            "gigg_config": gigg_cfg,
            "enforce_bayes_convergence": bool(enforce_bayes_convergence),
            "max_convergence_retries": int(retry_limit),
            "until_bayes_converged": bool(until_bayes_converged),
            "until_converged_retry_hard_cap": int(_UNTIL_CONVERGED_RETRY_HARD_CAP),
            "comparison_table_policy": "paired_converged_if_available",
        },
        out / "exp4_design_meta.json",
    )
    log.info("Completed exp4 with repeats=%d, snr_list=%s, profile=%s", repeats, snr_vals, profile_name)
    return {"raw": str(out / "raw_results.csv"), "table": str(tab_dir / "table_benchmark_linear.csv")}


def run_exp5_heterogeneity(
    n_jobs: int = 1,
    seed: int = MASTER_SEED,
    repeats: int = 100,
    save_dir: str = "simulation_project",
    *,
    profile: str = "full",
    methods: Sequence[str] | None = None,
    enforce_bayes_convergence: bool = True,
    max_convergence_retries: int | None = None,
    until_bayes_converged: bool = True,
) -> Dict[str, str]:
    # ============================================================
    # EXP5 鈥?寮傝川鎬х粍缁撴瀯涓嬬殑鍒嗙粍杈ㄨ瘑锛堣繛鎺ュ畾鐞?3.34 鐨?simultaneous separation锛?
    #
    # 銆愭暟鎹敓鎴愯璁¤鏄庛€?
    #   group_sizes = [50, 50, 20, 10, 10, 10]锛宻igma2=1.0锛宼au_ref=0.1
    #   mu = [0, 0,  mu_boundary,  2,  8,  25]
    #         闆? 闆? 闃堝€奸檮杩?      寮? 涓? 寮?
    #   mu_boundary = 1.2 脳 尉_crit(0.5, 0.1) 脳 20  锛堟瘮鐩稿彉闃堝€奸珮 20%锛?
    #   鈫?Group 1,2 (p_g=50)锛氬簲琚己鍔涙敹缂╋紙魏鈮?锛?
    #   鈫?Group 3   (p_g=20)锛氫俊鍙疯交寰秴杩囬槇鍊硷紝魏 搴斿浜庤繃娓″尯锛?.3~0.7锛?
    #   鈫?Group 4,5,6 (p_g=10)锛氬己淇″彿锛屛衡増1
    #
    # 銆愮悊鎯崇粨鏋溿€?
    #   GR_RHS AUROC > 0.90锛堟樉钁椾紭浜?RHS 鐨?~0.60~0.70锛?
    #   闆剁粍 魏 鍧囧€?< 0.05锛屽己淇″彿缁?魏 鍧囧€?> 0.80
    #   Group 3 鐨?魏 鍒嗗竷瀹戒笖灞呬腑锛堜綋鐜扮浉鍙樿竟鐣岀殑涓嶇‘瀹氭€э級
    #
    # 銆愯嫢杈ㄨ瘑澶辫触锛圓UROC 浣庯級鐨勮皟鏁存柟鍚戙€?
    #   鐥囩姸 A 鈥?GR_RHS AUROC 鈮?RHS锛堢粍缁撴瀯鏈鍒╃敤锛夛細
    #     鈫?妫€鏌?use_group_scale=True 鍦?fit_gr_rhs 璋冪敤涓槸鍚︽湁鏁?
    #     鈫?tau_ref=0.1 浣垮緱 尉_crit 鏋佸皬锛孏roup 3 鐨勪俊鍙峰凡瓒呴槇鍊艰緝澶氾紱
    #       鑻ヤ粛涓嶆敹鏁涳紝灏濊瘯 tau_ref=0.3 浣块槇鍊兼洿澶э紝鍒嗙鏇撮毦
    #
    #   鐥囩姸 B 鈥?Group 3 鐨?魏 鏃犲樊寮傦紙瑕佷箞鍏?0 瑕佷箞鍏?1锛夛細
    #     鈫?mu_boundary 鍊嶆暟浠?1.2 璋冩暣涓?1.0锛堟伆濂藉湪闃堝€间笂锛?
    #     鈫?澧炲ぇ p_g=20 缁勫埌 30 鎴?50 浠ヤ娇鐩稿彉鏇撮攼鍒?
    #
    #   鐥囩姸 C 鈥?澶ч浂缁勶紙p_g=50锛夌殑 魏 涓嶈秼鍚?0锛?
    #     鈫?涓?EXP1 涓€鑷寸殑闂锛歱_g=50 鍦?tau_ref=0.1 涓嬪彲鑳戒粛鏈厖鍒嗘敹缂?
    #     鈫?澧炲ぇ n锛堝綋鍓?n=300锛夛紝浣挎暟鎹鍏堥獙鐨勬敮閰嶅姏鏇村己
    # ============================================================
    import pandas as pd
    from .plotting import plot_exp5_group_ranking, plot_exp5_kappa_stratification, plot_exp5_null_signal_mse
    base = Path(save_dir)
    out = ensure_dir(base / "results" / "exp5_heterogeneity")
    fig_dir = ensure_dir(base / "figures")
    tab_dir = ensure_dir(base / "tables")
    log = setup_logger("exp5", base / "logs" / "exp5_heterogeneity.log")
    profile_name = _normalize_compute_profile(profile)
    sampler = _sampler_for_profile(profile_name, experiment="exp5")
    methods_use = _resolve_method_list(methods, profile=profile_name)
    gigg_cfg = _gigg_config_for_profile(profile_name)
    retry_limit = _resolve_convergence_retry_limit(
        profile_name,
        max_convergence_retries,
        until_bayes_converged=bool(until_bayes_converged),
    )
    sigma2 = 1.0
    tau_ref = 0.1
    group_sizes = [50, 50, 20, 10, 10, 10]
    xi_boundary = xi_crit_u0_rho(u0=0.5, rho=tau_ref / math.sqrt(sigma2))
    mu_boundary = 1.2 * xi_boundary * group_sizes[2]
    mu = [0.0, 0.0, float(mu_boundary), 2.0, 8.0, 25.0]
    labels = (np.asarray(mu) > 0.0).astype(int)
    tasks = [
        (r, seed, group_sizes, mu, sampler, methods_use, gigg_cfg, bool(enforce_bayes_convergence), int(retry_limit))
        for r in range(1, int(repeats) + 1)
    ]

    row_rep, row_group, row_grrhs_kappa = [], [], []
    for rep_rows, group_rows, kappa_rows in _parallel_rows(tasks, _exp5_worker, n_jobs=n_jobs, prefer_process=True, progress_desc="Exp5 Heterogeneity"):
        row_rep.extend(rep_rows)
        row_group.extend(group_rows)
        row_grrhs_kappa.extend(kappa_rows)
    raw_rep = pd.DataFrame(row_rep)
    raw_group = pd.DataFrame(row_group)
    raw_kappa = pd.DataFrame(row_grrhs_kappa)
    raw = raw_rep.merge(raw_group, on=["replicate_id", "method"], how="left")
    run_counts = raw_rep.groupby("method", as_index=False).agg(n_total_runs=("replicate_id", "count"))
    convergence_audit = raw_rep.groupby("method", as_index=False).agg(
        n_total_runs=("replicate_id", "count"),
        n_converged=("converged", "sum"),
        convergence_rate=("converged", "mean"),
        fit_attempts_mean=("fit_attempts", "mean"),
        fit_attempts_max=("fit_attempts", "max"),
    )
    auroc_table = raw_rep.groupby("method", as_index=False).agg(
        group_auroc=("group_auroc_score", "mean"),
        avg_null_group_mse=("null_group_mse_avg", "mean"),
        avg_signal_group_mse=("signal_group_mse_avg", "mean"),
        n_effective=("converged", "sum"),
        n_valid_metrics=("group_auroc_score", lambda s: int(s.notna().sum())),
    )
    auroc_table = auroc_table.merge(run_counts, on="method", how="left")
    auroc_table["valid_rate"] = auroc_table["n_effective"] / auroc_table["n_total_runs"].clip(lower=1)

    paired_rep, paired_stats = _paired_converged_subset(
        raw_rep,
        group_cols=[],
        method_col="method",
        replicate_col="replicate_id",
        converged_col="converged",
        required_cols=["group_auroc_score", "null_group_mse_avg", "signal_group_mse_avg", "overall_mse"],
        method_levels=methods_use,
    )
    auroc_table_paired = paired_rep.groupby("method", as_index=False).agg(
        group_auroc=("group_auroc_score", "mean"),
        avg_null_group_mse=("null_group_mse_avg", "mean"),
        avg_signal_group_mse=("signal_group_mse_avg", "mean"),
        n_effective=("converged", "sum"),
        n_paired=("replicate_id", "nunique"),
    )
    if not auroc_table_paired.empty:
        auroc_table_paired = auroc_table_paired.merge(run_counts, on="method", how="left")
        if not paired_stats.empty:
            stats_row = paired_stats.iloc[0].to_dict()
            for k, v in stats_row.items():
                auroc_table_paired[k] = v
        auroc_table_paired["paired_rate"] = auroc_table_paired["n_paired"] / auroc_table_paired["n_total_runs"].clip(lower=1)

    table_main = auroc_table_paired if not auroc_table_paired.empty else auroc_table
    save_dataframe(raw, out / "raw_results.csv")
    save_dataframe(raw_rep, out / "summary_replicate.csv")
    save_dataframe(paired_rep, out / "summary_replicate_paired_converged.csv")
    save_dataframe(raw_kappa, out / "summary_kappa.csv")
    save_dataframe(auroc_table, out / "summary_all.csv")
    save_dataframe(auroc_table_paired, out / "summary_paired_converged.csv")
    save_dataframe(convergence_audit, out / "convergence_audit.csv")
    save_dataframe(paired_stats, out / "paired_convergence_audit.csv")
    save_dataframe(table_main, tab_dir / "table_heterogeneity_auroc.csv")
    save_dataframe(auroc_table, tab_dir / "table_heterogeneity_auroc_all.csv")
    save_dataframe(auroc_table_paired, tab_dir / "table_heterogeneity_auroc_paired_converged.csv")
    save_json(
        {
            "group_sizes": [int(v) for v in group_sizes],
            "mu": [float(v) for v in mu],
            "tau_ref": float(tau_ref),
            "xi_boundary": float(xi_boundary),
            "mu_boundary": float(mu_boundary),
            "boundary_multiplier_vs_xi_crit": 1.2,
            "repeats": int(repeats),
            "profile": profile_name,
            "methods": methods_use,
            "gigg_config": gigg_cfg,
            "enforce_bayes_convergence": bool(enforce_bayes_convergence),
            "max_convergence_retries": int(retry_limit),
            "until_bayes_converged": bool(until_bayes_converged),
            "until_converged_retry_hard_cap": int(_UNTIL_CONVERGED_RETRY_HARD_CAP),
            "comparison_table_policy": "paired_converged_if_available",
        },
        out / "exp5_meta.json",
    )
    plot_methods = [m for m in METHODS if m in set(methods_use)]
    if not plot_methods:
        plot_methods = sorted(set(table_main["method"].astype(str).tolist()))
    if not raw_kappa.empty:
        plot_exp5_kappa_stratification(raw_kappa, out_path=fig_dir / "fig5_kappa_stratification.png")
        plot_exp5_group_ranking(raw_kappa, table_main.loc[table_main["method"].isin(plot_methods)], out_path=fig_dir / "fig5_group_ranking.png")
    plot_exp5_null_signal_mse(table_main.loc[table_main["method"].isin(plot_methods)], out_path=fig_dir / "fig5_null_signal_mse.png")
    log.info("Completed exp5 with repeats=%d, profile=%s", repeats, profile_name)
    return {"raw": str(out / "raw_results.csv"), "table": str(tab_dir / "table_heterogeneity_auroc.csv")}


def run_exp6_grouped_logistic(
    n_jobs: int = 1,
    seed: int = MASTER_SEED,
    repeats: int = 50,
    save_dir: str = "simulation_project",
    *,
    min_separator_auc: float = 0.8,
    profile: str = "full",
    methods: Sequence[str] | None = None,
    enforce_bayes_convergence: bool = True,
    max_convergence_retries: int | None = None,
    until_bayes_converged: bool = True,
) -> Dict[str, str]:
    # ============================================================
    # EXP6 鈥?鍒嗙粍 Logistic 鍥炲綊锛堜簩鍏冪粨鏋滅殑閫傜敤鎬ч獙璇侊級
    #
    # 銆愭暟鎹敓鎴愯璁¤鏄庛€?
    #   3 缁勫悇 5 涓娴嬪彉閲忥紙p_g=5锛屾敞鎰?p_g 寰堝皬锛屼笉澶勪簬鐞嗚娓愯繎鍖哄煙锛?
    #   beta0 = [1.5,1.5,0,0,0 | 0,0,0,0,0 | 0.5,0.5,0,0,0]
    #            寮轰俊鍙风粍          闆剁粍          寮变俊鍙风粍
    #   鈫? Group 1: ||尾||虏=4.5锛堝己锛夛紝Group 2: ||尾||虏=0锛堥浂锛夛紝Group 3: ||尾||虏=0.5锛堝急锛?
    #   n=200锛宮in_separator_auc=0.8 杩囨护鏋侀毦鏍锋湰
    #
    # 銆愮悊鎯崇粨鏋溿€?
    #   P(魏鈧?> 0.5) 鈮?1.0锛堝己淇″彿缁勬槑纭娴嬶級
    #   P(魏鈧?> 0.5) 鈮?0.0锛堥浂缁勬槑纭姂鍒讹級
    #   P(魏鈧?> 0.5) 鈭?(0.3, 0.7)锛堝急淇″彿缁勪笉纭畾锛岃涓烘垚鍔熷睍绀虹伒鏁忓害锛?
    #   尾_{11}, 尾_{12} 鍚庨獙鍧囧€兼帴杩?1.5
    #
    # 銆愯嫢 魏 鍒嗙粍鏃犳硶鍖哄垎鐨勮皟鏁存柟鍚戙€?
    #   鐥囩姸 A 鈥?P(魏鈧?> 0.5) 鍋忎綆锛堝己淇″彿涔熸湭琚娴嬶級锛?
    #     鈫?妫€鏌?logistic 鍥炲綊鐨?MCMC 鏀舵暃锛坉ivergence_ratio < 0.01锛?
    #     鈫?n=200 鍙兘涓嶈冻浠ユ敮鎸?p_g=5 鐨勪俊鍙凤紱灏濊瘯 n=500
    #     鈫?beta0 涓殑 1.5 瀵?logistic 灏哄害鏄惁鍚堥€傦紙log-odds 涓?1.5锛孉UC 绾?0.85锛?
    #
    #   鐥囩姸 B 鈥?Group 3锛堝急淇″彿锛変笌 Group 2锛堥浂缁勶級鐨?魏 瀹屽叏鐩稿悓锛?
    #     鈫?寮变俊鍙?||尾||虏=0.5 鍦?n=200 涓嬪彲鑳戒笌闆剁粍鏃犳硶鍖哄垎锛堣繖鏈韩鏄悎鐞嗙幇璞★級
    #     鈫?鑻ュ笇鏈涘尯鍒嗭紝澧炲ぇ beta0[10:12] 鍒?[0.8, 0.8] 鎴栧澶?n
    #
    #   鐥囩姸 C 鈥?鏀舵暃鐜囷紙n_effective锛変綆锛? 0.8锛夛細
    #     鈫?Logistic 妯″瀷瀹规槗鍑虹幇鍙戞暎锛涗娇鐢?SamplerConfig(adapt_delta=0.95)
    #     鈫?闄嶄綆 min_separator_auc 闃堝€硷紙杩囨护澶弗瀵艰嚧鏍锋湰閲忎笉瓒筹級
    # ============================================================
    import pandas as pd
    from .plotting import plot_exp6_coefficients, plot_exp6_diagnostics, plot_exp6_kappa, plot_exp6_null_group
    base = Path(save_dir)
    out = ensure_dir(base / "results" / "exp6_grouped_logistic")
    fig_dir = ensure_dir(base / "figures")
    log = setup_logger("exp6", base / "logs" / "exp6_grouped_logistic.log")
    profile_name = _normalize_compute_profile(profile)
    sampler = _sampler_for_profile(profile_name, experiment="exp6")
    methods_use = _resolve_method_list(methods, profile=profile_name)
    gigg_cfg = _gigg_config_for_profile(profile_name)
    retry_limit = _resolve_convergence_retry_limit(
        profile_name,
        max_convergence_retries,
        until_bayes_converged=bool(until_bayes_converged),
    )
    beta0 = np.array([1.5, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0], dtype=float)
    tasks = [
        (
            r,
            seed,
            beta0,
            sampler,
            float(min_separator_auc),
            methods_use,
            gigg_cfg,
            bool(enforce_bayes_convergence),
            int(retry_limit),
        )
        for r in range(1, int(repeats) + 1)
    ]

    rows = []
    for chunk in _parallel_rows(tasks, _exp6_worker, n_jobs=n_jobs, prefer_process=True, progress_desc="Exp6 Grouped Logistic"):
        rows.extend(chunk)
    raw = pd.DataFrame(rows)
    save_dataframe(raw, out / "raw_results.csv")
    run_counts = raw.groupby("method", as_index=False).agg(n_total_runs=("replicate_id", "count"))
    convergence_audit = raw.groupby("method", as_index=False).agg(
        n_total_runs=("replicate_id", "count"),
        n_converged=("converged", "sum"),
        convergence_rate=("converged", "mean"),
        fit_attempts_mean=("fit_attempts", "mean"),
        fit_attempts_max=("fit_attempts", "max"),
    )
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
        n_valid_metrics=("beta_group1_l2_norm", lambda s: int(s.notna().sum())),
    )
    summary = summary.merge(run_counts, on="method", how="left")
    summary["valid_rate"] = summary["n_effective"] / summary["n_total_runs"].clip(lower=1)

    paired_raw, paired_stats = _paired_converged_subset(
        raw,
        group_cols=[],
        method_col="method",
        replicate_col="replicate_id",
        converged_col="converged",
        required_cols=[
            "beta11_post_mean",
            "beta12_post_mean",
            "beta_group1_l2_norm",
            "beta_group2_l2_norm",
            "beta_group3_l2_norm",
            "overall_runtime",
        ],
        method_levels=methods_use,
    )
    summary_paired = paired_raw.groupby("method", as_index=False).agg(
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
        n_paired=("replicate_id", "nunique"),
    )
    if not summary_paired.empty:
        summary_paired = summary_paired.merge(run_counts, on="method", how="left")
        if not paired_stats.empty:
            stats_row = paired_stats.iloc[0].to_dict()
            for k, v in stats_row.items():
                summary_paired[k] = v
        summary_paired["paired_rate"] = summary_paired["n_paired"] / summary_paired["n_total_runs"].clip(lower=1)

    summary_main = summary_paired if not summary_paired.empty else summary
    save_dataframe(summary_main, out / "summary.csv")
    save_dataframe(summary, out / "summary_all.csv")
    save_dataframe(summary_paired, out / "summary_paired_converged.csv")
    save_dataframe(convergence_audit, out / "convergence_audit.csv")
    save_dataframe(paired_stats, out / "paired_convergence_audit.csv")
    ok = raw.loc[raw["converged"] == True].copy()
    if not ok.empty:
        plot_exp6_coefficients(ok, out_path=fig_dir / "fig6_logistic_coefficients.png")
        plot_exp6_null_group(ok, out_path=fig_dir / "fig6_logistic_null_group.png")
        plot_exp6_diagnostics(ok, out_path=fig_dir / "fig6_logistic_diagnostics.png")
        plot_exp6_kappa(ok, out_path=fig_dir / "fig6_kappa_logistic.png")
    save_json(
        {
            "repeats": int(repeats),
            "min_separator_auc": float(min_separator_auc),
            "profile": profile_name,
            "methods": methods_use,
            "gigg_config": gigg_cfg,
            "enforce_bayes_convergence": bool(enforce_bayes_convergence),
            "max_convergence_retries": int(retry_limit),
            "until_bayes_converged": bool(until_bayes_converged),
            "until_converged_retry_hard_cap": int(_UNTIL_CONVERGED_RETRY_HARD_CAP),
            "comparison_table_policy": "paired_converged_if_available",
        },
        out / "exp6_meta.json",
    )
    log.info("Completed exp6 with repeats=%d, min_separator_auc=%.3f, profile=%s", repeats, float(min_separator_auc), profile_name)
    return {"raw": str(out / "raw_results.csv")}


def run_exp7_ablation(
    n_jobs: int = 1,
    seed: int = MASTER_SEED,
    repeats: int = 100,
    save_dir: str = "simulation_project",
    *,
    profile: str = "full",
    enforce_bayes_convergence: bool = True,
    max_convergence_retries: int | None = None,
    until_bayes_converged: bool = True,
) -> Dict[str, str]:
    # ============================================================
    # EXP7 鈥?娑堣瀺鐮旂┒锛堢粍浠朵环鍊煎垎鏋愶級
    #
    # 銆愭秷铻嶅彉浣撹鏄庛€?
    #   GR_RHS_full         锛氬畬鏁存ā鍨嬶紙鍩哄噯锛屽簲鏈€浼橈級
    #   GR_RHS_no_ag        锛氬幓闄ょ粍灏哄害 a_g锛堟棤缁勯棿寮傝川鎬ф牎姝ｏ級
    #   GR_RHS_no_local_scales锛氬幓闄ょ粍鍐呭眬閮ㄥ昂搴?位_j锛堝叏缁勫潎鍖€鏀剁缉锛?
    #   GR_RHS_shared_kappa 锛氭墍鏈夌粍鍏变韩涓€涓?魏锛堟棤缁勭壒寮傛€ф敹缂╋級
    #   GR_RHS_no_kappa     锛氶€€鍖栦负鏍囧噯 RHS锛堟棤 魏 鏈哄埗锛?
    #   RHS                 锛氱函 horseshoe 鍩哄噯
    #
    # 銆怐GP 绫诲瀷涓庨鏈熷樊寮傘€?
    #   dense_uniform锛氱粍鍐呮墍鏈夊彉閲忓潎鏈変俊鍙?
    #     鈫?no_local_scales 褰卞搷搴旇緝灏忥紙缁勫唴鍚岃川锛?
    #     鈫?no_ag 褰卞搷搴旇緝澶э紙缁勯棿寮哄害宸紓闇€瑕?a_g 鎹曡幏锛?
    #   sparse_within_group锛氱粍鍐呬粎 20% 鍙橀噺鏈変俊鍙?
    #     鈫?no_local_scales 搴旀樉钁楀彉宸紙位_j 鏄粍鍐呯█鐤忔€ф娴嬬殑鍏抽敭锛?
    #     鈫?GR_RHS_full 浼樺娍浣撶幇鍦?mse_signal 涓婏紙绮惧噯瀹氫綅娲昏穬鍙橀噺锛?
    #
    # 銆愮悊鎯崇粨鏋溿€?
    #   GR_RHS_full 鍦ㄤ袱绉?DGP 涓嬮兘鍏锋湁鏈€浣?null_group_mse 鍜屾渶楂?AUROC
    #   GR_RHS_no_local_scales 鍦?sparse_within_group 涓?signal_group_mse 鏄庢樉鏇撮珮
    #   GR_RHS_shared_kappa 鐨?AUROC < GR_RHS_full锛堝け鍘荤粍鐗瑰紓鎬э級
    #
    # 銆愯嫢娑堣瀺宸紓涓嶆樉钁楃殑璋冩暣鏂瑰悜銆?
    #   鐥囩姸 A 鈥?鎵€鏈夊彉浣撴€ц兘鐩歌繎锛堟秷铻嶆棤鏁堟灉锛夛細
    #     鈫?mu 涓急淇″彿缁勶紙mu=2锛夊彲鑳藉凡瓒呭嚭 尉_crit锛屼俊鍙疯繃寮哄鑷存墍鏈夋柟娉曢兘鑳芥娴?
    #     鈫?灏?mu[2] 闄嶄綆鑷虫帴杩戠浉鍙橀槇鍊硷紙瑙?EXP5 鐨?mu_boundary 璁＄畻鏂规硶锛?
    #     鈫?rho_within=0.7 涓嬬粍鍐呭彉閲忛珮搴︾浉鍏筹紝灞€閮ㄥ昂搴︿綔鐢ㄥ彲鑳借鐩稿叧鎬ф帺鐩?
    #
    #   鐥囩姸 B 鈥?GR_RHS_no_ag 涓?GR_RHS_full 鏃犲樊寮傦細
    #     鈫?缁勯棿淇″彿寮哄害宸紓涓嶅澶э紙mu=[0,0,2,8,25,80] 鐩稿樊鎮畩锛屼絾 a_g 鐨勪綔鐢?
    #       浣撶幇鍦ㄤ腑绛夊樊寮傚満鏅級锛岃€冭檻鏀逛负 [0,0,5,10,20,40]
    #
    #   鐥囩姸 C 鈥?repeats=100 涓嬬疆淇″尯闂磋繃瀹斤細
    #     鈫?澧炲姞鍒?repeats=200锛屾垨浣跨敤 paired t-test 鑰岄潪鍧囧€兼瘮杈?
    # ============================================================
    import pandas as pd
    from .plotting import plot_exp7_ablation_bars
    base = Path(save_dir)
    out = ensure_dir(base / "results" / "exp7_ablation")
    fig_dir = ensure_dir(base / "figures")
    tab_dir = ensure_dir(base / "tables")
    log = setup_logger("exp7", base / "logs" / "exp7_ablation.log")
    profile_name = _normalize_compute_profile(profile)
    sampler = _sampler_for_profile(profile_name, experiment="exp7")
    retry_limit = _resolve_convergence_retry_limit(
        profile_name,
        max_convergence_retries,
        until_bayes_converged=bool(until_bayes_converged),
    )
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
        (
            r,
            seed,
            [10, 10, 10, 10, 10, 10],
            mu,
            dgp_type,
            variants,
            sampler,
            bool(enforce_bayes_convergence),
            int(retry_limit),
        )
        for dgp_type in dgp_types
        for r in range(1, int(repeats) + 1)
    ]

    rows = []
    for chunk in _parallel_rows(tasks, _exp7_worker, n_jobs=n_jobs, prefer_process=True, progress_desc="Exp7 Ablation"):
        rows.extend(chunk)
    raw = pd.DataFrame(rows)
    run_counts = raw.groupby(["dgp_type", "variant"], as_index=False).agg(n_total_runs=("replicate_id", "count"))
    convergence_audit = raw.groupby(["dgp_type", "variant"], as_index=False).agg(
        n_total_runs=("replicate_id", "count"),
        n_converged=("converged", "sum"),
        convergence_rate=("converged", "mean"),
        fit_attempts_mean=("fit_attempts", "mean"),
        fit_attempts_max=("fit_attempts", "max"),
    )
    table = raw.groupby(["dgp_type", "variant"], as_index=False).agg(
        null_group_mse_avg=("null_group_mse_avg", "mean"),
        signal_group_mse_avg=("signal_group_mse_avg", "mean"),
        overall_mse=("overall_mse", "mean"),
        group_auroc=("group_auroc", "mean"),
        n_effective=("converged", "sum"),
        n_valid_metrics=("group_auroc", lambda s: int(s.notna().sum())),
    )
    table = table.merge(run_counts, on=["dgp_type", "variant"], how="left")
    table["valid_rate"] = table["n_effective"] / table["n_total_runs"].clip(lower=1)

    paired_raw, paired_stats = _paired_converged_subset(
        raw,
        group_cols=["dgp_type"],
        method_col="variant",
        replicate_col="replicate_id",
        converged_col="converged",
        required_cols=["null_group_mse_avg", "signal_group_mse_avg", "overall_mse", "group_auroc"],
        method_levels=list(variants.keys()),
    )
    table_paired = paired_raw.groupby(["dgp_type", "variant"], as_index=False).agg(
        null_group_mse_avg=("null_group_mse_avg", "mean"),
        signal_group_mse_avg=("signal_group_mse_avg", "mean"),
        overall_mse=("overall_mse", "mean"),
        group_auroc=("group_auroc", "mean"),
        n_effective=("converged", "sum"),
        n_paired=("replicate_id", "nunique"),
    )
    if not table_paired.empty:
        table_paired = table_paired.merge(run_counts, on=["dgp_type", "variant"], how="left")
        table_paired = table_paired.merge(paired_stats, on=["dgp_type"], how="left")
        table_paired["paired_rate"] = table_paired["n_paired"] / table_paired["n_total_runs"].clip(lower=1)

    table_main = table_paired if not table_paired.empty else table
    save_dataframe(raw, out / "raw_results.csv")
    save_dataframe(table_main, tab_dir / "table_ablation.csv")
    save_dataframe(table_main, out / "summary.csv")
    save_dataframe(table, out / "summary_all.csv")
    save_dataframe(table_paired, out / "summary_paired_converged.csv")
    save_dataframe(table, tab_dir / "table_ablation_all.csv")
    save_dataframe(table_paired, tab_dir / "table_ablation_paired_converged.csv")
    save_dataframe(convergence_audit, out / "convergence_audit.csv")
    save_dataframe(paired_stats, out / "paired_convergence_audit.csv")
    plot_exp7_ablation_bars(table_main, out_path=fig_dir / "fig7_ablation_metrics.png")
    save_json(
        {
            "mu": [float(v) for v in mu],
            "dgp_types": dgp_types,
            "variants": list(variants.keys()),
            "repeats": int(repeats),
            "sparse_within_group_rho_within": 0.3,
            "profile": profile_name,
            "enforce_bayes_convergence": bool(enforce_bayes_convergence),
            "max_convergence_retries": int(retry_limit),
            "until_bayes_converged": bool(until_bayes_converged),
            "until_converged_retry_hard_cap": int(_UNTIL_CONVERGED_RETRY_HARD_CAP),
            "comparison_table_policy": "paired_converged_if_available",
        },
        out / "exp7_meta.json",
    )
    log.info("Completed exp7 with repeats=%d, profile=%s", repeats, profile_name)
    return {"raw": str(out / "raw_results.csv"), "table": str(tab_dir / "table_ablation.csv")}


def run_exp8_tau_calibration(
    n_jobs: int = 1,
    seed: int = MASTER_SEED,
    repeats: int = 100,
    save_dir: str = "simulation_project",
    *,
    profile: str = "full",
    enforce_bayes_convergence: bool = True,
    max_convergence_retries: int | None = None,
    until_bayes_converged: bool = True,
) -> Dict[str, str]:
    # ============================================================
    # EXP8 鈥?蟿 鑷姩鏍″噯楠岃瘉
    #
    # 銆愮悊璁哄熀鍑嗐€?
    #   Carvalho-Polson-Scott 鎺ㄨ崘锛毾刜target = p鈧€ / ((p鈭抪鈧€)鈭歯)
    #   鑷姩鏍″噯鍚庨獙鍧囧€煎簲鎺ヨ繎 蟿_target锛坱au_rel_error < 0.20 瑙嗕负鎴愬姛锛?
    #
    # 銆愬弬鏁伴€夋嫨璇存槑銆?
    #   p0_list=[2,6,12,30]锛氳鐩栫█鐤忥紙p鈧€/p=3%锛夊埌绋犲瘑锛坧鈧€/p=50%锛?
    #   tau_scales=[0.5,1.0,2.0]锛氬浐瀹?蟿 鍒嗗埆浣跨敤 0.5x/1x/2x 蟿_target
    #   SamplerConfig(chains=4, warmup=600)锛毾?鏄叏灞€鍙傛暟锛屾贩鍚堣緝鎱紝闇€瑕佽冻澶熼鐑?
    #
    # 銆愯嫢鑷姩鏍″噯鏈兘杈惧埌 tau_target 鐨勮皟鏁存柟鍚戙€?
    #   鐥囩姸 A 鈥?tau_post_mean 绯荤粺浣庝簬 tau_target锛堣嚜鍔ㄦ牎鍑嗘瑺浼拌锛夛細
    #     鈫?妫€鏌?auto_calibrate_tau 鐨勫疄鐜帮細鏄惁鐢ㄤ簡姝ｇ‘鐨?p鈧€ 浣滀负鍏堥獙淇℃伅
    #     鈫?tau_target 鏈韩璁＄畻锛歱0/((p-p0)*sqrt(n))锛岀‘璁?p鈧€ 鏄椿璺冨彉閲忔暟鑰岄潪缁勬暟
    #     鈫?灏濊瘯 tau_prior_scale=2.0 缁欒嚜鍔ㄦ牎鍑嗘洿瀹界殑鎼滅储鑼冨洿
    #
    #   鐥囩姸 B 鈥?tau_post_sd 寰堝ぇ锛堝悗楠屽彂鏁ｏ級锛?
    #     鈫?chains=4 浣?warmup=600 鍙兘瀵?tau 涓嶈冻锛涘鍔犲埌 warmup=1000
    #     鈫?澶?p鈧€锛堝 p鈧€=30/p=60锛夋椂鍏堥獙淇℃伅寮憋紝tau 鍚庨獙澶╃劧鏇村锛屽睘姝ｅ父鐜拌薄
    #
    #   鐥囩姸 C 鈥?Fixed 1x tau 涓?auto 琛ㄧ幇鐩歌繎浣?0.5x/2x 宸紓涓嶆槑鏄撅細
    #     鈫?beta0 鍏ㄩ儴璁句负 2.0锛堟亽绛変俊鍙凤級锛屼俊鍙疯繃寮哄鑷?tau 鍙樺寲瀵逛及璁″奖鍝嶅皬
    #     鈫?鑰冭檻娣峰悎淇″彿寮哄害锛堥儴鍒?beta=0.5锛岄儴鍒?beta=2.0锛変互澧炲己 tau 鐨勫奖鍝嶅姏
    #
    #   鐥囩姸 D 鈥?n_effective锛堟敹鏁涚巼锛夊亸浣庯細
    #     鈫?Logistic 鐗堟湰鎴栭珮鐩稿叧璁捐浼氬鑷?tau 娣峰悎鍥伴毦
    #     鈫?纭 use_auto=True 鏃?tau0=None锛堜笉鑳藉悓鏃舵寚瀹氳捣濮嬪€硷級
    # ============================================================
    import pandas as pd
    from .plotting import plot_exp8_tau
    base = Path(save_dir)
    out = ensure_dir(base / "results" / "exp8_tau_calibration")
    fig_dir = ensure_dir(base / "figures")
    log = setup_logger("exp8", base / "logs" / "exp8_tau_calibration.log")
    profile_name = _normalize_compute_profile(profile)
    sampler = _sampler_for_profile(profile_name, experiment="exp8")
    retry_limit = _resolve_convergence_retry_limit(
        profile_name,
        max_convergence_retries,
        until_bayes_converged=bool(until_bayes_converged),
    )

    n = 500
    group_sizes = [10, 10, 10, 10, 10, 10]
    p = int(sum(group_sizes))
    p0_list = [2, 6, 12, 30]
    tau_scales = [0.5, 1.0, 2.0]
    tasks: list[tuple[int, int, int, list[int], SamplerConfig, float, list[float], bool, int]] = []
    for p0 in p0_list:
        tau_target = p0 / ((p - p0) * math.sqrt(n))
        for r in range(1, int(repeats) + 1):
            tasks.append(
                (
                    int(p0),
                    int(r),
                    seed,
                    group_sizes,
                    sampler,
                    float(tau_target),
                    [float(sc) for sc in tau_scales],
                    bool(enforce_bayes_convergence),
                    int(retry_limit),
                )
            )

    rows_nested = _parallel_rows(tasks, _exp8_worker, n_jobs=n_jobs, prefer_process=True, progress_desc="Exp8 Tau Calibration")
    rows: list[dict[str, Any]] = []
    for chunk in rows_nested:
        rows.extend(chunk)
    raw = pd.DataFrame(rows)
    raw["tau_abs_error"] = (raw["tau_post_mean"] - raw["tau_target"]).abs()
    raw["tau_rel_error"] = raw["tau_abs_error"] / raw["tau_target"].clip(lower=1e-12)
    run_counts = raw.groupby(["p0", "tau_mode"], as_index=False).agg(n_total_runs=("replicate_id", "count"))
    convergence_audit = raw.groupby(["p0", "tau_mode"], as_index=False).agg(
        n_total_runs=("replicate_id", "count"),
        n_converged=("converged", "sum"),
        convergence_rate=("converged", "mean"),
        fit_attempts_mean=("fit_attempts", "mean"),
        fit_attempts_max=("fit_attempts", "max"),
    )
    summary = raw.groupby(["p0", "tau_mode"], as_index=False).agg(
        tau_target=("tau_target", "mean"),
        tau_post_mean=("tau_post_mean", "mean"),
        tau_post_sd=("tau_post_sd", "mean"),
        tau_abs_error=("tau_abs_error", "mean"),
        tau_rel_error=("tau_rel_error", "mean"),
        kappa_eff_sum_post_mean=("kappa_eff_sum_post_mean", "mean"),
        n_effective=("converged", "sum"),
        n_valid_metrics=("tau_post_mean", lambda s: int(s.notna().sum())),
    )
    summary = summary.merge(run_counts, on=["p0", "tau_mode"], how="left")
    summary["valid_rate"] = summary["n_effective"] / summary["n_total_runs"].clip(lower=1)

    mode_levels = ["auto_calibrated"] + [f"fixed_{float(sc):.2f}x" for sc in tau_scales]
    paired_raw, paired_stats = _paired_converged_subset(
        raw,
        group_cols=["p0"],
        method_col="tau_mode",
        replicate_col="replicate_id",
        converged_col="converged",
        required_cols=["tau_post_mean", "tau_post_sd", "kappa_eff_sum_post_mean", "tau_abs_error", "tau_rel_error"],
        method_levels=mode_levels,
    )
    summary_paired = paired_raw.groupby(["p0", "tau_mode"], as_index=False).agg(
        tau_target=("tau_target", "mean"),
        tau_post_mean=("tau_post_mean", "mean"),
        tau_post_sd=("tau_post_sd", "mean"),
        tau_abs_error=("tau_abs_error", "mean"),
        tau_rel_error=("tau_rel_error", "mean"),
        kappa_eff_sum_post_mean=("kappa_eff_sum_post_mean", "mean"),
        n_effective=("converged", "sum"),
        n_paired=("replicate_id", "nunique"),
    )
    if not summary_paired.empty:
        summary_paired = summary_paired.merge(run_counts, on=["p0", "tau_mode"], how="left")
        summary_paired = summary_paired.merge(paired_stats, on=["p0"], how="left")
        summary_paired["paired_rate"] = summary_paired["n_paired"] / summary_paired["n_total_runs"].clip(lower=1)

    summary_main = summary_paired if not summary_paired.empty else summary
    save_dataframe(raw, out / "raw_results.csv")
    save_dataframe(summary_main, out / "summary.csv")
    save_dataframe(summary, out / "summary_all.csv")
    save_dataframe(summary_paired, out / "summary_paired_converged.csv")
    save_dataframe(convergence_audit, out / "convergence_audit.csv")
    save_dataframe(paired_stats, out / "paired_convergence_audit.csv")
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
            "signal_profile": {"strong_value": 2.0, "weak_value": 0.5, "strong_share": 0.5},
            "sampler": {
                "chains": int(sampler.chains),
                "warmup": int(sampler.warmup),
                "post_warmup_draws": int(sampler.post_warmup_draws),
            },
            "profile": profile_name,
            "enforce_bayes_convergence": bool(enforce_bayes_convergence),
            "max_convergence_retries": int(retry_limit),
            "until_bayes_converged": bool(until_bayes_converged),
            "until_converged_retry_hard_cap": int(_UNTIL_CONVERGED_RETRY_HARD_CAP),
            "primary_error_metric": "tau_rel_error",
            "comparison_table_policy": "paired_converged_if_available",
        },
        out / "exp8_meta.json",
    )
    log.info("Completed exp8 with repeats=%d, profile=%s", repeats, profile_name)
    return {"raw": str(out / "raw_results.csv"), "summary": str(out / "summary.csv"), "figure": str(fig_dir / "fig8_tau_calibration.png")}


def run_exp9_beta_prior_sensitivity(
    n_jobs: int = 1,
    seed: int = MASTER_SEED,
    repeats: int = 120,
    save_dir: str = "simulation_project",
    *,
    profile: str = "full",
    enforce_bayes_convergence: bool = True,
    max_convergence_retries: int | None = None,
    until_bayes_converged: bool = True,
) -> Dict[str, str]:
    # ============================================================
    # EXP9 鈥?Beta(伪_魏, 尾_魏) 鍏堥獙鏁忔劅鎬э紙Theorem 2.8锛?
    #
    # 銆愮悊璁鸿繛鎺ャ€?
    #   Theorem 2.8锛毼瞋魏 鎺у埗杈归檯鍏堥獙 蟺(尾_j) 鐨勫熬閮ㄦ寚鏁般€?
    #   尾_魏 瓒婂ぇ 鈫?鍏堥獙瓒婁繚瀹堬紙瀵瑰ぇ |尾_j| 鎯╃綒鏇撮噸锛?鈫?闆剁粍鏀剁缉鏇村揩
    #   伪_魏 瓒婂ぇ 鈫?鍏堥獙鍧囧€兼洿楂橈紙魏 鏇村€惧悜 1锛?鈫?淇″彿妫€娴嬫洿绉瀬
    #
    # 銆愬弬鏁伴€夋嫨璇存槑銆?
    #   priors = [(0.5,0.5),(1,1),(1,2),(0.5,1),(2.5,1)]
    #     (0.5,1.0)锛氶粯璁ゆ帹鑽愶紝杞诲井鍊惧悜 魏鈫?
    #     (2.5,1.0)锛氭洿绉瀬鐨勪俊鍙锋娴嬶紙伪 澶э級
    #     (1.0,2.0)锛氭洿淇濆畧鐨勬敹缂╋紙尾 澶э級
    #   pg_levels=[20,50]锛氭湁闄愭牱鏈笅鍏堥獙褰卞搷搴斿湪灏?p_g 鏃舵洿澶э紙澶?p_g 浼肩劧涓诲锛?
    #   scenarios锛歜aseline锛堟搴︿俊鍙凤級+ tail_extreme锛堝崟鏋佺淇″彿锛?
    #
    # 銆愮悊鎯崇粨鏋溿€?
    #   缁撴灉瀵瑰厛楠?绋冲仴"锛? 绉嶅厛楠岀殑 AUROC 宸紓 < 0.05锛堥瞾妫掓€э級
    #   null_group_kappa_mean锛毼瞋魏 澧炲ぇ鏃跺簲鏇村皬锛堥浂缁勬洿濂芥敹缂╋級
    #   null_group_prob_kappa_gt_0_1锛埼篲null > 0.1 鐨勬鐜囷級锛毼瞋魏 澧炲ぇ鏃跺簲闄嶄綆
    #   tail_extreme 鍦烘櫙锛氭墍鏈夊厛楠岄兘搴旀纭瘑鍒瀬寮轰俊鍙风粍锛圓UROC鈮?锛?
    #
    # 銆愯嫢鍏堥獙鏁忔劅鎬ц繃澶х殑璋冩暣鏂瑰悜銆?
    #   鐥囩姸 A 鈥?AUROC 闅忓厛楠屽墽鐑堝彉鍖栵紙宸窛 > 0.1锛夛細
    #     鈫?pg 澶皬锛坧_g=20,50 澶勪簬鍏堥獙涓诲鍖哄煙锛夛紝澧炲ぇ鑷?pg_levels=[50,100]
    #     鈫?鎴栬繖鏄悎鐞嗙殑缁撴灉锛岃鏄庡厛楠岄€夋嫨瀵瑰皬鏍锋湰鏈夊疄璐ㄥ奖鍝嶏紝鍦ㄨ鏂囦腑璁ㄨ
    #
    #   鐥囩姸 B 鈥?(0.5,0.5) 鍏堥獙鐨勯浂缁?魏 鏃犳硶鏀剁缉锛坣ull_kappa_mean 鍋忓ぇ锛夛細
    #     鈫?Beta(0.5,0.5) 鏄弻妯?U 褰㈠垎甯冿紝澶ч噺璐ㄩ噺鍦?0 鍜?1 闄勮繎锛屼絾涔熷厑璁镐腑闂村€?
    #     鈫?鍦ㄨ繖绉嶅厛楠屼笅 p_g=20 鐨勪技鐒跺彲鑳戒笉瓒充互鍘嬪埗鍏堥獙锛岃繖鏄鏈熻涓?
    #
    #   鐥囩姸 C 鈥?kappa_curve锛堟寜 p_g 鍒嗗眰鐨?魏 鏇茬嚎锛夋棤鍒嗗眰宸紓锛?
    #     鈫?p_g=20 vs 50 鐨勫樊璺濅笉澶燂紱澧炲姞 pg_levels=[20,50,100,200]
    #     鈫?鎴栬€呭厛楠屾晥搴斿湪鎵€鏈?p_g 涓嬮兘灏忥紝璇存槑浼肩劧宸蹭富瀵?鈫?椴佹鎬х粨璁?
    # ============================================================
    import pandas as pd
    from .plotting import plot_exp9_prior_sensitivity
    base = Path(save_dir)
    out = ensure_dir(base / "results" / "exp9_beta_prior_sensitivity")
    fig_dir = ensure_dir(base / "figures")
    tab_dir = ensure_dir(base / "tables")
    log = setup_logger("exp9", base / "logs" / "exp9_beta_prior_sensitivity.log")
    profile_name = _normalize_compute_profile(profile)
    sampler = _sampler_for_profile(profile_name, experiment="exp9")
    retry_limit = _resolve_convergence_retry_limit(
        profile_name,
        max_convergence_retries,
        until_bayes_converged=bool(until_bayes_converged),
    )
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
    tasks: list[tuple[int, int, str, str, list[float], list[int], int, SamplerConfig, list[tuple[float, float]], bool, int]] = []
    for sid, (scenario, scenario_base, mu, group_sizes, p_g) in enumerate(scenarios, start=1):
        for r in range(1, int(repeats) + 1):
            tasks.append(
                (
                    sid,
                    r,
                    scenario,
                    scenario_base,
                    list(mu),
                    list(group_sizes),
                    seed,
                    sampler,
                    list(priors),
                    bool(enforce_bayes_convergence),
                    int(retry_limit),
                )
            )
    rows_nested = _parallel_rows(tasks, _exp9_worker, n_jobs=n_jobs, prefer_process=True, progress_desc="Exp9 Beta Prior Sensitivity")
    rows: list[dict[str, Any]] = []
    for chunk in rows_nested:
        rows.extend(chunk)
    raw = pd.DataFrame(rows)
    raw["prior_id"] = raw.apply(lambda r: f"a={float(r['alpha_kappa']):.3g}|b={float(r['beta_kappa']):.3g}", axis=1)
    prior_levels = [f"a={float(a):.3g}|b={float(b):.3g}" for a, b in priors]
    convergence_audit = raw.groupby(["scenario", "scenario_base", "p_g", "alpha_kappa", "beta_kappa"], as_index=False).agg(
        n_total_runs=("replicate_id", "count"),
        n_converged=("converged", "sum"),
        convergence_rate=("converged", "mean"),
        fit_attempts_mean=("fit_attempts", "mean"),
        fit_attempts_max=("fit_attempts", "max"),
    )
    run_counts = raw.groupby(["scenario", "scenario_base", "p_g", "alpha_kappa", "beta_kappa"], as_index=False).agg(
        n_total_runs=("replicate_id", "count")
    )
    table = raw.groupby(["scenario", "scenario_base", "p_g", "alpha_kappa", "beta_kappa"], as_index=False).agg(
        null_group_mse_avg=("null_group_mse_avg", "mean"),
        signal_group_mse_avg=("signal_group_mse_avg", "mean"),
        group_auroc=("group_auroc", "mean"),
        null_group_kappa_mean=("null_group_kappa_mean", "mean"),
        signal_group_kappa_mean=("signal_group_kappa_mean", "mean"),
        null_group_prob_kappa_gt_0_1=("null_group_prob_kappa_gt_0_1", "mean"),
        n_effective=("converged", "sum"),
        n_valid_metrics=("group_auroc", lambda s: int(s.notna().sum())),
    )
    table = table.merge(run_counts, on=["scenario", "scenario_base", "p_g", "alpha_kappa", "beta_kappa"], how="left")
    table["valid_rate"] = table["n_effective"] / table["n_total_runs"].clip(lower=1)

    paired_raw, paired_stats = _paired_converged_subset(
        raw,
        group_cols=["scenario", "scenario_base", "p_g"],
        method_col="prior_id",
        replicate_col="replicate_id",
        converged_col="converged",
        required_cols=[
            "null_group_mse_avg",
            "signal_group_mse_avg",
            "group_auroc",
            "null_group_kappa_mean",
            "null_group_prob_kappa_gt_0_1",
        ],
        method_levels=prior_levels,
    )
    table_paired = paired_raw.groupby(["scenario", "scenario_base", "p_g", "alpha_kappa", "beta_kappa"], as_index=False).agg(
        null_group_mse_avg=("null_group_mse_avg", "mean"),
        signal_group_mse_avg=("signal_group_mse_avg", "mean"),
        group_auroc=("group_auroc", "mean"),
        null_group_kappa_mean=("null_group_kappa_mean", "mean"),
        signal_group_kappa_mean=("signal_group_kappa_mean", "mean"),
        null_group_prob_kappa_gt_0_1=("null_group_prob_kappa_gt_0_1", "mean"),
        n_effective=("converged", "sum"),
        n_paired=("replicate_id", "nunique"),
    )
    if not table_paired.empty:
        table_paired = table_paired.merge(run_counts, on=["scenario", "scenario_base", "p_g", "alpha_kappa", "beta_kappa"], how="left")
        table_paired = table_paired.merge(paired_stats, on=["scenario", "scenario_base", "p_g"], how="left")
        table_paired["paired_rate"] = table_paired["n_paired"] / table_paired["n_total_runs"].clip(lower=1)

    kappa_counts = raw.groupby(["scenario_base", "p_g", "alpha_kappa", "beta_kappa"], as_index=False).agg(n_total_runs=("replicate_id", "count"))
    kappa_curve = raw.groupby(["scenario_base", "p_g", "alpha_kappa", "beta_kappa"], as_index=False).agg(
        null_group_kappa_mean=("null_group_kappa_mean", "mean"),
        null_group_prob_kappa_gt_0_1=("null_group_prob_kappa_gt_0_1", "mean"),
        n_effective=("converged", "sum"),
        n_valid_metrics=("null_group_kappa_mean", lambda s: int(s.notna().sum())),
    )
    kappa_curve = kappa_curve.merge(kappa_counts, on=["scenario_base", "p_g", "alpha_kappa", "beta_kappa"], how="left")
    kappa_curve["valid_rate"] = kappa_curve["n_effective"] / kappa_curve["n_total_runs"].clip(lower=1)

    kappa_curve_paired = paired_raw.groupby(["scenario_base", "p_g", "alpha_kappa", "beta_kappa"], as_index=False).agg(
        null_group_kappa_mean=("null_group_kappa_mean", "mean"),
        null_group_prob_kappa_gt_0_1=("null_group_prob_kappa_gt_0_1", "mean"),
        n_effective=("converged", "sum"),
        n_paired=("replicate_id", "nunique"),
    )
    if not kappa_curve_paired.empty:
        kappa_curve_paired = kappa_curve_paired.merge(kappa_counts, on=["scenario_base", "p_g", "alpha_kappa", "beta_kappa"], how="left")
        kappa_curve_paired["paired_rate"] = kappa_curve_paired["n_paired"] / kappa_curve_paired["n_total_runs"].clip(lower=1)

    table_main = table_paired if not table_paired.empty else table
    kappa_main = kappa_curve_paired if not kappa_curve_paired.empty else kappa_curve
    save_dataframe(raw, out / "raw_results.csv")
    save_dataframe(paired_raw, out / "raw_results_paired_converged.csv")
    save_dataframe(table_main, out / "summary.csv")
    save_dataframe(kappa_main, out / "summary_kappa_curve.csv")
    save_dataframe(table, out / "summary_all.csv")
    save_dataframe(table_paired, out / "summary_paired_converged.csv")
    save_dataframe(kappa_curve, out / "summary_kappa_curve_all.csv")
    save_dataframe(kappa_curve_paired, out / "summary_kappa_curve_paired_converged.csv")
    save_dataframe(convergence_audit, out / "convergence_audit.csv")
    save_dataframe(paired_stats, out / "paired_convergence_audit.csv")
    save_dataframe(table_main, tab_dir / "table_beta_prior_sensitivity.csv")
    save_dataframe(table, tab_dir / "table_beta_prior_sensitivity_all.csv")
    save_dataframe(table_paired, tab_dir / "table_beta_prior_sensitivity_paired_converged.csv")
    plot_exp9_prior_sensitivity(table_main, kappa_main, out_path=fig_dir / "fig9_beta_prior_sensitivity.png")
    save_json(
        {
            "repeats": int(repeats),
            "pg_levels": pg_levels,
            "scenario_templates": [{"name": name, "mu": mu} for name, mu in scenario_templates],
            "priors": [{"alpha_kappa": float(a), "beta_kappa": float(b)} for a, b in priors],
            "profile": profile_name,
            "enforce_bayes_convergence": bool(enforce_bayes_convergence),
            "max_convergence_retries": int(retry_limit),
            "until_bayes_converged": bool(until_bayes_converged),
            "until_converged_retry_hard_cap": int(_UNTIL_CONVERGED_RETRY_HARD_CAP),
            "comparison_table_policy": "paired_converged_if_available",
        },
        out / "exp9_meta.json",
    )
    log.info("Completed exp9 with repeats=%d, profile=%s", repeats, profile_name)
    return {"raw": str(out / "raw_results.csv"), "table": str(tab_dir / "table_beta_prior_sensitivity.csv")}


def run_all(
    save_dir: str = "simulation_project",
    seed: int = MASTER_SEED,
    n_jobs: int = 1,
    *,
    profile: str = "full",
    enforce_bayes_convergence: bool = True,
    max_convergence_retries: int | None = None,
    until_bayes_converged: bool = True,
) -> Dict[str, Dict[str, str]]:
    profile_name = _normalize_compute_profile(profile)
    out: Dict[str, Dict[str, str]] = {}
    jobs = [
        ("exp1", lambda: run_exp1_null_contraction(save_dir=save_dir, seed=seed, n_jobs=n_jobs, repeats=_default_repeats("exp1", profile_name))),
        ("exp2", lambda: run_exp2_adaptive_localization(save_dir=save_dir, seed=seed, n_jobs=n_jobs, repeats=_default_repeats("exp2", profile_name))),
        ("exp3", lambda: run_exp3_phase_diagram(save_dir=save_dir, seed=seed, n_jobs=n_jobs, repeats=_default_repeats("exp3", profile_name))),
        (
            "exp4",
            lambda: run_exp4_benchmark_linear(
                save_dir=save_dir,
                seed=seed,
                n_jobs=n_jobs,
                repeats=_default_repeats("exp4", profile_name),
                profile=profile_name,
                enforce_bayes_convergence=bool(enforce_bayes_convergence),
                max_convergence_retries=max_convergence_retries,
                until_bayes_converged=bool(until_bayes_converged),
            ),
        ),
        (
            "exp5",
            lambda: run_exp5_heterogeneity(
                save_dir=save_dir,
                seed=seed,
                n_jobs=n_jobs,
                repeats=_default_repeats("exp5", profile_name),
                profile=profile_name,
                enforce_bayes_convergence=bool(enforce_bayes_convergence),
                max_convergence_retries=max_convergence_retries,
                until_bayes_converged=bool(until_bayes_converged),
            ),
        ),
        (
            "exp6",
            lambda: run_exp6_grouped_logistic(
                save_dir=save_dir,
                seed=seed,
                n_jobs=n_jobs,
                repeats=_default_repeats("exp6", profile_name),
                profile=profile_name,
                enforce_bayes_convergence=bool(enforce_bayes_convergence),
                max_convergence_retries=max_convergence_retries,
                until_bayes_converged=bool(until_bayes_converged),
            ),
        ),
        (
            "exp7",
            lambda: run_exp7_ablation(
                save_dir=save_dir,
                seed=seed,
                n_jobs=n_jobs,
                repeats=_default_repeats("exp7", profile_name),
                profile=profile_name,
                enforce_bayes_convergence=bool(enforce_bayes_convergence),
                max_convergence_retries=max_convergence_retries,
                until_bayes_converged=bool(until_bayes_converged),
            ),
        ),
        (
            "exp8",
            lambda: run_exp8_tau_calibration(
                save_dir=save_dir,
                seed=seed,
                n_jobs=n_jobs,
                repeats=_default_repeats("exp8", profile_name),
                profile=profile_name,
                enforce_bayes_convergence=bool(enforce_bayes_convergence),
                max_convergence_retries=max_convergence_retries,
                until_bayes_converged=bool(until_bayes_converged),
            ),
        ),
        (
            "exp9",
            lambda: run_exp9_beta_prior_sensitivity(
                save_dir=save_dir,
                seed=seed,
                n_jobs=n_jobs,
                repeats=_default_repeats("exp9", profile_name),
                profile=profile_name,
                enforce_bayes_convergence=bool(enforce_bayes_convergence),
                max_convergence_retries=max_convergence_retries,
                until_bayes_converged=bool(until_bayes_converged),
            ),
        ),
    ]
    for name, runner in tqdm(jobs, total=len(jobs), desc="All Experiments", leave=True):
        out[name] = runner()
    save_json(
        {
            "profile": profile_name,
            "enforce_bayes_convergence": bool(enforce_bayes_convergence),
            "max_convergence_retries": None if max_convergence_retries is None else int(max_convergence_retries),
            "until_bayes_converged": bool(until_bayes_converged),
            "until_converged_retry_hard_cap": int(_UNTIL_CONVERGED_RETRY_HARD_CAP),
            "results": out,
        },
        Path(save_dir) / "results" / "run_manifest.json",
    )
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
    parser.add_argument("--profile", type=str, default="full", choices=list(COMPUTE_PROFILES))
    parser.add_argument("--no-enforce-bayes-convergence", action="store_true")
    parser.add_argument("--max-convergence-retries", type=int, default=None)
    parser.add_argument("--until-bayes-converged", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=8)
    args = parser.parse_args()
    profile_name = _normalize_compute_profile(args.profile)
    enforce_conv = not bool(args.no_enforce_bayes_convergence)
    until_conv = bool(args.until_bayes_converged) or (enforce_conv and args.max_convergence_retries is None)

    if args.experiment == "all":
        run_all(
            save_dir=args.save_dir,
            seed=args.seed,
            n_jobs=args.n_jobs,
            profile=profile_name,
            enforce_bayes_convergence=enforce_conv,
            max_convergence_retries=args.max_convergence_retries,
            until_bayes_converged=until_conv,
        )
    elif args.experiment == "1":
        run_exp1_null_contraction(
            repeats=args.repeats or _default_repeats("exp1", profile_name),
            save_dir=args.save_dir,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )
    elif args.experiment == "2":
        run_exp2_adaptive_localization(
            repeats=args.repeats or _default_repeats("exp2", profile_name),
            save_dir=args.save_dir,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )
    elif args.experiment == "3":
        run_exp3_phase_diagram(
            repeats=args.repeats or _default_repeats("exp3", profile_name),
            save_dir=args.save_dir,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )
    elif args.experiment == "4":
        run_exp4_benchmark_linear(
            repeats=args.repeats or _default_repeats("exp4", profile_name),
            save_dir=args.save_dir,
            seed=args.seed,
            n_jobs=args.n_jobs,
            profile=profile_name,
            enforce_bayes_convergence=enforce_conv,
            max_convergence_retries=args.max_convergence_retries,
            until_bayes_converged=until_conv,
        )
    elif args.experiment == "5":
        run_exp5_heterogeneity(
            repeats=args.repeats or _default_repeats("exp5", profile_name),
            save_dir=args.save_dir,
            seed=args.seed,
            n_jobs=args.n_jobs,
            profile=profile_name,
            enforce_bayes_convergence=enforce_conv,
            max_convergence_retries=args.max_convergence_retries,
            until_bayes_converged=until_conv,
        )
    elif args.experiment == "6":
        run_exp6_grouped_logistic(
            repeats=args.repeats or _default_repeats("exp6", profile_name),
            save_dir=args.save_dir,
            seed=args.seed,
            n_jobs=args.n_jobs,
            profile=profile_name,
            enforce_bayes_convergence=enforce_conv,
            max_convergence_retries=args.max_convergence_retries,
            until_bayes_converged=until_conv,
        )
    elif args.experiment == "7":
        run_exp7_ablation(
            repeats=args.repeats or _default_repeats("exp7", profile_name),
            save_dir=args.save_dir,
            seed=args.seed,
            n_jobs=args.n_jobs,
            profile=profile_name,
            enforce_bayes_convergence=enforce_conv,
            max_convergence_retries=args.max_convergence_retries,
            until_bayes_converged=until_conv,
        )
    elif args.experiment == "8":
        run_exp8_tau_calibration(
            repeats=args.repeats or _default_repeats("exp8", profile_name),
            save_dir=args.save_dir,
            seed=args.seed,
            n_jobs=args.n_jobs,
            profile=profile_name,
            enforce_bayes_convergence=enforce_conv,
            max_convergence_retries=args.max_convergence_retries,
            until_bayes_converged=until_conv,
        )
    elif args.experiment == "9":
        run_exp9_beta_prior_sensitivity(
            repeats=args.repeats or _default_repeats("exp9", profile_name),
            save_dir=args.save_dir,
            seed=args.seed,
            n_jobs=args.n_jobs,
            profile=profile_name,
            enforce_bayes_convergence=enforce_conv,
            max_convergence_retries=args.max_convergence_retries,
            until_bayes_converged=until_conv,
        )
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
