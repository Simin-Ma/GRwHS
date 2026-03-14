"""Unified multi-chain posterior diagnostics workflow for GRRHS models.

This script implements a convergence-and-credibility checklist that combines:
- parameter-level diagnostics (R-hat, bulk/tail ESS, MCSE)
- shrinkage function diagnostics (kappa, edf_g, slab/spike ratios)
- diagnostic plots (trace, rank, autocorrelation, pair, running summaries)
- posterior predictive checks on held-out data when available
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from grrhs.diagnostics.convergence import effective_sample_size, split_rhat
from grrhs.metrics.evaluation import _predictive_draws
from grrhs.viz.diagnostics import load_run_artifacts


Array = np.ndarray


@dataclass
class DiagnosticThresholds:
    rhat_strict: float = 1.01
    rhat_relaxed: float = 1.03
    ess_global: float = 1000.0
    ess_group: float = 400.0
    ess_local: float = 200.0
    mcse_ratio_good: float = 0.05
    mcse_ratio_ok: float = 0.10


@dataclass
class SeriesDiagnostic:
    name: str
    rhat: float
    ess_bulk: float
    ess_tail: float
    post_mean: float
    post_sd: float
    mcse_mean: float
    mcse_ratio: float
    pass_rhat: bool
    pass_ess: bool
    pass_mcse: bool
    category: str


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _flatten_groups(groups: Sequence[Sequence[int]], p: int) -> Array:
    group_index = np.zeros(p, dtype=int)
    for gid, idxs in enumerate(groups):
        group_index[np.asarray(list(idxs), dtype=int)] = gid
    return group_index


def _as_chain_draws(arr: Array, *, scalar_param: bool = False) -> Array:
    x = np.asarray(arr, dtype=float)
    if x.ndim == 0:
        raise ValueError("samples must have at least one dimension")

    if scalar_param:
        if x.ndim == 1:
            return x.reshape(1, x.shape[0])
        if x.ndim == 2:
            return x
        trailing = int(np.prod(x.shape[2:], dtype=int))
        if trailing == 1:
            return x.reshape(x.shape[0], x.shape[1])
        raise ValueError(f"scalar parameter expected, got shape {x.shape}")

    if x.ndim == 1:
        return x.reshape(1, x.shape[0], 1)
    if x.ndim == 2:
        return x.reshape(1, x.shape[0], x.shape[1])
    return x


def _trim_burnin(arr: Array, burn_in: int, *, scalar_param: bool = False) -> Array:
    x = _as_chain_draws(arr, scalar_param=scalar_param)
    if scalar_param:
        if burn_in <= 0:
            return x
        if burn_in >= x.shape[1]:
            raise ValueError(f"burn-in {burn_in} removes all draws for scalar samples with {x.shape[1]} draws")
        return x[:, burn_in:]

    if burn_in <= 0:
        return x
    if burn_in >= x.shape[1]:
        raise ValueError(f"burn-in {burn_in} removes all draws for samples with {x.shape[1]} draws")
    return x[:, burn_in:, ...]


def _interleave_chains(x: Array) -> Array:
    if x.ndim != 2:
        raise ValueError(f"expected 2D chain-by-draw array, got {x.shape}")
    return np.asarray(x.T.reshape(-1), dtype=float)


def _lag1_autocorr(seq: Array) -> float:
    z = np.asarray(seq, dtype=float)
    if z.size < 3:
        return float("nan")
    z = z - np.mean(z)
    denom = float(np.dot(z, z))
    if denom <= 0:
        return 0.0
    return float(np.dot(z[:-1], z[1:]) / denom)


def _tail_ess(samples: Array, *, scalar_param: bool = False, probs: Tuple[float, float] = (0.10, 0.90)) -> float:
    x = _as_chain_draws(samples, scalar_param=scalar_param)
    if scalar_param:
        flat = x.reshape(-1)
    else:
        flat = x.reshape(x.shape[0] * x.shape[1], -1)
    if flat.size == 0:
        return float("nan")

    tails: List[float] = []
    for q in probs:
        qv = np.quantile(flat, q, axis=0)
        if scalar_param:
            indicator = (x <= float(qv)).astype(float)
            ess = float(np.asarray(effective_sample_size(indicator, scalar_param=True)).reshape(-1)[0])
            tails.append(ess)
        else:
            indicator = (x <= qv.reshape((1, 1) + qv.shape)).astype(float)
            ess_arr = np.asarray(effective_sample_size(indicator, scalar_param=False)).reshape(-1)
            tails.append(float(np.min(ess_arr)))
    return float(min(tails))


def _rank_per_chain(samples_2d: Array) -> List[Array]:
    chains, draws = samples_2d.shape
    pooled = samples_2d.reshape(-1)
    order = np.argsort(pooled, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(pooled.size)
    ranks = ranks.reshape(chains, draws)
    return [ranks[c] for c in range(chains)]


def _acf(x: Array, max_lag: int) -> Array:
    seq = np.asarray(x, dtype=float)
    seq = seq - np.mean(seq)
    if seq.size == 0:
        return np.zeros(max_lag + 1, dtype=float)
    corr = np.correlate(seq, seq, mode="full")
    mid = corr.size // 2
    denom = corr[mid]
    if denom <= 0:
        return np.zeros(max_lag + 1, dtype=float)
    return corr[mid : mid + max_lag + 1] / denom


def _running_summary(seq: Array, checkpoints: Array) -> Tuple[Array, Array, Array, Array]:
    means = []
    q10 = []
    q50 = []
    q90 = []
    for k in checkpoints:
        head = seq[:k]
        means.append(float(np.mean(head)))
        q10.append(float(np.quantile(head, 0.10)))
        q50.append(float(np.quantile(head, 0.50)))
        q90.append(float(np.quantile(head, 0.90)))
    return np.asarray(means), np.asarray(q10), np.asarray(q50), np.asarray(q90)


def _representative_indices(beta_cd: Array, k_top: int = 4, k_zero: int = 4) -> Tuple[List[int], List[int]]:
    pooled = beta_cd.reshape(beta_cd.shape[0] * beta_cd.shape[1], beta_cd.shape[2])
    mean_abs = np.mean(np.abs(pooled), axis=0)
    top = np.argsort(-mean_abs)[: max(1, min(k_top, mean_abs.size))]
    zeros = np.argsort(mean_abs)[: max(1, min(k_zero, mean_abs.size))]
    return top.tolist(), zeros.tolist()


def _compute_grrhs_function_draws(
    *,
    X: Array,
    group_index: Array,
    c: float,
    tau_cd: Array,
    phi_cdg: Array,
    lambda_cdp: Array,
    sigma_cd: Array,
) -> Dict[str, Array]:
    chains, draws, p = lambda_cdp.shape
    G = int(group_index.max()) + 1 if group_index.size else 1

    tau = np.maximum(tau_cd, 1e-12)
    sigma = np.maximum(sigma_cd, 1e-12)
    phi = np.maximum(phi_cdg, 1e-12)
    lam = np.maximum(lambda_cdp, 1e-12)

    tau2 = tau[..., None] ** 2
    sigma2 = sigma[..., None] ** 2
    lam2 = lam ** 2
    c2 = float(c) ** 2

    denom = np.maximum(c2 + tau2 * lam2, 1e-12)
    tilde_lam_sq = (c2 * lam2) / denom
    tilde_lam = np.sqrt(np.maximum(tilde_lam_sq, 1e-12))

    phi_j = phi[:, :, group_index]
    prior_prec = 1.0 / np.maximum((phi_j**2) * tau2 * tilde_lam_sq * sigma2, 1e-12)

    xtx_diag = np.sum(np.asarray(X, dtype=float) ** 2, axis=0)
    q = xtx_diag.reshape(1, 1, p) / sigma2
    kappa = q / np.maximum(q + prior_prec, 1e-12)
    r = (tau2 * lam2) / c2

    term_group = 2.0 * np.log(np.maximum(phi_j, 1e-8))
    term_tau = 2.0 * np.log(np.maximum(tau[..., None], 1e-8))
    term_lambda = 2.0 * np.log(np.maximum(tilde_lam, 1e-8))

    def _sym_clip(x: Array, eps: float = 1e-8) -> Array:
        sign = np.where(x >= 0.0, 1.0, -1.0)
        return sign * np.maximum(np.abs(x), eps)

    cg = _sym_clip(term_group)
    ct = _sym_clip(term_tau)
    cl = _sym_clip(term_lambda)
    omega_denom = cg + ct + cl
    omega_denom = np.where(np.abs(omega_denom) < 1e-8, np.sign(omega_denom + 1e-8) * 3e-8, omega_denom)

    omega_group = cg / omega_denom
    omega_tau = ct / omega_denom
    omega_lambda = cl / omega_denom

    edf = np.zeros((chains, draws, G), dtype=float)
    for g in range(G):
        mask = group_index == g
        if np.any(mask):
            edf[:, :, g] = np.sum(kappa[:, :, mask], axis=2)

    return {
        "kappa": kappa,
        "r": r,
        "edf": edf,
        "omega_group": omega_group,
        "omega_tau": omega_tau,
        "omega_lambda": omega_lambda,
    }


def _series_diagnostic(
    *,
    name: str,
    samples: Array,
    category: str,
    thresholds: DiagnosticThresholds,
    scalar_param: bool = True,
) -> SeriesDiagnostic:
    x = _as_chain_draws(samples, scalar_param=scalar_param)
    if scalar_param:
        flat = x.reshape(-1)
        sd = float(np.std(flat, ddof=1)) if flat.size > 1 else 0.0
        mean = float(np.mean(flat))
        ess_bulk = float(np.asarray(effective_sample_size(x, scalar_param=True)).reshape(-1)[0])
        rhat = float(np.asarray(split_rhat(x, scalar_param=True)).reshape(-1)[0]) if x.shape[0] >= 2 else float("nan")
        ess_tail = float(_tail_ess(x, scalar_param=True))
    else:
        flat = x.reshape(x.shape[0] * x.shape[1], -1)
        mean = float(np.mean(flat))
        sd = float(np.std(flat, ddof=1)) if flat.size > 1 else 0.0
        ess_bulk = float(np.min(np.asarray(effective_sample_size(x, scalar_param=False)).reshape(-1)))
        rhat = (
            float(np.max(np.asarray(split_rhat(x, scalar_param=False)).reshape(-1)))
            if x.shape[0] >= 2
            else float("nan")
        )
        ess_tail = float(_tail_ess(x, scalar_param=False))

    safe_ess = max(ess_bulk, 1.0)
    mcse_mean = float(sd / math.sqrt(safe_ess))
    mcse_ratio = float(mcse_mean / max(sd, 1e-12))

    if category == "global":
        ess_cut = thresholds.ess_global
    elif category == "group":
        ess_cut = thresholds.ess_group
    else:
        ess_cut = thresholds.ess_local

    pass_rhat = np.isfinite(rhat) and (rhat < thresholds.rhat_strict)
    pass_ess = np.isfinite(ess_bulk) and np.isfinite(ess_tail) and ess_bulk >= ess_cut and ess_tail >= ess_cut
    pass_mcse = mcse_ratio < thresholds.mcse_ratio_ok

    return SeriesDiagnostic(
        name=name,
        rhat=rhat,
        ess_bulk=ess_bulk,
        ess_tail=ess_tail,
        post_mean=mean,
        post_sd=sd,
        mcse_mean=mcse_mean,
        mcse_ratio=mcse_ratio,
        pass_rhat=pass_rhat,
        pass_ess=pass_ess,
        pass_mcse=pass_mcse,
        category=category,
    )


def _save_table_csv(rows: Sequence[SeriesDiagnostic], path: Path) -> None:
    header = (
        "name,category,rhat,ess_bulk,ess_tail,post_mean,post_sd,mcse_mean,mcse_ratio,"
        "pass_rhat,pass_ess,pass_mcse\n"
    )
    lines = [header]
    for r in rows:
        lines.append(
            f"{r.name},{r.category},{r.rhat:.6g},{r.ess_bulk:.6g},{r.ess_tail:.6g},"
            f"{r.post_mean:.6g},{r.post_sd:.6g},{r.mcse_mean:.6g},{r.mcse_ratio:.6g},"
            f"{int(r.pass_rhat)},{int(r.pass_ess)},{int(r.pass_mcse)}\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def _plot_trace(dest: Path, series_map: Mapping[str, Array]) -> None:
    keys = list(series_map.keys())
    fig, axes = plt.subplots(len(keys), 1, figsize=(11, max(2.5, 1.8 * len(keys))), sharex=True)
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        s = np.asarray(series_map[key], dtype=float)
        chains, draws = s.shape
        x = np.arange(draws)
        for c in range(chains):
            ax.plot(x, s[c], linewidth=0.9, alpha=0.85, label=f"chain {c + 1}")
        ax.set_ylabel(key)
        if chains <= 4:
            ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("Draw")
    fig.tight_layout()
    fig.savefig(dest / "trace_panel.png", dpi=180)
    plt.close(fig)


def _plot_rank(dest: Path, series_map: Mapping[str, Array], bins: int = 20) -> None:
    keys = list(series_map.keys())
    fig, axes = plt.subplots(len(keys), 1, figsize=(11, max(2.5, 1.8 * len(keys))), sharex=False)
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        s = np.asarray(series_map[key], dtype=float)
        ranks_by_chain = _rank_per_chain(s)
        max_rank = s.size
        for cid, ranks in enumerate(ranks_by_chain):
            ax.hist(
                ranks,
                bins=bins,
                range=(0, max_rank),
                density=True,
                histtype="step",
                linewidth=1.2,
                label=f"chain {cid + 1}",
            )
        ax.set_ylabel(key)
        ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(dest / "rank_hist_panel.png", dpi=180)
    plt.close(fig)


def _plot_acf(dest: Path, series_map: Mapping[str, Array], max_lag: int = 80) -> None:
    keys = list(series_map.keys())
    fig, axes = plt.subplots(len(keys), 1, figsize=(10, max(2.5, 1.8 * len(keys))), sharex=True)
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        s = np.asarray(series_map[key], dtype=float)
        mean_acf = np.zeros(max_lag + 1, dtype=float)
        for c in range(s.shape[0]):
            mean_acf += _acf(s[c], max_lag)
        mean_acf /= float(s.shape[0])
        ax.stem(np.arange(max_lag + 1), mean_acf, linefmt="-", markerfmt="o", basefmt=" ")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel(key)
    axes[-1].set_xlabel("Lag")
    fig.tight_layout()
    fig.savefig(dest / "autocorr_panel.png", dpi=180)
    plt.close(fig)


def _plot_pairs(
    dest: Path,
    *,
    log_tau_cd: Array,
    log_phi_cdg: Array,
    log_lambda_cdp: Array,
    group_sel: Sequence[int],
    coef_sel: Sequence[int],
) -> None:
    x_tau = log_tau_cd.reshape(-1)

    rows = max(len(group_sel), len(coef_sel))
    if rows == 0:
        return
    fig, axes = plt.subplots(rows, 2, figsize=(10, 3 * rows), squeeze=False)

    for ridx in range(rows):
        left = axes[ridx, 0]
        right = axes[ridx, 1]

        if ridx < len(group_sel):
            g = int(group_sel[ridx])
            y = log_phi_cdg[:, :, g].reshape(-1)
            left.scatter(x_tau, y, s=6, alpha=0.25)
            left.set_xlabel("log(tau)")
            left.set_ylabel(f"log(phi[{g}])")
        else:
            left.axis("off")

        if ridx < len(coef_sel):
            j = int(coef_sel[ridx])
            y = log_lambda_cdp[:, :, j].reshape(-1)
            right.scatter(x_tau, y, s=6, alpha=0.25)
            right.set_xlabel("log(tau)")
            right.set_ylabel(f"log(lambda[{j}])")
        else:
            right.axis("off")

    fig.tight_layout()
    fig.savefig(dest / "pair_panel_logtau_vs_scales.png", dpi=180)
    plt.close(fig)


def _plot_running(dest: Path, series_map: Mapping[str, Array], max_points: int = 220) -> None:
    keys = list(series_map.keys())
    fig, axes = plt.subplots(len(keys), 1, figsize=(11, max(2.5, 2.0 * len(keys))), sharex=False)
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        seq = _interleave_chains(np.asarray(series_map[key], dtype=float))
        n = seq.size
        checkpoints = np.unique(np.linspace(10, n, min(max_points, n), dtype=int))
        means, q10, q50, q90 = _running_summary(seq, checkpoints)
        ax.plot(checkpoints, means, label="running mean", linewidth=1.2)
        ax.plot(checkpoints, q50, label="running q50", linewidth=1.0)
        ax.fill_between(checkpoints, q10, q90, alpha=0.2, label="running q10-q90")
        ax.set_ylabel(key)
        ax.legend(loc="best", fontsize=7)
    axes[-1].set_xlabel("Interleaved draw index")
    fig.tight_layout()
    fig.savefig(dest / "running_summary_panel.png", dpi=180)
    plt.close(fig)


def _plot_ppc(
    dest: Path,
    *,
    y_true: Array,
    yrep: Array,
) -> Dict[str, float]:
    pred_mean = np.mean(yrep, axis=0)
    residual = y_true - pred_mean

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].hist(y_true, bins=35, alpha=0.6, label="y", density=True)
    axes[0].hist(yrep.reshape(-1), bins=35, alpha=0.5, label="y_rep draws", density=True)
    axes[0].legend(loc="best", fontsize=8)
    axes[0].set_title("Observed vs posterior predictive")

    axes[1].scatter(pred_mean, y_true, s=10, alpha=0.5)
    mn = float(min(np.min(pred_mean), np.min(y_true)))
    mx = float(max(np.max(pred_mean), np.max(y_true)))
    axes[1].plot([mn, mx], [mn, mx], color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Predictive mean")
    axes[1].set_ylabel("Observed y")
    axes[1].set_title("Calibration scatter")

    axes[2].hist(residual, bins=35, alpha=0.8, density=True)
    axes[2].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[2].set_title("Residual distribution")

    fig.tight_layout()
    fig.savefig(dest / "posterior_predictive_check.png", dpi=180)
    plt.close(fig)

    rmse = float(np.sqrt(np.mean((pred_mean - y_true) ** 2)))
    sigma2 = np.var(yrep - pred_mean[np.newaxis, :], axis=0) + 1e-8
    loglik = -0.5 * (np.log(2.0 * np.pi * sigma2) + (y_true - pred_mean) ** 2 / sigma2)

    return {
        "predictive_rmse": rmse,
        "heldout_log_predictive_density_mean": float(np.mean(loglik)),
        "residual_mean": float(np.mean(residual)),
        "residual_sd": float(np.std(residual, ddof=1)) if residual.size > 1 else 0.0,
    }


def _sampling_health(tau_cd: Array, sigma2_cd: Array, log_lambda_cdp: Array) -> Dict[str, float]:
    tau_seq = _interleave_chains(np.log(np.maximum(tau_cd, 1e-12)))
    sigma_seq = _interleave_chains(np.log(np.maximum(sigma2_cd, 1e-12)))

    d_tau = np.diff(tau_seq)
    d_sigma = np.diff(sigma_seq)
    stuck_tau_frac = float(np.mean(np.abs(d_tau) < 1e-8)) if d_tau.size else 0.0
    stuck_sigma_frac = float(np.mean(np.abs(d_sigma) < 1e-8)) if d_sigma.size else 0.0

    jump = np.abs(np.diff(np.exp(sigma_seq)))
    jump_q95 = float(np.quantile(jump, 0.95)) if jump.size else 0.0
    jump_q99 = float(np.quantile(jump, 0.99)) if jump.size else 0.0

    lambda_seq = log_lambda_cdp.reshape(-1, log_lambda_cdp.shape[-1])
    lambda_stuck = float(np.mean(np.std(lambda_seq, axis=0) < 1e-6))

    return {
        "tau_log_lag1_autocorr": _lag1_autocorr(tau_seq),
        "sigma2_log_lag1_autocorr": _lag1_autocorr(sigma_seq),
        "tau_stuck_step_fraction": stuck_tau_frac,
        "sigma2_stuck_step_fraction": stuck_sigma_frac,
        "sigma2_jump_q95": jump_q95,
        "sigma2_jump_q99": jump_q99,
        "log_lambda_near_constant_fraction": lambda_stuck,
    }


def _json_default(obj: object) -> object:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Type not serializable: {type(obj)!r}")


def _build_series_set(
    *,
    beta_cdp: Array,
    tau_cd: Array,
    sigma2_cd: Array,
    phi_cdg: Array,
    lambda_cdp: Array,
    fdraws: Mapping[str, Array],
    top_idx: Sequence[int],
    zero_idx: Sequence[int],
    group_sel: Sequence[int],
) -> Dict[str, Tuple[Array, str]]:
    targets: Dict[str, Tuple[Array, str]] = {
        "log_tau": (np.log(np.maximum(tau_cd, 1e-12)), "global"),
        "log_sigma2": (np.log(np.maximum(sigma2_cd, 1e-12)), "global"),
    }

    for g in group_sel:
        targets[f"log_phi[{g}]"] = (np.log(np.maximum(phi_cdg[:, :, g], 1e-12)), "group")
        targets[f"edf[{g}]"] = (fdraws["edf"][:, :, g], "group")

    for j in list(top_idx) + list(zero_idx):
        targets[f"beta[{j}]"] = (beta_cdp[:, :, j], "local")
        targets[f"log_lambda[{j}]"] = (np.log(np.maximum(lambda_cdp[:, :, j], 1e-12)), "local")
        targets[f"kappa[{j}]"] = (fdraws["kappa"][:, :, j], "local")
        targets[f"r[{j}]"] = (fdraws["r"][:, :, j], "local")
        targets[f"Pr(r[{j}]>1)"] = ((fdraws["r"][:, :, j] > 1.0).astype(float), "local")
        targets[f"omega_group[{j}]"] = (fdraws["omega_group"][:, :, j], "local")
        targets[f"omega_tau[{j}]"] = (fdraws["omega_tau"][:, :, j], "local")
        targets[f"omega_lambda[{j}]"] = (fdraws["omega_lambda"][:, :, j], "local")

    return targets


def _to_plot_map(targets: Mapping[str, Tuple[Array, str]], names: Iterable[str]) -> Dict[str, Array]:
    out: Dict[str, Array] = {}
    for name in names:
        if name in targets:
            out[name] = targets[name][0]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full GRRHS posterior convergence workflow on a run directory.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path containing posterior_samples.npz")
    parser.add_argument("--dest", type=Path, default=None, help="Output directory (default: <run>/diagnostics_bundle)")
    parser.add_argument("--burn-in", type=int, default=None, help="Discard this many draws from each chain")
    parser.add_argument("--burn-in-frac", type=float, default=0.5, help="Used when --burn-in is not provided")
    parser.add_argument("--top-k", type=int, default=4, help="Top |beta| coefficients to monitor")
    parser.add_argument("--zero-k", type=int, default=4, help="Near-zero coefficients to monitor")
    parser.add_argument("--group-k", type=int, default=4, help="Groups to monitor by median edf")
    parser.add_argument("--max-lag", type=int, default=80, help="Max lag for autocorrelation panels")
    parser.add_argument("--seed", type=int, default=2026, help="Seed for predictive checks")
    parser.add_argument("--strict-rhat", type=float, default=1.01)
    parser.add_argument("--relaxed-rhat", type=float, default=1.03)
    parser.add_argument("--ess-global", type=float, default=1000.0)
    parser.add_argument("--ess-group", type=float, default=400.0)
    parser.add_argument("--ess-local", type=float, default=200.0)
    args = parser.parse_args()

    artifacts = load_run_artifacts(args.run_dir)
    posterior = artifacts.posterior
    if not posterior:
        raise RuntimeError(f"No posterior_samples.npz found in {args.run_dir}")

    if "beta" not in posterior:
        raise RuntimeError("Posterior samples missing beta")

    beta = _as_chain_draws(np.asarray(posterior["beta"], dtype=float), scalar_param=False)
    tau = _as_chain_draws(np.asarray(posterior.get("tau"), dtype=float), scalar_param=True) if "tau" in posterior else None
    if tau is None:
        raise RuntimeError("Posterior samples missing tau")

    if "sigma2" in posterior:
        sigma2 = _as_chain_draws(np.asarray(posterior["sigma2"], dtype=float), scalar_param=True)
    elif "sigma" in posterior:
        sigma = _as_chain_draws(np.asarray(posterior["sigma"], dtype=float), scalar_param=True)
        sigma2 = np.square(np.maximum(sigma, 1e-12))
    else:
        raise RuntimeError("Posterior samples missing sigma/sigma2")

    if "phi" not in posterior:
        raise RuntimeError("Posterior samples missing phi")
    phi = _as_chain_draws(np.asarray(posterior["phi"], dtype=float), scalar_param=False)

    lambda_key = "lambda" if "lambda" in posterior else ("lambda_" if "lambda_" in posterior else None)
    if lambda_key is None:
        raise RuntimeError("Posterior samples missing lambda/lambda_")
    lambda_ = _as_chain_draws(np.asarray(posterior[lambda_key], dtype=float), scalar_param=False)

    chains = int(beta.shape[0])
    draws = int(beta.shape[1])
    if not (tau.shape[:2] == (chains, draws) and sigma2.shape[:2] == (chains, draws)):
        raise RuntimeError("beta/tau/sigma draw shapes are inconsistent")

    if args.burn_in is None:
        burn_in = int(draws * float(args.burn_in_frac))
    else:
        burn_in = int(args.burn_in)
    burn_in = max(0, min(burn_in, draws - 2))

    beta = _trim_burnin(beta, burn_in, scalar_param=False)
    tau = _trim_burnin(tau, burn_in, scalar_param=True)
    sigma2 = _trim_burnin(sigma2, burn_in, scalar_param=True)
    phi = _trim_burnin(phi, burn_in, scalar_param=False)
    lambda_ = _trim_burnin(lambda_, burn_in, scalar_param=False)

    if not (beta.shape[0] == tau.shape[0] == sigma2.shape[0] == phi.shape[0] == lambda_.shape[0]):
        raise RuntimeError("chain count mismatch after burn-in")
    if not (beta.shape[1] == tau.shape[1] == sigma2.shape[1] == phi.shape[1] == lambda_.shape[1]):
        raise RuntimeError("draw count mismatch after burn-in")

    dataset_arrays = artifacts.dataset_arrays
    X_train = dataset_arrays.get("X_train")
    if X_train is None:
        raise RuntimeError("dataset.npz missing X_train; cannot compute kappa/edf diagnostics")
    X_train = np.asarray(X_train, dtype=float)

    p = X_train.shape[1]
    if beta.shape[2] != p or lambda_.shape[2] != p:
        raise RuntimeError("feature dimension mismatch between posterior and dataset")

    groups = artifacts.dataset_meta.get("groups") or [[j] for j in range(p)]
    group_index = _flatten_groups(groups, p)
    G = int(group_index.max()) + 1 if group_index.size else 1
    if phi.shape[2] < G:
        raise RuntimeError("phi group dimension is smaller than required by dataset groups")

    c_value = 1.0
    if artifacts.resolved_config:
        c_value = float(artifacts.resolved_config.get("model", {}).get("c", c_value))

    fdraws = _compute_grrhs_function_draws(
        X=X_train,
        group_index=group_index,
        c=c_value,
        tau_cd=tau,
        phi_cdg=phi,
        lambda_cdp=lambda_,
        sigma_cd=np.sqrt(np.maximum(sigma2, 1e-12)),
    )

    top_idx, zero_idx = _representative_indices(beta, k_top=args.top_k, k_zero=args.zero_k)
    edf_median = np.median(fdraws["edf"].reshape(-1, fdraws["edf"].shape[2]), axis=0)
    group_sel = np.argsort(-edf_median)[: max(1, min(args.group_k, edf_median.size))].tolist()

    targets = _build_series_set(
        beta_cdp=beta,
        tau_cd=tau,
        sigma2_cd=sigma2,
        phi_cdg=phi,
        lambda_cdp=lambda_,
        fdraws=fdraws,
        top_idx=top_idx,
        zero_idx=zero_idx,
        group_sel=group_sel,
    )

    thresholds = DiagnosticThresholds(
        rhat_strict=float(args.strict_rhat),
        rhat_relaxed=float(args.relaxed_rhat),
        ess_global=float(args.ess_global),
        ess_group=float(args.ess_group),
        ess_local=float(args.ess_local),
    )

    dest = args.dest or (args.run_dir / "diagnostics_bundle")
    dest = dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    rows: List[SeriesDiagnostic] = []
    for name, (arr, category) in targets.items():
        rows.append(
            _series_diagnostic(
                name=name,
                samples=arr,
                category=category,
                thresholds=thresholds,
                scalar_param=True,
            )
        )

    rows_sorted = sorted(rows, key=lambda r: (not (r.pass_rhat and r.pass_ess and r.pass_mcse), -r.rhat if np.isfinite(r.rhat) else -1.0))

    sampler_health = _sampling_health(tau, sigma2, np.log(np.maximum(lambda_, 1e-12)))

    predictive_summary = {}
    X_test = dataset_arrays.get("X_test")
    y_test = dataset_arrays.get("y_test")
    if X_test is not None and y_test is not None:
        coef_samples = beta.reshape(beta.shape[0] * beta.shape[1], beta.shape[2])
        sigma_samples = np.sqrt(np.maximum(sigma2.reshape(-1), 1e-12))
        yrep = _predictive_draws(
            X=np.asarray(X_test, dtype=float),
            coef_samples=coef_samples,
            intercept=0.0,
            sigma_samples=sigma_samples,
            rng_seed=int(args.seed),
        )
        if yrep is not None:
            predictive_summary = _plot_ppc(
                dest,
                y_true=np.asarray(y_test, dtype=float).reshape(-1),
                yrep=np.asarray(yrep, dtype=float),
            )
    else:
        _warn("X_test/y_test unavailable: skipping posterior predictive check plot.")

    _save_table_csv(rows_sorted, dest / "diagnostics_table.csv")

    trace_names = [
        "log_tau",
        "log_sigma2",
        *[f"log_phi[{g}]" for g in group_sel[:2]],
        *[f"beta[{j}]" for j in top_idx[:2]],
        *[f"kappa[{j}]" for j in top_idx[:2]],
        *[f"r[{j}]" for j in top_idx[:2]],
        *[f"edf[{g}]" for g in group_sel[:2]],
    ]
    rank_names = trace_names[:8]
    acf_names = [
        "log_tau",
        "log_sigma2",
        *[f"log_phi[{g}]" for g in group_sel[:2]],
        *[f"r[{j}]" for j in top_idx[:2]],
    ]
    running_names = [
        "log_tau",
        *[f"edf[{g}]" for g in group_sel[:2]],
        *[f"Pr(r[{j}]>1)" for j in top_idx[:2]],
    ]

    trace_map = _to_plot_map(targets, trace_names)
    rank_map = _to_plot_map(targets, rank_names)
    acf_map = _to_plot_map(targets, acf_names)
    running_map = _to_plot_map(targets, running_names)

    if trace_map:
        _plot_trace(dest, trace_map)
    if rank_map and chains >= 2:
        _plot_rank(dest, rank_map)
    if acf_map:
        _plot_acf(dest, acf_map, max_lag=max(5, int(args.max_lag)))

    _plot_pairs(
        dest,
        log_tau_cd=np.log(np.maximum(tau, 1e-12)),
        log_phi_cdg=np.log(np.maximum(phi, 1e-12)),
        log_lambda_cdp=np.log(np.maximum(lambda_, 1e-12)),
        group_sel=group_sel[: min(3, len(group_sel))],
        coef_sel=top_idx[: min(3, len(top_idx))],
    )

    if running_map:
        _plot_running(dest, running_map)

    fail_rows = [r for r in rows_sorted if not (r.pass_rhat and r.pass_ess and r.pass_mcse)]
    pass_rate = 1.0 - (len(fail_rows) / max(len(rows_sorted), 1))

    summary = {
        "run_dir": str(args.run_dir),
        "output_dir": str(dest),
        "sampler": {
            "chains": int(beta.shape[0]),
            "draws_per_chain_after_burnin": int(beta.shape[1]),
            "burn_in": int(burn_in),
            "minimum_chain_requirement": 4,
            "minimum_chain_check_pass": bool(beta.shape[0] >= 4),
            "health": sampler_health,
        },
        "thresholds": {
            "rhat_strict": thresholds.rhat_strict,
            "rhat_relaxed": thresholds.rhat_relaxed,
            "ess_global": thresholds.ess_global,
            "ess_group": thresholds.ess_group,
            "ess_local": thresholds.ess_local,
            "mcse_ratio_good": thresholds.mcse_ratio_good,
            "mcse_ratio_ok": thresholds.mcse_ratio_ok,
        },
        "diagnostics": {
            "count": len(rows_sorted),
            "failed": len(fail_rows),
            "pass_rate": pass_rate,
            "worst_rhat": max((r.rhat for r in rows_sorted if np.isfinite(r.rhat)), default=float("nan")),
            "min_bulk_ess": min((r.ess_bulk for r in rows_sorted if np.isfinite(r.ess_bulk)), default=float("nan")),
            "min_tail_ess": min((r.ess_tail for r in rows_sorted if np.isfinite(r.ess_tail)), default=float("nan")),
            "max_mcse_ratio": max((r.mcse_ratio for r in rows_sorted if np.isfinite(r.mcse_ratio)), default=float("nan")),
            "failed_items": [r.name for r in fail_rows[:50]],
        },
        "representatives": {
            "top_beta_indices": top_idx,
            "near_zero_beta_indices": zero_idx,
            "group_indices": group_sel,
        },
        "predictive": predictive_summary,
    }

    (dest / "diagnostics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    print(f"[OK] Diagnostics bundle written to {dest}")


if __name__ == "__main__":
    main()
