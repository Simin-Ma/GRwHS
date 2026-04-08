"""Run GR-RHS theory-aligned simulation studies (Experiments 1-3).

This script implements the minimal executable versions described in the
simulation plan:

1) Cross-group inflation factor behavior (eta_g)
2) Non-vacuousness of the Theorem-2 style B_g* bound
3) Slab-induced stabilization of blockwise covariance/mean
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import invgamma  # noqa: E402


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _operator_norm(mat: np.ndarray) -> float:
    if mat.size == 0:
        return 0.0
    return float(np.linalg.svd(mat, compute_uv=False, full_matrices=False)[0])


def _project_to_spd(cov: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    sym = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(sym)
    vals = np.maximum(vals, eps)
    out = (vecs * vals) @ vecs.T
    d = np.sqrt(np.maximum(np.diag(out), eps))
    out = out / np.outer(d, d)
    return 0.5 * (out + out.T)


def _sample_multivariate_normal(n: int, cov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    p = cov.shape[0]
    cov_spd = _project_to_spd(cov)
    jitter = 1e-10
    for _ in range(8):
        try:
            chol = np.linalg.cholesky(cov_spd + jitter * np.eye(p))
            z = rng.normal(size=(n, p))
            return z @ chol.T
        except np.linalg.LinAlgError:
            jitter *= 10.0
    vals, vecs = np.linalg.eigh(cov_spd)
    vals = np.maximum(vals, 1e-10)
    z = rng.normal(size=(n, p))
    return z @ (vecs * np.sqrt(vals)).T


def _standardize_columns_to_sqrt_n(x: np.ndarray) -> np.ndarray:
    n, p = x.shape
    x = x - x.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(x, axis=0)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    return x * (math.sqrt(n) / norms)


def build_groups(g: int, m: int) -> list[np.ndarray]:
    groups = []
    for idx in range(g):
        lo = idx * m
        hi = (idx + 1) * m
        groups.append(np.arange(lo, hi, dtype=int))
    return groups


def generate_block_design(
    n: int,
    g: int,
    m: int,
    rho_within: float,
    rho_cross: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[np.ndarray]]:
    p = g * m
    groups = build_groups(g, m)
    cov = np.full((p, p), rho_cross, dtype=float)
    np.fill_diagonal(cov, 1.0)
    for idxs in groups:
        for i in idxs:
            for j in idxs:
                if i != j:
                    cov[i, j] = rho_within
    x = _sample_multivariate_normal(n=n, cov=cov, rng=rng)
    x = _standardize_columns_to_sqrt_n(x)
    return x, groups


def generate_weak_identification_design(
    n: int,
    g: int,
    m: int,
    target_group: int,
    rho_within_target: float,
    rho_within_other: float,
    rho_cross: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[np.ndarray]]:
    p = g * m
    groups = build_groups(g, m)
    cov = np.full((p, p), rho_cross, dtype=float)
    np.fill_diagonal(cov, 1.0)
    for group_idx, idxs in enumerate(groups):
        rho_w = rho_within_target if group_idx == target_group else rho_within_other
        for i in idxs:
            for j in idxs:
                if i != j:
                    cov[i, j] = rho_w
    x = _sample_multivariate_normal(n=n, cov=cov, rng=rng)
    x = _standardize_columns_to_sqrt_n(x)
    return x, groups


def _inv_sqrt_spd(mat: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(0.5 * (mat + mat.T))
    vals = np.maximum(vals, 1e-12)
    return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T


@dataclass
class EtaMetrics:
    eta_floor: float
    eta_upper_bound: float
    delta_g: float
    delta_minus_g: float
    rho_g: float
    inflation: float


def compute_eta_floor(
    x: np.ndarray,
    groups: Sequence[np.ndarray],
    target_group: int,
    c_target_sq: float,
    c_out_sq: float,
) -> EtaMetrics:
    idx_g = np.asarray(groups[target_group], dtype=int)
    idx_other = np.concatenate([np.asarray(groups[k], dtype=int) for k in range(len(groups)) if k != target_group])

    xg = x[:, idx_g]
    xo = x[:, idx_other]
    gram_g = xg.T @ xg
    gram_o = xo.T @ xo
    cross = xg.T @ xo

    delta_g = float(np.min(np.linalg.eigvalsh(0.5 * (gram_g + gram_g.T))) + 1.0 / c_target_sq)
    delta_o = float(np.min(np.linalg.eigvalsh(0.5 * (gram_o + gram_o.T))) + 1.0 / c_out_sq)
    rho_g = _operator_norm(cross)

    ub = (rho_g**2) / max(delta_g * delta_o, 1e-12)

    ag = gram_g + (1.0 / c_target_sq) * np.eye(gram_g.shape[0])
    dg = gram_o + (1.0 / c_out_sq) * np.eye(gram_o.shape[0])
    m = _inv_sqrt_spd(ag) @ cross @ _inv_sqrt_spd(dg)
    eta = _operator_norm(m) ** 2
    eta = float(max(0.0, eta))
    inflation = float(np.inf if eta >= 1.0 else 1.0 / (1.0 - eta))

    return EtaMetrics(
        eta_floor=eta,
        eta_upper_bound=float(ub),
        delta_g=delta_g,
        delta_minus_g=delta_o,
        rho_g=float(rho_g),
        inflation=inflation,
    )


def run_experiment_1(
    outdir: Path,
    seed: int,
    quick: bool,
    profile: str,
) -> None:
    rng = np.random.default_rng(seed)
    n = 800 if profile == "adjusted" else 200
    g = 10
    m = 10
    rho_within = 0.95
    rho_cross_grid = [0.0, 0.1, 0.3, 0.5]
    if profile == "adjusted":
        c_grid = [0.01, 0.05, 0.25, 1.0, 4.0, 16.0, 64.0]
    else:
        c_grid = [0.25, 1.0, 4.0, 16.0]
    reps = 30 if quick else 200
    target_group = 0

    rows: list[dict[str, float]] = []
    total = len(rho_cross_grid) * len(c_grid) * len(c_grid) * reps
    done = 0
    for rho_cross in rho_cross_grid:
        for c_target_sq in c_grid:
            for c_out_sq in c_grid:
                for rep in range(reps):
                    x, groups = generate_block_design(
                        n=n,
                        g=g,
                        m=m,
                        rho_within=rho_within,
                        rho_cross=rho_cross,
                        rng=rng,
                    )
                    met = compute_eta_floor(
                        x=x,
                        groups=groups,
                        target_group=target_group,
                        c_target_sq=c_target_sq,
                        c_out_sq=c_out_sq,
                    )
                    rows.append(
                        {
                            "rho_cross": rho_cross,
                            "c_target_sq": c_target_sq,
                            "c_out_sq": c_out_sq,
                            "rep": rep,
                            "eta_floor": met.eta_floor,
                            "eta_upper_bound": met.eta_upper_bound,
                            "delta_g": met.delta_g,
                            "delta_minus_g": met.delta_minus_g,
                            "rho_g": met.rho_g,
                            "inflation": met.inflation,
                        }
                    )
                    done += 1
                    if done % 200 == 0:
                        print(f"[Exp1] {done}/{total}")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "exp1_raw.csv", index=False)

    grp = df.groupby(["rho_cross", "c_target_sq", "c_out_sq"], dropna=False)
    summary = grp.agg(
        eta_median=("eta_floor", "median"),
        eta_q90=("eta_floor", lambda s: float(np.quantile(s.to_numpy(), 0.9))),
        infl_median=("inflation", lambda s: float(np.median(np.clip(s.to_numpy(), 0, 20)))),
        prop_eta_lt_025=("eta_floor", lambda s: float(np.mean(s.to_numpy() < 0.25))),
        prop_eta_lt_05=("eta_floor", lambda s: float(np.mean(s.to_numpy() < 0.5))),
        ub_median=("eta_upper_bound", "median"),
    ).reset_index()
    summary.to_csv(outdir / "exp1_summary.csv", index=False)

    # Heatmaps by c_target
    for c_target_sq in c_grid:
        sub = summary[summary["c_target_sq"] == c_target_sq].copy()
        pivot_eta = sub.pivot(index="c_out_sq", columns="rho_cross", values="eta_median").sort_index().sort_index(axis=1)
        pivot_inf = sub.pivot(index="c_out_sq", columns="rho_cross", values="infl_median").sort_index().sort_index(axis=1)

        fig, ax = plt.subplots(figsize=(6, 4.5))
        im = ax.imshow(pivot_eta.to_numpy(), aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(pivot_eta.columns)))
        ax.set_xticklabels([str(v) for v in pivot_eta.columns])
        ax.set_yticks(range(len(pivot_eta.index)))
        ax.set_yticklabels([str(v) for v in pivot_eta.index])
        ax.set_xlabel("rho_cross")
        ax.set_ylabel("c_out^2")
        ax.set_title(f"Exp1 eta_floor median (c_target^2={c_target_sq})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(outdir / f"exp1_eta_heatmap_ctarget_{c_target_sq}.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4.5))
        im = ax.imshow(pivot_inf.to_numpy(), aspect="auto", origin="lower", cmap="magma")
        ax.set_xticks(range(len(pivot_inf.columns)))
        ax.set_xticklabels([str(v) for v in pivot_inf.columns])
        ax.set_yticks(range(len(pivot_inf.index)))
        ax.set_yticklabels([str(v) for v in pivot_inf.index])
        ax.set_xlabel("rho_cross")
        ax.set_ylabel("c_out^2")
        ax.set_title(f"Exp1 inflation median (c_target^2={c_target_sq})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(outdir / f"exp1_inflation_heatmap_ctarget_{c_target_sq}.png", dpi=180)
        plt.close(fig)

    # Bound tightness scatter
    fig, ax = plt.subplots(figsize=(5.5, 5))
    plot_df = df.copy()
    if len(plot_df) > 5000:
        plot_df = plot_df.sample(5000, random_state=seed)
    ax.scatter(plot_df["eta_floor"], plot_df["eta_upper_bound"], s=8, alpha=0.25, edgecolors="none")
    lo = 0.0
    hi = float(max(plot_df["eta_floor"].max(), plot_df["eta_upper_bound"].max(), 1e-3))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="black")
    ax.set_xlabel("eta_floor")
    ax.set_ylabel("eta_upper_bound")
    ax.set_title("Exp1 bound tightness")
    fig.tight_layout()
    fig.savefig(outdir / "exp1_bound_tightness_scatter.png", dpi=180)
    plt.close(fig)


def compute_bg_star(
    y_g: np.ndarray,
    alpha_c: float,
    beta_c: float,
    u_grid: np.ndarray,
) -> tuple[float, float]:
    pg = float(y_g.size)
    norm2 = float(np.dot(y_g, y_g))
    cdf = invgamma.cdf(u_grid, a=alpha_c, scale=beta_c)
    cdf = np.clip(cdf, 1e-14, 1.0 - 1e-14)
    tail_ratio = (1.0 - cdf) / cdf
    phi_u = u_grid / (1.0 + u_grid)
    # Numerically stable log-space evaluation:
    # B(u) = exp(a) * (term1 + term2), with
    # a = 0.5 * p_g * u
    # term1 = phi(u) * exp(0.5 * ||Y||^2 * phi(u))
    # term2 = exp(0.5 * ||Y||^2) * tail_ratio
    a = 0.5 * pg * u_grid
    log_term1 = np.log(np.maximum(phi_u, 1e-300)) + 0.5 * norm2 * phi_u
    log_term2 = 0.5 * norm2 + np.log(np.maximum(tail_ratio, 1e-300))
    max_log = np.maximum(log_term1, log_term2)
    log_sum = max_log + np.log(np.exp(log_term1 - max_log) + np.exp(log_term2 - max_log))
    log_bg = a + log_sum
    idx = int(np.argmin(log_bg))
    return float(np.exp(log_bg[idx])), float(u_grid[idx])


def run_experiment_2(outdir: Path, seed: int, quick: bool) -> None:
    rng = np.random.default_rng(seed + 1000)
    p_grid = [2, 5, 10, 20, 40]
    prior_grid = [(2.0, 2.0), (2.0, 0.5), (1.0, 1.0), (3.0, 3.0)]
    # Add stronger regularization priors in adjusted profile to probe
    # non-vacuous informative regimes.
    if getattr(run_experiment_2, "_profile", "main") == "adjusted":
        prior_grid = prior_grid + [(4.0, 0.1), (6.0, 0.05)]
    reps = 200 if quick else 1000
    u_grid = np.logspace(-3, 2, 150)

    rows: list[dict[str, float]] = []
    total = len(p_grid) * len(prior_grid) * reps
    done = 0
    for pg in p_grid:
        for alpha_c, beta_c in prior_grid:
            for rep in range(reps):
                y = rng.normal(size=pg)
                bg_star, u_star = compute_bg_star(y_g=y, alpha_c=alpha_c, beta_c=beta_c, u_grid=u_grid)
                rows.append(
                    {
                        "p_g": pg,
                        "alpha_c": alpha_c,
                        "beta_c": beta_c,
                        "rep": rep,
                        "bg_star": bg_star,
                        "u_star": u_star,
                        "non_vacuous_lt1": float(bg_star < 1.0),
                        "inform_lt08": float(bg_star < 0.8),
                        "strong_lt05": float(bg_star < 0.5),
                    }
                )
                done += 1
                if done % 500 == 0:
                    print(f"[Exp2] {done}/{total}")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "exp2_raw.csv", index=False)

    summary = (
        df.groupby(["p_g", "alpha_c", "beta_c"], dropna=False)
        .agg(
            bg_median=("bg_star", "median"),
            bg_q75=("bg_star", lambda s: float(np.quantile(s.to_numpy(), 0.75))),
            prob_lt1=("non_vacuous_lt1", "mean"),
            prob_lt08=("inform_lt08", "mean"),
            prob_lt05=("strong_lt05", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(outdir / "exp2_summary.csv", index=False)

    # Boxplot of B_g*
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    width = 0.18
    prior_labels = [f"({a:g},{b:g})" for a, b in prior_grid]
    x = np.arange(len(p_grid))
    for k, (a, b) in enumerate(prior_grid):
        vals = []
        for pg in p_grid:
            vals.append(df[(df["p_g"] == pg) & (df["alpha_c"] == a) & (df["beta_c"] == b)]["bg_star"].to_numpy())
        positions = x + (k - 1.5) * width
        box = ax.boxplot(vals, positions=positions, widths=width * 0.9, patch_artist=True, showfliers=False)
        color = plt.cm.tab10(k)
        for patch in box["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
        for med in box["medians"]:
            med.set_color(color)
            med.set_linewidth(1.2)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in p_grid])
    ax.set_xlabel("p_g")
    ax.set_ylabel("B_g*")
    ax.set_title("Exp2 distribution of B_g* (log scale)")
    handles = [plt.Line2D([0], [0], color=plt.cm.tab10(i), lw=2, label=f"IG{prior_labels[i]}") for i in range(len(prior_grid))]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "exp2_bgstar_boxplot.png", dpi=180)
    plt.close(fig)

    # Non-vacuousness probabilities
    for metric, title, fname in [
        ("prob_lt1", "P(B_g* < 1.0)", "exp2_prob_lt1.png"),
        ("prob_lt08", "P(B_g* < 0.8)", "exp2_prob_lt08.png"),
        ("prob_lt05", "P(B_g* < 0.5)", "exp2_prob_lt05.png"),
    ]:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for k, (a, b) in enumerate(prior_grid):
            sub = summary[(summary["alpha_c"] == a) & (summary["beta_c"] == b)].sort_values("p_g")
            ax.plot(sub["p_g"], sub[metric], marker="o", linewidth=1.8, label=f"IG({a:g},{b:g})")
        ax.set_xlabel("p_g")
        ax.set_ylabel("Probability")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"Exp2 {title}")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=180)
        plt.close(fig)


def run_experiment_3(outdir: Path, seed: int, quick: bool) -> None:
    rng = np.random.default_rng(seed + 2000)
    n = 100
    g = 5
    m = 10
    target_group = 0
    rho_within_target = 0.95
    rho_within_other = 0.5
    rho_cross = 0.1
    reps = 40 if quick else 200
    if getattr(run_experiment_3, "_profile", "main") == "adjusted":
        c_grid = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1.0, 5.0, 20.0, 100.0], dtype=float)
    else:
        c_grid = np.array([0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 20.0, 100.0], dtype=float)
    c2_grid = c_grid  # already c_g^2

    rows: list[dict[str, float]] = []
    total = reps * len(c2_grid)
    done = 0
    for rep in range(reps):
        x, groups = generate_weak_identification_design(
            n=n,
            g=g,
            m=m,
            target_group=target_group,
            rho_within_target=rho_within_target,
            rho_within_other=rho_within_other,
            rho_cross=rho_cross,
            rng=rng,
        )
        p = x.shape[1]
        beta0 = np.zeros(p, dtype=float)
        idx_g = groups[target_group]
        beta0[idx_g[:3]] = np.array([2.0, 1.5, 1.0], dtype=float)
        y = x @ beta0 + rng.normal(size=n)

        xg = x[:, idx_g]
        gram = xg.T @ xg
        eigvals, eigvecs = np.linalg.eigh(0.5 * (gram + gram.T))
        umin = eigvecs[:, int(np.argmin(eigvals))]
        lam_min = float(np.min(eigvals))
        p_g = gram.shape[0]
        i_pg = np.eye(p_g)

        p_mat = gram + i_pg
        b_vec = xg.T @ y
        m_inf = np.linalg.solve(p_mat, b_vec)

        for c2 in c2_grid:
            sigma_g = np.linalg.inv(gram + (1.0 + 1.0 / c2) * i_pg)
            lam_max = float(np.max(np.linalg.eigvalsh(0.5 * (sigma_g + sigma_g.T))))
            weak_var = float(umin.T @ sigma_g @ umin)
            ceiling_dir = float(1.0 / (lam_min + 1.0 / c2))

            m_c = np.linalg.solve(p_mat + (1.0 / c2) * i_pg, b_vec)
            mean_norm = float(np.linalg.norm(m_c))
            mean_diff = float(np.linalg.norm(m_c - m_inf))

            rows.append(
                {
                    "rep": rep,
                    "c2": float(c2),
                    "lambda_max_sigma_g": lam_max,
                    "weak_dir_var": weak_var,
                    "ceiling_dir": ceiling_dir,
                    "c2_ceiling": float(c2),
                    "mean_norm": mean_norm,
                    "mean_diff_to_infty": mean_diff,
                }
            )
            done += 1
            if done % 200 == 0:
                print(f"[Exp3] {done}/{total}")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "exp3_raw.csv", index=False)
    summary = (
        df.groupby("c2", dropna=False)
        .agg(
            lambda_max_med=("lambda_max_sigma_g", "median"),
            weak_var_med=("weak_dir_var", "median"),
            ceiling_dir_med=("ceiling_dir", "median"),
            c2_med=("c2_ceiling", "median"),
            mean_norm_med=("mean_norm", "median"),
            mean_diff_med=("mean_diff_to_infty", "median"),
        )
        .reset_index()
    )
    summary.to_csv(outdir / "exp3_summary.csv", index=False)

    # Fig 6
    fig, ax = plt.subplots(figsize=(6.3, 4.6))
    ax.plot(summary["c2"], summary["lambda_max_med"], marker="o", label="median lambda_max(Sigma_g)")
    ax.plot(summary["c2"], summary["ceiling_dir_med"], marker="s", label="median directional ceiling")
    ax.plot(summary["c2"], summary["c2_med"], linestyle="--", label="c_g^2 ceiling")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("c_g^2")
    ax.set_ylabel("Scale")
    ax.set_title("Exp3 blockwise covariance ceilings")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "exp3_covariance_ceilings.png", dpi=180)
    plt.close(fig)

    # Fig 7
    fig, ax = plt.subplots(figsize=(6.3, 4.6))
    ax.plot(summary["c2"], summary["weak_var_med"], marker="o")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("c_g^2")
    ax.set_ylabel("median weak-direction variance")
    ax.set_title("Exp3 weak-direction variance")
    fig.tight_layout()
    fig.savefig(outdir / "exp3_weak_direction_variance.png", dpi=180)
    plt.close(fig)

    # Fig 8
    fig, ax = plt.subplots(figsize=(6.3, 4.6))
    ax.plot(summary["c2"], summary["mean_norm_med"], marker="o")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("c_g^2")
    ax.set_ylabel("median ||m_g(c_g^2)||_2")
    ax.set_title("Exp3 blockwise mean magnitude")
    fig.tight_layout()
    fig.savefig(outdir / "exp3_mean_norm_curve.png", dpi=180)
    plt.close(fig)

    # Fig 9
    fig, ax = plt.subplots(figsize=(6.3, 4.6))
    ax.plot(summary["c2"], summary["mean_diff_med"], marker="o")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("c_g^2")
    ax.set_ylabel("median ||m_g(c_g^2)-m_g^(inf)||_2")
    ax.set_title("Exp3 approach to large-slab limit")
    fig.tight_layout()
    fig.savefig(outdir / "exp3_mean_diff_curve.png", dpi=180)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GR-RHS theory-aligned simulations (Exp1-Exp3).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: outputs/simulations/grrhs_theory_<timestamp>).",
    )
    parser.add_argument("--seed", type=int, default=20260408, help="Master seed.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick debug mode with reduced repetitions.",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default="1,2,3",
        help="Comma-separated subset, e.g. 1,2 or 3.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="main",
        choices=["main", "adjusted"],
        help="Simulation profile. 'adjusted' sharpens informative-regime visibility.",
    )
    return parser.parse_args()


def _parse_experiments(spec: str) -> set[int]:
    out: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        val = int(token)
        if val not in {1, 2, 3}:
            raise ValueError(f"Unsupported experiment id: {val}")
        out.add(val)
    if not out:
        raise ValueError("At least one experiment must be selected.")
    return out


def main() -> None:
    args = _parse_args()
    chosen = _parse_experiments(args.experiments)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = args.output_dir or Path("outputs") / "simulations" / f"grrhs_theory_{stamp}"
    _ensure_dir(base)

    (base / "meta.txt").write_text(
        f"seed={args.seed}\nquick={args.quick}\nexperiments={sorted(chosen)}\n",
        encoding="utf-8",
    )
    print(f"Output: {base}")

    run_experiment_2._profile = args.profile  # type: ignore[attr-defined]
    run_experiment_3._profile = args.profile  # type: ignore[attr-defined]

    if 1 in chosen:
        out1 = base / "experiment_1"
        _ensure_dir(out1)
        run_experiment_1(outdir=out1, seed=args.seed, quick=args.quick, profile=args.profile)
    if 2 in chosen:
        out2 = base / "experiment_2"
        _ensure_dir(out2)
        run_experiment_2(outdir=out2, seed=args.seed, quick=args.quick)
    if 3 in chosen:
        out3 = base / "experiment_3"
        _ensure_dir(out3)
        run_experiment_3(outdir=out3, seed=args.seed, quick=args.quick)

    print("Done.")


if __name__ == "__main__":
    main()
