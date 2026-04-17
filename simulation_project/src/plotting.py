from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _records(df: Any) -> list[dict[str, Any]]:
    if isinstance(df, list):
        return [dict(r) for r in df]
    if hasattr(df, "to_dict"):
        try:
            return list(df.to_dict(orient="records"))
        except TypeError:
            pass
    return [dict(r) for r in df]


def plot_exp1(df: Any, slope: float, slope_ci: tuple[float, float], out_path: Path) -> None:
    rows = sorted(_records(df), key=lambda r: float(r["p_g"]))
    p_g = np.asarray([float(r["p_g"]) for r in rows], dtype=float)
    med = np.asarray([float(r["median_post_mean_kappa"]) for r in rows], dtype=float)
    tail = np.asarray([float(r["mean_tail_prob"]) for r in rows], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.log(p_g)
    y = np.log(med)
    axes[0].plot(x, y, "o-")
    coef = np.polyfit(x, y, deg=1)
    axes[0].plot(x, coef[0] * x + coef[1], "--", color="black")
    axes[0].set_xlabel("log p_g")
    axes[0].set_ylabel("log median E[kappa|Y]")
    axes[0].set_title(f"Slope={slope:.3f} [{slope_ci[0]:.3f},{slope_ci[1]:.3f}]")

    axes[1].plot(p_g, tail, "o-")
    axes[1].set_xlabel("p_g")
    axes[1].set_ylabel("Mean P(kappa>2/sqrt(p_g)|Y)")
    _save(fig, out_path)


def plot_exp2(df: Any, out_path: Path) -> None:
    rows = sorted(_records(df), key=lambda r: float(r["p_g"]))
    x = np.asarray([float(r["p_g"]) for r in rows], dtype=float)
    m = np.asarray([float(r["median_ratio_R"]) for r in rows], dtype=float)
    lo = np.asarray([float(r["iqr_ratio_R_low"]) for r in rows], dtype=float)
    hi = np.asarray([float(r["iqr_ratio_R_high"]) for r in rows], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(x, m, "o-")
    axes[0].fill_between(x, lo, hi, alpha=0.25)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("p_g")
    axes[0].set_ylabel("Median R")

    axes[1].plot(x, np.asarray([float(r["mean_window_prob"]) for r in rows], dtype=float), "o-")
    axes[1].set_xlabel("p_g")
    axes[1].set_ylabel("Mean window prob")
    _save(fig, out_path)


def plot_exp3_heatmap(df: Any, out_path: Path) -> None:
    rows = _records(df)
    p_vals = sorted({float(r["p_g"]) for r in rows})
    x_vals = sorted({float(r["xi"]) for r in rows})
    mat = np.full((len(p_vals), len(x_vals)), np.nan, dtype=float)
    p_idx = {v: i for i, v in enumerate(p_vals)}
    x_idx = {v: i for i, v in enumerate(x_vals)}
    for r in rows:
        mat[p_idx[float(r["p_g"])], x_idx[float(r["xi"])]] = float(r["mean_prob_gt_u0"])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(np.arange(len(x_vals)), labels=[f"{v:.3g}" for v in x_vals])
    ax.set_yticks(np.arange(len(p_vals)), labels=[str(int(v)) for v in p_vals])
    ax.set_xlabel("xi")
    ax.set_ylabel("p_g")
    fig.colorbar(im, ax=ax, label="Mean P(kappa>u0|Y)")
    _save(fig, out_path)


def plot_exp3_curves(df: Any, xi_crit: float, out_path: Path) -> None:
    rows = _records(df)
    p_vals = sorted({float(r["p_g"]) for r in rows})
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for pg in p_vals:
        sub = sorted((r for r in rows if float(r["p_g"]) == pg), key=lambda r: float(r["xi"]))
        ax.plot([float(r["xi"]) for r in sub], [float(r["mean_prob_gt_u0"]) for r in sub], marker="o", label=f"p_g={int(pg)}")
    ax.axvline(float(xi_crit), color="black", linestyle="--", label="xi_crit")
    ax.set_xscale("log")
    ax.set_xlabel("xi")
    ax.set_ylabel("Mean P(kappa>u0|Y)")
    ax.legend()
    _save(fig, out_path)


def plot_exp4_overall_mse(df: Any, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for method, sub in df.groupby("method"):
        s = sub.sort_values("setting")
        ax.plot(s["setting"], s["mse_overall"], marker="o", label=method)
    ax.set_xlabel("Setting")
    ax.set_ylabel("Overall MSE")
    ax.legend()
    _save(fig, out_path)


def plot_exp5_kappa_stratification(df: Any, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    groups = sorted(df["group_id"].unique())
    data = [df.loc[df["group_id"] == g, "post_mean_kappa_g"].dropna().to_numpy() for g in groups]
    ax.boxplot(data, positions=np.arange(1, len(groups) + 1))
    ax.set_xticks(np.arange(1, len(groups) + 1), labels=[str(int(g) + 1) for g in groups])
    ax.set_xlabel("Group")
    ax.set_ylabel("Posterior mean kappa_g")
    _save(fig, out_path)


def plot_exp5_null_signal_mse(df: Any, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    xpos = np.arange(len(df))
    w = 0.35
    ax.bar(xpos - w / 2, df["avg_null_group_mse"], width=w, label="null")
    ax.bar(xpos + w / 2, df["avg_signal_group_mse"], width=w, label="signal")
    ax.set_xticks(xpos, labels=df["method"].tolist())
    ax.set_ylabel("Group MSE")
    ax.legend()
    _save(fig, out_path)


def plot_exp5_group_ranking(df_kappa: Any, df_auroc: Any, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    agg = df_kappa.groupby("mu_g", as_index=False)["post_mean_kappa_g"].mean()
    axes[0].plot(agg["mu_g"], agg["post_mean_kappa_g"], "o-")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("mu_g")
    axes[0].set_ylabel("Avg E[kappa_g|Y]")

    axes[1].bar(df_auroc["method"], df_auroc["group_auroc"])
    axes[1].set_ylabel("Group AUROC")
    _save(fig, out_path)


def plot_exp6_coefficients(df: Any, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for method, sub in df.groupby("method"):
        ax.scatter(sub["beta11_post_mean"], sub["beta12_post_mean"], alpha=0.6, label=method)
    ax.set_xlabel("beta11 posterior mean")
    ax.set_ylabel("beta12 posterior mean")
    ax.legend()
    _save(fig, out_path)


def plot_exp6_null_group(df: Any, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    vals = [sub["beta_group2_l2_norm"].dropna().to_numpy() for _, sub in df.groupby("method")]
    labels = [m for m, _ in df.groupby("method")]
    ax.boxplot(vals, labels=labels)
    ax.set_ylabel("||beta_group2||_2")
    _save(fig, out_path)


def plot_exp6_diagnostics(df: Any, out_path: Path) -> None:
    agg = df.groupby("method", as_index=False)[["divergence_ratio", "overall_runtime"]].mean(numeric_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(agg["method"], agg["divergence_ratio"])
    axes[0].set_ylabel("Divergence ratio")
    axes[1].bar(agg["method"], agg["overall_runtime"])
    axes[1].set_ylabel("Runtime (s)")
    _save(fig, out_path)


def plot_exp6_kappa(df: Any, out_path: Path) -> None:
    gr = df.loc[df["method"] == "GR_RHS"].copy()
    agg = gr[["post_prob_kappa_group1_gt_0_5", "post_prob_kappa_group2_gt_0_5"]].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["group1", "group2"], [agg.iloc[0], agg.iloc[1]])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Posterior prob")
    _save(fig, out_path)


def plot_exp8_tau(df: Any, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for name, sub in df.groupby("tau_prior"):
        vals = sub["m_eff"].to_numpy()
        ax.hist(vals, bins=40, density=True, alpha=0.35, label=name)
    ax.set_xlabel("m_eff")
    ax.set_ylabel("Density")
    ax.legend()
    _save(fig, out_path)
