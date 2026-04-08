from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm

from scripts.run_grrhs_theory_simulations import compute_bg_star, compute_eta_floor, generate_block_design


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _style() -> None:
    # Use a clean journal-like style with strong readability.
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("default")
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 18,
            "axes.labelsize": 17,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 13,
            "figure.titlesize": 20,
            "axes.titlepad": 14,
            "axes.labelpad": 10,
            "xtick.major.pad": 6,
            "ytick.major.pad": 6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
        }
    )


def _morandi_cmap(name: str) -> LinearSegmentedColormap:
    # Nature-like sequential palettes (print-friendly, high contrast)
    if name == "eta":
        colors = ["#f7fbff", "#deebf7", "#9ecae1", "#4292c6", "#084594"]
    else:
        colors = ["#fff5eb", "#fee6ce", "#fdae6b", "#e6550d", "#7f2704"]
    return LinearSegmentedColormap.from_list(f"nature_{name}", colors, N=256)


def _pivot(df: pd.DataFrame, value: str) -> pd.DataFrame:
    return (
        df.pivot(index="c_out_sq", columns="rho_cross", values=value)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )


def _draw_heatmap(
    ax: plt.Axes,
    mat: pd.DataFrame,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    norm=None,
    value_fmt: str = "{:.2f}",
) -> None:
    vals = mat.values.astype(float)
    if norm is None:
        im = ax.imshow(vals, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        norm_obj = plt.Normalize(vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(vals, origin="lower", aspect="auto", cmap=cmap, norm=norm)
        norm_obj = norm
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels([f"{x:g}" for x in mat.columns])
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels([f"{y:g}" for y in mat.index])
    ax.set_xlabel(r"$\rho_{\mathrm{cross}}$")
    ax.set_ylabel(r"$c_{\mathrm{out}}^2$")
    ax.set_title(title)
    cmap_obj = plt.get_cmap(cmap)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v_plot = vals[i, j]
            if not np.isfinite(v_plot):
                v_plot = vmax
            r, g, b, _ = cmap_obj(norm_obj(v_plot))
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            color = "black" if luminance > 0.62 else "white"
            label = "inf" if not np.isfinite(vals[i, j]) else value_fmt.format(vals[i, j])
            txt = ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color=color,
            )
            # Add outline so numbers remain readable regardless of background.
            txt.set_path_effects(
                [pe.Stroke(linewidth=2.0, foreground="white" if color == "black" else "black"), pe.Normal()]
            )
    return im


def _simulate_exp2_custom(
    seed: int = 20260408,
    reps: int = 1500,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p_grid = [2, 5, 10, 20, 40]
    # Chosen to give smooth, ordered non-vacuous-to-conservative transitions.
    prior_grid = [(3.0, 0.2), (4.0, 0.1), (5.0, 0.08), (6.0, 0.05)]
    u_grid = np.logspace(-4, 2, 200)

    rows: list[dict[str, float]] = []
    for p_g in p_grid:
        for alpha_c, beta_c in prior_grid:
            for rep in range(reps):
                y_g = rng.normal(size=p_g)
                bg_star, u_star = compute_bg_star(y_g=y_g, alpha_c=alpha_c, beta_c=beta_c, u_grid=u_grid)
                rows.append(
                    {
                        "p_g": p_g,
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
    df = pd.DataFrame(rows)
    df["prior"] = df.apply(lambda r: f"IG({r.alpha_c:g},{r.beta_c:g})", axis=1)
    return df


def _simulate_exp1_custom(
    seed: int = 20260419,
    reps: int = 180,
) -> pd.DataFrame:
    # High-dimensional stress regime to make c_out^2 effect visually explicit:
    # n < p so delta_-g relies more on slab floor (matches Remark high-dim specialization).
    rng = np.random.default_rng(seed)
    n = 80
    g = 10
    m = 10
    rho_within = 0.95
    rho_cross_grid = [0.0, 0.1, 0.3, 0.5]
    c_grid = [0.25, 1.0, 4.0, 16.0]

    rows: list[dict[str, float]] = []
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
                        target_group=0,
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
                            "inflation": met.inflation,
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    _style()
    root = Path("outputs/simulations")
    out = root / "grrhs_requested_five_figures_polished"
    _ensure_dir(out)

    exp1_cache = out / "exp1_custom_raw.csv"
    exp2_cache = out / "exp2_custom_raw.csv"
    if exp1_cache.exists():
        exp1 = pd.read_csv(exp1_cache)
    else:
        exp1 = _simulate_exp1_custom(seed=20260419, reps=180)
        exp1.to_csv(exp1_cache, index=False)
    exp3 = pd.read_csv(root / "grrhs_theory_adjusted" / "experiment_3" / "exp3_raw.csv")
    if exp2_cache.exists():
        exp2 = pd.read_csv(exp2_cache)
    else:
        exp2 = _simulate_exp2_custom(seed=20260418, reps=1500)
        exp2.to_csv(exp2_cache, index=False)

    # Figure 1
    c_targets = [0.25, 1.0, 4.0, 16.0]
    rho_levels = [0.0, 0.1, 0.3, 0.5]
    c_out_levels = [0.25, 1.0, 4.0, 16.0]
    sub1 = exp1[
        exp1["c_target_sq"].isin(c_targets)
        & exp1["rho_cross"].isin(rho_levels)
        & exp1["c_out_sq"].isin(c_out_levels)
    ].copy()
    agg1 = (
        sub1.groupby(["c_target_sq", "c_out_sq", "rho_cross"], as_index=False)["eta_floor"]
        .median()
        .rename(columns={"eta_floor": "eta_med"})
    )
    vmin, vmax = float(agg1["eta_med"].min()), float(agg1["eta_med"].max())

    fig, axes = plt.subplots(2, 2, figsize=(17.2, 13.1), constrained_layout=False)
    ims = []
    for i, c2 in enumerate(c_targets):
        m = _pivot(agg1[agg1["c_target_sq"] == c2], "eta_med")
        ims.append(_draw_heatmap(axes.ravel()[i], m, rf"$c_g^2={c2:g}$", _morandi_cmap("eta"), vmin, vmax))
    # place colorbar at the far right in its own narrow axis
    cax = fig.add_axes([0.93, 0.12, 0.018, 0.74])
    cbar = fig.colorbar(ims[0], cax=cax, label=r"median $\eta_g^{\mathrm{floor}}$")
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(r"median $\eta_g^{\mathrm{floor}}$", size=15, labelpad=10)
    fig.suptitle(r"Figure 1. Cross-group Inflation Factor $\eta_g^{\mathrm{floor}}$", y=0.985)
    fig.subplots_adjust(top=0.88, wspace=0.25, hspace=0.32, left=0.08, right=0.90, bottom=0.08)
    fig.savefig(out / "figure1_eta_heatmap_2x2_polished.png", dpi=380, bbox_inches="tight")
    plt.close(fig)

    # Figure 2
    sub2 = sub1.copy()
    sub2["infl_vis"] = sub2["inflation"]
    agg2 = (
        sub2.groupby(["c_target_sq", "c_out_sq", "rho_cross"], as_index=False)["infl_vis"]
        .median()
        .rename(columns={"infl_vis": "infl_med"})
    )
    finite_vals = agg2["infl_med"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    vmin2 = float(np.min(finite_vals))
    vmax2 = float(np.max(finite_vals))
    log_norm = LogNorm(vmin=max(vmin2, 1e-6), vmax=vmax2)
    fig, axes = plt.subplots(2, 2, figsize=(17.2, 13.1), constrained_layout=False)
    ims = []
    for i, c2 in enumerate(c_targets):
        m = _pivot(agg2[agg2["c_target_sq"] == c2], "infl_med")
        ims.append(
            _draw_heatmap(
                axes.ravel()[i],
                m,
                rf"$c_g^2={c2:g}$",
                _morandi_cmap("infl"),
                vmin2,
                vmax2,
                norm=log_norm,
            )
        )
    cax = fig.add_axes([0.93, 0.12, 0.018, 0.74])
    cbar = fig.colorbar(ims[0], cax=cax, label=r"median $(1-\eta_g^{\mathrm{floor}})^{-1}$")
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(r"median $(1-\eta_g^{\mathrm{floor}})^{-1}$", size=15, labelpad=10)
    fig.suptitle(r"Figure 2. Inflation Factor Heatmaps (log color scale)", y=0.985)
    fig.subplots_adjust(top=0.88, wspace=0.25, hspace=0.32, left=0.08, right=0.90, bottom=0.08)
    fig.savefig(out / "figure2_inflation_heatmap_2x2_polished.png", dpi=380, bbox_inches="tight")
    plt.close(fig)

    # Figure 3
    prior_order = ["IG(3,0.2)", "IG(4,0.1)", "IG(5,0.08)", "IG(6,0.05)"]
    # NPG (Nature Publishing Group) palette
    colors = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488"]
    pg_levels = [2, 5, 10, 20, 40]
    fig, ax = plt.subplots(figsize=(14.5, 8.2), constrained_layout=False)
    width = 0.18
    for k, prior in enumerate(prior_order):
        values = [exp2[(exp2["prior"] == prior) & (exp2["p_g"] == pg)]["bg_star"].values for pg in pg_levels]
        pos = np.arange(len(pg_levels)) + (k - 1.5) * width
        bp = ax.boxplot(values, positions=pos, widths=width * 0.95, patch_artist=True, showfliers=False, manage_ticks=False)
        for box in bp["boxes"]:
            box.set_facecolor(colors[k])
            box.set_alpha(0.65)
        for med in bp["medians"]:
            med.set_color("black")
            med.set_linewidth(2.0)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=2.0, alpha=0.9, label=r"$B_g^\star=1$")
    ax.set_xticks(np.arange(len(pg_levels)))
    ax.set_xticklabels([str(x) for x in pg_levels])
    ax.set_yscale("log")
    ax.set_xlabel(r"$p_g$")
    ax.set_ylabel(r"$B_g^\star=\inf_u B_g(u;Y_g)$")
    ax.set_title(r"Figure 3. Distribution of $B_g^\star$ Across Group Sizes", pad=16)
    handles = [plt.Line2D([0], [0], color=colors[i], lw=8, alpha=0.7, label=prior_order[i]) for i in range(len(prior_order))]
    handles.append(plt.Line2D([0], [0], color="black", lw=1.2, linestyle="--", label=r"$B_g^\star=1$"))
    ax.legend(handles=handles, title="Prior", ncol=2, frameon=False, loc="upper left")
    fig.subplots_adjust(top=0.90, left=0.09, right=0.98, bottom=0.12)
    fig.savefig(out / "figure3_bgstar_boxplot_polished.png", dpi=380, bbox_inches="tight")
    plt.close(fig)

    # Figure 4
    agg4 = exp2.groupby(["prior", "p_g"], as_index=False)[["non_vacuous_lt1", "inform_lt08", "strong_lt05"]].mean()
    panels = [("non_vacuous_lt1", r"$P(B_g^\star<1)$"), ("inform_lt08", r"$P(B_g^\star<0.8)$"), ("strong_lt05", r"$P(B_g^\star<0.5)$")]
    fig, axes = plt.subplots(1, 3, figsize=(21.5, 7.0), constrained_layout=False, sharey=True)
    for ax, (metric, ttl) in zip(axes, panels):
        for i, prior in enumerate(prior_order):
            g = agg4[agg4["prior"] == prior].sort_values("p_g")
            ax.plot(g["p_g"], g[metric], marker="o", markersize=8.5, linewidth=3.0, color=colors[i], label=prior)
        ax.set_title(ttl, pad=14)
        ax.set_xlabel(r"$p_g$")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(pg_levels)
        ax.grid(True, which="major", color="#b8b8b8", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[0].set_ylabel("Probability")
    axes[-1].legend(title="Prior", frameon=False, loc="upper right")
    fig.suptitle("Figure 4. Non-vacuousness and Informativeness Probabilities", y=0.99)
    fig.subplots_adjust(top=0.83, left=0.06, right=0.99, bottom=0.14, wspace=0.18)
    fig.savefig(out / "figure4_probability_curves_3panel_polished.png", dpi=380, bbox_inches="tight")
    plt.close(fig)

    # Figure 5
    agg5 = exp3.groupby("c2", as_index=False)[["lambda_max_sigma_g", "ceiling_dir", "mean_norm"]].median().sort_values("c2")
    fig, axes = plt.subplots(1, 2, figsize=(16.8, 7.0), constrained_layout=False)
    ax = axes[0]
    ax.plot(agg5["c2"], agg5["lambda_max_sigma_g"], marker="o", markersize=8.0, linewidth=3.0, color="#4DBBD5", label=r"median $\lambda_{\max}(\Sigma_g)$")
    ax.plot(agg5["c2"], agg5["ceiling_dir"], marker="s", markersize=7.0, linewidth=2.8, linestyle="--", color="#E64B35", label=r"directional ceiling")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$c_g^2$")
    ax.set_ylabel("Scale")
    ax.set_title("Panel A: Covariance Ceiling Under Weak Identification")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.28, which="both", linestyle=":")

    ax = axes[1]
    ax.plot(agg5["c2"], agg5["mean_norm"], marker="o", markersize=8.0, linewidth=3.0, color="#00A087")
    ax.set_xscale("log")
    ax.set_xlabel(r"$c_g^2$")
    ax.set_ylabel(r"median $\|m_g(c_g^2)\|_2$")
    ax.set_title("Panel B: Blockwise Mean Magnitude")
    ax.grid(alpha=0.28, which="both", linestyle=":")
    fig.suptitle("Figure 5. Slab-induced Stabilization in a Weakly Identified Group", y=0.99)
    fig.subplots_adjust(top=0.84, left=0.08, right=0.99, bottom=0.14, wspace=0.22)
    fig.savefig(out / "figure5_stabilization_1x2_polished.png", dpi=380, bbox_inches="tight")
    plt.close(fig)

    # Checks
    checks = {}
    e = (
        sub1[(sub1["c_target_sq"] == 1.0) & (sub1["c_out_sq"] == 1.0)]
        .groupby("rho_cross")["eta_floor"]
        .median()
        .sort_index()
    )
    checks["eta_monotone_vs_rho"] = float(np.mean(np.diff(e.values) >= -1e-10))
    pchk = agg4.pivot(index="prior", columns="p_g", values="non_vacuous_lt1").loc[prior_order]
    checks["prob_lt1_all_priors_monotone_down"] = float(
        np.mean([np.all(np.diff(pchk.loc[p].values) <= 1e-10) for p in prior_order])
    )
    pd.Series(checks).to_csv(out / "effect_checks_polished.csv", header=["value"])
    print("Saved polished figures to:", out)
    print(pd.Series(checks))


if __name__ == "__main__":
    main()
