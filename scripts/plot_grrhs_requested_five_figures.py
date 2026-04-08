from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def _prior_label(alpha: float, beta: float) -> str:
    return f"IG({alpha:g},{beta:g})"


def _pivot_grid(
    df: pd.DataFrame,
    value_col: str,
    index_col: str = "c_out_sq",
    columns_col: str = "rho_cross",
) -> pd.DataFrame:
    out = df.pivot(index=index_col, columns=columns_col, values=value_col)
    out = out.sort_index(axis=0).sort_index(axis=1)
    return out


def _plot_heatmap_panel(
    ax: plt.Axes,
    mat: pd.DataFrame,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
) -> None:
    vals = mat.values.astype(float)
    im = ax.imshow(vals, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels([f"{x:g}" for x in mat.columns])
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels([f"{y:g}" for y in mat.index])
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(
                j,
                i,
                f"{vals[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if vals[i, j] > (0.6 * (vmax if vmax is not None else vals.max())) else "black",
            )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel(r"$\rho_{\mathrm{cross}}$")
    ax.set_ylabel(r"$c_{\mathrm{out}}^2$")
    ax.grid(False)


def make_fig1_eta_heatmaps(exp1_raw: pd.DataFrame, out_path: Path) -> None:
    c_targets = [0.25, 1.0, 4.0, 16.0]
    rho_levels = [0.0, 0.1, 0.3, 0.5]
    cout_levels = [0.25, 1.0, 4.0, 16.0]

    sub = exp1_raw[
        exp1_raw["c_target_sq"].isin(c_targets)
        & exp1_raw["rho_cross"].isin(rho_levels)
        & exp1_raw["c_out_sq"].isin(cout_levels)
    ].copy()
    agg = (
        sub.groupby(["c_target_sq", "c_out_sq", "rho_cross"], as_index=False)["eta_floor"]
        .median()
        .rename(columns={"eta_floor": "eta_median"})
    )

    vmin = float(agg["eta_median"].min())
    vmax = float(agg["eta_median"].max())

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for i, c2 in enumerate(c_targets):
        ax = axes.ravel()[i]
        mat = _pivot_grid(agg[agg["c_target_sq"] == c2], "eta_median")
        _plot_heatmap_panel(
            ax,
            mat,
            title=rf"$c_g^2={c2:g}$",
            vmin=vmin,
            vmax=vmax,
            cmap="YlGnBu",
        )

    fig.suptitle(r"Figure 1: Median $\eta_g^{\mathrm{floor}}$ Heatmaps", fontsize=14)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_fig2_inflation_heatmaps(exp1_raw: pd.DataFrame, out_path: Path, cap: float = 10.0) -> None:
    c_targets = [0.25, 1.0, 4.0, 16.0]
    rho_levels = [0.0, 0.1, 0.3, 0.5]
    cout_levels = [0.25, 1.0, 4.0, 16.0]

    sub = exp1_raw[
        exp1_raw["c_target_sq"].isin(c_targets)
        & exp1_raw["rho_cross"].isin(rho_levels)
        & exp1_raw["c_out_sq"].isin(cout_levels)
    ].copy()
    sub["inflation_capped"] = np.minimum(sub["inflation"], cap)
    agg = (
        sub.groupby(["c_target_sq", "c_out_sq", "rho_cross"], as_index=False)["inflation_capped"]
        .median()
        .rename(columns={"inflation_capped": "infl_median"})
    )

    vmin = float(agg["infl_median"].min())
    vmax = float(agg["infl_median"].max())

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for i, c2 in enumerate(c_targets):
        ax = axes.ravel()[i]
        mat = _pivot_grid(agg[agg["c_target_sq"] == c2], "infl_median")
        _plot_heatmap_panel(
            ax,
            mat,
            title=rf"$c_g^2={c2:g}$",
            vmin=vmin,
            vmax=vmax,
            cmap="magma",
        )

    fig.suptitle(
        rf"Figure 2: Median Inflation $(1-\eta_g^{{\mathrm{{floor}}}})^{{-1}}$ Heatmaps (cap={cap:g})",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_fig3_bgstar_boxplot(exp2_raw: pd.DataFrame, out_path: Path) -> tuple[pd.DataFrame, list[tuple[float, float]]]:
    # Keep four priors, including two baseline and two regularizing priors,
    # to ensure visible informative/non-vacuous regimes.
    priors = [(2.0, 2.0), (2.0, 0.5), (4.0, 0.1), (6.0, 0.05)]
    mask = pd.Series(False, index=exp2_raw.index)
    for a, b in priors:
        mask = mask | ((exp2_raw["alpha_c"] == a) & (exp2_raw["beta_c"] == b))
    sub = exp2_raw[mask].copy()
    sub["prior"] = [_prior_label(a, b) for a, b in zip(sub["alpha_c"], sub["beta_c"])]
    sub["p_g"] = sub["p_g"].astype(int)

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    pg_levels = sorted(sub["p_g"].unique())
    prior_levels = sorted(sub["prior"].unique())
    colors = ["#355070", "#6d597a", "#b56576", "#e56b6f"]
    width = 0.18
    for k, prior in enumerate(prior_levels):
        data = [sub[(sub["p_g"] == pg) & (sub["prior"] == prior)]["bg_star"].values for pg in pg_levels]
        positions = np.arange(len(pg_levels)) + (k - 1.5) * width
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.95,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[k % len(colors)])
            patch.set_alpha(0.6)
        for m in bp["medians"]:
            m.set_color("black")
    ax.set_xticks(np.arange(len(pg_levels)))
    ax.set_xticklabels([str(x) for x in pg_levels])
    ax.set_yscale("log")
    ax.set_xlabel(r"$p_g$")
    ax.set_ylabel(r"$B_g^\star=\inf_u B_g(u;Y_g)$")
    ax.set_title(r"Figure 3: $B_g^\star$ by Group Size and Slab Prior")
    legend_handles = [
        plt.Line2D([0], [0], color=colors[i], lw=8, alpha=0.6, label=prior_levels[i]) for i in range(len(prior_levels))
    ]
    ax.legend(handles=legend_handles, title="Prior", ncol=2, fontsize=9)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return sub, priors


def make_fig4_probability_curves(exp2_sub: pd.DataFrame, out_path: Path) -> None:
    thresholds = [
        ("non_vacuous_lt1", r"$P(B_g^\star<1)$"),
        ("inform_lt08", r"$P(B_g^\star<0.8)$"),
        ("strong_lt05", r"$P(B_g^\star<0.5)$"),
    ]
    agg = (
        exp2_sub.groupby(["prior", "p_g"], as_index=False)[[t[0] for t in thresholds]]
        .mean()
        .sort_values(["prior", "p_g"])
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True, sharey=True)
    prior_levels = sorted(agg["prior"].unique())
    colors = ["#355070", "#6d597a", "#b56576", "#e56b6f"]
    for ax, (col, title) in zip(axes, thresholds):
        for idx, prior in enumerate(prior_levels):
            grp = agg[agg["prior"] == prior]
            ax.plot(grp["p_g"], grp[col], marker="o", linewidth=2, label=prior, color=colors[idx % len(colors)])
        ax.set_title(title)
        ax.set_xlabel(r"$p_g$")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Probability")
    axes[-1].legend(title="Prior", fontsize=8)
    fig.suptitle(r"Figure 4: Non-vacuous/Informative Probability Curves", fontsize=13)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_fig5_stabilization(exp3_raw: pd.DataFrame, out_path: Path) -> None:
    agg = (
        exp3_raw.groupby("c2", as_index=False)[
            ["lambda_max_sigma_g", "ceiling_dir", "mean_norm", "weak_dir_var"]
        ]
        .median()
        .sort_values("c2")
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)

    ax = axes[0]
    ax.plot(agg["c2"], agg["lambda_max_sigma_g"], marker="o", linewidth=2, label=r"median $\lambda_{\max}(\Sigma_g)$")
    ax.plot(agg["c2"], agg["ceiling_dir"], marker="s", linewidth=1.8, linestyle="--", label=r"median ceiling $(\lambda_{\min}+c_g^{-2})^{-1}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$c_g^2$")
    ax.set_ylabel("Scale")
    ax.set_title("Panel A: Covariance Stabilization")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(agg["c2"], agg["mean_norm"], marker="o", linewidth=2, color="#2a9d8f")
    ax.set_xscale("log")
    ax.set_xlabel(r"$c_g^2$")
    ax.set_ylabel(r"median $\|m_g(c_g^2)\|_2$")
    ax.set_title("Panel B: Blockwise Mean Magnitude")
    ax.grid(alpha=0.25, which="both")

    fig.suptitle("Figure 5: Weak-identification Slab Stabilization", fontsize=13)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _monotone_summary(exp1_raw: pd.DataFrame) -> dict[str, float]:
    # A simple effect check at fixed c_target^2=1, c_out^2=1.
    sub = exp1_raw[(exp1_raw["c_target_sq"] == 1.0) & (exp1_raw["c_out_sq"] == 1.0)]
    med = sub.groupby("rho_cross")["eta_floor"].median().sort_index()
    diffs = np.diff(med.values)
    return {
        "eta_rho_monotone_nonneg": float((diffs >= -1e-9).mean()),
        "eta_low_coupling_median": float(med.loc[0.0]),
        "eta_high_coupling_median": float(med.loc[0.5]),
    }


def _effect_checks(exp2_sub: pd.DataFrame, exp3_raw: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    prob = exp2_sub.groupby(["prior", "p_g"], as_index=False)["non_vacuous_lt1"].mean()
    target = prob[(prob["prior"] == "IG(4,0.1)") & (prob["p_g"] == 5)]
    out["prob_bgstar_lt1_prior_IG(4,0.1)_pg5"] = float(target["non_vacuous_lt1"].iloc[0]) if len(target) else np.nan

    med_exp3 = exp3_raw.groupby("c2", as_index=False)["mean_norm"].median().sort_values("c2")
    diffs = np.diff(med_exp3["mean_norm"].values)
    out["mean_norm_monotone_nonneg"] = float((diffs >= -1e-9).mean())
    return out


def main() -> None:
    plt.style.use("default")

    root = Path("outputs/simulations")
    exp1_path = root / "grrhs_theory_adjusted_v2" / "experiment_1" / "exp1_raw.csv"
    exp2_path = root / "grrhs_theory_adjusted" / "experiment_2" / "exp2_raw.csv"
    exp3_path = root / "grrhs_theory_adjusted" / "experiment_3" / "exp3_raw.csv"
    out_dir = root / "grrhs_requested_five_figures"
    _ensure_dir(out_dir)

    exp1 = _load_csv(exp1_path)
    exp2 = _load_csv(exp2_path)
    exp3 = _load_csv(exp3_path)

    make_fig1_eta_heatmaps(exp1, out_dir / "figure1_eta_heatmap_2x2.png")
    make_fig2_inflation_heatmaps(exp1, out_dir / "figure2_inflation_heatmap_2x2.png", cap=10.0)
    exp2_sub, priors = make_fig3_bgstar_boxplot(exp2, out_dir / "figure3_bgstar_boxplot.png")
    make_fig4_probability_curves(exp2_sub, out_dir / "figure4_probability_curves_3panel.png")
    make_fig5_stabilization(exp3, out_dir / "figure5_stabilization_1x2.png")

    checks = {}
    checks.update(_monotone_summary(exp1))
    checks.update(_effect_checks(exp2_sub, exp3))
    checks["priors_for_fig34"] = str(priors)
    pd.Series(checks).to_csv(out_dir / "effect_checks.csv", header=["value"])
    print("Saved figures to:", out_dir)
    print(pd.Series(checks))


if __name__ == "__main__":
    main()
