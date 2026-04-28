from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    root = Path(r"d:\FilesP\GR-RHS\outputs\history\simulation_mechanism\mechanism_main\20260426_030600_638018")
    summary = pd.read_csv(root / "summary_paired.csv")
    delta = pd.read_csv(root / "summary_paired_deltas.csv")
    group = pd.read_csv(root / "per_group_kappa.csv")

    summary = summary.loc[summary["experiment_id"].astype(str).eq("M1")].copy()
    delta = delta.loc[
        delta["experiment_id"].astype(str).eq("M1")
        & delta["method"].astype(str).eq("GR_RHS")
        & delta["metric"].astype(str).isin(["mse_overall", "mse_signal", "mse_null"])
    ].copy()
    group = group.loc[
        group["experiment_id"].astype(str).eq("M1")
        & group["method"].astype(str).eq("GR_RHS")
    ].copy()
    if "paired_common_converged" in group.columns:
        group = group.loc[group["paired_common_converged"].fillna(False).astype(bool)].copy()

    group["kappa_group_mean"] = pd.to_numeric(group["kappa_group_mean"], errors="coerce")
    group["true_group_l2_norm"] = pd.to_numeric(group["true_group_l2_norm"], errors="coerce")
    group["group_id"] = pd.to_numeric(group["group_id"], errors="coerce")
    group["replicate_id"] = pd.to_numeric(group["replicate_id"], errors="coerce")
    if "is_active_group" in group.columns:
        group["is_active_group"] = group["is_active_group"].fillna(False).astype(bool)
    else:
        group["is_active_group"] = group["group_role"].astype(str).eq("active")

    role_order = ["other_null", "active"]
    role_labels = {"other_null": "Null groups", "active": "Signal groups"}
    role_colors = {"other_null": "#9aa0a6", "active": "#1f77b4"}
    role_positions = {"other_null": 0.0, "active": 1.0}

    fig = plt.figure(figsize=(12.6, 5.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.45, 1.10])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    rng = np.random.default_rng(20260426)
    for role in role_order:
        sub = group.loc[group["group_role"].astype(str).eq(role)].copy()
        if sub.empty:
            continue
        y = sub["kappa_group_mean"].to_numpy(dtype=float)
        x = np.full(len(sub), role_positions[role], dtype=float)
        x = x + rng.uniform(-0.08, 0.08, size=len(sub))
        ax0.scatter(
            x,
            y,
            s=28,
            alpha=0.22,
            color=role_colors[role],
            edgecolors="none",
            zorder=1,
        )
        quantiles = np.nanquantile(y, [0.10, 0.50, 0.90])
        ax0.vlines(role_positions[role], quantiles[0], quantiles[2], color=role_colors[role], lw=4.0, alpha=0.9, zorder=3)
        ax0.scatter([role_positions[role]], [quantiles[1]], s=85, color=role_colors[role], edgecolors="white", linewidths=0.8, zorder=4)
        ax0.text(role_positions[role], quantiles[2] + 0.045, f"median={quantiles[1]:.3f}", ha="center", va="bottom", fontsize=8, color="#333333")
    ax0.set_xlim(-0.45, 1.45)
    ax0.set_ylim(-0.02, 0.58)
    ax0.set_xticks([role_positions[r] for r in role_order], labels=[role_labels[r] for r in role_order])
    ax0.set_ylabel("Posterior mean $\\kappa_g$")
    ax0.set_title("Signal and null groups separate")
    ax0.grid(axis="y", alpha=0.22)

    trend = (
        group.groupby(["group_id", "true_group_l2_norm", "group_role", "is_active_group"], dropna=False, sort=True)["kappa_group_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    trend["se"] = trend["std"].astype(float) / np.sqrt(np.maximum(trend["count"].astype(float), 1.0))
    trend["ci_lo"] = trend["mean"].astype(float) - 1.96 * trend["se"].astype(float)
    trend["ci_hi"] = trend["mean"].astype(float) + 1.96 * trend["se"].astype(float)
    trend = trend.sort_values(["true_group_l2_norm", "group_id"], kind="stable")
    xvals = np.arange(len(trend), dtype=float)
    colors = [role_colors["active"] if bool(flag) else role_colors["other_null"] for flag in trend["is_active_group"].tolist()]
    ax1.plot(xvals, trend["mean"].to_numpy(dtype=float), color="#6b7280", lw=1.5, alpha=0.7, zorder=1)
    ax1.vlines(xvals, trend["ci_lo"].to_numpy(dtype=float), trend["ci_hi"].to_numpy(dtype=float), colors=colors, lw=2.2, alpha=0.95, zorder=2)
    ax1.scatter(xvals, trend["mean"].to_numpy(dtype=float), s=90, c=colors, edgecolors="white", linewidths=0.8, zorder=3)
    for idx, row in trend.reset_index(drop=True).iterrows():
        strength = float(row["true_group_l2_norm"]) if np.isfinite(float(row["true_group_l2_norm"])) else 0.0
        label = f"g{int(row['group_id'])}\n||beta||={strength:.1f}"
        ax1.text(float(idx), float(row["mean"]) + 0.035, label, ha="center", va="bottom", fontsize=8, color="#333333")
    ax1.set_xlim(-0.6, len(trend) - 0.4)
    ax1.set_ylim(-0.02, 0.58)
    ax1.set_xticks(xvals, labels=["null", "null", "weak", "mid", "strong"])
    ax1.set_ylabel("Mean posterior $\\kappa_g$ across replicates")
    ax1.set_title("Stronger true groups receive larger $\\kappa_g$")
    ax1.grid(axis="y", alpha=0.22)

    ax2.axis("off")
    text_lines: list[str] = []
    gr = summary.loc[summary["method"].astype(str).eq("GR_RHS")]
    rhs = summary.loc[summary["method"].astype(str).eq("RHS")]
    if not gr.empty:
        text_lines.append("GR-RHS mechanism summary")
        text_lines.append(f"kappa gap: {float(gr['kappa_gap'].iloc[0]):.3f}")
        text_lines.append(f"group AUROC: {float(gr['group_auroc'].iloc[0]):.3f}")
        text_lines.append(f"overall MSE: {float(gr['mse_overall'].iloc[0]):.3f}")
        if not rhs.empty:
            text_lines.append(f"RHS overall MSE: {float(rhs['mse_overall'].iloc[0]):.3f}")
        text_lines.append(f"paired converged replicates: {int(float(gr['n_paired'].iloc[0]))}")
    if not delta.empty:
        mse_delta = delta.loc[delta["metric"].astype(str).eq("mse_overall")]
        if not mse_delta.empty:
            row = mse_delta.iloc[0]
            text_lines.append("")
            text_lines.append("Paired difference vs RHS")
            text_lines.append(f"GR-RHS - RHS MSE: {float(row['mean_diff']):+.3f}")
            text_lines.append(f"95% CI: [{float(row['ci95_lo']):+.3f}, {float(row['ci95_hi']):+.3f}]")
            text_lines.append(f"wins-losses: {int(float(row['wins_vs_baseline']))}-{int(float(row['losses_vs_baseline']))}")
    active_vals = group.loc[group["is_active_group"], "kappa_group_mean"].to_numpy(dtype=float)
    null_vals = group.loc[~group["is_active_group"], "kappa_group_mean"].to_numpy(dtype=float)
    if active_vals.size and null_vals.size:
        text_lines.append("")
        text_lines.append("Interpretation")
        text_lines.append(f"signal median kappa: {float(np.nanmedian(active_vals)):.3f}")
        text_lines.append(f"null median kappa: {float(np.nanmedian(null_vals)):.3f}")
        text_lines.append("The group gate rises smoothly with")
        text_lines.append("true group strength while keeping")
        text_lines.append("null groups close to zero.")
    ax2.text(
        0.02,
        0.98,
        "\n".join(text_lines),
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        linespacing=1.45,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8fafc", edgecolor="#d0d7de", linewidth=1.0),
    )

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=role_colors["active"], markersize=9, label="signal group"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=role_colors["other_null"], markersize=9, label="null group"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, bbox_to_anchor=(0.43, -0.01), fontsize=9)
    fig.suptitle("Figure 2. M1 mechanism evidence: GR-RHS learns graded group separation", fontsize=13, fontweight="bold")
    fig.tight_layout()

    out = root / "figures" / "figure2_group_separation_m1_story.png"
    fig.savefig(out, dpi=240, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
