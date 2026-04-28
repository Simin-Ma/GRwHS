from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_COLORS: dict[str, str] = {
    "GR_RHS": "#1f77b4",
    "GR_RHS_fixed_10x": "#6b8e23",
    "GR_RHS_oracle": "#2ca02c",
    "GR_RHS_no_local_scales": "#8c564b",
    "GR_RHS_shared_kappa": "#9467bd",
    "GR_RHS_no_kappa": "#d62728",
    "RHS_oracle": "#ff7f0e",
}


def main() -> None:
    root = Path(r"d:\FilesP\GR-RHS\outputs\history\simulation_mechanism\mechanism_main\20260426_030600_638018")
    summary = pd.read_csv(root / "paper_tables" / "figure_data" / "figure6_ablation.csv")
    delta = pd.read_csv(root / "paper_tables" / "figure_data" / "figure6_ablation_deltas.csv")

    summary["kappa_gap"] = pd.to_numeric(summary["kappa_gap"], errors="coerce")
    summary["mse_overall"] = pd.to_numeric(summary["mse_overall"], errors="coerce")
    summary["mse_signal"] = pd.to_numeric(summary["mse_signal"], errors="coerce")
    summary["tau_ratio_to_oracle"] = pd.to_numeric(summary["tau_ratio_to_oracle"], errors="coerce")

    base = summary.loc[summary["method"].astype(str).eq("GR_RHS")].copy()
    if base.empty:
        raise RuntimeError("GR_RHS baseline row not found in figure6_ablation.csv")
    base_row = base.iloc[0]

    methods = [
        "GR_RHS",
        "GR_RHS_oracle",
        "GR_RHS_fixed_10x",
        "GR_RHS_no_kappa",
        "GR_RHS_shared_kappa",
        "GR_RHS_no_local_scales",
        "RHS_oracle",
    ]
    available = [m for m in methods if m in set(summary["method"].astype(str))]
    plot_df = summary.loc[summary["method"].astype(str).isin(available)].copy()
    plot_df["method_order"] = plot_df["method"].map({name: idx for idx, name in enumerate(available)})
    plot_df = plot_df.sort_values(["method_order"], kind="stable").reset_index(drop=True)

    fig = plt.figure(figsize=(12.8, 5.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.05])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    x_base = float(base_row["kappa_gap"])
    y_base = float(base_row["mse_overall"])
    for _, row in plot_df.iterrows():
        method = str(row["method"])
        color = METHOD_COLORS.get(method, "#7f7f7f")
        x = float(row["kappa_gap"]) if np.isfinite(float(row["kappa_gap"])) else 0.0
        y = float(row["mse_overall"])
        size = 150 if method == "GR_RHS" else 120
        marker = "o"
        if method in {"GR_RHS_no_kappa", "GR_RHS_shared_kappa"}:
            marker = "X"
        elif method == "GR_RHS_no_local_scales":
            marker = "s"
        elif method in {"GR_RHS_fixed_10x", "GR_RHS_oracle"}:
            marker = "D"
        elif method == "RHS_oracle":
            marker = "^"
        ax0.scatter(x, y, s=size, color=color, marker=marker, edgecolors="white", linewidths=0.9, zorder=3)
        label = str(row["method_label"]).replace(" [stan_rstanarm_hs]", "")
        ax0.text(x + 0.010, y + 0.0008, label, fontsize=8.5, color="#333333", ha="left", va="bottom")
        if method != "GR_RHS":
            ax0.plot([x_base, x], [y_base, y], color=color, lw=1.1, alpha=0.55, zorder=1)

    ax0.axvline(x_base, color="#1f2937", lw=1.0, ls="--")
    ax0.axhline(y_base, color="#1f2937", lw=1.0, ls="--")
    ax0.set_xlabel("Mechanism strength: kappa gap")
    ax0.set_ylabel("Overall MSE")
    ax0.set_title("Which ablations kill the mechanism?")
    ax0.grid(alpha=0.22)
    ax0.annotate(
        "worse prediction",
        xy=(ax0.get_xlim()[0], y_base),
        xytext=(ax0.get_xlim()[0] + 0.015, y_base + 0.010),
        fontsize=8,
        color="#444444",
    )
    ax0.annotate(
        "weaker group separation",
        xy=(x_base, ax0.get_ylim()[0]),
        xytext=(x_base - 0.18, ax0.get_ylim()[0] + 0.004),
        fontsize=8,
        color="#444444",
    )

    keep = delta.loc[
        delta["metric"].astype(str).isin(["kappa_gap", "mse_overall"])
        & delta["method"].astype(str).isin([m for m in available if m != "GR_RHS"])
    ].copy()
    metric_order = ["kappa_gap", "mse_overall"]
    method_order = [m for m in available if m != "GR_RHS"]
    ytick_positions: list[float] = []
    ytick_labels: list[str] = []
    row_space = 1.25
    metric_gap = 1.2
    for m_idx, metric in enumerate(metric_order):
        metric_df = keep.loc[keep["metric"].astype(str).eq(metric)].copy()
        for idx, method in enumerate(method_order):
            cell = metric_df.loc[metric_df["method"].astype(str).eq(method)]
            if cell.empty:
                continue
            row = cell.iloc[0]
            ypos = m_idx * (len(method_order) + metric_gap) + idx * row_space
            mean_diff = float(row["mean_diff"])
            lo = float(row["ci95_lo"])
            hi = float(row["ci95_hi"])
            color = METHOD_COLORS.get(method, "#7f7f7f")
            ax1.hlines(ypos, lo, hi, color=color, lw=2.4, alpha=0.9)
            ax1.scatter(mean_diff, ypos, s=85, color=color, edgecolors="white", linewidths=0.8, zorder=3)
            ytick_positions.append(ypos)
            clean_label = str(row["method_label"]).replace(" [stan_rstanarm_hs]", "")
            ytick_labels.append(clean_label if metric == "kappa_gap" else "")
            wins = int(float(row["wins_vs_baseline"])) if "wins_vs_baseline" in row.index else 0
            losses = int(float(row["losses_vs_baseline"])) if "losses_vs_baseline" in row.index else 0
            ax1.text(hi + 0.006, ypos, f"{wins}-{losses}", va="center", ha="left", fontsize=7.5, color="#444444")
        mid = m_idx * (len(method_order) + metric_gap) + ((len(method_order) - 1) * row_space / 2.0)
        ax1.text(
            ax1.get_xlim()[0] if np.isfinite(ax1.get_xlim()[0]) else -0.1,
            mid,
            "delta kappa gap" if metric == "kappa_gap" else "delta overall MSE",
            rotation=90,
            va="center",
            ha="right",
            fontsize=8.5,
            color="#333333",
        )
    ax1.axvline(0.0, color="#1f2937", lw=1.0, ls="--")
    ax1.set_yticks(ytick_positions, labels=ytick_labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("Delta vs GR-RHS")
    ax1.set_title("Paired effect sizes with 95% CI")
    ax1.grid(axis="x", alpha=0.22)

    ax2.axis("off")
    no_kappa = plot_df.loc[plot_df["method"].astype(str).eq("GR_RHS_no_kappa")]
    shared = plot_df.loc[plot_df["method"].astype(str).eq("GR_RHS_shared_kappa")]
    no_local = plot_df.loc[plot_df["method"].astype(str).eq("GR_RHS_no_local_scales")]
    fixed = plot_df.loc[plot_df["method"].astype(str).eq("GR_RHS_fixed_10x")]
    rhs_oracle = plot_df.loc[plot_df["method"].astype(str).eq("RHS_oracle")]

    lines: list[str] = [
        "M4 mechanism takeaway",
        f"Baseline GR-RHS kappa gap: {float(base_row['kappa_gap']):.3f}",
        f"Baseline overall MSE: {float(base_row['mse_overall']):.3f}",
        "",
    ]
    if not no_kappa.empty:
        lines.append(f"No kappa: kappa gap -> {float(no_kappa['kappa_gap'].iloc[0]):.3f}")
    if not shared.empty:
        lines.append(f"Shared kappa: kappa gap -> {float(shared['kappa_gap'].iloc[0]):.3f}")
    if not no_local.empty:
        lines.append(f"No local scales: overall MSE -> {float(no_local['mse_overall'].iloc[0]):.3f}")
    if not fixed.empty:
        lines.append(f"Fixed 10x tau0: tau/oracle -> {float(fixed['tau_ratio_to_oracle'].iloc[0]):.2f}")
    if not rhs_oracle.empty:
        lines.append(f"RHS oracle overall MSE: {float(rhs_oracle['mse_overall'].iloc[0]):.3f}")
    lines.extend(
        [
            "",
            "Interpretation",
            "Destroying kappa or forcing one shared gate",
            "collapses the mechanism signal almost completely.",
            "Removing local scales hurts prediction badly,",
            "but does not erase group separation to the same degree.",
            "So the key mechanism is the group gate itself,",
            "not only tau calibration or coefficient-local shrinkage.",
        ]
    )
    ax2.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        linespacing=1.45,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8fafc", edgecolor="#d0d7de", linewidth=1.0),
    )

    fig.suptitle("Figure M4. Mechanism attribution under GR-RHS ablations", fontsize=13, fontweight="bold")
    fig.tight_layout()

    out = root / "figures" / "figure_m4_mechanism_attribution.png"
    fig.savefig(out, dpi=240, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
