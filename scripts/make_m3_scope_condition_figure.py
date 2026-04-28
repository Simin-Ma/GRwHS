from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_COLORS: dict[str, str] = {
    "GR_RHS": "#1f77b4",
    "RHS": "#ff7f0e",
}


def main() -> None:
    root = Path(r"d:\FilesP\GR-RHS\outputs\history\simulation_mechanism\mechanism_main\20260426_030600_638018")
    summary = pd.read_csv(root / "paper_tables" / "figure_data" / "figure5_complexity_unit.csv")
    deltas = pd.read_csv(root / "summary_paired_deltas.csv")

    summary["kappa_gap"] = pd.to_numeric(summary["kappa_gap"], errors="coerce")
    summary["mse_overall"] = pd.to_numeric(summary["mse_overall"], errors="coerce")
    summary["n_paired"] = pd.to_numeric(summary["n_paired"], errors="coerce")
    deltas["mean_diff"] = pd.to_numeric(deltas["mean_diff"], errors="coerce")
    deltas["ci95_lo"] = pd.to_numeric(deltas["ci95_lo"], errors="coerce")
    deltas["ci95_hi"] = pd.to_numeric(deltas["ci95_hi"], errors="coerce")

    summary = summary.loc[summary["method"].astype(str).isin(["GR_RHS", "RHS"])].copy()
    deltas = deltas.loc[
        deltas["experiment_id"].astype(str).eq("M3")
        & deltas["method"].astype(str).eq("GR_RHS")
        & deltas["metric"].astype(str).isin(["mse_overall", "mse_signal", "mse_null", "group_auroc"])
    ].copy()

    merged_rows: list[dict[str, object]] = []
    for setting_id, sub in summary.groupby("setting_id", sort=False):
        gr = sub.loc[sub["method"].astype(str).eq("GR_RHS")]
        rhs = sub.loc[sub["method"].astype(str).eq("RHS")]
        if gr.empty or rhs.empty:
            continue
        row_gr = gr.iloc[0]
        row_rhs = rhs.iloc[0]
        delta_overall = deltas.loc[
            deltas["setting_id"].astype(str).eq(str(setting_id))
            & deltas["metric"].astype(str).eq("mse_overall")
        ]
        delta_signal = deltas.loc[
            deltas["setting_id"].astype(str).eq(str(setting_id))
            & deltas["metric"].astype(str).eq("mse_signal")
        ]
        delta_null = deltas.loc[
            deltas["setting_id"].astype(str).eq(str(setting_id))
            & deltas["metric"].astype(str).eq("mse_null")
        ]
        delta_auroc = deltas.loc[
            deltas["setting_id"].astype(str).eq(str(setting_id))
            & deltas["metric"].astype(str).eq("group_auroc")
        ]
        merged_rows.append(
            {
                "setting_id": setting_id,
                "complexity_pattern": str(row_gr["complexity_pattern"]),
                "within_group_pattern": str(row_gr["within_group_pattern"]),
                "kappa_gap": float(row_gr["kappa_gap"]),
                "gr_mse_overall": float(row_gr["mse_overall"]),
                "rhs_mse_overall": float(row_rhs["mse_overall"]),
                "delta_mse_overall": float(delta_overall["mean_diff"].iloc[0]) if not delta_overall.empty else np.nan,
                "delta_mse_signal": float(delta_signal["mean_diff"].iloc[0]) if not delta_signal.empty else np.nan,
                "delta_mse_null": float(delta_null["mean_diff"].iloc[0]) if not delta_null.empty else np.nan,
                "delta_group_auroc": float(delta_auroc["mean_diff"].iloc[0]) if not delta_auroc.empty else np.nan,
                "delta_mse_overall_lo": float(delta_overall["ci95_lo"].iloc[0]) if not delta_overall.empty else np.nan,
                "delta_mse_overall_hi": float(delta_overall["ci95_hi"].iloc[0]) if not delta_overall.empty else np.nan,
                "wins_vs_rhs": int(float(delta_overall["wins_vs_baseline"].iloc[0])) if not delta_overall.empty else 0,
                "losses_vs_rhs": int(float(delta_overall["losses_vs_baseline"].iloc[0])) if not delta_overall.empty else 0,
                "n_paired": int(float(row_gr["n_paired"])) if pd.notna(row_gr["n_paired"]) else 0,
            }
        )

    plot_df = pd.DataFrame(merged_rows)
    x_map = {"few_groups": 0.0, "many_groups": 1.0}
    y_map = {"distributed": 0.0, "concentrated": 1.0}
    plot_df["x"] = plot_df["complexity_pattern"].map(x_map)
    plot_df["y"] = plot_df["within_group_pattern"].map(y_map)

    fig = plt.figure(figsize=(12.8, 5.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.25, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    x_labels = ["few groups", "many groups"]
    y_labels = ["distributed", "concentrated"]
    delta_matrix = np.full((2, 2), np.nan, dtype=float)
    gap_matrix = np.full((2, 2), np.nan, dtype=float)
    ann_matrix = np.full((2, 2), np.nan, dtype=float)
    for _, row in plot_df.iterrows():
        xi = int(row["x"])
        yi = int(row["y"])
        delta_matrix[yi, xi] = float(row["delta_mse_overall"])
        gap_matrix[yi, xi] = float(row["kappa_gap"])
        ann_matrix[yi, xi] = float(row["n_paired"])

    spread = np.nanmax(np.abs(delta_matrix[np.isfinite(delta_matrix)]))
    if not np.isfinite(spread) or spread <= 0:
        spread = 1.0
    im0 = ax0.imshow(delta_matrix, cmap="RdBu_r", vmin=-spread, vmax=spread, aspect="equal")
    ax0.set_xticks(np.arange(2), labels=x_labels)
    ax0.set_yticks(np.arange(2), labels=y_labels)
    ax0.set_title("Paired overall MSE: GR-RHS - RHS")
    for yi in range(2):
        for xi in range(2):
            val = delta_matrix[yi, xi]
            n = ann_matrix[yi, xi]
            if np.isfinite(val):
                ax0.text(xi, yi, f"{val:+.3f}\n(n={int(n)})", ha="center", va="center", fontsize=9, color="#111111")
    cbar0 = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    cbar0.set_label("negative = GR-RHS better")

    im1 = ax1.imshow(gap_matrix, cmap="YlGnBu", aspect="equal")
    ax1.set_xticks(np.arange(2), labels=x_labels)
    ax1.set_yticks(np.arange(2), labels=y_labels)
    ax1.set_title("GR-RHS mechanism strength: kappa gap")
    for yi in range(2):
        for xi in range(2):
            val = gap_matrix[yi, xi]
            if np.isfinite(val):
                ax1.text(xi, yi, f"{val:.3f}", ha="center", va="center", fontsize=9, color="#111111")
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("kappa gap")

    ax2.axis("off")
    strongest = plot_df.sort_values("delta_mse_overall").iloc[0]
    weakest = plot_df.sort_values("delta_mse_overall", ascending=False).iloc[0]
    near_tie = plot_df.loc[
        plot_df["complexity_pattern"].astype(str).eq("many_groups")
        & plot_df["within_group_pattern"].astype(str).eq("distributed")
    ]

    lines = [
        "M3 scope-condition takeaway",
        f"Strongest advantage: {str(strongest['complexity_pattern']).replace('_', ' ')} / {str(strongest['within_group_pattern']).replace('_', ' ')}",
        f"delta MSE: {float(strongest['delta_mse_overall']):+.3f}",
        f"kappa gap: {float(strongest['kappa_gap']):.3f}",
        "",
        f"Weakest advantage: {str(weakest['complexity_pattern']).replace('_', ' ')} / {str(weakest['within_group_pattern']).replace('_', ' ')}",
        f"delta MSE: {float(weakest['delta_mse_overall']):+.3f}",
        f"kappa gap: {float(weakest['kappa_gap']):.3f}",
        "",
    ]
    if not near_tie.empty:
        row = near_tie.iloc[0]
        lines.extend(
            [
                "Near-tie cell",
                f"many groups / distributed: {float(row['delta_mse_overall']):+.3f}",
                "",
            ]
        )
    lines.extend(
        [
            "Interpretation",
            "GR-RHS is reacting to how signal is allocated",
            "across groups, not only to the total coefficient budget.",
            "When signal is concentrated into fewer groups,",
            "the group-aware prior has a much clearer advantage.",
            "When signal is spread across many groups,",
            "especially in the distributed case, the edge narrows sharply.",
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

    fig.suptitle("Figure M3. Scope condition: the advantage depends on group allocation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = root / "figures" / "figure_m3_scope_condition.png"
    fig.savefig(out, dpi=240, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
