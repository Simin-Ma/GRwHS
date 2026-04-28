from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


DEFAULT_RESULTS_DIR = Path(
    r"d:\FilesP\GR-RHS\outputs\history\simulation_mechanism\mechanism_main\20260426_030600_638018"
)

FIG_FACE = "#fcfcfa"
AXIS_FACE = "#f5f6f2"
GRID_COLOR = "#d9ddd6"
SPINE_COLOR = "#c7cec8"
TEXT_DARK = "#1f2933"
TEXT_MID = "#5b6770"
TEXT_SOFT = "#8a949c"
HIGHLIGHT_COLOR = "#244d57"

METHOD_COLORS: dict[str, str] = {
    "GR_RHS": "#2f6b73",
    "RHS": "#c08a5c",
}

ROLE_COLORS: dict[str, str] = {
    "active": "#2f6b73",
    "decoy_null": "#ba7c4a",
    "other_null": "#a8b4c1",
}

VARIANT_COLORS: dict[str, str] = {
    "GR_RHS_oracle": "#6f879a",
    "GR_RHS_fixed_10x": "#8a9a68",
    "GR_RHS_no_local_scales": "#9e7c69",
    "GR_RHS_shared_kappa": "#9a6b79",
    "GR_RHS_no_kappa": "#b75d57",
    "RHS_oracle": "#b59c79",
}

VARIANT_MARKERS: dict[str, str] = {
    "GR_RHS_oracle": "o",
    "GR_RHS_fixed_10x": "D",
    "GR_RHS_no_local_scales": "s",
    "GR_RHS_shared_kappa": "^",
    "GR_RHS_no_kappa": "X",
    "RHS_oracle": "o",
}

DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "muted_diverging",
    ["#3d6c80", "#f7f7f3", "#d3a074"],
)
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "muted_sequential",
    ["#f1efe7", "#bfd1cb", "#4f7a76"],
)

plt.rcParams.update(
    {
        "figure.facecolor": FIG_FACE,
        "axes.facecolor": AXIS_FACE,
        "axes.edgecolor": SPINE_COLOR,
        "axes.labelcolor": TEXT_DARK,
        "axes.titlecolor": TEXT_DARK,
        "axes.titleweight": "bold",
        "xtick.color": TEXT_MID,
        "ytick.color": TEXT_MID,
        "text.color": TEXT_DARK,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.frameon": False,
    }
)


def _clean_method_label(label: object) -> str:
    return str(label).replace(" [stan_rstanarm_hs]", "")


def _card_kwargs(*, pad: float = 0.32, facecolor: str = "#ffffff") -> dict[str, object]:
    return {
        "boxstyle": f"round,pad={pad}",
        "facecolor": facecolor,
        "edgecolor": SPINE_COLOR,
        "linewidth": 0.9,
    }


def _panel_tag(ax: plt.Axes, tag: str) -> None:
    ax.text(
        -0.08,
        1.05,
        tag,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=TEXT_SOFT,
    )


def _style_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.set_facecolor(AXIS_FACE)
    if grid_axis in {"x", "y", "both"}:
        ax.grid(axis=grid_axis, color=GRID_COLOR, alpha=0.72, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(1.0)
    ax.tick_params(colors=TEXT_MID, labelcolor=TEXT_DARK)


def _annotated_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    *,
    cmap: LinearSegmentedColormap,
    title: str,
    cbar_label: str,
    extra_labels: dict[tuple[int, int], str] | None = None,
    center_zero: bool = False,
    highlight: tuple[int, int] | None = None,
    highlight_label: str | None = None,
) -> None:
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        spread = 1.0
        vmin = -spread if center_zero else 0.0
        vmax = spread
    elif center_zero:
        spread = float(np.nanmax(np.abs(finite)))
        spread = spread if spread > 0 else 1.0
        vmin, vmax = -spread, spread
    else:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        if np.isclose(vmin, vmax):
            vmin -= 0.5
            vmax += 0.5

    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
    ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)
    ax.set_title(title, pad=10)
    ax.set_facecolor(AXIS_FACE)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(1.0)

    for yi in range(matrix.shape[0]):
        for xi in range(matrix.shape[1]):
            ax.add_patch(
                Rectangle(
                    (xi - 0.5, yi - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor="#ffffff",
                    linewidth=1.2,
                )
            )
            value = matrix[yi, xi]
            if not np.isfinite(value):
                continue
            label = f"{value:+.3f}" if center_zero else f"{value:.3f}"
            if extra_labels and (yi, xi) in extra_labels:
                label = f"{label}\n{extra_labels[(yi, xi)]}"
            ax.text(
                xi,
                yi,
                label,
                ha="center",
                va="center",
                fontsize=10,
                color=TEXT_DARK,
                fontweight="bold" if highlight == (yi, xi) else None,
                linespacing=1.25,
            )

    if highlight is not None:
        yi, xi = highlight
        ax.add_patch(
            Rectangle(
                (xi - 0.5, yi - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor=HIGHLIGHT_COLOR,
                linewidth=2.6,
            )
        )
        if highlight_label:
            ax.text(
                xi,
                yi - 0.78,
                highlight_label,
                ha="center",
                va="bottom",
                fontsize=8.7,
                color=HIGHLIGHT_COLOR,
                fontweight="bold",
                bbox=_card_kwargs(pad=0.18, facecolor="#fdfdfb"),
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=9, colors=TEXT_MID)
    cbar.outline.set_edgecolor(SPINE_COLOR)
    cbar.outline.set_linewidth(0.9)


def _strength_labels(trend: pd.DataFrame) -> list[str]:
    if trend.empty:
        return []
    labels: list[str] = []
    active_count = int(trend["is_active_group"].fillna(False).astype(bool).sum())
    active_names = ["weak", "mid", "strong"]
    active_rank = 0
    for _, row in trend.iterrows():
        if not bool(row["is_active_group"]):
            labels.append("null")
            continue
        if active_count <= len(active_names):
            labels.append(active_names[active_rank])
        else:
            labels.append(f"active {active_rank + 1}")
        active_rank += 1
    return labels


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=280, bbox_inches="tight", pad_inches=0.10, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _build_m1_figure(results_dir: Path, out_dir: Path) -> Path:
    frame = pd.read_csv(results_dir / "paper_tables" / "figure_data" / "figure2_group_separation.csv")
    summary = frame.loc[frame["record_type"].astype(str).eq("method_summary")].copy()
    delta = frame.loc[
        frame["record_type"].astype(str).eq("paired_delta")
        & frame["metric"].astype(str).eq("mse_overall")
        & frame["method"].astype(str).eq("GR_RHS")
    ].copy()
    group = frame.loc[
        frame["record_type"].astype(str).eq("group_kappa")
        & frame["method"].astype(str).eq("GR_RHS")
    ].copy()

    group["kappa_group_mean"] = pd.to_numeric(group["kappa_group_mean"], errors="coerce")
    group["true_group_l2_norm"] = pd.to_numeric(group["true_group_l2_norm"], errors="coerce")
    group["group_id"] = pd.to_numeric(group["group_id"], errors="coerce")
    group["replicate_id"] = pd.to_numeric(group["replicate_id"], errors="coerce")
    group["is_active_group"] = group["is_active_group"].fillna(False).astype(bool)

    fig = plt.figure(figsize=(13.2, 6.1))
    fig.patch.set_facecolor(FIG_FACE)
    outer = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.02, 1.28, 0.96],
        left=0.06,
        right=0.985,
        top=0.84,
        bottom=0.19,
        wspace=0.28,
    )
    ax0 = fig.add_subplot(outer[0, 0])
    ax1 = fig.add_subplot(outer[0, 1])
    ax2 = fig.add_subplot(outer[0, 2])

    rng = np.random.default_rng(20260427)
    role_order = ["other_null", "active"]
    role_names = {"other_null": "Null groups", "active": "Signal groups"}
    role_positions = {"other_null": 0.0, "active": 1.0}
    medians: dict[str, float] = {}
    for role in role_order:
        sub = group.loc[group["group_role"].astype(str).eq(role)].copy()
        if sub.empty:
            continue
        values = sub["kappa_group_mean"].to_numpy(dtype=float)
        x = np.full(values.shape[0], role_positions[role], dtype=float) + rng.uniform(-0.08, 0.08, size=values.shape[0])
        color = ROLE_COLORS[role]
        q10, q50, q90 = np.nanquantile(values, [0.10, 0.50, 0.90])
        medians[role] = float(q50)
        ax0.scatter(x, values, s=28, alpha=0.18, color=color, edgecolors="none")
        ax0.vlines(role_positions[role], q10, q90, color=color, linewidth=4.4, alpha=0.95)
        ax0.hlines(q50, role_positions[role] - 0.10, role_positions[role] + 0.10, color=color, linewidth=3.4)
        ax0.scatter([role_positions[role]], [q50], s=96, color=color, edgecolors="white", linewidths=0.9, zorder=3)
        ax0.text(role_positions[role], q90 + 0.025, f"{q50:.3f}", ha="center", va="bottom", fontsize=9, color=TEXT_MID)
    median_gap = medians.get("active", np.nan) - medians.get("other_null", np.nan)
    ax0.text(
        0.03,
        0.96,
        f"median gap = {median_gap:.3f}",
        transform=ax0.transAxes,
        ha="left",
        va="top",
        fontsize=9.3,
        color=TEXT_MID,
        bbox=_card_kwargs(pad=0.22, facecolor="#fbfbf9"),
    )
    ax0.set_xlim(-0.45, 1.45)
    ax0.set_ylim(-0.01, 0.56)
    ax0.set_xticks([role_positions[r] for r in role_order], labels=[role_names[r] for r in role_order])
    ax0.set_ylabel("Posterior mean $\\kappa_g$")
    ax0.set_title("Null and signal groups separate cleanly")
    _style_axis(ax0)
    _panel_tag(ax0, "A")

    trend = (
        group.groupby(["group_id", "true_group_l2_norm", "group_role", "is_active_group"], dropna=False, sort=True)["kappa_group_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["true_group_l2_norm", "group_id"], kind="stable")
        .reset_index(drop=True)
    )
    trend["se"] = trend["std"].fillna(0.0).astype(float) / np.sqrt(np.maximum(trend["count"].astype(float), 1.0))
    trend["ci_lo"] = trend["mean"].astype(float) - 1.96 * trend["se"].astype(float)
    trend["ci_hi"] = trend["mean"].astype(float) + 1.96 * trend["se"].astype(float)
    xvals = np.arange(len(trend), dtype=float)
    point_colors = [ROLE_COLORS["active"] if flag else ROLE_COLORS["other_null"] for flag in trend["is_active_group"].tolist()]
    tick_labels = [
        f"{name}\n{float(strength):.1f}"
        for name, strength in zip(_strength_labels(trend), trend["true_group_l2_norm"].fillna(0.0).tolist())
    ]
    ax1.axvspan(1.5, len(trend) - 0.5, color="#edf3f1", alpha=0.95, zorder=0)
    ax1.plot(xvals, trend["mean"].to_numpy(dtype=float), color="#7d8998", linewidth=2.0, alpha=0.95)
    ax1.vlines(xvals, trend["ci_lo"].to_numpy(dtype=float), trend["ci_hi"].to_numpy(dtype=float), colors=point_colors, linewidth=2.4)
    ax1.scatter(xvals, trend["mean"].to_numpy(dtype=float), s=100, c=point_colors, edgecolors="white", linewidths=0.9, zorder=3)
    ax1.annotate(
        "gate rises with support",
        xy=(xvals[-1], float(trend["mean"].iloc[-1])),
        xytext=(2.5, 0.47),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "color": TEXT_MID, "linewidth": 1.0},
        fontsize=9,
        color=TEXT_MID,
    )
    ax1.set_xlim(-0.6, len(trend) - 0.4)
    ax1.set_ylim(-0.01, 0.56)
    ax1.set_xticks(xvals, labels=tick_labels)
    ax1.set_xlabel("Relative true group strength\n(second line shows $||\\beta_g||_2$)")
    ax1.set_ylabel("Mean posterior $\\kappa_g$")
    ax1.set_title("The learned gate tracks true group strength")
    _style_axis(ax1)
    _panel_tag(ax1, "B")

    summary = summary.loc[summary["method"].astype(str).isin(["GR_RHS", "RHS"])].copy()
    metric_rows = [
        ("mse_overall", "Overall MSE"),
        ("signal_group_mse", "Signal-group MSE"),
    ]
    compare_rows: list[dict[str, float | str]] = []
    for metric, label in metric_rows:
        row_gr = summary.loc[summary["method"].astype(str).eq("GR_RHS")].iloc[0]
        row_rhs = summary.loc[summary["method"].astype(str).eq("RHS")].iloc[0]
        compare_rows.append(
            {
                "label": label,
                "gr": float(row_gr[metric]),
                "rhs": float(row_rhs[metric]),
                "delta": float(row_gr[metric]) - float(row_rhs[metric]),
            }
        )
    comp = pd.DataFrame(compare_rows)
    y_positions = np.arange(len(comp), dtype=float)
    ax2.hlines(y_positions, comp["gr"], comp["rhs"], color="#d4dadf", linewidth=2.6)
    ax2.scatter(comp["rhs"], y_positions, s=96, color=METHOD_COLORS["RHS"], edgecolors="white", linewidths=0.9, zorder=3)
    ax2.scatter(comp["gr"], y_positions, s=96, color=METHOD_COLORS["GR_RHS"], edgecolors="white", linewidths=0.9, zorder=3)
    for ypos, row in zip(y_positions, comp.itertuples(index=False)):
        ax2.text(float(row.gr) - 0.004, ypos + 0.12, f"{float(row.gr):.3f}", ha="right", va="bottom", fontsize=8.7, color=TEXT_MID)
        ax2.text(float(row.rhs) + 0.004, ypos - 0.12, f"{float(row.rhs):.3f}", ha="left", va="top", fontsize=8.7, color=TEXT_MID)
        ax2.text(max(float(row.gr), float(row.rhs)) + 0.03, ypos, f"{float(row.delta):+.3f}", ha="left", va="center", fontsize=9, color=TEXT_DARK)
    ax2.text(
        0.03,
        0.96,
        f"{int(float(delta.iloc[0]['wins_vs_baseline']))}/{int(float(delta.iloc[0]['n_effective_pairs']))} paired wins on overall MSE",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=9.1,
        color=TEXT_MID,
        bbox=_card_kwargs(pad=0.22, facecolor="#fbfbf9"),
    )
    ax2.set_yticks(y_positions, labels=comp["label"].tolist())
    ax2.invert_yaxis()
    ax2.set_xlabel("Smaller is better")
    ax2.set_title("Prediction also improves in the paired comparison")
    _style_axis(ax2, grid_axis="x")
    _panel_tag(ax2, "C")
    ax2.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor=METHOD_COLORS["GR_RHS"], markersize=8.5, label="GR-RHS"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=METHOD_COLORS["RHS"], markersize=8.5, label="RHS"),
        ],
        loc="lower right",
        fontsize=8.8,
    )

    active_vals = group.loc[group["is_active_group"], "kappa_group_mean"].to_numpy(dtype=float)
    null_vals = group.loc[~group["is_active_group"], "kappa_group_mean"].to_numpy(dtype=float)
    footer = (
        f"Signal-group median $\\kappa_g$ = {float(np.nanmedian(active_vals)):.3f}; "
        f"null-group median $\\kappa_g$ = {float(np.nanmedian(null_vals)):.3f}; "
        f"paired overall-MSE difference = {float(delta.iloc[0]['mean_diff']):+.3f} "
        f"(95% CI [{float(delta.iloc[0]['ci95_lo']):+.3f}, {float(delta.iloc[0]['ci95_hi']):+.3f}])."
    )
    fig.text(
        0.5,
        0.08,
        footer,
        ha="center",
        va="center",
        fontsize=9.5,
        color=TEXT_MID,
        bbox=_card_kwargs(pad=0.28, facecolor="#ffffff"),
    )
    fig.suptitle("M1. Group separation: GR-RHS learns a usable group gate", y=0.94, fontsize=16, fontweight="bold")
    return _save_figure(fig, out_dir / "figure_m1_group_separation_story.png")


def _build_m2_figure(results_dir: Path, out_dir: Path) -> Path:
    heat = pd.read_csv(results_dir / "paper_tables" / "figure_data" / "figure3_correlation_ambiguity.csv")
    profile = pd.read_csv(results_dir / "paper_tables" / "figure_data" / "figure4_representative_profile.csv")

    heat["rho_within"] = pd.to_numeric(heat["rho_within"], errors="coerce")
    heat["kappa_gap"] = pd.to_numeric(heat["kappa_gap"], errors="coerce")
    heat["gr_minus_rhs_mse_overall"] = pd.to_numeric(heat["gr_minus_rhs_mse_overall"], errors="coerce")

    x_order = sorted(heat["rho_within"].dropna().unique().tolist())
    y_order = ["mixed_decoy", "concentrated"]
    x_labels = [f"$\\rho_w = {value:.1f}$" for value in x_order]
    y_labels = ["mixed decoy", "concentrated"]
    delta_matrix = np.full((len(y_order), len(x_order)), np.nan, dtype=float)
    for yi, pattern in enumerate(y_order):
        for xi, rho in enumerate(x_order):
            cell = heat.loc[
                heat["within_group_pattern"].astype(str).eq(pattern)
                & np.isclose(heat["rho_within"].astype(float), float(rho))
            ]
            if cell.empty:
                continue
            row = cell.iloc[0]
            delta_matrix[yi, xi] = float(row["gr_minus_rhs_mse_overall"])

    profile = profile.loc[profile["method"].astype(str).eq("GR_RHS")].copy()
    profile["group_id"] = pd.to_numeric(profile["group_id"], errors="coerce")
    profile["kappa_group_mean"] = pd.to_numeric(profile["kappa_group_mean"], errors="coerce")
    profile = profile.sort_values(["group_id"], kind="stable").reset_index(drop=True)

    fig = plt.figure(figsize=(12.8, 6.6))
    fig.patch.set_facecolor(FIG_FACE)
    outer = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.18, 1.00],
        height_ratios=[0.76, 1.00],
        left=0.06,
        right=0.985,
        top=0.84,
        bottom=0.19,
        wspace=0.28,
        hspace=0.24,
    )
    ax0 = fig.add_subplot(outer[:, 0])
    ax1 = fig.add_subplot(outer[0, 1])
    ax2 = fig.add_subplot(outer[1, 1])

    headline_cell = (0, len(x_order) - 1)
    _annotated_heatmap(
        ax0,
        delta_matrix,
        x_labels,
        y_labels,
        cmap=DIVERGING_CMAP,
        title="The performance edge appears only in the mixed-decoy regime",
        cbar_label="Paired overall-MSE difference\nnegative favors GR-RHS",
        center_zero=True,
        highlight=headline_cell,
        highlight_label="largest gain",
    )
    ax0.text(
        0.02,
        1.01,
        "All four cells use 100 paired replicates.",
        transform=ax0.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color=TEXT_MID,
    )
    _panel_tag(ax0, "A")

    gap_order = [
        ("mixed_decoy", 0.8),
        ("mixed_decoy", 0.9),
        ("concentrated", 0.8),
        ("concentrated", 0.9),
    ]
    gap_rows: list[dict[str, object]] = []
    for pattern, rho in gap_order:
        cell = heat.loc[
            heat["within_group_pattern"].astype(str).eq(pattern)
            & np.isclose(heat["rho_within"].astype(float), float(rho))
        ]
        if cell.empty:
            continue
        row = cell.iloc[0]
        gap_rows.append(
            {
                "label": f"{pattern.replace('_', ' ')}\n$\\rho_w = {rho:.1f}$",
                "kappa_gap": float(row["kappa_gap"]),
                "highlight": bool(pattern == "mixed_decoy" and np.isclose(float(rho), 0.9)),
            }
        )
    gap_df = pd.DataFrame(gap_rows)
    y_positions = np.arange(len(gap_df), dtype=float)
    colors = [HIGHLIGHT_COLOR if flag else "#91a9a4" for flag in gap_df["highlight"].tolist()]
    ax1.hlines(y_positions, 0.0, gap_df["kappa_gap"], color="#d5dbd6", linewidth=3.0)
    ax1.scatter(gap_df["kappa_gap"], y_positions, s=96, c=colors, edgecolors="white", linewidths=0.9, zorder=3)
    for ypos, row in zip(y_positions, gap_df.itertuples(index=False)):
        ax1.text(float(row.kappa_gap) + 0.004, ypos, f"{float(row.kappa_gap):.3f}", ha="left", va="center", fontsize=9, color=TEXT_DARK)
    ax1.axvline(0.0, color=SPINE_COLOR, linewidth=1.0)
    ax1.set_yticks(y_positions, labels=gap_df["label"].tolist())
    ax1.invert_yaxis()
    ax1.set_xlabel("GR-RHS kappa gap")
    ax1.set_title("The gate stays identifiable in every cell")
    _style_axis(ax1, grid_axis="x")
    ax1.text(
        0.03,
        0.86,
        "All four settings remain above 0.09.",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=TEXT_MID,
        bbox=_card_kwargs(pad=0.20, facecolor="#fbfbf9"),
    )
    _panel_tag(ax1, "B")

    xs = profile["group_id"].astype(int).to_numpy()
    ys = profile["kappa_group_mean"].to_numpy(dtype=float)
    colors = [ROLE_COLORS.get(str(role), "#8a949c") for role in profile["group_role"].astype(str).tolist()]
    ax2.plot(xs, ys, color="#8d98a6", linewidth=2.0, alpha=0.95)
    ax2.scatter(xs, ys, s=120, c=colors, edgecolors="white", linewidths=1.0, zorder=3)
    short_tags = {"active": "A", "decoy_null": "D", "other_null": "N"}
    for _, row in profile.iterrows():
        tag = short_tags.get(str(row["group_role"]), "?")
        ax2.text(float(row["group_id"]), float(row["kappa_group_mean"]) + 0.018, tag, ha="center", va="bottom", fontsize=8.5, color=TEXT_MID)
    decoy_row = profile.loc[profile["group_role"].astype(str).eq("decoy_null")].iloc[0]
    ax2.annotate(
        "decoy stays below the active groups",
        xy=(float(decoy_row["group_id"]), float(decoy_row["kappa_group_mean"])),
        xytext=(0.06, 0.90),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": TEXT_MID, "linewidth": 1.0},
        fontsize=9,
        color=TEXT_MID,
        ha="left",
        va="top",
    )
    ax2.set_xticks(xs)
    ax2.set_xlabel("Group id")
    ax2.set_ylabel("Posterior mean $\\kappa_g$")
    ax2.set_ylim(0.0, max(0.42, float(np.nanmax(ys)) + 0.08))
    ax2.set_title("Representative mixed-decoy replicate ($\\rho_w = 0.9$)")
    _style_axis(ax2)
    _panel_tag(ax2, "C")
    ax2.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor=ROLE_COLORS["active"], label="active", markersize=8.5),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=ROLE_COLORS["decoy_null"], label="decoy null", markersize=8.5),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=ROLE_COLORS["other_null"], label="other null", markersize=8.5),
        ],
        loc="lower right",
        fontsize=8.8,
    )

    footer = (
        "The largest gain appears at mixed decoy / $\\rho_w = 0.9$ "
        f"(paired overall-MSE difference = {float(delta_matrix[headline_cell]):+.3f}), "
        "while the GR-RHS gate remains positive in every setting."
    )
    fig.text(
        0.5,
        0.08,
        footer,
        ha="center",
        va="center",
        fontsize=9.5,
        color=TEXT_MID,
        bbox=_card_kwargs(pad=0.28, facecolor="#ffffff"),
    )
    fig.suptitle("M2. Correlation stress: the gain sharpens in the ambiguity regime", y=0.94, fontsize=16, fontweight="bold")
    return _save_figure(fig, out_dir / "figure_m2_correlation_ambiguity_story.png")


def _build_m3_figure(results_dir: Path, out_dir: Path) -> Path:
    summary = pd.read_csv(results_dir / "paper_tables" / "figure_data" / "figure5_complexity_unit.csv")
    deltas = pd.read_csv(results_dir / "summary_paired_deltas.csv")

    summary = summary.loc[summary["method"].astype(str).isin(["GR_RHS", "RHS"])].copy()
    summary["kappa_gap"] = pd.to_numeric(summary["kappa_gap"], errors="coerce")
    summary["mse_overall"] = pd.to_numeric(summary["mse_overall"], errors="coerce")
    summary["n_paired"] = pd.to_numeric(summary["n_paired"], errors="coerce")
    deltas = deltas.loc[
        deltas["experiment_id"].astype(str).eq("M3")
        & deltas["method"].astype(str).eq("GR_RHS")
        & deltas["baseline_method"].astype(str).eq("RHS")
        & deltas["metric"].astype(str).eq("mse_overall")
    ].copy()
    deltas["mean_diff"] = pd.to_numeric(deltas["mean_diff"], errors="coerce")

    merged: list[dict[str, object]] = []
    for setting_id, sub in summary.groupby("setting_id", sort=False):
        gr = sub.loc[sub["method"].astype(str).eq("GR_RHS")]
        rhs = sub.loc[sub["method"].astype(str).eq("RHS")]
        if gr.empty or rhs.empty:
            continue
        row_gr = gr.iloc[0]
        row_rhs = rhs.iloc[0]
        delta_row = deltas.loc[deltas["setting_id"].astype(str).eq(str(setting_id))]
        merged.append(
            {
                "setting_id": setting_id,
                "complexity_pattern": str(row_gr["complexity_pattern"]),
                "within_group_pattern": str(row_gr["within_group_pattern"]),
                "kappa_gap": float(row_gr["kappa_gap"]),
                "gr_mse_overall": float(row_gr["mse_overall"]),
                "rhs_mse_overall": float(row_rhs["mse_overall"]),
                "delta_mse_overall": float(delta_row["mean_diff"].iloc[0]) if not delta_row.empty else np.nan,
                "n_paired": int(float(row_gr["n_paired"])),
            }
        )
    plot_df = pd.DataFrame(merged)

    x_order = ["few_groups", "many_groups"]
    y_order = ["concentrated", "distributed"]
    x_labels = ["few groups", "many groups"]
    y_labels = ["concentrated", "distributed"]
    delta_matrix = np.full((len(y_order), len(x_order)), np.nan, dtype=float)
    extra_labels: dict[tuple[int, int], str] = {}
    for yi, within in enumerate(y_order):
        for xi, complexity in enumerate(x_order):
            cell = plot_df.loc[
                plot_df["within_group_pattern"].astype(str).eq(within)
                & plot_df["complexity_pattern"].astype(str).eq(complexity)
            ]
            if cell.empty:
                continue
            row = cell.iloc[0]
            delta_matrix[yi, xi] = float(row["delta_mse_overall"])
            extra_labels[(yi, xi)] = f"k-gap {float(row['kappa_gap']):.3f}"

    ordered_rows = plot_df.sort_values("delta_mse_overall", kind="stable").reset_index(drop=True)
    ordered_rows["cell_label"] = (
        ordered_rows["complexity_pattern"].astype(str).str.replace("_", " ", regex=False)
        + " / "
        + ordered_rows["within_group_pattern"].astype(str).str.replace("_", " ", regex=False)
    )

    fig = plt.figure(figsize=(12.8, 6.0))
    fig.patch.set_facecolor(FIG_FACE)
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.02, 1.18],
        left=0.06,
        right=0.985,
        top=0.84,
        bottom=0.19,
        wspace=0.30,
    )
    ax0 = fig.add_subplot(outer[0, 0])
    ax1 = fig.add_subplot(outer[0, 1])

    _annotated_heatmap(
        ax0,
        delta_matrix,
        x_labels,
        y_labels,
        cmap=DIVERGING_CMAP,
        title="Large gains appear only when signal is packed into few groups",
        cbar_label="Paired overall-MSE difference\nnegative favors GR-RHS",
        extra_labels=extra_labels,
        center_zero=True,
        highlight=(0, 0),
        highlight_label="largest edge",
    )
    ax0.add_patch(
        Rectangle(
            (1 - 0.5, 1 - 0.5),
            1.0,
            1.0,
            fill=False,
            edgecolor=TEXT_SOFT,
            linewidth=1.8,
            linestyle="--",
        )
    )
    ax0.text(
        1.0,
        1.48,
        "near tie",
        ha="center",
        va="bottom",
        fontsize=8.8,
        color=TEXT_MID,
        bbox=_card_kwargs(pad=0.18, facecolor="#fbfbf9"),
    )
    _panel_tag(ax0, "A")

    y_positions = np.arange(len(ordered_rows), dtype=float)
    ax1.hlines(y_positions, ordered_rows["gr_mse_overall"], ordered_rows["rhs_mse_overall"], color="#d4dadf", linewidth=2.8)
    ax1.scatter(ordered_rows["gr_mse_overall"], y_positions, s=100, color=METHOD_COLORS["GR_RHS"], edgecolors="white", linewidths=0.9, label="GR-RHS", zorder=3)
    ax1.scatter(ordered_rows["rhs_mse_overall"], y_positions, s=100, color=METHOD_COLORS["RHS"], edgecolors="white", linewidths=0.9, label="RHS", zorder=3)
    for ypos, row in zip(y_positions, ordered_rows.itertuples(index=False)):
        x_anchor = max(float(row.gr_mse_overall), float(row.rhs_mse_overall))
        ax1.text(x_anchor + 0.020, ypos, f"{float(row.delta_mse_overall):+.3f}", ha="left", va="center", fontsize=9, color=TEXT_DARK)
    near_tie = ordered_rows.iloc[-1]
    ax1.annotate(
        "almost no separation here",
        xy=(float(near_tie["gr_mse_overall"]), float(y_positions[-1])),
        xytext=(0.60, 0.12),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": TEXT_MID, "linewidth": 1.0},
        fontsize=9,
        color=TEXT_MID,
    )
    ax1.set_yticks(y_positions, labels=ordered_rows["cell_label"].tolist())
    ax1.invert_yaxis()
    ax1.set_xlabel("Overall MSE")
    ax1.set_title("The raw MSE scale tells the same story")
    _style_axis(ax1, grid_axis="x")
    _panel_tag(ax1, "B")
    ax1.legend(loc="lower right", fontsize=8.8)

    strongest = ordered_rows.iloc[0]
    footer = (
        f"The strongest edge occurs in few groups / concentrated "
        f"(paired overall-MSE difference = {float(strongest['delta_mse_overall']):+.3f}), "
        f"whereas many groups / distributed is effectively flat ({float(near_tie['delta_mse_overall']):+.3f})."
    )
    fig.text(
        0.5,
        0.08,
        footer,
        ha="center",
        va="center",
        fontsize=9.5,
        color=TEXT_MID,
        bbox=_card_kwargs(pad=0.28, facecolor="#ffffff"),
    )
    fig.suptitle("M3. Scope condition: the advantage depends on group complexity", y=0.94, fontsize=16, fontweight="bold")
    return _save_figure(fig, out_dir / "figure_m3_scope_condition_story.png")


def _forest_axis_limits(metric_df: pd.DataFrame) -> tuple[float, float]:
    finite_values: list[float] = []
    for column in ["mean_diff", "ci95_lo", "ci95_hi"]:
        if column in metric_df.columns:
            finite_values.extend(pd.to_numeric(metric_df[column], errors="coerce").dropna().astype(float).tolist())
    if not finite_values:
        return -1.0, 1.0
    xmin = min(finite_values)
    xmax = max(finite_values)
    spread = max(xmax - xmin, 0.02)
    pad = 0.16 * spread
    xmin = min(xmin - pad, -0.02 if xmax >= 0.0 else xmin - pad)
    xmax = max(xmax + pad, 0.02 if xmin <= 0.0 else xmax + pad)
    return xmin, xmax


def _draw_forest_panel(
    ax: plt.Axes,
    metric_df: pd.DataFrame,
    variant_order: list[str],
    label_map: dict[str, str],
    *,
    title: str,
    xlabel: str,
    show_ylabels: bool,
) -> None:
    bounds = _forest_axis_limits(metric_df)
    ax.set_xlim(*bounds)
    for ypos, method in enumerate(variant_order):
        row = metric_df.loc[metric_df["method"].astype(str).eq(method)]
        color = VARIANT_COLORS.get(method, "#8a949c")
        marker = VARIANT_MARKERS.get(method, "o")
        if row.empty:
            ax.text(bounds[1] - 0.01 * (bounds[1] - bounds[0]), ypos, "N/A", ha="right", va="center", fontsize=8.6, color=TEXT_SOFT)
            continue
        entry = row.iloc[0]
        mean = float(entry["mean_diff"])
        lo = float(entry["ci95_lo"])
        hi = float(entry["ci95_hi"])
        ax.hlines(ypos, lo, hi, color=color, linewidth=2.8, alpha=0.92)
        ax.scatter(mean, ypos, s=94, color=color, marker=marker, edgecolors="white", linewidths=0.9, zorder=3)
        x_pad = 0.02 * (bounds[1] - bounds[0])
        if mean >= 0:
            ax.text(mean + x_pad, ypos, f"{mean:+.3f}", ha="left", va="center", fontsize=8.7, color=TEXT_DARK)
        else:
            ax.text(mean - x_pad, ypos, f"{mean:+.3f}", ha="right", va="center", fontsize=8.7, color=TEXT_DARK)
    ax.axvline(0.0, color=TEXT_DARK, linewidth=1.0, linestyle="--")
    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_yticks(np.arange(len(variant_order), dtype=float), labels=[label_map[name] for name in variant_order] if show_ylabels else [])
    _style_axis(ax, grid_axis="x")


def _build_m4_figure(results_dir: Path, out_dir: Path) -> Path:
    summary = pd.read_csv(results_dir / "paper_tables" / "figure_data" / "figure6_ablation.csv")
    deltas = pd.read_csv(results_dir / "paper_tables" / "figure_data" / "figure6_ablation_deltas.csv")

    summary["tau_ratio_to_oracle"] = pd.to_numeric(summary["tau_ratio_to_oracle"], errors="coerce")
    summary["kappa_gap"] = pd.to_numeric(summary["kappa_gap"], errors="coerce")
    summary["mse_overall"] = pd.to_numeric(summary["mse_overall"], errors="coerce")
    summary["mse_signal"] = pd.to_numeric(summary["mse_signal"], errors="coerce")
    deltas["mean_diff"] = pd.to_numeric(deltas["mean_diff"], errors="coerce")
    deltas["ci95_lo"] = pd.to_numeric(deltas["ci95_lo"], errors="coerce")
    deltas["ci95_hi"] = pd.to_numeric(deltas["ci95_hi"], errors="coerce")

    variant_order = [
        "GR_RHS_oracle",
        "GR_RHS_fixed_10x",
        "GR_RHS_no_local_scales",
        "GR_RHS_shared_kappa",
        "GR_RHS_no_kappa",
        "RHS_oracle",
    ]
    label_map = {
        "GR_RHS_oracle": "oracle tau0",
        "GR_RHS_fixed_10x": "fixed 10x tau0",
        "GR_RHS_no_local_scales": "no local scales",
        "GR_RHS_shared_kappa": "shared kappa",
        "GR_RHS_no_kappa": "no kappa",
        "RHS_oracle": "RHS oracle",
    }

    baseline = summary.loc[summary["method"].astype(str).eq("GR_RHS")]
    if baseline.empty:
        raise RuntimeError("Baseline GR_RHS row was not found in figure6_ablation.csv.")
    baseline_row = baseline.iloc[0]

    fig = plt.figure(figsize=(13.2, 8.2))
    fig.patch.set_facecolor(FIG_FACE)
    outer = fig.add_gridspec(
        2,
        2,
        left=0.07,
        right=0.985,
        top=0.88,
        bottom=0.16,
        wspace=0.22,
        hspace=0.24,
    )
    ax00 = fig.add_subplot(outer[0, 0])
    ax01 = fig.add_subplot(outer[0, 1])
    ax10 = fig.add_subplot(outer[1, 0])
    ax11 = fig.add_subplot(outer[1, 1])

    _draw_forest_panel(
        ax00,
        deltas.loc[deltas["metric"].astype(str).eq("kappa_gap")].copy(),
        variant_order,
        label_map,
        title="Mechanism loss when each component is removed",
        xlabel="Delta kappa gap",
        show_ylabels=True,
    )
    _panel_tag(ax00, "A")

    _draw_forest_panel(
        ax01,
        deltas.loc[deltas["metric"].astype(str).eq("mse_overall")].copy(),
        variant_order,
        label_map,
        title="Prediction cost in overall MSE",
        xlabel="Delta overall MSE",
        show_ylabels=False,
    )
    _panel_tag(ax01, "B")

    _draw_forest_panel(
        ax10,
        deltas.loc[deltas["metric"].astype(str).eq("mse_signal")].copy(),
        variant_order,
        label_map,
        title="Prediction cost on the signal component",
        xlabel="Delta signal MSE",
        show_ylabels=True,
    )
    _panel_tag(ax10, "C")

    y_positions = np.arange(len(variant_order), dtype=float)
    tau_values = summary.set_index("method")["tau_ratio_to_oracle"].to_dict()
    finite_tau = [float(v) for v in tau_values.values() if np.isfinite(float(v))]
    tau_max = max(finite_tau) if finite_tau else 1.0
    ax11.set_xlim(0.0, tau_max + 0.7)
    for ypos, method in zip(y_positions, variant_order):
        value = tau_values.get(method, np.nan)
        color = VARIANT_COLORS.get(method, "#8a949c")
        marker = VARIANT_MARKERS.get(method, "o")
        if not np.isfinite(float(value)):
            ax11.text(0.10, ypos, "N/A", ha="left", va="center", fontsize=8.6, color=TEXT_SOFT)
            continue
        tau_ratio = float(value)
        ax11.hlines(ypos, 0.0, tau_ratio, color="#d7ddd8", linewidth=2.4)
        ax11.scatter(tau_ratio, ypos, s=96, color=color, marker=marker, edgecolors="white", linewidths=0.9, zorder=3)
        ax11.text(tau_ratio + 0.10, ypos, f"{tau_ratio:.2f}", ha="left", va="center", fontsize=8.7, color=TEXT_DARK)
    ax11.axvline(1.0, color=TEXT_DARK, linewidth=1.0, linestyle="--")
    ax11.text(
        1.0,
        1.02,
        "oracle",
        transform=ax11.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=TEXT_MID,
    )
    ax11.set_title("$\\tau_0$ calibration", pad=10)
    ax11.set_xlabel("$\\tau_0$ / oracle $\\tau_0$")
    ax11.set_yticks(y_positions, labels=[])
    _style_axis(ax11, grid_axis="x")
    ax11.text(
        0.03,
        0.95,
        f"baseline GR-RHS = {float(baseline_row['tau_ratio_to_oracle']):.2f}x oracle",
        transform=ax11.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=TEXT_MID,
        bbox=_card_kwargs(pad=0.20, facecolor="#fbfbf9"),
    )
    _panel_tag(ax11, "D")

    separator_y = variant_order.index("RHS_oracle") - 0.5
    for ax in [ax00, ax01, ax10, ax11]:
        ax.axhline(separator_y, color=GRID_COLOR, linewidth=1.0)

    ax00.invert_yaxis()
    footer = (
        f"Baseline GR-RHS uses kappa gap = {float(baseline_row['kappa_gap']):.3f}. "
        "Removing the group gate drives that gap to zero, while removing local scales mostly shows up as a large prediction penalty."
    )
    fig.text(
        0.5,
        0.08,
        footer,
        ha="center",
        va="center",
        fontsize=9.5,
        color=TEXT_MID,
        bbox=_card_kwargs(pad=0.28, facecolor="#ffffff"),
    )
    fig.suptitle("M4. Ablation attribution: the group gate drives the mechanism", y=0.96, fontsize=16, fontweight="bold")
    return _save_figure(fig, out_dir / "figure_m4_ablation_story.png")


def build_figure_suite(results_dir: Path, out_dir: Path | None = None) -> list[Path]:
    results_dir = Path(results_dir)
    target_dir = out_dir if out_dir is not None else results_dir / "figures"
    target_dir.mkdir(parents=True, exist_ok=True)
    return [
        _build_m1_figure(results_dir, target_dir),
        _build_m2_figure(results_dir, target_dir),
        _build_m3_figure(results_dir, target_dir),
        _build_m4_figure(results_dir, target_dir),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate unified narrative mechanism figures for M1-M4.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_figure_suite(args.results_dir, args.out_dir)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
