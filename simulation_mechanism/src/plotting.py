from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from simulation_project.src.utils import load_pandas

from .utils import ensure_dir, resolve_history_results_dir


_METHOD_COLORS: dict[str, str] = {
    "GR_RHS": "#2f6b73",
    "RHS": "#c08a5c",
    "RHS_oracle": "#b59c79",
    "GR_RHS_fixed_10x": "#8a9a68",
    "GR_RHS_oracle": "#6f879a",
    "GR_RHS_no_local_scales": "#9e7c69",
    "GR_RHS_shared_kappa": "#9a6b79",
    "GR_RHS_no_kappa": "#b75d57",
}


_GROUP_ROLE_COLORS: dict[str, str] = {
    "active": "#2f6b73",
    "decoy_null": "#ba7c4a",
    "other_null": "#a8b4c1",
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
        "figure.facecolor": "#fcfcfa",
        "axes.facecolor": "#f5f6f2",
        "axes.edgecolor": "#c7cec8",
        "axes.labelcolor": "#1f2933",
        "axes.titlecolor": "#1f2933",
        "axes.titleweight": "bold",
        "xtick.color": "#5b6770",
        "ytick.color": "#5b6770",
        "text.color": "#1f2933",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.frameon": False,
    }
)


def _method_color(name: str) -> str:
    return _METHOD_COLORS.get(str(name), "#7f7f7f")


def _group_role_color(name: str) -> str:
    return _GROUP_ROLE_COLORS.get(str(name), "#7f7f7f")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    save_kws = dict(dpi=240, bbox_inches="tight", pad_inches=0.10)
    fig.savefig(path, **save_kws)
    history_dir = path.parent / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    shutil.copy2(path, history_dir / f"{path.stem}_{ts}{path.suffix}")
    plt.close(fig)


def _as_frame(df: Any):
    pd = load_pandas()
    if hasattr(df, "groupby"):
        return df
    if hasattr(df, "to_dict"):
        try:
            return pd.DataFrame(df.to_dict(orient="records"))
        except TypeError:
            pass
    return pd.DataFrame(list(df))


def _read_csv_or_empty(path: Path):
    pd = load_pandas()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _draw_missing_panel(ax, title: str, message: str, *, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_alpha(0.18)
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="#4b5563",
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#f8fafc", edgecolor="#d0d7de", linewidth=1.0),
    )


def _figure2_strength_labels(trend: Any) -> list[str]:
    if trend.empty:
        return []

    ordered = trend.reset_index(drop=True).copy()
    labels = ["active"] * len(ordered)
    active_rank = 0
    active_total = int(ordered["is_active_group"].fillna(False).astype(bool).sum()) if "is_active_group" in ordered.columns else 0
    active_names = ["weak", "medium", "strong"]
    for idx, row in ordered.iterrows():
        is_active = bool(row.get("is_active_group", False))
        if not is_active:
            labels[idx] = "null"
            continue
        if active_total == 1:
            labels[idx] = "active"
        elif active_total <= len(active_names):
            labels[idx] = active_names[active_rank]
        else:
            labels[idx] = f"active {active_rank + 1}"
        active_rank += 1
    return labels


def _heatmap_panel(ax, matrix: np.ndarray, x_labels: Sequence[str], y_labels: Sequence[str], *, cmap: str, center: float | None = None, fmt: str = "{:.3f}", annotation: np.ndarray | None = None, title: str = ""):
    arr = np.asarray(matrix, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        vmin, vmax = -1.0, 1.0
    else:
        if center is None:
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
        else:
            spread = float(np.max(np.abs(finite - center)))
            vmin = center - spread
            vmax = center + spread
            if abs(vmax - vmin) < 1e-12:
                vmin = center - 1.0
                vmax = center + 1.0
    im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
    ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)
    ax.set_title(title, fontsize=10)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            label = "--" if not np.isfinite(val) else fmt.format(float(val))
            if annotation is not None:
                ann = annotation[i, j]
                if np.isfinite(ann):
                    label = label + f"\n(n={int(ann)})"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color="#111111")
    return im


def _card_kwargs(*, pad: float = 0.32, facecolor: str = "#ffffff") -> dict[str, object]:
    return {
        "boxstyle": f"round,pad={pad}",
        "facecolor": facecolor,
        "edgecolor": "#c7cec8",
        "linewidth": 0.9,
    }


def _style_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.set_facecolor("#f5f6f2")
    if grid_axis in {"x", "y", "both"}:
        ax.grid(axis=grid_axis, color="#d9ddd6", alpha=0.72, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#c7cec8")
        spine.set_linewidth(1.0)
    ax.tick_params(colors="#5b6770", labelcolor="#1f2933")


def _panel_tag(ax: plt.Axes, tag: str) -> None:
    ax.text(
        -0.08,
        1.04,
        tag,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="#8a949c",
    )


def _styled_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    *,
    cmap,
    title: str,
    cbar_label: str,
    fmt: str = "{:+.3f}",
    center_zero: bool = False,
    extra_labels: dict[tuple[int, int], str] | None = None,
    highlight: tuple[int, int] | None = None,
    highlight_label: str | None = None,
):
    from matplotlib.patches import Rectangle

    arr = np.asarray(matrix, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        vmin, vmax = (-1.0, 1.0) if center_zero else (0.0, 1.0)
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

    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(np.arange(len(x_labels)), labels=list(x_labels))
    ax.set_yticks(np.arange(len(y_labels)), labels=list(y_labels))
    ax.set_title(title, pad=10)
    ax.set_facecolor("#f5f6f2")
    for spine in ax.spines.values():
        spine.set_color("#c7cec8")
        spine.set_linewidth(1.0)
    ax.tick_params(colors="#5b6770", labelcolor="#1f2933")

    for yi in range(arr.shape[0]):
        for xi in range(arr.shape[1]):
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
            value = arr[yi, xi]
            if not np.isfinite(value):
                continue
            label = fmt.format(float(value))
            if extra_labels and (yi, xi) in extra_labels:
                label = f"{label}\n{extra_labels[(yi, xi)]}"
            ax.text(
                xi,
                yi,
                label,
                ha="center",
                va="center",
                fontsize=9.5,
                color="#1f2933",
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
                edgecolor="#244d57",
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
                color="#244d57",
                fontweight="bold",
                bbox=_card_kwargs(pad=0.18, facecolor="#fdfdfb"),
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=9, colors="#5b6770")
    cbar.outline.set_edgecolor("#c7cec8")
    cbar.outline.set_linewidth(0.9)
    return im


def plot_figure1_mechanism_schematic(out_dir: Path | str) -> str:
    out_dir = ensure_dir(out_dir)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    panels = [
        (
            axes[0],
            "RHS",
            [
                ("Global\nshrinkage", (0.50, 0.84), "#f5c26b"),
                ("Local scales\n$\\lambda_j$", (0.50, 0.56), "#d9d9d9"),
                ("Coefficients\n$\\beta_j$", (0.50, 0.27), "#9ecae1"),
            ],
            [
                ((0.50, 0.78), (0.50, 0.62)),
                ((0.50, 0.50), (0.50, 0.33)),
            ],
            "Coefficient-level shrinkage only",
        ),
        (
            axes[1],
            "GR-RHS",
            [
                ("Global /\ngroup calibration", (0.50, 0.88), "#f5c26b"),
                ("Group gate\n$\\kappa_g$", (0.50, 0.66), "#f28482"),
                ("Local scales\n$\\lambda_j$", (0.50, 0.44), "#d9d9d9"),
                ("Coefficients\n$\\beta_j$", (0.50, 0.22), "#9ecae1"),
            ],
            [
                ((0.50, 0.82), (0.50, 0.72)),
                ((0.50, 0.60), (0.50, 0.50)),
                ((0.50, 0.38), (0.50, 0.28)),
            ],
            "Group-aware two-level hierarchy",
        ),
    ]
    for ax, title, boxes, arrows, footer in panels:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, fontsize=14, fontweight="bold")
        for label, (x, y), color in boxes:
            ax.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.45", facecolor=color, edgecolor="#444444", linewidth=1.2),
            )
        for (x0, y0), (x1, y1) in arrows:
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.6, color="#333333"))
        ax.text(0.50, 0.04, footer, ha="center", va="center", fontsize=10, color="#444444")
    fig.suptitle("Figure 1. Conceptual hierarchy of RHS and GR-RHS", fontsize=14, fontweight="bold")
    out_path = out_dir / "figure1_mechanism_schematic.png"
    _save(fig, out_path)
    return str(out_path)


def plot_figure2_group_separation(df: Any, out_dir: Path | str) -> str | None:
    frame = _as_frame(df)
    out_dir = ensure_dir(out_dir)
    if frame.empty or "record_type" not in frame.columns:
        return None
    pd = load_pandas()
    frame = frame.copy()
    summary = frame.loc[frame["record_type"].astype(str).eq("method_summary")].copy()
    if {"metric", "method"}.issubset(set(frame.columns)):
        delta = frame.loc[
            frame["record_type"].astype(str).eq("paired_delta")
            & frame["metric"].astype(str).eq("mse_overall")
            & frame["method"].astype(str).eq("GR_RHS")
        ].copy()
    else:
        delta = frame.iloc[0:0].copy()
    group = frame.loc[
        frame["record_type"].astype(str).eq("group_kappa")
        & frame["method"].astype(str).eq("GR_RHS")
    ].copy()
    if group.empty:
        return None

    group["kappa_group_mean"] = pd.to_numeric(group["kappa_group_mean"], errors="coerce")
    group["true_group_l2_norm"] = pd.to_numeric(group["true_group_l2_norm"], errors="coerce")
    group["group_id"] = pd.to_numeric(group["group_id"], errors="coerce")
    group["replicate_id"] = pd.to_numeric(group["replicate_id"], errors="coerce")
    if "is_active_group" in group.columns:
        group["is_active_group"] = group["is_active_group"].fillna(False).astype(bool)
    else:
        group["is_active_group"] = group["group_role"].astype(str).eq("active")

    fig = plt.figure(figsize=(10.8, 4.8))
    fig.patch.set_facecolor("white")
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[0.92, 1.25],
        left=0.08,
        right=0.985,
        top=0.84,
        bottom=0.22,
        wspace=0.24,
    )
    ax0 = fig.add_subplot(outer[0, 0])
    ax1 = fig.add_subplot(outer[0, 1])

    for ax in (ax0, ax1):
        ax.set_facecolor("white")
        ax.grid(True, which="major", color="#d9d9d9", linewidth=0.8)
        ax.grid(True, which="minor", color="#ececec", linewidth=0.55)
        ax.minorticks_on()
        for spine in ax.spines.values():
            spine.set_color("#333333")
            spine.set_linewidth(1.1)
        ax.tick_params(colors="#333333", labelsize=10)

    rng = np.random.default_rng(20260427)
    role_order = ["other_null", "active"]
    role_names = {"other_null": "Null groups", "active": "Signal groups"}
    role_positions = {"other_null": 0.0, "active": 1.0}
    role_colors = {"other_null": "#ff6f61", "active": "#17becf"}
    medians: dict[str, float] = {}
    box_data: list[np.ndarray] = []
    box_positions: list[float] = []
    box_colors: list[str] = []
    for role in role_order:
        sub = group.loc[group["group_role"].astype(str).eq(role)].copy()
        if sub.empty:
            continue
        values = sub["kappa_group_mean"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        box_data.append(values)
        box_positions.append(role_positions[role])
        box_colors.append(role_colors[role])
        medians[role] = float(np.nanmedian(values))

    if box_data:
        bp = ax0.boxplot(
            box_data,
            positions=box_positions,
            widths=0.42,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.8),
            whiskerprops=dict(color="#4a4a4a", linewidth=1.2),
            capprops=dict(color="#4a4a4a", linewidth=1.2),
            boxprops=dict(edgecolor="#4a4a4a", linewidth=1.2),
        )
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.18)

    for role in role_order:
        sub = group.loc[group["group_role"].astype(str).eq(role)].copy()
        if sub.empty:
            continue
        values = pd.to_numeric(sub["kappa_group_mean"], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        x = np.full(values.shape[0], role_positions[role], dtype=float) + rng.uniform(-0.085, 0.085, size=values.shape[0])
        ax0.scatter(
            x,
            values,
            s=13,
            alpha=0.28,
            color=role_colors[role],
            edgecolors="none",
            zorder=2,
        )

    median_gap = medians.get("active", np.nan) - medians.get("other_null", np.nan)
    ax0.set_xlim(-0.48, 1.48)
    ax0.set_ylim(-0.01, 0.56)
    ax0.set_xticks([role_positions[r] for r in role_order], labels=[role_names[r] for r in role_order])
    ax0.set_ylabel("Posterior mean $\\kappa_g$")
    ax0.set_title("", fontsize=12)
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
    point_colors = [role_colors["active"] if flag else role_colors["other_null"] for flag in trend["is_active_group"].tolist()]
    tick_labels = _figure2_strength_labels(trend)
    ax1.plot(
        xvals,
        trend["mean"].to_numpy(dtype=float),
        color="black",
        linewidth=1.7,
        zorder=2,
    )
    ax1.vlines(
        xvals,
        trend["ci_lo"].to_numpy(dtype=float),
        trend["ci_hi"].to_numpy(dtype=float),
        colors=point_colors,
        linewidth=1.7,
        zorder=2,
    )
    ax1.scatter(
        xvals,
        trend["mean"].to_numpy(dtype=float),
        s=42,
        c=point_colors,
        edgecolors="white",
        linewidths=0.7,
        zorder=3,
    )
    ax1.set_xlim(-0.55, len(trend) - 0.45)
    ax1.set_ylim(-0.01, 0.56)
    ax1.set_xticks(xvals, labels=tick_labels)
    ax1.set_xlabel("True group strength")
    ax1.set_ylabel("Mean posterior $\\kappa_g$")
    ax1.set_title("", fontsize=12)
    ax1.legend(
        handles=[
            plt.Line2D([0], [0], color=role_colors["other_null"], marker="o", lw=1.5, markersize=5.5, label="null group"),
            plt.Line2D([0], [0], color=role_colors["active"], marker="o", lw=1.5, markersize=5.5, label="active group"),
            plt.Line2D([0], [0], color="black", lw=1.7, label="mean trend"),
        ],
        loc="upper left",
        fontsize=8.8,
        frameon=False,
    )
    _panel_tag(ax1, "B")

    fig.suptitle("")
    out_path = out_dir / "figure2_group_separation.png"
    _save(fig, out_path)
    return str(out_path)


def plot_figure3_correlation_ambiguity(df: Any, out_dir: Path | str) -> str | None:
    frame = _as_frame(df)
    out_dir = ensure_dir(out_dir)
    needed = {"rho_within", "within_group_pattern", "gr_minus_rhs_mse_overall", "kappa_gap"}
    if frame.empty or not needed.issubset(set(frame.columns)):
        return None
    pd = load_pandas()
    frame = frame.copy()
    frame["rho_within"] = pd.to_numeric(frame["rho_within"], errors="coerce")
    frame["kappa_gap"] = pd.to_numeric(frame["kappa_gap"], errors="coerce")
    frame["gr_minus_rhs_mse_overall"] = pd.to_numeric(frame["gr_minus_rhs_mse_overall"], errors="coerce")

    rho_vals = sorted(frame["rho_within"].dropna().unique().tolist())
    pattern_priority = ["mixed_decoy", "concentrated"]
    present_patterns = [p for p in pattern_priority if p in set(frame["within_group_pattern"].astype(str))]
    extras = [p for p in sorted(frame["within_group_pattern"].astype(str).unique()) if p not in set(present_patterns)]
    patterns = present_patterns + extras
    if not rho_vals or not patterns:
        return None

    delta_mat = np.full((len(patterns), len(rho_vals)), np.nan, dtype=float)
    gap_mat = np.full((len(patterns), len(rho_vals)), np.nan, dtype=float)
    for yi, pat in enumerate(patterns):
        for xi, rho in enumerate(rho_vals):
            cell = frame.loc[
                frame["within_group_pattern"].astype(str).eq(pat)
                & np.isclose(frame["rho_within"].astype(float), float(rho))
            ]
            if cell.empty:
                continue
            row = cell.iloc[0]
            delta_mat[yi, xi] = float(row["gr_minus_rhs_mse_overall"])
            gap_mat[yi, xi] = float(row["kappa_gap"])

    x_labels = [f"$\\rho_w = {rho:.1f}$" for rho in rho_vals]
    y_labels = [pat.replace("_", " ") for pat in patterns]

    finite_delta = delta_mat[np.isfinite(delta_mat)]
    if finite_delta.size > 0:
        flat_idx = int(np.nanargmin(delta_mat))
        highlight = (flat_idx // delta_mat.shape[1], flat_idx % delta_mat.shape[1])
    else:
        highlight = None

    fig = plt.figure(figsize=(12.6, 5.4))
    fig.patch.set_facecolor("#fcfcfa")
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.05, 1.0],
        left=0.06,
        right=0.985,
        top=0.84,
        bottom=0.20,
        wspace=0.28,
    )
    ax0 = fig.add_subplot(outer[0, 0])
    ax1 = fig.add_subplot(outer[0, 1])

    _styled_heatmap(
        ax0,
        delta_mat,
        x_labels,
        y_labels,
        cmap=DIVERGING_CMAP,
        title="Paired overall-MSE difference (negative favors GR-RHS)",
        cbar_label="GR-RHS - RHS overall MSE",
        fmt="{:+.3f}",
        center_zero=True,
        highlight=highlight,
        highlight_label="largest gain",
    )
    _panel_tag(ax0, "A")

    _styled_heatmap(
        ax1,
        gap_mat,
        x_labels,
        y_labels,
        cmap=SEQUENTIAL_CMAP,
        title="GR-RHS gate stays identifiable in every cell",
        cbar_label="GR-RHS $\\kappa$ gap",
        fmt="{:.3f}",
        center_zero=False,
    )
    _panel_tag(ax1, "B")

    if "n_effective_pairs" in frame.columns and frame["n_effective_pairs"].notna().any():
        n_eff = int(float(frame["n_effective_pairs"].dropna().iloc[0]))
        n_text = f"{n_eff} paired replicates per cell"
    elif "n_paired" in frame.columns and frame["n_paired"].notna().any():
        n_text = f"{int(float(frame['n_paired'].dropna().iloc[0]))} paired replicates per cell"
    else:
        n_text = ""
    if highlight is not None and finite_delta.size > 0:
        edge = float(delta_mat[highlight])
        footer_pieces = [
            f"Largest gain at {y_labels[highlight[0]]} / {x_labels[highlight[1]]}: {edge:+.3f}",
        ]
        if n_text:
            footer_pieces.append(n_text)
        fig.text(
            0.5,
            0.07,
            " | ".join(footer_pieces),
            ha="center",
            va="center",
            fontsize=9.5,
            color="#5b6770",
            bbox=_card_kwargs(pad=0.28, facecolor="#ffffff"),
        )

    fig.suptitle("M2. Correlation stress: GR-RHS sharpens in the ambiguity regime", y=0.94, fontsize=15, fontweight="bold")
    out_path = out_dir / "figure3_correlation_ambiguity.png"
    _save(fig, out_path)
    return str(out_path)


def plot_figure4_representative_profile(df: Any, out_dir: Path | str) -> str | None:
    frame = _as_frame(df)
    out_dir = ensure_dir(out_dir)
    needed = {"group_id", "kappa_group_mean", "group_role", "method"}
    if frame.empty or not needed.issubset(set(frame.columns)):
        return None
    pd = load_pandas()
    frame = frame.copy()
    frame["group_id"] = pd.to_numeric(frame["group_id"], errors="coerce")
    frame["kappa_group_mean"] = pd.to_numeric(frame["kappa_group_mean"], errors="coerce")
    methods = ["GR_RHS", "RHS"]
    present = [m for m in methods if m in set(frame["method"].astype(str))]
    if not present:
        present = sorted(frame["method"].astype(str).unique().tolist())

    fig = plt.figure(figsize=(5.6 * len(present) + 0.4, 5.0))
    fig.patch.set_facecolor("#fcfcfa")
    outer = fig.add_gridspec(
        1,
        len(present),
        left=0.07,
        right=0.985,
        top=0.84,
        bottom=0.22,
        wspace=0.18,
    )
    short_tags = {"active": "A", "decoy_null": "D", "other_null": "N"}
    role_rank = {"other_null": 0, "decoy_null": 1, "active": 2}
    panel_letters = ["A", "B", "C", "D"]
    for idx, method in enumerate(present):
        ax = fig.add_subplot(outer[0, idx])
        sub = frame.loc[frame["method"].astype(str).eq(method)].copy()
        sub = sub.sort_values(["group_id"], kind="stable")
        finite = sub["kappa_group_mean"].replace([np.inf, -np.inf], np.nan).notna()
        title = str(sub["method_label"].iloc[0]) if "method_label" in sub.columns and not sub.empty else method.replace("_", "-")
        title = title.replace(" [stan_rstanarm_hs]", "")
        if not finite.any():
            _draw_missing_panel(
                ax,
                title,
                "Group-level $\\kappa_g$ is not defined for this method,\nso this profile is intentionally unavailable.",
                xlabel="Group id",
            )
            ax.set_ylim(-0.02, 1.02)
            ax.axhline(0.5, color="#8a949c", lw=1.0, ls="--")
            _panel_tag(ax, panel_letters[idx])
            continue

        sub = sub.loc[finite].copy()
        sub["role_rank"] = sub["group_role"].astype(str).map(lambda x: role_rank.get(str(x), 99))
        sub["role_order"] = sub.groupby("group_role", dropna=False).cumcount() + 1
        sub = sub.sort_values(["role_rank", "group_id"], kind="stable").reset_index(drop=True)
        ys = sub["kappa_group_mean"].astype(float).to_numpy()
        roles = sub["group_role"].astype(str).tolist()
        colors = [_group_role_color(role) for role in roles]
        plot_x = np.arange(len(sub), dtype=float)

        xtick_labels: list[str] = []
        for row in sub.itertuples(index=False):
            role = str(row.group_role)
            order = int(row.role_order)
            if role == "active":
                xtick_labels.append(f"active {order}")
            elif role == "decoy_null":
                xtick_labels.append("decoy")
            elif role == "other_null":
                xtick_labels.append(f"null {order}")
            else:
                xtick_labels.append(f"group {int(row.group_id)}")

        ax.plot(plot_x, ys, color="#8d98a6", lw=2.0, alpha=0.95, zorder=2)
        ax.scatter(plot_x, ys, s=120, c=colors, edgecolors="white", linewidths=0.9, zorder=3)
        for xi, yi, role in zip(plot_x, ys, roles):
            ax.text(float(xi), float(yi) + 0.018, short_tags.get(role, "?"), ha="center", va="bottom", fontsize=8.5, color="#5b6770")

        decoy_mask = np.array([role == "decoy_null" for role in roles])
        if method == "GR_RHS" and decoy_mask.any():
            di = int(np.argmax(decoy_mask))
            ax.annotate(
                "decoy stays low",
                xy=(float(plot_x[di]), float(ys[di])),
                xytext=(0.05, 0.92),
                textcoords="axes fraction",
                arrowprops={"arrowstyle": "->", "color": "#5b6770", "linewidth": 1.0},
                fontsize=9,
                color="#5b6770",
                ha="left",
                va="top",
            )

        ax.set_title(title)
        ax.set_xlabel("Group role")
        ax.set_xticks(plot_x, labels=xtick_labels)
        ax.set_ylim(-0.02, max(0.42, float(np.nanmax(ys)) + 0.10))
        ax.set_xlim(-0.45, len(sub) - 0.55 if len(sub) > 0 else 0.45)
        if idx == 0:
            ax.set_ylabel("Posterior mean $\\kappa_g$")
        _style_axis(ax)
        ax.tick_params(axis="x", labelrotation=0)
        ax.axhline(0.5, color="#8a949c", lw=1.0, ls="--", alpha=0.7)
        _panel_tag(ax, panel_letters[idx])

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=_group_role_color("active"), markersize=9, label="active (A)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=_group_role_color("decoy_null"), markersize=9, label="decoy null (D)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=_group_role_color("other_null"), markersize=9, label="other null (N)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.05), fontsize=9)
    fig.suptitle("Representative mixed-decoy replicate: posterior group shrinkage", y=0.94, fontsize=14, fontweight="bold")
    out_path = out_dir / "figure4_representative_profile.png"
    _save(fig, out_path)
    return str(out_path)


def plot_figure5_complexity_unit(df: Any, out_dir: Path | str) -> str | None:
    frame = _as_frame(df)
    out_dir = ensure_dir(out_dir)
    needed = {"complexity_pattern", "within_group_pattern", "kappa_gap", "gr_minus_rhs_mse_overall"}
    if frame.empty or not needed.issubset(set(frame.columns)):
        return None
    pd = load_pandas()
    frame = frame.copy()
    for col in [
        "kappa_gap",
        "gr_minus_rhs_mse_overall",
        "gr_minus_rhs_mse_overall_ci95_lo",
        "gr_minus_rhs_mse_overall_ci95_hi",
        "gr_mse_overall",
        "rhs_mse_overall",
        "n_effective_pairs",
        "n_paired",
    ]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["complexity_pattern"] = frame["complexity_pattern"].astype(str)
    frame["within_group_pattern"] = frame["within_group_pattern"].astype(str)

    complexity_order = ["few_groups", "many_groups"]
    within_order = ["concentrated", "distributed"]
    complexity_levels = [c for c in complexity_order if c in set(frame["complexity_pattern"])]
    within_levels = [w for w in within_order if w in set(frame["within_group_pattern"])]
    if not complexity_levels or not within_levels:
        return None

    delta_mat = np.full((len(within_levels), len(complexity_levels)), np.nan, dtype=float)
    gap_mat = np.full((len(within_levels), len(complexity_levels)), np.nan, dtype=float)
    extra_labels: dict[tuple[int, int], str] = {}
    for yi, within in enumerate(within_levels):
        for xi, complexity in enumerate(complexity_levels):
            cell = frame.loc[
                frame["within_group_pattern"].eq(within)
                & frame["complexity_pattern"].eq(complexity)
            ]
            if cell.empty:
                continue
            row = cell.iloc[0]
            delta_mat[yi, xi] = float(row["gr_minus_rhs_mse_overall"])
            gap_mat[yi, xi] = float(row["kappa_gap"])
            extra_labels[(yi, xi)] = f"$\\kappa$ gap {float(row['kappa_gap']):.3f}"

    x_labels = [c.replace("_", " ") for c in complexity_levels]
    y_labels = [w for w in within_levels]
    finite_delta = delta_mat[np.isfinite(delta_mat)]
    highlight = None
    if finite_delta.size > 0:
        idx_min = int(np.nanargmin(delta_mat))
        highlight = (idx_min // delta_mat.shape[1], idx_min % delta_mat.shape[1])

    bar_rows = []
    for yi, within in enumerate(within_levels):
        for xi, complexity in enumerate(complexity_levels):
            cell = frame.loc[
                frame["within_group_pattern"].eq(within)
                & frame["complexity_pattern"].eq(complexity)
            ]
            if cell.empty:
                continue
            row = cell.iloc[0]
            bar_rows.append(
                {
                    "label": f"{complexity.replace('_', ' ')}\n{within}",
                    "delta": float(row["gr_minus_rhs_mse_overall"]),
                    "lo": float(row["gr_minus_rhs_mse_overall_ci95_lo"]) if "gr_minus_rhs_mse_overall_ci95_lo" in row.index else np.nan,
                    "hi": float(row["gr_minus_rhs_mse_overall_ci95_hi"]) if "gr_minus_rhs_mse_overall_ci95_hi" in row.index else np.nan,
                    "highlight": (yi, xi) == highlight,
                }
            )
    bar_df = pd.DataFrame(bar_rows).sort_values("delta", kind="stable").reset_index(drop=True)

    fig = plt.figure(figsize=(13.0, 5.6))
    fig.patch.set_facecolor("#fcfcfa")
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.0, 1.15],
        left=0.06,
        right=0.985,
        top=0.84,
        bottom=0.20,
        wspace=0.28,
    )
    ax0 = fig.add_subplot(outer[0, 0])
    ax1 = fig.add_subplot(outer[0, 1])

    _styled_heatmap(
        ax0,
        delta_mat,
        x_labels,
        y_labels,
        cmap=DIVERGING_CMAP,
        title="Edge concentrates where signal is packed",
        cbar_label="GR-RHS - RHS overall MSE",
        fmt="{:+.3f}",
        center_zero=True,
        extra_labels=extra_labels,
        highlight=highlight,
        highlight_label="largest edge",
    )
    _panel_tag(ax0, "A")

    y_positions = np.arange(len(bar_df), dtype=float)
    colors = ["#244d57" if flag else "#91a9a4" for flag in bar_df["highlight"].tolist()]
    ax1.hlines(y_positions, 0.0, bar_df["delta"], color="#d5dbd6", linewidth=3.0)
    ax1.scatter(bar_df["delta"], y_positions, s=110, c=colors, edgecolors="white", linewidths=0.9, zorder=3)
    has_ci = bar_df["lo"].notna().any() and bar_df["hi"].notna().any()
    if has_ci:
        for ypos, row in zip(y_positions, bar_df.itertuples(index=False)):
            if np.isfinite(row.lo) and np.isfinite(row.hi):
                ax1.hlines(ypos, row.lo, row.hi, color="#a8b4c1", linewidth=2.0, alpha=0.85, zorder=2)
    for ypos, row in zip(y_positions, bar_df.itertuples(index=False)):
        anchor = max(float(row.delta), float(row.hi) if np.isfinite(row.hi) else float(row.delta))
        ax1.text(anchor + 0.004, ypos, f"{float(row.delta):+.3f}", ha="left", va="center", fontsize=9, color="#1f2933")
    ax1.axvline(0.0, color="#5b6770", linewidth=1.0, linestyle="--")
    ax1.set_yticks(y_positions, labels=bar_df["label"].tolist())
    ax1.invert_yaxis()
    ax1.set_xlabel("Paired overall-MSE difference (negative favors GR-RHS)")
    ax1.set_title("Setting-level scope condition")
    _style_axis(ax1, grid_axis="x")
    _panel_tag(ax1, "B")

    if highlight is not None and finite_delta.size > 0:
        strongest = float(delta_mat[highlight])
        weakest = float(np.nanmax(delta_mat))
        footer = (
            f"Strongest edge: {y_labels[highlight[0]]} / {x_labels[highlight[1]]} ({strongest:+.3f}). "
            f"Weakest cell trends toward zero ({weakest:+.3f})."
        )
        fig.text(
            0.5,
            0.07,
            footer,
            ha="center",
            va="center",
            fontsize=9.5,
            color="#5b6770",
            bbox=_card_kwargs(pad=0.28, facecolor="#ffffff"),
        )

    fig.suptitle("M3. Scope condition: the advantage tracks group complexity", y=0.94, fontsize=15, fontweight="bold")
    out_path = out_dir / "figure5_complexity_unit.png"
    _save(fig, out_path)
    return str(out_path)


def plot_figure6_ablation(summary_df: Any, delta_df: Any, out_dir: Path | str) -> str | None:
    summary = _as_frame(summary_df)
    delta = _as_frame(delta_df)
    out_dir = ensure_dir(out_dir)
    if delta.empty or "metric" not in delta.columns:
        return None
    pd = load_pandas()
    delta = delta.copy()
    summary = summary.copy()
    for col in ["mean_diff", "ci95_lo", "ci95_hi"]:
        if col in delta.columns:
            delta[col] = pd.to_numeric(delta[col], errors="coerce")
    for col in ["kappa_gap", "null_group_mse", "signal_group_mse", "tau_ratio_to_oracle", "n_paired"]:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")

    metrics = [
        ("kappa_gap", "$\\Delta\\kappa$ gap", "Loss of group separation"),
        ("null_group_mse", "$\\Delta$ null-group MSE", "Cost on null groups"),
        ("signal_group_mse", "$\\Delta$ signal-group MSE", "Cost on signal groups"),
    ]
    delta = delta.loc[delta["metric"].astype(str).isin([m for m, _, _ in metrics])].copy()
    if delta.empty:
        return None

    variant_order = [
        "GR_RHS_oracle",
        "GR_RHS_fixed_10x",
        "GR_RHS_no_local_scales",
        "GR_RHS_shared_kappa",
        "GR_RHS_no_kappa",
        "RHS_oracle",
    ]
    label_map = {
        "GR_RHS_oracle": "oracle $\\tau_0$",
        "GR_RHS_fixed_10x": "fixed 10x $\\tau_0$",
        "GR_RHS_no_local_scales": "no local scales",
        "GR_RHS_shared_kappa": "shared $\\kappa$",
        "GR_RHS_no_kappa": "no $\\kappa$",
        "RHS_oracle": "RHS oracle",
    }
    available = list(delta["method"].astype(str).unique())
    variants = [v for v in variant_order if v in available]
    extras = [v for v in available if v not in set(variants) and v != "GR_RHS"]
    variants.extend(sorted(extras))
    if not variants:
        return None

    fig = plt.figure(figsize=(11.6, 4.8))
    fig.patch.set_facecolor("white")
    outer = fig.add_gridspec(
        1,
        2,
        left=0.08,
        right=0.985,
        top=0.88,
        bottom=0.18,
        wspace=0.22,
    )
    axes = {
        "kappa_gap": fig.add_subplot(outer[0, 0]),
        "tau": fig.add_subplot(outer[0, 1]),
    }

    def _draw_forest(ax, metric_key: str, metric_xlabel: str, metric_title: str, show_ylabels: bool):
        sub = delta.loc[delta["metric"].astype(str).eq(metric_key)].copy()
        finite = pd.concat([sub["mean_diff"], sub.get("ci95_lo"), sub.get("ci95_hi")]).dropna().astype(float)
        if finite.empty:
            xmin, xmax = -1.0, 1.0
        else:
            xmin, xmax = float(finite.min()), float(finite.max())
            spread = max(xmax - xmin, 0.02)
            pad = 0.18 * spread
            xmin = min(xmin - pad, -0.02 if xmax >= 0 else xmin - pad)
            xmax = max(xmax + pad, 0.02 if xmin <= 0 else xmax + pad)
        ax.set_xlim(xmin, xmax)
        for ypos, method in enumerate(variants):
            row_q = sub.loc[sub["method"].astype(str).eq(method)]
            color = _method_color(method)
            if row_q.empty:
                ax.text(xmax - 0.02 * (xmax - xmin), ypos, "N/A", ha="right", va="center", fontsize=8.6, color="#8a949c")
                continue
            row = row_q.iloc[0]
            mean_val = float(row["mean_diff"])
            lo = float(row["ci95_lo"]) if "ci95_lo" in row.index and np.isfinite(float(row["ci95_lo"])) else np.nan
            hi = float(row["ci95_hi"]) if "ci95_hi" in row.index and np.isfinite(float(row["ci95_hi"])) else np.nan
            if np.isfinite(lo) and np.isfinite(hi):
                ax.hlines(ypos, lo, hi, color=color, linewidth=2.6, alpha=0.92, zorder=2)
            ax.scatter(mean_val, ypos, s=100, color=color, edgecolors="white", linewidths=0.9, zorder=3)
            x_pad = 0.02 * (xmax - xmin)
            if mean_val >= 0:
                ax.text(mean_val + x_pad, ypos, f"{mean_val:+.3f}", ha="left", va="center", fontsize=8.7, color="#1f2933")
            else:
                ax.text(mean_val - x_pad, ypos, f"{mean_val:+.3f}", ha="right", va="center", fontsize=8.7, color="#1f2933")
        ax.axvline(0.0, color="#5b6770", linewidth=1.0, linestyle="--")
        ax.set_title(metric_title, pad=10)
        ax.set_xlabel(metric_xlabel)
        ax.set_yticks(np.arange(len(variants)), labels=[label_map.get(v, v.replace("_", " ")) for v in variants] if show_ylabels else [])
        ax.invert_yaxis()
        _style_axis(ax, grid_axis="x")
        ax.set_facecolor("white")

    _draw_forest(axes["kappa_gap"], "kappa_gap", "$\\Delta$ $\\kappa$ gap (vs GR-RHS)", "Loss of group separation", show_ylabels=True)
    _panel_tag(axes["kappa_gap"], "A")

    ax_tau = axes["tau"]
    tau_values: dict[str, float] = {}
    if "tau_ratio_to_oracle" in summary.columns:
        for method in variants:
            cell = summary.loc[summary["method"].astype(str).eq(method)]
            if cell.empty:
                continue
            val = float(cell["tau_ratio_to_oracle"].iloc[0])
            if np.isfinite(val):
                tau_values[method] = val
    finite_tau = list(tau_values.values())
    tau_max = max(finite_tau) if finite_tau else 1.0
    ax_tau.set_xlim(0.0, tau_max + 0.7)
    for ypos, method in enumerate(variants):
        color = _method_color(method)
        if method not in tau_values:
            ax_tau.text(0.10, ypos, "N/A", ha="left", va="center", fontsize=8.6, color="#8a949c")
            continue
        ratio = tau_values[method]
        ax_tau.hlines(ypos, 0.0, ratio, color="#d7ddd8", linewidth=2.4)
        ax_tau.scatter(ratio, ypos, s=100, color=color, edgecolors="white", linewidths=0.9, zorder=3)
        ax_tau.text(ratio + 0.10, ypos, f"{ratio:.2f}", ha="left", va="center", fontsize=8.7, color="#1f2933")
    ax_tau.axvline(1.0, color="#5b6770", linewidth=1.0, linestyle="--")
    ax_tau.text(
        1.0,
        1.02,
        "oracle",
        transform=ax_tau.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#5b6770",
    )
    ax_tau.set_title("$\\tau_0$ calibration", pad=10)
    ax_tau.set_xlabel("$\\tau_0$ / oracle $\\tau_0$")
    ax_tau.set_yticks(np.arange(len(variants)), labels=[])
    ax_tau.invert_yaxis()
    _style_axis(ax_tau, grid_axis="x")
    ax_tau.set_facecolor("white")
    _panel_tag(ax_tau, "B")

    fig.suptitle("")
    out_path = out_dir / "figure6_ablation.png"
    _save(fig, out_path)
    return str(out_path)




def build_mechanism_figures_from_results_dir(results_dir: Path | str) -> dict[str, str]:
    pd = load_pandas()
    root = resolve_history_results_dir(
        results_dir,
        required_files=(
            "paper_tables/figure_data/figure2_group_separation.csv",
            "paper_tables/figure_data/figure3_correlation_ambiguity.csv",
            "paper_tables/figure_data/figure4_representative_profile.csv",
            "paper_tables/figure_data/figure5_complexity_unit.csv",
            "paper_tables/figure_data/figure6_ablation.csv",
            "paper_tables/figure_data/figure6_ablation_deltas.csv",
        ),
    )
    paper_dir = ensure_dir(root / "paper_tables")
    fig_data_dir = paper_dir / "figure_data"
    fig_out_dir = ensure_dir(root / "figures")

    outputs: dict[str, str] = {}
    outputs["figure1_mechanism_schematic"] = plot_figure1_mechanism_schematic(fig_out_dir)

    maybe = plot_figure2_group_separation(_read_csv_or_empty(fig_data_dir / "figure2_group_separation.csv"), fig_out_dir)
    if maybe:
        outputs["figure2_group_separation"] = maybe
    maybe = plot_figure3_correlation_ambiguity(_read_csv_or_empty(fig_data_dir / "figure3_correlation_ambiguity.csv"), fig_out_dir)
    if maybe:
        outputs["figure3_correlation_ambiguity"] = maybe
    maybe = plot_figure4_representative_profile(_read_csv_or_empty(fig_data_dir / "figure4_representative_profile.csv"), fig_out_dir)
    if maybe:
        outputs["figure4_representative_profile"] = maybe
    maybe = plot_figure5_complexity_unit(_read_csv_or_empty(fig_data_dir / "figure5_complexity_unit.csv"), fig_out_dir)
    if maybe:
        outputs["figure5_complexity_unit"] = maybe
    summary_df = _read_csv_or_empty(fig_data_dir / "figure6_ablation.csv")
    delta_df = _read_csv_or_empty(fig_data_dir / "figure6_ablation_deltas.csv")
    maybe = plot_figure6_ablation(summary_df, delta_df, fig_out_dir)
    if maybe:
        outputs["figure6_ablation"] = maybe
    return outputs
