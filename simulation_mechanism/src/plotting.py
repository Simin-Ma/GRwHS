from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulation_project.src.utils import load_pandas

from .utils import ensure_dir


_METHOD_COLORS: dict[str, str] = {
    "GR_RHS": "#1f77b4",
    "RHS": "#ff7f0e",
    "RHS_oracle": "#ff7f0e",
    "GR_RHS_fixed_10x": "#6b8e23",
    "GR_RHS_oracle": "#2ca02c",
    "GR_RHS_no_local_scales": "#8c564b",
    "GR_RHS_shared_kappa": "#9467bd",
    "GR_RHS_no_kappa": "#d62728",
}


_GROUP_ROLE_COLORS: dict[str, str] = {
    "active": "#1f77b4",
    "decoy_null": "#d62728",
    "other_null": "#9aa0a6",
}


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
    fig.savefig(history_dir / f"{path.stem}_{ts}{path.suffix}", **save_kws)
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
    if frame.empty or "method" not in frame.columns:
        return None
    frame = frame.copy()
    frame["method_sort"] = frame["method"].map(lambda x: 0 if str(x) == "GR_RHS" else 1)
    frame = frame.sort_values(["method_sort", "method"], kind="stable")
    methods = frame["method_label"].astype(str).tolist()
    x = np.arange(len(methods), dtype=float)
    kappa_gap = frame["kappa_gap"].to_numpy(dtype=float) if "kappa_gap" in frame.columns else np.full(len(frame), np.nan)
    group_auroc = frame["group_auroc"].to_numpy(dtype=float) if "group_auroc" in frame.columns else np.full(len(frame), np.nan)
    mse_overall = frame["mse_overall"].to_numpy(dtype=float) if "mse_overall" in frame.columns else np.full(len(frame), np.nan)
    n_eff = frame["n_paired"].to_numpy(dtype=float) if "n_paired" in frame.columns else np.full(len(frame), np.nan)
    colors = [_method_color(m) for m in frame["method"].astype(str).tolist()]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    bar_heights = np.where(np.isfinite(kappa_gap), kappa_gap, 0.0)
    bars = ax.bar(x, bar_heights, color=colors, alpha=0.90, width=0.62, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x, labels=methods)
    ax.set_ylabel("kappa gap")
    ax.set_title("Signal-null group separation")
    ax.grid(axis="y", alpha=0.22)
    y_max = np.nanmax(kappa_gap[np.isfinite(kappa_gap)]) if np.isfinite(kappa_gap).any() else 1.0
    ax.set_ylim(min(0.0, np.nanmin(kappa_gap[np.isfinite(kappa_gap)]) if np.isfinite(kappa_gap).any() else 0.0) - 0.05, y_max * 1.25 + 0.08)
    for idx, bar in enumerate(bars):
        if np.isfinite(kappa_gap[idx]):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.02, f"{kappa_gap[idx]:.3f}", ha="center", va="bottom", fontsize=9)
        else:
            bar.set_hatch("//")
            ax.text(bar.get_x() + bar.get_width() / 2.0, 0.03, "N/A", ha="center", va="bottom", fontsize=8, color="#444444")
        text_bits = []
        if np.isfinite(mse_overall[idx]):
            text_bits.append(f"MSE={mse_overall[idx]:.3f}")
        if np.isfinite(n_eff[idx]):
            text_bits.append(f"n={int(n_eff[idx])}")
        if text_bits:
            ax.text(bar.get_x() + bar.get_width() / 2.0, ax.get_ylim()[0] + 0.03, "\n".join(text_bits), ha="center", va="bottom", fontsize=8, color="#444444")

    ax2 = axes[1]
    ax2.scatter(x, group_auroc, s=120, color=colors, zorder=3)
    ax2.plot(x, group_auroc, color="#666666", lw=1.2, alpha=0.7, zorder=2)
    ax2.set_xticks(x, labels=methods)
    ax2.set_ylabel("group AUROC")
    ax2.set_title("Group recovery ranking")
    ax2.grid(axis="y", alpha=0.22)
    ax2.set_ylim(0.45, 1.02)
    for idx in range(len(x)):
        if np.isfinite(group_auroc[idx]):
            ax2.text(x[idx], group_auroc[idx] + 0.02, f"{group_auroc[idx]:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Figure 2. Group-level separation in GA-V2-A", fontsize=13, fontweight="bold")
    out_path = out_dir / "figure2_group_separation.png"
    _save(fig, out_path)
    return str(out_path)


def plot_figure3_correlation_ambiguity(df: Any, out_dir: Path | str) -> str | None:
    frame = _as_frame(df)
    out_dir = ensure_dir(out_dir)
    needed = {"rho_within", "within_group_pattern", "gr_minus_rhs_mse_overall", "kappa_gap"}
    if frame.empty or not needed.issubset(set(frame.columns)):
        return None
    patterns = sorted(frame["within_group_pattern"].astype(str).unique().tolist())
    rho_vals = sorted(frame["rho_within"].astype(float).unique().tolist())
    mse_mat = np.full((len(patterns), len(rho_vals)), np.nan, dtype=float)
    gap_mat = np.full((len(patterns), len(rho_vals)), np.nan, dtype=float)
    ann = np.full((len(patterns), len(rho_vals)), np.nan, dtype=float)
    for i, pat in enumerate(patterns):
        for j, rho in enumerate(rho_vals):
            sub = frame.loc[
                frame["within_group_pattern"].astype(str).eq(pat)
                & np.isclose(frame["rho_within"].astype(float), float(rho))
            ]
            if sub.empty:
                continue
            mse_mat[i, j] = float(sub["gr_minus_rhs_mse_overall"].iloc[0])
            gap_mat[i, j] = float(sub["kappa_gap"].iloc[0])
            if "n_effective_pairs" in sub.columns and np.isfinite(pd := float(sub["n_effective_pairs"].iloc[0])):
                ann[i, j] = pd
            elif "n_paired" in sub.columns and np.isfinite(pd2 := float(sub["n_paired"].iloc[0])):
                ann[i, j] = pd2

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    x_labels = [f"{rho:.2f}" for rho in rho_vals]
    y_labels = [pat.replace("_", "\n") for pat in patterns]
    im_a = _heatmap_panel(
        axes[0],
        mse_mat,
        x_labels,
        y_labels,
        cmap="RdBu_r",
        center=0.0,
        fmt="{:+.3f}",
        annotation=ann,
        title="GR-RHS - RHS on paired overall MSE",
    )
    axes[0].set_xlabel("rho_within")
    axes[0].set_ylabel("within-group pattern")
    cbar_a = fig.colorbar(im_a, ax=axes[0], fraction=0.046, pad=0.04)
    cbar_a.set_label("negative = GR-RHS better")

    im_b = _heatmap_panel(
        axes[1],
        gap_mat,
        x_labels,
        y_labels,
        cmap="YlGnBu",
        center=None,
        fmt="{:.3f}",
        annotation=ann,
        title="GR-RHS kappa gap",
    )
    axes[1].set_xlabel("rho_within")
    axes[1].set_ylabel("within-group pattern")
    cbar_b = fig.colorbar(im_b, ax=axes[1], fraction=0.046, pad=0.04)
    cbar_b.set_label("kappa gap")

    fig.suptitle("Figure 3. Correlation stress under structural ambiguity", fontsize=13, fontweight="bold")
    out_path = out_dir / "figure3_correlation_ambiguity.png"
    _save(fig, out_path)
    return str(out_path)


def plot_figure4_representative_profile(df: Any, out_dir: Path | str) -> str | None:
    frame = _as_frame(df)
    out_dir = ensure_dir(out_dir)
    needed = {"group_id", "kappa_group_mean", "group_role", "method"}
    if frame.empty or not needed.issubset(set(frame.columns)):
        return None
    methods = ["GR_RHS", "RHS"]
    present = [m for m in methods if m in set(frame["method"].astype(str))]
    if not present:
        present = sorted(frame["method"].astype(str).unique().tolist())

    fig, axes = plt.subplots(1, len(present), figsize=(5.2 * len(present), 4.8), sharey=True)
    if len(present) == 1:
        axes = [axes]
    for ax, method in zip(axes, present):
        sub = frame.loc[frame["method"].astype(str).eq(method)].copy()
        sub = sub.sort_values(["group_id"], kind="stable")
        xs = sub["group_id"].astype(int).to_numpy()
        ys = sub["kappa_group_mean"].astype(float).to_numpy()
        colors = [_group_role_color(role) for role in sub["group_role"].astype(str).tolist()]
        ax.plot(xs, ys, color="#888888", lw=1.4, zorder=1)
        ax.scatter(xs, ys, s=120, c=colors, edgecolors="white", linewidths=0.8, zorder=3)
        for _, row in sub.iterrows():
            label = "A" if bool(row.get("is_active_group", False)) else ("D" if bool(row.get("is_decoy_group", False)) else "N")
            ax.text(float(row["group_id"]), float(row["kappa_group_mean"]) + 0.025, label, ha="center", va="bottom", fontsize=8)
        ax.set_title(str(sub["method_label"].iloc[0]) if "method_label" in sub.columns else method)
        ax.set_xlabel("group id")
        ax.set_xticks(xs)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", alpha=0.22)
        ax.axhline(0.5, color="#444444", lw=1.0, ls="--")
    axes[0].set_ylabel("posterior mean kappa_g")
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=_group_role_color("active"), markersize=9, label="active"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=_group_role_color("decoy_null"), markersize=9, label="decoy null"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=_group_role_color("other_null"), markersize=9, label="other null"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.suptitle("Figure 4. Posterior group shrinkage in a representative mixed-decoy replicate", fontsize=13, fontweight="bold")
    out_path = out_dir / "figure4_representative_profile.png"
    _save(fig, out_path)
    return str(out_path)


def plot_figure5_complexity_unit(df: Any, out_dir: Path | str) -> str | None:
    frame = _as_frame(df)
    out_dir = ensure_dir(out_dir)
    needed = {"complexity_pattern", "within_group_pattern", "method", "kappa_gap", "mse_overall"}
    if frame.empty or not needed.issubset(set(frame.columns)):
        return None
    patterns = ["few_groups", "many_groups"]
    within_levels = sorted(frame["within_group_pattern"].astype(str).unique().tolist())
    methods = [m for m in ["GR_RHS", "RHS"] if m in set(frame["method"].astype(str))]
    if not methods:
        methods = sorted(frame["method"].astype(str).unique().tolist())

    fig, axes = plt.subplots(2, max(len(within_levels), 1), figsize=(5.2 * max(len(within_levels), 1), 7.4), sharex=True)
    if len(within_levels) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    width = 0.34
    for col_idx, within in enumerate(within_levels):
        sub = frame.loc[frame["within_group_pattern"].astype(str).eq(within)].copy()
        x = np.arange(len(patterns), dtype=float)
        for row_idx, metric in enumerate(["kappa_gap", "mse_overall"]):
            ax = axes[row_idx, col_idx]
            for m_idx, method in enumerate(methods):
                vals = []
                for pat in patterns:
                    cell = sub.loc[
                        sub["complexity_pattern"].astype(str).eq(pat)
                        & sub["method"].astype(str).eq(method)
                    ]
                    vals.append(float(cell[metric].iloc[0]) if not cell.empty else np.nan)
                offset = (m_idx - (len(methods) - 1) / 2.0) * width
                ax.bar(x + offset, vals, width=width, color=_method_color(method), alpha=0.88, label=method if row_idx == 0 else None)
                for pos, val in zip(x + offset, vals):
                    if np.isfinite(val):
                        ax.text(pos, val + (0.02 if metric == "kappa_gap" else 0.005), f"{val:.3f}", ha="center", va="bottom", fontsize=8)
            ax.set_title(within.replace("_", " "))
            ax.set_xticks(x, labels=[pat.replace("_", "\n") for pat in patterns])
            ax.grid(axis="y", alpha=0.22)
            ax.set_ylabel("kappa gap" if metric == "kappa_gap" else "overall MSE")
            if metric == "mse_overall":
                tie_cell = sub.loc[
                    sub["complexity_pattern"].astype(str).eq("many_groups")
                    & sub["method"].astype(str).isin(methods)
                ]
                if tie_cell.shape[0] >= 2:
                    vals = tie_cell.sort_values("method")["mse_overall"].astype(float).to_numpy()
                    if np.isfinite(vals).all() and abs(vals.max() - vals.min()) < 0.03:
                        ax.annotate(
                            "near tie",
                            xy=(1.0, float(np.nanmax(vals))),
                            xytext=(1.12, float(np.nanmax(vals)) + 0.03),
                            arrowprops=dict(arrowstyle="->", lw=1.0, color="#444444"),
                            fontsize=8,
                            color="#444444",
                        )
    handles = [plt.Line2D([0], [0], color=_method_color(m), lw=8, label=m.replace("_", "-")) for m in methods]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.01), fontsize=9)
    fig.suptitle("Figure 5. Complexity allocation under cleaned GA-V2-B", fontsize=13, fontweight="bold")
    out_path = out_dir / "figure5_complexity_unit.png"
    _save(fig, out_path)
    return str(out_path)


def plot_figure6_ablation(summary_df: Any, delta_df: Any, out_dir: Path | str) -> str | None:
    summary = _as_frame(summary_df)
    delta = _as_frame(delta_df)
    out_dir = ensure_dir(out_dir)
    if delta.empty or "metric" not in delta.columns:
        return None
    metrics = ["kappa_gap", "mse_overall", "mse_signal"]
    delta = delta.loc[delta["metric"].astype(str).isin(metrics)].copy()
    if delta.empty:
        return None
    if not summary.empty:
        baseline_rows = []
        for metric in metrics:
            metric_col = metric if metric in summary.columns else None
            if metric_col is None:
                continue
            for p0_val, sub in summary.loc[summary["method"].astype(str).eq("GR_RHS")].groupby("total_active_coeff", dropna=False, sort=False):
                if sub.empty:
                    continue
                n_eff = float(sub["n_paired"].iloc[0]) if "n_paired" in sub.columns and not sub["n_paired"].isna().all() else float("nan")
                baseline_rows.append(
                    {
                        "experiment_id": "M4",
                        "setting_id": str(sub["setting_id"].iloc[0]) if "setting_id" in sub.columns else "",
                        "total_active_coeff": p0_val,
                        "method": "GR_RHS",
                        "method_label": str(sub["method_label"].iloc[0]) if "method_label" in sub.columns else "GR-RHS",
                        "baseline_method": "GR_RHS",
                        "baseline_method_label": "GR-RHS",
                        "metric": metric,
                        "metric_direction": "larger_is_better" if metric == "kappa_gap" else "smaller_is_better",
                        "n_effective_pairs": n_eff,
                        "mean_diff": 0.0,
                        "std_diff": 0.0,
                        "se_diff": 0.0,
                        "ci95_lo": 0.0,
                        "ci95_hi": 0.0,
                        "wins_vs_baseline": 0,
                        "losses_vs_baseline": 0,
                        "ties_vs_baseline": int(n_eff) if np.isfinite(n_eff) else 0,
                    }
                )
        if baseline_rows:
            pd = load_pandas()
            delta = pd.concat([delta, pd.DataFrame(baseline_rows)], ignore_index=True)
    variants_order = [
        "GR_RHS",
        "GR_RHS_fixed_10x",
        "RHS_oracle",
        "GR_RHS_no_local_scales",
        "GR_RHS_shared_kappa",
        "GR_RHS_no_kappa",
    ]
    available = list(delta["method"].astype(str).unique())
    variants = [v for v in variants_order if v in available]
    extras = [v for v in available if v not in set(variants)]
    variants.extend(sorted(extras))
    fig, axes = plt.subplots(1, len(metrics), figsize=(12.5, 5.0), sharey=True)
    p0_vals = sorted(delta["total_active_coeff"].dropna().astype(int).unique().tolist()) if "total_active_coeff" in delta.columns and delta["total_active_coeff"].notna().any() else []
    for ax, metric in zip(axes, metrics):
        sub = delta.loc[delta["metric"].astype(str).eq(metric)].copy()
        y = np.arange(len(variants), dtype=float)
        for idx, variant in enumerate(variants):
            cell = sub.loc[sub["method"].astype(str).eq(variant)]
            if cell.empty:
                continue
            mean_val = float(cell["mean_diff"].mean())
            lo = float(cell["ci95_lo"].mean()) if "ci95_lo" in cell.columns else np.nan
            hi = float(cell["ci95_hi"].mean()) if "ci95_hi" in cell.columns else np.nan
            ax.scatter(mean_val, y[idx], s=110, color=_method_color(variant), zorder=3)
            if np.isfinite(lo) and np.isfinite(hi):
                ax.hlines(y[idx], lo, hi, colors=_method_color(variant), lw=2.2, zorder=2)
            if "wins_vs_baseline" in cell.columns and "losses_vs_baseline" in cell.columns:
                wins = int(cell["wins_vs_baseline"].sum())
                losses = int(cell["losses_vs_baseline"].sum())
                ties = int(cell["ties_vs_baseline"].sum()) if "ties_vs_baseline" in cell.columns else 0
                if variant != "GR_RHS":
                    ax.text(mean_val, y[idx] + 0.18, f"{wins}-{losses}-{ties}", ha="center", va="bottom", fontsize=7, color="#444444")
        ax.axvline(0.0, color="#444444", ls="--", lw=1.0)
        ax.set_title(metric.replace("_", " "))
        ax.grid(axis="x", alpha=0.22)
        ax.set_xlabel("delta vs GR-RHS")
    axes[0].set_yticks(np.arange(len(variants)), labels=[v.replace("_", " ") for v in variants])
    axes[0].set_ylabel("variant")
    fig.suptitle("Figure 6. Mechanism ablation for GR-RHS", fontsize=13, fontweight="bold")
    out_path = out_dir / "figure6_ablation.png"
    _save(fig, out_path)
    return str(out_path)


def build_mechanism_figures_from_results_dir(results_dir: Path | str) -> dict[str, str]:
    pd = load_pandas()
    root = Path(results_dir)
    paper_dir = ensure_dir(root / "paper_tables")
    fig_data_dir = paper_dir / "figure_data"
    fig_out_dir = ensure_dir(root / "figures")

    outputs: dict[str, str] = {}
    outputs["figure1_mechanism_schematic"] = plot_figure1_mechanism_schematic(fig_out_dir)

    maybe = plot_figure2_group_separation(pd.read_csv(fig_data_dir / "figure2_group_separation.csv"), fig_out_dir)
    if maybe:
        outputs["figure2_group_separation"] = maybe
    maybe = plot_figure3_correlation_ambiguity(pd.read_csv(fig_data_dir / "figure3_correlation_ambiguity.csv"), fig_out_dir)
    if maybe:
        outputs["figure3_correlation_ambiguity"] = maybe
    maybe = plot_figure4_representative_profile(pd.read_csv(fig_data_dir / "figure4_representative_profile.csv"), fig_out_dir)
    if maybe:
        outputs["figure4_representative_profile"] = maybe
    maybe = plot_figure5_complexity_unit(pd.read_csv(fig_data_dir / "figure5_complexity_unit.csv"), fig_out_dir)
    if maybe:
        outputs["figure5_complexity_unit"] = maybe
    summary_df = pd.read_csv(fig_data_dir / "figure6_ablation.csv")
    delta_df = pd.read_csv(fig_data_dir / "figure6_ablation_deltas.csv")
    maybe = plot_figure6_ablation(summary_df, delta_df, fig_out_dir)
    if maybe:
        outputs["figure6_ablation"] = maybe
    return outputs
