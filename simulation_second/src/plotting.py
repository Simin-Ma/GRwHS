from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulation_project.src.utils import load_pandas

from .utils import ensure_dir, resolve_history_results_dir


_METHOD_COLORS: dict[str, str] = {
    "GR_RHS": "#0f766e",
    "RHS": "#c2410c",
    "GHS_plus": "#2563eb",
    "GIGG_MMLE": "#7c3aed",
    "LASSO_CV": "#b45309",
    "OLS": "#4b5563",
}


def _method_color(name: str) -> str:
    return _METHOD_COLORS.get(str(name), "#6b7280")


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


def _draw_missing_panel(ax, title: str, message: str) -> None:
    ax.set_title(title)
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


def plot_figure1_coefficient_recovery_profile(df: Any, out_dir: Path | str) -> str | None:
    frame = _as_frame(df)
    out_dir = ensure_dir(out_dir)
    needed = {"method", "method_label", "plot_order", "true_beta", "estimated_beta", "group_id"}
    if frame.empty or not needed.issubset(set(frame.columns)):
        return None

    pd = load_pandas()
    frame = frame.copy()
    for col in [
        "plot_order",
        "group_id",
        "true_beta",
        "estimated_beta",
        "group_plot_lo",
        "group_plot_hi",
        "group_plot_center",
        "method_order",
        "method_signal_rmse",
        "method_overall_rmse",
    ]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "is_active_group" in frame.columns:
        frame["is_active_group"] = frame["is_active_group"].fillna(False).astype(bool)
    else:
        frame["is_active_group"] = False
    if "is_active_coefficient" in frame.columns:
        frame["is_active_coefficient"] = frame["is_active_coefficient"].fillna(False).astype(bool)
    else:
        frame["is_active_coefficient"] = False

    if "method_order" in frame.columns and frame["method_order"].notna().any():
        methods = (
            frame.loc[:, ["method", "method_label", "method_order"]]
            .drop_duplicates()
            .sort_values(["method_order", "method"], kind="stable")
        )
    else:
        methods = frame.loc[:, ["method", "method_label"]].drop_duplicates().sort_values(["method"], kind="stable")
    method_records = methods.to_dict(orient="records")
    if not method_records:
        return None

    n_panels = len(method_records)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(14.0, max(3.8, 2.4 * n_panels)),
        sharex=True,
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    truth = (
        frame.loc[:, ["plot_order", "true_beta", "group_id", "group_plot_lo", "group_plot_hi", "group_plot_center", "is_active_group"]]
        .drop_duplicates(subset=["plot_order"])
        .sort_values(["plot_order"], kind="stable")
        .reset_index(drop=True)
    )
    x = truth["plot_order"].to_numpy(dtype=float)
    group_spans = (
        truth.loc[:, ["group_id", "group_plot_lo", "group_plot_hi", "group_plot_center", "is_active_group"]]
        .drop_duplicates(subset=["group_id"])
        .sort_values(["group_id"], kind="stable")
        .reset_index(drop=True)
    )

    y_values = np.concatenate(
        [
            pd.to_numeric(frame["true_beta"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(frame["estimated_beta"], errors="coerce").to_numpy(dtype=float),
        ]
    )
    y_values = y_values[np.isfinite(y_values)]
    if y_values.size == 0:
        y_lim = 1.0
    else:
        y_lim = float(np.max(np.abs(y_values)))
        y_lim = max(0.25, y_lim * 1.12)

    for ax, method_row in zip(axes, method_records):
        method = str(method_row["method"])
        label = str(method_row.get("method_label", method))
        sub = frame.loc[frame["method"].astype(str).eq(method)].copy()
        if sub.empty:
            _draw_missing_panel(ax, label, "No coefficient profile is available for this method.")
            continue
        sub = sub.sort_values(["plot_order"], kind="stable")

        for _, g_row in group_spans.iterrows():
            lo = float(g_row["group_plot_lo"])
            hi = float(g_row["group_plot_hi"])
            if bool(g_row["is_active_group"]):
                ax.axvspan(lo - 0.5, hi + 0.5, color="#d1fae5", alpha=0.45, zorder=0)
            ax.axvline(hi + 0.5, color="#d1d5db", lw=0.8, alpha=0.7, zorder=0)

        ax.axhline(0.0, color="#9ca3af", lw=1.0, zorder=1)
        ax.plot(
            x,
            truth["true_beta"].to_numpy(dtype=float),
            color="#111827",
            lw=2.2,
            alpha=0.95,
            label="True beta",
            zorder=3,
        )
        ax.plot(
            sub["plot_order"].to_numpy(dtype=float),
            sub["estimated_beta"].to_numpy(dtype=float),
            color=_method_color(method),
            lw=1.8,
            alpha=0.95,
            label=label,
            zorder=4,
        )
        ax.scatter(
            sub["plot_order"].to_numpy(dtype=float),
            sub["estimated_beta"].to_numpy(dtype=float),
            c=np.where(sub["is_active_coefficient"].to_numpy(dtype=bool), _method_color(method), "#94a3b8"),
            s=np.where(sub["is_active_coefficient"].to_numpy(dtype=bool), 26, 18),
            alpha=0.9,
            edgecolors="white",
            linewidths=0.25,
            zorder=5,
        )
        if "method_signal_rmse" in sub.columns and sub["method_signal_rmse"].notna().any():
            signal_rmse = float(pd.to_numeric(sub["method_signal_rmse"], errors="coerce").dropna().iloc[0])
            overall_rmse = float(pd.to_numeric(sub["method_overall_rmse"], errors="coerce").dropna().iloc[0]) if "method_overall_rmse" in sub.columns and pd.to_numeric(sub["method_overall_rmse"], errors="coerce").notna().any() else float("nan")
            stat_text = f"signal RMSE={signal_rmse:.3f}"
            if np.isfinite(overall_rmse):
                stat_text += f" | overall RMSE={overall_rmse:.3f}"
            ax.text(
                0.995,
                0.92,
                stat_text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                color="#374151",
                bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#d1d5db", linewidth=0.8, alpha=0.95),
            )
        ax.set_ylabel(label)
        ax.set_ylim(-y_lim, y_lim)
        ax.grid(axis="y", alpha=0.18)

    axes[-1].set_xlim(-0.5, float(np.max(x)) + 0.5 if x.size else 0.5)
    axes[-1].set_xlabel("Coefficients ordered within groups by true magnitude")

    if not group_spans.empty:
        xticks = group_spans["group_plot_center"].to_numpy(dtype=float)
        xlabels = [f"g{int(gid)}" for gid in group_spans["group_id"].tolist()]
        axes[-1].set_xticks(xticks, labels=xlabels)

    active_patch = plt.Rectangle((0, 0), 1, 1, facecolor="#d1fae5", edgecolor="none", alpha=0.45)
    handles = [
        plt.Line2D([0], [0], color="#111827", lw=2.2, label="True beta"),
        plt.Line2D([0], [0], color=_method_color("GR_RHS"), lw=1.8, label="Estimated beta"),
        active_patch,
    ]
    labels = ["True beta", "Estimated beta", "Active group band"]
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01), fontsize=9)

    title_bits = []
    if "setting_label" in frame.columns and frame["setting_label"].astype(str).nunique() == 1:
        title_bits.append(str(frame["setting_label"].iloc[0]))
    if "representative_replicate_id" in frame.columns and pd.to_numeric(frame["representative_replicate_id"], errors="coerce").notna().any():
        rep_id = int(pd.to_numeric(frame["representative_replicate_id"], errors="coerce").dropna().iloc[0])
        title_bits.append(f"replicate {rep_id}")
    suffix = " | ".join(title_bits)
    fig.suptitle(
        "Figure 1. Representative coefficient recovery profile" + (f" ({suffix})" if suffix else ""),
        fontsize=13,
        fontweight="bold",
    )
    out_path = Path(out_dir) / "figure1_coefficient_recovery_profile.png"
    _save(fig, out_path)
    return str(out_path)


def build_benchmark_figures_from_results_dir(results_dir: Path | str) -> dict[str, str]:
    root = resolve_history_results_dir(
        results_dir,
        required_files=("paper_tables/figure_data/figure1_coefficient_recovery_profile.csv",),
    )
    paper_dir = ensure_dir(root / "paper_tables")
    fig_data_dir = paper_dir / "figure_data"
    fig_out_dir = ensure_dir(root / "figures")

    outputs: dict[str, str] = {}
    maybe = plot_figure1_coefficient_recovery_profile(
        _read_csv_or_empty(fig_data_dir / "figure1_coefficient_recovery_profile.csv"),
        fig_out_dir,
    )
    if maybe:
        outputs["figure1_coefficient_recovery_profile"] = maybe
    return outputs
