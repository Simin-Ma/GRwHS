from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_SPECS: List[Tuple[str, str, bool]] = [
    ("RMSE", "RMSE", True),
    ("MLPD", "MLPD", False),
    ("PredictiveLogLikelihood", "Predictive log-likelihood", False),
    ("EffectiveDoF", "Effective DoF", False),
    ("MeanEffectiveNonzeros", "Mean effective nonzeros", False),
]

MODEL_LABELS: Dict[str, str] = {
    "nhanes_grrhs": "GR-RHS",
    "nhanes_rhs": "RHS",
    "nhanes_gigg": "GIGG",
    "nhanes_sgl": "SGL",
    "nhanes_lasso": "Lasso",
    "nhanes_ridge": "Ridge",
}

MODEL_COLORS: Dict[str, str] = {
    "GR-RHS": "#153B50",
    "RHS": "#2F6690",
    "GIGG": "#3A7D44",
    "SGL": "#D17A22",
    "Lasso": "#8E5572",
    "Ridge": "#7A7A7A",
}


def _load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise SystemExit(f"{path} is empty.")
    return frame


def _resolve_model_label(row: pd.Series) -> str:
    variation = str(row.get("variation", ""))
    model = str(row.get("model", ""))
    if variation in MODEL_LABELS:
        return MODEL_LABELS[variation]
    fallback = {
        "grrhs_gibbs": "GR-RHS",
        "gigg": "GIGG",
        "sparse_group_lasso": "SGL",
        "lasso": "Lasso",
        "ridge": "Ridge",
    }
    return fallback.get(model, variation or model)


def plot_overview(comparison_csv: Path, out_path: Path, *, title: str) -> None:
    frame = _load_frame(comparison_csv).copy()
    frame["label"] = frame.apply(_resolve_model_label, axis=1)
    status_col = "Status" if "Status" in frame.columns else "status"
    if status_col in frame.columns:
        frame = frame[frame[status_col].fillna("OK") != "ERROR"].copy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8))
    axes = axes.ravel()

    for ax, (metric, ylabel, lower_is_better) in zip(axes, METRIC_SPECS):
        if metric not in frame.columns:
            ax.set_visible(False)
            continue
        values = pd.to_numeric(frame[metric], errors="coerce")
        plot_frame = frame.loc[values.notna(), ["label", metric]].copy()
        plot_frame[metric] = pd.to_numeric(plot_frame[metric], errors="coerce")
        plot_frame.sort_values(metric, ascending=lower_is_better, inplace=True)

        labels = plot_frame["label"].tolist()
        vals = plot_frame[metric].to_numpy(dtype=float)
        colors = [MODEL_COLORS.get(label, "#4C78A8") for label in labels]
        bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.8)

        best_idx = int(np.argmin(vals) if lower_is_better else np.argmax(vals))
        bars[best_idx].set_edgecolor("#111111")
        bars[best_idx].set_linewidth(2.2)

        for idx, value in enumerate(vals):
            ax.text(
                idx,
                value,
                f"{value:.3f}" if abs(value) >= 1e-2 else f"{value:.2e}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=9,
                rotation=0,
            )

        direction = "lower is better" if lower_is_better else "higher is better"
        ax.set_title(f"{ylabel}\n({direction})", fontsize=14, fontweight="bold", pad=10)
        ax.tick_params(axis="x", labelrotation=20, labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)

    legend_ax = axes[-1]
    legend_ax.axis("off")
    legend_ax.text(
        0.02,
        0.95,
        "NHANES sweep overview",
        fontsize=17,
        fontweight="bold",
        va="top",
    )
    legend_ax.text(
        0.02,
        0.74,
        "Bar border marks the best model within each metric.",
        fontsize=11,
        va="top",
    )
    legend_ax.text(
        0.02,
        0.58,
        "Interpretation",
        fontsize=13,
        fontweight="bold",
        va="top",
    )
    legend_ax.text(
        0.02,
        0.48,
        "- Predictive metrics: RMSE, MLPD, Predictive log-likelihood",
        fontsize=10.5,
        va="top",
    )
    legend_ax.text(
        0.02,
        0.39,
        "- Complexity metrics: Effective DoF, Mean effective nonzeros",
        fontsize=10.5,
        va="top",
    )
    legend_ax.text(
        0.02,
        0.30,
        "- NHANES paper-facing uncertainty results should still come from",
        fontsize=10.5,
        va="top",
    )
    legend_ax.text(
        0.02,
        0.22,
        "  the 2x exposure -> % change in GGT summaries and CI-length plots.",
        fontsize=10.5,
        va="top",
    )

    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the NHANES sweep comparison metrics.")
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=Path("outputs/sweeps/real_nhanes_2003_2004_ggt/sweep_comparison_20260308-014419.csv"),
        help="Sweep comparison CSV emitted by run_sweep.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/reports/nhanes_sweep_overview.png"),
        help="Destination PNG path.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="NHANES 2003-2004 GGT Sweep Overview",
        help="Figure title.",
    )
    args = parser.parse_args()

    plot_overview(args.comparison_csv, args.out, title=args.title)
    print(f"[ok] plot written to {args.out}")


if __name__ == "__main__":
    main()
