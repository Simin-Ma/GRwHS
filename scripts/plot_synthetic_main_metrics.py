from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_METRICS = [
    "RMSE",
    "BetaRMSE",
    "AUC-PR",
    "BetaCoverage90",
]

MODEL_COLORS: Dict[str, str] = {
    "grrhs_nuts": "#153B50",
    "regularized_horseshoe": "#5B4B8A",
    "gigg": "#3A7D44",
    "ridge": "#7A7A7A",
    "lasso": "#8E5572",
    "sparse_group_lasso": "#D17A22",
}


def _load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise SystemExit(f"{path} is empty.")
    return frame


def plot_main_metrics(
    comparison_csv: Path,
    out_path: Path,
    *,
    title: str,
    metrics: List[str],
) -> None:
    frame = _load_frame(comparison_csv)
    metrics = [metric for metric in metrics if metric in frame.columns]
    if not metrics:
        raise SystemExit(f"No requested metric columns were found in {comparison_csv}.")

    display = frame[["variation", "model", *metrics]].copy()
    display["model_label"] = display["model"].fillna(display["variation"])

    n_panels = max(1, len(metrics))
    ncols = 2
    nrows = int((n_panels + ncols - 1) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, max(4.8, 4.0 * nrows)))
    axes = axes.ravel()
    for ax, metric in zip(axes, metrics):
        plot_frame = display[["model_label", metric]].dropna()
        if plot_frame.empty:
            ax.set_visible(False)
            continue
        colors = [MODEL_COLORS.get(str(label), "#4C78A8") for label in plot_frame["model_label"]]
        ax.bar(plot_frame["model_label"], plot_frame[metric], color=colors, edgecolor="white", linewidth=0.8)
        ax.set_title(metric, fontsize=15, fontweight="bold")
        ax.tick_params(axis="x", labelrotation=25, labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)
        if metric == "BetaCoverage90":
            ax.axhline(0.9, color="#AA3A38", linestyle="--", linewidth=1.4)
            ax.set_ylim(0.0, max(1.0, float(plot_frame[metric].max()) * 1.05))

    for ax in axes[len(metrics):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the main synthetic benchmark metrics from a sweep comparison CSV.")
    parser.add_argument("--comparison-csv", type=Path, required=True, help="CSV emitted by grrhs.cli.run_sweep.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/reports/synthetic_main_metrics.png"),
        help="Destination PNG path.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Synthetic Benchmark Main Metrics",
        help="Figure title.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metric columns to display.",
    )
    args = parser.parse_args()

    plot_main_metrics(args.comparison_csv, args.out, title=args.title, metrics=list(args.metrics))
    print(f"[ok] plot written to {args.out}")


if __name__ == "__main__":
    main()

