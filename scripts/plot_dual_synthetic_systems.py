from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_METRICS = ["RMSE", "BetaRMSE", "AUC-PR", "F1"]

MODEL_LABELS: Dict[str, str] = {
    "grrhs_gibbs": "GRRHS",
    "regularized_horseshoe": "RHS",
    "gigg": "GIGG",
    "sparse_group_lasso": "SGL",
    "lasso": "Lasso",
    "ridge": "Ridge",
}

MODEL_COLORS: Dict[str, str] = {
    "grrhs_gibbs": "#153B50",
    "regularized_horseshoe": "#5B4B8A",
    "gigg": "#3A7D44",
    "sparse_group_lasso": "#D17A22",
    "lasso": "#8E5572",
    "ridge": "#7A7A7A",
}


def _load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise SystemExit(f"{path} is empty.")
    return frame


def _prepare_frame(path: Path) -> pd.DataFrame:
    frame = _load_frame(path)
    frame = frame.copy()
    frame["model"] = frame["model"].fillna(frame["variation"])
    frame["model_label"] = frame["model"].map(lambda name: MODEL_LABELS.get(str(name), str(name)))
    frame["model_color"] = frame["model"].map(lambda name: MODEL_COLORS.get(str(name), "#4C78A8"))
    return frame


def _metric_direction(metric: str) -> str:
    higher_better = {"AUC-PR", "F1", "BetaCoverage90", "BetaPearson"}
    return "max" if metric in higher_better else "min"


def plot_dual_systems(
    system_a_csv: Path,
    system_b_csv: Path,
    out_path: Path,
    *,
    system_a_name: str,
    system_b_name: str,
    metrics: Iterable[str],
) -> None:
    frame_a = _prepare_frame(system_a_csv)
    frame_b = _prepare_frame(system_b_csv)
    metrics = [metric for metric in metrics if metric in frame_a.columns and metric in frame_b.columns]
    if not metrics:
        raise SystemExit("No common requested metric columns found in both comparison CSV files.")

    system_frames = [
        (system_a_name, frame_a),
        (system_b_name, frame_b),
    ]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(metrics),
        figsize=(4.2 * len(metrics), 9.0),
        squeeze=False,
    )

    for row_idx, (system_name, frame) in enumerate(system_frames):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx][col_idx]
            plot_frame = frame[["model", "model_label", "model_color", metric]].dropna().sort_values(metric)
            if plot_frame.empty:
                ax.set_visible(False)
                continue

            ascending = _metric_direction(metric) == "min"
            plot_frame = plot_frame.sort_values(metric, ascending=ascending)
            ax.bar(
                plot_frame["model_label"],
                plot_frame[metric],
                color=plot_frame["model_color"],
                edgecolor="white",
                linewidth=0.8,
            )
            best_value = plot_frame[metric].iloc[0]
            ax.axhline(best_value, color="#AA3A38", linestyle="--", linewidth=1.0, alpha=0.55)
            ax.set_title(f"{system_name}\n{metric}", fontsize=13, fontweight="bold")
            ax.tick_params(axis="x", labelrotation=25, labelsize=9)
            ax.tick_params(axis="y", labelsize=9)
            ax.grid(axis="y", alpha=0.25, linewidth=0.8)
            ax.set_axisbelow(True)

    fig.suptitle("Synthetic Benchmark Across Two Data-Generation Systems", fontsize=18, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot six-model comparisons for two synthetic systems.")
    parser.add_argument("--system-a-csv", type=Path, required=True)
    parser.add_argument("--system-b-csv", type=Path, required=True)
    parser.add_argument("--system-a-name", type=str, default="System A")
    parser.add_argument("--system-b-name", type=str, default="System B")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metric columns shared by both comparison CSV files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/reports/dual_synthetic_systems.png"),
    )
    args = parser.parse_args()

    plot_dual_systems(
        args.system_a_csv,
        args.system_b_csv,
        args.out,
        system_a_name=args.system_a_name,
        system_b_name=args.system_b_name,
        metrics=list(args.metrics),
    )
    print(f"[ok] plot written to {args.out}")


if __name__ == "__main__":
    main()
