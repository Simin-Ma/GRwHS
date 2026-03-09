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
    alias_by_model = {
        "grrhs_gibbs": "GR-RHS",
        "regularized_horseshoe": "RHS",
        "rhs": "RHS",
        "gigg": "GIGG",
        "sparse_group_lasso": "SGL",
        "lasso": "Lasso",
        "ridge": "Ridge",
    }
    alias_by_variation = {
        "trust_experts_grrhs": "GR-RHS",
        "trust_experts_rhs": "RHS",
        "trust_experts_gigg": "GIGG",
        "trust_experts_sgl": "SGL",
        "trust_experts_lasso": "Lasso",
        "trust_experts_ridge": "Ridge",
        "nhanes_grrhs": "GR-RHS",
        "nhanes_rhs": "RHS",
        "nhanes_gigg": "GIGG",
        "nhanes_sgl": "SGL",
        "nhanes_lasso": "Lasso",
        "nhanes_ridge": "Ridge",
    }
    if variation in alias_by_variation:
        return alias_by_variation[variation]
    return alias_by_model.get(model, variation or model)


def plot_overview(comparison_csv: Path, out_path: Path, *, title: str) -> None:
    frame = _load_frame(comparison_csv).copy()
    frame["label"] = frame.apply(_resolve_model_label, axis=1)
    status_col = "Status" if "Status" in frame.columns else "status"
    if status_col in frame.columns:
        frame = frame[frame[status_col].fillna("OK") != "ERROR"].copy()

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8))
    axes = axes.ravel()

    for ax, (metric, ylabel, lower_is_better) in zip(axes, METRIC_SPECS):
        if metric not in frame.columns:
            ax.set_visible(False)
            continue
        values = pd.to_numeric(frame[metric], errors="coerce")
        plot_frame = frame.loc[values.notna(), ["label", metric]].copy()
        if plot_frame.empty:
            ax.set_visible(False)
            continue
        plot_frame[metric] = pd.to_numeric(plot_frame[metric], errors="coerce")
        plot_frame.sort_values(metric, ascending=lower_is_better, inplace=True)

        labels = plot_frame["label"].tolist()
        vals = plot_frame[metric].to_numpy(dtype=float)
        colors = [MODEL_COLORS.get(label, "#4C78A8") for label in labels]
        bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.8)

        best_idx = int(np.argmin(vals) if lower_is_better else np.argmax(vals))
        bars[best_idx].set_edgecolor("#111111")
        bars[best_idx].set_linewidth(2.0)

        for idx, value in enumerate(vals):
            label = f"{value:.3f}" if abs(value) >= 1e-2 else f"{value:.2e}"
            ax.text(idx, value, label, ha="center", va="bottom" if value >= 0 else "top", fontsize=8.5)

        ax.set_title(f"{ylabel}\n({'lower' if lower_is_better else 'higher'} is better)", fontsize=13, fontweight="bold", pad=10)
        ax.tick_params(axis="x", labelrotation=20, labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)

    note_ax = axes[-1]
    note_ax.axis("off")
    note_ax.text(0.02, 0.95, title, fontsize=16, fontweight="bold", va="top")
    note_ax.text(0.02, 0.74, "Black border marks the best model for each metric.", fontsize=11, va="top")
    note_ax.text(0.02, 0.56, "Recommended thesis use", fontsize=12.5, fontweight="bold", va="top")
    note_ax.text(0.02, 0.46, "- Main predictive ranking: RMSE", fontsize=10.5, va="top")
    note_ax.text(0.02, 0.37, "- Bayesian calibration: MLPD / predictive log-likelihood", fontsize=10.5, va="top")
    note_ax.text(0.02, 0.28, "- Structural sparsity: Effective DoF / Mean effective nonzeros", fontsize=10.5, va="top")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a paper-style overview figure for a real-data sweep.")
    parser.add_argument("--comparison-csv", required=True, type=Path, help="Sweep comparison CSV emitted by run_sweep.")
    parser.add_argument("--out", required=True, type=Path, help="Destination PNG path.")
    parser.add_argument("--title", type=str, default="Real-data sweep overview", help="Figure title.")
    args = parser.parse_args()

    plot_overview(args.comparison_csv, args.out, title=args.title)
    print(f"[ok] plot written to {args.out}")


if __name__ == "__main__":
    main()
