from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


GROUP_ORDER = [
    "Metals",
    "Phthalates",
    "Organochlorine pesticides",
    "PBDEs",
    "PAHs",
]

MODEL_ORDER = [
    "GR-RHS",
    "RHS",
    "GIGG",
    "Sparse Group Lasso",
    "Lasso",
    "Ridge",
]

MODEL_COLORS = {
    "GR-RHS": "#153B50",
    "RHS": "#2F6690",
    "GIGG": "#3A7D44",
    "Sparse Group Lasso": "#D17A22",
    "Lasso": "#8E5572",
    "Ridge": "#7A7A7A",
}


def plot_group_ci(summary_csv: Path, out_path: Path, *, title: str) -> None:
    try:
        frame = pd.read_csv(summary_csv)
    except EmptyDataError:
        raise SystemExit(
            f"{summary_csv} is empty. Run the NHANES sweep to completion and regenerate the effect summary first."
        ) from None
    if frame.empty:
        raise SystemExit(
            f"{summary_csv} is empty. Run the NHANES sweep to completion and regenerate the effect summary first."
        )

    frame = frame.copy()
    frame["group"] = pd.Categorical(frame["group"], categories=GROUP_ORDER, ordered=True)
    frame["model"] = pd.Categorical(frame["model"], categories=MODEL_ORDER, ordered=True)
    frame.sort_values(["group", "model"], inplace=True)

    groups = [group for group in GROUP_ORDER if group in set(frame["group"].astype(str))]
    models = [model for model in MODEL_ORDER if model in set(frame["model"].astype(str))]
    if not groups or not models:
        raise SystemExit("No recognized groups/models found in the summary CSV.")

    x = np.arange(len(groups), dtype=float)
    width = 0.12 if len(models) >= 5 else 0.16

    fig, ax = plt.subplots(figsize=(max(8.0, 1.9 * len(groups)), 5.8))
    for idx, model in enumerate(models):
        model_frame = frame[frame["model"].astype(str) == model]
        y_vals = []
        for group in groups:
            match = model_frame[model_frame["group"].astype(str) == group]
            y_vals.append(float(match["mean_ci95_length"].iloc[0]) if not match.empty else np.nan)
        offset = (idx - (len(models) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            y_vals,
            width=width,
            label=model,
            color=MODEL_COLORS.get(model, "#4C78A8"),
            edgecolor="white",
            linewidth=0.8,
        )

    ax.set_title(title, fontsize=18, fontweight="bold", pad=12)
    ax.set_ylabel("Mean 95% CI length (% change in GGT)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10, ncol=2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot mean CI length by exposure group for NHANES method comparisons.")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("outputs/reports/nhanes_effects/nhanes_group_ci_summary.csv"),
        help="CSV created by scripts/summarize_nhanes_effects.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/reports/nhanes_effects/nhanes_group_ci_barplot.png"),
        help="Destination PNG path.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Mean 95% CI Length by Exposure Group",
        help="Plot title.",
    )
    args = parser.parse_args()

    plot_group_ci(args.summary_csv, args.out, title=args.title)
    print(f"[ok] plot written to {args.out}")


if __name__ == "__main__":
    main()
