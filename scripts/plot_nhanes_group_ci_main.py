from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GROUP_ORDER = [
    "Metals",
    "Phthalates",
    "Organochlorine pesticides",
    "PBDEs",
    "PAHs",
]

MODEL_ORDER = ["GR-RHS", "RHS", "GIGG", "Sparse Group Lasso", "Lasso", "Ridge"]

MODEL_COLORS: Dict[str, str] = {
    "GR-RHS": "#153B50",
    "RHS": "#2F6690",
    "GIGG": "#7A7A7A",
    "Sparse Group Lasso": "#D17A22",
    "Lasso": "#8E5572",
    "Ridge": "#B3B3B3",
}


def _load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise SystemExit(f"{path} is empty.")
    return frame


def plot_main(summary_csv: Path, out_path: Path, *, title: str) -> None:
    frame = _load_frame(summary_csv).copy()
    frame["group"] = pd.Categorical(frame["group"], categories=GROUP_ORDER, ordered=True)
    frame["model"] = pd.Categorical(frame["model"], categories=MODEL_ORDER, ordered=True)
    frame.sort_values(["group", "model"], inplace=True)

    groups = [g for g in GROUP_ORDER if g in set(frame["group"].astype(str))]
    models = [m for m in MODEL_ORDER if m in set(frame["model"].astype(str))]
    x = np.arange(len(groups), dtype=float)
    width = 0.12

    fig, ax = plt.subplots(figsize=(12.8, 6.8))
    for idx, model in enumerate(models):
        model_frame = frame[frame["model"].astype(str) == model]
        values: List[float] = []
        for group in groups:
            match = model_frame[model_frame["group"].astype(str) == group]
            values.append(float(match["mean_ci95_length"].iloc[0]) if not match.empty else np.nan)

        offset = (idx - (len(models) - 1) / 2.0) * width
        edge = "#111111" if model in {"GR-RHS", "RHS"} else "white"
        lw = 1.5 if model in {"GR-RHS", "RHS"} else 0.8
        alpha = 1.0 if model in {"GR-RHS", "RHS"} else 0.72
        ax.bar(
            x + offset,
            values,
            width=width,
            label=model,
            color=MODEL_COLORS.get(model, "#4C78A8"),
            edgecolor=edge,
            linewidth=lw,
            alpha=alpha,
        )

    ax.set_title(title, fontsize=19, fontweight="bold", pad=12)
    ax.set_ylabel("Mean 95% CI length (% change in GGT)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)

    note = (
        "GR-RHS and RHS are highlighted with darker fills and dark borders.\n"
        "GIGG intervals are near zero in the current summary and should be interpreted with caution."
    )
    fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=10.2, color="#444444")
    ax.legend(frameon=False, fontsize=10, ncol=3, loc="upper right")

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the main NHANES group mean CI-length figure.")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("outputs/reports/nhanes_effects/nhanes_group_ci_summary.csv"),
        help="Group summary CSV from summarize_nhanes_effects.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/reports/nhanes_effects/nhanes_group_ci_main.png"),
        help="Destination PNG path.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="NHANES 2003-2004: Mean CI Length by Exposure Group",
        help="Figure title.",
    )
    args = parser.parse_args()

    plot_main(args.summary_csv, args.out, title=args.title)
    print(f"[ok] plot written to {args.out}")


if __name__ == "__main__":
    main()
