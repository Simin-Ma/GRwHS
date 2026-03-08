from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_MODELS = ["GR-RHS", "RHS"]
GROUP_ORDER = [
    "Metals",
    "Phthalates",
    "Organochlorine pesticides",
    "PBDEs",
    "PAHs",
]
MODEL_COLORS: Dict[str, str] = {
    "GR-RHS": "#153B50",
    "RHS": "#2F6690",
}


def _load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise SystemExit(f"{path} is empty.")
    return frame


def _prepare_plot_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[frame["model"].isin(TARGET_MODELS)].copy()
    if frame.empty:
        raise SystemExit("No GR-RHS / RHS rows found in the exposure summary.")

    summary = (
        frame.groupby(["feature", "label", "group"], as_index=False)
        .agg(sort_key=("median_percent_change", lambda s: float(np.mean(np.abs(s)))))
    )
    summary["group"] = pd.Categorical(summary["group"], categories=GROUP_ORDER, ordered=True)
    summary.sort_values(["group", "sort_key", "label"], ascending=[True, False, True], inplace=True)
    summary["order"] = np.arange(summary.shape[0], dtype=int)
    return frame.merge(summary[["feature", "order"]], on="feature", how="left")


def plot_forest(exposure_csv: Path, out_path: Path, *, title: str) -> None:
    frame = _prepare_plot_frame(_load_frame(exposure_csv))
    labels = (
        frame[["feature", "label", "group", "order"]]
        .drop_duplicates()
        .sort_values("order")
        .reset_index(drop=True)
    )

    n = labels.shape[0]
    fig_h = max(10.0, 0.34 * n + 2.6)
    fig, ax = plt.subplots(figsize=(11.8, fig_h))

    offsets = {"GR-RHS": -0.18, "RHS": 0.18}
    for model in TARGET_MODELS:
        model_frame = frame[frame["model"] == model].copy()
        model_frame = labels.merge(
            model_frame[["feature", "median_percent_change", "ci95_low", "ci95_high"]],
            on="feature",
            how="left",
        )
        y = model_frame["order"].to_numpy(dtype=float) + offsets[model]
        center = model_frame["median_percent_change"].to_numpy(dtype=float)
        low = model_frame["ci95_low"].to_numpy(dtype=float)
        high = model_frame["ci95_high"].to_numpy(dtype=float)
        xerr = np.vstack([center - low, high - center])

        ax.errorbar(
            center,
            y,
            xerr=xerr,
            fmt="o",
            color=MODEL_COLORS[model],
            ecolor=MODEL_COLORS[model],
            elinewidth=1.8,
            capsize=2.8,
            markersize=5.2,
            label=model,
            alpha=0.95,
        )

    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1.2, alpha=0.8)

    ax.set_yticks(labels["order"].to_numpy(dtype=float))
    ax.set_yticklabels(labels["label"].tolist(), fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlabel("Percent change in GGT for 2x exposure", fontsize=13)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(axis="x", alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)

    group_blocks: List[tuple[str, float, float]] = []
    for group in GROUP_ORDER:
        sub = labels[labels["group"] == group]
        if sub.empty:
            continue
        y0 = float(sub["order"].min()) - 0.6
        y1 = float(sub["order"].max()) + 0.6
        group_blocks.append((group, y0, y1))

    for idx, (group, y0, y1) in enumerate(group_blocks):
        if idx % 2 == 0:
            ax.axhspan(y0, y1, color="#F5F2EB", alpha=0.7, zorder=0)
        if idx > 0:
            ax.axhline(y0, color="#B0B0B0", linewidth=0.8, alpha=0.9)
        ax.text(
            0.995,
            (y0 + y1) / 2.0,
            group,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color="#444444",
        )

    ax.legend(loc="lower right", frameon=False, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a GR-RHS vs RHS NHANES forest plot.")
    parser.add_argument(
        "--exposure-csv",
        type=Path,
        default=Path("outputs/reports/nhanes_effects/nhanes_exposure_effects.csv"),
        help="Exposure summary CSV from summarize_nhanes_effects.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/reports/nhanes_effects/nhanes_grrhs_rhs_forest.png"),
        help="Destination PNG path.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="NHANES 2003-2004: GR-RHS vs RHS Effect Sizes",
        help="Figure title.",
    )
    args = parser.parse_args()

    plot_forest(args.exposure_csv, args.out, title=args.title)
    print(f"[ok] plot written to {args.out}")


if __name__ == "__main__":
    main()
