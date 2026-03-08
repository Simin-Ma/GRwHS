from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise SystemExit(f"{path} is empty.")
    return frame


def plot_sensitivity(summary_csv: Path, out_path: Path, *, rhs_rmse: float, rhs_mlpd: float) -> None:
    frame = _load_frame(summary_csv)

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8))
    legend_handles = []
    legend_labels = []
    specs = [
        ("eta", "eta sweep"),
        ("p0", "p0 sweep"),
        ("c", "c sweep"),
    ]

    for ax, (param, title) in zip(axes, specs):
        sub = frame.sort_values(param).copy()
        if param == "eta":
            sub = sub[sub["p0"] == 20]
            sub = sub[sub["c"].round(12) == 1.0]
        elif param == "p0":
            sub = sub[sub["eta"].round(12) == 0.6]
            sub = sub[sub["c"].round(12) == 1.0]
        elif param == "c":
            sub = sub[sub["eta"].round(12) == 0.6]
            sub = sub[sub["p0"] == 20]

        ax.plot(sub[param], sub["RMSE"], marker="o", linewidth=2.0, color="#153B50", label="GR-RHS RMSE")
        ax.axhline(rhs_rmse, color="#2F6690", linestyle="--", linewidth=1.4, label="RHS RMSE")
        ax.set_title(title, fontsize=13.5, fontweight="bold")
        ax.set_xlabel(param, fontsize=11)
        ax.set_ylabel("RMSE", fontsize=11)
        ax.grid(alpha=0.25, linewidth=0.8)

        ax2 = ax.twinx()
        ax2.plot(sub[param], sub["MLPD"], marker="s", linewidth=1.8, color="#8E5572", label="GR-RHS MLPD")
        ax2.axhline(rhs_mlpd, color="#B279A2", linestyle=":", linewidth=1.4, label="RHS MLPD")
        ax2.set_ylabel("MLPD", fontsize=11)

        if not legend_handles:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            legend_handles = h1 + h2
            legend_labels = l1 + l2

    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=4, frameon=False, fontsize=10)
    fig.suptitle("NHANES GR-RHS sensitivity analysis", fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot NHANES GR-RHS sensitivity results.")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("outputs/reports/nhanes_grrhs_sensitivity_summary.csv"),
        help="CSV from summarize_nhanes_grrhs_sensitivity.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/reports/nhanes_grrhs_sensitivity.png"),
        help="Destination PNG path.",
    )
    parser.add_argument("--rhs-rmse", type=float, default=0.5958629598532049)
    parser.add_argument("--rhs-mlpd", type=float, default=-0.9009611069874411)
    args = parser.parse_args()

    plot_sensitivity(args.summary_csv, args.out, rhs_rmse=args.rhs_rmse, rhs_mlpd=args.rhs_mlpd)
    print(f"[ok] plot written to {args.out}")


if __name__ == "__main__":
    main()
