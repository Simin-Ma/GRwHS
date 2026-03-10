from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd


SWEEPS = ["sim_s1", "sim_s2", "sim_s3"]

SIM_TITLES = {
    "sim_s1": "S1  Sparse-Strong",
    "sim_s2": "S2  Dense-Weak",
    "sim_s3": "S3  Mixed",
}

MODEL_ORDER = ["grrhs", "rhs", "gigg", "sgl", "lasso", "ridge"]
MODEL_LABELS = {
    "grrhs": "GR-RHS",
    "rhs": "RHS",
    "gigg": "GIGG",
    "sgl": "SGL",
    "lasso": "Lasso",
    "ridge": "Ridge",
}
MODEL_COLORS = {
    "grrhs": "#0f6b50",
    "rhs": "#6b7280",
    "gigg": "#b42318",
    "sgl": "#1769aa",
    "lasso": "#d97706",
    "ridge": "#374151",
}


def _latest_sweep_csv(sweep_dir: Path) -> Path:
    files = sorted(sweep_dir.glob("sweep_comparison_*.csv"))
    if not files:
        raise FileNotFoundError(f"No sweep_comparison_*.csv found in {sweep_dir}")
    return files[-1]


def _load_metric_frame(repo_root: Path) -> pd.DataFrame:
    rows = []
    for sim in SWEEPS:
        path = _latest_sweep_csv(repo_root / "outputs" / "sweeps" / sim)
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            match = re.match(r"snr([0-9p]+)_(.+)", str(row["variation"]))
            if match is None:
                continue
            rows.append(
                {
                    "sim": sim,
                    "snr": float(match.group(1).replace("p", ".")),
                    "model": match.group(2),
                    "RMSE": pd.to_numeric(row.get("RMSE"), errors="coerce"),
                    "BetaRMSE": pd.to_numeric(row.get("BetaRMSE"), errors="coerce"),
                }
            )
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out["model"], categories=MODEL_ORDER, ordered=True)
    return out.sort_values(["sim", "snr", "model"]).reset_index(drop=True)


def _plot_metric(ax: plt.Axes, df: pd.DataFrame, metric: str, ylabel: str) -> None:
    plt.style.use("default")
    for model in MODEL_ORDER:
        curve = df[df["model"] == model].sort_values("snr")
        if curve.empty:
            continue
        is_focus = model == "grrhs"
        ax.plot(
            curve["snr"],
            curve[metric],
            color=MODEL_COLORS[model],
            lw=3.0 if is_focus else 1.6,
            marker="o" if is_focus else "s",
            ms=6 if is_focus else 4.5,
            alpha=1.0 if is_focus else 0.78,
            label=MODEL_LABELS[model],
            zorder=4 if is_focus else 2,
        )

    ax.set_xticks([0.1, 0.5, 1.0, 3.0])
    ax.set_xticklabels(["0.1", "0.5", "1.0", "3.0"])
    ax.set_xlabel("SNR")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22, linewidth=0.8)
    ax.set_facecolor("#fbfbfa")
    for spine in ax.spines.values():
        spine.set_color("#d1d5db")
        spine.set_linewidth(1.0)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "outputs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_metric_frame(repo_root)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    for sim in SWEEPS:
        sub = df[df["sim"] == sim]
        if sub.empty:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.6), constrained_layout=False)
        _plot_metric(axes[0], sub, metric="RMSE", ylabel="RMSE")
        axes[0].set_title(f"{SIM_TITLES[sim]}  |  RMSE", loc="left", fontsize=12, fontweight="bold")
        _plot_metric(axes[1], sub, metric="BetaRMSE", ylabel="BetaRMSE")
        axes[1].set_title(f"{SIM_TITLES[sim]}  |  BetaRMSE", loc="left", fontsize=12, fontweight="bold")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=6,
            frameon=False,
            bbox_to_anchor=(0.5, 0.99),
            fontsize=9.5,
        )
        plt.subplots_adjust(top=0.90, hspace=0.35, left=0.10, right=0.97, bottom=0.08)
        fig.savefig(out_dir / f"{sim}_metrics_3x4_{timestamp}.png", dpi=240, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
