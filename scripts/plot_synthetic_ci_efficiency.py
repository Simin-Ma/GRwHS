from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SWEEP_FILES = {
    "sim_s1": "20260309-115403",
    "sim_s2": "20260309-120416",
    "sim_s3": "20260309-121319",
    "sim_s4": "20260309-122319",
}

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

ALL_MODELS = ["grrhs", "rhs", "gigg", "sgl", "lasso", "ridge"]
BAYES_MODELS = ["grrhs", "rhs", "gigg"]
SNR_ORDER = [0.1, 0.5, 1.0, 3.0]


def _load_data(repo_root: Path) -> pd.DataFrame:
    rows = []
    for sim, timestamp in SWEEP_FILES.items():
        path = repo_root / "outputs" / "sweeps" / sim / f"sweep_comparison_{timestamp}.csv"
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
                    "BetaCoverage90": pd.to_numeric(row.get("BetaCoverage90"), errors="coerce"),
                    "ActiveBetaIntervalWidth90": pd.to_numeric(row.get("ActiveBetaIntervalWidth90"), errors="coerce"),
                }
            )
    return pd.DataFrame(rows)


def _write_tables(df: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    for model in ALL_MODELS:
        sub = df[df["model"] == model]
        summary_rows.append(
            {
                "model": MODEL_LABELS[model],
                "has_ci": bool(sub["ActiveBetaIntervalWidth90"].notna().any()),
                "median_ci_width": float(sub["ActiveBetaIntervalWidth90"].median()) if sub["ActiveBetaIntervalWidth90"].notna().any() else np.nan,
                "mean_ci_width": float(sub["ActiveBetaIntervalWidth90"].mean()) if sub["ActiveBetaIntervalWidth90"].notna().any() else np.nan,
                "median_coverage": float(sub["BetaCoverage90"].median()) if sub["BetaCoverage90"].notna().any() else np.nan,
                "coverage_ge_085_count": int((sub["BetaCoverage90"] >= 0.85).sum()) if sub["BetaCoverage90"].notna().any() else 0,
                "total_scenarios": int(len(sub)),
            }
        )
    capability = pd.DataFrame(summary_rows)
    capability.to_csv(out_dir / "synthetic_ci_capability_table.csv", index=False)

    bayes = df[df["model"].isin(BAYES_MODELS)].copy()
    margins = []
    pivot = bayes.pivot_table(index=["sim", "snr"], columns="model", values=["BetaCoverage90", "ActiveBetaIntervalWidth90"])
    matched_rhs = (pivot["BetaCoverage90"]["grrhs"] >= 0.85) & (pivot["BetaCoverage90"]["rhs"] >= 0.85)
    if matched_rhs.any():
        width_ratio = pivot["ActiveBetaIntervalWidth90"]["grrhs"] / pivot["ActiveBetaIntervalWidth90"]["rhs"]
        margins.append(
            {
                "comparison": "GR-RHS vs RHS (matched coverage >= 0.85)",
                "mean_width_ratio": float(width_ratio[matched_rhs].mean()),
                "median_width_ratio": float(width_ratio[matched_rhs].median()),
                "n": int(matched_rhs.sum()),
            }
        )
    margins_df = pd.DataFrame(margins)
    margins_df.to_csv(out_dir / "synthetic_ci_efficiency_notes.csv", index=False)
    return capability, margins_df


def _plot_figure(df: pd.DataFrame, capability: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8), constrained_layout=True)

    ax = axes[0]
    ax.axvspan(0.85, 0.95, color="#e8f5e9", alpha=0.8, zorder=0)
    ax.axvline(0.90, color="#2e7d32", linestyle="--", linewidth=1.5, alpha=0.9)
    for model in BAYES_MODELS:
        sub = df[df["model"] == model]
        ax.scatter(
            sub["BetaCoverage90"],
            sub["ActiveBetaIntervalWidth90"],
            color=MODEL_COLORS[model],
            alpha=0.35,
            s=55,
            label=MODEL_LABELS[model],
        )
        ax.scatter(
            [sub["BetaCoverage90"].median()],
            [sub["ActiveBetaIntervalWidth90"].median()],
            color=MODEL_COLORS[model],
            edgecolor="black",
            linewidth=0.8,
            s=170,
            zorder=4,
        )
        ax.text(
            float(sub["BetaCoverage90"].median()) + 0.01,
            float(sub["ActiveBetaIntervalWidth90"].median()),
            MODEL_LABELS[model],
            fontsize=10,
            va="center",
        )
    ax.set_title("Coverage-Length Efficiency (Bayesian models only)")
    ax.set_xlabel("BetaCoverage90")
    ax.set_ylabel("Median active CI length (ActiveBetaIntervalWidth90)")
    ax.grid(alpha=0.25)

    ax2 = axes[1]
    models = capability["model"].tolist()
    x = np.arange(len(models))
    for idx, row in capability.iterrows():
        if row["has_ci"]:
            ax2.bar(
                idx,
                row["median_ci_width"],
                color=MODEL_COLORS[[k for k, v in MODEL_LABELS.items() if v == row["model"]][0]],
                alpha=0.85,
                width=0.72,
            )
            ax2.text(
                idx,
                row["median_ci_width"],
                f"cov={row['median_coverage']:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        else:
            ax2.bar(idx, 0.02, color="#d1d5db", width=0.72, hatch="///", edgecolor="#6b7280")
            ax2.text(idx, 0.07, "No CI", ha="center", va="bottom", fontsize=9, color="#4b5563")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=20)
    ax2.set_ylabel("Median active CI length")
    ax2.set_title("Six-model view: CI availability and typical length")
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Synthetic uncertainty comparison: do not read CI length without coverage",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _write_md(capability: pd.DataFrame, notes: pd.DataFrame, out_path: Path) -> None:
    grrhs = capability[capability["model"] == "GR-RHS"].iloc[0]
    rhs = capability[capability["model"] == "RHS"].iloc[0]
    gigg = capability[capability["model"] == "GIGG"].iloc[0]
    lines = [
        "# Synthetic CI Efficiency Notes",
        "",
        "Recommended interpretation:",
        "- CI length alone is not a fair quality measure.",
        "- A short interval with very poor coverage is overconfident, not better.",
        "- For this synthetic benchmark, GR-RHS should be presented as a calibrated interval method, not as the shortest interval method.",
        "",
        "Headline numbers:",
        f"- GR-RHS median coverage = {grrhs['median_coverage']:.3f}, median active CI length = {grrhs['median_ci_width']:.3f}.",
        f"- RHS median coverage = {rhs['median_coverage']:.3f}, median active CI length = {rhs['median_ci_width']:.3f}.",
        f"- GIGG median coverage = {gigg['median_coverage']:.3f}, median active CI length = {gigg['median_ci_width']:.3f}.",
        f"- Deterministic baselines (SGL / Lasso / Ridge) do not output coefficient CIs in the current synthetic pipeline.",
    ]
    if not notes.empty:
        row = notes.iloc[0]
        lines.append(
            f"- In scenarios where both GR-RHS and RHS reach coverage >= 0.85, "
            f"GR-RHS median width ratio is {row['mean_width_ratio']:.3f}x of RHS on average "
            f"(n={int(row['n'])})."
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "outputs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_data(repo_root)
    capability, notes = _write_tables(df, out_dir)
    _plot_figure(df, capability, out_dir / "synthetic_ci_efficiency_summary.png")
    _write_md(capability, notes, out_dir / "synthetic_ci_efficiency_notes.md")


if __name__ == "__main__":
    main()
