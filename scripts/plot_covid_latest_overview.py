from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SWEEP_DIR = Path("outputs/sweeps/real_covid19_trust_experts_thesis")
TIMESTAMP = "20260309-025417"

MODEL_LABELS = {
    "trust_experts_grrhs": "GR-RHS",
    "trust_experts_rhs": "RHS",
    "trust_experts_gigg": "GIGG",
    "trust_experts_sgl": "SGL",
    "trust_experts_lasso": "Lasso",
    "trust_experts_ridge": "Ridge",
}

MODEL_ORDER = [
    "trust_experts_grrhs",
    "trust_experts_rhs",
    "trust_experts_gigg",
    "trust_experts_sgl",
    "trust_experts_lasso",
    "trust_experts_ridge",
]

MODEL_COLORS = {
    "GR-RHS": "#0f6b50",
    "RHS": "#6b7280",
    "GIGG": "#b42318",
    "SGL": "#1769aa",
    "Lasso": "#d97706",
    "Ridge": "#374151",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _aggregate_from_summary(run_summary: dict) -> dict[str, float]:
    fold_metrics: dict[str, list[float]] = {}
    for repeat in run_summary.get("repeat_summaries", []):
        for fold in repeat.get("folds", []):
            metrics = fold.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            for key, value in metrics.items():
                numeric = _safe_float(value)
                if numeric is None:
                    continue
                fold_metrics.setdefault(str(key), []).append(numeric)
    return {k: float(np.mean(v)) for k, v in fold_metrics.items() if v}


def _build_table(sweep_dir: Path) -> pd.DataFrame:
    rows = []
    for run_name in MODEL_ORDER:
        run_dir = sweep_dir / f"{run_name}-{TIMESTAMP}"
        summary = _load_json(run_dir / "summary.json")
        metrics = summary.get("metrics", {}) or {}
        if not metrics:
            metrics = _aggregate_from_summary(summary)
        rows.append(
            {
                "run_name": run_name,
                "model": MODEL_LABELS[run_name],
                "status": summary.get("status"),
                "RMSE": _safe_float(metrics.get("RMSE")),
                "MLPD": _safe_float(metrics.get("MLPD")),
                "EffectiveDoF": _safe_float(metrics.get("EffectiveDoF")),
                "MeanEffectiveNonzeros": _safe_float(metrics.get("MeanEffectiveNonzeros")),
                "Coverage90": _safe_float(metrics.get("Coverage90")),
                "IntervalWidth90": _safe_float(metrics.get("IntervalWidth90")),
            }
        )
    return pd.DataFrame(rows)


def _plot_overview(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.8), constrained_layout=True)

    x = np.arange(len(df))
    labels = df["model"].tolist()
    colors = [MODEL_COLORS[label] for label in labels]

    panels = [
        ("RMSE", "RMSE", "Lower is better"),
        ("EffectiveDoF", "Effective DoF", "Model complexity"),
        ("MeanEffectiveNonzeros", "Mean effective nonzeros", "Shrinkage sparsity"),
    ]

    for ax, (col, title, subtitle) in zip(axes, panels):
        vals = df[col].to_numpy(dtype=float)
        bars = ax.bar(x, vals, color=colors, alpha=0.88)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        ax.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("COVID latest sweep: six-model predictive and complexity overview", fontsize=15, fontweight="bold")
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_bayes_uncertainty(df: pd.DataFrame, out_path: Path) -> None:
    sub = df[df["model"].isin(["GR-RHS", "RHS", "GIGG"])].copy()
    fig, ax = plt.subplots(figsize=(7.6, 6.1), constrained_layout=True)
    ax.axvspan(0.85, 0.95, color="#e8f5e9", alpha=0.75, zorder=0)
    ax.axvline(0.90, color="#2e7d32", linestyle="--", linewidth=1.3)
    for _, row in sub.iterrows():
        ax.scatter(
            row["Coverage90"],
            row["IntervalWidth90"],
            s=180,
            color=MODEL_COLORS[row["model"]],
            alpha=0.9,
            edgecolor="black",
            linewidth=0.8,
        )
        ax.text(row["Coverage90"] + 0.005, row["IntervalWidth90"], row["model"], fontsize=10, va="center")
    ax.set_xlabel("PredictiveCoverage90")
    ax.set_ylabel("PredictiveIntervalWidth90")
    ax.set_title("COVID latest sweep: predictive coverage vs interval width", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sweep_dir = repo_root / SWEEP_DIR
    out_dir = repo_root / "outputs" / "reports" / "covid_latest"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _build_table(sweep_dir)
    df.to_csv(out_dir / "covid_latest_overview_table.csv", index=False)
    _plot_overview(df, out_dir / "covid_latest_six_model_overview.png")
    _plot_bayes_uncertainty(df, out_dir / "covid_latest_bayes_uncertainty.png")


if __name__ == "__main__":
    main()
