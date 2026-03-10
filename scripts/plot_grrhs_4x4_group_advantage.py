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

SIM_LABELS = {
    "sim_s1": "Sim S1",
    "sim_s2": "Sim S2",
    "sim_s3": "Sim S3",
    "sim_s4": "Sim S4",
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
    "grrhs": "#0b6e4f",
    "rhs": "#6c757d",
    "gigg": "#b02a37",
    "sgl": "#1f77b4",
    "lasso": "#f59f00",
    "ridge": "#495057",
}

MODEL_ORDER = ["grrhs", "rhs", "gigg", "sgl", "lasso", "ridge"]


def _load_all_sweeps(root: Path) -> pd.DataFrame:
    rows = []
    for sim, timestamp in SWEEP_FILES.items():
        path = root / "outputs" / "sweeps" / sim / f"sweep_comparison_{timestamp}.csv"
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            match = re.match(r"snr([0-9p]+)_(.+)", str(row["variation"]))
            if match is None:
                continue
            snr = float(match.group(1).replace("p", "."))
            model = match.group(2)
            payload = {
                "sim": sim,
                "sim_label": SIM_LABELS.get(sim, sim),
                "snr": snr,
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "status": row["status"],
            }
            for metric in [
                "RMSE",
                "BetaRMSE",
                "GroupNormRMSE",
                "AUC-PR",
                "F1",
                "BetaCoverage90",
            ]:
                payload[metric] = pd.to_numeric(row.get(metric), errors="coerce")
            rows.append(payload)
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out["model"], categories=MODEL_ORDER, ordered=True)
    out = out.sort_values(["sim", "snr", "model"]).reset_index(drop=True)
    return out


def _winner_counts(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    specs = [
        ("RMSE", False),
        ("BetaRMSE", False),
        ("GroupNormRMSE", False),
        ("AUC-PR", True),
        ("F1", True),
        ("BetaCoverage90", True),
    ]
    for metric, higher_better in specs:
        grouped = df.groupby(["sim", "snr"], observed=True)
        idx = grouped[metric].idxmax() if higher_better else grouped[metric].idxmin()
        winners = df.loc[idx, ["sim", "snr", "model", metric]].copy()
        winners["is_grrhs_win"] = winners["model"].eq("grrhs")
        records.append(
            {
                "metric": metric,
                "grrhs_win_count": int(winners["is_grrhs_win"].sum()),
                "total_scenarios": int(len(winners)),
            }
        )
    return pd.DataFrame(records)


def _baseline_margins(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index=["sim", "snr"],
        columns="model",
        values=["RMSE", "BetaRMSE", "GroupNormRMSE", "AUC-PR", "F1", "BetaCoverage90"],
    )
    records = []
    for baseline in ["rhs", "gigg", "sgl"]:
        for metric in ["RMSE", "BetaRMSE", "GroupNormRMSE"]:
            if ("grrhs" not in pivot[metric]) or (baseline not in pivot[metric]):
                continue
            diff = pivot[metric][baseline] - pivot[metric]["grrhs"]
            records.append(
                {
                    "baseline": baseline,
                    "metric": metric,
                    "better_direction": "lower",
                    "mean_margin_for_grrhs": float(diff.mean()),
                    "grrhs_better_count": int((diff > 0).sum()),
                    "n": int(diff.notna().sum()),
                }
            )
        for metric in ["AUC-PR", "F1", "BetaCoverage90"]:
            if ("grrhs" not in pivot[metric]) or (baseline not in pivot[metric]):
                continue
            diff = pivot[metric]["grrhs"] - pivot[metric][baseline]
            records.append(
                {
                    "baseline": baseline,
                    "metric": metric,
                    "better_direction": "higher",
                    "mean_margin_for_grrhs": float(diff.mean()),
                    "grrhs_better_count": int((diff > 0).sum()),
                    "n": int(diff.notna().sum()),
                }
            )
    return pd.DataFrame(records)


def _plot_metric_grid(df: pd.DataFrame, metric: str, ylabel: str, out_path: Path) -> None:
    sims = list(SWEEP_FILES.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.flatten()
    for ax, sim in zip(axes, sims):
        sub = df[df["sim"] == sim].copy()
        for model in MODEL_ORDER:
            curve = sub[sub["model"] == model].sort_values("snr")
            if curve.empty:
                continue
            alpha = 1.0 if model == "grrhs" else 0.65
            linewidth = 3.0 if model == "grrhs" else 1.8
            zorder = 5 if model == "grrhs" else 2
            ax.plot(
                curve["snr"],
                curve[metric],
                marker="o",
                linewidth=linewidth,
                alpha=alpha,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
                zorder=zorder,
            )
        ax.set_title(SIM_LABELS[sim])
        ax.set_xticks([0.1, 0.5, 1.0, 3.0])
        ax.set_xticklabels(["0.1", "0.5", "1.0", "3.0"])
        ax.set_xlabel("SNR")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(f"4x4 Synthetic Benchmark: {metric}", fontsize=14, y=1.08)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_margin_heatmaps(df: pd.DataFrame, out_path: Path) -> None:
    pivot = df.pivot_table(
        index=["sim", "snr"],
        columns="model",
        values=["GroupNormRMSE", "AUC-PR"],
    )
    sim_snr_index = pd.MultiIndex.from_product(
        [list(SWEEP_FILES.keys()), [0.1, 0.5, 1.0, 3.0]],
        names=["sim", "snr"],
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), constrained_layout=True)
    heat_specs = [
        ("GroupNormRMSE", "rhs", "gigg", "sgl", "GR-RHS margin on GroupNormRMSE\n(positive = better)"),
        ("AUC-PR", "rhs", "gigg", "sgl", "GR-RHS margin on AUC-PR\n(positive = better)"),
    ]
    for ax, (metric, b1, b2, b3, title) in zip(axes, heat_specs):
        cols = [b1, b2, b3]
        mat = []
        for sim in SWEEP_FILES:
            row_block = []
            for snr in [0.1, 0.5, 1.0, 3.0]:
                values = []
                for baseline in cols:
                    if metric == "GroupNormRMSE":
                        val = pivot.loc[(sim, snr), (metric, baseline)] - pivot.loc[(sim, snr), (metric, "grrhs")]
                    else:
                        val = pivot.loc[(sim, snr), (metric, "grrhs")] - pivot.loc[(sim, snr), (metric, baseline)]
                    values.append(val)
                row_block.append(values)
            mat.append(row_block)
        arr = np.asarray(mat, dtype=float).reshape(16, 3)
        vmax = np.nanmax(np.abs(arr))
        im = ax.imshow(arr, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(range(3))
        ax.set_xticklabels([MODEL_LABELS[c] for c in cols], rotation=20)
        ylabels = [f"{SIM_LABELS[sim]} | {snr}" for sim in SWEEP_FILES for snr in [0.1, 0.5, 1.0, 3.0]]
        ax.set_yticks(range(16))
        ax.set_yticklabels(ylabels, fontsize=8)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if np.isnan(arr[i, j]):
                    text = "NA"
                else:
                    text = f"{arr[i, j]:.2f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_markdown_summary(
    win_counts: pd.DataFrame,
    margins: pd.DataFrame,
    out_path: Path,
) -> None:
    lines = ["# GR-RHS 4x4 Group Advantage Summary", ""]
    lines.append("## Win Counts")
    lines.append("")
    lines.append("| Metric | GR-RHS Wins | Total Scenarios |")
    lines.append("| --- | ---: | ---: |")
    for _, row in win_counts.iterrows():
        lines.append(f"| {row['metric']} | {int(row['grrhs_win_count'])} | {int(row['total_scenarios'])} |")
    lines.append("")
    lines.append("## Baseline Margins")
    lines.append("")
    lines.append("| Baseline | Metric | Direction | Mean Margin for GR-RHS | GR-RHS Better Count | n |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: |")
    for _, row in margins.iterrows():
        lines.append(
            f"| {MODEL_LABELS.get(row['baseline'], row['baseline'])} | {row['metric']} | {row['better_direction']} | "
            f"{row['mean_margin_for_grrhs']:.6f} | {int(row['grrhs_better_count'])} | {int(row['n'])} |"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "outputs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_all_sweeps(repo_root)
    win_counts = _winner_counts(df)
    margins = _baseline_margins(df)

    win_counts.to_csv(out_dir / "grrhs_4x4_win_counts.csv", index=False)
    margins.to_csv(out_dir / "grrhs_4x4_baseline_margins.csv", index=False)
    _write_markdown_summary(
        win_counts,
        margins,
        out_dir / "grrhs_4x4_summary.md",
    )

    _plot_metric_grid(
        df,
        metric="GroupNormRMSE",
        ylabel="GroupNormRMSE (lower is better)",
        out_path=out_dir / "grrhs_4x4_groupnormrmse.png",
    )
    _plot_metric_grid(
        df,
        metric="BetaRMSE",
        ylabel="BetaRMSE (lower is better)",
        out_path=out_dir / "grrhs_4x4_betarmse.png",
    )
    _plot_margin_heatmaps(
        df,
        out_path=out_dir / "grrhs_4x4_margin_heatmaps.png",
    )


if __name__ == "__main__":
    main()
