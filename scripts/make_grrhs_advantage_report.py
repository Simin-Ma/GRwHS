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
    "sim_s1": "S1  Sparse-Strong",
    "sim_s2": "S2  Dense-Weak",
    "sim_s3": "S3  Mixed",
    "sim_s4": "S4  Half-Dense",
}

MODEL_ORDER = ["grrhs", "rhs", "gigg", "sgl", "lasso", "ridge"]
BASELINES = ["rhs", "gigg", "sgl"]

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

SNR_ORDER = [0.1, 0.5, 1.0, 3.0]


def _load_frame(repo_root: Path) -> pd.DataFrame:
    rows = []
    for sim, timestamp in SWEEP_FILES.items():
        path = repo_root / "outputs" / "sweeps" / sim / f"sweep_comparison_{timestamp}.csv"
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            match = re.match(r"snr([0-9p]+)_(.+)", str(row["variation"]))
            if match is None:
                continue
            payload = {
                "sim": sim,
                "sim_label": SIM_LABELS[sim],
                "snr": float(match.group(1).replace("p", ".")),
                "model": match.group(2),
            }
            for metric in ["RMSE", "BetaRMSE", "GroupNormRMSE", "AUC-PR", "F1"]:
                payload[metric] = pd.to_numeric(row.get(metric), errors="coerce")
            rows.append(payload)
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out["model"], categories=MODEL_ORDER, ordered=True)
    return out.sort_values(["sim", "snr", "model"]).reset_index(drop=True)


def _plot_advantage_heatmap(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
    title: str,
    *,
    higher_is_better: bool = False,
) -> None:
    pivot = df.pivot_table(index=["sim", "snr"], columns="model", values=metric)
    mat = []
    ylabels = []
    for sim in SWEEP_FILES:
        for snr in SNR_ORDER:
            row = []
            for baseline in BASELINES:
                if higher_is_better:
                    value = pivot.loc[(sim, snr), "grrhs"] - pivot.loc[(sim, snr), baseline]
                else:
                    value = pivot.loc[(sim, snr), baseline] - pivot.loc[(sim, snr), "grrhs"]
                row.append(float(value))
            mat.append(row)
            ylabels.append(f"{SIM_LABELS[sim]} | {snr}")
    arr = np.asarray(mat, dtype=float)
    vmax = float(np.nanmax(np.abs(arr)))

    fig, ax = plt.subplots(figsize=(8.6, 8.2), constrained_layout=True)
    im = ax.imshow(arr, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(BASELINES)))
    ax.set_xticklabels([MODEL_LABELS[x] for x in BASELINES], fontsize=11)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=9)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.88)
    cbar.set_label("Positive = GR-RHS better", rotation=90)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_betarmse_rank(df: pd.DataFrame, out_path: Path) -> None:
    rank_rows = []
    for (sim, snr), sub in df.groupby(["sim", "snr"], observed=True):
        ordered = sub.sort_values("BetaRMSE")
        for rank, (_, row) in enumerate(ordered.iterrows(), start=1):
            rank_rows.append(
                {
                    "scenario": f"{SIM_LABELS[sim]} | {snr}",
                    "rank": rank,
                    "model": row["model"],
                }
            )
    rank_df = pd.DataFrame(rank_rows)
    scenario_order = [f"{SIM_LABELS[sim]} | {snr}" for sim in SWEEP_FILES for snr in SNR_ORDER]
    y_map = {name: idx for idx, name in enumerate(scenario_order)}

    fig, ax = plt.subplots(figsize=(10.2, 8.6), constrained_layout=True)
    for model in MODEL_ORDER:
        sub = rank_df[rank_df["model"] == model]
        ax.scatter(
            sub["rank"],
            [y_map[s] for s in sub["scenario"]],
            s=82 if model == "grrhs" else 42,
            color=MODEL_COLORS[model],
            alpha=1.0 if model == "grrhs" else 0.72,
            label=MODEL_LABELS[model],
        )
    ax.set_xlim(0.5, 6.5)
    ax.set_xticks(range(1, 7))
    ax.set_xlabel("Rank by BetaRMSE (1 = best)", fontsize=12)
    ax.set_yticks(range(len(scenario_order)))
    ax.set_yticklabels(scenario_order, fontsize=8)
    ax.set_title("GR-RHS rank stability across all 16 scenarios", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", frameon=False, ncol=2)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _write_tables(df: pd.DataFrame, out_dir: Path) -> None:
    pivot = df.pivot_table(index=["sim", "snr"], columns="model", values=["BetaRMSE", "GroupNormRMSE", "RMSE", "AUC-PR"])

    summary_rows = []
    for baseline in BASELINES:
        beta_diff = pivot["BetaRMSE"][baseline] - pivot["BetaRMSE"]["grrhs"]
        group_diff = pivot["GroupNormRMSE"][baseline] - pivot["GroupNormRMSE"]["grrhs"]
        rmse_diff = pivot["RMSE"][baseline] - pivot["RMSE"]["grrhs"]
        auc_diff = pivot["AUC-PR"]["grrhs"] - pivot["AUC-PR"][baseline]
        summary_rows.append(
            {
                "Baseline": MODEL_LABELS[baseline],
                "Mean_BetaRMSE_Gain": float(beta_diff.mean()),
                "BetaRMSE_Wins": int((beta_diff > 0).sum()),
                "Mean_GroupNormRMSE_Gain": float(group_diff.mean()),
                "GroupNormRMSE_Wins": int((group_diff > 0).sum()),
                "Mean_RMSE_Gain": float(rmse_diff.mean()),
                "RMSE_Wins": int((rmse_diff > 0).sum()),
                "Mean_AUCPR_Gain": float(auc_diff.mean()),
                "AUCPR_Wins": int((auc_diff > 0).sum()),
                "Total": int(beta_diff.notna().sum()),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "grrhs_advantage_summary_table.csv", index=False)

    scenario_rows = []
    for sim in SWEEP_FILES:
        for snr in SNR_ORDER:
            sub = df[(df["sim"] == sim) & (df["snr"] == snr)].sort_values("BetaRMSE")
            best = sub.iloc[0]
            scenario_rows.append(
                {
                    "Scenario": SIM_LABELS[sim],
                    "SNR": snr,
                    "Best_BetaRMSE_Model": MODEL_LABELS[str(best["model"])],
                    "Best_BetaRMSE": float(best["BetaRMSE"]),
                    "GR-RHS_BetaRMSE": float(sub[sub["model"] == "grrhs"]["BetaRMSE"].iloc[0]),
                    "RHS_BetaRMSE": float(sub[sub["model"] == "rhs"]["BetaRMSE"].iloc[0]),
                    "GIGG_BetaRMSE": float(sub[sub["model"] == "gigg"]["BetaRMSE"].iloc[0]),
                    "SGL_BetaRMSE": float(sub[sub["model"] == "sgl"]["BetaRMSE"].iloc[0]),
                }
            )
    scenario_df = pd.DataFrame(scenario_rows)
    scenario_df.to_csv(out_dir / "grrhs_advantage_scenario_table.csv", index=False)

    md_lines = ["# GR-RHS Advantage Summary", ""]
    md_lines.append("## Summary Table")
    md_lines.append("")
    md_lines.append("| Baseline | Mean BetaRMSE Gain | BetaRMSE Wins | Mean GroupNormRMSE Gain | Group Wins | Mean RMSE Gain | RMSE Wins | Mean AUC-PR Gain | AUC-PR Wins | Total |")
    md_lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, row in summary_df.iterrows():
        md_lines.append(
            f"| {row['Baseline']} | {row['Mean_BetaRMSE_Gain']:.3f} | {int(row['BetaRMSE_Wins'])} | "
            f"{row['Mean_GroupNormRMSE_Gain']:.3f} | {int(row['GroupNormRMSE_Wins'])} | "
            f"{row['Mean_RMSE_Gain']:.3f} | {int(row['RMSE_Wins'])} | "
            f"{row['Mean_AUCPR_Gain']:.3f} | {int(row['AUCPR_Wins'])} | {int(row['Total'])} |"
        )
    (out_dir / "grrhs_advantage_summary_table.md").write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "outputs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_frame(repo_root)
    _plot_advantage_heatmap(
        df,
        metric="BetaRMSE",
        out_path=out_dir / "grrhs_advantage_heatmap_betarmse.png",
        title="GR-RHS advantage on BetaRMSE",
    )
    _plot_advantage_heatmap(
        df,
        metric="GroupNormRMSE",
        out_path=out_dir / "grrhs_advantage_heatmap_groupnormrmse.png",
        title="GR-RHS advantage on GroupNormRMSE",
    )
    _plot_betarmse_rank(df, out_dir / "grrhs_advantage_betarmse_rank.png")
    _write_tables(df, out_dir)


if __name__ == "__main__":
    main()
