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
    "sim_s1": "S1 Concentrated",
    "sim_s2": "S2 Distributed",
    "sim_s3": "S3 Dense",
    "sim_s4": "S4 Half-Dense",
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

FOCUS_MODELS = ["grrhs", "rhs", "sgl", "gigg"]
BASELINES = ["rhs", "sgl", "gigg"]
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
                    "sim_label": SIM_LABELS[sim],
                    "snr": float(match.group(1).replace("p", ".")),
                    "model": match.group(2),
                    "BetaRMSE": pd.to_numeric(row.get("BetaRMSE"), errors="coerce"),
                    "RMSE": pd.to_numeric(row.get("RMSE"), errors="coerce"),
                    "status": row.get("status"),
                }
            )
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out["model"], categories=FOCUS_MODELS + ["lasso", "ridge"], ordered=True)
    return out.sort_values(["sim", "snr", "model"]).reset_index(drop=True)


def _write_summary_tables(df: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = df.pivot_table(index=["sim", "snr"], columns="model", values=["BetaRMSE", "RMSE"])

    rows = []
    for sim in SWEEP_FILES:
        for snr in SNR_ORDER:
            record = {"sim": sim, "snr": snr}
            for model in ["grrhs", "rhs", "sgl", "gigg", "lasso", "ridge"]:
                record[MODEL_LABELS.get(model, model)] = float(pivot.loc[(sim, snr), ("BetaRMSE", model)])
            best_model = min(
                [(model, record[MODEL_LABELS[model]]) for model in ["grrhs", "rhs", "sgl", "gigg", "lasso", "ridge"]],
                key=lambda item: item[1],
            )[0]
            record["Best"] = MODEL_LABELS[best_model]
            rows.append(record)
    scenario_table = pd.DataFrame(rows)
    scenario_table.to_csv(out_dir / "betarmse_4x4_scenario_table.csv", index=False)

    margin_rows = []
    for baseline in BASELINES:
        diff = pivot["BetaRMSE"][baseline] - pivot["BetaRMSE"]["grrhs"]
        margin_rows.append(
            {
                "baseline": MODEL_LABELS[baseline],
                "mean_margin": float(diff.mean()),
                "median_margin": float(diff.median()),
                "grrhs_better_count": int((diff > 0).sum()),
                "total": int(diff.notna().sum()),
            }
        )
    margin_table = pd.DataFrame(margin_rows)
    margin_table.to_csv(out_dir / "betarmse_4x4_margin_table.csv", index=False)
    return scenario_table, margin_table


def _plot_main_figure(df: pd.DataFrame, margin_table: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.9])

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]

    for ax, sim in zip(axes, SWEEP_FILES):
        sub = df[(df["sim"] == sim) & (df["model"].isin(FOCUS_MODELS))].copy()
        for model in FOCUS_MODELS:
            curve = sub[sub["model"] == model].sort_values("snr")
            alpha = 1.0 if model == "grrhs" else 0.72
            linewidth = 3.2 if model == "grrhs" else 2.0
            marker = "o" if model == "grrhs" else "s"
            ax.plot(
                curve["snr"],
                curve["BetaRMSE"],
                color=MODEL_COLORS[model],
                linewidth=linewidth,
                marker=marker,
                alpha=alpha,
                label=MODEL_LABELS[model],
            )
        ax.set_title(SIM_LABELS[sim])
        ax.set_xticks(SNR_ORDER)
        ax.set_xticklabels(["0.1", "0.5", "1.0", "3.0"])
        ax.set_xlabel("SNR")
        ax.set_ylabel("BetaRMSE")
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01))

    ax_heat = fig.add_subplot(gs[2, 0])
    pivot = df.pivot_table(index=["sim", "snr"], columns="model", values="BetaRMSE")
    mat = []
    ylabels = []
    for sim in SWEEP_FILES:
        for snr in SNR_ORDER:
            row = []
            for baseline in BASELINES:
                row.append(float(pivot.loc[(sim, snr), baseline] - pivot.loc[(sim, snr), "grrhs"]))
            mat.append(row)
            ylabels.append(f"{SIM_LABELS[sim]} | {snr}")
    arr = np.asarray(mat)
    vmax = float(np.abs(arr).max())
    im = ax_heat.imshow(arr, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax_heat.set_title("GR-RHS advantage on BetaRMSE\n(positive = lower error than baseline)")
    ax_heat.set_xticks(range(len(BASELINES)))
    ax_heat.set_xticklabels([MODEL_LABELS[x] for x in BASELINES])
    ax_heat.set_yticks(range(len(ylabels)))
    ax_heat.set_yticklabels(ylabels, fontsize=8)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax_heat.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax_heat, shrink=0.85)

    ax_bar = fig.add_subplot(gs[2, 1])
    bar_vals = margin_table["mean_margin"].to_numpy()
    bar_labels = margin_table["baseline"].tolist()
    colors = [MODEL_COLORS["rhs"], MODEL_COLORS["sgl"], MODEL_COLORS["gigg"]]
    bars = ax_bar.bar(bar_labels, bar_vals, color=colors, alpha=0.85)
    ax_bar.axhline(0.0, color="black", linewidth=1)
    ax_bar.set_title("Average BetaRMSE gain of GR-RHS")
    ax_bar.set_ylabel("Baseline BetaRMSE - GR-RHS BetaRMSE")
    ax_bar.grid(axis="y", alpha=0.25)
    for bar, value, wins, total in zip(
        bars,
        bar_vals,
        margin_table["grrhs_better_count"].tolist(),
        margin_table["total"].tolist(),
    ):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:.2f}\n{wins}/{total}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )

    fig.suptitle(
        "4x4 synthetic benchmark: BetaRMSE-focused view\n"
        "Use this figure to explain coefficient recovery, not generic prediction",
        fontsize=15,
        y=1.03,
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_rank_strip(df: pd.DataFrame, out_path: Path) -> None:
    rank_rows = []
    for (sim, snr), sub in df.groupby(["sim", "snr"], observed=True):
        ordered = sub.sort_values("BetaRMSE")
        for rank, (_, row) in enumerate(ordered.iterrows(), start=1):
            rank_rows.append(
                {
                    "scenario": f"{SIM_LABELS[sim]} | {snr}",
                    "rank": rank,
                    "model": row["model"],
                    "BetaRMSE": row["BetaRMSE"],
                }
            )
    rank_df = pd.DataFrame(rank_rows)

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    scenario_order = [f"{SIM_LABELS[sim]} | {snr}" for sim in SWEEP_FILES for snr in SNR_ORDER]
    y_map = {name: i for i, name in enumerate(scenario_order)}
    for model in ["grrhs", "rhs", "sgl", "lasso", "ridge", "gigg"]:
        sub = rank_df[rank_df["model"] == model]
        ax.scatter(
            sub["rank"],
            [y_map[s] for s in sub["scenario"]],
            s=80 if model == "grrhs" else 42,
            color=MODEL_COLORS.get(model, "#333333"),
            alpha=1.0 if model == "grrhs" else 0.75,
            label=MODEL_LABELS.get(model, model),
        )
    ax.set_xlim(0.5, 6.5)
    ax.set_xticks(range(1, 7))
    ax.set_xlabel("Rank by BetaRMSE (1 = best)")
    ax.set_yticks(range(len(scenario_order)))
    ax.set_yticklabels(scenario_order, fontsize=8)
    ax.set_title("Model rank in each scenario using BetaRMSE")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_notes(margin_table: pd.DataFrame, out_path: Path) -> None:
    rhs = margin_table[margin_table["baseline"] == "RHS"].iloc[0]
    sgl = margin_table[margin_table["baseline"] == "SGL"].iloc[0]
    gigg = margin_table[margin_table["baseline"] == "GIGG"].iloc[0]
    lines = [
        "# BetaRMSE-focused interpretation",
        "",
        "Recommended simple explanation:",
        "- BetaRMSE is coefficient recovery error. Lower means the estimated coefficients are closer to the true beta vector.",
        "- If the paper wants to show grouped-shrinkage quality, BetaRMSE is easier to explain than posterior metrics and more structural than plain RMSE.",
        "- Read the heatmap as: positive values mean GR-RHS has lower coefficient recovery error than the baseline.",
        "",
        "Headline numbers:",
        f"- Versus RHS: average BetaRMSE gain = {rhs['mean_margin']:.3f}, better in {int(rhs['grrhs_better_count'])}/{int(rhs['total'])} scenarios.",
        f"- Versus SGL: average BetaRMSE gain = {sgl['mean_margin']:.3f}, better in {int(sgl['grrhs_better_count'])}/{int(sgl['total'])} scenarios.",
        f"- Versus GIGG: average BetaRMSE gain = {gigg['mean_margin']:.3f}, better in {int(gigg['grrhs_better_count'])}/{int(gigg['total'])} scenarios.",
        "",
        "Suggested narrative:",
        "- Main claim: GR-RHS is not only competitive in prediction; it more reliably reconstructs the true coefficient pattern.",
        "- Secondary claim: the advantage is strongest against GIGG, consistent across almost all scenarios, and still usually positive against RHS and SGL.",
        "- Interpretation shortcut: if the audience only remembers one panel, let it be the BetaRMSE heatmap.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "outputs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_data(repo_root)
    scenario_table, margin_table = _write_summary_tables(df, out_dir)
    _plot_main_figure(df, margin_table, out_dir / "grrhs_4x4_betarmse_focus.png")
    _plot_rank_strip(df, out_dir / "grrhs_4x4_betarmse_ranks.png")
    _write_notes(margin_table, out_dir / "grrhs_4x4_betarmse_notes.md")


if __name__ == "__main__":
    main()
