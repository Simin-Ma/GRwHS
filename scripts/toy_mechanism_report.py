from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from data.generators import generate_synthetic, synthetic_config_from_dict


MODEL_COLORS: Dict[str, str] = {
    "truth": "#222222",
    "grrhs": "#153B50",
    "gigg": "#3A7D44",
    "rhs": "#5B4B8A",
}


def _latest_comparison_csv(sweep_dir: Path) -> Path:
    candidates = sorted(sweep_dir.glob("sweep_comparison_*.csv"))
    if not candidates:
        raise SystemExit(f"No sweep_comparison_*.csv found in {sweep_dir}")
    return candidates[-1]


def _load_posterior_beta_mean(run_dir: Path) -> np.ndarray:
    summary = run_dir / "repeat_001" / "fold_01" / "posterior_summary.csv"
    if not summary.exists():
        raise FileNotFoundError(f"Missing posterior_summary.csv at {summary}")
    frame = pd.read_csv(summary)
    beta_rows = frame[frame["parameter"] == "beta"].copy()
    if beta_rows.empty:
        raise ValueError(f"No beta rows found in {summary}")
    beta_rows = beta_rows.sort_values("index")
    return beta_rows["mean"].to_numpy(dtype=float)


def _load_truth_from_resolved_config(run_dir: Path) -> Tuple[np.ndarray, List[List[int]]]:
    cfg_path = run_dir / "resolved_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data_cfg = cfg.get("data", {})
    task = str(cfg.get("task", "regression")).lower()
    synth_cfg = synthetic_config_from_dict(data_cfg, task=task, name=str(cfg.get("name", "toy_mechanism_main")))
    dataset = generate_synthetic(synth_cfg)
    return np.asarray(dataset.beta, dtype=float), [list(map(int, g)) for g in dataset.groups]


def _group_mass(beta: np.ndarray, groups: List[List[int]]) -> np.ndarray:
    masses = np.array([float(np.sum(np.abs(beta[idxs]))) for idxs in groups], dtype=float)
    denom = float(np.sum(masses))
    if denom <= 0:
        return masses
    return masses / denom


def _plot_metrics_bar(comparison_csv: Path, out_path: Path) -> None:
    frame = pd.read_csv(comparison_csv)
    metrics = [c for c in ["RMSE", "BetaRMSE", "GroupNormRMSE"] if c in frame.columns]
    if not metrics:
        raise SystemExit(f"No metrics columns found in {comparison_csv}")

    display = frame[["variation", "model", *metrics]].copy()
    display["label"] = display["variation"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.2))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        sub = display[["label", metric]].dropna()
        colors = [MODEL_COLORS.get(str(lbl), "#4C78A8") for lbl in sub["label"]]
        ax.bar(sub["label"], sub[metric], color=colors, edgecolor="white", linewidth=0.8)
        ax.set_title(metric, fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_beta_scatter(truth: np.ndarray, betas: Dict[str, np.ndarray], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.axhline(0.0, color="#999999", linewidth=1.0, alpha=0.5)
    ax.axvline(0.0, color="#999999", linewidth=1.0, alpha=0.5)
    lim = max(1.0, float(np.max(np.abs(truth))) * 1.2)
    ax.plot([-lim, lim], [-lim, lim], linestyle="--", color="#444444", linewidth=1.0, alpha=0.7)
    for name, mean_beta in betas.items():
        ax.scatter(
            truth,
            mean_beta,
            s=28,
            alpha=0.85,
            color=MODEL_COLORS.get(name, "#4C78A8"),
            label=name,
            edgecolors="white",
            linewidths=0.5,
        )
    ax.set_xlabel("True β", fontsize=12)
    ax.set_ylabel("Posterior mean β", fontsize=12)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.legend(frameon=False, fontsize=10)
    ax.set_title("Coefficient recovery (posterior mean vs truth)", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_group_mass(truth: np.ndarray, groups: List[List[int]], betas: Dict[str, np.ndarray], out_path: Path) -> None:
    labels = ["truth", *betas.keys()]
    rows = [truth, *[betas[k] for k in betas.keys()]]
    masses = np.stack([_group_mass(row, groups) for row in rows], axis=0)
    g = masses.shape[1]
    x = np.arange(len(labels))
    bottoms = np.zeros(len(labels), dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    group_colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    for gi in range(g):
        vals = masses[:, gi]
        ax.bar(
            x,
            vals,
            bottom=bottoms,
            label=f"group {gi+1}",
            color=group_colors[gi % len(group_colors)],
            edgecolor="white",
            linewidth=0.6,
        )
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Share of Σ|β| by group", fontsize=12)
    ax.set_title("Group-level |β| mass share", fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, ncols=2)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_recall_at_k(truth: np.ndarray, betas: Dict[str, np.ndarray], out_path: Path) -> None:
    active = np.flatnonzero(np.abs(truth) > 1e-12).astype(int)
    if active.size == 0:
        return
    p = int(truth.shape[0])
    k_grid = np.arange(1, p + 1, dtype=int)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for name, mean_beta in betas.items():
        order = np.argsort(-np.abs(mean_beta))
        hits = np.isin(order, active).astype(int)
        cum_hits = np.cumsum(hits)
        recall = cum_hits / float(active.size)
        ax.plot(
            k_grid,
            recall,
            linewidth=2.2,
            color=MODEL_COLORS.get(name, "#4C78A8"),
            label=name,
        )
    ax.set_xlabel("k (top-k by |posterior mean β|)", fontsize=12)
    ax.set_ylabel("Recall of true nonzeros", fontsize=12)
    ax.set_title("Coefficient support recovery (recall@k)", fontsize=13, fontweight="bold")
    ax.set_xlim(1, p)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_weak_groups_bar(
    truth: np.ndarray,
    groups: List[List[int]],
    betas: Dict[str, np.ndarray],
    *,
    weak_group_ids: List[int],
    out_path: Path,
) -> None:
    idxs: List[int] = []
    for gid in weak_group_ids:
        if 0 <= gid < len(groups):
            idxs.extend(list(groups[gid]))
    idxs = sorted(set(int(i) for i in idxs))
    if not idxs:
        return
    x = np.arange(len(idxs))
    width = 0.22
    fig, ax = plt.subplots(figsize=(max(8.0, 0.35 * len(idxs)), 4.3))
    ax.bar(x - width, truth[idxs], width=width, color=MODEL_COLORS["truth"], label="truth", alpha=0.9)
    for i, (name, mean_beta) in enumerate(betas.items()):
        ax.bar(
            x + (i * width),
            mean_beta[idxs],
            width=width,
            color=MODEL_COLORS.get(name, "#4C78A8"),
            label=name,
            alpha=0.9,
        )
    ax.axhline(0.0, color="#999999", linewidth=1.0, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in idxs], rotation=0)
    ax.set_xlabel("Coefficient index (weak-signal groups)", fontsize=12)
    ax.set_ylabel("β (truth vs posterior mean)", fontsize=12)
    ax.set_title("Weak-group coefficient recovery", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncols=3, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small report for the toy_mechanism_main sweep.")
    parser.add_argument("--sweep-dir", type=Path, required=True, help="Sweep directory under outputs/sweeps/...")
    args = parser.parse_args()

    sweep_dir = args.sweep_dir
    comparison_csv = _latest_comparison_csv(sweep_dir)
    frame = pd.read_csv(comparison_csv)
    if frame.empty:
        raise SystemExit(f"{comparison_csv} is empty")

    run_dirs: Dict[str, Path] = {}
    for _, row in frame.iterrows():
        var = str(row.get("variation", "")).strip()
        run_dir = str(row.get("run_dir", "")).strip()
        if var and run_dir:
            run_dirs[var] = Path(run_dir)

    if "grrhs" not in run_dirs:
        raise SystemExit("Missing variation 'grrhs' in comparison CSV.")

    # Optional variations.
    present = [name for name in ["grrhs", "gigg", "rhs"] if name in run_dirs]
    if len(present) < 2:
        raise SystemExit(f"Need at least two variations in comparison CSV, found: {present}")

    truth, groups = _load_truth_from_resolved_config(run_dirs["grrhs"])
    betas = {name: _load_posterior_beta_mean(run_dirs[name]) for name in present if name != "truth"}

    report_dir = sweep_dir / "report_latest"
    report_dir.mkdir(parents=True, exist_ok=True)

    _plot_metrics_bar(comparison_csv, report_dir / "metrics_bar.png")
    _plot_beta_scatter(truth, betas, report_dir / "beta_scatter.png")
    _plot_group_mass(truth, groups, betas, report_dir / "group_mass_share.png")
    _plot_recall_at_k(truth, betas, report_dir / "recall_at_k.png")
    _plot_weak_groups_bar(truth, groups, betas, weak_group_ids=[0, 1], out_path=report_dir / "weak_groups_bar.png")

    meta = {
        "comparison_csv": str(comparison_csv),
        "run_dirs": {k: str(v) for k, v in run_dirs.items()},
        "report_dir": str(report_dir),
    }
    (report_dir / "report_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[ok] wrote report to {report_dir}")


if __name__ == "__main__":
    main()
