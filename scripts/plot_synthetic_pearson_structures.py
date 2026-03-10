from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import yaml

from data.generators import generate_synthetic, synthetic_config_from_dict


SCENARIOS: List[Tuple[str, Path, str]] = [
    ("sim_g5", Path("configs/experiments/sim_g5_grouped_mixed.yaml"), "Synthetic dataset: grouped mixed signal"),
]

TAG_COLORS = {
    "strong": "#b42318",
    "noise": "#1f77b4",
    "weak": "#6b7280",
    "null": "#6b7280",
}
TAG_PRIORITY = {"strong": 3, "noise": 2, "weak": 1, "null": 0}


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload or {}


def _feature_corr(X: np.ndarray) -> np.ndarray:
    return np.corrcoef(X, rowvar=False)


def _group_correlation_matrix(corr: np.ndarray, groups: Sequence[Sequence[int]]) -> np.ndarray:
    mat = np.zeros((len(groups), len(groups)), dtype=float)
    for i, gi in enumerate(groups):
        idx_i = np.asarray(gi, dtype=int)
        for j, gj in enumerate(groups):
            idx_j = np.asarray(gj, dtype=int)
            block = corr[np.ix_(idx_i, idx_j)]
            if i == j:
                if block.shape[0] <= 1:
                    value = 1.0
                else:
                    mask = ~np.eye(block.shape[0], dtype=bool)
                    value = float(block[mask].mean()) if np.any(mask) else 1.0
            else:
                value = float(block.mean())
            mat[i, j] = value
    return mat


def _group_boundaries(groups: Sequence[Sequence[int]]) -> tuple[List[int], List[Tuple[float, str]]]:
    boundaries: List[int] = []
    centers: List[Tuple[float, str]] = []
    cursor = 0
    for gid, group in enumerate(groups):
        size = len(group)
        centers.append((cursor + (size - 1) / 2.0, f"G{gid + 1}"))
        cursor += size
        if gid < len(groups) - 1:
            boundaries.append(cursor)
    return boundaries, centers


def _group_active_summary(beta: np.ndarray, groups: Sequence[Sequence[int]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    active = np.abs(beta) > 1e-10
    for gid, group in enumerate(groups):
        idx = np.asarray(group, dtype=int)
        rows.append(
            {
                "group": f"G{gid + 1}",
                "group_index": gid,
                "group_size": int(idx.size),
                "active_count": int(active[idx].sum()),
                "active_fraction": float(active[idx].mean()),
                "beta_norm": float(np.linalg.norm(beta[idx])),
            }
        )
    return pd.DataFrame(rows)


def _feature_tags(info: Mapping[str, Any], p: int) -> List[str]:
    tags = ["null"] * p
    blueprint = info.get("signal_blueprint") or {}
    tagged = blueprint.get("tags") or {}
    if not isinstance(tagged, Mapping):
        return tags
    for tag, indices in tagged.items():
        if tag not in TAG_PRIORITY:
            continue
        for idx in indices:
            j = int(idx)
            if 0 <= j < p and TAG_PRIORITY[tag] >= TAG_PRIORITY.get(tags[j], 0):
                tags[j] = str(tag)
    return tags


def _group_summary(beta: np.ndarray, groups: Sequence[Sequence[int]], feature_tags: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    active = np.abs(beta) > 1e-10
    for gid, group in enumerate(groups):
        idx = np.asarray(group, dtype=int)
        tag_counts = {
            tag: int(sum(1 for j in idx if feature_tags[int(j)] == tag))
            for tag in ("strong", "noise", "weak")
        }
        if any(tag_counts.values()):
            group_tag = max(tag_counts, key=lambda key: (tag_counts[key], TAG_PRIORITY[key]))
        else:
            group_tag = "null"
        rows.append(
            {
                "group": f"G{gid + 1}",
                "group_index": gid,
                "group_size": int(idx.size),
                "active_count": int(active[idx].sum()),
                "beta_norm": float(np.linalg.norm(beta[idx])),
                "group_tag": group_tag,
            }
        )
    return pd.DataFrame(rows)


def _style_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#222222")
    ax.set_facecolor("white")


def _plot_full_heatmap(ax: plt.Axes, corr: np.ndarray, groups: Sequence[Sequence[int]], title: str) -> None:
    _style_axis(ax)
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    boundaries, centers = _group_boundaries(groups)
    for boundary in boundaries:
        ax.axhline(boundary - 0.5, color="white", linewidth=1.1)
        ax.axvline(boundary - 0.5, color="white", linewidth=1.1)
    for center, label in centers:
        ax.text(center, -6.0, label, ha="center", va="bottom", fontsize=10.2, fontweight="bold", clip_on=False)
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
    cbar.ax.set_ylabel("Pearson r", rotation=90, fontsize=10)
    cbar.ax.tick_params(labelsize=8.5)


def _plot_group_heatmap(ax: plt.Axes, group_corr: np.ndarray, active_summary: pd.DataFrame) -> None:
    _style_axis(ax)
    labels = [
        f"{row['group']}\n(n={int(row['group_size'])}, act={int(row['active_count'])})"
        for _, row in active_summary.iterrows()
    ]
    im = ax.imshow(group_corr, cmap="coolwarm", vmin=-1.0, vmax=1.0, interpolation="nearest")
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8.8)
    ax.set_yticklabels(labels, fontsize=8.8)
    ax.set_title("Block-averaged group Pearson heatmap", fontsize=13, fontweight="bold", pad=14)
    for i in range(group_corr.shape[0]):
        for j in range(group_corr.shape[1]):
            val = float(group_corr[i, j])
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=8.8,
                color="white" if abs(val) > 0.45 else "#222222",
            )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Pearson r", rotation=90, fontsize=10)
    cbar.ax.tick_params(labelsize=8.5)


def _plot_group_signal_contrast(
    ax: plt.Axes,
    beta: np.ndarray,
    groups: Sequence[Sequence[int]],
    feature_tags: Sequence[str],
    summary: pd.DataFrame,
    ymax: float,
) -> None:
    _style_axis(ax)
    gap = 4.0
    left_pad = 4.5
    spans: List[Tuple[float, float, float]] = []
    cursor = left_pad
    for group in groups:
        size = float(len(group))
        start = cursor
        end = cursor + size
        center = (start + end) / 2.0
        spans.append((start, end, center))
        cursor = end + gap

    for (start, end, _), (_, row) in zip(spans, summary.iterrows()):
        ax.add_patch(
            Rectangle(
                (start, -0.02 * ymax),
                end - start,
                1.04 * ymax,
                facecolor=TAG_COLORS[row["group_tag"]],
                edgecolor="none",
                alpha=0.08,
                zorder=0,
            )
        )
        ax.vlines([start, end], ymin=-0.02 * ymax, ymax=1.04 * ymax, color="#e5e7eb", linewidth=0.8, zorder=1)

    for (start, end, _), group in zip(spans, groups):
        idx = np.asarray(group, dtype=int)
        x = np.linspace(start + 0.5, end - 0.5, idx.size)
        y = np.abs(beta[idx])
        colors = [TAG_COLORS[feature_tags[int(j)]] for j in idx]
        ax.vlines(x, 0.0, y, colors=colors, linewidth=0.9, alpha=0.65, zorder=2)
        ax.scatter(x, y, s=14, color=colors, edgecolors="white", linewidths=0.25, zorder=3)

    for (start, end, center), (_, row) in zip(spans, summary.iterrows()):
        strip_y0 = ymax * 1.02
        strip_h = ymax * 0.07
        ax.add_patch(
            Rectangle(
                (start, strip_y0),
                end - start,
                strip_h,
                facecolor=TAG_COLORS[row["group_tag"]],
                edgecolor="none",
                alpha=0.9,
                clip_on=False,
                zorder=2,
            )
        )
        # no text label on the strip

    ax.axhline(0.0, color="#6b7280", linewidth=0.9)
    ax.grid(axis="y", alpha=0.18, linewidth=0.8)
    ax.set_xlim(0.0, spans[-1][1] + 1.2)
    ax.set_ylim(-0.03 * ymax, 1.15 * ymax)
    ax.set_ylabel(r"$|\beta_j|$")
    ax.set_title("Grouped signal contrast", fontsize=13, fontweight="bold", pad=22)
    ax.set_xticks([center for _, _, center in spans])
    ax.set_xticklabels([f"{row['group']}" for _, row in summary.iterrows()], fontsize=8.6)


def _save_full_heatmap_figure(
    out_dir: Path,
    scenario_key: str,
    title: str,
    corr: np.ndarray,
    groups: Sequence[Sequence[int]],
) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 9.0), constrained_layout=False)
    _plot_full_heatmap(ax, corr, groups, title="Full feature Pearson heatmap")
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.97)
    fig.text(
        0.03,
        0.03,
        "Feature-feature Pearson structure with group boundaries.",
        fontsize=10,
        color="#222222",
    )
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.08)
    fig.savefig(out_dir / f"{scenario_key}_full_pearson_heatmap.png", dpi=230, bbox_inches="tight")
    plt.close(fig)


def _save_group_heatmap_figure(
    out_dir: Path,
    scenario_key: str,
    title: str,
    group_corr: np.ndarray,
    active_summary: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 8.4), constrained_layout=False)
    _plot_group_heatmap(ax, group_corr, active_summary)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.97)
    fig.text(
        0.03,
        0.03,
        "Block-averaged group correlations annotated with group size and true active count.",
        fontsize=10,
        color="#222222",
    )
    plt.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.15)
    fig.savefig(out_dir / f"{scenario_key}_group_pearson_heatmap.png", dpi=230, bbox_inches="tight")
    plt.close(fig)


def _write_scenario_summary(
    out_dir: Path,
    scenario_key: str,
    active_summary: pd.DataFrame,
    corr: np.ndarray,
    group_corr: np.ndarray,
) -> None:
    active_summary.to_csv(out_dir / f"{scenario_key}_group_activity.csv", index=False)
    payload = {
        "scenario": scenario_key,
        "within_group_corr_mean": [float(group_corr[i, i]) for i in range(group_corr.shape[0])],
        "max_abs_feature_corr_offdiag": float(np.max(np.abs(corr - np.eye(corr.shape[0])))),
        "group_activity": active_summary.to_dict(orient="records"),
    }
    (out_dir / f"{scenario_key}_pearson_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _render_scenario(out_dir: Path, scenario_key: str, title: str, cfg_path: Path) -> None:
    cfg = _load_yaml(cfg_path)
    data_cfg = dict(cfg.get("data", {}) or {})
    synthetic_cfg = synthetic_config_from_dict(data_cfg, seed=data_cfg.get("seed"), name=scenario_key, task="regression")
    dataset = generate_synthetic(synthetic_cfg)

    corr = _feature_corr(np.asarray(dataset.X, dtype=float))
    group_corr = _group_correlation_matrix(corr, dataset.groups)
    active_summary = _group_active_summary(np.asarray(dataset.beta, dtype=float), dataset.groups)
    feature_tags = _feature_tags(dataset.info or {}, dataset.beta.size)
    group_summary = _group_summary(np.asarray(dataset.beta, dtype=float), dataset.groups, feature_tags)

    ymax = max(0.6, float(np.max(np.abs(dataset.beta))))

    fig = plt.figure(figsize=(18.0, 7.4), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=[4.0, 6.0])
    ax_full = fig.add_subplot(gs[0, 0])
    ax_group = fig.add_subplot(gs[0, 1])

    _plot_full_heatmap(ax_full, corr, dataset.groups, title="Full feature Pearson heatmap")
    _plot_group_signal_contrast(ax_group, np.asarray(dataset.beta, dtype=float), dataset.groups, feature_tags, group_summary, ymax)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=7, color=TAG_COLORS["strong"], label="Signal"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=7, color=TAG_COLORS["noise"], label="Noise (looks like signal)"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=7, color=TAG_COLORS["null"], label="Other"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.72, 0.98),
    )

    plt.subplots_adjust(left=0.05, right=0.98, top=0.94, bottom=0.10, wspace=0.16)
    fig.savefig(out_dir / f"{scenario_key}_pearson_with_signal_contrast.png", dpi=230, bbox_inches="tight")
    plt.close(fig)

    _save_full_heatmap_figure(out_dir, scenario_key, title, corr, dataset.groups)
    _save_group_heatmap_figure(out_dir, scenario_key, title, group_corr, active_summary)

    _write_scenario_summary(out_dir, scenario_key, active_summary, corr, group_corr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Pearson-structure figures for the four synthetic benchmark datasets.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/reports/synthetic_pearson_structures"),
        help="Destination directory for the four synthetic Pearson plots.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for scenario_key, rel_cfg, title in SCENARIOS:
        _render_scenario(out_dir, scenario_key, title, repo_root / rel_cfg)

    print(f"[ok] synthetic Pearson figures written to {out_dir}")


if __name__ == "__main__":
    main()
