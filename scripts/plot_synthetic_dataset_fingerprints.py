from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
TAG_PRIORITY = {"strong": 3, "medium": 2, "weak": 1}


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload or {}


def _feature_corr(X: np.ndarray) -> np.ndarray:
    return np.corrcoef(X, rowvar=False)


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


def _infer_group_tags(info: Mapping[str, Any], groups: Sequence[Sequence[int]]) -> List[str]:
    tags = ["null"] * len(groups)
    blueprint = info.get("signal_blueprint") or {}
    tagged_indices = blueprint.get("tags") or {}
    if not isinstance(tagged_indices, Mapping):
        return tags

    for gid, group in enumerate(groups):
        group_idx = set(int(idx) for idx in group)
        counts: Dict[str, int] = {}
        for tag, indices in tagged_indices.items():
            if tag not in TAG_PRIORITY:
                continue
            overlap = sum(1 for idx in indices if int(idx) in group_idx)
            if overlap > 0:
                counts[str(tag)] = overlap
        if counts:
            tags[gid] = max(counts, key=lambda key: (counts[key], TAG_PRIORITY.get(key, 0)))
    return tags


def _group_summary(beta: np.ndarray, groups: Sequence[Sequence[int]], tags: Sequence[str]) -> pd.DataFrame:
    active = np.abs(beta) > 1e-10
    rows: List[Dict[str, Any]] = []
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
                "tag": tags[gid],
            }
        )
    return pd.DataFrame(rows)


def _feature_tags(info: Mapping[str, Any], p: int) -> List[str]:
    tags = ["null"] * p
    blueprint = info.get("signal_blueprint") or {}
    tagged_indices = blueprint.get("tags") or {}
    if not isinstance(tagged_indices, Mapping):
        return tags
    for tag, indices in tagged_indices.items():
        for idx in indices:
            i = int(idx)
            if 0 <= i < p:
                tags[i] = str(tag)
    return tags


def _style_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#222222")
    ax.set_facecolor("white")


def _plot_corr_heatmap(ax: plt.Axes, corr: np.ndarray, groups: Sequence[Sequence[int]]) -> None:
    _style_axis(ax)
    offdiag = corr[~np.eye(corr.shape[0], dtype=bool)]
    limit = max(0.15, float(np.quantile(np.abs(offdiag), 0.995)))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-limit, vmax=limit, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Design correlation structure", fontsize=13, fontweight="bold", pad=14)
    boundaries, centers = _group_boundaries(groups)
    for boundary in boundaries:
        ax.axhline(boundary - 0.5, color="white", linewidth=1.1)
        ax.axvline(boundary - 0.5, color="white", linewidth=1.1)
    for center, label in centers:
        ax.text(center, -3.6, label, ha="center", va="bottom", fontsize=9.2, fontweight="bold", clip_on=False)
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
    cbar.ax.set_ylabel("Pearson r", rotation=90, fontsize=10)
    cbar.ax.tick_params(labelsize=8.5)


def _plot_feature_lines(
    ax: plt.Axes,
    beta: np.ndarray,
    groups: Sequence[Sequence[int]],
    tags: Sequence[str],
    group_tags: Sequence[str],
) -> None:
    _style_axis(ax)
    p = beta.size
    x = np.arange(p, dtype=float)
    colors = [TAG_COLORS.get(tag, "#777777") for tag in tags]
    y = np.abs(beta)
    ax.set_ylabel(r"$|\beta|$")
    ax.set_title("Per-feature coefficients (grouped)", fontsize=13, fontweight="bold", pad=14)
    ax.grid(axis="y", alpha=0.18)
    ax.set_xticks([])

    # group shading + separators
    boundaries, centers = _group_boundaries(groups)
    start = 0
    for gid, group in enumerate(groups):
        end = start + len(group)
        # colored group band at the bottom of the panel
        band_color = TAG_COLORS.get(group_tags[gid], "#d1d5db")
        ax.axvspan(start - 0.5, end - 0.5, color=band_color, alpha=0.12, zorder=0)
        start = end
    for boundary in boundaries:
        ax.axvline(boundary - 0.5, color="#cfcfcf", linewidth=1.0, zorder=1)
    for center, label in centers:
        ax.text(center, -0.06, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9.2, fontweight="bold")

    # thin lollipop lines + points
    for xi, yi, ci in zip(x, y, colors):
        ax.plot([xi, xi], [0, yi], color=ci, linewidth=0.8, alpha=0.9, zorder=2)
        ax.scatter([xi], [yi], s=16, color=ci, edgecolor="white", linewidth=0.3, zorder=3)

    handles = [
        plt.Line2D([0], [0], color=TAG_COLORS["strong"], marker="o", linestyle="", markersize=6, label="Signal"),
        plt.Line2D([0], [0], color=TAG_COLORS["noise"], marker="o", linestyle="", markersize=6, label="Noise (looks like signal)"),
        plt.Line2D([0], [0], color=TAG_COLORS["null"], marker="o", linestyle="", markersize=6, label="Other"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=8.6)


def _write_summary(out_dir: Path, scenario_key: str, summary: pd.DataFrame) -> None:
    summary.to_csv(out_dir / f"{scenario_key}_fingerprint_summary.csv", index=False)
    payload = {
        "scenario": scenario_key,
        "groups": summary.to_dict(orient="records"),
    }
    (out_dir / f"{scenario_key}_fingerprint_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _render_scenario(out_dir: Path, scenario_key: str, title: str, cfg_path: Path) -> None:
    cfg = _load_yaml(cfg_path)
    data_cfg = dict(cfg.get("data", {}) or {})
    synthetic_cfg = synthetic_config_from_dict(data_cfg, seed=data_cfg.get("seed"), name=scenario_key, task="regression")
    dataset = generate_synthetic(synthetic_cfg)

    X = np.asarray(dataset.X, dtype=float)
    beta = np.asarray(dataset.beta, dtype=float)
    corr = _feature_corr(X)
    tags = _infer_group_tags(dataset.info or {}, dataset.groups)
    summary = _group_summary(beta, dataset.groups, tags)
    feature_tags = _feature_tags(dataset.info or {}, beta.size)

    fig = plt.figure(figsize=(16.8, 6.8), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.15, 1.65])
    ax_corr = fig.add_subplot(gs[0, 0])
    ax_group = fig.add_subplot(gs[0, 1])

    _plot_corr_heatmap(ax_corr, corr, dataset.groups)
    _plot_feature_lines(ax_group, beta, dataset.groups, feature_tags, tags)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    plt.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.12, wspace=0.22)
    fig.savefig(out_dir / f"{scenario_key}_dataset_fingerprint.png", dpi=230, bbox_inches="tight")
    plt.close(fig)

    _write_summary(out_dir, scenario_key, summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic dataset fingerprint figures for the four benchmark scenarios.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/reports/synthetic_dataset_fingerprints"),
        help="Destination directory for the synthetic dataset fingerprint figures.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for scenario_key, rel_cfg, title in SCENARIOS:
        _render_scenario(out_dir, scenario_key, title, repo_root / rel_cfg)

    print(f"[ok] synthetic dataset fingerprints written to {out_dir}")


if __name__ == "__main__":
    main()
