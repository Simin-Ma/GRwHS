from __future__ import annotations

import argparse
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
                "active_fraction": float(active[idx].mean()),
                "beta_norm": float(np.linalg.norm(beta[idx])),
                "group_tag": group_tag,
            }
        )
    return pd.DataFrame(rows)


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_color("#222222")
    ax.spines["bottom"].set_color("#222222")
    ax.set_facecolor("white")


def _plot_panel(
    ax: plt.Axes,
    rng: np.random.Generator,
    beta: np.ndarray,
    groups: Sequence[Sequence[int]],
    feature_tags: Sequence[str],
    summary: pd.DataFrame,
    title: str,
    ymax: float,
) -> None:
    _style_axis(ax)
    min_display_width = 12.0
    gap = 4.0
    left_pad = 4.5
    spans: List[Tuple[float, float, float]] = []
    cursor = left_pad
    for group in groups:
        size = max(float(len(group)), min_display_width)
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
                alpha=0.09,
                zorder=0,
            )
        )
        ax.vlines([start, end], ymin=-0.02 * ymax, ymax=1.04 * ymax, color="#e5e7eb", linewidth=0.8, zorder=1)

    for (start, end, _), group in zip(spans, groups):
        idx = np.asarray(group, dtype=int)
        x = np.linspace(start + 0.5, end - 0.5, idx.size)
        x = x + rng.uniform(-0.18, 0.18, size=idx.size)
        y = np.abs(beta[idx])
        colors = [TAG_COLORS[feature_tags[int(j)]] for j in idx]
        sizes = np.where(y > 1e-10, 26.0, 12.0)
        alphas = np.where(y > 1e-10, 0.95, 0.38)

        ax.vlines(x, 0.0, y, colors=colors, linewidth=0.9, alpha=0.65, zorder=2)
        for xx, yy, cc, ss, aa in zip(x, y, colors, sizes, alphas):
            ax.scatter(xx, yy, s=float(ss), color=cc, alpha=float(aa), edgecolors="white", linewidths=0.32, zorder=3)

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
        ax.text(
            center,
            strip_y0 + strip_h / 2.0,
            str(row["group_tag"]),
            ha="center",
            va="center",
            fontsize=8.2,
            fontweight="bold",
            color="white",
            clip_on=False,
            zorder=4,
        )

    ax.axhline(0.0, color="#6b7280", linewidth=0.9)
    ax.grid(axis="y", alpha=0.20, linewidth=0.8)
    ax.set_xlim(0.0, spans[-1][1] + 1.2)
    ax.set_ylim(-0.03 * ymax, 1.15 * ymax)
    ax.set_ylabel(r"$|\beta_j|$")
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=10)
    ax.set_xticks([center for _, _, center in spans])
    ax.set_xticklabels(
        [
            f"{row['group']}\n(n={int(row['group_size'])}, act={int(row['active_count'])})"
            for _, row in summary.iterrows()
        ],
        fontsize=8.4,
    )


def _render(out_dir: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    payloads: List[Dict[str, Any]] = []
    ymax = 0.0
    for scenario_key, rel_cfg, title in SCENARIOS:
        cfg = _load_yaml(repo_root / rel_cfg)
        data_cfg = dict(cfg.get("data", {}) or {})
        synthetic_cfg = synthetic_config_from_dict(data_cfg, seed=data_cfg.get("seed"), name=scenario_key, task="regression")
        dataset = generate_synthetic(synthetic_cfg)
        beta = np.asarray(dataset.beta, dtype=float)
        feature_tags = _feature_tags(dataset.info or {}, beta.size)
        summary = _group_summary(beta, dataset.groups, feature_tags)
        ymax = max(ymax, float(np.max(np.abs(beta))))
        payloads.append(
            {
                "scenario_key": scenario_key,
                "title": title,
                "beta": beta,
                "groups": dataset.groups,
                "feature_tags": feature_tags,
                "summary": summary,
            }
        )

    ymax = max(0.6, ymax)
    fig, axes = plt.subplots(len(payloads), 1, figsize=(17.6, 5.6), sharex=False, constrained_layout=False)
    fig.patch.set_facecolor("white")
    rng = np.random.default_rng(20260309)

    for ax, payload in zip(np.atleast_1d(axes), payloads):
        _plot_panel(
            ax=ax,
            rng=rng,
            beta=payload["beta"],
            groups=payload["groups"],
            feature_tags=payload["feature_tags"],
            summary=payload["summary"],
            title=payload["title"],
            ymax=ymax,
        )

    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=7, color=TAG_COLORS["strong"], label="Signal"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=7, color=TAG_COLORS["noise"], label="Noise (looks like signal)"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=7, color=TAG_COLORS["null"], label="Other"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.955),
    )
    fig.suptitle("Synthetic signal contrast by group", fontsize=17, fontweight="bold", y=0.98)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.10, hspace=0.28)
    fig.savefig(out_dir / "sim_g5_group_signal_contrast.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    summary_df = pd.concat(
        [payload["summary"].assign(scenario=payload["scenario_key"]) for payload in payloads],
        ignore_index=True,
    )
    summary_df.to_csv(out_dir / "sim_g5_group_signal_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot grouped signal-contrast panels for synthetic scenarios s1-s3.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/reports/synthetic_signal_contrast"),
        help="Destination directory for the grouped signal contrast plot.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _render(out_dir)
    print(f"[ok] grouped synthetic signal contrast figure written to {out_dir}")


if __name__ == "__main__":
    main()
