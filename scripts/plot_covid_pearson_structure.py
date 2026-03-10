from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GROUP_LABELS = {
    0: "Period",
    1: "Region",
    2: "Age",
    3: "Gender",
    4: "Race/Eth",
    5: "CLI spline",
    6: "Community CLI spline",
}

SHORT_LEVEL_LABELS = {
    "NonHispanicAmericanIndianAlaskaNative": "NH AI/AN",
    "NonHispanicAsian": "NH Asian",
    "NonHispanicBlackAfricanAmerican": "NH Black",
    "NonHispanicMultipleOther": "NH Multi/Other",
    "NonHispanicNativeHawaiianPacificIslander": "NH NH/PI",
    "NonHispanicWhite": "NH White",
    "NotReported": "Not rep.",
}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _feature_short_label(name: str) -> str:
    if name.startswith("period="):
        return name.split("=")[1][2:]  # YY-MM-DD -> MM-DD readable tail
    if name.startswith("region="):
        return name.split("=")[1]
    if name.startswith("age="):
        return name.split("=")[1]
    if name.startswith("gender="):
        return name.split("=")[1]
    if name.startswith("raceethnicity="):
        level = name.split("=", 1)[1]
        return SHORT_LEVEL_LABELS.get(level, level)
    if name.startswith("cli_bs_"):
        return "cli" + name.rsplit("_", 1)[1]
    if name.startswith("hh_cmnty_cli_bs_"):
        return "ccli" + name.rsplit("_", 1)[1]
    return name


def _load_design(processed_dir: Path) -> tuple[np.ndarray, np.ndarray, List[str], Dict[str, int]]:
    runner_dir = processed_dir / "runner_ready"
    X = np.load(runner_dir / "X.npy").astype(float)
    y = np.load(runner_dir / "y.npy").astype(float)
    feature_names = (runner_dir / "feature_names.txt").read_text(encoding="utf-8").splitlines()
    group_map = _load_json(runner_dir / "group_map.json")
    return X, y, feature_names, {str(k): int(v) for k, v in group_map.items()}


def _group_boundaries(feature_names: Sequence[str], group_map: Dict[str, int]) -> tuple[List[int], List[Tuple[float, str]]]:
    gids = [group_map[name] for name in feature_names]
    boundaries: List[int] = []
    centers: List[Tuple[float, str]] = []
    start = 0
    current_gid = gids[0]
    for idx, gid in enumerate(gids[1:], start=1):
        if gid != current_gid:
            boundaries.append(idx)
            centers.append(((start + idx - 1) / 2.0, GROUP_LABELS.get(current_gid, f"G{current_gid+1}")))
            start = idx
            current_gid = gid
    centers.append(((start + len(gids) - 1) / 2.0, GROUP_LABELS.get(current_gid, f"G{current_gid+1}")))
    return boundaries, centers


def _correlation_matrix(X: np.ndarray) -> np.ndarray:
    return np.corrcoef(X, rowvar=False)


def _feature_target_correlations(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    y_centered = y - y.mean()
    y_scale = np.sqrt(np.sum(y_centered**2))
    rows: List[Dict[str, Any]] = []
    for j in range(X.shape[1]):
        x = X[:, j]
        x_centered = x - x.mean()
        denom = np.sqrt(np.sum(x_centered**2)) * y_scale
        corr = 0.0 if denom <= 0 else float(np.sum(x_centered * y_centered) / denom)
        rows.append({"feature_index": j, "pearson_r": corr})
    return pd.DataFrame(rows)


def _style_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#222222")
    ax.set_facecolor("white")


def _plot_heatmap(
    ax: plt.Axes,
    corr: np.ndarray,
    feature_names: Sequence[str],
    group_map: Dict[str, int],
) -> None:
    _style_axis(ax)
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0, interpolation="nearest")
    short_labels = [_feature_short_label(name) for name in feature_names]
    ticks = np.arange(len(feature_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(short_labels, rotation=90, fontsize=6.2)
    ax.set_yticklabels(short_labels, fontsize=6.2)
    ax.set_title("Feature Pearson correlation heatmap", fontsize=13, fontweight="bold", pad=18)

    boundaries, centers = _group_boundaries(feature_names, group_map)
    for boundary in boundaries:
        ax.axhline(boundary - 0.5, color="white", linewidth=1.15)
        ax.axvline(boundary - 0.5, color="white", linewidth=1.15)
    for center, label in centers:
        ax.text(center, -6.8, label, ha="center", va="bottom", fontsize=8.2, fontweight="bold", clip_on=False)

    cbar = plt.colorbar(im, ax=ax, fraction=0.024, pad=0.01)
    cbar.ax.set_ylabel("Pearson r", rotation=90, fontsize=10)
    cbar.ax.tick_params(labelsize=8.5)


def _group_correlation_matrix(
    corr: np.ndarray,
    feature_names: Sequence[str],
    group_map: Dict[str, int],
) -> tuple[np.ndarray, List[str]]:
    unique_group_ids = sorted({int(group_map[name]) for name in feature_names})
    group_labels: List[str] = []
    block_indices: List[List[int]] = []
    for gid in unique_group_ids:
        idx = [i for i, name in enumerate(feature_names) if int(group_map[name]) == gid]
        block_indices.append(idx)
        group_labels.append(GROUP_LABELS.get(gid, f"G{gid+1}"))

    mat = np.zeros((len(block_indices), len(block_indices)), dtype=float)
    for i, idx_i in enumerate(block_indices):
        for j, idx_j in enumerate(block_indices):
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
    return mat, group_labels


def _plot_group_heatmap(ax: plt.Axes, mat: np.ndarray, labels: Sequence[str]) -> None:
    _style_axis(ax)
    im = ax.imshow(mat, cmap="coolwarm", vmin=-1.0, vmax=1.0, interpolation="nearest")
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_title("Block-averaged Pearson heatmap", fontsize=13, fontweight="bold", pad=14)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(
                j,
                i,
                f"{mat[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if abs(mat[i, j]) > 0.48 else "#222222",
            )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Pearson r", rotation=90, fontsize=10)
    cbar.ax.tick_params(labelsize=8.5)


def _plot_target_bar(
    ax: plt.Axes,
    target_corr: pd.DataFrame,
    feature_names: Sequence[str],
    group_map: Dict[str, int],
    top_k: int,
) -> None:
    _style_axis(ax)
    frame = target_corr.copy()
    frame["feature"] = [feature_names[idx] for idx in frame["feature_index"]]
    frame["group_id"] = frame["feature"].map(group_map)
    frame["group_label"] = frame["group_id"].map(GROUP_LABELS)
    frame["short_label"] = frame["feature"].map(_feature_short_label)
    frame["abs_r"] = frame["pearson_r"].abs()
    frame = frame.sort_values("abs_r", ascending=False).head(top_k).iloc[::-1].copy()

    group_palette = {
        0: "#153b50",
        1: "#4c78a8",
        2: "#0f6b50",
        3: "#2a9d8f",
        4: "#b42318",
        5: "#e9c46a",
        6: "#f4a261",
    }
    colors = [group_palette.get(int(g), "#777777") for g in frame["group_id"]]
    ax.barh(frame["short_label"], frame["pearson_r"], color=colors, alpha=0.9)
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Pearson r with trust_experts")
    ax.set_title(f"Top {top_k} feature-target correlations", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.22)

    handles = []
    seen = set()
    for gid in frame["group_id"]:
        if gid in seen:
            continue
        seen.add(gid)
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=group_palette.get(int(gid), "#777777"),
                marker="s",
                linestyle="",
                markersize=8,
                label=GROUP_LABELS.get(int(gid), f"G{gid+1}"),
            )
        )
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=8.8)


def _write_summary(
    out_dir: Path,
    feature_target: pd.DataFrame,
    feature_names: Sequence[str],
    group_map: Dict[str, int],
) -> None:
    frame = feature_target.copy()
    frame["feature"] = [feature_names[idx] for idx in frame["feature_index"]]
    frame["group_id"] = frame["feature"].map(group_map)
    frame["group_label"] = frame["group_id"].map(GROUP_LABELS)
    frame["abs_r"] = frame["pearson_r"].abs()
    frame.sort_values("abs_r", ascending=False, inplace=True)
    frame.to_csv(out_dir / "covid_feature_target_pearson.csv", index=False)

    summary = {
        "top_positive": frame.sort_values("pearson_r", ascending=False).head(12).to_dict(orient="records"),
        "top_negative": frame.sort_values("pearson_r", ascending=True).head(12).to_dict(orient="records"),
        "group_abs_mean": (
            frame.groupby("group_label", as_index=False)["abs_r"].mean().sort_values("abs_r", ascending=False).to_dict(orient="records")
        ),
    }
    (out_dir / "covid_pearson_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def make_plot(processed_dir: Path, out_dir: Path, *, top_k: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    X, y, feature_names, group_map = _load_design(processed_dir)
    corr = _correlation_matrix(X)
    feature_target = _feature_target_correlations(X, y)
    group_corr, group_labels = _group_correlation_matrix(corr, feature_names, group_map)

    fig = plt.figure(figsize=(22, 17), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.8, 1.6])
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])

    _plot_heatmap(ax_heat, corr, feature_names, group_map)
    _plot_target_bar(ax_bar, feature_target, feature_names, group_map, top_k=top_k)

    fig.suptitle(
        "COVID-19 Trust in Experts: Pearson correlation structure of encoded features",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.02,
        0.02,
        "Heatmap uses the 101 runner-ready design columns. White separators mark the seven modeling groups. "
        "The right panel shows the strongest linear associations with the outcome.",
        fontsize=10.5,
        color="#222222",
    )
    plt.subplots_adjust(left=0.05, right=0.98, top=0.94, bottom=0.06, wspace=0.12)
    fig.savefig(out_dir / "covid_pearson_structure.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(18, 18))
    _plot_heatmap(ax2, corr, feature_names, group_map)
    fig2.suptitle(
        "COVID-19 Trust in Experts: full Pearson heatmap of encoded features",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )
    fig2.text(
        0.02,
        0.02,
        "This is the full NHANES-style Pearson heatmap using the 101 runner-ready design columns.",
        fontsize=10.5,
        color="#222222",
    )
    plt.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.06)
    fig2.savefig(out_dir / "covid_pearson_heatmap_full.png", dpi=240, bbox_inches="tight")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(8.6, 7.6))
    _plot_group_heatmap(ax3, group_corr, group_labels)
    fig3.suptitle(
        "COVID-19 Trust in Experts: group-level Pearson heatmap",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.subplots_adjust(left=0.14, right=0.95, top=0.90, bottom=0.12)
    fig3.savefig(out_dir / "covid_group_pearson_heatmap.png", dpi=240, bbox_inches="tight")
    plt.close(fig3)

    _write_summary(out_dir, feature_target, feature_names, group_map)
    print(f"[ok] plot written to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot NHANES-style Pearson correlation figures for the COVID trust_experts dataset.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/real/covid19_trust_experts/processed"),
        help="Processed COVID dataset directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/reports/covid19_trust_experts_pearson"),
        help="Destination directory for the Pearson figure and tables.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=24,
        help="Number of strongest feature-target correlations to show in the right panel.",
    )
    args = parser.parse_args()
    make_plot(args.processed_dir, args.out_dir, top_k=max(8, int(args.top_k)))


if __name__ == "__main__":
    main()
