"""Generate GRwHS vs RHS group-level comparison plots and summaries."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import sys  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.generators import (  # noqa: E402
    generate_synthetic,
    synthetic_config_from_dict,
)

TAG_COLORS = {
    "strong": "#d62728",
    "medium": "#ff7f0e",
    "weak": "#9467bd",
    "null": "#7f7f7f",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot group-level estimation error, scales, and calibration for GRwHS vs RHS."
    )
    parser.add_argument("--grwhs-dir", required=True, type=Path, help="Path to GRwHS sweep variation directory.")
    parser.add_argument("--rhs-dir", required=True, type=Path, help="Path to RHS sweep variation directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write figures (default: <grwhs-dir>/figures_group_comparison).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="sim_s3 (SNR=3.0)",
        help="Title prefix used for the plots.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-10,
        help="Numerical epsilon used for log transforms.",
    )
    parser.add_argument(
        "--sweep-csv",
        type=Path,
        default=None,
        help="Optional sweep comparison CSV for MeanEffectiveNonzeros vs SNR (enables triptych panel).",
    )
    parser.add_argument(
        "--stacked-bar",
        action="store_true",
        help="Emit stacked bar plot showing tag-level distributions of effective nonzeros.",
    )
    return parser.parse_args()


def _load_resolved_config(run_dir: Path) -> Mapping[str, object]:
    cfg_path = run_dir / "resolved_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"resolved_config.yaml not found under {run_dir}")
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def _derive_context(grwhs_dir: Path, resolved_cfg: Mapping[str, object]) -> tuple[str, str, float]:
    """Return (scenario_slug, snr_token, snr_value) for unique filenames."""
    parts = list(grwhs_dir.parts)
    scenario = "scenario"
    if "sweeps" in parts:
        try:
            idx = parts.index("sweeps")
            scenario = parts[idx + 1]
        except Exception:
            pass
    snr_val = float(resolved_cfg.get("data", {}).get("snr", 0.0))
    token = ("%0.1f" % snr_val).replace(".", "p")
    return scenario, token, snr_val


def _load_groups_and_meta(run_dir: Path) -> tuple[List[List[int]], MutableMapping[str, object]]:
    for repeat_dir in sorted(run_dir.glob("repeat_*")):
        meta_path = repeat_dir / "dataset_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            groups = meta.get("groups")
            if not groups:
                raise RuntimeError(f"No 'groups' entry found in {meta_path}")
            return [list(map(int, grp)) for grp in groups], meta
    raise RuntimeError(f"No dataset_meta.json found under {run_dir}")


def _infer_group_tags(meta: Mapping[str, object], num_groups: int) -> List[str]:
    tags = ["null"] * num_groups
    info = (meta.get("metadata") or {}).get("info") or {}
    blueprint = info.get("signal_blueprint") or {}
    assignments = blueprint.get("assignments") or []
    for assignment in assignments:
        g_idx = assignment.get("group")
        if g_idx is None or not (0 <= int(g_idx) < num_groups):
            continue
        entry_label = str(
            assignment.get("entry") or assignment.get("label") or assignment.get("component") or ""
        ).lower()
        tag = "null"
        for candidate in ("strong", "medium", "weak"):
            if candidate in entry_label:
                tag = candidate
                break
        tags[int(g_idx)] = tag
    return tags


def _regenerate_beta(resolved_cfg: Mapping[str, object]) -> np.ndarray:
    data_cfg = resolved_cfg.get("data")
    if data_cfg is None:
        raise ValueError("Resolved config missing 'data' section.")
    seed = data_cfg.get("seed")
    seeds_block = resolved_cfg.get("seeds") or {}
    seed = seed or seeds_block.get("data_generation")
    synthetic_cfg = synthetic_config_from_dict(
        data_cfg,
        seed=seed,
        name=resolved_cfg.get("name"),
        task=resolved_cfg.get("task"),
    )
    dataset = generate_synthetic(synthetic_cfg)
    return np.asarray(dataset.beta, dtype=float)


def _iter_fold_dirs(run_dir: Path) -> Iterable[Path]:
    for repeat_dir in sorted(run_dir.glob("repeat_*")):
        for fold_dir in sorted(repeat_dir.glob("fold_*")):
            if (fold_dir / "posterior_summary.parquet").exists():
                yield fold_dir


def _load_beta_mean(fold_dir: Path) -> np.ndarray:
    summary_path = fold_dir / "posterior_summary.parquet"
    df = pd.read_parquet(summary_path)
    beta_df = df[df["parameter"] == "beta"].sort_values("index")
    if beta_df.empty:
        raise RuntimeError(f"No beta entries found in {summary_path}")
    return beta_df["mean"].to_numpy(dtype=float)


def _load_x_scale(fold_dir: Path) -> np.ndarray:
    arrays = np.load(fold_dir / "fold_arrays.npz")
    if "x_scale" not in arrays:
        raise RuntimeError(f"x_scale missing in {fold_dir/'fold_arrays.npz'}")
    return np.asarray(arrays["x_scale"], dtype=float)


def _collect_group_metrics(
    run_dir: Path,
    beta_true: np.ndarray,
    groups: Sequence[Sequence[int]],
) -> Dict[str, Dict[int, List[float]]]:
    group_metrics: Dict[str, Dict[int, List[float]]] = {
        "mse": {gid: [] for gid in range(len(groups))},
        "est_norm": {gid: [] for gid in range(len(groups))},
        "true_norm": {gid: [] for gid in range(len(groups))},
    }
    for fold_dir in _iter_fold_dirs(run_dir):
        beta_mean = _load_beta_mean(fold_dir)
        x_scale = _load_x_scale(fold_dir)
        if beta_mean.shape[0] != beta_true.shape[0]:
            raise ValueError(f"beta length mismatch in {fold_dir}")
        if x_scale.shape[0] != beta_true.shape[0]:
            raise ValueError(f"x_scale length mismatch in {fold_dir}")
        beta_true_std = beta_true * x_scale
        sq_error = (beta_mean - beta_true_std) ** 2
        for gid, idxs in enumerate(groups):
            idx_arr = np.asarray(idxs, dtype=int)
            if idx_arr.size == 0:
                continue
            group_metrics["mse"][gid].append(float(np.mean(sq_error[idx_arr])))
            group_metrics["est_norm"][gid].append(float(np.linalg.norm(beta_mean[idx_arr])))
            group_metrics["true_norm"][gid].append(float(np.linalg.norm(beta_true_std[idx_arr])))
    return group_metrics


def _collect_active_counts(groups: Sequence[Sequence[int]], meta: Mapping[str, object]) -> Dict[int, int]:
    info = (meta.get("metadata") or {}).get("info") or {}
    active_idx = [int(i) for i in info.get("signal_blueprint", {}).get("active_idx", info.get("active_idx", []))]
    counts = {gid: 0 for gid in range(len(groups))}
    for gid, idxs in enumerate(groups):
        idx_set = set(int(i) for i in idxs)
        counts[gid] = sum(1 for idx in active_idx if idx in idx_set)
    return counts

def _summarize_group_stats(group_values: Mapping[int, Sequence[float]]) -> Dict[int, Dict[str, float]]:
    stats: Dict[int, Dict[str, float]] = {}
    for gid, values in group_values.items():
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            stats[gid] = {"mean": float("nan"), "stderr": float("nan")}
        elif arr.size == 1:
            stats[gid] = {"mean": float(arr[0]), "stderr": 0.0}
        else:
            stats[gid] = {
                "mean": float(arr.mean()),
                "stderr": float(arr.std(ddof=1) / math.sqrt(arr.size)),
            }
    return stats


def _summarize_group_quantiles(
    group_values: Mapping[int, Sequence[float]],
    low: float = 5.0,
    high: float = 95.0,
) -> Dict[int, Dict[str, float]]:
    stats: Dict[int, Dict[str, float]] = {}
    for gid, values in group_values.items():
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            stats[gid] = {"mean": float("nan"), "p_low": float("nan"), "p_high": float("nan")}
        else:
            stats[gid] = {
                "mean": float(arr.mean()),
                "p_low": float(np.percentile(arr, low)),
                "p_high": float(np.percentile(arr, high)),
            }
    return stats


def _collect_log_phi(run_dir: Path, eps: float) -> np.ndarray:
    blocks = []
    for fold_dir in _iter_fold_dirs(run_dir):
        samples = np.load(fold_dir / "posterior_samples.npz")
        if "phi" not in samples:
            raise RuntimeError(f"phi samples missing in {fold_dir/'posterior_samples.npz'}")
        phi = np.asarray(samples["phi"], dtype=float)
        phi = phi.reshape(phi.shape[0], -1)
        blocks.append(np.log(np.maximum(phi, eps)))
    if not blocks:
        raise RuntimeError("No phi samples collected.")
    return np.concatenate(blocks, axis=0)


def _collect_log_tau(run_dir: Path, eps: float) -> np.ndarray:
    values = []
    for fold_dir in _iter_fold_dirs(run_dir):
        samples = np.load(fold_dir / "posterior_samples.npz")
        if "tau" not in samples:
            raise RuntimeError(f"tau samples missing in {fold_dir/'posterior_samples.npz'}")
        tau = np.asarray(samples["tau"], dtype=float).reshape(-1)
        values.append(np.log(np.maximum(tau, eps)))
    if not values:
        raise RuntimeError("No tau samples collected.")
    return np.concatenate(values, axis=0)


def _format_tick_labels(group_tags: Sequence[str]) -> List[str]:
    labels = []
    for gid, tag in enumerate(group_tags):
        labels.append(f"g{gid}\n({tag})")
    return labels


def _tick_colors(group_tags: Sequence[str]) -> List[str]:
    return [TAG_COLORS.get(tag, TAG_COLORS["null"]) for tag in group_tags]


def _format_label(label: str) -> str:
    mapping = {"grwhs": "GRwHS", "rhs": "RHS"}
    return mapping.get(str(label).lower(), label)


def _plot_group_mse(
    *,
    save_path: Path | None,
    groups: Sequence[Sequence[int]],
    stats_grwhs: Mapping[int, Mapping[str, float]],
    stats_rhs: Mapping[int, Mapping[str, float]],
    title: str,
    xtick_labels: Sequence[str],
    tick_colors: Sequence[str],
    ax: plt.Axes | None = None,
    show_legend: bool = True,
) -> plt.Axes:
    own_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        own_fig = True
    else:
        fig = ax.figure

    group_ids = np.arange(len(groups))
    width = 0.35
    mse_grwhs = np.array([stats_grwhs[g]["mean"] for g in group_ids])
    err_grwhs = np.array([stats_grwhs[g]["stderr"] for g in group_ids])
    mse_rhs = np.array([stats_rhs[g]["mean"] for g in group_ids])
    err_rhs = np.array([stats_rhs[g]["stderr"] for g in group_ids])

    ax.bar(
        group_ids - width / 2,
        mse_grwhs,
        width,
        yerr=err_grwhs,
        capsize=4,
        label="GRwHS",
        color="#1f77b4",
    )
    ax.bar(
        group_ids + width / 2,
        mse_rhs,
        width,
        yerr=err_rhs,
        capsize=4,
        label="RHS",
        color="#7f7f7f",
    )

    ax.set_xticks(group_ids, xtick_labels)
    for tick, color in zip(ax.get_xticklabels(), tick_colors):
        tick.set_color(color)
    ax.set_ylabel("Group mean squared error (standardized β)")
    ax.set_title(title)
    if show_legend:
        ax.legend()
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6)

    if own_fig:
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return ax


def _plot_group_scales(
    *,
    save_path: Path | None,
    log_phi: np.ndarray,
    log_tau: np.ndarray,
    num_groups: int,
    title: str,
    xtick_labels: Sequence[str],
    tick_colors: Sequence[str],
    ax: plt.Axes | None = None,
    show_legend: bool = True,
) -> plt.Axes:
    own_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        own_fig = True
    else:
        fig = ax.figure

    phi_mean = np.mean(log_phi, axis=0)
    phi_lo, phi_hi = np.percentile(log_phi, [5, 95], axis=0)
    rhs_mean = float(np.mean(log_tau))
    rhs_lo, rhs_hi = np.percentile(log_tau, [5, 95])

    x = np.arange(num_groups)
    yerr_lower = phi_mean - phi_lo
    yerr_upper = phi_hi - phi_mean
    ax.errorbar(
        x,
        phi_mean,
        yerr=[yerr_lower, yerr_upper],
        fmt="o-",
        color="#1f77b4",
        ecolor="#1f77b4",
        capsize=4,
        label="GRwHS $E[\\log \\phi_g]$",
    )

    span_x = np.linspace(-0.5, num_groups - 0.5, 200)
    ax.hlines(rhs_mean, span_x.min(), span_x.max(), colors="#7f7f7f", linestyles="--", label="RHS $E[\\log \\tau]$")
    ax.fill_between(span_x, rhs_lo, rhs_hi, color="#7f7f7f", alpha=0.2, linewidth=0)

    ax.set_xticks(x, xtick_labels)
    for tick, color in zip(ax.get_xticklabels(), tick_colors):
        tick.set_color(color)
    ax.set_ylabel("log-scale (natural log)")
    ax.set_title(title)
    if show_legend:
        ax.legend()
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6)

    if own_fig:
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return ax


def _plot_combined_panel(
    out_path: Path,
    *,
    groups: Sequence[Sequence[int]],
    stats_grwhs: Mapping[int, Mapping[str, float]],
    stats_rhs: Mapping[int, Mapping[str, float]],
    log_phi: np.ndarray,
    log_tau: np.ndarray,
    title: str,
    xtick_labels: Sequence[str],
    tick_colors: Sequence[str],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    _plot_group_mse(
        save_path=None,
        groups=groups,
        stats_grwhs=stats_grwhs,
        stats_rhs=stats_rhs,
        title=f"{title}: per-group error",
        xtick_labels=xtick_labels,
        tick_colors=tick_colors,
        ax=axes[0],
        show_legend=True,
    )
    _plot_group_scales(
        save_path=None,
        log_phi=log_phi,
        log_tau=log_tau,
        num_groups=len(groups),
        title=f"{title}: shrinkage scales",
        xtick_labels=xtick_labels,
        tick_colors=tick_colors,
        ax=axes[1],
        show_legend=True,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_group_calibration(
    out_path: Path,
    *,
    true_norm_stats: Mapping[int, Mapping[str, float]],
    est_norm_grwhs: Mapping[int, Mapping[str, float]],
    est_norm_rhs: Mapping[int, Mapping[str, float]],
    group_tags: Sequence[str],
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    group_ids = np.arange(len(group_tags))
    x_vals = np.array([true_norm_stats[g]["mean"] for g in group_ids])
    max_val = float(np.nanmax(x_vals)) if np.isfinite(x_vals).any() else 1.0

    ax.plot([0, max_val * 1.1], [0, max_val * 1.1], color="#444444", linestyle="--", linewidth=1.0, label="Ideal")

    for gid in group_ids:
        color = TAG_COLORS.get(group_tags[gid], TAG_COLORS["null"])
        x = true_norm_stats[gid]["mean"]
        if not np.isfinite(x):
            continue
        est_g = est_norm_grwhs[gid]
        est_r = est_norm_rhs[gid]
        ax.errorbar(
            x,
            est_g["mean"],
            yerr=[[est_g["mean"] - est_g["p_low"]], [est_g["p_high"] - est_g["mean"]]],
            fmt="o",
            color=color,
            ecolor=color,
            markersize=7,
            markeredgecolor="black",
            alpha=0.9,
        )
        ax.errorbar(
            x,
            est_r["mean"],
            yerr=[[est_r["mean"] - est_r["p_low"]], [est_r["p_high"] - est_r["mean"]]],
            fmt="^",
            color=color,
            ecolor=color,
            markersize=6,
            markeredgecolor="black",
            alpha=0.85,
        )

    ax.set_xlabel("True group ‖β‖ (standardized)")
    ax.set_ylabel("Estimated group ‖β̂‖")
    ax.set_title(f"{title}: group-level calibration")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

    model_handles = [
        Line2D([0], [0], marker="o", color="white", markerfacecolor="white", markeredgecolor="black", label="GRwHS"),
        Line2D([0], [0], marker="^", color="white", markerfacecolor="white", markeredgecolor="black", label="RHS"),
    ]
    legend1 = ax.legend(handles=model_handles, loc="upper left", title="Model")
    ax.add_artist(legend1)
    tag_handles = [
        Patch(facecolor=TAG_COLORS[tag], edgecolor="black", label=tag.title()) for tag in sorted(set(group_tags))
    ]
    ax.legend(handles=tag_handles, loc="lower right", title="Group tag")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _aggregate_tag_distribution(
    *,
    group_tags: Sequence[str],
    group_values: Mapping[int, float],
) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    for gid, tag in enumerate(group_tags):
        agg[tag] = agg.get(tag, 0.0) + float(group_values.get(gid, 0.0))
    total = sum(agg.values())
    if total > 0:
        agg = {tag: val / total for tag, val in agg.items()}
    return agg


def _plot_tag_stacked_bar(
    out_path: Path,
    *,
    group_tags: Sequence[str],
    true_counts: Mapping[int, int],
    est_grwhs: Mapping[int, Mapping[str, float]],
    est_rhs: Mapping[int, Mapping[str, float]],
) -> None:
    order = ["strong", "medium", "weak", "null"]
    truth_values = _aggregate_tag_distribution(
        group_tags=group_tags,
        group_values={gid: count for gid, count in true_counts.items()},
    )
    grwhs_values = _aggregate_tag_distribution(
        group_tags=group_tags,
        group_values={gid: est_grwhs[gid]["mean"] for gid in range(len(group_tags))},
    )
    rhs_values = _aggregate_tag_distribution(
        group_tags=group_tags,
        group_values={gid: est_rhs[gid]["mean"] for gid in range(len(group_tags))},
    )

    labels = ["Truth", "GRwHS", "RHS"]
    data = [truth_values, grwhs_values, rhs_values]

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    bottoms = [0.0, 0.0, 0.0]
    for tag in order:
        heights = [vals.get(tag, 0.0) for vals in data]
        ax.bar(labels, heights, bottom=bottoms, color=TAG_COLORS.get(tag, "#bababa"), edgecolor="black", label=tag.title())
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    ax.set_ylabel("Share of effective nonzeros")
    ax.set_ylim(0, 1.0)
    ax.set_title("Distribution of signal mass by group type")
    ax.legend(loc="upper right")
    for i, total in enumerate(bottoms):
        ax.text(i, 1.02, labels[i], ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _load_sweep_mean_effective(sweep_csv: Path, labels: Iterable[str]) -> pd.DataFrame:
    import pandas as pd  # local import to avoid global dependency if unused

    from math import isnan

    df = pd.read_csv(sweep_csv)
    requested = {label.lower() for label in labels}

    def parse_var(var: str) -> tuple[float | None, str | None]:
        parts = var.split("_", 1)
        if len(parts) != 2:
            return None, None
        head = parts[0]
        if not head.startswith("snr"):
            return None, None
        snr_token = head[3:].replace("p", ".")
        try:
            snr_val = float(snr_token)
        except ValueError:
            return None, None
        return snr_val, parts[1]

    rows = []
    for _, row in df.iterrows():
        variation = str(row["variation"])
        snr, label = parse_var(variation)
        if snr is None or label is None:
            continue
        if label.lower() not in requested:
            continue
        value = row["MeanEffectiveNonzeros"]
        if value is None or (hasattr(value, "__float__") and isnan(float(value))):
            continue
        rows.append({"snr": snr, "label": label.lower(), "mean_effective": float(value)})
    if not rows:
        raise RuntimeError(f"No MeanEffectiveNonzeros entries found in {sweep_csv}")
    return pd.DataFrame(rows)


def _plot_triptych(
    out_path: Path,
    *,
    groups: Sequence[Sequence[int]],
    stats_grwhs: Mapping[int, Mapping[str, float]],
    stats_rhs: Mapping[int, Mapping[str, float]],
    log_phi: np.ndarray,
    log_tau: np.ndarray,
    xtick_labels: Sequence[str],
    tick_colors: Sequence[str],
    sweep_df: pd.DataFrame,
    current_snr: float,
) -> None:
    from matplotlib import gridspec

    fig = plt.figure(figsize=(11, 7))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.7])

    ax_mse = fig.add_subplot(gs[0, 0])
    _plot_group_mse(
        save_path=None,
        groups=groups,
        stats_grwhs=stats_grwhs,
        stats_rhs=stats_rhs,
        title="Per-group error",
        xtick_labels=xtick_labels,
        tick_colors=tick_colors,
        ax=ax_mse,
        show_legend=True,
    )

    ax_scale = fig.add_subplot(gs[0, 1])
    _plot_group_scales(
        save_path=None,
        log_phi=log_phi,
        log_tau=log_tau,
        num_groups=len(groups),
        title="Group log-scales",
        xtick_labels=xtick_labels,
        tick_colors=tick_colors,
        ax=ax_scale,
        show_legend=True,
    )

    ax_mean = fig.add_subplot(gs[1, :])
    for label, group in sweep_df.groupby("label"):
        style = {"grwhs": {"color": "#1f77b4", "marker": "o"}, "rhs": {"color": "#7f7f7f", "marker": "^"}}.get(label, {})
        group = group.sort_values("snr")
        ax_mean.plot(
            group["snr"],
            group["mean_effective"],
            label=_format_label(label),
            color=style.get("color"),
            marker=style.get("marker", "o"),
            linewidth=2.0,
        )
        if current_snr in set(group["snr"]):
            subset = group[group["snr"] == current_snr]
            ax_mean.scatter(
                subset["snr"],
                subset["mean_effective"],
                color=style.get("color"),
                s=80,
                edgecolor="black",
                zorder=5,
            )
    ax_mean.set_xlabel("SNR")
    ax_mean.set_ylabel("MeanEffectiveNonzeros")
    ax_mean.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
    ax_mean.legend(loc="best")
    ax_mean.set_title("Global effective nonzeros vs SNR (highlight = current run)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    grwhs_dir = args.grwhs_dir.expanduser().resolve()
    rhs_dir = args.rhs_dir.expanduser().resolve()
    if not grwhs_dir.exists():
        raise FileNotFoundError(f"GRwHS directory not found: {grwhs_dir}")
    if not rhs_dir.exists():
        raise FileNotFoundError(f"RHS directory not found: {rhs_dir}")

    output_dir = args.output_dir or (grwhs_dir / "figures_group_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    groups, meta = _load_groups_and_meta(grwhs_dir)
    groups_rhs, meta_rhs = _load_groups_and_meta(rhs_dir)
    if len(groups) != len(groups_rhs) or any(g1 != g2 for g1, g2 in zip(groups, groups_rhs)):
        raise RuntimeError("GRwHS and RHS runs use mismatched group structures.")

    group_tags = _infer_group_tags(meta, len(groups))
    xtick_labels = _format_tick_labels(group_tags)
    tick_colors = _tick_colors(group_tags)

    resolved_cfg = _load_resolved_config(grwhs_dir)
    beta_true = _regenerate_beta(resolved_cfg)
    if beta_true.shape[0] != sum(len(g) for g in groups):
        raise RuntimeError("β dimension does not match group specification.")

    grwhs_metrics = _collect_group_metrics(grwhs_dir, beta_true, groups)
    rhs_metrics = _collect_group_metrics(rhs_dir, beta_true, groups)
    stats_grwhs = _summarize_group_stats(grwhs_metrics["mse"])
    stats_rhs = _summarize_group_stats(rhs_metrics["mse"])
    true_norm_stats = _summarize_group_quantiles(grwhs_metrics["true_norm"], low=5, high=95)
    est_norm_grwhs = _summarize_group_quantiles(grwhs_metrics["est_norm"])
    est_norm_rhs = _summarize_group_quantiles(rhs_metrics["est_norm"])

    log_phi = _collect_log_phi(grwhs_dir, args.eps)
    log_tau = _collect_log_tau(rhs_dir, args.eps)

    scenario_slug, snr_token, _snr_val = _derive_context(grwhs_dir, resolved_cfg)
    prefix = f"{scenario_slug}_snr{snr_token}"
    mse_path = output_dir / f"{prefix}_group_mse.png"
    scales_path = output_dir / f"{prefix}_group_scales.png"
    combined_path = output_dir / f"{prefix}_group_combined.png"
    calibration_path = output_dir / f"{prefix}_group_calibration.png"
    stacked_path = output_dir / f"{prefix}_group_stacked.png"
    triptych_path = output_dir / f"{prefix}_triptych.png"

    _plot_group_mse(
        save_path=mse_path,
        groups=groups,
        stats_grwhs=stats_grwhs,
        stats_rhs=stats_rhs,
        title=f"{args.title}: per-group estimation error",
        xtick_labels=xtick_labels,
        tick_colors=tick_colors,
        show_legend=True,
    )
    _plot_group_scales(
        save_path=scales_path,
        log_phi=log_phi,
        log_tau=log_tau,
        num_groups=len(groups),
        title=f"{args.title}: group-level shrinkage scales",
        xtick_labels=xtick_labels,
        tick_colors=tick_colors,
        show_legend=True,
    )
    _plot_combined_panel(
        combined_path,
        groups=groups,
        stats_grwhs=stats_grwhs,
        stats_rhs=stats_rhs,
        log_phi=log_phi,
        log_tau=log_tau,
        title=args.title,
        xtick_labels=xtick_labels,
        tick_colors=tick_colors,
    )
    _plot_group_calibration(
        calibration_path,
        true_norm_stats=true_norm_stats,
        est_norm_grwhs=est_norm_grwhs,
        est_norm_rhs=est_norm_rhs,
        group_tags=group_tags,
        title=args.title,
    )

    true_counts = _collect_active_counts(groups, meta)
    if args.stacked_bar:
        _plot_tag_stacked_bar(
            stacked_path,
            group_tags=group_tags,
            true_counts=true_counts,
            est_grwhs=est_norm_grwhs,
            est_rhs=est_norm_rhs,
        )

    triptych_generated = False
    if args.sweep_csv and args.sweep_csv.exists():
        sweep_df = _load_sweep_mean_effective(args.sweep_csv, labels=["grwhs", "rhs"])
        current_snr = float(resolved_cfg.get("data", {}).get("snr", meta.get("metadata", {}).get("info", {}).get("snr", 0.0)))
        _plot_triptych(
            triptych_path,
            groups=groups,
            stats_grwhs=stats_grwhs,
            stats_rhs=stats_rhs,
            log_phi=log_phi,
            log_tau=log_tau,
            xtick_labels=xtick_labels,
            tick_colors=tick_colors,
            sweep_df=sweep_df,
            current_snr=current_snr,
        )
        triptych_generated = True

    summary = {
        "groups": groups,
        "group_tags": group_tags,
        "beta_dimension": int(beta_true.shape[0]),
        "mse": {
            "grwhs": {gid: stats_grwhs[gid] for gid in range(len(groups))},
            "rhs": {gid: stats_rhs[gid] for gid in range(len(groups))},
        },
        "log_phi": {
            "mean": log_phi.mean(axis=0).tolist(),
            "p05": np.percentile(log_phi, 5, axis=0).tolist(),
            "p95": np.percentile(log_phi, 95, axis=0).tolist(),
        },
        "log_tau": {
            "mean": float(log_tau.mean()),
            "p05": float(np.percentile(log_tau, 5)),
            "p95": float(np.percentile(log_tau, 95)),
        },
        "group_norms": {
            "true": {gid: true_norm_stats[gid] for gid in range(len(groups))},
            "grwhs": {gid: est_norm_grwhs[gid] for gid in range(len(groups))},
            "rhs": {gid: est_norm_rhs[gid] for gid in range(len(groups))},
        },
        "tag_distribution": {
            "truth_counts": true_counts,
            "truth_share": _aggregate_tag_distribution(group_tags=group_tags, group_values=true_counts),
            "grwhs_share": _aggregate_tag_distribution(
                group_tags=group_tags,
                group_values={gid: est_norm_grwhs[gid]["mean"] for gid in range(len(groups))},
            ),
            "rhs_share": _aggregate_tag_distribution(
                group_tags=group_tags,
                group_values={gid: est_norm_rhs[gid]["mean"] for gid in range(len(groups))},
            ),
        },
        "figures": {
            "group_mse": str(mse_path),
            "group_scales": str(scales_path),
            "group_combined": str(combined_path),
            "group_calibration": str(calibration_path),
        },
    }
    if args.stacked_bar:
        summary["figures"]["group_stacked"] = str(stacked_path)
    if triptych_generated:
        summary["figures"]["triptych"] = str(triptych_path)

    summary_path = output_dir / "group_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] Figures written to {output_dir}")
    for key, path in summary["figures"].items():
        print(f" - {key}: {path}")


if __name__ == "__main__":
    main()
