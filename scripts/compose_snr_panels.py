"""Compose 2x2 SNR panels (MSE and scales) for a given scenario.

Reads group_comparison_summary.json from the per-SNR figure folders and
plots 2x2 mosaics across SNR ∈ {0.1, 0.5, 1.0, 3.0}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


SNR_TOKENS = ["0p1", "0p5", "1p0", "3p0"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compose SNR panels for a scenario")
    p.add_argument("--scenario", required=True, choices=["sim_s1", "sim_s2", "sim_s3"], help="Scenario name")
    p.add_argument(
        "--fig-root",
        type=Path,
        default=Path("outputs/figures"),
        help="Root where per-SNR figures live (default: outputs/figures)",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory for composed panels (default: <fig-root>/<scenario>_panels)",
    )
    return p.parse_args()


def _load_summary(path: Path) -> Mapping[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_scales_panel(ax: plt.Axes, summary: Mapping[str, object], title: str, xtick_labels: Sequence[str], colors: Sequence[str]) -> None:
    log_phi = np.asarray(summary["log_phi"]["mean"], dtype=float)
    p05 = np.asarray(summary["log_phi"]["p05"], dtype=float)
    p95 = np.asarray(summary["log_phi"]["p95"], dtype=float)
    rhs_mean = float(summary["log_tau"]["mean"])
    rhs_p05 = float(summary["log_tau"]["p05"])
    rhs_p95 = float(summary["log_tau"]["p95"])

    x = np.arange(log_phi.size)
    ax.errorbar(x, log_phi, yerr=[log_phi - p05, p95 - log_phi], fmt="o-", color="#1f77b4", ecolor="#1f77b4", capsize=3)
    span_x = np.linspace(-0.5, log_phi.size - 0.5, 200)
    ax.hlines(rhs_mean, span_x.min(), span_x.max(), colors="#7f7f7f", linestyles="--")
    ax.fill_between(span_x, rhs_p05, rhs_p95, color="#7f7f7f", alpha=0.2, linewidth=0)
    ax.set_xticks(x, xtick_labels)
    for tick, color in zip(ax.get_xticklabels(), colors):
        tick.set_color(color)
    ax.set_ylabel("log-scale (ln)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)


def _plot_mse_panel(ax: plt.Axes, summary: Mapping[str, object], title: str, xtick_labels: Sequence[str], colors: Sequence[str]) -> None:
    mse_g = [summary["mse"]["grrhs"][str(g)]["mean"] for g in range(len(xtick_labels))]
    mse_r = [summary["mse"]["rhs"][str(g)]["mean"] for g in range(len(xtick_labels))]
    x = np.arange(len(xtick_labels))
    width = 0.35
    ax.bar(x - width / 2, mse_g, width, color="#1f77b4", label="GRRHS")
    ax.bar(x + width / 2, mse_r, width, color="#7f7f7f", label="RHS")
    ax.set_xticks(x, xtick_labels)
    for tick, color in zip(ax.get_xticklabels(), colors):
        tick.set_color(color)
    ax.set_ylabel("group MSE")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)


def main() -> None:
    args = _parse_args()
    fig_root: Path = args.fig_root
    outdir = args.outdir or (fig_root / f"{args.scenario}_panels")
    outdir.mkdir(parents=True, exist_ok=True)

    # Load one summary to recover group tags
    sample_dir = fig_root / f"{args.scenario}_snr0p1_grrhs_vs_rhs"
    sample_summary = _load_summary(sample_dir / "group_comparison_summary.json")
    tags = list(sample_summary["group_tags"])  # strong/medium/weak/null
    xtick_labels = [f"g{g}\n({t})" for g, t in enumerate(tags)]
    tick_colors = []
    color_map = {"strong": "#d62728", "medium": "#ff7f0e", "weak": "#9467bd", "null": "#7f7f7f"}
    for t in tags:
        tick_colors.append(color_map.get(t, "#7f7f7f"))

    # Compose scales panel
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    titles = {"0p1": "SNR=0.1", "0p5": "SNR=0.5", "1p0": "SNR=1.0", "3p0": "SNR=3.0"}
    for ax, tok in zip(axes.flat, SNR_TOKENS):
        summary = _load_summary(fig_root / f"{args.scenario}_snr{tok}_grrhs_vs_rhs" / "group_comparison_summary.json")
        _plot_scales_panel(ax, summary, titles[tok], xtick_labels, tick_colors)
    fig.suptitle(f"{args.scenario}: Group shrinkage scales across SNR", y=0.98)
    fig.tight_layout()
    fig.savefig(outdir / f"{args.scenario}_snr_all_scales.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Compose MSE panel
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, tok in zip(axes.flat, SNR_TOKENS):
        summary = _load_summary(fig_root / f"{args.scenario}_snr{tok}_grrhs_vs_rhs" / "group_comparison_summary.json")
        _plot_mse_panel(ax, summary, titles[tok], xtick_labels, tick_colors)
    fig.suptitle(f"{args.scenario}: Per-group error across SNR", y=0.98)
    fig.tight_layout()
    fig.savefig(outdir / f"{args.scenario}_snr_all_mse.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Panels written to {outdir}")


if __name__ == "__main__":
    main()
