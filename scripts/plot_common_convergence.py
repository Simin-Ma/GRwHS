"""Generate common MCMC convergence diagnostics from posterior_samples.npz.

Outputs:
- trace_panel.png
- rank_hist_panel.png
- acf_panel.png
- running_mean_panel.png
- pair_panel.png
- convergence_summary.csv / convergence_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from grrhs.diagnostics.convergence import effective_sample_size, split_rhat


def _as_chain_draws(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim >= 2:
        return arr.reshape(arr.shape[0], arr.shape[1], -1)
    raise ValueError(f"Unsupported sample shape: {arr.shape}")


def _extract_scalar_series(posterior: Dict[str, np.ndarray], name: str) -> np.ndarray | None:
    if name not in posterior:
        return None
    x = _as_chain_draws(posterior[name])
    if x.ndim == 3:
        # choose first component for scalar-like arrays with trailing 1.
        return x[:, :, 0]
    return x


def _extract_vector_component(posterior: Dict[str, np.ndarray], name: str, idx: int = 0) -> np.ndarray | None:
    if name not in posterior:
        return None
    x = _as_chain_draws(posterior[name])
    if x.ndim == 2:
        return x
    comp = max(0, min(int(idx), x.shape[2] - 1))
    return x[:, :, comp]


def _rank_per_chain(x_cd: np.ndarray) -> List[np.ndarray]:
    c, d = x_cd.shape
    flat = x_cd.reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(flat.size)
    ranks = ranks.reshape(c, d)
    return [ranks[i] for i in range(c)]


def _acf_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    y = np.asarray(x, dtype=float)
    y = y - np.mean(y)
    if y.size == 0:
        return np.zeros(max_lag + 1, dtype=float)
    corr = np.correlate(y, y, mode="full")
    mid = corr.size // 2
    denom = corr[mid]
    if denom <= 0:
        return np.zeros(max_lag + 1, dtype=float)
    return corr[mid : mid + max_lag + 1] / denom


def _series_stats(series: Dict[str, np.ndarray]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for name, x in series.items():
        rhat = float(np.asarray(split_rhat(x, scalar_param=True)).reshape(-1)[0])
        ess = float(np.asarray(effective_sample_size(x, scalar_param=True)).reshape(-1)[0])
        sd = float(np.std(x.reshape(-1), ddof=1))
        mcse = float(sd / np.sqrt(max(ess, 1.0)))
        rows.append(
            {
                "name": name,
                "rhat": rhat,
                "ess_bulk": ess,
                "post_mean": float(np.mean(x)),
                "post_sd": sd,
                "mcse_mean": mcse,
                "mcse_over_sd": float(mcse / max(sd, 1e-12)),
            }
        )
    return rows


def _plot_trace(series: Dict[str, np.ndarray], out: Path) -> None:
    keys = list(series.keys())
    fig, axes = plt.subplots(len(keys), 1, figsize=(10, max(2.2, 1.9 * len(keys))), sharex=True)
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        x = series[key]
        for c in range(x.shape[0]):
            ax.plot(np.arange(x.shape[1]), x[c], lw=0.8, alpha=0.85, label=f"chain {c+1}")
        ax.set_title(f"Trace: {key}", fontsize=10)
    axes[-1].set_xlabel("draw")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _plot_rank(series: Dict[str, np.ndarray], out: Path) -> None:
    keys = list(series.keys())
    fig, axes = plt.subplots(len(keys), 1, figsize=(10, max(2.2, 1.8 * len(keys))), sharex=False)
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        ranks = _rank_per_chain(series[key])
        bins = 20
        for c, r in enumerate(ranks):
            ax.hist(r, bins=bins, alpha=0.45, label=f"chain {c+1}")
        ax.set_title(f"Rank Histogram: {key}", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _plot_acf(series: Dict[str, np.ndarray], out: Path, max_lag: int) -> None:
    keys = list(series.keys())
    fig, axes = plt.subplots(len(keys), 1, figsize=(10, max(2.2, 1.8 * len(keys))), sharex=True)
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        x = series[key]
        for c in range(x.shape[0]):
            ac = _acf_1d(x[c], max_lag=max_lag)
            ax.plot(np.arange(ac.size), ac, lw=0.9, alpha=0.85, label=f"chain {c+1}")
        ax.set_ylim(-0.2, 1.0)
        ax.set_title(f"ACF: {key}", fontsize=10)
    axes[-1].set_xlabel("lag")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _plot_running_mean(series: Dict[str, np.ndarray], out: Path) -> None:
    keys = list(series.keys())
    fig, axes = plt.subplots(len(keys), 1, figsize=(10, max(2.2, 1.8 * len(keys))), sharex=False)
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        x = series[key]
        pooled = x.reshape(-1)
        checkpoints = np.linspace(20, pooled.size, num=min(220, pooled.size), dtype=int)
        means = np.array([np.mean(pooled[:k]) for k in checkpoints], dtype=float)
        ax.plot(checkpoints, means, lw=1.1)
        ax.axhline(np.mean(pooled), ls="--", lw=0.8, color="black")
        ax.set_title(f"Running Mean: {key}", fontsize=10)
    axes[-1].set_xlabel("draws (pooled)")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _plot_pair(series: Dict[str, np.ndarray], out: Path) -> None:
    keys = list(series.keys())[:4]
    if len(keys) < 2:
        return
    rows = len(keys) - 1
    fig, axes = plt.subplots(rows, 1, figsize=(7, max(2.5, 2.4 * rows)))
    if rows == 1:
        axes = [axes]
    base = keys[0]
    xb = series[base].reshape(-1)
    for i, ax in enumerate(axes, start=1):
        ky = keys[i]
        y = series[ky].reshape(-1)
        n = min(xb.size, y.size, 12000)
        ax.scatter(xb[:n], y[:n], s=3, alpha=0.15)
        ax.set_xlabel(base)
        ax.set_ylabel(ky)
        ax.set_title(f"Pair: {base} vs {ky}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate common MCMC convergence plots.")
    p.add_argument("--run-dir", required=True, type=Path, help="Fold directory containing posterior_samples.npz")
    p.add_argument("--dest", required=True, type=Path, help="Output directory")
    p.add_argument("--max-lag", type=int, default=80)
    args = p.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    dest = args.dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    posterior_path = run_dir / "posterior_samples.npz"
    if not posterior_path.exists():
        raise FileNotFoundError(f"Missing posterior file: {posterior_path}")
    z = np.load(posterior_path)
    posterior = {k: np.asarray(z[k]) for k in z.files}

    series: Dict[str, np.ndarray] = {}
    tau = _extract_scalar_series(posterior, "tau")
    if tau is not None:
        series["tau"] = tau
    sigma = _extract_scalar_series(posterior, "sigma")
    if sigma is None:
        sigma2 = _extract_scalar_series(posterior, "sigma2")
        if sigma2 is not None:
            sigma = np.sqrt(np.maximum(sigma2, 1e-12))
    if sigma is not None:
        series["sigma"] = sigma
    intercept = _extract_scalar_series(posterior, "intercept")
    if intercept is not None:
        series["intercept"] = intercept

    for nm in ("beta", "gamma", "phi", "lambda"):
        comp = _extract_vector_component(posterior, nm, idx=0)
        if comp is not None:
            series[f"{nm}[0]"] = comp

    if not series:
        raise RuntimeError("No compatible scalar series found in posterior_samples.npz")

    _plot_trace(series, dest / "trace_panel.png")
    _plot_rank(series, dest / "rank_hist_panel.png")
    _plot_acf(series, dest / "acf_panel.png", max_lag=max(10, int(args.max_lag)))
    _plot_running_mean(series, dest / "running_mean_panel.png")
    _plot_pair(series, dest / "pair_panel.png")

    rows = _series_stats(series)
    import csv

    with (dest / "convergence_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "rhat", "ess_bulk", "post_mean", "post_sd", "mcse_mean", "mcse_over_sd"],
        )
        writer.writeheader()
        writer.writerows(rows)
    (dest / "convergence_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[OK] Wrote convergence bundle to {dest}")


if __name__ == "__main__":
    main()

