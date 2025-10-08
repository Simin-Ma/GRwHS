"""Configurable diagnostics plots for GRwHS experiment runs."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml

from grwhs.diagnostics.shrinkage import (
    prior_precision,
    regularized_lambda,
    shrinkage_kappa,
)
from grwhs.metrics.evaluation import _predictive_draws


@dataclass
class RunArtifacts:
    """Lightweight container for persisted experiment artifacts used in diagnostics."""

    run_dir: Path
    metrics: Dict
    dataset_meta: Dict
    resolved_config: Dict
    posterior: Dict[str, np.ndarray]
    dataset_arrays: Dict[str, np.ndarray]


def load_run_artifacts(run_dir: Path) -> RunArtifacts:
    """Load metrics, metadata, posterior samples, and dataset arrays for a run."""

    run_dir = run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    def _read_json(path: Path) -> Dict:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    metrics = _read_json(run_dir / "metrics.json")
    dataset_meta = _read_json(run_dir / "dataset_meta.json")
    resolved_config: Dict = {}
    cfg_path = run_dir / "resolved_config.yaml"
    if cfg_path.exists():
        resolved_config = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    posterior: Dict[str, np.ndarray] = {}
    posterior_path = run_dir / "posterior_samples.npz"
    if posterior_path.exists():
        posterior_arrays = np.load(posterior_path)
        posterior = {k: posterior_arrays[k] for k in posterior_arrays.files}

    dataset_arrays: Dict[str, np.ndarray] = {}
    dataset_path = run_dir / "dataset.npz"
    if dataset_path.exists():
        ds = np.load(dataset_path)
        dataset_arrays = {k: ds[k] for k in ds.files}

    return RunArtifacts(
        run_dir=run_dir,
        metrics=metrics,
        dataset_meta=dataset_meta,
        resolved_config=resolved_config,
        posterior=posterior,
        dataset_arrays=dataset_arrays,
    )


def _flatten_groups(groups: Sequence[Sequence[int]], p: int) -> np.ndarray:
    """Convert nested group index lists into a length-p index map."""
    group_index = np.zeros(p, dtype=int)
    for gid, idxs in enumerate(groups):
        group_index[np.asarray(list(idxs), dtype=int)] = gid
    return group_index


def _xtx_diagonal(X: np.ndarray) -> np.ndarray:
    """Compute the diagonal of X^T X."""
    return np.sum(np.square(X), axis=0)


def compute_mean_kappa_series(
    *,
    X: np.ndarray,
    group_index: np.ndarray,
    lambda_samples: np.ndarray,
    tau_samples: np.ndarray,
    phi_samples: np.ndarray,
    sigma_samples: np.ndarray,
    slab_width: float,
) -> np.ndarray:
    """Compute mean shrinkage (kappa) per posterior draw."""

    xtx_diag = _xtx_diagonal(X)
    T, p = lambda_samples.shape
    if tau_samples.shape[0] != T or sigma_samples.shape[0] != T:
        raise ValueError("Tau and sigma samples must align with lambda samples.")
    if phi_samples.shape[0] != T:
        raise ValueError("Phi samples must align with lambda samples.")

    mean_kappa = np.empty(T, dtype=float)
    sigma_sq = np.square(np.maximum(sigma_samples, 1e-12))

    for t in range(T):
        lam_t = lambda_samples[t]
        tau_t = float(max(tau_samples[t], 1e-12))
        sig_t = float(math.sqrt(sigma_sq[t]))
        phi_t = phi_samples[t, group_index]

        tilde_lambda_sq = regularized_lambda(lam_t, tau_t, slab_width)
        prior_prec = prior_precision(phi_t, tau_t, tilde_lambda_sq, sig_t)
        kappa = shrinkage_kappa(xtx_diag, sigma_sq[t], prior_prec)
        mean_kappa[t] = float(np.mean(kappa))

    return mean_kappa


def trace_plot(
    traces: Dict[str, np.ndarray],
    *,
    burn_in: Optional[int] = None,
    titles: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Create trace plots for the given series."""

    keys = list(traces.keys())
    fig, axes = plt.subplots(len(keys), 1, figsize=figsize, sharex=True)
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        series = np.asarray(traces[key])
        ax.plot(series, linewidth=0.9)
        ax.set_ylabel(key)
        if titles and key in titles:
            ax.set_title(titles[key])
        if burn_in is not None and 0 < burn_in < series.size:
            ax.axvline(burn_in, color="red", linestyle="--", linewidth=1.0, label="burn-in")
            ax.legend(loc="upper right")
    axes[-1].set_xlabel("Iteration")
    fig.tight_layout()
    return fig


def autocorrelation(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute autocorrelation up to max_lag."""

    x = np.asarray(series, dtype=float)
    x = x - np.mean(x)
    if x.size == 0:
        return np.zeros(max_lag + 1, dtype=float)
    corr = np.correlate(x, x, mode="full")
    mid = corr.size // 2
    denominator = corr[mid]
    if denominator == 0:
        return np.zeros(max_lag + 1, dtype=float)
    acf = corr[mid : mid + max_lag + 1] / denominator
    return acf


def autocorrelation_plot(
    acf_series: Dict[str, np.ndarray],
    *,
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """Plot autocorrelation functions."""

    keys = list(acf_series.keys())
    fig, axes = plt.subplots(len(keys), 1, figsize=figsize, sharex=True)
    if len(keys) == 1:
        axes = [axes]
    lags = None
    for ax, key in zip(axes, keys):
        acf = np.asarray(acf_series[key])
        if lags is None:
            lags = np.arange(acf.size)
        markerline, stemlines, baseline = ax.stem(lags, acf, basefmt=" ")
        plt.setp(stemlines, linewidth=1.2)
        plt.setp(markerline, markersize=4)
        ax.set_ylabel(key)
        ax.axhline(0.0, color="black", linewidth=0.8)
    axes[-1].set_xlabel("Lag")
    fig.tight_layout()
    return fig


def _select_indices(candidates: Sequence[int], count: int) -> List[int]:
    arr = list(dict.fromkeys(int(idx) for idx in candidates))
    return arr[:count]


def posterior_density_grid(
    samples: np.ndarray,
    *,
    indices: Sequence[int],
    truths: Optional[np.ndarray] = None,
    title_prefix: str,
    bins: int = 60,
    figsize: Tuple[int, int] = (12, 4),
) -> plt.Figure:
    """Plot posterior densities (histograms) for a subset of coefficients."""

    idx = list(indices)
    cols = min(len(idx), 5)
    rows = int(math.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    for ax, coef_idx in zip(axes.flat, idx):
        draw = samples[:, coef_idx]
        ax.hist(draw, bins=bins, density=True, alpha=0.75, color="#4c72b0")
        ax.set_title(f"{title_prefix} β[{coef_idx}]")
        if truths is not None and coef_idx < truths.size:
            ax.axvline(truths[coef_idx], color="black", linestyle="--", linewidth=1.0, label="truth")
            ax.legend(loc="upper right", fontsize=8)
        ax.axvline(0.0, color="grey", linestyle=":", linewidth=1.0)
    # Hide unused axes
    for ax in axes.flat[len(idx) :]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def phi_violin_plot(
    phi_samples: np.ndarray,
    *,
    group_sizes: Sequence[int],
    max_groups: int = 12,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Create violin plots of group-level phi samples ordered by median."""

    medians = np.median(phi_samples, axis=0)
    order = np.argsort(medians)[::-1]
    order = order[: max_groups + 1]

    ordered_samples = [phi_samples[:, g] for g in order]
    labels = [f"g{g}\n(|G|={group_sizes[g]})" for g in order]

    fig, ax = plt.subplots(figsize=figsize)
    parts = ax.violinplot(ordered_samples, showmeans=True, showmedians=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#55a868")
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)
    ax.scatter(np.arange(1, len(ordered_samples) + 1), [np.median(s) for s in ordered_samples], color="black", zorder=3)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("ϕ_g")
    ax.set_title("Group-level shrinkage (ϕ_g)")
    fig.tight_layout()
    return fig


def coverage_width_curve(
    *,
    predictive_draws: np.ndarray,
    y_true: np.ndarray,
    levels: Sequence[float],
    baseline_points: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    figsize: Tuple[int, int] = (7, 5),
) -> plt.Figure:
    """Plot empirical coverage against interval width across coverage levels."""

    draws = np.asarray(predictive_draws, dtype=float)
    y = np.asarray(y_true, dtype=float)
    if draws.ndim != 2 or draws.shape[1] != y.size:
        raise ValueError("Predictive draws must have shape (S, n) matching y_true length.")

    coverage = []
    widths = []

    for level in levels:
        alpha = (1.0 - level) / 2.0
        lower = np.quantile(draws, alpha, axis=0)
        upper = np.quantile(draws, 1.0 - alpha, axis=0)
        cov = np.mean((y >= lower) & (y <= upper))
        width = np.mean(upper - lower)
        coverage.append(cov)
        widths.append(width)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(widths, coverage, marker="o", label="GRwHS")
    ax.axhline(levels[-1], color="grey", linestyle="--", linewidth=1.0, label=f"Target {levels[-1]:.0%}")
    if baseline_points is not None:
        base_widths, base_cov = baseline_points
        ax.scatter(base_widths, base_cov, marker="s", color="#c44e52", label="Baseline")
    ax.set_xlabel("Average interval width")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Coverage vs. interval width")
    ax.legend()
    fig.tight_layout()
    return fig


def prepare_predictive_draws(
    artifacts: RunArtifacts,
    *,
    rng_seed: int = 0,
    burn_in: int = 0,
) -> Optional[np.ndarray]:
    """Compute predictive draws using posterior coefficient and sigma samples."""

    posterior = artifacts.posterior
    if not posterior or "beta" not in posterior:
        return None
    coef_samples = posterior["beta"]
    if burn_in > 0 and burn_in < coef_samples.shape[0]:
        coef_samples = coef_samples[burn_in:]
    sigma_samples = None
    if "sigma" in posterior:
        sigma_samples = posterior["sigma"]
    elif "sigma2" in posterior:
        sigma2 = posterior["sigma2"]
        if burn_in > 0 and burn_in < sigma2.shape[0]:
            sigma2 = sigma2[burn_in:]
        sigma_samples = np.sqrt(np.maximum(sigma2, 1e-12))
    elif "sigma" in posterior and burn_in > 0:
        sigma_samples = posterior["sigma"][burn_in:]
    if sigma_samples is not None and burn_in > 0 and sigma_samples.shape[0] > coef_samples.shape[0]:
        sigma_samples = sigma_samples[-coef_samples.shape[0] :]
    intercept = 0.0
    metrics = artifacts.metrics or {}
    model_section = metrics.get("model_params", {})
    if isinstance(model_section, dict) and "intercept" in model_section:
        intercept = float(model_section["intercept"])

    X_test = artifacts.dataset_arrays.get("X_test")
    if X_test is None:
        return None
    return _predictive_draws(
        X=np.asarray(X_test, dtype=float),
        coef_samples=np.asarray(coef_samples, dtype=float),
        intercept=intercept,
        sigma_samples=sigma_samples,
        rng_seed=rng_seed,
    )


def reconstruction_plot(
    *,
    X: np.ndarray,
    y_obs: np.ndarray,
    beta_samples: np.ndarray,
    beta_true: Optional[np.ndarray] = None,
    burn_in: int = 0,
    title: str = "Posterior reconstruction vs. truth",
    figsize: Tuple[int, int] = (10, 4),
) -> plt.Figure:
    """Plot observed responses, posterior mean reconstruction, and true signal."""

    X = np.asarray(X, dtype=float)
    y = np.asarray(y_obs, dtype=float).ravel()
    beta_samples = np.asarray(beta_samples, dtype=float)
    if beta_samples.ndim != 2:
        raise ValueError("beta_samples must be (S, p).")
    if X.shape[0] != y.size:
        raise ValueError("X and y dimensions mismatch.")

    samples = beta_samples[burn_in:] if burn_in > 0 else beta_samples
    if samples.size == 0:
        raise ValueError("No posterior samples remain after burn-in.")
    beta_mean = samples.mean(axis=0)
    y_post = X @ beta_mean
    y_true = None
    if beta_true is not None:
        beta_true = np.asarray(beta_true, dtype=float).ravel()
        if beta_true.size == X.shape[1]:
            y_true = X @ beta_true

    fig, ax = plt.subplots(figsize=figsize)
    idx = np.arange(y.size)
    ax.scatter(idx, y, marker="x", color="grey", label="observed $y$")
    ax.scatter(idx, y_post, color="black", s=15, label=r"posterior mean $\hat{y}$")
    if y_true is not None:
        ax.plot(idx, y_true, color="red", linewidth=1.5, label="true signal")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Response")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def build_group_sizes(groups: Sequence[Sequence[int]], p: int) -> List[int]:
    sizes = [len(g) for g in groups]
    if len(sizes) < 1:
        return [1] * p
    return sizes
