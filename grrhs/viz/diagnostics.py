"""Configurable diagnostics plots for GRRHS experiment runs."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import yaml
import warnings

from grrhs.diagnostics.shrinkage import (
    prior_precision,
    regularized_lambda,
    shrinkage_kappa,
)
from grrhs.metrics.evaluation import _predictive_draws


def _tight_layout(fig: plt.Figure) -> None:
    """Apply tight_layout while silencing compatibility warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible with tight_layout",
            category=UserWarning,
        )
        fig.tight_layout()


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
    _tight_layout(fig)
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
    _tight_layout(fig)
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
    _tight_layout(fig)
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
    _tight_layout(fig)
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
    ax.plot(widths, coverage, marker="o", label="GRRHS")
    ax.axhline(levels[-1], color="grey", linestyle="--", linewidth=1.0, label=f"Target {levels[-1]:.0%}")
    if baseline_points is not None:
        base_widths, base_cov = baseline_points
        ax.scatter(base_widths, base_cov, marker="s", color="#c44e52", label="Baseline")
    ax.set_xlabel("Average interval width")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Coverage vs. interval width")
    ax.legend()
    _tight_layout(fig)
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

    # Use a publication-friendly style for consistent typography.
    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=figsize)
    idx = np.arange(y.size)
    ax.scatter(idx, y, marker="x", color="lightgray", alpha=0.4, label="observed $y$")
    ax.scatter(idx, y_post, color="black", s=15, label=r"posterior mean $\hat{y}$")

    median_pred = None
    lower = upper = None
    if beta_samples.shape[0] > 1:
        draws = X @ samples.T  # (n, S')
        median_pred = np.median(draws, axis=1)
        lower = np.quantile(draws, 0.1, axis=1)
        upper = np.quantile(draws, 0.9, axis=1)

    if y_true is not None:
        ax.plot(idx, y_true, color="red", linewidth=1.5, alpha=0.8, label="true signal")
    if median_pred is not None:
        ax.plot(idx, median_pred, color="black", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Response")
    ax.set_title(title)
    all_values = [np.abs(y), np.abs(y_post)]
    if y_true is not None:
        all_values.append(np.abs(y_true))
    if lower is not None and upper is not None:
        all_values.extend([np.abs(lower), np.abs(upper)])
    vmax = max(np.max(v) for v in all_values) if all_values else 1.0
    ax.set_ylim(-1.1 * vmax, 1.1 * vmax)
    ax.legend(loc="upper left", frameon=False)
    _tight_layout(fig)
    return fig


def build_group_sizes(groups: Sequence[Sequence[int]], p: int) -> List[int]:
    sizes = [len(g) for g in groups]
    if len(sizes) < 1:
        return [1] * p
    return sizes


def group_shrinkage_landscape(
    phi_samples: np.ndarray,
    *,
    groups: Sequence[Sequence[int]],
    active_idx: Optional[Sequence[int]] = None,
    figsize: Tuple[int, int] = (10, 4),
) -> plt.Figure:
    """Scatter plot of group-level shrinkage scales with signal highlighting."""

    plt.style.use("seaborn-v0_8-paper")
    phi_samples = np.asarray(phi_samples, dtype=float)
    G = phi_samples.shape[1]
    means = phi_samples.mean(axis=0)
    ci_lower, ci_upper = np.quantile(phi_samples, [0.05, 0.95], axis=0)
    variances = phi_samples.var(axis=0)
    if np.allclose(variances.max(), 0.0):
        alphas = np.ones(G)
    else:
        scaled = (variances - variances.min()) / max(variances.max() - variances.min(), 1e-12)
        alphas = 0.3 + 0.7 * (1.0 - scaled)

    active_idx = set(int(i) for i in (active_idx or []))
    signal_mask = np.zeros(G, dtype=bool)
    for g, idxs in enumerate(groups):
        if any(int(j) in active_idx for j in idxs):
            signal_mask[g] = True

    x = np.arange(1, G + 1)
    colors = np.where(signal_mask, "red", "gray")

    fig, ax = plt.subplots(figsize=figsize)
    for g in range(G):
        ax.errorbar(
            x[g],
            means[g],
            yerr=[[means[g] - ci_lower[g]], [ci_upper[g] - means[g]]],
            fmt="o",
            color=colors[g],
            alpha=alphas[g],
            ecolor=colors[g],
            elinewidth=1.0,
            capsize=3,
        )

    ax.set_xlabel("Group index")
    ax.set_ylabel(r"Posterior mean $\phi_g$")
    ax.set_title("Group-level shrinkage landscape")
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="red", linestyle="", label="signal group"),
        plt.Line2D([0], [0], marker="o", color="gray", linestyle="", label="null group"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False)
    _tight_layout(fig)
    return fig


def posterior_mean_heatmap_by_model(
    beta_means: Sequence[np.ndarray],
    *,
    labels: Sequence[str],
    groups: Sequence[Sequence[int]],
    active_idx: Optional[Sequence[int]] = None,
    cmap: str = "RdBu_r",
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """Visualize posterior coefficient means for one or multiple models."""

    if len(beta_means) != len(labels):
        raise ValueError("beta_means and labels must have the same length.")
    if not beta_means:
        raise ValueError("beta_means must contain at least one entry.")

    converted = [np.asarray(b, dtype=float).reshape(-1) for b in beta_means]
    p = converted[0].size
    if any(arr.size != p for arr in converted):
        raise ValueError("All beta_means must share the same dimensionality.")

    matrix = np.vstack(converted)
    vmax = float(np.max(np.abs(matrix))) if np.max(np.abs(matrix)) > 0 else 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    if figsize is None:
        height = max(2.5, len(converted) * 1.8)
        figsize = (12.0, height)

    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Coefficient index")
    ax.set_ylabel("Model")
    ax.set_title("Posterior mean of coefficients by model")

    group_sizes = [len(g) for g in groups]
    cumulative = np.cumsum(group_sizes)
    for boundary in cumulative[:-1]:
        ax.axvline(boundary - 0.5, color="black", linewidth=0.6, linestyle="--", alpha=0.6)

    active_set = {int(i) for i in (active_idx or [])}
    if active_set:
        cols = np.asarray(sorted(active_set), dtype=float)
        for row in range(matrix.shape[0]):
            ax.scatter(
                cols,
                np.full(cols.shape, row, dtype=float),
                marker="o",
                facecolors="none",
                edgecolors="black",
                linewidths=0.8,
                s=30,
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(r"$\mathbb{E}[\beta_j]$")
    ax.set_xlim(-0.5, p - 0.5)
    _tight_layout(fig)
    return fig



def group_scale_vs_lambda_panel(
    lambda_samples: Sequence[np.ndarray],
    *,
    labels: Sequence[str],
    groups: Sequence[Sequence[int]],
    phi_samples: Optional[Sequence[Optional[np.ndarray]]] = None,
    active_idx: Optional[Sequence[int]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    beta_true: Optional[np.ndarray] = None,
    normalize_truth: bool = True,
    tau_samples: Optional[Sequence[Optional[np.ndarray]]] = None,
    slab_widths: Optional[Sequence[float]] = None,
    normalize_scales: bool = True,
) -> plt.Figure:
    """Compare group-level scales and normalized local scales across models."""

    num_models = len(lambda_samples)
    if num_models == 0:
        raise ValueError("lambda_samples must contain at least one entry.")
    if len(labels) != num_models:
        raise ValueError("lambda_samples and labels must have the same length.")

    phi_samples = phi_samples or [None] * num_models
    if len(phi_samples) != num_models:
        raise ValueError("phi_samples must match the number of models.")

    if tau_samples is None:
        tau_samples = [None] * num_models
    if len(tau_samples) != num_models:
        raise ValueError("tau_samples must match the number of models.")

    if slab_widths is None:
        slab_widths = [1.0] * num_models
    if len(slab_widths) != num_models:
        raise ValueError("slab_widths must match the number of models.")

    lambda_arrays = [np.asarray(arr, dtype=float) for arr in lambda_samples]
    p = lambda_arrays[0].shape[1]
    if any(arr.ndim != 2 for arr in lambda_arrays):
        raise ValueError("Each lambda_samples entry must be a 2D array (samples, coefficients).")
    if any(arr.shape[1] != p for arr in lambda_arrays):
        raise ValueError("All lambda_samples must share the same number of coefficients.")

    tau_arrays: List[Optional[np.ndarray]] = []
    for tau_entry in tau_samples:
        if tau_entry is None:
            tau_arrays.append(None)
        else:
            tau_arr = np.asarray(tau_entry, dtype=float)
            if tau_arr.ndim != 1:
                raise ValueError("tau_samples entries must be one-dimensional.")
            tau_arrays.append(tau_arr)

    beta_true_arr = None
    if beta_true is not None:
        beta_true_arr = np.asarray(beta_true, dtype=float).reshape(-1)
        if beta_true_arr.size != p:
            raise ValueError("beta_true length must match the coefficient dimensionality.")

    phi_arrays: List[Optional[np.ndarray]] = []
    for entry in phi_samples:
        if entry is None:
            phi_arrays.append(None)
        else:
            phi_arr = np.asarray(entry, dtype=float)
            if phi_arr.ndim != 2:
                raise ValueError("phi_samples entries must be 2D arrays (samples, groups).")
            phi_arrays.append(phi_arr)

    active_set = {int(i) for i in (active_idx or [])}
    signal_flags = np.zeros(len(groups), dtype=bool)
    for g, idxs in enumerate(groups):
        if any(int(idx) in active_set for idx in idxs):
            signal_flags[g] = True

    truth_scale = 1.0
    truth_by_group: Optional[np.ndarray] = None
    truth_label = None
    coef_truth: Optional[np.ndarray] = None
    if beta_true_arr is not None:
        truth_vals = []
        for idxs in groups:
            idx_array = np.asarray(list(idxs), dtype=int)
            if idx_array.size == 0:
                truth_vals.append(0.0)
            else:
                truth_vals.append(float(np.linalg.norm(beta_true_arr[idx_array], ord=2)))
        truth_vals = np.asarray(truth_vals, dtype=float)
        max_truth = float(np.max(np.abs(truth_vals))) if truth_vals.size else 0.0
        if normalize_truth and max_truth > 0.0:
            truth_scale = max_truth
            truth_by_group = truth_vals / truth_scale
            truth_label = r"True group amplitude (normalized $\|\beta_g\|_2$)"
        else:
            truth_by_group = truth_vals
            truth_label = r"True group amplitude $\|\beta_g\|_2$"
            truth_scale = 1.0 if max_truth == 0.0 else max_truth

        abs_coef = np.abs(beta_true_arr)
        if normalize_truth and truth_scale > 0.0:
            coef_truth = abs_coef / truth_scale
        else:
            coef_truth = abs_coef

    group_positions = np.arange(1, len(groups) + 1, dtype=float)
    tick_labels = [f"g{g}" for g in range(len(groups))]
    include_truth_panel = beta_true_arr is not None and truth_by_group is not None and coef_truth is not None

    total_panels = num_models + (1 if include_truth_panel else 0)
    if figsize is None:
        height = max(3.0, total_panels * 2.6)
        figsize = (12.0, height)

    plt.style.use("seaborn-v0_8-paper")
    fig, axes = plt.subplots(total_panels, 1, figsize=figsize, sharex=True)
    axes = list(np.atleast_1d(axes))

    lambda_stats_raw: List[Dict[str, np.ndarray]] = []
    for lam_arr, tau_arr, slab in zip(lambda_arrays, tau_arrays, slab_widths):
        if tau_arr is None:
            raise ValueError("tau_samples are required to compute regularized local scales.")
        tau_arr = np.asarray(tau_arr, dtype=float)
        if tau_arr.shape[0] != lam_arr.shape[0]:
            raise ValueError("tau_samples and lambda_samples must share the same number of draws.")
        lam_tilde = np.empty_like(lam_arr)
        for draw_idx in range(lam_arr.shape[0]):
            lam_tilde[draw_idx] = np.sqrt(
                regularized_lambda(lam_arr[draw_idx], float(tau_arr[draw_idx]), float(slab))
            )
        lambda_stats_raw.append(
            {
                "median": np.median(lam_tilde, axis=0),
                "q25": np.quantile(lam_tilde, 0.25, axis=0),
                "q75": np.quantile(lam_tilde, 0.75, axis=0),
            }
        )

    phi_means_list: List[Optional[np.ndarray]] = []
    phi_q10_list: List[Optional[np.ndarray]] = []
    phi_q90_list: List[Optional[np.ndarray]] = []
    for phi_entry in phi_arrays:
        if phi_entry is None:
            phi_means_list.append(None)
            phi_q10_list.append(None)
            phi_q90_list.append(None)
        else:
            phi_means_list.append(np.mean(phi_entry, axis=0))
            q10, q90 = np.quantile(phi_entry, [0.10, 0.90], axis=0)
            phi_q10_list.append(q10)
            phi_q90_list.append(q90)

    lambda_scale = 1.0
    if normalize_scales:
        maxima = [float(np.max(stats["median"])) for stats in lambda_stats_raw if np.any(stats["median"] > 0)]
        if maxima:
            lambda_scale = max(maxima)
        if lambda_scale <= 0:
            lambda_scale = 1.0

    phi_scale = 1.0
    if normalize_scales:
        maxima = [float(np.max(mean)) for mean in phi_means_list if mean is not None and np.any(mean > 0)]
        if maxima:
            phi_scale = max(maxima)
        if phi_scale <= 0:
            phi_scale = 1.0

    panel_idx = 0
    if include_truth_panel:
        truth_ax = axes[panel_idx]
        panel_idx += 1
        coef_color = "#fdd5a2"
        for g_idx, idxs in enumerate(groups):
            idxs_array = np.asarray(list(idxs), dtype=int)
            if idxs_array.size == 0:
                continue
            jitter = (
                np.linspace(-0.2, 0.2, idxs_array.size, dtype=float)
                if idxs_array.size > 1
                else np.zeros(1, dtype=float)
            )
            truth_ax.scatter(
                np.full(idxs_array.size, group_positions[g_idx], dtype=float) + jitter,
                coef_truth[idxs_array],
                color=coef_color,
                alpha=0.4,
                s=28,
                marker="o",
                edgecolors="none",
                zorder=3,
            )

        marker_signal = "#e75480"
        marker_null = "#6897bb"
        group_colors = np.where(signal_flags, marker_signal, marker_null)
        truth_ax.scatter(
            group_positions,
            truth_by_group,
            c=group_colors,
            marker="+",
            s=90,
            linewidths=1.5,
            zorder=4,
            alpha=0.95,
        )

        legend_handles_truth: List[Line2D] = []
        if np.any(signal_flags):
            legend_handles_truth.append(
                Line2D(
                    [0],
                    [0],
                    marker="+",
                    color=marker_signal,
                    linestyle="",
                    linewidth=1.5,
                    markersize=9,
                    markeredgewidth=1.8,
                    markerfacecolor="none",
                    label=r"Signal group truth ($\|\beta_g\|_2$)",
                )
            )
        if np.any(~signal_flags):
            legend_handles_truth.append(
                Line2D(
                    [0],
                    [0],
                    marker="+",
                    color=marker_null,
                    linestyle="",
                    linewidth=1.5,
                    markersize=9,
                    markeredgewidth=1.8,
                    markerfacecolor="none",
                    label=r"Null group truth ($\|\beta_g\|_2$)",
                )
            )
        legend_handles_truth.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color=coef_color,
                linestyle="",
                markersize=6,
                alpha=0.6,
                markeredgewidth=0.0,
                label=r"Coefficient truth $|\beta_j|$",
            )
        )

        truth_ax.set_ylabel(truth_label or r"$\|\beta_g\|_2$")
        if normalize_truth:
            truth_ax.set_ylim(-0.05, 1.05)
        truth_ax.set_title("True shrinkage structure")
        truth_ax.set_xticks(group_positions)
        truth_ax.set_xticklabels(tick_labels)
        truth_ax.legend(
            handles=legend_handles_truth,
            loc="center left",
            bbox_to_anchor=(1.1, 0.5),
            frameon=False,
        )
        truth_ax.tick_params(axis="x", labelbottom=False)

    model_axes = axes[panel_idx:]
    for model_idx, ax in enumerate(model_axes):
        label = labels[model_idx]
        phi_samples_model = phi_arrays[model_idx]
        lambda_stats = lambda_stats_raw[model_idx]
        lambda_median = lambda_stats["median"] / lambda_scale if normalize_scales else lambda_stats["median"]
        lambda_q25 = lambda_stats["q25"] / lambda_scale if normalize_scales else lambda_stats["q25"]
        lambda_q75 = lambda_stats["q75"] / lambda_scale if normalize_scales else lambda_stats["q75"]

        ax.set_title(label)
        phi_ax = ax if phi_samples_model is not None else None
        lambda_ax = ax if phi_ax is None else ax.twinx()
        legend_handles: List[Line2D] = []

        if phi_ax is not None and phi_samples_model is not None:
            phi_mean_raw = phi_means_list[model_idx]
            phi_q10_raw = phi_q10_list[model_idx]
            phi_q90_raw = phi_q90_list[model_idx]
            if phi_mean_raw is None or phi_q10_raw is None or phi_q90_raw is None:
                raise ValueError("Inconsistent phi_samples detected.")
            phi_mean = phi_mean_raw / phi_scale if normalize_scales else phi_mean_raw
            phi_q10 = phi_q10_raw / phi_scale if normalize_scales else phi_q10_raw
            phi_q90 = phi_q90_raw / phi_scale if normalize_scales else phi_q90_raw
            colors = np.where(signal_flags, "#d7191c", "#2c7bb6")
            phi_ax.bar(group_positions, phi_mean, width=0.6, color=colors, alpha=0.45, label=r"$\phi_g$")
            for g_idx in range(len(groups)):
                phi_ax.errorbar(
                    group_positions[g_idx],
                    phi_mean[g_idx],
                    yerr=[
                        [phi_mean[g_idx] - phi_q10[g_idx]],
                        [phi_q90[g_idx] - phi_mean[g_idx]],
                    ],
                    fmt="none",
                    color=colors[g_idx],
                    linewidth=1.0,
                    alpha=0.7,
                )
            phi_ax.set_ylabel(
                r"Normalized $\mathbb{E}[\phi_g]$" if normalize_scales else r"$\mathbb{E}[\phi_g]$"
            )
            phi_ax.set_ylim(bottom=0.0)
            if np.any(signal_flags):
                legend_handles.append(
                    Line2D([0], [0], marker="s", color="#d7191c", linestyle="", label=r"signal $\phi_g$")
                )
            if np.any(~signal_flags):
                legend_handles.append(
                    Line2D([0], [0], marker="s", color="#2c7bb6", linestyle="", label=r"null $\phi_g$")
                )

        lambda_color = "#808080"
        for g_idx, idxs in enumerate(groups):
            idxs_array = np.asarray(list(idxs), dtype=int)
            jitter = (
                np.linspace(-0.18, 0.18, idxs_array.size, dtype=float)
                if idxs_array.size > 1
                else np.zeros(1, dtype=float)
            )
            lambda_ax.scatter(
                np.full(idxs_array.size, group_positions[g_idx], dtype=float) + jitter,
                lambda_median[idxs_array],
                color=lambda_color,
                alpha=0.75,
                s=28,
                edgecolors="none",
            )
            lambda_ax.vlines(
                np.full(idxs_array.size, group_positions[g_idx], dtype=float) + jitter,
                lambda_q25[idxs_array],
                lambda_q75[idxs_array],
                color=lambda_color,
                linewidth=0.8,
                alpha=0.5,
            )

        lambda_ax.set_ylabel(
            r"Normalized median $\tilde{\lambda}_j$" if normalize_scales else r"Median $\tilde{\lambda}_j$"
        )
        lambda_ax.set_ylim(bottom=0.0)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color=lambda_color,
                linestyle="",
                label=(
                    r"Normalized median $\tilde{\lambda}_j$ (IQR)"
                    if normalize_scales
                    else r"Median $\tilde{\lambda}_j$ (IQR)"
                ),
                markerfacecolor=lambda_color,
            )
        )

        if phi_ax is None:
            ax.set_ylabel(
                r"Normalized median $\tilde{\lambda}_j$"
                if normalize_scales
                else r"Median $\tilde{\lambda}_j$"
            )

        ax.set_xticks(group_positions)
        ax.set_xticklabels(tick_labels, rotation=0)
        if legend_handles:
            ax.legend(
                handles=legend_handles,
                loc="center left",
                bbox_to_anchor=(1.1, 0.5),
                frameon=False,
            )
        if ax is not axes[-1]:
            ax.tick_params(labelbottom=False)

    axes[-1].set_xlabel("Group index")
    _tight_layout(fig)
    return fig

def group_interval_calibration_scatter(
    beta_samples: Sequence[np.ndarray],
    *,
    labels: Sequence[str],
    groups: Sequence[Sequence[int]],
    beta_true: np.ndarray,
    level: float = 0.9,
    normalize_width: bool = True,
    figsize: Tuple[int, int] = (7, 5),
) -> plt.Figure:
    """Scatter plot comparing empirical coverage and interval widths per group."""

    if not beta_samples:
        raise ValueError("beta_samples must contain at least one entry.")
    if len(beta_samples) != len(labels):
        raise ValueError("beta_samples and labels must have the same length.")

    beta_true = np.asarray(beta_true, dtype=float).reshape(-1)
    converted = [np.asarray(samples, dtype=float) for samples in beta_samples]
    p = beta_true.size
    if any(arr.ndim != 2 for arr in converted):
        raise ValueError("Each beta_samples entry must be a 2D array (samples, coefficients).")
    if any(arr.shape[1] != p for arr in converted):
        raise ValueError("All beta_samples must share the same coefficient dimension as beta_true.")

    lower_q = (1.0 - level) / 2.0
    upper_q = 1.0 - lower_q

    coverages: Dict[str, np.ndarray] = {}
    widths: Dict[str, np.ndarray] = {}

    for label, draws in zip(labels, converted):
        lower = np.quantile(draws, lower_q, axis=0)
        upper = np.quantile(draws, upper_q, axis=0)
        width_coef = upper - lower
        contain = (beta_true >= lower) & (beta_true <= upper)

        cov_group = []
        width_group = []
        for idxs in groups:
            idx_array = np.asarray(list(idxs), dtype=int)
            if idx_array.size == 0:
                cov_group.append(np.nan)
                width_group.append(np.nan)
                continue
            cov_group.append(float(np.mean(contain[idx_array])))
            width_group.append(float(np.mean(width_coef[idx_array])))
        coverages[label] = np.asarray(cov_group, dtype=float)
        widths[label] = np.asarray(width_group, dtype=float)

    width_chunks = [w[~np.isnan(w)] for w in widths.values() if np.any(~np.isnan(w))]
    if width_chunks:
        all_widths = np.concatenate(width_chunks, axis=0)
        width_max = float(np.max(all_widths)) if all_widths.size else 1.0
    else:
        width_max = 1.0
    if width_max <= 0:
        width_max = 1.0

    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=figsize)

    markers = ["o", "s", "D", "^", "v"]
    for idx, label in enumerate(labels):
        cov = coverages[label]
        wid = widths[label]
        mask = ~np.isnan(cov) & ~np.isnan(wid)
        if not np.any(mask):
            continue
        cov_valid = cov[mask]
        wid_valid = wid[mask]
        if normalize_width:
            wid_plot = wid_valid / width_max
        else:
            wid_plot = wid_valid
        ax.scatter(
            cov_valid,
            wid_plot,
            label=label,
            marker=markers[idx % len(markers)],
            s=55,
            alpha=0.8,
        )

    if normalize_width:
        ax.plot([0.0, 1.0], [0.0, 1.0], color="black", linestyle="--", linewidth=0.9, label="45° reference")
        ax.set_ylabel("Normalized interval width")
    else:
        ax.set_ylabel("Average interval width")

    ax.axvline(level, color="#444444", linestyle=":", linewidth=0.9, label=f"Target coverage {level:.0%}")
    ax.set_xlabel(f"Empirical coverage @ {level:.0%}")
    ax.set_xlim(0.0, 1.0)
    if normalize_width:
        ax.set_ylim(0.0, 1.05)
    ax.legend(loc="best", frameon=False)
    _tight_layout(fig)
    return fig


def true_vs_estimated_panel(
    beta_means: Sequence[np.ndarray],
    *,
    labels: Sequence[str],
    beta_true: np.ndarray,
    group_index: np.ndarray,
    active_idx: Optional[Sequence[int]] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """Scatter plot comparing true vs. estimated coefficients for one or multiple models."""

    if len(beta_means) != len(labels):
        raise ValueError("beta_means and labels must have the same length.")
    if not beta_means:
        raise ValueError("beta_means must contain at least one entry.")

    beta_true = np.asarray(beta_true, dtype=float).reshape(-1)
    group_index = np.asarray(group_index, dtype=int).reshape(-1)
    converted = [np.asarray(b, dtype=float).reshape(-1) for b in beta_means]
    p = beta_true.size
    if any(arr.size != p for arr in converted):
        raise ValueError("All beta_means must match the dimensionality of beta_true.")
    if group_index.size != p:
        raise ValueError("group_index length must match coefficient dimensionality.")

    if figsize is None:
        width = max(5.0, len(converted) * 4.0)
        figsize = (width, 4.5)

    plt.style.use("seaborn-v0_8-paper")
    fig, axes = plt.subplots(1, len(converted), figsize=figsize, sharex=True, sharey=True)
    if len(converted) == 1:
        axes = [axes]  # type: ignore[assignment]

    cmap = plt.get_cmap("tab20", int(group_index.max()) + 1 if group_index.size else 1)
    limit = max(
        np.max(np.abs(beta_true)),
        max(np.max(np.abs(arr)) for arr in converted),
        1e-3,
    )
    limit *= 1.05

    active_set = {int(i) for i in (active_idx or [])}

    for ax, beta_mean, label in zip(axes, converted, labels):
        scatter = ax.scatter(
            beta_true,
            beta_mean,
            c=group_index,
            cmap=cmap,
            s=24,
            alpha=0.8,
            edgecolors="none",
        )
        if active_set:
            idxs = np.asarray(sorted(active_set), dtype=int)
            ax.scatter(
                beta_true[idxs],
                beta_mean[idxs],
                facecolors="none",
                edgecolors="black",
                linewidths=0.8,
                s=50,
            )
        ax.axline((0.0, 0.0), slope=1.0, color="black", linestyle="--", linewidth=0.9)
        ax.axhline(0.0, color="#888888", linewidth=0.6, linestyle=":")
        ax.axvline(0.0, color="#888888", linewidth=0.6, linestyle=":")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_xlabel(r"True $\beta_j$")
        ax.set_title(label)

    axes[0].set_ylabel(r"Posterior mean $\hat{\beta}_j$")
    norm = Normalize(vmin=float(group_index.min()) if group_index.size else 0.0, vmax=float(group_index.max()) if group_index.size else 1.0)
    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        fraction=0.03,
        pad=0.02,
    )
    cbar.set_label("Group index")
    _tight_layout(fig)
    return fig


def group_coefficient_heatmap(
    beta_samples: np.ndarray,
    phi_samples: np.ndarray,
    *,
    groups: Sequence[Sequence[int]],
    active_idx: Optional[Sequence[int]] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Heatmap of posterior |beta_j| means by group with phi_g bars."""

    plt.style.use("seaborn-v0_8-paper")
    beta_samples = np.asarray(beta_samples, dtype=float)
    phi_samples = np.asarray(phi_samples, dtype=float)
    abs_mean = np.mean(np.abs(beta_samples), axis=0)
    phi_mean = phi_samples.mean(axis=0)
    G = len(groups)
    p = abs_mean.size

    heat = np.zeros((G, p), dtype=float)
    for g, idxs in enumerate(groups):
        idx_array = np.asarray(list(idxs), dtype=int)
        heat[g, idx_array] = abs_mean[idx_array]

    active_set = set(int(i) for i in (active_idx or []))

    cmap = LinearSegmentedColormap.from_list(
        "beta_heatmap",
        ["#ffffff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
    )
    vmax = float(np.max(heat)) if np.max(heat) > 0 else 1.0
    norm = Normalize(vmin=0.0, vmax=vmax)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1], wspace=0.1)
    ax = fig.add_subplot(gs[0, 0])
    img = ax.imshow(heat, aspect="auto", cmap=cmap, norm=norm)
    ax.set_yticks(np.arange(G))
    ax.set_yticklabels([f"g{g}" for g in range(G)])
    ax.set_xlabel("Coefficient index")
    ax.set_ylabel("Group index")
    ax.set_title(r"Group-wise posterior $|\beta_j|$ means")

    for g, idxs in enumerate(groups):
        for j in idxs:
            if int(j) in active_set:
                rect = Rectangle((j - 0.5, g - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=0.8)
                ax.add_patch(rect)

    ax_bar = fig.add_subplot(gs[0, 1], sharey=ax)
    ax_bar.barh(np.arange(G), phi_mean, color="#4c72b0")
    ax_bar.set_xlabel(r"$\phi_g$")
    ax_bar.set_title("Group scale")
    ax_bar.invert_yaxis()
    ax_bar.tick_params(labelleft=False)

    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\mathbb{E}[|\beta_j|]$")
    _tight_layout(fig)
    return fig


def group_vs_individual_scatter(
    phi_samples: np.ndarray,
    lambda_samples: np.ndarray,
    *,
    groups: Sequence[Sequence[int]],
    active_idx: Optional[Sequence[int]] = None,
    figsize: Tuple[int, int] = (6, 5),
) -> plt.Figure:
    """Scatter plot illustrating relationship between group and individual shrinkage."""

    plt.style.use("seaborn-v0_8-paper")
    phi_samples = np.asarray(phi_samples, dtype=float)
    lambda_samples = np.asarray(lambda_samples, dtype=float)
    phi_mean = phi_samples.mean(axis=0)
    lambda_med_coeff = np.median(lambda_samples, axis=0)

    active_idx = set(int(i) for i in (active_idx or []))
    phi_group = []
    lambda_group = []
    colores = []
    sizes = []

    for g, idxs in enumerate(groups):
        idxs = [int(i) for i in idxs]
        lambda_group.append(np.median(lambda_med_coeff[idxs]))
        phi_group.append(phi_mean[g])
        count_active = sum(1 for j in idxs if j in active_idx)
        sizes.append(80 + 40 * count_active)
        colores.append("red" if count_active > 0 else "gray")

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(phi_group, lambda_group, s=sizes, c=colores, alpha=0.8, edgecolor="black", linewidth=0.6)
    ax.set_xlabel(r"$\mathbb{E}[\phi_g]$")
    ax.set_ylabel(r"Median $\lambda_j$ within group")
    ax.set_title("Group vs. individual shrinkage scales")
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="red", linestyle="", label="signal group"),
        plt.Line2D([0], [0], marker="o", color="gray", linestyle="", label="null group"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=False)
    _tight_layout(fig)
    return fig
