"""CLI utility to generate configurable diagnostics plots for Bayesian runs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import numpy as np

from grrhs.viz.diagnostics import (
    RunArtifacts,
    autocorrelation,
    autocorrelation_plot,
    build_group_sizes,
    compute_mean_kappa_series,
    load_run_artifacts,
    phi_violin_plot,
    posterior_density_grid,
    prepare_predictive_draws,
    group_shrinkage_landscape,
    group_coefficient_heatmap,
    group_vs_individual_scatter,
    reconstruction_plot,
    trace_plot,
)
from grrhs.viz.diagnostics import coverage_width_curve as coverage_width_plot
from grrhs.viz.diagnostics import _flatten_groups, _select_indices


def _resolve_groups(artifacts: RunArtifacts, p: int) -> Sequence[Sequence[int]]:
    groups = artifacts.dataset_meta.get("groups")
    if groups:
        return groups
    return [[idx] for idx in range(p)]


def _extract_truths(artifacts: RunArtifacts) -> np.ndarray:
    truth = artifacts.dataset_arrays.get("beta_true")
    if truth is None:
        return np.ndarray(0, dtype=float)
    return np.asarray(truth, dtype=float).reshape(-1)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _flatten_chain_samples(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim <= 2:
        return arr
    lead = arr.shape[0] * arr.shape[1]
    return arr.reshape((lead,) + arr.shape[2:])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate diagnostics plots for a Bayesian run.")
    parser.add_argument("--run-dir", required=True, type=Path, help="Path to the primary run directory.")
    parser.add_argument("--dest", type=Path, default=None, help="Directory to write plots (default: <run>/figures).")
    parser.add_argument("--burn-in", type=int, default=None, help="Burn-in iterations to discard in plots.")
    parser.add_argument(
        "--burn-in-frac",
        type=float,
        default=0.5,
        help="If --burn-in is omitted, use this fraction of samples as burn-in.",
    )
    parser.add_argument("--max-lag", type=int, default=150, help="Maximum lag for autocorrelation plots.")
    parser.add_argument("--strong-count", type=int, default=4, help="Number of strong coefficients to plot.")
    parser.add_argument("--weak-count", type=int, default=4, help="Number of weak/zero coefficients to plot.")
    parser.add_argument("--groups-to-plot", type=int, default=12, help="Top-k groups to display in φ_g violin plot.")
    parser.add_argument(
        "--coverage-levels",
        nargs="+",
        type=float,
        default=[0.5, 0.7, 0.8, 0.9, 0.95],
        help="Coverage levels used in coverage-width calibration plot.",
    )
    parser.add_argument("--rng-seed", type=int, default=123, help="Random seed for predictive draws.")
    parser.add_argument("--dpi", type=int, default=180, help="DPI for saved figures.")
    args = parser.parse_args()

    artifacts = load_run_artifacts(args.run_dir)
    posterior = artifacts.posterior
    if not posterior:
        raise RuntimeError(f"No posterior_samples.npz found under {args.run_dir}")

    if "beta" not in posterior:
        raise RuntimeError("Posterior samples missing 'beta'; diagnostics plots require coefficient draws.")
    beta_samples = _flatten_chain_samples(np.asarray(posterior["beta"], dtype=float))
    if beta_samples.ndim > 2 and beta_samples.shape[-1] == 1:
        beta_samples = np.squeeze(beta_samples, axis=-1)
    if beta_samples.ndim != 2:
        beta_samples = beta_samples.reshape(beta_samples.shape[0], -1)
    total_samples = beta_samples.shape[0]
    burn_in = args.burn_in
    if burn_in is None:
        burn_in = int(total_samples * args.burn_in_frac)
    burn_in = max(0, min(burn_in, total_samples - 1))

    trimmed_slice = slice(burn_in, None)
    beta_trimmed = beta_samples[trimmed_slice]

    tau_samples = None
    if "tau" in posterior:
        tau_samples = _flatten_chain_samples(np.asarray(posterior["tau"], dtype=float))[trimmed_slice]
    else:
        _warn("Posterior samples missing 'tau'; skipping tau trace/ACF.")

    sigma_samples = None
    sigma2_series = None
    if "sigma2" in posterior:
        sigma2_series = _flatten_chain_samples(np.asarray(posterior["sigma2"], dtype=float))[trimmed_slice]
        sigma_samples = np.sqrt(np.maximum(sigma2_series, 1e-12))
    elif "sigma" in posterior:
        sigma_samples = _flatten_chain_samples(np.asarray(posterior["sigma"], dtype=float))[trimmed_slice]
        sigma2_series = np.square(sigma_samples)
    else:
        _warn("Posterior samples missing sigma/sigma2; skipping sigma trace/ACF and coverage plot.")
    if sigma_samples is not None and sigma_samples.ndim > 1 and sigma_samples.shape[-1] == 1:
        sigma_samples = np.squeeze(sigma_samples, axis=-1)
    if sigma2_series is not None and sigma2_series.ndim > 1 and sigma2_series.shape[-1] == 1:
        sigma2_series = np.squeeze(sigma2_series, axis=-1)

    lambda_key = "lambda" if "lambda" in posterior else ("lambda_" if "lambda_" in posterior else None)
    lambda_samples = None
    if lambda_key is not None:
        lambda_samples = _flatten_chain_samples(np.asarray(posterior[lambda_key], dtype=float))[trimmed_slice]
    else:
        _warn("Posterior samples missing 'lambda'; skipping shrinkage-specific diagnostics.")
    if lambda_samples is not None and lambda_samples.ndim > 1 and lambda_samples.shape[-1] == 1:
        lambda_samples = np.squeeze(lambda_samples, axis=-1)

    phi_samples = None
    if "phi" in posterior:
        phi_samples = _flatten_chain_samples(np.asarray(posterior["phi"], dtype=float))[trimmed_slice]
        if phi_samples.ndim > 1 and phi_samples.shape[-1] == 1:
            phi_samples = np.squeeze(phi_samples, axis=-1)

    dataset_arrays = artifacts.dataset_arrays
    X_train = dataset_arrays.get("X_train")
    if X_train is not None:
        X_train = np.asarray(X_train, dtype=float)
        p = X_train.shape[1]
    else:
        p = beta_samples.shape[1]
        _warn("dataset.npz missing X_train; group-aware diagnostics will be skipped.")
    groups = _resolve_groups(artifacts, p)
    group_index = _flatten_groups(groups, p)
    active_idx = artifacts.dataset_meta.get("active_idx") or artifacts.dataset_meta.get("strong_idx") or []

    slab_width = 1.0
    if artifacts.resolved_config:
        slab_width = float(artifacts.resolved_config.get("model", {}).get("c", slab_width))

    mean_kappa = None
    series = {}
    titles = {}
    if tau_samples is not None and tau_samples.size:
        series["tau"] = np.ravel(tau_samples)
        titles["tau"] = "Trace of global scale tau"
    if sigma2_series is not None and sigma2_series.size:
        series["sigma2"] = np.ravel(sigma2_series)
        titles["sigma2"] = "Trace of noise variance sigma2"
    phi_for_kappa = (
        phi_samples is not None
        and phi_samples.ndim == 2
        and phi_samples.shape[1] > int(np.max(group_index))
    )
    if (
        phi_for_kappa
        and lambda_samples is not None
        and tau_samples is not None
        and sigma_samples is not None
        and X_train is not None
    ):
        mean_kappa = compute_mean_kappa_series(
            X=X_train,
            group_index=group_index,
            lambda_samples=lambda_samples,
            tau_samples=tau_samples,
            phi_samples=phi_samples,
            sigma_samples=sigma_samples,
            slab_width=slab_width,
        )
        series["mean_kappa"] = mean_kappa
        titles["mean_kappa"] = "Mean shrinkage kappa"

    trace_fig = None
    acf_fig = None
    if series:
        trace_fig = trace_plot(series, burn_in=None, titles=titles)
        acf_series = {}
        if tau_samples is not None:
            acf_series["tau"] = autocorrelation(np.ravel(tau_samples), min(args.max_lag, tau_samples.size - 1))
        if sigma2_series is not None:
            acf_series["sigma2"] = autocorrelation(np.ravel(sigma2_series), min(args.max_lag, sigma2_series.size - 1))
        if mean_kappa is not None:
            acf_series["mean_kappa"] = autocorrelation(mean_kappa, min(args.max_lag, mean_kappa.size - 1))
        if acf_series:
            acf_fig = autocorrelation_plot(acf_series)


    truths = _extract_truths(artifacts)
    strong_idx = artifacts.dataset_meta.get("strong_idx", [])
    weak_idx = artifacts.dataset_meta.get("weak_idx", [])
    strong_sel: List[int] = _select_indices(strong_idx, args.strong_count)
    weak_sel: List[int] = _select_indices(weak_idx, args.weak_count)
    if not strong_sel:
        strong_sel = list(range(min(args.strong_count, p)))
    if not weak_sel:
        weak_sel = list(range(min(args.weak_count, p)))

    strong_fig = posterior_density_grid(
        beta_trimmed,
        indices=strong_sel,
        truths=truths,
        title_prefix="Strong",
    )
    weak_fig = posterior_density_grid(
        beta_trimmed,
        indices=weak_sel,
        truths=truths,
        title_prefix="Weak/Null",
    )

    phi_fig = None
    phi_ok = False
    if phi_samples is not None:
        if phi_samples.ndim == 2 and phi_samples.shape[1] == len(groups):
            phi_ok = True
            group_sizes = build_group_sizes(groups, p)
            phi_fig = phi_violin_plot(phi_samples, group_sizes=group_sizes, max_groups=args.groups_to_plot)
        else:
            _warn("phi samples do not align with groups; skipping group-level plots.")

    predictive_draws = None
    if X_train is not None:
        predictive_draws = prepare_predictive_draws(
            artifacts,
            rng_seed=args.rng_seed,
            burn_in=burn_in,
        )
    coverage_fig = None
    if predictive_draws is not None and "y_test" in dataset_arrays:
        y_test = np.asarray(dataset_arrays["y_test"], dtype=float)
        coverage_fig = coverage_width_plot(
            predictive_draws=predictive_draws,
            y_true=y_test,
            levels=args.coverage_levels,
        )

    dest = args.dest or (args.run_dir / "figures")
    dest = dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    if trace_fig is not None:
        trace_fig.savefig(dest / "trace_tau_sigma_kappa.png", dpi=args.dpi)
    if acf_fig is not None:
        acf_fig.savefig(dest / "acf_tau_sigma_kappa.png", dpi=args.dpi)
    strong_fig.savefig(dest / "posterior_density_strong.png", dpi=args.dpi)
    weak_fig.savefig(dest / "posterior_density_weak.png", dpi=args.dpi)
    if phi_fig is not None:
        phi_fig.savefig(dest / "group_phi_violin.png", dpi=args.dpi)
    if coverage_fig is not None:
        coverage_fig.savefig(dest / "coverage_width.png", dpi=args.dpi)

    beta_true = dataset_arrays.get("beta_true")
    X_plot = dataset_arrays.get("X_test")
    y_plot = dataset_arrays.get("y_test")
    if X_plot is None or y_plot is None or y_plot.size == 0:
        X_plot = dataset_arrays.get("X_train")
        y_plot = dataset_arrays.get("y_train")
    if X_plot is not None and y_plot is not None and beta_true is not None and beta_true.size:
        recon_fig = reconstruction_plot(
            X=np.asarray(X_plot, dtype=float),
            y_obs=np.asarray(y_plot, dtype=float),
            beta_samples=beta_trimmed,
            beta_true=np.asarray(beta_true, dtype=float),
            burn_in=0,
        )
        recon_fig.savefig(dest / "posterior_reconstruction.png", dpi=args.dpi)

    if phi_ok:
        landscape_fig = group_shrinkage_landscape(
            phi_samples=phi_samples,
            groups=groups,
            active_idx=active_idx,
        )
        landscape_fig.savefig(dest / "group_shrinkage_landscape.png", dpi=args.dpi)

        heatmap_fig = group_coefficient_heatmap(
            beta_samples=beta_trimmed,
            phi_samples=phi_samples,
            groups=groups,
            active_idx=active_idx,
        )
        heatmap_fig.savefig(dest / "group_coefficient_heatmap.png", dpi=args.dpi)

        if lambda_samples is not None and lambda_samples.ndim == 2 and lambda_samples.shape[1] == p:
            scatter_fig = group_vs_individual_scatter(
                phi_samples=phi_samples,
                lambda_samples=lambda_samples,
                groups=groups,
                active_idx=active_idx,
            )
            scatter_fig.savefig(dest / "group_vs_individual_scatter.png", dpi=args.dpi)
        elif lambda_samples is not None:
            _warn("lambda samples do not align with feature dimension; skipping group-vs-individual scatter.")

    print(f"[OK] Figures written to {dest}")


if __name__ == "__main__":
    main()
