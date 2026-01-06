"""CLI utility to visualize shrinkage structure for RHS and GRRHS runs."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from grrhs.viz.diagnostics import (
    RunArtifacts,
    _flatten_groups,
    group_interval_calibration_scatter,
    group_scale_vs_lambda_panel,
    load_run_artifacts,
    posterior_mean_heatmap_by_model,
    true_vs_estimated_panel,
)


@dataclass
class ShrinkageRunSummary:
    """Container holding trimmed posterior draws and metadata for plotting."""

    label: str
    run_dir: Path
    beta_samples: np.ndarray
    lambda_samples: np.ndarray
    tau_samples: np.ndarray
    beta_mean: np.ndarray
    groups: Tuple[Tuple[int, ...], ...]
    group_index: np.ndarray
    beta_true: np.ndarray
    active_idx: Tuple[int, ...]
    slab_width: float
    phi_samples: Optional[np.ndarray] = None


def _resolve_label(model_name: Optional[str]) -> str:
    if not model_name:
        return "Model"
    name = model_name.lower()
    if name in {"regularized_horseshoe", "rhs", "rhs_gibbs"}:
        return "RHS"
    if name in {"grrhs", "grrhs_gibbs"}:
        return "GRRHS"
    if name == "horseshoe":
        return "HS"
    return model_name


def _ensure_groups(artifacts: RunArtifacts, p: int) -> Tuple[Tuple[int, ...], ...]:
    groups = artifacts.dataset_meta.get("groups")
    if not groups:
        return tuple((i,) for i in range(p))
    return tuple(tuple(int(idx) for idx in grp) for grp in groups)


def _infer_burn_in(total_samples: int, requested: Optional[int], frac: float) -> int:
    if total_samples <= 0:
        raise ValueError("Total samples must be positive.")
    if requested is not None:
        burn = max(0, min(requested, total_samples - 1))
    else:
        burn = int(max(0.0, min(1.0, frac)) * total_samples)
        burn = min(burn, total_samples - 1)
    return burn


def _trim_samples(array: np.ndarray, burn_in: int) -> np.ndarray:
    if burn_in < 0 or burn_in >= array.shape[0]:
        burn_in = max(0, array.shape[0] - 1)
    return array[burn_in:]


def _summarize_run(artifacts: RunArtifacts, burn_in: int, label: Optional[str] = None) -> ShrinkageRunSummary:
    posterior = artifacts.posterior
    if not posterior:
        raise RuntimeError(f"No posterior samples found for run {artifacts.run_dir}.")
    if "beta" not in posterior:
        raise RuntimeError("Posterior samples must include 'beta' draws.")
    beta_samples = np.asarray(posterior["beta"], dtype=float)
    beta_trimmed = _trim_samples(beta_samples, burn_in)
    beta_mean = np.mean(beta_trimmed, axis=0)

    lambda_key = "lambda" if "lambda" in posterior else None
    if lambda_key is None:
        raise RuntimeError("Posterior samples must include 'lambda' draws for shrinkage diagnostics.")
    lambda_samples = np.asarray(posterior[lambda_key], dtype=float)
    lambda_trimmed = _trim_samples(lambda_samples, burn_in)

    if "tau" not in posterior:
        raise RuntimeError("Posterior samples must include 'tau' draws for scale normalization.")
    tau_samples = np.asarray(posterior["tau"], dtype=float)
    tau_trimmed = _trim_samples(tau_samples, burn_in)

    phi_samples = None
    if "phi" in posterior:
        phi_samples = np.asarray(posterior["phi"], dtype=float)
        phi_samples = _trim_samples(phi_samples, burn_in)

    dataset_arrays = artifacts.dataset_arrays
    if "beta_true" not in dataset_arrays:
        raise RuntimeError("dataset.npz must include 'beta_true' for shrinkage structure plots.")
    beta_true = np.asarray(dataset_arrays["beta_true"], dtype=float).reshape(-1)
    p = beta_true.size

    groups = _ensure_groups(artifacts, p)
    group_index = _flatten_groups(groups, p)
    active_idx = artifacts.dataset_meta.get("active_idx") or artifacts.dataset_meta.get("strong_idx") or []
    active_tuple = tuple(int(idx) for idx in active_idx)

    metrics_model = artifacts.metrics.get("model")
    resolved_model = artifacts.resolved_config.get("model", {}).get("name") if artifacts.resolved_config else None
    resolved_label = label or _resolve_label(metrics_model or resolved_model)
    slab_width = 1.0
    if artifacts.resolved_config:
        slab_width = float(artifacts.resolved_config.get("model", {}).get("c", slab_width))

    return ShrinkageRunSummary(
        label=resolved_label,
        run_dir=artifacts.run_dir,
        beta_samples=beta_trimmed,
        lambda_samples=lambda_trimmed,
        tau_samples=tau_trimmed,
        beta_mean=beta_mean,
        groups=groups,
        group_index=group_index,
        beta_true=beta_true,
        active_idx=active_tuple,
        slab_width=slab_width,
        phi_samples=phi_samples,
    )


def _common_output_dir(run_dirs: Sequence[Path]) -> Path:
    if len(run_dirs) == 1:
        return run_dirs[0] / "figures"
    parents = [rd.parent for rd in run_dirs]
    common_root = Path(os.path.commonpath([str(p) for p in parents]))
    return common_root / "comparison_figures"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create shrinkage structure visualizations for one or two experiment runs."
    )
    parser.add_argument("--run-dirs", nargs="+", required=True, type=Path, help="One or two run directories.")
    parser.add_argument("--dest", type=Path, default=None, help="Destination directory for generated figures.")
    parser.add_argument("--labels", nargs="+", default=None, help="Optional override labels for the runs.")
    parser.add_argument("--burn-in", type=int, default=None, help="Number of samples to discard as burn-in.")
    parser.add_argument(
        "--burn-in-frac",
        type=float,
        default=0.5,
        help="Fraction of samples to discard if --burn-in is not provided.",
    )
    parser.add_argument(
        "--level",
        type=float,
        default=0.9,
        help="Credible interval level used in coverage-width comparisons.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="DPI for saved figures.")
    args = parser.parse_args()

    run_dirs = [path.expanduser().resolve() for path in args.run_dirs]
    if len(run_dirs) not in {1, 2}:
        raise ValueError("Please provide one or two run directories.")

    artifacts: List[RunArtifacts] = [load_run_artifacts(path) for path in run_dirs]

    summaries: List[ShrinkageRunSummary] = []
    for idx, art in enumerate(artifacts):
        total_samples = int(np.asarray(art.posterior["beta"]).shape[0])
        burn_in = _infer_burn_in(total_samples, args.burn_in, args.burn_in_frac)
        label_override = None
        if args.labels and idx < len(args.labels):
            label_override = args.labels[idx]
        summaries.append(_summarize_run(art, burn_in, label_override))

    reference_groups = summaries[0].groups
    reference_beta_true = summaries[0].beta_true
    reference_group_index = summaries[0].group_index

    for summary in summaries[1:]:
        if summary.groups != reference_groups:
            raise ValueError("All runs must share the same group structure for comparison plots.")
        if not np.allclose(summary.beta_true, reference_beta_true):
            raise ValueError("All runs must share the same ground-truth coefficients.")
        if not np.array_equal(summary.group_index, reference_group_index):
            raise ValueError("All runs must share the same group index mapping.")

    dest = args.dest.expanduser().resolve() if args.dest else _common_output_dir(run_dirs)
    dest.mkdir(parents=True, exist_ok=True)

    labels = [summary.label for summary in summaries]
    beta_means = [summary.beta_mean for summary in summaries]
    lambda_samples = [summary.lambda_samples for summary in summaries]
    tau_samples = [summary.tau_samples for summary in summaries]
    phi_samples = [summary.phi_samples for summary in summaries]
    beta_samples = [summary.beta_samples for summary in summaries]
    active_idx = summaries[0].active_idx
    slab_widths = [summary.slab_width for summary in summaries]

    heatmap_fig = posterior_mean_heatmap_by_model(
        beta_means,
        labels=labels,
        groups=reference_groups,
        active_idx=active_idx,
    )
    scales_fig = group_scale_vs_lambda_panel(
        lambda_samples,
        labels=labels,
        groups=reference_groups,
        phi_samples=phi_samples,
        active_idx=active_idx,
        beta_true=reference_beta_true,
        tau_samples=tau_samples,
        slab_widths=slab_widths,
    )
    coverage_fig = group_interval_calibration_scatter(
        beta_samples,
        labels=labels,
        groups=reference_groups,
        beta_true=reference_beta_true,
        level=args.level,
    )
    truth_fig = true_vs_estimated_panel(
        beta_means,
        labels=labels,
        beta_true=reference_beta_true,
        group_index=reference_group_index,
        active_idx=active_idx,
    )

    figures = {
        "posterior_coefficients_heatmap.png": heatmap_fig,
        "shrinkage_scales_panel.png": scales_fig,
        "coverage_width_scatter.png": coverage_fig,
        "truth_vs_estimated.png": truth_fig,
    }

    for filename, fig in figures.items():
        output_path = dest / filename
        fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
