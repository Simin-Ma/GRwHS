from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from real_data_experiment.src.dataset import load_prepared_real_dataset
from real_data_experiment.src.schemas import DatasetSpec
from simulation_project.src.experiments.methods.fit_gr_rhs import fit_gr_rhs
from simulation_project.src.utils import SamplerConfig


OUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "gse40279_local_block_micro_grrhs"
ANALYSIS_DIR = (
    PROJECT_ROOT
    / "data"
    / "real"
    / "gse40279_methylation_age"
    / "processed"
    / "analysis_bundle"
)
RUNNER_DIR = (
    PROJECT_ROOT
    / "data"
    / "real"
    / "gse40279_methylation_age"
    / "processed"
    / "runner_ready_micro_local_block_gap1000"
)


GROUP_COLORS = {
    0: "#8b5e3c",
    1: "#2f6b7c",
    2: "#4d6a2f",
    3: "#8c3b3b",
    4: "#5a4fcf",
    5: "#9a7a13",
}


def _dataset_spec() -> DatasetSpec:
    return DatasetSpec(
        dataset_id="gse40279_age_local_block_micro_gap1000_forest",
        label="GSE40279 Methylation Age (micro, local block gap1000)",
        description="Posterior forest plot fit on the local-block micro methylation-age dataset.",
        loader={
            "path_X": "data/real/gse40279_methylation_age/processed/runner_ready_micro_local_block_gap1000/X.npy",
            "path_y": "data/real/gse40279_methylation_age/processed/runner_ready_micro_local_block_gap1000/y.npy",
            "path_feature_names": "data/real/gse40279_methylation_age/processed/runner_ready_micro_local_block_gap1000/feature_names.txt",
            "path_group_map": "data/real/gse40279_methylation_age/processed/runner_ready_micro_local_block_gap1000/group_map.json",
            "path_group_labels": "data/real/gse40279_methylation_age/processed/runner_ready_micro_local_block_gap1000/group_labels.txt",
        },
        task="gaussian",
        methods=("GR_RHS",),
        target_label="chronological_age",
        target_transform="none",
        response_standardization="train_center",
        covariate_mode="none",
        p0_strategy="sqrt_p",
        p0_groups_strategy="half_groups",
        repeats=1,
        shuffle=True,
    )


def _standardize_columns(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(X, dtype=float)
    centered = arr - np.mean(arr, axis=0, keepdims=True)
    scale = np.std(centered, axis=0, ddof=0, keepdims=True)
    scale = np.where(scale < 1e-10, 1.0, scale)
    return centered / scale, scale.reshape(-1)


def _center_vector(y: np.ndarray) -> tuple[np.ndarray, float]:
    arr = np.asarray(y, dtype=float).reshape(-1)
    offset = float(np.mean(arr))
    return arr - offset, offset


def _load_feature_metadata() -> dict[str, dict[str, object]]:
    meta_path = ANALYSIS_DIR / "micro_local_block_gap1000_selected_features.tsv"
    out: dict[str, dict[str, object]] = {}
    with meta_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            probe_id = str(row["probe_id"])
            out[probe_id] = {
                "variance": float(row["variance"]),
                "chromosome": str(row["chromosome"]),
                "position": int(row["position"]),
                "group_label": str(row["group_label"]),
                "anchor_gene": str(row["anchor_gene"]),
                "group_id": int(row["group_id"]),
            }
    return out


def _fit_grrhs(
    X: np.ndarray,
    y: np.ndarray,
    groups: list[list[int]],
    *,
    seed: int,
    chains: int,
    warmup: int,
    draws: int,
    gibbs_budget_scale: float,
):
    sampler = SamplerConfig(
        chains=int(chains),
        warmup=int(warmup),
        post_warmup_draws=int(draws),
        adapt_delta=0.90,
        max_treedepth=10,
        strict_adapt_delta=0.95,
        strict_max_treedepth=12,
        max_divergence_ratio=0.05,
        rhat_threshold=1.05,
        ess_threshold=50.0,
    )
    p0_groups = max(1, int(np.ceil(len(groups) / 2.0)))
    return fit_gr_rhs(
        X,
        y,
        groups,
        task="gaussian",
        seed=int(seed),
        p0=int(p0_groups),
        sampler=sampler,
        tau_target="groups",
        progress_bar=False,
        gibbs_budget_scale=float(gibbs_budget_scale),
    )


def _flatten_draws(draws: np.ndarray) -> np.ndarray:
    arr = np.asarray(draws, dtype=float)
    if arr.ndim == 3:
        return arr.reshape(-1, arr.shape[-1])
    if arr.ndim == 2:
        return arr
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    raise ValueError(f"Unexpected beta draw shape: {arr.shape}")


def _display_label(feature_name: str, meta: dict[str, object]) -> str:
    gene = str(meta.get("anchor_gene", "")).strip()
    chrom = str(meta.get("chromosome", "")).strip()
    pos = int(meta.get("position", 0))
    if gene:
        return f"{feature_name} | {gene} | chr{chrom}:{pos}"
    return f"{feature_name} | chr{chrom}:{pos}"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_top_features(
    *,
    rows: list[dict[str, object]],
    png_path: Path,
    pdf_path: Path,
    top_k: int,
) -> None:
    ranked = sorted(rows, key=lambda r: abs(float(r["posterior_mean"])), reverse=True)[: int(top_k)]
    ranked = list(reversed(ranked))
    labels = [str(item["display_label"]) for item in ranked]
    means = np.asarray([float(item["posterior_mean"]) for item in ranked], dtype=float)
    lowers = np.asarray([float(item["ci_lower_95"]) for item in ranked], dtype=float)
    uppers = np.asarray([float(item["ci_upper_95"]) for item in ranked], dtype=float)
    group_ids = [int(item["group_id"]) for item in ranked]

    y_pos = np.arange(len(ranked))
    fig_h = max(10.0, len(ranked) * 0.34 + 2.0)
    fig, ax = plt.subplots(figsize=(12.5, fig_h), constrained_layout=True)
    fig.patch.set_facecolor("#fbfaf6")
    ax.set_facecolor("#fffdfa")

    for idx, (mean, lo, hi, gid) in enumerate(zip(means, lowers, uppers, group_ids)):
        color = GROUP_COLORS.get(int(gid), "#374151")
        ax.hlines(idx, lo, hi, color=color, lw=2.0, alpha=0.95, zorder=2)
        ax.scatter(mean, idx, s=34, color=color, edgecolor="#111827", linewidth=0.4, zorder=3)

    ax.axvline(0.0, color="#111827", lw=1.2, alpha=0.85, zorder=1)
    ax.grid(axis="x", color="#e7e5e4", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.6)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_xlabel("Posterior mean age effect (years) per +1 SD CpG methylation", fontsize=11)
    ax.set_title("GSE40279 Local-Block Micro: GR-RHS Posterior Means and 95% Credible Intervals", fontsize=14, pad=12)
    ax.text(
        0.0,
        1.01,
        f"Top {len(ranked)} CpGs by |posterior mean|; fitted on all 656 samples with centered age and standardized CpGs",
        transform=ax.transAxes,
        fontsize=9.2,
        color="#44403c",
        ha="left",
    )

    xmin = float(np.min(lowers))
    xmax = float(np.max(uppers))
    pad = max(0.25, 0.08 * (xmax - xmin if xmax > xmin else 1.0))
    ax.set_xlim(xmin - pad, xmax + pad)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#78716c")

    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Posterior forest plot for GSE40279 local-block micro GR-RHS.")
    parser.add_argument("--seed", type=int, default=20260427, help="Model seed.")
    parser.add_argument("--chains", type=int, default=2, help="Number of chains.")
    parser.add_argument("--warmup", type=int, default=120, help="Warmup iterations per chain.")
    parser.add_argument("--draws", type=int, default=120, help="Posterior draws per chain.")
    parser.add_argument(
        "--gibbs-budget-scale",
        type=float,
        default=0.05,
        help="Exploratory staged-Gibbs budget scale for GR_RHS.",
    )
    parser.add_argument("--top-k", type=int, default=40, help="How many top CpGs to show in the forest plot.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    spec = _dataset_spec()
    prepared = load_prepared_real_dataset(spec)
    feature_meta = _load_feature_metadata()

    X_used, x_scale = _standardize_columns(prepared.X)
    y_used, y_offset = _center_vector(prepared.y)
    groups = [[int(idx) for idx in group] for group in prepared.groups]

    result = _fit_grrhs(
        X_used,
        y_used,
        groups,
        seed=int(args.seed),
        chains=int(args.chains),
        warmup=int(args.warmup),
        draws=int(args.draws),
        gibbs_budget_scale=float(args.gibbs_budget_scale),
    )
    if result.beta_draws is None or result.beta_mean is None:
        raise RuntimeError(f"GR_RHS fit did not return posterior draws: {result.error}")

    beta_draws = _flatten_draws(np.asarray(result.beta_draws, dtype=float))
    beta_mean = np.asarray(result.beta_mean, dtype=float).reshape(-1)
    lower = np.quantile(beta_draws, 0.025, axis=0)
    upper = np.quantile(beta_draws, 0.975, axis=0)
    posterior_sd = np.std(beta_draws, axis=0, ddof=1) if beta_draws.shape[0] > 1 else np.zeros_like(beta_mean)

    rows: list[dict[str, object]] = []
    for idx, feature_name in enumerate(prepared.feature_names):
        meta = feature_meta.get(
            str(feature_name),
            {
                "variance": float("nan"),
                "chromosome": "",
                "position": -1,
                "group_label": prepared.group_labels[int(prepared.groups and next(
                    (gid for gid, group in enumerate(prepared.groups) if idx in group),
                    0,
                ))],
                "anchor_gene": "",
                "group_id": int(next((gid for gid, group in enumerate(prepared.groups) if idx in group), 0)),
            },
        )
        gid = int(meta["group_id"])
        rows.append(
            {
                "feature_rank_abs_mean": 0,
                "feature_name": str(feature_name),
                "display_label": _display_label(str(feature_name), meta),
                "posterior_mean": float(beta_mean[idx]),
                "posterior_sd": float(posterior_sd[idx]),
                "ci_lower_95": float(lower[idx]),
                "ci_upper_95": float(upper[idx]),
                "abs_posterior_mean": float(abs(beta_mean[idx])),
                "group_id": int(gid),
                "group_label": str(meta["group_label"]),
                "anchor_gene": str(meta["anchor_gene"]),
                "chromosome": str(meta["chromosome"]),
                "position": int(meta["position"]),
                "variance": float(meta["variance"]),
                "x_scale_sd": float(x_scale[idx]),
            }
        )

    rows.sort(key=lambda item: abs(float(item["posterior_mean"])), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["feature_rank_abs_mean"] = int(rank)

    csv_path = OUT_DIR / "gse40279_local_block_micro_grrhs_posterior_summary.csv"
    png_path = OUT_DIR / "gse40279_local_block_micro_grrhs_posterior_forest.png"
    pdf_path = OUT_DIR / "gse40279_local_block_micro_grrhs_posterior_forest.pdf"
    json_path = OUT_DIR / "gse40279_local_block_micro_grrhs_fit_summary.json"

    _write_csv(csv_path, rows)
    _plot_top_features(rows=rows, png_path=png_path, pdf_path=pdf_path, top_k=int(args.top_k))

    fit_summary = {
        "method": "GR_RHS",
        "dataset_id": spec.dataset_id,
        "dataset_label": spec.label,
        "n": int(X_used.shape[0]),
        "p": int(X_used.shape[1]),
        "group_count": int(len(groups)),
        "group_sizes": [int(len(group)) for group in groups],
        "target_label": spec.target_label,
        "response_preprocessing": {
            "y_offset_mean_age": float(y_offset),
            "y_transform": "center only",
            "X_transform": "column standardize on full sample",
        },
        "fit": {
            "status": str(result.status),
            "converged": bool(result.converged),
            "runtime_seconds": float(result.runtime_seconds),
            "rhat_max": float(result.rhat_max),
            "bulk_ess_min": float(result.bulk_ess_min),
            "divergence_ratio": float(result.divergence_ratio),
            "chains": int(args.chains),
            "warmup": int(args.warmup),
            "draws": int(args.draws),
            "gibbs_budget_scale": float(args.gibbs_budget_scale),
            "seed": int(args.seed),
        },
        "artifacts": {
            "forest_png": str(png_path),
            "forest_pdf": str(pdf_path),
            "posterior_summary_csv": str(csv_path),
        },
        "top_10_features": [
            {
                "feature_name": row["feature_name"],
                "posterior_mean": row["posterior_mean"],
                "ci_lower_95": row["ci_lower_95"],
                "ci_upper_95": row["ci_upper_95"],
                "group_label": row["group_label"],
            }
            for row in rows[:10]
        ],
        "diagnostics": result.diagnostics,
    }
    json_path.write_text(json.dumps(fit_summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(str(png_path))
    print(str(pdf_path))
    print(str(csv_path))
    print(str(json_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
