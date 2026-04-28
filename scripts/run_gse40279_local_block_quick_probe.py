from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from real_data_experiment.src.dataset import load_prepared_real_dataset, prepare_split
from real_data_experiment.src.evaluation import evaluate_method_result
from real_data_experiment.src.fitting import fit_real_data_methods
from real_data_experiment.src.reporting import (
    DEFAULT_REQUIRED_METRICS,
    build_paired_deltas,
    build_paired_summary,
    build_summary,
    default_dataset_group_cols,
)
from real_data_experiment.src.schemas import DatasetSpec
from simulation_second.src.schemas import ConvergenceGateSpec
from simulation_project.src.utils import load_pandas

OUT_DIR = PROJECT_ROOT / "outputs" / "history" / "real_data_experiment" / "quick_probes"


def _quick_dataset_spec(*, repeats: int) -> DatasetSpec:
    return DatasetSpec(
        dataset_id="gse40279_age_local_block_micro_gap1000_quick_probe",
        label="GSE40279 Methylation Age (micro, local block gap1000, quick 5-method probe)",
        description="Fast repeated five-method probe on the local methylation block grouped variant.",
        loader={
            "path_X": "data/real/gse40279_methylation_age/processed/runner_ready_micro_local_block_gap1000/X.npy",
            "path_y": "data/real/gse40279_methylation_age/processed/runner_ready_micro_local_block_gap1000/y.npy",
            "path_feature_names": "data/real/gse40279_methylation_age/processed/runner_ready_micro_local_block_gap1000/feature_names.txt",
            "path_group_map": "data/real/gse40279_methylation_age/processed/runner_ready_micro_local_block_gap1000/group_map.json",
            "path_group_labels": "data/real/gse40279_methylation_age/processed/runner_ready_micro_local_block_gap1000/group_labels.txt",
        },
        task="gaussian",
        methods=("GR_RHS", "RHS", "GHS_plus", "LASSO_CV", "OLS"),
        target_label="chronological_age",
        target_transform="none",
        response_standardization="train_center",
        covariate_mode="none",
        p0_strategy="sqrt_p",
        p0_groups_strategy="half_groups",
        test_fraction=0.2,
        repeats=int(repeats),
        shuffle=True,
        notes=(
            "Quick repeated probe only. No formal convergence enforcement. "
            "GIGG_MMLE excluded for runtime reasons."
        ),
    )


def _quick_gate(*, warmup: int, draws: int) -> ConvergenceGateSpec:
    return ConvergenceGateSpec(
        enforce_bayes_convergence=False,
        max_convergence_retries=0,
        bayes_min_chains=1,
        chains=1,
        warmup=int(warmup),
        post_warmup_draws=int(draws),
        adapt_delta=0.90,
        max_treedepth=8,
        strict_adapt_delta=0.92,
        strict_max_treedepth=9,
        rhat_threshold=2.5,
        ess_threshold=5.0,
        max_divergence_ratio=1.0,
    )


def _heuristic_p0(spec: DatasetSpec, p: int) -> int:
    if spec.p0_override is not None:
        return max(1, min(int(spec.p0_override), int(p)))
    if str(spec.p0_strategy).strip().lower() == "sqrt_p":
        return max(1, min(int(np.ceil(np.sqrt(max(int(p), 1)))), int(p)))
    if str(spec.p0_strategy).strip().lower() == "half_p":
        return max(1, min(int(np.ceil(max(int(p), 1) / 2.0)), int(p)))
    return max(1, min(int(np.ceil(np.sqrt(max(int(p), 1)))), int(p)))


def _heuristic_p0_groups(spec: DatasetSpec, n_groups: int) -> int:
    if spec.p0_groups_override is not None:
        return max(1, min(int(spec.p0_groups_override), int(n_groups)))
    if str(spec.p0_groups_strategy).strip().lower() == "half_groups":
        return max(1, min(int(np.ceil(max(int(n_groups), 1) / 2.0)), int(n_groups)))
    if str(spec.p0_groups_strategy).strip().lower() == "sqrt_groups":
        return max(1, min(int(np.ceil(np.sqrt(max(int(n_groups), 1)))), int(n_groups)))
    return max(1, min(int(np.ceil(max(int(n_groups), 1) / 2.0)), int(n_groups)))


def _build_row(
    *,
    spec: DatasetSpec,
    split,
    method: str,
    result,
    eval_row: dict[str, object],
    p0: int,
    p0_groups: int,
) -> dict[str, object]:
    return {
        "dataset_id": str(spec.dataset_id),
        "dataset_label": str(spec.label),
        "description": str(spec.description),
        "task": str(spec.task),
        "target_label": str(spec.target_label),
        "target_transform": str(spec.target_transform),
        "covariate_mode": str(spec.covariate_mode),
        "response_standardization": str(spec.response_standardization),
        "notes": str(spec.notes),
        "replicate_id": int(split.replicate_id),
        "seed": int(split.seed),
        "split_hash": str(split.split_hash),
        "sample_count": int(split.dataset.X.shape[0]),
        "feature_count": int(split.dataset.X.shape[1]),
        "covariate_count": int(split.dataset.covariates.shape[1]) if split.dataset.covariates is not None else 0,
        "group_count": int(len(split.groups)),
        "group_sizes_json": json.dumps([int(len(group)) for group in split.groups]),
        "group_labels_json": json.dumps(list(split.dataset.group_labels)),
        "n_train": int(split.train_idx.size),
        "n_test": int(split.test_idx.size),
        "p0_estimated": int(p0),
        "p0_groups_estimated": int(p0_groups),
        "method": str(method),
        "method_label": str(eval_row["method_label"]),
        "method_type": str(eval_row["method_type"]),
        "status": str(result.status),
        "converged": bool(result.converged),
        "error": str(result.error),
        "runtime_seconds": float(result.runtime_seconds),
        "rhat_max": float(result.rhat_max),
        "bulk_ess_min": float(result.bulk_ess_min),
        "divergence_ratio": float(result.divergence_ratio),
        **{key: value for key, value in eval_row.items() if key not in {"method_label", "method_type"}},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repeated quick probe for GSE40279 local methylation blocks.")
    parser.add_argument("--repeats", type=int, default=10, help="Number of repeated train/test splits to run.")
    parser.add_argument("--seed", type=int, default=20260427, help="Master seed for split generation.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup draws for Bayesian quick probe methods.")
    parser.add_argument("--draws", type=int, default=10, help="Post-warmup draws for Bayesian quick probe methods.")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional subset of methods, e.g. --methods GR_RHS RHS.",
    )
    parser.add_argument(
        "--grrhs-budget-scale",
        type=float,
        default=0.03,
        help="Exploratory GR_RHS Gibbs budget scale for faster repeated probing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pd = load_pandas()
    spec = _quick_dataset_spec(repeats=int(args.repeats))
    gate = _quick_gate(warmup=int(args.warmup), draws=int(args.draws))
    prepared = load_prepared_real_dataset(spec)
    methods = list(spec.methods)
    if args.methods:
        wanted = {str(item) for item in args.methods}
        methods = [method for method in methods if method in wanted]
    if not methods:
        raise SystemExit("No methods selected.")
    rows: list[dict[str, object]] = []

    print(
        f"running_quick_probe dataset={spec.dataset_id} repeats={int(args.repeats)} "
        f"methods={','.join(methods)} warmup={int(args.warmup)} draws={int(args.draws)}",
        flush=True,
    )

    for replicate_id in range(1, int(args.repeats) + 1):
        split = prepare_split(prepared, replicate_id=replicate_id, master_seed=int(args.seed))
        p0 = _heuristic_p0(spec, split.X_train_used.shape[1])
        p0_groups = _heuristic_p0_groups(spec, len(split.groups))
        print(
            f"replicate_start rep={replicate_id} split_hash={split.split_hash} "
            f"n_train={int(split.train_idx.size)} n_test={int(split.test_idx.size)}",
            flush=True,
        )
        results = fit_real_data_methods(
            split.X_train_used,
            split.y_train_used,
            split.groups,
            task=str(spec.task),
            seed=int(split.seed) + 17,
            p0=int(p0),
            p0_groups=int(p0_groups),
            methods=methods,
            gate=gate,
            grrhs_kwargs={
                "tau_target": "groups",
                "progress_bar": False,
                "gibbs_budget_scale": float(args.grrhs_budget_scale),
            },
            gigg_config={"allow_budget_retry": True, "extra_retry": 0, "no_retry": True},
            method_jobs=1,
        )
        for method in methods:
            result = results[method]
            eval_row = evaluate_method_result(result, split)
            row = _build_row(
                spec=spec,
                split=split,
                method=method,
                result=result,
                eval_row=eval_row,
                p0=int(p0),
                p0_groups=int(p0_groups),
            )
            rows.append(row)
            print(
                f"replicate_method rep={replicate_id} method={method} status={row['status']} "
                f"converged={row['converged']} rmse={float(row['rmse_test']):.4f} "
                f"cov95={float(row['pred_coverage_95']):.4f} len95={float(row['avg_pred_interval_length_95']):.4f} "
                f"runtime={float(row['runtime_seconds']):.2f}s",
                flush=True,
            )

    raw = pd.DataFrame(rows)
    if not raw.empty:
        raw = raw.sort_values(["replicate_id", "method"], kind="stable").reset_index(drop=True)
    group_cols = default_dataset_group_cols(raw)
    summary = build_summary(
        raw,
        group_cols=group_cols,
        method_order=methods,
        required_metric_cols=DEFAULT_REQUIRED_METRICS,
    )
    paired_raw, paired_stats, summary_paired = build_paired_summary(
        raw,
        group_cols=group_cols,
        method_levels=methods,
        required_metric_cols=DEFAULT_REQUIRED_METRICS,
        method_order=methods,
    )
    paired_deltas = build_paired_deltas(
        paired_raw,
        group_cols=group_cols,
        baseline_method="RHS",
    )

    summary_cols = [
        "method",
        "n_runs",
        "n_ok",
        "n_converged",
        "n_summary_reps",
        "rmse_test",
        "mae_test",
        "r2_test",
        "lpd_test",
        "pred_coverage_90",
        "pred_coverage_95",
        "avg_pred_interval_length_90",
        "avg_pred_interval_length_95",
        "runtime_mean",
    ]
    summary_view = summary[[col for col in summary_cols if col in summary.columns]].copy()
    summary_view = summary_view.sort_values(["rmse_test", "method"], kind="stable").reset_index(drop=True)

    run_dir = OUT_DIR / f"gse40279_local_block_quick_probe_r{int(args.repeats)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(run_dir / "raw_results.csv", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)
    summary_paired.to_csv(run_dir / "summary_paired.csv", index=False)
    paired_stats.to_csv(run_dir / "paired_replicate_stats.csv", index=False)
    paired_deltas.to_csv(run_dir / "paired_deltas.csv", index=False)
    summary_view.to_csv(run_dir / "coverage_focus_summary.csv", index=False)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": spec.dataset_id,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "repeats": int(args.repeats),
                "seed": int(args.seed),
                "methods": methods,
                "gate": asdict(gate),
                "grrhs_budget_scale": float(args.grrhs_budget_scale),
                "result_files": {
                    "raw_results": "raw_results.csv",
                    "summary": "summary.csv",
                    "summary_paired": "summary_paired.csv",
                    "paired_replicate_stats": "paired_replicate_stats.csv",
                    "paired_deltas": "paired_deltas.csv",
                    "coverage_focus_summary": "coverage_focus_summary.csv",
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(summary_view.to_string(index=False))
    print(str(run_dir.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
