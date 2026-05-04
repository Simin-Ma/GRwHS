from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from ..config import load_real_data_config
from ..dataset import load_prepared_real_dataset
from ..runner import finalize_real_data_results_dir, run_real_data_experiment
from ..table_builder import build_paper_tables_from_results_dir
from ..utils import save_json


def _print_datasets(datasets: Iterable[object]) -> None:
    for dataset in datasets:
        print(f"{dataset.dataset_id}: {dataset.label}")


def _filtered_config(config, *, dataset_ids: list[str] | None = None, methods: list[str] | None = None):
    out = config
    if dataset_ids:
        wanted = {str(item) for item in dataset_ids}
        out = replace(out, datasets=tuple(dataset for dataset in out.datasets if dataset.dataset_id in wanted))
    if methods:
        out = replace(out, methods=replace(out.methods, roster=tuple(str(item) for item in methods)))
        out = replace(
            out,
            datasets=tuple(
                replace(dataset, methods=tuple(method for method in dataset.methods if method in set(methods)))
                for dataset in out.datasets
            ),
        )
    return out


def _override_runner_from_args(config, args: argparse.Namespace):
    out = _filtered_config(config, dataset_ids=getattr(args, "datasets", None), methods=getattr(args, "methods", None))
    runner = out.runner
    if getattr(args, "save_dir", ""):
        runner = replace(runner, output_dir=str(args.save_dir))
    if getattr(args, "seed", None) is not None:
        runner = replace(runner, seed=int(args.seed))
    if getattr(args, "n_jobs", None) is not None:
        runner = replace(runner, n_jobs=int(args.n_jobs))
    if getattr(args, "method_jobs", None) is not None:
        runner = replace(runner, method_jobs=int(args.method_jobs))
    if getattr(args, "no_build_tables", False):
        runner = replace(runner, build_tables=False)
    if getattr(args, "no_save_splits", False):
        runner = replace(runner, save_splits=False)
    if getattr(args, "baseline_method", ""):
        runner = replace(runner, baseline_method=str(args.baseline_method))
    return replace(out, runner=runner)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the real-data GR-RHS comparison pipeline.")
    parser.add_argument(
        "--config",
        default="",
        help="Optional YAML config. Defaults to real_data_experiment/config/real_data.yaml if present.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list-datasets", help="List configured real datasets.")

    dump = sub.add_parser("dump-manifest", help="Export the loaded real-data manifest as JSON.")
    dump.add_argument("--save-path", default="", help="Optional JSON path to write the manifest.")

    describe = sub.add_parser("describe-dataset", help="Load and summarize one dataset.")
    describe.add_argument("--dataset-id", required=True, help="Dataset identifier from list-datasets.")

    run = sub.add_parser("run-real-data", help="Run the full real-data comparison pipeline.")
    run.add_argument("--save-dir", default="", help="Optional override for the output history directory.")
    run.add_argument("--seed", type=int, default=None, help="Override master seed.")
    run.add_argument("--n-jobs", type=int, default=None, help="Number of replicate workers.")
    run.add_argument("--method-jobs", type=int, default=None, help="Number of per-task method workers.")
    run.add_argument("--methods", nargs="*", default=None, help="Optional subset of methods.")
    run.add_argument("--datasets", nargs="*", default=None, help="Optional subset of dataset ids.")
    run.add_argument("--baseline-method", default="", help="Optional baseline method for paired deltas.")
    run.add_argument("--no-build-tables", action="store_true", help="Skip paper-table generation.")
    run.add_argument("--no-save-splits", action="store_true", help="Skip persisted split artifacts.")

    tables = sub.add_parser("build-tables", help="Rebuild paper tables from an existing results directory.")
    tables.add_argument(
        "--results-dir",
        default="outputs/history/real_data_experiment/main",
        help="Directory containing raw_results.csv or a history root with latest_run.json.",
    )

    finalize = sub.add_parser(
        "finalize-results",
        help="Build CSV summaries and paper tables from an existing run directory, including incremental JSONL checkpoints.",
    )
    finalize.add_argument(
        "--results-dir",
        required=True,
        help="Run directory containing raw_results.csv or raw_results_incremental.jsonl.",
    )
    finalize.add_argument("--baseline-method", default="", help="Optional baseline method for paired deltas.")
    finalize.add_argument("--no-build-tables", action="store_true", help="Skip paper-table generation.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_real_data_config(args.config or None)

    if args.command == "list-datasets":
        _print_datasets(config.datasets)
        return 0

    if args.command == "dump-manifest":
        manifest = config.to_manifest()
        if str(args.save_path).strip():
            save_json(manifest, Path(args.save_path))
            print(Path(args.save_path))
        else:
            print(json.dumps(manifest, indent=2))
        return 0

    if args.command == "describe-dataset":
        dataset_map = config.dataset_map()
        if args.dataset_id not in dataset_map:
            parser.error(f"Unknown dataset-id: {args.dataset_id!r}")
        prepared = load_prepared_real_dataset(dataset_map[args.dataset_id])
        print(json.dumps(prepared.to_summary(), indent=2))
        return 0

    if args.command == "run-real-data":
        run_config = _override_runner_from_args(config, args)
        if not run_config.datasets:
            parser.error("No datasets selected for run-real-data.")
        if not run_config.methods.roster:
            parser.error("No methods selected for run-real-data.")
        result = run_real_data_experiment(run_config)
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "build-tables":
        result = build_paper_tables_from_results_dir(
            args.results_dir,
            method_order=config.methods.roster,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "finalize-results":
        result = finalize_real_data_results_dir(
            args.results_dir,
            method_order=config.methods.roster,
            baseline_method=str(args.baseline_method or config.runner.baseline_method),
            required_metrics_for_pairing=config.runner.required_metrics_for_pairing,
            build_tables=not bool(args.no_build_tables),
        )
        print(json.dumps(result, indent=2))
        return 0

    parser.error(f"Unknown command: {args.command!r}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
