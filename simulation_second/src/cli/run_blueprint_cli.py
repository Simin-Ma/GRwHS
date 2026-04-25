from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from ..config import BenchmarkConfig, load_benchmark_config
from ..dataset import generate_grouped_dataset, save_grouped_dataset
from ..runner import run_benchmark
from ..table_builder import build_paper_tables_from_results_dir
from ..utils import prepare_history_run_dir, save_json, write_history_run_index


def _print_settings(settings: Iterable[object]) -> None:
    for setting in settings:
        print(f"{setting.setting_id}: {setting.label}")


def _filtered_config(config: BenchmarkConfig, *, setting_ids: list[str] | None = None, methods: list[str] | None = None) -> BenchmarkConfig:
    out = config
    if setting_ids:
        wanted = {str(item) for item in setting_ids}
        out = replace(out, settings=tuple(setting for setting in out.settings if setting.setting_id in wanted))
    if methods:
        out = replace(out, methods=replace(out.methods, roster=tuple(str(item) for item in methods)))
    return out


def _override_n_test(config: BenchmarkConfig, n_test: int | None) -> BenchmarkConfig:
    if n_test is None:
        return config
    settings = tuple(replace(setting, n_test=int(n_test)) for setting in config.settings)
    return replace(config, settings=settings)


def _override_runner_from_args(config: BenchmarkConfig, args: argparse.Namespace) -> BenchmarkConfig:
    out = _filtered_config(config, setting_ids=getattr(args, "settings", None), methods=getattr(args, "methods", None))
    out = _override_n_test(out, getattr(args, "n_test", None))

    runner = out.runner
    if getattr(args, "save_dir", ""):
        runner = replace(runner, output_dir=str(args.save_dir))
    if getattr(args, "repeats", None) is not None:
        runner = replace(runner, repeats=int(args.repeats))
    if getattr(args, "seed", None) is not None:
        runner = replace(runner, seed=int(args.seed))
    if getattr(args, "n_jobs", None) is not None:
        runner = replace(runner, n_jobs=int(args.n_jobs))
    if getattr(args, "method_jobs", None) is not None:
        runner = replace(runner, method_jobs=int(args.method_jobs))
    if getattr(args, "save_datasets", False):
        runner = replace(runner, save_datasets=True)
    if getattr(args, "no_build_tables", False):
        runner = replace(runner, build_tables=False)
    return replace(out, runner=runner)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the second-generation GR-RHS benchmark utilities."
    )
    parser.add_argument(
        "--config",
        default="",
        help="Optional YAML benchmark config. Defaults to simulation_second/config/benchmark.yaml.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list-settings", help="List benchmark settings.")
    list_parser.add_argument("--n-test", type=int, default=None, help="Optional override for test sample size.")

    dump = sub.add_parser("dump-manifest", help="Export the loaded benchmark manifest as JSON.")
    dump.add_argument("--save-path", default="", help="Optional JSON path to write the manifest.")
    dump.add_argument("--n-test", type=int, default=None, help="Optional override for test sample size.")

    sample = sub.add_parser("sample-setting", help="Generate one train/test dataset for a setting.")
    sample.add_argument("--setting-id", required=True, help="Setting identifier from list-settings.")
    sample.add_argument("--replicate", type=int, default=1, help="Replicate index.")
    sample.add_argument("--seed", type=int, default=None, help="Master seed override.")
    sample.add_argument("--n-test", type=int, default=None, help="Override test sample size.")
    sample.add_argument(
        "--save-dir",
        default="outputs/history/simulation_second/samples",
        help="Directory used to store the sampled dataset.",
    )

    suite = sub.add_parser("sample-suite", help="Generate datasets for the whole benchmark suite.")
    suite.add_argument("--repeats", type=int, default=1, help="Number of replicates per setting.")
    suite.add_argument("--seed", type=int, default=None, help="Master seed override.")
    suite.add_argument("--n-test", type=int, default=None, help="Override test sample size.")
    suite.add_argument("--settings", nargs="*", default=None, help="Optional subset of setting ids.")
    suite.add_argument(
        "--save-dir",
        default="outputs/history/simulation_second/suite_samples",
        help="Directory used to store the sampled datasets.",
    )

    run = sub.add_parser("run-benchmark", help="Run the full benchmark pipeline from config to tables.")
    run.add_argument("--save-dir", default="", help="Optional override for the benchmark output directory.")
    run.add_argument("--repeats", type=int, default=None, help="Override repeats.")
    run.add_argument("--seed", type=int, default=None, help="Override master seed.")
    run.add_argument("--n-test", type=int, default=None, help="Override test sample size.")
    run.add_argument("--n-jobs", type=int, default=None, help="Number of replicate workers.")
    run.add_argument("--method-jobs", type=int, default=None, help="Number of per-task method workers.")
    run.add_argument("--methods", nargs="*", default=None, help="Optional subset of methods.")
    run.add_argument("--settings", nargs="*", default=None, help="Optional subset of setting ids.")
    run.add_argument("--save-datasets", action="store_true", help="Persist sampled datasets alongside the run.")
    run.add_argument("--no-build-tables", action="store_true", help="Skip paper-table generation.")

    tables = sub.add_parser("build-tables", help="Rebuild paper tables from an existing results directory.")
    tables.add_argument(
        "--results-dir",
        default="outputs/history/simulation_second/benchmark_main",
        help="Directory containing raw_results.csv.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_benchmark_config(args.config or None)

    if args.command == "list-settings":
        config = _override_n_test(config, args.n_test)
        _print_settings(config.settings)
        return 0

    if args.command == "dump-manifest":
        config = _override_n_test(config, args.n_test)
        manifest = config.to_manifest()
        if str(args.save_path).strip():
            save_json(manifest, Path(args.save_path))
            print(Path(args.save_path))
        else:
            print(json.dumps(manifest, indent=2))
        return 0

    if args.command == "sample-setting":
        config = _override_n_test(config, args.n_test)
        setting_map = config.setting_map()
        if args.setting_id not in setting_map:
            parser.error(f"Unknown setting-id: {args.setting_id!r}")
        runner_seed = int(args.seed) if args.seed is not None else int(config.runner.seed)
        setting = setting_map[args.setting_id]
        dataset = generate_grouped_dataset(
            setting,
            replicate_id=int(args.replicate),
            master_seed=runner_seed,
            family_specs=config.families,
        )
        history_root, out_dir, run_timestamp = prepare_history_run_dir(args.save_dir)
        produced = save_grouped_dataset(dataset, out_dir)
        result = {
            "history_root": str(history_root),
            "run_dir": str(out_dir),
            "run_timestamp": str(run_timestamp),
            **produced,
        }
        result.update(
            write_history_run_index(
                history_root,
                run_timestamp=run_timestamp,
                run_dir=out_dir,
                result_paths=produced,
            )
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "sample-suite":
        suite_config = _filtered_config(config, setting_ids=args.settings)
        suite_config = _override_n_test(suite_config, args.n_test)
        if not suite_config.settings:
            parser.error("No settings selected for sample-suite.")
        history_root, root, run_timestamp = prepare_history_run_dir(args.save_dir)
        produced: list[dict[str, str]] = []
        seed = int(args.seed) if args.seed is not None else int(suite_config.runner.seed)
        for setting in suite_config.settings:
            setting_dir = root / setting.setting_id
            for rep in range(1, int(args.repeats) + 1):
                dataset = generate_grouped_dataset(
                    setting,
                    replicate_id=rep,
                    master_seed=seed,
                    family_specs=suite_config.families,
                )
                produced.append(save_grouped_dataset(dataset, setting_dir))
        manifest_path = root / "suite_manifest.json"
        save_json(suite_config.to_manifest(), manifest_path)
        result = {
            "history_root": str(history_root),
            "run_dir": str(root),
            "run_timestamp": str(run_timestamp),
            "manifest": str(manifest_path),
            "datasets": produced,
        }
        result.update(
            write_history_run_index(
                history_root,
                run_timestamp=run_timestamp,
                run_dir=root,
                result_paths={
                    "manifest": str(manifest_path),
                    "dataset_count": len(produced),
                },
            )
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "run-benchmark":
        run_config = _override_runner_from_args(config, args)
        if not run_config.settings:
            parser.error("No settings selected for run-benchmark.")
        if not run_config.methods.roster:
            parser.error("No methods selected for run-benchmark.")
        result = run_benchmark(run_config)
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "build-tables":
        result = build_paper_tables_from_results_dir(
            args.results_dir,
            method_order=config.methods.roster,
        )
        print(json.dumps(result, indent=2))
        return 0

    parser.error(f"Unknown command: {args.command!r}")
    return 1
