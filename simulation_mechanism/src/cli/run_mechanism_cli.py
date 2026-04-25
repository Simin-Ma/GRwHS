from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from ..config import MechanismConfig, load_mechanism_config
from ..dgp import generate_mechanism_dataset, save_mechanism_dataset
from ..plotting import build_mechanism_figures_from_results_dir
from ..runner import run_mechanism
from ..table_builder import build_paper_tables_from_results_dir
from ..utils import ensure_dir, save_json


def _print_settings(settings: Iterable[object]) -> None:
    for setting in settings:
        print(f"{setting.setting_id}: {setting.setting_label}")


def _filtered_config(
    config: MechanismConfig,
    *,
    setting_ids: list[str] | None = None,
    experiment_ids: list[str] | None = None,
    methods: list[str] | None = None,
) -> MechanismConfig:
    out = config
    if setting_ids:
        wanted = {str(item) for item in setting_ids}
        out = replace(out, settings=tuple(setting for setting in out.settings if setting.setting_id in wanted))
    if experiment_ids:
        wanted = {str(item) for item in experiment_ids}
        out = replace(out, settings=tuple(setting for setting in out.settings if setting.experiment_id in wanted))
    if methods:
        wanted = {str(item) for item in methods}
        settings = []
        for setting in out.settings:
            keep = tuple(method for method in setting.methods if str(method) in wanted)
            settings.append(replace(setting, methods=keep))
        out = replace(out, settings=tuple(setting for setting in settings if setting.methods))
    return out


def _override_n_test(config: MechanismConfig, n_test: int | None) -> MechanismConfig:
    if n_test is None:
        return config
    settings = tuple(replace(setting, n_test=int(n_test)) for setting in config.settings)
    return replace(config, settings=settings)


def _override_runner_from_args(config: MechanismConfig, args: argparse.Namespace) -> MechanismConfig:
    out = _filtered_config(
        config,
        setting_ids=getattr(args, "settings", None),
        experiment_ids=getattr(args, "experiments", None),
        methods=getattr(args, "methods", None),
    )
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
    parser = argparse.ArgumentParser(description="Run mechanism-first GR-RHS simulation utilities.")
    parser.add_argument(
        "--config",
        default="",
        help="Optional YAML config. Defaults to simulation_mechanism/config/mechanism.yaml.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list-settings", help="List mechanism settings.")
    list_parser.add_argument("--n-test", type=int, default=None, help="Optional override for test sample size.")

    dump = sub.add_parser("dump-manifest", help="Export the loaded mechanism manifest as JSON.")
    dump.add_argument("--save-path", default="", help="Optional JSON path to write the manifest.")
    dump.add_argument("--n-test", type=int, default=None, help="Optional override for test sample size.")

    sample = sub.add_parser("sample-setting", help="Generate one dataset for a mechanism setting.")
    sample.add_argument("--setting-id", required=True, help="Setting identifier from list-settings.")
    sample.add_argument("--replicate", type=int, default=1, help="Replicate index.")
    sample.add_argument("--seed", type=int, default=None, help="Master seed override.")
    sample.add_argument("--n-test", type=int, default=None, help="Override test sample size.")
    sample.add_argument(
        "--save-dir",
        default="outputs/simulation_mechanism/samples",
        help="Directory used to store the sampled dataset.",
    )

    suite = sub.add_parser("sample-suite", help="Generate datasets for the whole mechanism suite.")
    suite.add_argument("--repeats", type=int, default=1, help="Number of replicates per setting.")
    suite.add_argument("--seed", type=int, default=None, help="Master seed override.")
    suite.add_argument("--n-test", type=int, default=None, help="Override test sample size.")
    suite.add_argument("--settings", nargs="*", default=None, help="Optional subset of setting ids.")
    suite.add_argument("--experiments", nargs="*", default=None, help="Optional subset of experiment ids.")
    suite.add_argument(
        "--save-dir",
        default="outputs/simulation_mechanism/suite_samples",
        help="Directory used to store the sampled datasets.",
    )

    run = sub.add_parser("run-mechanism", help="Run the full mechanism pipeline from config to tables.")
    run.add_argument("--save-dir", default="", help="Optional override for the output directory.")
    run.add_argument("--repeats", type=int, default=None, help="Override repeats.")
    run.add_argument("--seed", type=int, default=None, help="Override master seed.")
    run.add_argument("--n-test", type=int, default=None, help="Override test sample size.")
    run.add_argument("--n-jobs", type=int, default=None, help="Number of replicate workers.")
    run.add_argument("--method-jobs", type=int, default=None, help="Number of per-task method workers.")
    run.add_argument("--methods", nargs="*", default=None, help="Optional subset of methods or ablation variants.")
    run.add_argument("--settings", nargs="*", default=None, help="Optional subset of setting ids.")
    run.add_argument("--experiments", nargs="*", default=None, help="Optional subset of experiment ids.")
    run.add_argument("--save-datasets", action="store_true", help="Persist sampled datasets alongside the run.")
    run.add_argument("--no-build-tables", action="store_true", help="Skip paper-table generation.")

    tables = sub.add_parser("build-tables", help="Rebuild mechanism tables from an existing results directory.")
    tables.add_argument(
        "--results-dir",
        default="outputs/simulation_mechanism/mechanism_main",
        help="Directory containing mechanism CSV outputs.",
    )

    figures = sub.add_parser("build-figures", help="Rebuild mechanism figures from an existing results directory.")
    figures.add_argument(
        "--results-dir",
        default="outputs/simulation_mechanism/mechanism_main",
        help="Directory containing mechanism CSV outputs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_mechanism_config(args.config or None)

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
        seed = int(args.seed) if args.seed is not None else int(config.runner.seed)
        dataset = generate_mechanism_dataset(
            setting_map[args.setting_id],
            replicate_id=int(args.replicate),
            master_seed=seed,
        )
        produced = save_mechanism_dataset(dataset, args.save_dir)
        print(json.dumps(produced, indent=2))
        return 0

    if args.command == "sample-suite":
        suite_config = _filtered_config(
            config,
            setting_ids=args.settings,
            experiment_ids=args.experiments,
        )
        suite_config = _override_n_test(suite_config, args.n_test)
        if not suite_config.settings:
            parser.error("No settings selected for sample-suite.")
        root = ensure_dir(args.save_dir)
        produced: list[dict[str, str]] = []
        seed = int(args.seed) if args.seed is not None else int(suite_config.runner.seed)
        for setting in suite_config.settings:
            setting_dir = root / setting.setting_id
            for rep in range(1, int(args.repeats) + 1):
                dataset = generate_mechanism_dataset(setting, replicate_id=rep, master_seed=seed)
                produced.append(save_mechanism_dataset(dataset, setting_dir))
        manifest_path = root / "suite_manifest.json"
        save_json(suite_config.to_manifest(), manifest_path)
        print(json.dumps({"manifest": str(manifest_path), "datasets": produced}, indent=2))
        return 0

    if args.command == "run-mechanism":
        run_config = _override_runner_from_args(config, args)
        if not run_config.settings:
            parser.error("No settings selected for run-mechanism.")
        result = run_mechanism(run_config)
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "build-tables":
        result = build_paper_tables_from_results_dir(args.results_dir)
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "build-figures":
        result = build_mechanism_figures_from_results_dir(args.results_dir)
        print(json.dumps(result, indent=2))
        return 0

    parser.error(f"Unknown command: {args.command!r}")
    return 1
