from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from ..experiment_aliases import CLI_EXPERIMENT_CHOICES, cli_choice_to_key
from ..output_layout import resolve_analysis_dir, resolve_run_save_dir
from ..utils import MASTER_SEED
from .exp_ga_v2_complexity_mismatch import run_ga_v2_complexity_mismatch
from .exp_ga_v2_correlation_stress import run_ga_v2_correlation_stress
from .exp_ga_v2_group_separation import run_ga_v2_group_separation


def run_all_experiments(
    n_jobs: int = 1,
    method_jobs: int = 1,
    seed: int = MASTER_SEED,
    save_dir: str = "outputs/simulation_project",
    *,
    skip_run_analysis: bool = False,
    enforce_bayes_convergence: bool = True,
    max_convergence_retries: int | None = None,
    until_bayes_converged: bool = True,
    skip_analysis: bool = False,
    archive_artifacts: bool = True,
    all_parallel_jobs: int = 1,
) -> Dict[str, Any]:
    _ = all_parallel_jobs
    out: Dict[str, Any] = {}
    common = {
        "n_jobs": int(n_jobs),
        "method_jobs": int(method_jobs),
        "seed": int(seed),
        "save_dir": str(save_dir),
        "skip_run_analysis": bool(skip_run_analysis),
        "archive_artifacts": bool(archive_artifacts),
        "enforce_bayes_convergence": bool(enforce_bayes_convergence),
        "max_convergence_retries": max_convergence_retries,
        "until_bayes_converged": bool(until_bayes_converged),
    }
    out["ga_v2a"] = run_ga_v2_group_separation(repeats=100, **common)
    out["ga_v2b"] = run_ga_v2_complexity_mismatch(repeats=40, **common)
    out["ga_v2c"] = run_ga_v2_correlation_stress(repeats=30, **common)
    if not bool(skip_analysis) and not bool(skip_run_analysis):
        from .analysis.report import run_analysis

        run_analysis(save_dir=save_dir)
    return out


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Run the active simulation pipeline (GA-V2 suite) or analysis."
    )
    parser.add_argument("--experiment", default="all", choices=list(CLI_EXPERIMENT_CHOICES))
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--workspace", default="simulation_project")
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--method-jobs", type=int, default=1)
    parser.add_argument("--all-parallel-jobs", type=int, default=1)
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--no-archive-artifacts", action="store_true")
    parser.add_argument("--no-enforce-bayes-convergence", action="store_true")
    parser.add_argument("--max-convergence-retries", type=int, default=None)
    parser.add_argument("--until-bayes-converged", action="store_true")
    args = parser.parse_args()

    exp_key = cli_choice_to_key(args.experiment)
    enforce_conv = not bool(args.no_enforce_bayes_convergence)
    until_conv = bool(args.until_bayes_converged) or (enforce_conv and args.max_convergence_retries is None)
    if exp_key == "analysis":
        save_dir_resolved = resolve_analysis_dir(args.save_dir, workspace=args.workspace)
    else:
        save_dir_resolved = resolve_run_save_dir(
            args.save_dir,
            workspace=args.workspace,
            run_label=f"cli_{exp_key}",
        )
    common = {
        "n_jobs": int(args.n_jobs),
        "method_jobs": int(args.method_jobs),
        "seed": int(args.seed),
        "save_dir": str(save_dir_resolved),
        "skip_run_analysis": bool(args.skip_analysis),
        "archive_artifacts": not bool(args.no_archive_artifacts),
        "enforce_bayes_convergence": bool(enforce_conv),
        "max_convergence_retries": args.max_convergence_retries,
        "until_bayes_converged": bool(until_conv),
    }

    if exp_key == "all":
        run_all_experiments(
            **common,
            skip_analysis=bool(args.skip_analysis),
            all_parallel_jobs=max(1, int(args.all_parallel_jobs)),
        )
        return
    if exp_key == "analysis":
        from .analysis.report import run_analysis

        run_analysis(save_dir=str(save_dir_resolved))
        return

    dispatch = {
        "ga_v2a": lambda: run_ga_v2_group_separation(repeats=args.repeats or 100, **common),
        "ga_v2b": lambda: run_ga_v2_complexity_mismatch(repeats=args.repeats or 40, **common),
        "ga_v2c": lambda: run_ga_v2_correlation_stress(repeats=args.repeats or 30, **common),
    }
    result = dispatch[exp_key]()
    print(json.dumps(result, ensure_ascii=False, indent=2))
