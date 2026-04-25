from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

from tqdm.auto import tqdm

from .schemas import RunCommonConfig
from ..experiment_aliases import CLI_EXPERIMENT_CHOICES, cli_choice_to_key
from .runtime import EXP3_GIGG_MODES, _default_repeats, _normalize_exp3_gigg_mode
from ..output_layout import resolve_analysis_dir, resolve_run_save_dir
from ..utils import MASTER_SEED, save_json
from .exp1 import run_exp1_kappa_profile_regimes
from .exp2 import run_exp2_group_separation
from .exp3 import (
    run_exp3_linear_benchmark,
    run_exp3a_main_benchmark,
    run_exp3b_boundary_stress,
    run_exp3c_highdim_stress,
    run_exp3d_within_group_mixed,
)
from .exp4 import run_exp4_variant_ablation
from .exp5 import run_exp5_prior_sensitivity
from .exp_ga_v2_group_separation import run_ga_v2_group_separation
from .exp_ga_v2_complexity_mismatch import run_ga_v2_complexity_mismatch
from .exp_ga_v2_correlation_stress import run_ga_v2_correlation_stress


def _parse_csv_ints(value: str | None) -> list[int] | None:
    if value is None:
        return None
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts:
        return None
    return [int(p) for p in parts]


def _parse_csv_floats(value: str | None) -> list[float] | None:
    if value is None:
        return None
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts:
        return None
    return [float(p) for p in parts]


def _parse_csv_strings(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    return parts or None


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
    exp3_gigg_mode: str = "paper_ref",
    skip_analysis: bool = False,
    archive_artifacts: bool = True,
    all_parallel_jobs: int = 1,
) -> Dict[str, Any]:
    skip_analysis = bool(skip_analysis) or bool(skip_run_analysis)
    exp3_gigg_mode_name = _normalize_exp3_gigg_mode(exp3_gigg_mode)
    common_cfg = RunCommonConfig(
        n_jobs=int(n_jobs),
        method_jobs=int(method_jobs),
        seed=seed,
        save_dir=save_dir,
        skip_run_analysis=bool(skip_analysis),
        archive_artifacts=bool(archive_artifacts),
        enforce_bayes_convergence=bool(enforce_bayes_convergence),
        max_convergence_retries=max_convergence_retries,
        until_bayes_converged=bool(until_bayes_converged),
    )

    out: Dict[str, Any] = {}
    jobs: list[tuple[str, Any]] = [
        (
            "exp1",
            lambda: run_exp1_kappa_profile_regimes(
                n_jobs=int(n_jobs),
                seed=seed,
                save_dir=save_dir,
                repeats=int(_default_repeats("exp1")),
                skip_run_analysis=bool(skip_analysis),
                archive_artifacts=bool(archive_artifacts),
            ),
        ),
        (
            "exp2",
            lambda: run_exp2_group_separation(
                repeats=int(_default_repeats("exp2")),
                **common_cfg.as_kwargs(),
            ),
        ),
        (
            "exp3a",
            lambda: run_exp3a_main_benchmark(
                repeats=int(_default_repeats("exp3")),
                gigg_mode=exp3_gigg_mode_name,
                **common_cfg.as_kwargs(),
            ),
        ),
        (
            "exp3b",
            lambda: run_exp3b_boundary_stress(
                repeats=int(_default_repeats("exp3")),
                gigg_mode=exp3_gigg_mode_name,
                **common_cfg.as_kwargs(),
            ),
        ),
        (
            "exp4",
            lambda: run_exp4_variant_ablation(
                repeats=int(_default_repeats("exp4")),
                **common_cfg.as_kwargs(),
            ),
        ),
        (
            "exp5",
            lambda: run_exp5_prior_sensitivity(
                repeats=int(_default_repeats("exp5")),
                **common_cfg.as_kwargs(),
            ),
        ),
    ]

    workers = max(1, min(int(all_parallel_jobs), len(jobs)))
    if workers <= 1:
        for name, runner in tqdm(jobs, total=len(jobs), desc="All Experiments", leave=True):
            out[name] = runner()
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_map = {ex.submit(runner): name for name, runner in jobs}
            done: dict[str, Any] = {}
            for fut in tqdm(as_completed(fut_map), total=len(jobs), desc=f"All Experiments (parallel={workers})", leave=True):
                name = fut_map[fut]
                done[name] = fut.result()
            out = {name: done[name] for name, _ in jobs}

    save_json(
        {
            "protocol": "single_full",
            "method_jobs": int(method_jobs),
            "archive_artifacts": bool(archive_artifacts),
            "enforce_bayes_convergence": bool(enforce_bayes_convergence),
            "max_convergence_retries": max_convergence_retries,
            "until_bayes_converged": bool(until_bayes_converged),
            "exp3_gigg_mode": str(exp3_gigg_mode_name),
            "all_parallel_jobs": int(workers),
            "results": out,
        },
        Path(save_dir) / "results" / "run_manifest.json",
    )
    if not skip_analysis:
        from .analysis.report import run_analysis

        run_analysis(save_dir=save_dir)
    return out


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the unified full simulation pipeline (Exp1, Exp2, Exp3a, Exp3b, Exp4, Exp5), "
            "or run individual experiment entries. "
            "On Windows, process-pool parallelism is allowed from spawn-safe script entrypoints "
            "and disabled only in interactive/non-spawn-safe contexts."
        )
    )
    parser.add_argument("--experiment", default="all", choices=list(CLI_EXPERIMENT_CHOICES))
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Explicit output directory. Relative paths are normalized under the workspace.",
    )
    parser.add_argument(
        "--workspace",
        default="simulation_project",
        help="Workspace root for organized outputs (default resolves to outputs/simulation_project).",
    )
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--method-jobs",
        type=int,
        default=1,
        help="Concurrent fits within a single replicate/task (default 1 = serial within task).",
    )
    parser.add_argument(
        "--all-parallel-jobs",
        type=int,
        default=1,
        help="Number of concurrent experiments when --experiment all (default 1 = serial).",
    )
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--no-archive-artifacts", action="store_true")
    parser.add_argument("--no-enforce-bayes-convergence", action="store_true")
    parser.add_argument("--max-convergence-retries", type=int, default=None)
    parser.add_argument("--until-bayes-converged", action="store_true")
    parser.add_argument(
        "--exp3-gigg-mode",
        type=str,
        default="paper_ref",
        choices=list(EXP3_GIGG_MODES),
        help="Exp3 GIGG mode. Only paper_ref is supported; it matches the gigg-master reference path.",
    )
    parser.add_argument("--ga-v2-group-sizes", type=str, default=None, help="Comma-separated group sizes for GA-V2 runs, e.g. 10,10,10,10,10")
    parser.add_argument("--ga-v2-methods", type=str, default=None, help="Comma-separated method list for GA-V2 runs, e.g. GR_RHS,RHS")
    parser.add_argument("--ga-v2-n-train", type=int, default=None, help="Override training sample size for GA-V2 runs.")
    parser.add_argument("--ga-v2-n-test", type=int, default=None, help="Override test sample size for GA-V2 runs.")
    parser.add_argument("--ga-v2-rho-within", type=float, default=None, help="Override within-group correlation for GA-V2 runs.")
    parser.add_argument("--ga-v2-rho-between", type=float, default=None, help="Override between-group correlation for GA-V2 runs.")
    parser.add_argument("--ga-v2a-mu", type=str, default=None, help="Comma-separated group-level means for GA-V2-A, e.g. 0,0,1.5,4,10")
    parser.add_argument("--ga-v2a-sigma2", type=float, default=None, help="Override residual variance for GA-V2-A.")
    parser.add_argument("--ga-v2b-patterns", type=str, default=None, help="Comma-separated GA-V2-B complexity patterns, e.g. few_groups,many_groups")
    parser.add_argument("--ga-v2b-within-patterns", type=str, default=None, help="Comma-separated GA-V2-B within-group patterns, e.g. concentrated,distributed")
    parser.add_argument("--ga-v2b-total-active-coeff", type=int, default=None, help="Override total active coefficient count for GA-V2-B.")
    parser.add_argument("--ga-v2b-target-snr", type=float, default=None, help="Override target SNR for GA-V2-B.")
    parser.add_argument("--ga-v2c-rho-list", type=str, default=None, help="Comma-separated rho_within grid for GA-V2-C, e.g. 0.4,0.6,0.8,0.95")
    parser.add_argument("--ga-v2c-within-patterns", type=str, default=None, help="Comma-separated within-group patterns for GA-V2-C.")
    parser.add_argument("--ga-v2c-active-groups", type=str, default=None, help="Comma-separated active group ids for GA-V2-C, e.g. 0,1")
    parser.add_argument("--ga-v2c-target-snr", type=float, default=None, help="Override target SNR for GA-V2-C.")
    args = parser.parse_args()
    exp_key = cli_choice_to_key(args.experiment)
    n_jobs_use = int(args.n_jobs)
    method_jobs_use = int(args.method_jobs)
    repeats_use = args.repeats
    exp3_gigg_mode_name = _normalize_exp3_gigg_mode(args.exp3_gigg_mode)
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
    common_cfg = RunCommonConfig(
        n_jobs=n_jobs_use,
        method_jobs=method_jobs_use,
        seed=args.seed,
        save_dir=str(save_dir_resolved),
        skip_run_analysis=bool(args.skip_analysis),
        archive_artifacts=(not bool(args.no_archive_artifacts)),
        enforce_bayes_convergence=enforce_conv,
        max_convergence_retries=args.max_convergence_retries,
        until_bayes_converged=until_conv,
    )
    reps = repeats_use
    ga_v2_common_overrides: dict[str, Any] = {}
    ga_v2_group_sizes = _parse_csv_ints(args.ga_v2_group_sizes)
    ga_v2_methods = _parse_csv_strings(args.ga_v2_methods)
    if ga_v2_group_sizes is not None:
        ga_v2_common_overrides["group_sizes"] = ga_v2_group_sizes
    if ga_v2_methods is not None:
        ga_v2_common_overrides["methods"] = ga_v2_methods
    if args.ga_v2_n_train is not None:
        ga_v2_common_overrides["n_train"] = int(args.ga_v2_n_train)
    if args.ga_v2_n_test is not None:
        ga_v2_common_overrides["n_test"] = int(args.ga_v2_n_test)
    if args.ga_v2_rho_within is not None:
        ga_v2_common_overrides["rho_within"] = float(args.ga_v2_rho_within)
    if args.ga_v2_rho_between is not None:
        ga_v2_common_overrides["rho_between"] = float(args.ga_v2_rho_between)

    ga_v2a_overrides = dict(ga_v2_common_overrides)
    ga_v2a_mu = _parse_csv_floats(args.ga_v2a_mu)
    if ga_v2a_mu is not None:
        ga_v2a_overrides["mu"] = ga_v2a_mu
    if args.ga_v2a_sigma2 is not None:
        ga_v2a_overrides["sigma2"] = float(args.ga_v2a_sigma2)

    ga_v2b_overrides = dict(ga_v2_common_overrides)
    ga_v2b_patterns = _parse_csv_strings(args.ga_v2b_patterns)
    ga_v2b_within_patterns = _parse_csv_strings(args.ga_v2b_within_patterns)
    if ga_v2b_patterns is not None:
        ga_v2b_overrides["patterns"] = ga_v2b_patterns
    if ga_v2b_within_patterns is not None:
        ga_v2b_overrides["within_group_patterns"] = ga_v2b_within_patterns
    if args.ga_v2b_total_active_coeff is not None:
        ga_v2b_overrides["total_active_coeff"] = int(args.ga_v2b_total_active_coeff)
    if args.ga_v2b_target_snr is not None:
        ga_v2b_overrides["target_snr"] = float(args.ga_v2b_target_snr)

    ga_v2c_overrides = dict(ga_v2_common_overrides)
    ga_v2c_rho_list = _parse_csv_floats(args.ga_v2c_rho_list)
    ga_v2c_within_patterns = _parse_csv_strings(args.ga_v2c_within_patterns)
    ga_v2c_active_groups = _parse_csv_ints(args.ga_v2c_active_groups)
    if ga_v2c_rho_list is not None:
        ga_v2c_overrides["rho_within_list"] = ga_v2c_rho_list
    if ga_v2c_within_patterns is not None:
        ga_v2c_overrides["within_group_patterns"] = ga_v2c_within_patterns
    if ga_v2c_active_groups is not None:
        ga_v2c_overrides["active_groups"] = ga_v2c_active_groups
    if args.ga_v2c_target_snr is not None:
        ga_v2c_overrides["target_snr"] = float(args.ga_v2c_target_snr)

    from .analysis.report import (
        analyze_exp1,
        analyze_exp2,
        analyze_exp3,
        analyze_exp4,
        analyze_exp5,
        analyze_ga_v2_complexity_mismatch,
        analyze_ga_v2_correlation_stress,
        analyze_ga_v2_group_separation,
        _safe_print,
        run_analysis,
    )

    base = Path(str(save_dir_resolved))

    def _print_exp_analysis(label: str, result: dict) -> None:
        sep = "=" * 68
        lines = [sep, f"ANALYSIS: {label}", sep]
        for finding in result.get("findings", []):
            lines.append(finding)
        lines.append(sep)
        _safe_print("\n".join(lines))
        out_path = base / "results" / f"analysis_{label.lower().replace(' ', '_').replace(':', '')}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result.get("metrics", {}), f, indent=2)

    if exp_key == "all":
        run_all_experiments(
            **common_cfg.as_kwargs(),
            exp3_gigg_mode=exp3_gigg_mode_name,
            skip_analysis=bool(args.skip_analysis),
            all_parallel_jobs=max(1, int(args.all_parallel_jobs)),
        )
    elif exp_key == "analysis":
        run_analysis(save_dir=str(save_dir_resolved))
    else:
        dispatch: dict[str, dict[str, Any]] = {
            "exp1": {
                "run": lambda: run_exp1_kappa_profile_regimes(
                    n_jobs=n_jobs_use,
                    skip_run_analysis=bool(args.skip_analysis),
                    archive_artifacts=(not bool(args.no_archive_artifacts)),
                    seed=args.seed,
                    save_dir=str(save_dir_resolved),
                    repeats=reps or _default_repeats("exp1"),
                ),
                "analyze": analyze_exp1,
                "label": "Exp1: kappa_g Profile Regimes",
                "results_subdir": "exp1_kappa_profile_regimes",
            },
            "exp2": {
                "run": lambda: run_exp2_group_separation(
                    repeats=reps or _default_repeats("exp2"),
                    **common_cfg.as_kwargs(),
                ),
                "analyze": analyze_exp2,
                "label": "Exp2: Group Separation",
                "results_subdir": "exp2_group_separation",
            },
            "exp3": {
                "run": lambda: run_exp3_linear_benchmark(
                    repeats=reps or _default_repeats("exp3"),
                    gigg_mode=exp3_gigg_mode_name,
                    **common_cfg.as_kwargs(),
                ),
                "analyze": analyze_exp3,
                "label": "Exp3: Linear Benchmark",
                "results_subdir": "exp3_linear_benchmark",
            },
            "exp3a": {
                "run": lambda: run_exp3a_main_benchmark(
                    repeats=reps or _default_repeats("exp3"),
                    gigg_mode=exp3_gigg_mode_name,
                    **common_cfg.as_kwargs(),
                ),
                "analyze": analyze_exp3,
                "label": "Exp3a: Main Benchmark",
                "results_subdir": "exp3a_main_benchmark",
            },
            "exp3b": {
                "run": lambda: run_exp3b_boundary_stress(
                    repeats=reps or _default_repeats("exp3"),
                    gigg_mode=exp3_gigg_mode_name,
                    **common_cfg.as_kwargs(),
                ),
                "analyze": analyze_exp3,
                "label": "Exp3b: Boundary Stress",
                "results_subdir": "exp3b_boundary_stress",
            },
            "exp3c": {
                "run": lambda: run_exp3c_highdim_stress(
                    repeats=reps or _default_repeats("exp3c"),
                    gigg_mode=exp3_gigg_mode_name,
                    **common_cfg.as_kwargs(),
                ),
                "analyze": analyze_exp3,
                "label": "Exp3c: Highdim Stress",
                "results_subdir": "exp3c_highdim_stress",
            },
            "exp3d": {
                "run": lambda: run_exp3d_within_group_mixed(
                    repeats=reps or _default_repeats("exp3d"),
                    gigg_mode=exp3_gigg_mode_name,
                    **common_cfg.as_kwargs(),
                ),
                "analyze": analyze_exp3,
                "label": "Exp3d: Legacy Boundary Stress",
                "results_subdir": "exp3d_within_group_mixed",
            },
            "exp4": {
                "run": lambda: run_exp4_variant_ablation(
                    repeats=reps or _default_repeats("exp4"),
                    **common_cfg.as_kwargs(),
                ),
                "analyze": analyze_exp4,
                "label": "Exp4: Variant Ablation",
                "results_subdir": "exp4_variant_ablation",
            },
            "exp5": {
                "run": lambda: run_exp5_prior_sensitivity(
                    repeats=reps or _default_repeats("exp5"),
                    **common_cfg.as_kwargs(),
                ),
                "analyze": analyze_exp5,
                "label": "Exp5: Prior Sensitivity",
                "results_subdir": "exp5_prior_sensitivity",
            },
            "ga_v2a": {
                "run": lambda: run_ga_v2_group_separation(
                    repeats=reps or 100,
                    **common_cfg.as_kwargs(),
                    **ga_v2a_overrides,
                ),
                "analyze": analyze_ga_v2_group_separation,
                "label": "GA-V2-A: Group Separation",
                "results_subdir": "group_aware_v2/ga_v2_group_separation",
            },
            "ga_v2b": {
                "run": lambda: run_ga_v2_complexity_mismatch(
                    repeats=reps or 40,
                    **common_cfg.as_kwargs(),
                    **ga_v2b_overrides,
                ),
                "analyze": analyze_ga_v2_complexity_mismatch,
                "label": "GA-V2-B: Complexity Mismatch",
                "results_subdir": "group_aware_v2/ga_v2_complexity_mismatch",
            },
            "ga_v2c": {
                "run": lambda: run_ga_v2_correlation_stress(
                    repeats=reps or 30,
                    **common_cfg.as_kwargs(),
                    **ga_v2c_overrides,
                ),
                "analyze": analyze_ga_v2_correlation_stress,
                "label": "GA-V2-C: Correlation Stress",
                "results_subdir": "group_aware_v2/ga_v2_correlation_stress",
            },
        }
        spec = dispatch[exp_key]
        spec["run"]()
        if not bool(args.skip_analysis):
            analyzer = spec["analyze"]
            label = str(spec["label"])
            results_subdir = str(spec["results_subdir"])
            _print_exp_analysis(label, analyzer(base / "results" / results_subdir))
