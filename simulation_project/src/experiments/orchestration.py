from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from tqdm.auto import tqdm

from .schemas import RunCommonConfig
from ..experiment_aliases import CLI_EXPERIMENT_CHOICES, cli_choice_to_key
from .runtime import (
    COMPUTE_PROFILES,
    EXP3_GIGG_MODES,
    _default_repeats,
    _normalize_compute_profile,
    _normalize_exp3_gigg_mode,
    _resolve_sampler_backend_for_experiment,
)
from ..utils import MASTER_SEED, save_json
from .exp1 import run_exp1_kappa_profile_regimes
from .exp2 import run_exp2_group_separation
from .exp3 import run_exp3_linear_benchmark, run_exp3a_main_benchmark, run_exp3b_boundary_stress
from .exp4 import run_exp4_variant_ablation
from .exp5 import run_exp5_prior_sensitivity


def run_all_experiments(
    n_jobs: int = 1,
    seed: int = MASTER_SEED,
    save_dir: str = "outputs/simulation_project",
    *,
    profile: str = "full",
    enforce_bayes_convergence: bool = True,
    max_convergence_retries: int | None = None,
    until_bayes_converged: bool = True,
    sampler_backend: str = "nuts",
    exp3_gigg_mode: str = "stable",
    skip_analysis: bool = False,
) -> Dict[str, Any]:
    profile_name = _normalize_compute_profile(profile)
    exp3_gigg_mode_name = _normalize_exp3_gigg_mode(exp3_gigg_mode)
    common_cfg = RunCommonConfig(
        n_jobs=n_jobs,
        seed=seed,
        save_dir=save_dir,
        profile=profile_name,
        enforce_bayes_convergence=bool(enforce_bayes_convergence),
        max_convergence_retries=max_convergence_retries,
        until_bayes_converged=bool(until_bayes_converged),
        sampler_backend=str(sampler_backend),
    )

    out: Dict[str, Any] = {}
    jobs: list[tuple[str, Any]] = [
        ("exp1", lambda: run_exp1_kappa_profile_regimes(n_jobs=n_jobs, seed=seed, save_dir=save_dir, repeats=_default_repeats("exp1", profile_name))),
        ("exp2", lambda: run_exp2_group_separation(repeats=_default_repeats("exp2", profile_name), **common_cfg.as_kwargs())),
        ("exp3a", lambda: run_exp3a_main_benchmark(repeats=_default_repeats("exp3", profile_name), gigg_mode=exp3_gigg_mode_name, **common_cfg.as_kwargs())),
        ("exp3b", lambda: run_exp3b_boundary_stress(repeats=_default_repeats("exp3", profile_name), gigg_mode=exp3_gigg_mode_name, **common_cfg.as_kwargs())),
        ("exp4", lambda: run_exp4_variant_ablation(repeats=_default_repeats("exp4", profile_name), **common_cfg.as_kwargs())),
        ("exp5", lambda: run_exp5_prior_sensitivity(repeats=_default_repeats("exp5", profile_name), **common_cfg.as_kwargs())),
    ]
    for name, runner in tqdm(jobs, total=len(jobs), desc="All Experiments", leave=True):
        out[name] = runner()
    save_json(
        {
            "profile": profile_name,
            "enforce_bayes_convergence": bool(enforce_bayes_convergence),
            "max_convergence_retries": max_convergence_retries,
            "until_bayes_converged": bool(until_bayes_converged),
            "exp3_gigg_mode": str(exp3_gigg_mode_name),
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
            "Run the unified simulation pipeline (Exp1, Exp2, Exp3a, Exp3b, Exp4, Exp5). "
            "On Windows, process-pool parallelism is disabled by default; "
            "set SIM_ALLOW_WINDOWS_PROCESS_POOL=1 to force-enable from a spawn-safe script entrypoint."
        )
    )
    parser.add_argument("--experiment", default="all", choices=list(CLI_EXPERIMENT_CHOICES))
    parser.add_argument("--save-dir", default="outputs/simulation_project")
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--profile", type=str, default="full", choices=list(COMPUTE_PROFILES))
    parser.add_argument("--no-enforce-bayes-convergence", action="store_true")
    parser.add_argument("--max-convergence-retries", type=int, default=None)
    parser.add_argument("--until-bayes-converged", action="store_true")
    parser.add_argument(
        "--exp3-gigg-mode",
        type=str,
        default="stable",
        choices=list(EXP3_GIGG_MODES),
        help="Exp3 GIGG mode: stable (enhanced, default) or paper_ref (strict baseline).",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="nuts",
        choices=["nuts", "collapsed", "gibbs"],
        help="GR-RHS posterior sampler: nuts (global default), collapsed (beta marginalized, Gaussian only), gibbs (Gibbs+slice, Gaussian only); Exp4 defaults to gibbs when --sampler is omitted",
    )
    args = parser.parse_args()
    exp_key = cli_choice_to_key(args.experiment)
    profile_name = _normalize_compute_profile(args.profile)
    exp3_gigg_mode_name = _normalize_exp3_gigg_mode(args.exp3_gigg_mode)
    enforce_conv = not bool(args.no_enforce_bayes_convergence)
    until_conv = bool(args.until_bayes_converged) or (enforce_conv and args.max_convergence_retries is None)
    sampler_backend_cli = _resolve_sampler_backend_for_experiment(exp_key, args.sampler)
    common_cfg = RunCommonConfig(
        n_jobs=args.n_jobs,
        seed=args.seed,
        save_dir=args.save_dir,
        profile=profile_name,
        enforce_bayes_convergence=enforce_conv,
        max_convergence_retries=args.max_convergence_retries,
        until_bayes_converged=until_conv,
        sampler_backend=sampler_backend_cli,
    )
    reps = args.repeats

    from .analysis.report import analyze_exp1, analyze_exp2, analyze_exp3, analyze_exp4, analyze_exp5, _safe_print, run_analysis

    base = Path(args.save_dir)

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
        run_all_experiments(**common_cfg.as_kwargs(), exp3_gigg_mode=exp3_gigg_mode_name)
    elif exp_key == "analysis":
        run_analysis(save_dir=args.save_dir)
    else:
        dispatch: dict[str, dict[str, Any]] = {
            "exp1": {
                "run": lambda: run_exp1_kappa_profile_regimes(
                    n_jobs=args.n_jobs,
                    seed=args.seed,
                    save_dir=args.save_dir,
                    repeats=reps or _default_repeats("exp1", profile_name),
                ),
                "analyze": analyze_exp1,
                "label": "Exp1: kappa_g Profile Regimes",
                "results_subdir": "exp1_kappa_profile_regimes",
            },
            "exp2": {
                "run": lambda: run_exp2_group_separation(
                    repeats=reps or _default_repeats("exp2", profile_name),
                    **common_cfg.as_kwargs(),
                ),
                "analyze": analyze_exp2,
                "label": "Exp2: Group Separation",
                "results_subdir": "exp2_group_separation",
            },
            "exp3": {
                "run": lambda: run_exp3_linear_benchmark(
                    repeats=reps or _default_repeats("exp3", profile_name),
                    gigg_mode=exp3_gigg_mode_name,
                    **common_cfg.as_kwargs(),
                ),
                "analyze": analyze_exp3,
                "label": "Exp3: Linear Benchmark",
                "results_subdir": "exp3_linear_benchmark",
            },
            "exp3a": {
                "run": lambda: run_exp3a_main_benchmark(
                    repeats=reps or _default_repeats("exp3", profile_name),
                    gigg_mode=exp3_gigg_mode_name,
                    **common_cfg.as_kwargs(),
                ),
                "analyze": analyze_exp3,
                "label": "Exp3a: Main Benchmark",
                "results_subdir": "exp3a_main_benchmark",
            },
            "exp3b": {
                "run": lambda: run_exp3b_boundary_stress(
                    repeats=reps or _default_repeats("exp3", profile_name),
                    gigg_mode=exp3_gigg_mode_name,
                    **common_cfg.as_kwargs(),
                ),
                "analyze": analyze_exp3,
                "label": "Exp3b: Boundary Stress",
                "results_subdir": "exp3b_boundary_stress",
            },
            "exp4": {
                "run": lambda: run_exp4_variant_ablation(
                    repeats=reps or _default_repeats("exp4", profile_name),
                    **common_cfg.as_kwargs(),
                ),
                "analyze": analyze_exp4,
                "label": "Exp4: Variant Ablation",
                "results_subdir": "exp4_variant_ablation",
            },
            "exp5": {
                "run": lambda: run_exp5_prior_sensitivity(
                    repeats=reps or _default_repeats("exp5", profile_name),
                    **common_cfg.as_kwargs(),
                ),
                "analyze": analyze_exp5,
                "label": "Exp5: Prior Sensitivity",
                "results_subdir": "exp5_prior_sensitivity",
            },
        }
        spec = dispatch[exp_key]
        spec["run"]()
        analyzer = spec["analyze"]
        label = str(spec["label"])
        results_subdir = str(spec["results_subdir"])
        _print_exp_analysis(label, analyzer(base / "results" / results_subdir))




