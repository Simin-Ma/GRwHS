#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random sweep driver for Scenario B GRwHS tuning.

Features
--------
1) Random search with hard constraints linking model and Gibbs parameters.
2) Convergence gatekeeper based on diagnostics.json (supports multi-chain).
3) Sampler-only escalation ladder to recover convergence before scoring.
4) Composite scoring that balances predictive accuracy and calibration.
5) (NEW) --fast mode to reduce average runtime per candidate.
6) (NEW) --stop_after_accepted to stop early once enough candidates pass.

Usage
-----
python random_sweep_selector.py \
  --base_config configs/scenario_B.yaml \
  --budget 40 \
  --chains 2 \
  --work_root outputs/random_search_B \
  [--fast] \
  [--stop_after_accepted 8]

Adjust RUN_EXPERIMENT_ENTRY, DIAG_FILENAME, and METRICS_FILENAME below if
your repository uses different entrypoints or output filenames.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import string
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

# =============================================================================
# Repository-specific defaults
# =============================================================================

# Command used to launch one experiment run. Override if needed.
RUN_EXPERIMENT_ENTRY: List[str] = [sys.executable, "-m", "grwhs.cli.run_experiment", "--config"]

# Expected output filenames within each run directory.
DIAG_FILENAME = "convergence.json"
METRICS_FILENAME = "metrics.json"


# =============================================================================
# Random search space
# =============================================================================

def sample_log_uniform(low: float, high: float) -> float:
    """Sample a strictly positive value from a log-uniform distribution."""
    lo, hi = math.log(low), math.log(high)
    return math.exp(random.uniform(lo, hi))


def sample_search_space(fast: bool = False) -> Dict[str, Any]:
    """
    Draw a single candidate configuration override.

    Parameters
    ----------
    fast : bool
        If True, restrict the iters set to {10000, 12000} for faster screening.

    Returns
    -------
    dict
        Nested overrides for model and inference sections. The structure
        mirrors the YAML layout to allow deep merging with the base config.
    """
    s_guess = random.randint(16, 30)
    c = round(random.uniform(1.5, 2.4), 2)
    eta = round(random.uniform(1.5, 1.9), 2)
    s0 = round(random.uniform(2.8, 3.6), 2)

    slice_w = round(random.uniform(12.0, 16.5), 1)
    slice_m = int(round(100 * slice_w))
    jitter = sample_log_uniform(1e-6, 8e-6)

    iters_choices = [10_000, 12_000] if fast else [10_000, 12_000, 14_000, 16_000]
    iters = random.choice(iters_choices)
    burn_in = iters // 2
    thin = 2

    return {
        "model": {
            "s_guess": s_guess,
            "c": c,
            "eta": eta,
            "s0": s0,
            "size_adjust": True,
        },
        "inference": {
            "gibbs": {
                "slice_w": slice_w,
                "slice_m": slice_m,
                "jitter": jitter,
                "iters": iters,
                "burn_in": burn_in,
                "thin": thin,
                "adapt_slice": False,
                "seed": 123,
            }
        },
    }


def violates_hard_rules(cfg_overrides: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Apply hard constraints to prune infeasible samples early.

    Parameters
    ----------
    cfg_overrides : dict
        Candidate overrides returned by ``sample_search_space``.

    Returns
    -------
    (bool, str)
        ``(True, reason)`` if the sample should be rejected, otherwise
        ``(False, "")``.
    """
    model_cfg = cfg_overrides["model"]
    gibbs_cfg = cfg_overrides["inference"]["gibbs"]

    if model_cfg["s_guess"] <= 18 and model_cfg["c"] <= 1.6:
        return True, "overshrink: s_guess<=18 & c<=1.6"

    if (
        model_cfg["s_guess"] >= 26
        and model_cfg["c"] >= 2.2
        and model_cfg["eta"] >= 1.8
    ):
        return True, "overrelax: s_guess>=26 & c>=2.2 & eta>=1.8"

    if model_cfg["eta"] < 1.6 and model_cfg["c"] < 1.7:
        return True, "sticky: eta<1.6 & c<1.7"

    if gibbs_cfg["slice_w"] > 16.5:
        return True, "slice_w too large (>16.5)"

    if gibbs_cfg["slice_m"] != int(round(100 * gibbs_cfg["slice_w"])):
        return True, "slice_m must equal round(100 * slice_w)"

    if not (1e-6 <= gibbs_cfg["jitter"] <= 8e-6):
        return True, "jitter outside [1e-6, 8e-6]"

    if gibbs_cfg["burn_in"] != gibbs_cfg["iters"] // 2:
        return True, "burn_in must be iters//2"

    return False, ""


# =============================================================================
# Convergence diagnostics
# =============================================================================

CONV_TARGETS = {
    "phi": {"rhat_max": 1.03, "ess_median": 100.0, "ess_min": 60.0},
    "sigma2": {"rhat_max": 1.01, "ess_median": 90.0},
    "beta": {"ess_min": 100.0},
    "tau": {"rhat_max": 1.05, "ess_median": 200.0},
    "lambda": {"rhat_max": 1.05, "ess_median": 200.0},
}


def check_convergence(diag: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Decide whether diagnostics meet convergence thresholds.

    Parameters
    ----------
    diag : dict
        Parsed JSON diagnostics for a single run.

    Returns
    -------
    (bool, str)
        ``(True, "")`` if all checks pass, else ``(False, reason)``.
    """
    try:
        phi = diag["phi"]
        if phi["rhat_max"] > CONV_TARGETS["phi"]["rhat_max"]:
            return False, f"phi Rhat={phi['rhat_max']:.3f} > 1.03"
        if phi["ess_median"] < CONV_TARGETS["phi"]["ess_median"]:
            return False, f"phi ESS_med={phi['ess_median']:.1f} < 100"
        if phi["ess_min"] < CONV_TARGETS["phi"]["ess_min"]:
            return False, f"phi ESS_min={phi['ess_min']:.1f} < 60"

        sigma2 = diag["sigma2"]
        if sigma2["rhat_max"] > CONV_TARGETS["sigma2"]["rhat_max"]:
            return False, f"sigma2 Rhat={sigma2['rhat_max']:.3f} > 1.01"
        if sigma2["ess_median"] < CONV_TARGETS["sigma2"]["ess_median"]:
            return False, f"sigma2 ESS_med={sigma2['ess_median']:.1f} < 90"

        beta = diag["beta"]
        if beta["ess_min"] < CONV_TARGETS["beta"]["ess_min"]:
            return False, f"beta ESS_min={beta['ess_min']:.1f} < 100"

        tau = diag["tau"]
        if tau["rhat_max"] > CONV_TARGETS["tau"]["rhat_max"]:
            return False, f"tau Rhat={tau['rhat_max']:.3f} > 1.05"

        lamb = diag["lambda"]
        if lamb["rhat_max"] > CONV_TARGETS["lambda"]["rhat_max"]:
            return False, f"lambda Rhat={lamb['rhat_max']:.3f} > 1.05"

        return True, ""
    except KeyError as exc:
        return False, f"missing diagnostics key: {exc}"


# =============================================================================
# Sampler escalation ladder
# =============================================================================

def escalate_sampler(base_gibbs_cfg: Dict[str, Any], fast: bool = False) -> Iterable[Tuple[Dict[str, Any], str]]:
    """
    Generate up to three sampler-only escalations for a failing chain.

    Parameters
    ----------
    base_gibbs_cfg : dict
        Gibbs configuration to escalate from.
    fast : bool
        If True, C-escalation only adds +1000 iterations (instead of +2000).

    Yields
    ------
    (dict, str)
        A pair of (gibbs_config, label) describing the escalation.
    """
    # Escalation A: increase slice step size.
    esc_a = deepcopy(base_gibbs_cfg)
    esc_a["slice_w"] = min(esc_a["slice_w"] + 1.0, 16.5)
    esc_a["slice_m"] = int(round(100 * esc_a["slice_w"]))
    yield esc_a, "A_bigger_step"

    # Escalation B: add jitter for numerical stability.
    esc_b = deepcopy(esc_a)
    esc_b["jitter"] = min(esc_b["jitter"] + 1e-6, 8e-6)
    yield esc_b, "B_more_jitter"

    # Escalation C: lengthen the chain.
    esc_c = deepcopy(esc_b)
    delta = 1_000 if fast else 2_000
    esc_c["iters"] = esc_c["iters"] + delta
    esc_c["burn_in"] = esc_c["iters"] // 2
    yield esc_c, "C_longer_chain"


# =============================================================================
# Scoring
# =============================================================================

def score_from_metrics(metrics: Dict[str, Any]) -> float:
    """
    Compute a scalar score given metrics.json contents.

    The objective favours low RMSE, high PLL, and discourages excessive
    interval width or degrees of freedom. Under-coverage is penalised
    heavily to enforce calibration.
    """
    rmse = float(metrics["metrics"]["RMSE"])
    pll = float(metrics["metrics"]["PredictiveLogLikelihood"])
    coverage = float(metrics["metrics"]["Coverage90"])
    interval_width = float(metrics["metrics"]["IntervalWidth90"])
    effective_df = float(metrics["metrics"]["EffectiveDoF"])

    penalty_edf = max(effective_df - 220.0, 0.0)
    penalty_cov = max(0.9 - coverage, 0.0)
    penalty_iw = max(interval_width - 20.0, 0.0)

    score = (
        -rmse
        + 0.02 * pll
        - 0.002 * penalty_edf
        - 0.5 * penalty_cov
        - 0.02 * penalty_iw
    )
    return score


# =============================================================================
# YAML helpers
# =============================================================================

def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(obj: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(obj, handle, sort_keys=False, allow_unicode=True)


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` in-place and return ``base``."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


# =============================================================================
# Runner utilities
# =============================================================================

def gen_run_dir(root: Path, prefix: str) -> Path:
    """Create a unique run directory rooted at ``root``."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    run_dir = root / f"{prefix}-{timestamp}-{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _infer_repo_root(base_cfg_path: Path) -> Path:
    """Heuristic to locate the repository root given a base config path."""
    candidates = list(base_cfg_path.parents)
    for candidate in candidates:
        if (candidate / "grwhs").exists():
            return candidate
    return base_cfg_path.parent


def run_once(
    overrides: Dict[str, Any],
    base_cfg_path: Path,
    work_root: Path,
    tag: str,
) -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
    """
    Execute one experiment with merged configuration.

    Returns the run directory together with parsed diagnostics and metrics.
    """
    configs_root = work_root / "configs"
    configs_root.mkdir(parents=True, exist_ok=True)
    cfg_dir = gen_run_dir(configs_root, tag)

    merged_cfg = load_yaml(base_cfg_path)
    merged_cfg = deep_update(merged_cfg, deepcopy(overrides))

    cfg_path = cfg_dir / "config.generated.yaml"
    dump_yaml(merged_cfg, cfg_path)

    runs_root = work_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    repo_root = _infer_repo_root(base_cfg_path)
    cmd = RUN_EXPERIMENT_ENTRY + [
        str(cfg_path),
        "--outdir",
        str(runs_root),
        "--name",
        tag,
    ]
    result = subprocess.run(
        cmd,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    (cfg_dir / "stdout.log").write_text(result.stdout or "", encoding="utf-8")
    (cfg_dir / "stderr.log").write_text(result.stderr or "", encoding="utf-8")

    if result.returncode != 0:
        raise RuntimeError(
            f"run_experiment failed (exit {result.returncode})\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    artifact_path: Path | None = None
    marker = "Artifacts in:"
    for line in (result.stdout or "").splitlines():
        if marker in line:
            candidate = line.split(marker, 1)[1].strip()
            if candidate:
                artifact_path = Path(candidate).expanduser().resolve()
                break

    if artifact_path is None or not artifact_path.exists():
        candidates = sorted(
            runs_root.glob(f"{tag}-*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            artifact_path = candidates[0].resolve()

    if artifact_path is None or not artifact_path.exists():
        raise RuntimeError(
            "Could not locate run artifacts after execution.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    diag_path = artifact_path / DIAG_FILENAME
    met_path = artifact_path / METRICS_FILENAME
    if not diag_path.exists() or not met_path.exists():
        raise RuntimeError(
            f"Expected outputs missing in {artifact_path}: {DIAG_FILENAME}, {METRICS_FILENAME}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    with diag_path.open("r", encoding="utf-8") as handle:
        diagnostics = json.load(handle)
    with met_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    return artifact_path, diagnostics, metrics


# =============================================================================
# Main sweep loop
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Random search with convergence safety checks for GRwHS."
    )
    parser.add_argument(
        "--base_config",
        type=str,
        required=True,
        help="Path to the base scenario YAML file.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=40,
        help="Maximum number of random prior+sampler proposals to evaluate.",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=2,
        help="Number of independent chains per candidate.",
    )
    parser.add_argument(
        "--work_root",
        type=str,
        default="outputs/random_search",
        help="Directory where sweep artifacts will be stored.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed controlling the random sweep.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Faster sweep: restrict iters to {10000,12000} and use +1000 on C-escalation.",
    )
    parser.add_argument(
        "--stop_after_accepted",
        type=int,
        default=0,
        help="Stop early once this many candidates have converged and been accepted.",
    )

    args = parser.parse_args()
    random.seed(args.seed)

    base_cfg_path = Path(args.base_config).resolve()
    work_root = Path(args.work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    best = {
        "score": -1e18,
        "run_dir": None,
        "overrides": None,
        "metrics": None,
    }

    tried = 0
    accepted = 0

    for trial in range(1, args.budget + 1):
        # Draw candidate satisfying hard rules.
        for _ in range(200):
            overrides = sample_search_space(args.fast)
            rejected, reason = violates_hard_rules(overrides)
            if not rejected:
                break
        else:
            print("[WARN] Unable to sample a valid configuration after 200 attempts.")
            continue

        tried += 1
        print(f"\n=== Trial {trial}/{args.budget} ===")
        print(json.dumps(overrides, indent=2, default=float))

        chain_success = True
        chain_metrics: List[Dict[str, Any]] = []
        chain_dirs: List[str] = []

        for chain_idx in range(args.chains):
            base_gibbs = deepcopy(overrides["inference"]["gibbs"])
            chain_seed = 1000 + chain_idx
            overrides["inference"]["gibbs"]["seed"] = chain_seed

            candidate_queue: List[Tuple[Dict[str, Any], str]] = [
                (deepcopy(overrides), "base")
            ]
            for esc_cfg, esc_label in escalate_sampler(base_gibbs, args.fast):
                esc_override = deepcopy(overrides)
                esc_override["inference"]["gibbs"].update(esc_cfg)
                candidate_queue.append((esc_override, esc_label))

            chain_ok = False
            last_reason = ""

            for cand_cfg, label in candidate_queue:
                tag = f"trial{trial:03d}_chain{chain_idx+1}_{label}"
                try:
                    run_dir, diagnostics, metrics = run_once(
                        cand_cfg, base_cfg_path, work_root, tag
                    )
                except Exception as exc:
                    last_reason = f"execution error: {exc}"
                    print(f"  [Chain {chain_idx+1}] {last_reason}")
                    continue

                ok, reason = check_convergence(diagnostics)
                if ok:
                    chain_ok = True
                    chain_dirs.append(str(run_dir))
                    chain_metrics.append(metrics)
                    print(f"  [Chain {chain_idx+1}] converged with {label} @ {run_dir}")
                    break

                last_reason = reason
                print(
                    f"  [Chain {chain_idx+1}] failed convergence ({reason}); trying escalation."
                )

            if not chain_ok:
                print(
                    f"  [Chain {chain_idx+1}] exhausted escalations without convergence ({last_reason})."
                )
                chain_success = False
                break

        if not chain_success:
            print("-> Discarding configuration due to failed convergence.")
            continue

        accepted += 1
        metrics = chain_metrics[0]
        score = score_from_metrics(metrics)
        print(f"-> ACCEPTED configuration. Score={score:.4f}")

        if score > best["score"]:
            best = {
                "score": score,
                "run_dir": chain_dirs[0],
                "overrides": deepcopy(overrides),
                "metrics": metrics,
            }
            print(f"   New best score achieved at {chain_dirs[0]}")

        # Early stopping if enough accepted candidates were found
        if args.stop_after_accepted and accepted >= args.stop_after_accepted:
            print("\nReached stop_after_accepted target; ending sweep early.")
            break

    print("\n================ SUMMARY ================")
    print(f"Trials attempted: {tried}")
    print(f"Converged candidates: {accepted}")

    if best["run_dir"] is None:
        print("No converged configuration outperformed the baseline.")
        return

    print(f"\nBest score: {best['score']:.4f}")
    print(f"Best run directory: {best['run_dir']}")
    print("Best overrides:")
    print(json.dumps(best["overrides"], indent=2, default=float))

    summary = {
        "trials": tried,
        "accepted": accepted,
        "best_score": best["score"],
        "best_run_dir": best["run_dir"],
        "best_overrides": best["overrides"],
        "best_metrics": best["metrics"],
    }
    summary_path = work_root / "random_search_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
