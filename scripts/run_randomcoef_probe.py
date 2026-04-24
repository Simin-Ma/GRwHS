from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation_project.src.experiments.analysis.metrics import mse_null_signal_overall
from simulation_project.src.experiments.exp3 import _build_benchmark_beta
from simulation_project.src.experiments.fitting import _fit_all_methods
from simulation_project.src.utils import MASTER_SEED, SamplerConfig, canonical_groups, ensure_dir, sample_correlated_design


def _sigma2_for_target_snr(beta: np.ndarray, cov_x: np.ndarray, target_snr: float) -> float:
    signal_var = float(beta.T @ cov_x @ beta)
    return max(signal_var / max(float(target_snr), 1e-8), 1e-8)


def _count_active_groups(beta: np.ndarray, group_sizes: list[int]) -> int:
    groups = canonical_groups(group_sizes)
    return int(
        sum(
            1
            for group in groups
            if np.any(np.abs(np.asarray(beta, dtype=float)[np.asarray(group, dtype=int)]) > 1e-12)
        )
    )


def _single_replicate(
    *,
    n: int,
    group_sizes: list[int],
    rho_within: float,
    rho_between: float,
    target_snr: float,
    methods: list[str],
    seed: int,
    sampler: SamplerConfig,
    enforce_bayes_convergence: bool,
    max_convergence_retries: int,
    grrhs_kwargs: dict[str, Any] | None,
    gigg_config: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    beta0 = _build_benchmark_beta(
        signal="random_coefficient",
        group_sizes=group_sizes,
        group_cfg={"paper_random_coefficients": True},
        rng=rng,
        rho_within=float(rho_within),
    )
    X, cov_x = sample_correlated_design(
        n=int(n),
        group_sizes=group_sizes,
        rho_within=float(rho_within),
        rho_between=float(rho_between),
        seed=int(seed),
    )
    sigma2 = _sigma2_for_target_snr(beta0, cov_x, target_snr=float(target_snr))
    y = X @ beta0 + rng.normal(loc=0.0, scale=math.sqrt(float(sigma2)), size=int(n))
    groups = canonical_groups(group_sizes)

    results = _fit_all_methods(
        X,
        y,
        groups,
        task="gaussian",
        seed=int(seed),
        p0=int(np.sum(np.abs(beta0) > 1e-12)),
        p0_groups=_count_active_groups(beta0, group_sizes),
        sampler=sampler,
        grrhs_kwargs=dict(grrhs_kwargs or {}),
        methods=methods,
        gigg_config=dict(gigg_config or {}),
        enforce_bayes_convergence=bool(enforce_bayes_convergence),
        max_convergence_retries=int(max_convergence_retries),
        method_jobs=1,
    )

    rows: list[dict[str, Any]] = []
    for method, res in results.items():
        metric = {
            "mse_overall": float("nan"),
            "mse_signal": float("nan"),
            "mse_null": float("nan"),
        }
        if res.beta_mean is not None:
            metric = mse_null_signal_overall(res.beta_mean, beta0)

        mmle_q = None
        if isinstance(res.diagnostics, dict):
            mmle_est = res.diagnostics.get("mmle_estimate")
            if isinstance(mmle_est, dict):
                q_vals = mmle_est.get("q_estimate")
                if isinstance(q_vals, list):
                    mmle_q = [float(v) for v in q_vals]

        rows.append(
            {
                "method": str(method),
                "status": str(res.status),
                "converged": bool(res.converged),
                "runtime_seconds": float(res.runtime_seconds),
                "mse_overall": float(metric["mse_overall"]),
                "mse_signal": float(metric["mse_signal"]),
                "mse_null": float(metric["mse_null"]),
                "beta_nonzero_count": int(np.sum(np.abs(beta0) > 1e-12)),
                "sigma2": float(sigma2),
                "mmle_q_estimate_json": json.dumps(mmle_q) if mmle_q is not None else "",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe methods on the project paper-style random-coefficient grouped regression task."
    )
    parser.add_argument("--scenario", choices=["lowdim", "highdim"], default="lowdim")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--target-snr", type=float, default=0.7 / 0.3)
    parser.add_argument("--rho-within", type=float, default=0.8)
    parser.add_argument("--rho-between", type=float, default=0.2)
    parser.add_argument(
        "--coef-mode",
        choices=["paper", "gaussian", "mixed"],
        default="paper",
        help="Deprecated compatibility flag. The probe always uses the paper random-coefficient generator.",
    )
    parser.add_argument(
        "--signal-scale",
        type=float,
        default=0.45,
        help="Deprecated compatibility flag kept for older command lines; ignored by the paper generator.",
    )
    parser.add_argument("--methods", type=str, default="GR_RHS,GIGG_MMLE,GIGG_GHS,RHS,GHS_plus,OLS,LASSO_CV")
    parser.add_argument("--seed", type=int, default=MASTER_SEED + 9000)
    parser.add_argument("--save-dir", type=str, default="outputs/randomcoef_probe")
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=250)
    parser.add_argument("--draws", type=int, default=250)
    parser.add_argument("--ess-threshold", type=float, default=100.0)
    parser.add_argument("--max-convergence-retries", type=int, default=0)
    parser.add_argument("--grrhs-backend", choices=["gibbs_staged", "nuts"], default="gibbs_staged")
    parser.add_argument("--grrhs-tau-target", choices=["coefficients", "groups"], default="coefficients")
    parser.add_argument("--gigg-q-constraint-mode", choices=["hard", "soft", "none"], default="hard")
    parser.add_argument("--gigg-b-max", type=float, default=4.0)
    parser.add_argument("--gigg-mmle-step-size", type=float, default=1.0)
    parser.add_argument("--no-enforce-bayes-convergence", action="store_true")
    args = parser.parse_args()

    if args.scenario == "lowdim":
        n = 500
        group_sizes = [10, 10, 10, 10, 10]
    else:
        n = 200
        group_sizes = [10] * 50

    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    sampler = SamplerConfig(
        chains=int(args.chains),
        warmup=int(args.warmup),
        post_warmup_draws=int(args.draws),
        ess_threshold=float(args.ess_threshold),
    )
    grrhs_kwargs = {
        "tau_target": str(args.grrhs_tau_target),
        "sampler_backend": str(args.grrhs_backend),
        "progress_bar": False,
    }
    gigg_config = {
        "q_constraint_mode": str(args.gigg_q_constraint_mode),
        "b_max": float(args.gigg_b_max),
        "mmle_step_size": float(args.gigg_mmle_step_size),
    }

    out_dir = ensure_dir(Path(args.save_dir) / f"{args.scenario}_{args.coef_mode}")
    raw_rows: list[dict[str, Any]] = []

    for rep in range(int(args.repeats)):
        rep_seed = int(args.seed) + 1000 * rep
        rep_rows = _single_replicate(
            n=int(n),
            group_sizes=list(group_sizes),
            rho_within=float(args.rho_within),
            rho_between=float(args.rho_between),
            target_snr=float(args.target_snr),
            methods=list(methods),
            seed=rep_seed,
            sampler=sampler,
            enforce_bayes_convergence=not bool(args.no_enforce_bayes_convergence),
            max_convergence_retries=int(args.max_convergence_retries),
            grrhs_kwargs=grrhs_kwargs,
            gigg_config=gigg_config,
        )
        for row in rep_rows:
            row["replicate_id"] = int(rep)
            row["scenario"] = str(args.scenario)
            row["coef_mode"] = "paper"
            raw_rows.append(row)

    import pandas as pd

    raw = pd.DataFrame(raw_rows)
    raw_path = out_dir / "raw_results.csv"
    raw.to_csv(raw_path, index=False)

    summary = (
        raw.groupby("method", as_index=False)
        .agg(
            n_runs=("method", "count"),
            n_ok=("status", lambda s: int(np.sum(np.asarray(s) == "ok"))),
            n_converged=("converged", lambda s: int(np.sum(np.asarray(s, dtype=bool)))),
            runtime_mean=("runtime_seconds", "mean"),
            mse_null_mean=("mse_null", "mean"),
            mse_signal_mean=("mse_signal", "mean"),
            mse_overall_mean=("mse_overall", "mean"),
        )
        .sort_values(["mse_overall_mean", "mse_null_mean"], ascending=[True, True])
    )
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    meta = {
        "scenario": str(args.scenario),
        "coef_mode": "paper",
        "repeats": int(args.repeats),
        "n": int(n),
        "p": int(sum(group_sizes)),
        "group_sizes": list(group_sizes),
        "signal_mechanism": "paper_random_coefficient",
        "paper_random_design": {
            "group_size": 10,
            "first_group_assignment": "concentrated_or_distributed_with_equal_probability",
            "remaining_group_probabilities": {
                "concentrated": 0.2,
                "distributed": 0.2,
                "null": 0.6,
            },
        },
        "rho_within": float(args.rho_within),
        "rho_between": float(args.rho_between),
        "target_snr": float(args.target_snr),
        "deprecated_args_ignored": {
            "coef_mode": str(args.coef_mode),
            "signal_scale": float(args.signal_scale),
        },
        "methods": list(methods),
        "sampler": {
            "chains": int(args.chains),
            "warmup": int(args.warmup),
            "draws": int(args.draws),
            "ess_threshold": float(args.ess_threshold),
        },
        "max_convergence_retries": int(args.max_convergence_retries),
        "grrhs_kwargs": grrhs_kwargs,
        "gigg_config": gigg_config,
        "raw_results": str(raw_path),
        "summary": str(summary_path),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(summary.to_string(index=False))
    print(f"\nSaved raw results to: {raw_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
