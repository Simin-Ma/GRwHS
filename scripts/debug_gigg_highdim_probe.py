from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation_project.src.core.models.gigg_regression import GIGGRegression
from simulation_project.src.experiments.methods.fit_gigg import fit_gigg_mmle
from simulation_project.src.utils import SamplerConfig
from simulation_second.src.config import load_benchmark_config
from simulation_second.src.dataset import generate_grouped_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Tiny high-dimensional GIGG_MMLE probe.")
    parser.add_argument("--config", default="Simulation_highdimension/config/highdimension.yaml")
    parser.add_argument("--setting-id", default="hd_setting_1_classical_anchor")
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--iter-mult", type=int, default=1)
    parser.add_argument("--iter-floor", type=int, default=10)
    parser.add_argument("--iter-cap", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--manual-two-stage", action="store_true")
    parser.add_argument("--manual-stage-a-burnin", type=int, default=2)
    parser.add_argument("--manual-stage-a-draws", type=int, default=2)
    parser.add_argument("--manual-stage-b-burnin", type=int, default=2)
    parser.add_argument("--manual-stage-b-draws", type=int, default=2)
    parser.add_argument("--mmle-samp-size", type=int, default=8)
    parser.add_argument("--mmle-tol-scale", type=float, default=1e-2)
    parser.add_argument("--mmle-max-iters", type=int, default=16)
    args = parser.parse_args()

    cfg = load_benchmark_config(args.config)
    setting = cfg.setting_map()[str(args.setting_id)]
    dataset = generate_grouped_dataset(
        setting,
        replicate_id=int(args.replicate),
        master_seed=int(cfg.runner.seed),
        family_specs=cfg.families,
    )

    sampler = SamplerConfig(
        warmup=int(args.warmup),
        post_warmup_draws=int(args.draws),
        adapt_delta=0.9,
        max_treedepth=10,
        chains=int(args.chains),
    )

    print(
        json.dumps(
            {
                "setting_id": setting.setting_id,
                "shape": list(map(int, dataset.X_train.shape)),
                "chains": int(args.chains),
                "warmup": int(args.warmup),
                "draws": int(args.draws),
                "iter_mult": int(args.iter_mult),
                "iter_floor": int(args.iter_floor),
                "iter_cap": int(args.iter_cap),
            },
            indent=2,
        )
    )

    start = time.perf_counter()
    try:
        if args.manual_two_stage:
            stage_a = GIGGRegression(
                method="mmle",
                n_burn_in=int(args.manual_stage_a_burnin),
                n_samples=int(args.manual_stage_a_draws),
                n_thin=1,
                seed=int(args.seed),
                num_chains=1,
                fit_intercept=False,
                store_lambda=True,
                tau_sq_init=1.0,
                btrick=True,
                stable_solve=True,
                mmle_burnin_only=False,
                mmle_highdim_fastpath=True,
                init_strategy="zero",
                init_ridge=1.0,
                init_scale_blend=0.0,
                randomize_group_order=False,
                lambda_vectorized_update=True,
                extra_beta_refresh_prob=0.0,
                mmle_step_size=1.0,
                mmle_update_every=1,
                mmle_window=1,
                lambda_constraint_mode="none",
                q_constraint_mode="hard",
                progress_bar=False,
            )
            stage_a.fit(dataset.X_train, dataset.y_train, groups=dataset.groups, method="mmle")
            b_hat = np.asarray(stage_a.b_mean_, dtype=float).reshape(-1)
            a_hat = np.full(len(dataset.groups), 1.0 / max(int(dataset.X_train.shape[0]), 1), dtype=float)
            stage_b = GIGGRegression(
                method="fixed",
                n_burn_in=int(args.manual_stage_b_burnin),
                n_samples=int(args.manual_stage_b_draws),
                n_thin=1,
                seed=int(args.seed),
                num_chains=int(args.chains),
                fit_intercept=False,
                store_lambda=True,
                a_value=None,
                b_init=float(np.mean(b_hat)) if b_hat.size else 0.5,
                tau_sq_init=1.0,
                sigma_sq_init=1.0,
                btrick=True,
                stable_solve=True,
                init_strategy="zero",
                init_ridge=1.0,
                init_scale_blend=0.0,
                randomize_group_order=True,
                locals_first_beta_update=False,
                extra_local_scale_sweeps=0,
                lambda_vectorized_update=True,
                extra_beta_refresh_prob=0.0,
                lambda_constraint_mode="none",
                q_constraint_mode="hard",
                progress_bar=False,
            )
            fitted = stage_b.fit(
                dataset.X_train,
                dataset.y_train,
                groups=dataset.groups,
                a=a_hat,
                b=b_hat,
                method="fixed",
            )
            payload = {
                "elapsed_seconds": float(time.perf_counter() - start),
                "mode": "manual_two_stage",
                "stage_a_b_mean_finite": bool(np.all(np.isfinite(b_hat))),
                "coef_samples_shape": None if fitted.coef_samples_ is None else list(np.asarray(fitted.coef_samples_).shape),
                "gamma2_samples_shape": None if fitted.gamma2_samples_ is None else list(np.asarray(fitted.gamma2_samples_).shape),
                "lambda_samples_shape": None if fitted.lambda_samples_ is None else list(np.asarray(fitted.lambda_samples_).shape),
            }
        else:
            result = fit_gigg_mmle(
                dataset.X_train,
                dataset.y_train,
                dataset.groups,
                task="gaussian",
                seed=int(args.seed),
                sampler=sampler,
                p0=int(np.count_nonzero(dataset.beta)),
                iter_mult=int(args.iter_mult),
                iter_floor=int(args.iter_floor),
                iter_cap=int(args.iter_cap),
                mmle_samp_size=int(args.mmle_samp_size),
                mmle_tol_scale=float(args.mmle_tol_scale),
                mmle_max_iters=int(args.mmle_max_iters),
                exact_highdim_fastpath=True,
                no_retry=True,
                progress_bar=False,
            )
            payload = {
                "elapsed_seconds": float(time.perf_counter() - start),
                "mode": "fit_gigg_mmle",
                "status": str(result.status),
                "converged": bool(result.converged),
                "runtime_seconds": None if result.runtime_seconds is None else float(result.runtime_seconds),
                "rhat_max": None if result.rhat_max is None else float(result.rhat_max),
                "bulk_ess_min": None if result.bulk_ess_min is None else float(result.bulk_ess_min),
                "has_beta_mean": result.beta_mean is not None,
                "has_beta_draws": result.beta_draws is not None,
                "diagnostics": result.diagnostics,
            }
    except Exception as exc:
        elapsed = time.perf_counter() - start
        print(f"UNCAUGHT_EXCEPTION after {elapsed:.3f}s: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 2

    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
