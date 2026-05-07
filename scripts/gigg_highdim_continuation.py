from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation_project.src.core.diagnostics.convergence import summarize_convergence
from simulation_project.src.core.models.gigg_regression import GIGGRegression
from simulation_project.src.utils import FitResult, save_fit_result_artifacts
from simulation_second.src.config import load_benchmark_config
from simulation_second.src.dataset import generate_grouped_dataset


def _json_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Continuation workflow for high-dimensional GIGG stage-B chains.")
    parser.add_argument("--config", default="Simulation_highdimension/config/highdimension.yaml")
    parser.add_argument("--setting-id", default="hd_setting_1_classical_anchor")
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--stage-a-burnin", type=int, default=4)
    parser.add_argument("--stage-a-draws", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--draws", type=int, default=2)
    parser.add_argument("--target-rhat", type=float, default=float("nan"))
    parser.add_argument("--min-rounds", type=int, default=1)
    parser.add_argument("--beta-jitter-scale", type=float, default=0.005)
    parser.add_argument("--lambda-log-jitter-scale", type=float, default=0.01)
    parser.add_argument("--gamma-log-jitter-scale", type=float, default=0.01)
    parser.add_argument("--use-stagea-quantiles", action="store_true")
    parser.add_argument("--tau-scale", type=float, default=1.0)
    parser.add_argument("--sigma-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--outdir", default="tmp/gigg_highdim_continuation")
    args = parser.parse_args()

    cfg = load_benchmark_config(args.config)
    setting = cfg.setting_map()[str(args.setting_id)]
    ds = generate_grouped_dataset(
        setting,
        replicate_id=int(args.replicate),
        master_seed=int(cfg.runner.seed),
        family_specs=cfg.families,
    )
    X = ds.X_train
    y = ds.y_train
    groups = ds.groups
    G = len(groups)
    outdir = ROOT / str(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stage_a_start = time.perf_counter()
    stage_a = GIGGRegression(
        method="mmle",
        n_burn_in=int(args.stage_a_burnin),
        n_samples=int(args.stage_a_draws),
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
        init_scale_blend=0.0,
        randomize_group_order=False,
        lambda_vectorized_update=True,
        lambda_constraint_mode="none",
        q_constraint_mode="hard",
        mmle_step_size=1.0,
        mmle_update_every=1,
        mmle_window=1,
        mmle_samp_size=2,
        mmle_tol_scale=1.0,
        mmle_max_iters=2,
        progress_bar=False,
    )
    fit_a = stage_a.fit(X, y, groups=groups, method="mmle")
    b_hat = np.asarray(fit_a.b_mean_, dtype=float).reshape(-1)
    a_hat = np.full(G, 1.0 / max(int(X.shape[0]), 1), dtype=float)
    stage_a_sec = time.perf_counter() - stage_a_start

    payloads: list[dict[str, np.ndarray] | None] = []
    beta_pool = np.asarray(fit_a.coef_samples_, dtype=float).reshape(-1, X.shape[1])
    lambda_pool = np.asarray(fit_a.lambda_samples_, dtype=float).reshape(-1, X.shape[1])
    gamma_pool = np.asarray(fit_a.gamma2_samples_, dtype=float).reshape(-1, G)
    if bool(args.use_stagea_quantiles) and beta_pool.shape[0] > 1:
        grid = np.linspace(0.2, 0.8, int(args.chains))
        pick = np.clip(np.round(grid * max(beta_pool.shape[0] - 1, 0)).astype(int), 0, beta_pool.shape[0] - 1)
        beta_seeds = beta_pool[pick]
        lambda_seeds = lambda_pool[np.clip(pick, 0, lambda_pool.shape[0] - 1)]
        gamma_seeds = gamma_pool[np.clip(pick, 0, gamma_pool.shape[0] - 1)]
    else:
        base_beta = np.asarray(beta_pool[-1], dtype=float).reshape(-1)
        base_lambda = np.asarray(lambda_pool[-1], dtype=float).reshape(-1)
        base_gamma = np.asarray(gamma_pool[-1], dtype=float).reshape(-1)
        beta_seeds = np.repeat(base_beta.reshape(1, -1), int(args.chains), axis=0)
        lambda_seeds = np.repeat(base_lambda.reshape(1, -1), int(args.chains), axis=0)
        gamma_seeds = np.repeat(base_gamma.reshape(1, -1), int(args.chains), axis=0)
    rng = np.random.default_rng(int(args.seed) + 9000)
    for chain_idx in range(int(args.chains)):
        beta_seed = np.asarray(beta_seeds[chain_idx], dtype=float).reshape(-1)
        lambda_seed = np.asarray(lambda_seeds[chain_idx], dtype=float).reshape(-1)
        gamma_seed = np.asarray(gamma_seeds[chain_idx], dtype=float).reshape(-1)
        payloads.append(
            {
                "beta_inits": beta_seed + rng.normal(0.0, float(args.beta_jitter_scale), size=beta_seed.shape[0]),
                "lambda_sq_inits": np.clip(
                    lambda_seed * np.exp(rng.normal(0.0, float(args.lambda_log_jitter_scale), size=lambda_seed.shape[0])),
                    1e-8,
                    1e3,
                ),
                "gamma_sq_inits": np.clip(
                    gamma_seed * np.exp(rng.normal(0.0, float(args.gamma_log_jitter_scale), size=gamma_seed.shape[0])),
                    1e-8,
                    1e3,
                ),
            }
        )

    chain_beta_rounds: list[list[np.ndarray]] = [[] for _ in range(int(args.chains))]
    history: list[dict[str, Any]] = []
    best_round_idx = -1
    best_rhat = float("inf")
    total_start = time.perf_counter()
    for round_idx in range(1, int(args.rounds) + 1):
        round_rec: dict[str, Any] = {"round": int(round_idx), "chains": []}
        for chain_idx in range(int(args.chains)):
            payload = payloads[chain_idx]
            t0 = time.perf_counter()
            model = GIGGRegression(
                method="fixed",
                n_burn_in=int(args.warmup),
                n_samples=int(args.draws),
                n_thin=1,
                seed=int(args.seed) + 1000 * int(round_idx) + chain_idx,
                num_chains=1,
                fit_intercept=False,
                store_lambda=True,
                a_value=None,
                b_init=float(np.mean(b_hat)),
                tau_sq_init=float(max(1.0 * float(args.tau_scale), 1e-8)),
                sigma_sq_init=float(max(1.0 * float(args.sigma_scale), 1e-8)),
                btrick=True,
                stable_solve=True,
                init_strategy="zero",
                init_scale_blend=0.0,
                randomize_group_order=False,
                locals_first_beta_update=False,
                extra_local_scale_sweeps=0,
                lambda_vectorized_update=True,
                extra_beta_refresh_prob=0.0,
                lambda_constraint_mode="none",
                q_constraint_mode="hard",
                mmle_highdim_fastpath=True,
                progress_bar=False,
            )
            fit = model.fit(
                X,
                y,
                groups=groups,
                a=a_hat,
                b=b_hat,
                method="fixed",
                beta_inits=None if payload is None else payload.get("beta_inits"),
                lambda_sq_inits=None if payload is None else payload.get("lambda_sq_inits"),
                gamma_sq_inits=None if payload is None else payload.get("gamma_sq_inits"),
            )
            wall = time.perf_counter() - t0
            draws = np.asarray(fit.coef_samples_, dtype=float)
            chain_beta_rounds[chain_idx].append(draws)
            payloads[chain_idx] = {
                "beta_inits": np.asarray(fit.coef_samples_[-1], dtype=float).reshape(-1),
                "lambda_sq_inits": np.asarray(fit.lambda_samples_[-1], dtype=float).reshape(-1),
                "gamma_sq_inits": np.asarray(fit.gamma2_samples_[-1], dtype=float).reshape(-1),
            }
            round_rec["chains"].append(
                {
                    "chain": int(chain_idx + 1),
                    "wall_seconds": float(wall),
                    "has_beta_draws": True,
                }
            )

        aligned = [np.concatenate(chain_beta_rounds[idx], axis=0) for idx in range(int(args.chains))]
        min_draws = min(arr.shape[0] for arr in aligned)
        aligned = [arr[:min_draws] for arr in aligned]
        beta_chains = np.stack(aligned, axis=0)
        conv = summarize_convergence({"beta": beta_chains})
        round_rec["merged_beta_diag"] = dict(conv.get("beta", {}))
        round_rec["merged_shape"] = [int(x) for x in beta_chains.shape]
        history.append(round_rec)
        rhat_now = float(round_rec["merged_beta_diag"].get("rhat_max", float("inf")))
        if np.isfinite(rhat_now) and rhat_now < best_rhat:
            best_rhat = rhat_now
            best_round_idx = len(history) - 1
        _json_dump({"history": history}, outdir / "continuation_history.json")
        if (
            np.isfinite(float(args.target_rhat))
            and round_idx >= int(args.min_rounds)
            and np.isfinite(rhat_now)
            and rhat_now <= float(args.target_rhat)
        ):
            break

    final_round_count = len(history)
    if best_round_idx >= 0:
        use_round_count = best_round_idx + 1
    else:
        use_round_count = final_round_count
    aligned = [
        np.concatenate(chain_beta_rounds[idx][:use_round_count], axis=0)
        for idx in range(int(args.chains))
    ]
    min_draws = min(arr.shape[0] for arr in aligned)
    aligned = [arr[:min_draws] for arr in aligned]
    beta_chains = np.stack(aligned, axis=0)
    conv = summarize_convergence({"beta": beta_chains})
    beta_mean = beta_chains.reshape(-1, beta_chains.shape[-1]).mean(axis=0)

    final = FitResult(
        method="GIGG_MMLE_continuation",
        status="ok",
        beta_mean=beta_mean,
        beta_draws=beta_chains,
        kappa_draws=None,
        group_scale_draws=None,
        tau_draws=None,
        runtime_seconds=float(time.perf_counter() - total_start + stage_a_sec),
        rhat_max=float(conv["beta"].get("rhat_max", float("nan"))),
        bulk_ess_min=float(conv["beta"].get("ess_min", float("nan"))),
        divergence_ratio=float("nan"),
        converged=False,
        diagnostics={
            "continuation_history": history,
            "convergence_detail": conv,
            "continuation_config": {
                "chains": int(args.chains),
                "rounds": int(args.rounds),
                "warmup_per_round": int(args.warmup),
                "draws_per_round": int(args.draws),
                "stage_a_burnin": int(args.stage_a_burnin),
                "stage_a_draws": int(args.stage_a_draws),
                "target_rhat": None if not np.isfinite(float(args.target_rhat)) else float(args.target_rhat),
                "min_rounds": int(args.min_rounds),
                "beta_jitter_scale": float(args.beta_jitter_scale),
                "lambda_log_jitter_scale": float(args.lambda_log_jitter_scale),
                "gamma_log_jitter_scale": float(args.gamma_log_jitter_scale),
                "use_stagea_quantiles": bool(args.use_stagea_quantiles),
                "tau_scale": float(args.tau_scale),
                "sigma_scale": float(args.sigma_scale),
                "best_round": int(best_round_idx + 1) if best_round_idx >= 0 else None,
                "final_round_executed": int(final_round_count),
                "rounds_used_for_final_artifact": int(use_round_count),
                "setting_id": str(args.setting_id),
            },
        },
    )
    written = save_fit_result_artifacts(outdir / "final", result=final, coefficient_truth=ds.beta)
    payload = {
        "artifact_dir": written.get("fit_dir", ""),
        "fit_summary": written.get("fit_summary", ""),
        "convergence_beta_summary": written.get("convergence_beta_summary", ""),
        "rhat_max": float(final.rhat_max),
        "ess_min": float(final.bulk_ess_min),
        "runtime_seconds": float(final.runtime_seconds),
        "merged_shape": [int(x) for x in beta_chains.shape],
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
