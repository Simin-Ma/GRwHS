import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from simulation_project.src.dgp_grouped_linear import generate_orthonormal_block_design, sigma2_for_target_snr
from simulation_project.src.fit_gigg import fit_gigg_mmle
from simulation_project.src.run_experiment import _build_benchmark_beta, _scale_gigg_config_for_retry, _scale_sampler_for_retry
from simulation_project.src.utils import SamplerConfig, canonical_groups, experiment_seed


GROUP_SIZES = [5, 5, 5, 5, 5]
N_TRAIN = 100
N_TEST = 30
TARGET_SNR = 2.0
SIGNALS = ["concentrated", "distributed"]
MASTER_SEED = 20260415


@dataclass
class Strategy:
    name: str
    sampler: SamplerConfig
    gigg: dict[str, Any]
    retries: int


def make_dataset(signal: str, replicate: int) -> tuple[np.ndarray, np.ndarray, list[list[int]], int]:
    sid = 1 if signal == "concentrated" else 2
    s = experiment_seed(3, sid, replicate, master_seed=MASTER_SEED)
    beta0 = _build_benchmark_beta(signal, GROUP_SIZES)
    p = int(sum(GROUP_SIZES))
    X_train = generate_orthonormal_block_design(n=N_TRAIN, group_sizes=GROUP_SIZES, seed=s)
    cov_x = np.eye(p, dtype=float)
    sigma2 = sigma2_for_target_snr(beta=beta0, cov_x=cov_x, target_snr=float(TARGET_SNR))
    rng_y = np.random.default_rng(s + 17)
    y_train = X_train @ beta0 + rng_y.normal(0.0, np.sqrt(sigma2), N_TRAIN)
    groups = canonical_groups(GROUP_SIZES)
    p0 = int(np.sum(np.abs(beta0) > 1e-12))
    return X_train, y_train, groups, p0


def run_one_strategy(strategy: Strategy, repeats: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    for signal in SIGNALS:
        for r in range(1, repeats + 1):
            X, y, groups, p0 = make_dataset(signal, r)
            success = False
            total_runtime = 0.0
            final_ess = float("nan")
            final_rhat = float("nan")
            attempts = 0

            for attempt in range(max(0, int(strategy.retries)) + 1):
                attempts = attempt + 1
                sampler_try = _scale_sampler_for_retry(strategy.sampler, attempt)
                gigg_try = _scale_gigg_config_for_retry(dict(strategy.gigg), attempt)
                seed = experiment_seed(33, 11 if signal == "concentrated" else 12, 100 * r + attempt, master_seed=MASTER_SEED)

                res = fit_gigg_mmle(
                    X,
                    y,
                    groups,
                    task="gaussian",
                    seed=int(seed),
                    sampler=sampler_try,
                    p0=p0,
                    **gigg_try,
                )
                total_runtime += float(res.runtime_seconds) if np.isfinite(res.runtime_seconds) else 0.0
                final_ess = float(res.bulk_ess_min)
                final_rhat = float(res.rhat_max)

                if res.status == "ok" and bool(res.converged):
                    success = True
                    break

            rows.append(
                {
                    "signal": signal,
                    "replicate": int(r),
                    "converged": bool(success),
                    "attempts": int(attempts),
                    "runtime_seconds": float(total_runtime),
                    "ess_last": float(final_ess),
                    "rhat_last": float(final_rhat),
                }
            )

    rt = np.array([x["runtime_seconds"] for x in rows], dtype=float)
    ess = np.array([x["ess_last"] for x in rows], dtype=float)
    conv = np.array([1.0 if x["converged"] else 0.0 for x in rows], dtype=float)
    att = np.array([x["attempts"] for x in rows], dtype=float)

    by_signal = {}
    for sig in SIGNALS:
        ss = [x for x in rows if x["signal"] == sig]
        c = np.mean([1.0 if x["converged"] else 0.0 for x in ss]) if ss else float("nan")
        t = np.median([x["runtime_seconds"] for x in ss]) if ss else float("nan")
        e = np.median([x["ess_last"] for x in ss]) if ss else float("nan")
        by_signal[sig] = {"conv_rate": float(c), "runtime_median": float(t), "ess_median": float(e)}

    return {
        "name": strategy.name,
        "n_tasks": int(len(rows)),
        "conv_rate": float(np.mean(conv)),
        "runtime_median": float(np.median(rt)),
        "runtime_p90": float(np.quantile(rt, 0.9)),
        "ess_median": float(np.nanmedian(ess)),
        "ess_p90": float(np.nanquantile(ess, 0.9)),
        "attempts_mean": float(np.mean(att)),
        "by_signal": by_signal,
        "rows": rows,
    }


def make_strategies() -> list[Strategy]:
    base_sampler = SamplerConfig(
        chains=1,
        warmup=250,
        post_warmup_draws=250,
        adapt_delta=0.92,
        max_treedepth=10,
        strict_adapt_delta=0.97,
        strict_max_treedepth=12,
        max_divergence_ratio=0.01,
        rhat_threshold=1.03,
        ess_threshold=120.0,
    )
    base_gigg = {
        "iter_mult": 2,
        "iter_floor": 500,
        "iter_cap": 1500,
        "btrick": True,
        "mmle_burnin_only": True,
        "init_strategy": "ridge",
        "init_ridge": 1.0,
        "init_scale_blend": 0.5,
        "randomize_group_order": False,
        "lambda_vectorized_update": False,
        "extra_beta_refresh_prob": 0.0,
    }

    return [
        Strategy("S0_baseline", base_sampler, dict(base_gigg), retries=0),
        Strategy("S1_more_draws_1chain", SamplerConfig(**{**base_sampler.__dict__, "warmup": 500, "post_warmup_draws": 500}), dict(base_gigg), retries=0),
        Strategy("S2_more_draws_1chain_more_iter", SamplerConfig(**{**base_sampler.__dict__, "warmup": 500, "post_warmup_draws": 500}), {**base_gigg, "iter_mult": 3, "iter_floor": 1000, "iter_cap": 3000}, retries=0),
        Strategy("S3_two_chains", SamplerConfig(**{**base_sampler.__dict__, "chains": 2}), dict(base_gigg), retries=0),
        Strategy("S4_two_chains_more_draws", SamplerConfig(**{**base_sampler.__dict__, "chains": 2, "warmup": 500, "post_warmup_draws": 500}), dict(base_gigg), retries=0),
        Strategy("S5_retry1", base_sampler, dict(base_gigg), retries=1),
        Strategy("S6_retry2", base_sampler, dict(base_gigg), retries=2),
        Strategy("S7_mmle_full", base_sampler, {**base_gigg, "mmle_burnin_only": False}, retries=0),
        Strategy("S8_block_mix", base_sampler, {**base_gigg, "randomize_group_order": True, "lambda_vectorized_update": True, "extra_beta_refresh_prob": 0.25}, retries=0),
        Strategy("S9_no_btrick", base_sampler, {**base_gigg, "btrick": False}, retries=0),
    ]


def sort_key(item: dict[str, Any]) -> tuple:
    return (-float(item["conv_rate"]), float(item["runtime_median"]))


def main() -> None:
    out_dir = Path(r"D:\FilesP\GR-RHS\ab_runs\gigg_conv_search")
    out_dir.mkdir(parents=True, exist_ok=True)

    strategies = make_strategies()

    t0 = time.time()
    stage1 = []
    for s in strategies:
        stage1.append(run_one_strategy(s, repeats=2))
    stage1 = sorted(stage1, key=sort_key)

    top_names = [x["name"] for x in stage1[:4]]
    top_map = {s.name: s for s in strategies if s.name in set(top_names)}

    stage2 = []
    for name in top_names:
        stage2.append(run_one_strategy(top_map[name], repeats=5))
    stage2 = sorted(stage2, key=sort_key)

    out = {
        "elapsed_seconds": float(time.time() - t0),
        "stage1_repeats": 2,
        "stage2_repeats": 5,
        "stage1": stage1,
        "stage2": stage2,
        "top_names": top_names,
    }

    p = out_dir / "search_results.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
