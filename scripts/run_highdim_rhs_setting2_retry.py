from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation_project.src.experiments.evaluation import _evaluate_row
from simulation_project.src.experiments.methods.fit_rhs import fit_rhs
from simulation_project.src.utils import SamplerConfig
from simulation_second.src.config import load_benchmark_config
from simulation_second.src.dataset import generate_grouped_dataset


def _json_scalar(value):
    if value is None:
        return None
    try:
        val = float(value)
    except Exception:
        return value
    return val if math.isfinite(val) else None


def _json_clean(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _json_clean(value.item())
        except Exception:
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (int, str, bool)):
        return value
    try:
        val = float(value)
    except Exception:
        return str(value)
    return val if math.isfinite(val) else None


def run_rhs_retry(
    *,
    config: str,
    setting_id: str,
    replicate: int,
    outdir: str,
    warmup: int,
    draws: int,
    chains: int,
    adapt_delta: float,
    max_treedepth: int,
    seed_offset: int,
) -> dict[str, object]:
    total_t0 = time.perf_counter()
    cfg = load_benchmark_config(config)
    setting = cfg.setting_map()[str(setting_id)]
    ds = generate_grouped_dataset(
        setting,
        replicate_id=int(replicate),
        master_seed=int(cfg.runner.seed),
        family_specs=cfg.families,
    )
    p0 = int((abs(ds.beta) > 1e-12).sum())
    sampler = SamplerConfig(
        chains=int(chains),
        warmup=int(warmup),
        post_warmup_draws=int(draws),
        adapt_delta=float(adapt_delta),
        max_treedepth=int(max_treedepth),
        strict_adapt_delta=max(0.999, float(adapt_delta)),
        strict_max_treedepth=max(int(max_treedepth), 16),
        max_divergence_ratio=float(cfg.convergence_gate.max_divergence_ratio),
        rhat_threshold=float(cfg.convergence_gate.rhat_threshold),
        ess_threshold=float(cfg.convergence_gate.ess_threshold),
    )

    fit_t0 = time.perf_counter()
    result = fit_rhs(
        ds.X_train,
        ds.y_train,
        ds.groups,
        task=str(cfg.runner.task),
        seed=int(cfg.runner.seed) + 2 + 10000 * int(seed_offset),
        p0=int(p0),
        sampler=sampler,
        method_name="RHS",
        progress_bar=False,
    )
    fit_seconds = time.perf_counter() - fit_t0

    metrics = _evaluate_row(
        result,
        ds.beta,
        X_train=ds.X_train,
        y_train=ds.y_train,
        X_test=ds.X_test,
        y_test=ds.y_test,
    )
    payload = {
        "replicate": int(replicate),
        "setting_id": str(setting_id),
        "method": "RHS",
        "wall_seconds": float(time.perf_counter() - total_t0),
        "runtime_seconds": _json_scalar(result.runtime_seconds),
        "status": str(result.status),
        "converged": bool(result.converged),
        "rhat_max": _json_scalar(result.rhat_max),
        "ess_min": _json_scalar(result.bulk_ess_min),
        "divergence_ratio": _json_scalar(result.divergence_ratio),
        "mse_overall": _json_scalar(metrics.get("mse_overall")),
        "mse_signal": _json_scalar(metrics.get("mse_signal")),
        "mse_null": _json_scalar(metrics.get("mse_null")),
        "coverage_95": _json_scalar(metrics.get("coverage_95")),
        "lpd_test": _json_scalar(metrics.get("lpd_test")),
        "attempts_used": 1,
        "error": str(getattr(result, "error", "") or ""),
        "diagnostics": _json_clean(result.diagnostics),
        "retry_budget": {
            "warmup": int(warmup),
            "draws": int(draws),
            "chains": int(chains),
            "adapt_delta": float(adapt_delta),
            "max_treedepth": int(max_treedepth),
            "seed_offset": int(seed_offset),
        },
        "timing_breakdown": {
            "fit_wrapper_seconds": float(fit_seconds),
            "fit_runtime_seconds": _json_scalar(result.runtime_seconds),
            "total_seconds": float(time.perf_counter() - total_t0),
        },
    }

    outdir_path = ROOT / str(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    stem = f"{setting_id}__RHS__r{int(replicate)}__w{int(warmup)}_d{int(draws)}_s{int(seed_offset)}"
    out_path = outdir_path / f"{stem}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"out_path": str(out_path), **payload}


def main() -> int:
    parser = argparse.ArgumentParser(description="Targeted strict RHS retry for high-dimensional setting 2.")
    parser.add_argument("--config", default="Simulation_highdimension/config/highdimension.yaml")
    parser.add_argument("--setting-id", default="hd_setting_2_single_mode")
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--outdir", default="tmp/highdim_rhs_setting2_retry_full")
    parser.add_argument("--warmup", type=int, default=4000)
    parser.add_argument("--draws", type=int, default=8000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--adapt-delta", type=float, default=0.995)
    parser.add_argument("--max-treedepth", type=int, default=15)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()

    payload = run_rhs_retry(
        config=str(args.config),
        setting_id=str(args.setting_id),
        replicate=int(args.replicate),
        outdir=str(args.outdir),
        warmup=int(args.warmup),
        draws=int(args.draws),
        chains=int(args.chains),
        adapt_delta=float(args.adapt_delta),
        max_treedepth=int(args.max_treedepth),
        seed_offset=int(args.seed_offset),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
