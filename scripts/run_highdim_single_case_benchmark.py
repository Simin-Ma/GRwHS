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
from simulation_project.src.experiments.fitting import _fit_all_methods
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
    if not math.isfinite(val):
        return None
    return val


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one high-dimensional setting x method benchmark case.")
    parser.add_argument("--config", default="Simulation_highdimension/config/highdimension.yaml")
    parser.add_argument("--setting-id", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--outdir", default="tmp/highdim_single_case_runs")
    args = parser.parse_args()

    cfg = load_benchmark_config(args.config)
    setting = cfg.setting_map()[str(args.setting_id)]
    ds = generate_grouped_dataset(
        setting,
        replicate_id=int(args.replicate),
        master_seed=int(cfg.runner.seed),
        family_specs=cfg.families,
    )
    p0 = int((abs(ds.beta) > 1e-12).sum())
    p0_groups = int(sum(any(abs(ds.beta[idx]) > 1e-12 for idx in g) for g in ds.groups))
    sampler = SamplerConfig(
        chains=int(cfg.convergence_gate.chains),
        warmup=int(cfg.convergence_gate.warmup),
        post_warmup_draws=int(cfg.convergence_gate.post_warmup_draws),
        adapt_delta=float(cfg.convergence_gate.adapt_delta),
        max_treedepth=int(cfg.convergence_gate.max_treedepth),
        strict_adapt_delta=float(cfg.convergence_gate.strict_adapt_delta),
        strict_max_treedepth=int(cfg.convergence_gate.strict_max_treedepth),
        rhat_threshold=float(cfg.convergence_gate.rhat_threshold),
        ess_threshold=float(cfg.convergence_gate.ess_threshold),
        max_divergence_ratio=float(cfg.convergence_gate.max_divergence_ratio),
    )

    t0 = time.perf_counter()
    result = _fit_all_methods(
        ds.X_train,
        ds.y_train,
        ds.groups,
        task=str(cfg.runner.task),
        seed=int(cfg.runner.seed),
        p0=int(p0),
        p0_groups=int(p0_groups),
        sampler=sampler,
        rhs_kwargs=dict(cfg.methods.rhs_kwargs),
        grrhs_kwargs=dict(cfg.methods.grrhs_kwargs),
        methods=[str(args.method)],
        gigg_config=dict(cfg.methods.gigg_config),
        bayes_min_chains=int(cfg.convergence_gate.bayes_min_chains),
        enforce_bayes_convergence=bool(cfg.convergence_gate.enforce_bayes_convergence),
        max_convergence_retries=int(cfg.convergence_gate.max_convergence_retries),
        method_jobs=1,
        rhs_sampler_strategy="high_dim",
    )[str(args.method)]
    wall_seconds = time.perf_counter() - t0

    metrics = _evaluate_row(
        result,
        ds.beta,
        X_train=ds.X_train,
        y_train=ds.y_train,
        X_test=ds.X_test,
        y_test=ds.y_test,
    )

    payload = {
        "replicate": int(args.replicate),
        "setting_id": str(args.setting_id),
        "method": str(args.method),
        "wall_seconds": float(wall_seconds),
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
        "attempts_used": int((((result.diagnostics or {}).get("convergence_retry", {}) or {}).get("attempts_used", 1))),
        "error": str(getattr(result, "error", "") or ""),
        "diagnostics": _json_clean(result.diagnostics),
    }

    outdir = ROOT / str(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.setting_id}__{args.method}__r{int(args.replicate)}"
    out_path = outdir / f"{stem}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out_path": str(out_path), **payload}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
