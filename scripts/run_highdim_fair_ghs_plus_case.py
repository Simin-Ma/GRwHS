from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation_project.src.core.models.baselines import GroupedHorseshoePlus
from simulation_project.src.experiments.evaluation import _evaluate_row
from simulation_project.src.experiments.methods.helpers import as_int_groups
from simulation_project.src.utils import (
    FitResult,
    SamplerConfig,
    diagnostics_summary_for_method,
    save_fit_result_artifacts,
    timed_call,
)
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


def _fit_fair_ghs_plus(
    X,
    y,
    groups,
    *,
    seed: int,
    sampler: SamplerConfig,
    tau0: float,
    sample_global_scale: bool,
    group_scale_prior: float,
    local_scale_prior: float,
    use_process_pool: bool,
) -> FitResult:
    model = GroupedHorseshoePlus(
        fit_intercept=True,
        tau0=float(tau0),
        group_scale_prior=float(group_scale_prior),
        local_scale_prior=float(local_scale_prior),
        sample_global_scale=bool(sample_global_scale),
        iters=int(sampler.warmup + sampler.post_warmup_draws),
        burnin=int(sampler.warmup),
        thin=1,
        seed=int(seed),
        num_chains=int(sampler.chains),
        progress_bar=False,
        use_process_pool=bool(use_process_pool),
    )
    model, runtime = timed_call(model.fit, X, y, groups=as_int_groups(groups))
    beta_draws = getattr(model, "coef_samples_", None)
    beta_mean = getattr(model, "coef_mean_", None)
    group_draws = getattr(model, "group_lambda_samples_", None)
    tau_draws = getattr(model, "tau_samples_", None)

    tracked = ["beta", "group_scale", "lambda", "sigma"]
    if bool(sample_global_scale):
        tracked.insert(1, "tau")
    rhat_max, ess_min, div_ratio, converged, details = diagnostics_summary_for_method(
        model=model,
        tracked_params=tracked,
        beta_draws=beta_draws,
        config=sampler,
    )
    diagnostics = dict(details or {})
    diagnostics["fair_ghs_plus_protocol"] = {
        "tracked_params": tracked,
        "route": "full_gibbs_formal" if bool(sample_global_scale) else "fixed_tau_gibbs_formal",
        "tau0": float(tau0),
        "sample_global_scale": bool(sample_global_scale),
        "group_scale_prior": float(group_scale_prior),
        "local_scale_prior": float(local_scale_prior),
        "iters": int(sampler.warmup + sampler.post_warmup_draws),
        "burnin": int(sampler.warmup),
        "post_warmup_draws": int(sampler.post_warmup_draws),
        "chains": int(sampler.chains),
        "rhat_threshold": float(sampler.rhat_threshold),
        "ess_threshold": float(sampler.ess_threshold),
    }

    return FitResult(
        method="GHS_plus_fair",
        status="ok",
        beta_mean=None if beta_mean is None else np.asarray(beta_mean, dtype=float),
        beta_draws=None if beta_draws is None else np.asarray(beta_draws, dtype=float),
        kappa_draws=None,
        group_scale_draws=None if group_draws is None else np.asarray(group_draws, dtype=float),
        tau_draws=None if tau_draws is None else np.asarray(tau_draws, dtype=float),
        runtime_seconds=float(runtime),
        rhat_max=float(rhat_max),
        bulk_ess_min=float(ess_min),
        divergence_ratio=float(div_ratio),
        converged=bool(converged),
        diagnostics=diagnostics,
    )


def run_case(
    *,
    config: str,
    setting_id: str,
    replicate: int,
    outdir: str,
    warmup: int,
    draws: int,
    tau0: float,
    tau0_auto: bool,
    sample_global_scale: bool,
    group_scale_prior: float,
    local_scale_prior: float,
    use_process_pool: bool,
    save_artifacts: bool = False,
) -> dict[str, object]:
    total_t0 = time.perf_counter()
    cfg = load_benchmark_config(config)
    setting = cfg.setting_map()[str(setting_id)]
    dataset_t0 = time.perf_counter()
    ds = generate_grouped_dataset(
        setting,
        replicate_id=int(replicate),
        master_seed=int(cfg.runner.seed),
        family_specs=cfg.families,
    )
    dataset_seconds = time.perf_counter() - dataset_t0
    p0 = int((np.abs(ds.beta) > 1e-12).sum())
    p = int(np.asarray(ds.beta).reshape(-1).size)
    n = int(ds.X_train.shape[0])
    tau0_use = float(tau0)
    if bool(tau0_auto):
        tau0_use = float((max(p0, 1) / max(p - max(p0, 1), 1)) / math.sqrt(max(n, 1)))

    sampler = SamplerConfig(
        chains=max(4, int(cfg.convergence_gate.chains)),
        warmup=int(warmup),
        post_warmup_draws=int(draws),
        adapt_delta=max(0.95, float(cfg.convergence_gate.adapt_delta)),
        max_treedepth=max(12, int(cfg.convergence_gate.max_treedepth)),
        strict_adapt_delta=max(0.99, float(cfg.convergence_gate.strict_adapt_delta)),
        strict_max_treedepth=max(14, int(cfg.convergence_gate.strict_max_treedepth)),
        max_divergence_ratio=min(0.005, float(cfg.convergence_gate.max_divergence_ratio)),
        rhat_threshold=min(1.015, float(cfg.convergence_gate.rhat_threshold)),
        ess_threshold=max(400.0, float(cfg.convergence_gate.ess_threshold)),
    )

    fit_t0 = time.perf_counter()
    result = _fit_fair_ghs_plus(
        ds.X_train,
        ds.y_train,
        ds.groups,
        seed=int(cfg.runner.seed) + 5,
        sampler=sampler,
        tau0=float(tau0_use),
        sample_global_scale=bool(sample_global_scale),
        group_scale_prior=float(group_scale_prior),
        local_scale_prior=float(local_scale_prior),
        use_process_pool=bool(use_process_pool),
    )
    fit_call_seconds = time.perf_counter() - fit_t0

    eval_t0 = time.perf_counter()
    metrics = _evaluate_row(
        result,
        ds.beta,
        X_train=ds.X_train,
        y_train=ds.y_train,
        X_test=ds.X_test,
        y_test=ds.y_test,
    )
    evaluation_seconds = time.perf_counter() - eval_t0

    payload = {
        "replicate": int(replicate),
        "setting_id": str(setting_id),
        "method": "GHS_plus_fair" if bool(sample_global_scale) else "GHS_plus_fixed_tau",
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
        "p0": int(p0),
        "tau0_used": float(tau0_use),
        "sample_global_scale": bool(sample_global_scale),
        "diagnostics": _json_clean(result.diagnostics),
        "timing_breakdown": {
            "dataset_seconds": float(dataset_seconds),
            "fit_wrapper_seconds": float(fit_call_seconds),
            "fit_runtime_seconds": _json_scalar(result.runtime_seconds),
            "evaluation_seconds": float(evaluation_seconds),
            "total_seconds": float(time.perf_counter() - total_t0),
        },
    }
    outdir_path = ROOT / str(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    method_stem = "GHS_plus_fair" if bool(sample_global_scale) else "GHS_plus_fixed_tau"
    stem = f"{setting_id}__{method_stem}__r{int(replicate)}"
    out_path = outdir_path / f"{stem}.json"
    if bool(save_artifacts):
        artifacts = save_fit_result_artifacts(
            outdir_path / stem,
            result=result,
            run_context={
                "setting_id": str(setting_id),
                "method": "GHS_plus_NUTS",
                "replicate": int(replicate),
                "source_script": "run_highdim_fair_ghs_plus_case.py",
            },
            coefficient_truth=ds.beta,
            dataset_arrays={
                "X_train": ds.X_train,
                "y_train": ds.y_train,
                "X_test": ds.X_test,
                "y_test": ds.y_test,
                "beta": ds.beta,
            },
            dataset_metadata={"groups": [[int(i) for i in g] for g in ds.groups]},
            save_dataset_bundle=True,
        )
        payload["artifacts"] = _json_clean(artifacts)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"out_path": str(out_path), **payload}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one formal fair GHS+ high-dimensional case.")
    parser.add_argument("--config", default="Simulation_highdimension/config/highdimension.yaml")
    parser.add_argument("--setting-id", required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--outdir", default="tmp/highdim_bayes_fair_quality")
    parser.add_argument("--warmup", type=int, default=2500)
    parser.add_argument("--draws", type=int, default=2500)
    parser.add_argument("--tau0", type=float, default=1.0)
    parser.add_argument("--tau0-auto", action="store_true", help="Use simulation p0 to set RHS-style global scale.")
    parser.add_argument("--fixed-tau", action="store_true", help="Treat tau0 as a fixed global-scale hyperparameter.")
    parser.add_argument("--group-scale-prior", type=float, default=1.0)
    parser.add_argument("--local-scale-prior", type=float, default=1.0)
    parser.add_argument("--use-process-pool", action="store_true")
    parser.add_argument("--save-artifacts", action="store_true")
    args = parser.parse_args()
    payload = run_case(
        config=str(args.config),
        setting_id=str(args.setting_id),
        replicate=int(args.replicate),
        outdir=str(args.outdir),
        warmup=int(args.warmup),
        draws=int(args.draws),
        tau0=float(args.tau0),
        tau0_auto=bool(args.tau0_auto),
        sample_global_scale=not bool(args.fixed_tau),
        group_scale_prior=float(args.group_scale_prior),
        local_scale_prior=float(args.local_scale_prior),
        use_process_pool=bool(args.use_process_pool),
        save_artifacts=bool(args.save_artifacts),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
