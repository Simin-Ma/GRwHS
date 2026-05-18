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
from simulation_project.src.experiments.fitting import _fit_all_methods, _fit_with_convergence_retry
from simulation_project.src.experiments.runtime import _sampler_for_bayesian_default, highdim_sampler_budget
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


def run_case(
    *,
    config: str,
    setting_id: str,
    method: str,
    replicate: int = 1,
    outdir: str = "tmp/highdim_single_case_runs",
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

    def _fit_direct_single_method():
        method_name = str(method)
        if method_name == "GR_RHS":
            from simulation_project.src.experiments.methods.fit_gr_rhs import fit_gr_rhs

            grrhs_kwargs = dict(cfg.methods.grrhs_kwargs)
            grrhs_kwargs.setdefault("collapsed_hard_min_warmup", 150)
            grrhs_kwargs.setdefault("collapsed_hard_min_draws", 320)
            tau_target_use = str(grrhs_kwargs.get("tau_target", "coefficients")).strip().lower()
            grrhs_p0 = int(p0_groups) if tau_target_use == "groups" else int(p0)
            sampler_base = _sampler_for_bayesian_default(
                sampler,
                min_chains=int(cfg.convergence_gate.bayes_min_chains),
            )

            def _runner(sampler_try, attempt, resume_payload=None):
                return fit_gr_rhs(
                    ds.X_train,
                    ds.y_train,
                    ds.groups,
                    task=str(cfg.runner.task),
                    seed=int(cfg.runner.seed) + 1,
                    p0=int(grrhs_p0),
                    sampler=sampler_try,
                    method_name="GR_RHS",
                    retry_resume_payload=resume_payload,
                    **{**grrhs_kwargs, "retry_attempt": int(attempt)},
                )

            return _fit_with_convergence_retry(
                _runner,
                method="GR_RHS",
                sampler=sampler_base,
                bayes_min_chains=int(cfg.convergence_gate.bayes_min_chains),
                max_convergence_retries=int(cfg.convergence_gate.max_convergence_retries),
                enforce_bayes_convergence=bool(cfg.convergence_gate.enforce_bayes_convergence),
                continue_on_retry=True,
            )
        if method_name == "GIGG_MMLE":
            from simulation_project.src.experiments.methods.fit_gigg import fit_gigg_mmle

            gigg_kwargs = {"exact_highdim_fastpath": True, "progress_bar": False}
            sampler_base = _sampler_for_bayesian_default(
                highdim_sampler_budget(
                    sampler,
                    ds.X_train,
                    ds.groups,
                    role="gigg_mmle",
                ),
                min_chains=int(cfg.convergence_gate.bayes_min_chains),
            )

            def _runner(sampler_try, attempt, resume_payload=None):
                _ = resume_payload
                return fit_gigg_mmle(
                    ds.X_train,
                    ds.y_train,
                    ds.groups,
                    task=str(cfg.runner.task),
                    seed=int(cfg.runner.seed) + 1,
                    sampler=sampler_try,
                    p0=int(p0),
                    method_label="GIGG_MMLE",
                    **gigg_kwargs,
                )

            return _fit_with_convergence_retry(
                _runner,
                method="GIGG_MMLE",
                sampler=sampler_base,
                bayes_min_chains=int(cfg.convergence_gate.bayes_min_chains),
                max_convergence_retries=int(cfg.convergence_gate.max_convergence_retries),
                enforce_bayes_convergence=bool(cfg.convergence_gate.enforce_bayes_convergence),
                continue_on_retry=False,
            )
        if method_name == "RHS":
            from simulation_project.src.experiments.methods.fit_rhs import fit_rhs

            sampler_base = _sampler_for_bayesian_default(
                highdim_sampler_budget(sampler, ds.X_train, ds.groups, role="rhs_exact"),
                min_chains=int(cfg.convergence_gate.bayes_min_chains),
            )

            def _runner(sampler_try, attempt, resume_payload=None):
                _ = resume_payload
                return fit_rhs(
                    ds.X_train,
                    ds.y_train,
                    ds.groups,
                    task=str(cfg.runner.task),
                    seed=int(cfg.runner.seed) + 1 + 100 * int(attempt),
                    p0=int(p0),
                    sampler=sampler_try,
                    method_name="RHS",
                    progress_bar=False,
                )

            return _fit_with_convergence_retry(
                _runner,
                method="RHS",
                sampler=sampler_base,
                bayes_min_chains=int(cfg.convergence_gate.bayes_min_chains),
                max_convergence_retries=int(cfg.convergence_gate.max_convergence_retries),
                enforce_bayes_convergence=bool(cfg.convergence_gate.enforce_bayes_convergence),
                continue_on_retry=False,
            )
        return None

    fit_t0 = time.perf_counter()
    direct_result = _fit_direct_single_method()
    if direct_result is not None:
        result = direct_result
    else:
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
            methods=[str(method)],
            gigg_config=dict(cfg.methods.gigg_config),
            bayes_min_chains=int(cfg.convergence_gate.bayes_min_chains),
            enforce_bayes_convergence=bool(cfg.convergence_gate.enforce_bayes_convergence),
            max_convergence_retries=int(cfg.convergence_gate.max_convergence_retries),
            method_jobs=1,
            rhs_sampler_strategy="high_dim",
        )[str(method)]
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

    payload_t0 = time.perf_counter()
    payload = {
        "replicate": int(replicate),
        "setting_id": str(setting_id),
        "method": str(method),
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
        "attempts_used": int((((result.diagnostics or {}).get("convergence_retry", {}) or {}).get("attempts_used", 1))),
        "error": str(getattr(result, "error", "") or ""),
        "diagnostics": _json_clean(result.diagnostics),
        "timing_breakdown": {
            "dataset_seconds": float(dataset_seconds),
            "fit_wrapper_seconds": float(fit_call_seconds),
            "fit_runtime_seconds": _json_scalar(result.runtime_seconds),
            "evaluation_seconds": float(evaluation_seconds),
        },
    }
    payload_build_seconds = time.perf_counter() - payload_t0

    outdir_path = ROOT / str(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    stem = f"{setting_id}__{method}__r{int(replicate)}"
    out_path = outdir_path / f"{stem}.json"
    write_t0 = time.perf_counter()
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_seconds = time.perf_counter() - write_t0
    total_seconds = time.perf_counter() - total_t0
    payload["wall_seconds"] = float(total_seconds)
    payload["timing_breakdown"]["payload_build_seconds"] = float(payload_build_seconds)
    payload["timing_breakdown"]["write_seconds"] = float(write_seconds)
    payload["timing_breakdown"]["total_seconds"] = float(total_seconds)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"out_path": str(out_path), **payload}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one high-dimensional setting x method benchmark case.")
    parser.add_argument("--config", default="Simulation_highdimension/config/highdimension.yaml")
    parser.add_argument("--setting-id", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--outdir", default="tmp/highdim_single_case_runs")
    args = parser.parse_args()

    payload = run_case(
        config=str(args.config),
        setting_id=str(args.setting_id),
        method=str(args.method),
        replicate=int(args.replicate),
        outdir=str(args.outdir),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
