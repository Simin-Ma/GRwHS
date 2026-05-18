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
from simulation_project.src.experiments.methods.fit_gr_rhs_adaptive import fit_gr_rhs_adaptive_beta
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


def run_case(
    *,
    config: str,
    setting_id: str,
    replicate: int,
    outdir: str,
    warmup: int,
    draws: int,
    adapt_delta: float,
    max_treedepth: int,
    seed_offset: int,
    calibration_warmup: int,
    calibration_draws: int,
    n_initial_points: int,
    n_refine_points: int,
    validation_fraction: float,
    log_beta_min: float,
    log_beta_max: float,
    min_beta_kappa: float | None,
    adaptive_strategy: str,
    mcem_rounds: int,
    mcem_step_size: float,
    mcem_init_strategy: str,
    mcem_calibration_chains: int | None,
    mcem_calibration_adapt_delta: float | None,
    mcem_calibration_max_treedepth: int | None,
    label: str | None,
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
    p0_groups = int(sum(any(abs(ds.beta[idx]) > 1e-12 for idx in g) for g in ds.groups))
    sampler = SamplerConfig(
        chains=max(4, int(cfg.convergence_gate.chains)),
        warmup=int(warmup),
        post_warmup_draws=int(draws),
        adapt_delta=float(adapt_delta),
        max_treedepth=int(max_treedepth),
        strict_adapt_delta=max(0.995, float(adapt_delta)),
        strict_max_treedepth=max(int(max_treedepth), 15),
        max_divergence_ratio=float(cfg.convergence_gate.max_divergence_ratio),
        rhat_threshold=float(cfg.convergence_gate.rhat_threshold),
        ess_threshold=float(cfg.convergence_gate.ess_threshold),
    )
    grrhs_kwargs = dict(cfg.methods.grrhs_kwargs)
    grrhs_kwargs.pop("collapsed_hard_min_warmup", None)
    grrhs_kwargs.pop("collapsed_hard_min_draws", None)
    grrhs_kwargs.update(
        {
            "tau_target": "groups",
            "sampler_backend": "collapsed_profile",
            "use_local_scale": False,
            "progress_bar": False,
            "adaptive_strategy": str(adaptive_strategy),
            "alpha_kappa": 0.5,
            "screening_null_quantile": float(validation_fraction),
            "screening_permutations": int(calibration_draws),
            "ridge_screening_scale": "sqrt_np",
            "calibration_warmup": int(calibration_warmup),
            "calibration_draws": int(calibration_draws),
            "mcem_rounds": int(mcem_rounds),
            "mcem_step_size": float(mcem_step_size),
            "mcem_init_strategy": str(mcem_init_strategy),
            "mcem_calibration_chains": mcem_calibration_chains,
            "mcem_calibration_adapt_delta": mcem_calibration_adapt_delta,
            "mcem_calibration_max_treedepth": mcem_calibration_max_treedepth,
            "min_beta_kappa": min_beta_kappa,
            "max_beta_kappa": 16.0,
        }
    )

    result = fit_gr_rhs_adaptive_beta(
        ds.X_train,
        ds.y_train,
        ds.groups,
        task=str(cfg.runner.task),
        seed=int(cfg.runner.seed) + 15 + 10000 * int(seed_offset),
        p0=int(p0_groups),
        sampler=sampler,
        method_name="GR_RHS_EB",
        **{
            **grrhs_kwargs,
            "retry_attempt": 0,
            "collapsed_hard_min_warmup": int(warmup),
            "collapsed_hard_min_draws": int(draws),
        },
    )
    metrics = _evaluate_row(
        result,
        ds.beta,
        X_train=ds.X_train,
        y_train=ds.y_train,
        X_test=ds.X_test,
        y_test=ds.y_test,
    )
    adaptive = {}
    if isinstance(result.diagnostics, dict):
        adaptive = result.diagnostics.get("grrhs_adaptive_beta", {}) or {}
    payload = {
        "replicate": int(replicate),
        "setting_id": str(setting_id),
        "method": "GR_RHS_EB",
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
        "wall_seconds": float(time.perf_counter() - total_t0),
        "selected_alpha_kappa": _json_scalar(adaptive.get("alpha_kappa")),
        "selected_beta_kappa": _json_scalar(adaptive.get("beta_kappa")),
        "adaptive_calibration": _json_clean(adaptive),
        "grrhs_parameters": _json_clean(grrhs_kwargs),
        "diagnostics": _json_clean(result.diagnostics),
        "probe_budget": {
            "warmup": int(warmup),
            "draws": int(draws),
            "adapt_delta": float(adapt_delta),
            "max_treedepth": int(max_treedepth),
            "seed_offset": int(seed_offset),
        },
    }

    outdir_path = ROOT / str(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    label_part = ""
    if label:
        safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(label))
        label_part = f"__{safe_label}"
    stem = f"{setting_id}__GR_RHS_EB{label_part}__r{int(replicate)}__cw{int(calibration_warmup)}_cd{int(calibration_draws)}__fw{int(warmup)}_fd{int(draws)}_s{int(seed_offset)}"
    out_path = outdir_path / f"{stem}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"out_path": str(out_path), **payload}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run registered continuous-validation GR_RHS_EB on a high-dimensional setting.")
    parser.add_argument("--config", default="Simulation_highdimension/config/highdimension.yaml")
    parser.add_argument("--setting-id", required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--outdir", default="tmp/highdim_grrhs_registered_eb_rep1")
    parser.add_argument("--warmup", type=int, default=700)
    parser.add_argument("--draws", type=int, default=1600)
    parser.add_argument("--adapt-delta", type=float, default=0.97)
    parser.add_argument("--max-treedepth", type=int, default=14)
    parser.add_argument("--seed-offset", type=int, default=9901)
    parser.add_argument("--calibration-warmup", type=int, default=300)
    parser.add_argument("--calibration-draws", type=int, default=500)
    parser.add_argument("--n-initial-points", type=int, default=5)
    parser.add_argument("--n-refine-points", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.95, help="For ridge_screening_moment this is the empirical-null quantile.")
    parser.add_argument("--log-beta-min", type=float, default=math.log(0.5))
    parser.add_argument("--log-beta-max", type=float, default=math.log(16.0))
    parser.add_argument("--min-beta-kappa", type=float, default=1.0)
    parser.add_argument("--adaptive-strategy", default="ridge_screening_moment")
    parser.add_argument("--mcem-rounds", type=int, default=1)
    parser.add_argument("--mcem-step-size", type=float, default=1.0)
    parser.add_argument("--mcem-init-strategy", default="ridge_screening_moment")
    parser.add_argument("--mcem-calibration-chains", type=int, default=None)
    parser.add_argument("--mcem-calibration-adapt-delta", type=float, default=None)
    parser.add_argument("--mcem-calibration-max-treedepth", type=int, default=None)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()
    payload = run_case(
        config=str(args.config),
        setting_id=str(args.setting_id),
        replicate=int(args.replicate),
        outdir=str(args.outdir),
        warmup=int(args.warmup),
        draws=int(args.draws),
        adapt_delta=float(args.adapt_delta),
        max_treedepth=int(args.max_treedepth),
        seed_offset=int(args.seed_offset),
        calibration_warmup=int(args.calibration_warmup),
        calibration_draws=int(args.calibration_draws),
        n_initial_points=int(args.n_initial_points),
        n_refine_points=int(args.n_refine_points),
        validation_fraction=float(args.validation_fraction),
        log_beta_min=float(args.log_beta_min),
        log_beta_max=float(args.log_beta_max),
        min_beta_kappa=args.min_beta_kappa,
        adaptive_strategy=str(args.adaptive_strategy),
        mcem_rounds=int(args.mcem_rounds),
        mcem_step_size=float(args.mcem_step_size),
        mcem_init_strategy=str(args.mcem_init_strategy),
        mcem_calibration_chains=args.mcem_calibration_chains,
        mcem_calibration_adapt_delta=args.mcem_calibration_adapt_delta,
        mcem_calibration_max_treedepth=args.mcem_calibration_max_treedepth,
        label=args.label,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
