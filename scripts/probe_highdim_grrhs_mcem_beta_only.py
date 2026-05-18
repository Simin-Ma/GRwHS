from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation_project.src.experiments.methods.fit_gr_rhs_adaptive import (
    calibrate_grrhs_beta_mcem,
    calibrate_grrhs_beta_ridge_screening_multiplicity,
)
from simulation_project.src.utils import SamplerConfig
from simulation_second.src.config import load_benchmark_config
from simulation_second.src.dataset import generate_grouped_dataset


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
    if isinstance(value, (str, bool, int)):
        return value
    try:
        val = float(value)
    except Exception:
        return str(value)
    return val if val == val and abs(val) < float("inf") else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe GR-RHS MCEM beta calibration only, without final posterior sampling.")
    parser.add_argument("--config", default="Simulation_highdimension/config/highdimension.yaml")
    parser.add_argument("--setting-id", required=True)
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--outdir", default="tmp/highdim_grrhs_mcem_beta_only")
    parser.add_argument("--calibration-chains", type=int, default=1)
    parser.add_argument("--calibration-warmup", type=int, default=80)
    parser.add_argument("--calibration-draws", type=int, default=120)
    parser.add_argument("--calibration-adapt-delta", type=float, default=0.8)
    parser.add_argument("--calibration-max-treedepth", type=int, default=7)
    parser.add_argument("--screening-permutations", type=int, default=120)
    parser.add_argument("--screening-null-quantile", type=float, default=0.95)
    parser.add_argument("--strategy", default="mcem_beta", choices=["mcem_beta", "multiplicity"])
    parser.add_argument("--multiplicity-correction", default="fwer")
    parser.add_argument("--multiplicity-level", type=float, default=0.10)
    parser.add_argument("--multiplicity-min-active-groups", type=float, default=1.0)
    parser.add_argument("--mcem-rounds", type=int, default=1)
    parser.add_argument("--mcem-step-size", type=float, default=0.7)
    parser.add_argument("--seed-offset", type=int, default=13001)
    args = parser.parse_args()

    t0 = time.perf_counter()
    cfg = load_benchmark_config(str(args.config))
    setting = cfg.setting_map()[str(args.setting_id)]
    ds = generate_grouped_dataset(
        setting,
        replicate_id=int(args.replicate),
        master_seed=int(cfg.runner.seed),
        family_specs=cfg.families,
    )
    p0_groups = int(sum(any(abs(ds.beta[idx]) > 1e-12 for idx in g) for g in ds.groups))
    sampler = SamplerConfig(
        chains=int(args.calibration_chains),
        warmup=int(args.calibration_warmup),
        post_warmup_draws=int(args.calibration_draws),
        adapt_delta=float(args.calibration_adapt_delta),
        max_treedepth=int(args.calibration_max_treedepth),
        max_divergence_ratio=float(cfg.convergence_gate.max_divergence_ratio),
        rhat_threshold=float(cfg.convergence_gate.rhat_threshold),
        ess_threshold=float(cfg.convergence_gate.ess_threshold),
    )
    grrhs_kwargs = dict(cfg.methods.grrhs_kwargs)
    grrhs_kwargs.pop("collapsed_hard_min_warmup", None)
    grrhs_kwargs.pop("collapsed_hard_min_draws", None)
    grrhs_kwargs.update({
        "tau_target": "groups",
        "sampler_backend": "collapsed_profile",
        "use_local_scale": False,
        "progress_bar": False,
    })
    if str(args.strategy) == "multiplicity":
        calib = calibrate_grrhs_beta_ridge_screening_multiplicity(
            ds.X_train,
            ds.y_train,
            ds.groups,
            alpha_kappa=0.5,
            null_level=float(args.multiplicity_level),
            n_permutations=int(args.screening_permutations),
            ridge_scale="sqrt_np",
            correction=str(args.multiplicity_correction),
            min_active_groups=float(args.multiplicity_min_active_groups),
            min_beta_kappa=1.0,
            max_beta_kappa=16.0,
            seed=int(cfg.runner.seed) + 15 + 10000 * int(args.seed_offset),
        )
    else:
        calib = calibrate_grrhs_beta_mcem(
            ds.X_train,
            ds.y_train,
            ds.groups,
            task=str(cfg.runner.task),
            seed=int(cfg.runner.seed) + 15 + 10000 * int(args.seed_offset),
            p0=int(p0_groups),
            sampler=sampler,
            grrhs_kwargs=grrhs_kwargs,
            alpha_kappa=0.5,
            beta_kappa=1.0,
            init_strategy="ridge_screening_moment",
            rounds=int(args.mcem_rounds),
            calibration_chains=int(args.calibration_chains),
            calibration_warmup=int(args.calibration_warmup),
            calibration_draws=int(args.calibration_draws),
            calibration_adapt_delta=float(args.calibration_adapt_delta),
            calibration_max_treedepth=int(args.calibration_max_treedepth),
            step_size=float(args.mcem_step_size),
            screening_null_quantile=float(args.screening_null_quantile),
            screening_permutations=int(args.screening_permutations),
            ridge_screening_scale="sqrt_np",
            min_beta_kappa=1.0,
            max_beta_kappa=16.0,
        )
    payload = {
        "setting_id": str(args.setting_id),
        "replicate": int(args.replicate),
        "strategy": str(calib.strategy),
        "alpha_kappa": float(calib.alpha_kappa),
        "beta_kappa": float(calib.beta_kappa),
        "details": _json_clean(calib.details),
        "wall_seconds": float(time.perf_counter() - t0),
    }
    outdir = ROOT / str(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{args.setting_id}__mcem_beta_only__r{int(args.replicate)}__s{int(args.seed_offset)}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out_path": str(out_path), **payload}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
