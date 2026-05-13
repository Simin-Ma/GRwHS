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
from simulation_project.src.utils import SamplerConfig, save_fit_result_artifacts
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
    method: str,
    replicate: int,
    outdir: str,
    warmup: int,
    draws: int,
    adapt_delta: float,
    max_treedepth: int,
    seed_offset: int,
    save_artifacts: bool = False,
    label: str | None = None,
    grrhs_tau0: float | None = None,
    grrhs_alpha_kappa: float | None = None,
    grrhs_beta_kappa: float | None = None,
    grrhs_use_local_scale: bool | None = None,
    grrhs_shared_kappa: bool | None = None,
    grrhs_tau_target: str | None = None,
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

    method_name = str(method)
    fit_t0 = time.perf_counter()
    if method_name == "RHS":
        from simulation_project.src.experiments.methods.fit_rhs import fit_rhs

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
    elif method_name == "GR_RHS":
        from simulation_project.src.experiments.methods.fit_gr_rhs import fit_gr_rhs

        grrhs_kwargs = dict(cfg.methods.grrhs_kwargs)
        grrhs_kwargs.pop("collapsed_hard_min_warmup", None)
        grrhs_kwargs.pop("collapsed_hard_min_draws", None)
        if grrhs_tau0 is not None:
            grrhs_kwargs["tau0"] = float(grrhs_tau0)
        if grrhs_alpha_kappa is not None:
            grrhs_kwargs["alpha_kappa"] = float(grrhs_alpha_kappa)
        if grrhs_beta_kappa is not None:
            grrhs_kwargs["beta_kappa"] = float(grrhs_beta_kappa)
        if grrhs_use_local_scale is not None:
            grrhs_kwargs["use_local_scale"] = bool(grrhs_use_local_scale)
        if grrhs_shared_kappa is not None:
            grrhs_kwargs["shared_kappa"] = bool(grrhs_shared_kappa)
        if grrhs_tau_target is not None:
            grrhs_kwargs["tau_target"] = str(grrhs_tau_target)
        tau_target_use = str(grrhs_kwargs.get("tau_target", "coefficients")).strip().lower()
        grrhs_p0 = int(p0_groups) if tau_target_use == "groups" else int(p0)
        result = fit_gr_rhs(
            ds.X_train,
            ds.y_train,
            ds.groups,
            task=str(cfg.runner.task),
            seed=int(cfg.runner.seed) + 1 + 10000 * int(seed_offset),
            p0=int(grrhs_p0),
            sampler=sampler,
            method_name="GR_RHS",
            retry_resume_payload=None,
            **{
                **grrhs_kwargs,
                "retry_attempt": 0,
                "collapsed_hard_min_warmup": int(warmup),
                "collapsed_hard_min_draws": int(draws),
            },
        )
    else:
        raise ValueError(f"Unsupported probe method: {method_name}")
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
        "method": method_name,
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
        "error": str(getattr(result, "error", "") or ""),
        "diagnostics": _json_clean(result.diagnostics),
        "probe_budget": {
            "warmup": int(warmup),
            "draws": int(draws),
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
    if method_name == "GR_RHS":
        payload["grrhs_parameters"] = _json_clean(grrhs_kwargs)
    outdir_path = ROOT / str(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    label_part = ""
    if label:
        safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(label))
        label_part = f"__{safe_label}"
    stem = f"{setting_id}__{method_name}{label_part}__r{int(replicate)}__w{int(warmup)}_d{int(draws)}_s{int(seed_offset)}"
    out_path = outdir_path / f"{stem}.json"
    if bool(save_artifacts):
        artifacts = save_fit_result_artifacts(
            outdir_path / stem,
            result=result,
            run_context={
                "setting_id": str(setting_id),
                "method": method_name,
                "replicate": int(replicate),
                "source_script": "run_highdim_budget_probe_case.py",
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
    parser = argparse.ArgumentParser(description="Probe smaller budgets for high-dimensional Bayesian methods.")
    parser.add_argument("--config", default="Simulation_highdimension/config/highdimension.yaml")
    parser.add_argument("--setting-id", required=True)
    parser.add_argument("--method", required=True, choices=["RHS", "GR_RHS"])
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--outdir", default="tmp/highdim_budget_probe")
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--draws", type=int, required=True)
    parser.add_argument("--adapt-delta", type=float, default=0.99)
    parser.add_argument("--max-treedepth", type=int, default=14)
    parser.add_argument("--seed-offset", type=int, default=31)
    parser.add_argument("--save-artifacts", action="store_true")
    parser.add_argument("--label", default=None)
    parser.add_argument("--grrhs-tau0", type=float, default=None)
    parser.add_argument("--grrhs-alpha-kappa", type=float, default=None)
    parser.add_argument("--grrhs-beta-kappa", type=float, default=None)
    parser.add_argument("--grrhs-use-local-scale", dest="grrhs_use_local_scale", action="store_true")
    parser.add_argument("--grrhs-no-local-scale", dest="grrhs_use_local_scale", action="store_false")
    parser.set_defaults(grrhs_use_local_scale=None)
    parser.add_argument("--grrhs-shared-kappa", dest="grrhs_shared_kappa", action="store_true")
    parser.add_argument("--grrhs-no-shared-kappa", dest="grrhs_shared_kappa", action="store_false")
    parser.set_defaults(grrhs_shared_kappa=None)
    parser.add_argument("--grrhs-tau-target", choices=["groups", "coefficients"], default=None)
    args = parser.parse_args()
    payload = run_case(
        config=str(args.config),
        setting_id=str(args.setting_id),
        method=str(args.method),
        replicate=int(args.replicate),
        outdir=str(args.outdir),
        warmup=int(args.warmup),
        draws=int(args.draws),
        adapt_delta=float(args.adapt_delta),
        max_treedepth=int(args.max_treedepth),
        seed_offset=int(args.seed_offset),
        save_artifacts=bool(args.save_artifacts),
        label=args.label,
        grrhs_tau0=args.grrhs_tau0,
        grrhs_alpha_kappa=args.grrhs_alpha_kappa,
        grrhs_beta_kappa=args.grrhs_beta_kappa,
        grrhs_use_local_scale=args.grrhs_use_local_scale,
        grrhs_shared_kappa=args.grrhs_shared_kappa,
        grrhs_tau_target=args.grrhs_tau_target,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
