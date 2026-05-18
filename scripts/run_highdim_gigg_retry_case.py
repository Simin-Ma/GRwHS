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
from simulation_project.src.experiments.methods.fit_gigg import fit_gigg_mmle
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
    replicate: int,
    outdir: str,
    rounds: int,
    draws_per_round: int,
    chain_workers: int,
    seed_offset: int,
    save_artifacts: bool = False,
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
        chains=max(4, int(cfg.convergence_gate.chains)),
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
    gigg_kwargs = dict(cfg.methods.gigg_config)
    for key in ("allow_budget_retry", "extra_retry", "retry_cap"):
        gigg_kwargs.pop(key, None)
    gigg_kwargs.update(
        {
            "exact_highdim_fastpath": True,
            "highdim_continuation_rounds": int(rounds),
            "highdim_continuation_warmup": 2,
            "highdim_continuation_draws": int(draws_per_round),
            "highdim_stage_a_burnin": max(8, int(gigg_kwargs.get("highdim_stage_a_burnin", 8) or 8)),
            "highdim_stage_a_draws": max(8, int(gigg_kwargs.get("highdim_stage_a_draws", 8) or 8)),
            "highdim_stage_a_reference_mmle": True,
            "highdim_select_best_round": True,
            "highdim_diagnostic_interval": 10,
            "highdim_early_stop": True,
            "highdim_early_stop_min_rounds": 120,
            "highdim_early_stop_patience": 2,
            "highdim_store_history": False,
            "highdim_chain_workers": int(max(1, chain_workers)),
            "progress_bar": False,
        }
    )
    fit_t0 = time.perf_counter()
    result = fit_gigg_mmle(
        ds.X_train,
        ds.y_train,
        ds.groups,
        task=str(cfg.runner.task),
        seed=int(cfg.runner.seed) + 4 + 10000 * int(seed_offset),
        sampler=sampler,
        p0=int(p0),
        method_label="GIGG_MMLE",
        **gigg_kwargs,
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
        "method": "GIGG_MMLE",
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
            "rounds": int(rounds),
            "draws_per_round": int(draws_per_round),
            "chain_workers": int(max(1, chain_workers)),
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
    stem = f"{setting_id}__GIGG_MMLE__r{int(replicate)}__rounds{int(rounds)}_dpr{int(draws_per_round)}_s{int(seed_offset)}"
    out_path = outdir_path / f"{stem}.json"
    if bool(save_artifacts):
        artifacts = save_fit_result_artifacts(
            outdir_path / stem,
            result=result,
            run_context={
                "setting_id": str(setting_id),
                "method": "GIGG_MMLE",
                "replicate": int(replicate),
                "source_script": "run_highdim_gigg_retry_case.py",
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
    parser = argparse.ArgumentParser(description="Targeted high-dimensional GIGG retry case.")
    parser.add_argument("--config", default="Simulation_highdimension/config/highdimension.yaml")
    parser.add_argument("--setting-id", required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--outdir", default="tmp/highdim_bayes_rerun_20260512_full/gigg_retries")
    parser.add_argument("--rounds", type=int, default=420)
    parser.add_argument("--draws-per-round", type=int, default=5)
    parser.add_argument("--chain-workers", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=21)
    parser.add_argument("--save-artifacts", action="store_true")
    args = parser.parse_args()
    payload = run_case(
        config=str(args.config),
        setting_id=str(args.setting_id),
        replicate=int(args.replicate),
        outdir=str(args.outdir),
        rounds=int(args.rounds),
        draws_per_round=int(args.draws_per_round),
        chain_workers=int(args.chain_workers),
        seed_offset=int(args.seed_offset),
        save_artifacts=bool(args.save_artifacts),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
