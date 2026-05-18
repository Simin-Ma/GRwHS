from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_highdim_grrhs_registered_eb_case import run_case
from simulation_second.src.config import load_benchmark_config


def _setting_ids(config_path: str) -> list[str]:
    cfg = load_benchmark_config(config_path)
    return [str(setting.setting_id) for setting in cfg.settings]


def _result_ok(payload: dict[str, object]) -> bool:
    return bool(payload.get("converged"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen ridge-screen adaptive EB GR-RHS across high-dimensional "
            "settings/repeats. The EB rule is fixed; retries only increase the "
            "sampling budget."
        )
    )
    parser.add_argument("--config", default="Simulation_highdimension/config/highdimension.yaml")
    parser.add_argument("--outdir", default="tmp/highdim_grrhs_eb_frozen_reps5")
    parser.add_argument("--settings", nargs="*", default=None)
    parser.add_argument("--repeats", nargs="*", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--calibration-draws", type=int, default=300)
    parser.add_argument("--screening-null-quantile", type=float, default=0.95)
    parser.add_argument("--min-beta-kappa", type=float, default=1.0)
    parser.add_argument("--seed-offset-base", type=int, default=9911)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    settings = args.settings if args.settings else _setting_ids(str(args.config))
    schedules = [
        {
            "label": "ridge_screen_q95",
            "warmup": 700,
            "draws": 1600,
            "adapt_delta": 0.97,
            "max_treedepth": 14,
        },
        {
            "label": "ridge_screen_q95_retry",
            "warmup": 1200,
            "draws": 3000,
            "adapt_delta": 0.99,
            "max_treedepth": 15,
        },
    ]

    records: list[dict[str, object]] = []
    for replicate in args.repeats:
        for setting_id in settings:
            case_records: list[dict[str, object]] = []
            for attempt, budget in enumerate(schedules):
                seed_offset = int(args.seed_offset_base) + 100 * int(replicate) + attempt
                if args.dry_run:
                    payload = {
                        "setting_id": setting_id,
                        "replicate": replicate,
                        "attempt": attempt,
                        "seed_offset": seed_offset,
                        **budget,
                    }
                    print(json.dumps(payload, ensure_ascii=False))
                    case_records.append(payload)
                    continue
                payload = run_case(
                    config=str(args.config),
                    setting_id=str(setting_id),
                    replicate=int(replicate),
                    outdir=str(args.outdir),
                    warmup=int(budget["warmup"]),
                    draws=int(budget["draws"]),
                    adapt_delta=float(budget["adapt_delta"]),
                    max_treedepth=int(budget["max_treedepth"]),
                    seed_offset=seed_offset,
                    calibration_warmup=300,
                    calibration_draws=int(args.calibration_draws),
                    n_initial_points=5,
                    n_refine_points=0,
                    validation_fraction=float(args.screening_null_quantile),
                    log_beta_min=0.0,
                    log_beta_max=0.0,
                    min_beta_kappa=float(args.min_beta_kappa),
                    label=str(budget["label"]),
                )
                payload["attempt"] = attempt
                case_records.append(payload)
                print(json.dumps(payload, ensure_ascii=False))
                if _result_ok(payload):
                    break
            records.extend(case_records)

    outdir = ROOT / str(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_log = {
        "config": str(args.config),
        "settings": settings,
        "repeats": [int(rep) for rep in args.repeats],
        "frozen_eb": {
            "adaptive_strategy": "ridge_screening_moment",
            "alpha_kappa": 0.5,
            "screening_null_quantile": float(args.screening_null_quantile),
            "screening_permutations": int(args.calibration_draws),
            "ridge_screening_scale": "sqrt_np",
            "min_beta_kappa": float(args.min_beta_kappa),
            "max_beta_kappa": 16.0,
        },
        "schedules": schedules,
        "records": records,
    }
    (outdir / "run_log.json").write_text(json.dumps(run_log, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
