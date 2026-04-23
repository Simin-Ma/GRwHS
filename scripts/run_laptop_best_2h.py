"""Laptop-friendly Exp1-Exp5 protocol runner (2-3 hour target).

This script codifies the "main-result-first" protocol:
- smoke mode: repeats=1 for each experiment to validate run-chain quickly
- main mode: fixed low-budget settings intended for a single laptop
- acceptance mode: evaluates plan-level acceptance criteria on saved outputs

Examples:
    # 1) smoke check
    python scripts/run_laptop_best_2h.py --mode smoke

    # 2) main run + analysis + acceptance checks
    python scripts/run_laptop_best_2h.py --mode main

    # 3) run everything in one command
    python scripts/run_laptop_best_2h.py --mode both

    # 4) acceptance-only on existing results
    python scripts/run_laptop_best_2h.py --mode acceptance --save-dir outputs/simulation_project/laptop_best_2h
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation_project.src.experiments.analysis.report import run_analysis  # noqa: E402
from simulation_project.src.run_experiment import (  # noqa: E402
    run_exp1_kappa_profile_regimes,
    run_exp2_group_separation,
    run_exp3a_main_benchmark,
    run_exp3b_boundary_stress,
    run_exp4_variant_ablation,
    run_exp5_prior_sensitivity,
)

DEFAULT_SAVE_DIR = Path("outputs/simulation_project/laptop_best_2h")


@dataclass(frozen=True)
class ProtocolConfig:
    save_dir: Path
    seed: int
    n_jobs: int
    skip_analysis: bool
    skip_acceptance: bool
    exp3b_repeats: int
    exp5_repeats: int
    distributed_noninferiority_ratio: float
    exp4_max_degrade_ratio: float
    runtime_target_minutes: float
    require_all_bayes_converged: bool
    nuts_max_convergence_retries: int
    exp4_max_convergence_retries: int


def _common_kwargs(cfg: ProtocolConfig, *, max_retries: int, sampler_backend: str) -> dict[str, Any]:
    return {
        "n_jobs": int(cfg.n_jobs),
        "seed": int(cfg.seed),
        "save_dir": str(cfg.save_dir),
        "enforce_bayes_convergence": True,
        "max_convergence_retries": int(max_retries),
        "until_bayes_converged": True,
        "sampler_backend": str(sampler_backend),
    }


def _group_configs_main() -> list[dict[str, Any]]:
    return [
        {"name": "G10x5", "group_sizes": [10, 10, 10, 10, 10], "active_groups": [0, 1]},
        {"name": "CL", "group_sizes": [30, 10, 5, 3, 2], "active_groups": [0, 1]},
    ]


def _env_points_e0() -> list[dict[str, Any]]:
    return [
        {
            "env_id": "E0",
            "setting_block": "anchor",
            "rho_within": 0.3,
            "rho_between": 0.1,
            "target_snr": 1.0,
            "signals": ["concentrated", "distributed", "boundary"],
        }
    ]


def _methods_exp3() -> list[str]:
    return ["GR_RHS", "RHS", "GIGG_MMLE", "GHS_plus", "OLS", "LASSO_CV"]


BAYES_METHODS = {"GR_RHS", "RHS", "GIGG_MMLE", "GHS_plus"}


def _run_smoke(cfg: ProtocolConfig) -> None:
    print("[smoke] starting protocol smoke run (repeats=1 for each experiment)...")
    smoke_save_dir = Path(str(cfg.save_dir) + "_smoke")
    smoke_cfg = ProtocolConfig(
        save_dir=smoke_save_dir,
        seed=cfg.seed,
        n_jobs=max(1, min(cfg.n_jobs, 2)),
        skip_analysis=cfg.skip_analysis,
        skip_acceptance=cfg.skip_acceptance,
        exp3b_repeats=1,
        exp5_repeats=1,
        distributed_noninferiority_ratio=cfg.distributed_noninferiority_ratio,
        exp4_max_degrade_ratio=cfg.exp4_max_degrade_ratio,
        runtime_target_minutes=cfg.runtime_target_minutes,
        require_all_bayes_converged=cfg.require_all_bayes_converged,
        nuts_max_convergence_retries=cfg.nuts_max_convergence_retries,
        exp4_max_convergence_retries=cfg.exp4_max_convergence_retries,
    )
    _run_main(smoke_cfg, smoke_mode=True)


def _save_runtime_manifest(save_dir: Path, *, total_minutes: float, smoke_mode: bool) -> None:
    path = save_dir / "results" / "run_runtime.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "save_dir": str(save_dir),
        "total_minutes": float(total_minutes),
        "smoke_mode": bool(smoke_mode),
        "generated_at_unix": time.time(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_main(cfg: ProtocolConfig, *, smoke_mode: bool = False) -> None:
    cfg.save_dir.mkdir(parents=True, exist_ok=True)

    repeats = {
        "exp1": 1 if smoke_mode else 100,
        "exp2": 1 if smoke_mode else 12,
        "exp3a": 1 if smoke_mode else 4,
        "exp4": 1 if smoke_mode else 6,
        "exp3b": 1 if smoke_mode else int(cfg.exp3b_repeats),
        "exp5": 1 if smoke_mode else int(cfg.exp5_repeats),
    }

    common_collapsed = _common_kwargs(cfg, max_retries=cfg.nuts_max_convergence_retries, sampler_backend="collapsed")

    print(f"[main] save_dir={cfg.save_dir}")
    print(f"[main] n_jobs={cfg.n_jobs}, seed={cfg.seed}, smoke_mode={smoke_mode}")
    print(f"[main] repeats={repeats}")
    t0 = time.time()

    # Exp1
    run_exp1_kappa_profile_regimes(
        n_jobs=cfg.n_jobs,
        seed=cfg.seed,
        save_dir=str(cfg.save_dir),
        repeats=repeats["exp1"],
        pg_null_list=[10, 20, 50, 100, 200, 500],
        pg_phase_list=[30, 120, 480],
        tau_phase_list=[0.5, 1.0],
        xi_multiplier_list=[0.5, 0.8, 1.0, 1.2, 1.5],
        include_full_null_curve=False,
    )

    # Exp2
    run_exp2_group_separation(
        repeats=repeats["exp2"],
        methods=["GR_RHS", "RHS"],
        rho_ref=0.8,
        n_test=50,
        **common_collapsed,
    )

    # Exp3a
    run_exp3a_main_benchmark(
        repeats=repeats["exp3a"],
        group_configs=_group_configs_main(),
        env_points=_env_points_e0(),
        methods=_methods_exp3(),
        **common_collapsed,
    )

    # Exp4
    run_exp4_variant_ablation(
        repeats=repeats["exp4"],
        p0_list=[5, 30],
        include_oracle=False,
        n_jobs=cfg.n_jobs,
        seed=cfg.seed,
        save_dir=str(cfg.save_dir),
        enforce_bayes_convergence=True,
        max_convergence_retries=int(cfg.exp4_max_convergence_retries),
        until_bayes_converged=True,
        sampler_backend="collapsed",
    )

    # Exp3b
    run_exp3b_boundary_stress(
        repeats=repeats["exp3b"],
        group_configs=[{"name": "G10x5", "group_sizes": [10, 10, 10, 10, 10], "active_groups": [0, 1]}],
        env_points=_env_points_e0(),
        boundary_xi_ratio_list=[0.8, 1.0, 1.2],
        methods=_methods_exp3(),
        **common_collapsed,
    )

    # Exp5
    run_exp5_prior_sensitivity(
        repeats=repeats["exp5"],
        **common_collapsed,
    )

    total_min = (time.time() - t0) / 60.0
    _save_runtime_manifest(cfg.save_dir, total_minutes=total_min, smoke_mode=smoke_mode)
    print(f"[main] total_minutes={total_min:.1f}")

    if not cfg.skip_analysis:
        run_analysis(save_dir=str(cfg.save_dir))

    if not cfg.skip_acceptance and not smoke_mode:
        _run_acceptance(cfg)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def _check_exp1(save_dir: Path) -> tuple[bool, str]:
    path = save_dir / "results" / "exp1_kappa_profile_regimes" / "null_slope_check.json"
    if not path.exists():
        return False, f"missing {path}"
    payload = _read_json(path)
    ci = payload.get("slope_ci", [float("nan"), float("nan")])
    lo, hi = _to_float(ci[0]), _to_float(ci[1])
    ok = bool(lo < -0.5 < hi)
    return ok, f"Exp1 ci=[{lo:.3f}, {hi:.3f}] contains -0.5={ok}"


def _check_exp2(save_dir: Path) -> tuple[bool, str]:
    path = save_dir / "results" / "exp2_group_separation" / "raw_results.csv"
    if not path.exists():
        return False, f"missing {path}"
    rows = _read_csv(path)
    conv = [r for r in rows if _to_bool(r.get("converged", False))]
    gr = [_to_float(r.get("null_group_mse")) for r in conv if r.get("method") == "GR_RHS"]
    rhs = [_to_float(r.get("null_group_mse")) for r in conv if r.get("method") == "RHS"]
    gr = [v for v in gr if math.isfinite(v)]
    rhs = [v for v in rhs if math.isfinite(v)]
    if not gr or not rhs:
        return False, "Exp2 missing converged GR_RHS/RHS null_group_mse values"
    gr_m, rhs_m = float(sum(gr) / len(gr)), float(sum(rhs) / len(rhs))
    ok = gr_m < rhs_m
    return ok, f"Exp2 null_group_mse: GR_RHS={gr_m:.5f}, RHS={rhs_m:.5f}, pass={ok}"


def _check_exp3a(save_dir: Path, *, noninferiority_ratio: float) -> tuple[bool, str]:
    path = save_dir / "results" / "exp3a_main_benchmark" / "summary.csv"
    if not path.exists():
        return False, f"missing {path}"
    rows = _read_csv(path)

    def mean_metric(signal: str, method: str, metric: str) -> float:
        vals = [
            _to_float(r.get(metric))
            for r in rows
            if r.get("signal") == signal and r.get("method") == method
        ]
        vals = [v for v in vals if math.isfinite(v)]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    c_gr = mean_metric("concentrated", "GR_RHS", "mse_null")
    c_rhs = mean_metric("concentrated", "RHS", "mse_null")
    d_gr = mean_metric("distributed", "GR_RHS", "mse_overall")
    d_rhs = mean_metric("distributed", "RHS", "mse_overall")

    if not all(math.isfinite(v) for v in [c_gr, c_rhs, d_gr, d_rhs]):
        return False, "Exp3a missing concentrated/distributed GR_RHS or RHS metrics"

    concentrated_ok = c_gr <= c_rhs
    distributed_ok = d_gr <= d_rhs * float(noninferiority_ratio)
    ok = concentrated_ok and distributed_ok
    return ok, (
        f"Exp3a concentrated mse_null GR_RHS={c_gr:.5f}, RHS={c_rhs:.5f}, pass={concentrated_ok}; "
        f"distributed mse_overall GR_RHS={d_gr:.5f}, RHS={d_rhs:.5f}, "
        f"threshold={noninferiority_ratio:.3f}, pass={distributed_ok}"
    )


def _check_exp4(save_dir: Path, *, max_degrade_ratio: float) -> tuple[bool, str]:
    path = save_dir / "results" / "exp4_variant_ablation" / "summary.csv"
    if not path.exists():
        return False, f"missing {path}"
    rows = _read_csv(path)
    cal_rows = [r for r in rows if r.get("variant") == "calibrated"]
    vals = [_to_float(r.get("mse_rel_rhs_oracle")) for r in cal_rows]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return False, "Exp4 has no calibrated mse_rel_rhs_oracle values"
    has_leq_one = any(v <= 1.0 for v in vals)
    all_not_too_bad = all(v <= max_degrade_ratio for v in vals)
    ok = has_leq_one and all_not_too_bad
    return ok, (
        f"Exp4 calibrated mse_rel_rhs_oracle={['%.4f' % v for v in vals]}, "
        f"any<=1.0={has_leq_one}, all<={max_degrade_ratio:.3f}={all_not_too_bad}"
    )


def _check_exp5(save_dir: Path) -> tuple[bool, str]:
    path = save_dir / "results" / "exp5_prior_sensitivity" / "summary.csv"
    if not path.exists():
        return False, f"missing {path}"
    rows = _read_csv(path)

    by_setting: dict[int, list[dict[str, str]]] = {}
    for r in rows:
        sid = int(_to_float(r.get("setting_id", 0)))
        by_setting.setdefault(sid, []).append(r)

    settings = sorted(by_setting.keys())
    if not settings:
        return False, "Exp5 has no settings in summary.csv"

    mse_worst_count = 0
    auroc_worst_count = 0
    detail_lines: list[str] = []

    for sid in settings:
        sub = by_setting[sid]
        default = [r for r in sub if abs(_to_float(r.get("alpha_kappa")) - 0.5) < 1e-9 and abs(_to_float(r.get("beta_kappa")) - 1.0) < 1e-9]
        if not default:
            return False, f"Exp5 setting={sid} missing default prior (0.5,1.0)"
        d = default[0]
        d_mse = _to_float(d.get("mse_signal"))
        d_auc = _to_float(d.get("group_auroc"))

        others = [
            r for r in sub
            if not (abs(_to_float(r.get("alpha_kappa")) - 0.5) < 1e-9 and abs(_to_float(r.get("beta_kappa")) - 1.0) < 1e-9)
        ]
        o_mse = [_to_float(r.get("mse_signal")) for r in others if math.isfinite(_to_float(r.get("mse_signal")))]
        o_auc = [_to_float(r.get("group_auroc")) for r in others if math.isfinite(_to_float(r.get("group_auroc")))]
        if not math.isfinite(d_mse) or not math.isfinite(d_auc) or not o_mse or not o_auc:
            return False, f"Exp5 setting={sid} missing finite mse_signal/group_auroc values"

        worst_mse = d_mse > max(o_mse)
        worst_auc = d_auc < min(o_auc)
        mse_worst_count += int(worst_mse)
        auroc_worst_count += int(worst_auc)
        detail_lines.append(
            f"setting={sid}: default mse_signal={d_mse:.5f}, other_range=[{min(o_mse):.5f},{max(o_mse):.5f}], "
            f"default auroc={d_auc:.3f}, other_range=[{min(o_auc):.3f},{max(o_auc):.3f}]"
        )

    threshold = len(settings) / 2.0
    ok = (mse_worst_count <= threshold) and (auroc_worst_count <= threshold)
    detail = "; ".join(detail_lines)
    return ok, (
        f"Exp5 default prior systematic degrade check: "
        f"mse_worst_count={mse_worst_count}/{len(settings)}, "
        f"auroc_worst_count={auroc_worst_count}/{len(settings)}, pass={ok}. {detail}"
    )


def _check_runtime_target(save_dir: Path, *, target_minutes: float) -> tuple[bool, str]:
    path = save_dir / "results" / "run_runtime.json"
    if not path.exists():
        return False, f"missing {path}"
    payload = _read_json(path)
    total = _to_float(payload.get("total_minutes"))
    if not math.isfinite(total):
        return False, "runtime total_minutes is non-finite"
    ok = total <= float(target_minutes)
    return ok, f"Runtime total_minutes={total:.1f}, target<={target_minutes:.1f}, pass={ok}"


def _check_bayes_convergence_strict(save_dir: Path) -> tuple[bool, str]:
    specs = [
        {
            "name": "Exp2",
            "path": save_dir / "results" / "exp2_group_separation" / "raw_results.csv",
            "method_col": "method",
            "include_methods": {"GR_RHS", "RHS"},
        },
        {
            "name": "Exp3a",
            "path": save_dir / "results" / "exp3a_main_benchmark" / "raw_results.csv",
            "method_col": "method",
            "include_methods": set(BAYES_METHODS),
        },
        {
            "name": "Exp3b",
            "path": save_dir / "results" / "exp3b_boundary_stress" / "raw_results.csv",
            "method_col": "method",
            "include_methods": set(BAYES_METHODS),
        },
        {
            "name": "Exp4",
            "path": save_dir / "results" / "exp4_variant_ablation" / "raw_results.csv",
            "method_col": "method_type",
            "include_methods": {"GR_RHS", "RHS"},
        },
        {
            "name": "Exp5",
            "path": save_dir / "results" / "exp5_prior_sensitivity" / "raw_results.csv",
            "method_col": None,
            "include_methods": None,
        },
    ]

    all_ok = True
    parts: list[str] = []
    for spec in specs:
        path = Path(spec["path"])
        name = str(spec["name"])
        method_col = spec["method_col"]
        include_methods = spec["include_methods"]

        if not path.exists():
            all_ok = False
            parts.append(f"{name}: missing raw_results.csv")
            continue

        rows = _read_csv(path)
        if not rows:
            all_ok = False
            parts.append(f"{name}: empty raw_results.csv")
            continue
        if "converged" not in rows[0]:
            all_ok = False
            parts.append(f"{name}: missing converged column")
            continue

        target_rows = rows
        if method_col is not None:
            if method_col not in rows[0]:
                all_ok = False
                parts.append(f"{name}: missing {method_col} column")
                continue
            allowed = set(include_methods or [])
            target_rows = [r for r in rows if str(r.get(method_col, "")).strip() in allowed]
            if not target_rows:
                all_ok = False
                parts.append(f"{name}: no bayes rows found via {method_col}")
                continue

        ok_rows = [
            r for r in target_rows
            if _to_bool(r.get("converged", False)) and str(r.get("status", "")).strip().lower() == "ok"
        ]
        n_ok = len(ok_rows)
        n_total = len(target_rows)
        is_ok = n_ok == n_total
        all_ok = all_ok and is_ok

        if not is_ok:
            err_msgs = []
            for r in target_rows:
                if _to_bool(r.get("converged", False)) and str(r.get("status", "")).strip().lower() == "ok":
                    continue
                err = str(r.get("error", "")).strip()
                if err:
                    err_msgs.append(err)
            err_uniq = sorted(set(err_msgs))
            err_preview = "; ".join(err_uniq[:2]) if err_uniq else "n/a"
            parts.append(f"{name}: {n_ok}/{n_total} converged&ok, errors={err_preview}")
        else:
            parts.append(f"{name}: {n_ok}/{n_total} converged&ok")

    return all_ok, " | ".join(parts)


def _run_acceptance(cfg: ProtocolConfig) -> int:
    checks: list[tuple[str, tuple[bool, str]]] = []
    if bool(cfg.require_all_bayes_converged):
        checks.append(("Convergence", _check_bayes_convergence_strict(cfg.save_dir)))

    checks.extend([
        ("Exp1", _check_exp1(cfg.save_dir)),
        ("Exp2", _check_exp2(cfg.save_dir)),
        ("Exp3a", _check_exp3a(cfg.save_dir, noninferiority_ratio=cfg.distributed_noninferiority_ratio)),
        ("Exp4", _check_exp4(cfg.save_dir, max_degrade_ratio=cfg.exp4_max_degrade_ratio)),
        ("Exp5", _check_exp5(cfg.save_dir)),
        ("Runtime", _check_runtime_target(cfg.save_dir, target_minutes=cfg.runtime_target_minutes)),
    ])

    lines = ["Laptop 2-3h protocol acceptance report:"]
    all_ok = True
    report_items: list[dict[str, Any]] = []
    for name, (ok, msg) in checks:
        flag = "PASS" if ok else "FAIL"
        lines.append(f"  [{flag}] {name}: {msg}")
        report_items.append({"check": name, "pass": bool(ok), "message": msg})
        all_ok = all_ok and bool(ok)

    report_text = "\n".join(lines)
    print(report_text)

    out_dir = cfg.save_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "laptop_best_2h_acceptance.txt").write_text(report_text, encoding="utf-8")
    (out_dir / "laptop_best_2h_acceptance.json").write_text(
        json.dumps({"all_pass": all_ok, "checks": report_items}, indent=2),
        encoding="utf-8",
    )

    if not all_ok:
        print(
            "[acceptance] One or more checks failed. "
            "If runtime is the blocker, downscale in this order: "
            "Exp5 repeats 3->2, then Exp3b repeats 3->2."
        )
    return 0 if all_ok else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the laptop_best_2h simulation protocol.")
    parser.add_argument(
        "--mode",
        choices=["smoke", "main", "both", "acceptance"],
        default="main",
        help="smoke: repeats=1 chain check; main: full laptop protocol; both: smoke then main; acceptance: checks only.",
    )
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--seed", type=int, default=20260415)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="Print configuration and exit.")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--skip-acceptance", action="store_true")
    parser.add_argument("--exp3b-repeats", type=int, default=3)
    parser.add_argument("--exp5-repeats", type=int, default=3)
    parser.add_argument("--distributed-noninferiority-ratio", type=float, default=1.05)
    parser.add_argument("--exp4-max-degrade-ratio", type=float, default=1.10)
    parser.add_argument("--runtime-target-minutes", type=float, default=180.0)
    parser.add_argument("--nuts-max-convergence-retries", type=int, default=1)
    parser.add_argument("--exp4-max-convergence-retries", type=int, default=2)
    parser.add_argument(
        "--allow-partial-convergence",
        action="store_true",
        help="Disable strict all-bayes convergence gating in acceptance (not recommended for final claims).",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    cfg = ProtocolConfig(
        save_dir=args.save_dir,
        seed=int(args.seed),
        n_jobs=max(1, int(args.n_jobs)),
        skip_analysis=bool(args.skip_analysis),
        skip_acceptance=bool(args.skip_acceptance),
        exp3b_repeats=max(1, int(args.exp3b_repeats)),
        exp5_repeats=max(1, int(args.exp5_repeats)),
        distributed_noninferiority_ratio=float(args.distributed_noninferiority_ratio),
        exp4_max_degrade_ratio=float(args.exp4_max_degrade_ratio),
        runtime_target_minutes=float(args.runtime_target_minutes),
        require_all_bayes_converged=not bool(args.allow_partial_convergence),
        nuts_max_convergence_retries=max(1, int(args.nuts_max_convergence_retries)),
        exp4_max_convergence_retries=max(1, int(args.exp4_max_convergence_retries)),
    )

    if args.dry_run:
        print("Dry-run configuration:")
        print(json.dumps(cfg.__dict__, indent=2, default=str))
        return 0

    if args.mode == "smoke":
        _run_smoke(cfg)
        return 0
    if args.mode == "main":
        _run_main(cfg, smoke_mode=False)
        return 0
    if args.mode == "both":
        _run_smoke(cfg)
        _run_main(cfg, smoke_mode=False)
        return 0
    if args.mode == "acceptance":
        return _run_acceptance(cfg)

    raise ValueError(f"unexpected mode: {args.mode!r}")


if __name__ == "__main__":
    raise SystemExit(main())
