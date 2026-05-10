from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_highdim_single_case_benchmark import run_case
from simulation_second.src.config import load_benchmark_config


def _prewarm_method(method: str) -> dict[str, object]:
    t0 = time.perf_counter()
    name = str(method).strip()
    note = ""
    if name == "GR_RHS":
        _fit_gr_rhs_mod = importlib.import_module(
            "simulation_project.src.experiments.methods.fit_gr_rhs"
        )

        _fit_gr_rhs_mod._load_grrhs_classes()
        note = "loaded fit_gr_rhs module and grrhs_nuts classes"
    elif name == "GIGG_MMLE":
        from simulation_project.src.experiments.methods import fit_gigg as _fit_gigg_mod

        note = f"loaded {getattr(_fit_gigg_mod, '__name__', 'fit_gigg')}"
    elif name in {"RHS", "RHS_Gibbs"}:
        if name == "RHS":
            from simulation_project.src.experiments.methods import fit_rhs as _fit_rhs_mod

            note = f"loaded {getattr(_fit_rhs_mod, '__name__', 'fit_rhs')}"
        else:
            from simulation_project.src.experiments.methods import fit_rhs_gibbs as _fit_rhs_gibbs_mod

            note = f"loaded {getattr(_fit_rhs_gibbs_mod, '__name__', 'fit_rhs_gibbs')}"
    elif name == "GHS_plus":
        from simulation_project.src.experiments.methods import fit_ghs_plus as _fit_ghs_plus_mod

        note = f"loaded {getattr(_fit_ghs_plus_mod, '__name__', 'fit_ghs_plus')}"
    elif name in {"OLS", "LASSO_CV"}:
        from simulation_project.src.experiments.methods import fit_classical as _fit_classical_mod

        note = f"loaded {getattr(_fit_classical_mod, '__name__', 'fit_classical')}"
    else:
        note = "no explicit prewarm action"
    return {
        "method": name,
        "seconds": float(time.perf_counter() - t0),
        "note": note,
    }


def _kill_process_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    try:
        os.killpg(pid, 9)
    except Exception:
        try:
            os.kill(pid, 9)
        except Exception:
            pass


def _run_direct_case(cmd: list[str], *, timeout_seconds: int) -> tuple[int, str, str]:
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    popen_kwargs: dict[str, object] = {
        "cwd": str(ROOT),
        "env": env,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True

    start = time.perf_counter()
    proc = subprocess.Popen(cmd, **popen_kwargs)
    try:
        out, _ = proc.communicate(timeout=max(1, int(timeout_seconds)))
        return int(proc.returncode or 0), out or "", ""
    except subprocess.TimeoutExpired:
        _kill_process_tree(proc.pid)
        out, _ = proc.communicate()
        elapsed = time.perf_counter() - start
        tail = out or ""
        err = f"[main-singlecases] timed out after {elapsed:.1f}s; killed process tree rooted at PID {proc.pid}\n"
        return 124, tail, err


def main() -> int:
    parser = argparse.ArgumentParser(description="Run high-dimensional main benchmark as sequential single-case jobs.")
    parser.add_argument("--config", default="Simulation_highdimension/config/highdimension.yaml")
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--outdir", default="tmp/highdim_main_singlecases")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--isolated", action="store_true", help="Use the extra isolated wrapper process per case.")
    parser.add_argument("--method-first", action="store_true", help="Group jobs by method to improve warm-cache reuse.")
    parser.add_argument("--methods", nargs="*", default=None, help="Optional subset of methods to run.")
    parser.add_argument("--settings", nargs="*", default=None, help="Optional subset of setting ids to run.")
    parser.add_argument("--prewarm-methods", nargs="*", default=None, help="Optional methods to prewarm before running jobs.")
    args = parser.parse_args()

    cfg = load_benchmark_config(args.config)
    outdir = ROOT / str(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    requested_methods = {str(m) for m in (args.methods or [])}
    requested_settings = {str(s) for s in (args.settings or [])}
    prewarm_methods = [str(m) for m in (args.prewarm_methods or [])]

    settings_use = [
        setting for setting in cfg.settings
        if not requested_settings or str(setting.setting_id) in requested_settings
    ]

    jobs: list[dict[str, str]] = []
    if bool(args.method_first):
        roster = list(cfg.methods.roster)
        for method in roster:
            if requested_methods and str(method) not in requested_methods:
                continue
            for setting in settings_use:
                methods = list(setting.methods or cfg.methods.roster)
                if str(method) not in {str(m) for m in methods}:
                    continue
                jobs.append(
                    {
                        "setting_id": str(setting.setting_id),
                        "method": str(method),
                    }
                )
    else:
        for setting in settings_use:
            methods = list(setting.methods or cfg.methods.roster)
            for method in methods:
                if requested_methods and str(method) not in requested_methods:
                    continue
                jobs.append(
                    {
                        "setting_id": str(setting.setting_id),
                        "method": str(method),
                    }
                )

    manifest = {
        "config": str(args.config),
        "replicate": int(args.replicate),
        "timeout_seconds": int(args.timeout_seconds),
        "method_first": bool(args.method_first),
        "methods_filter": sorted(requested_methods),
        "settings_filter": sorted(requested_settings),
        "prewarm_methods": list(prewarm_methods),
        "jobs": jobs,
    }
    (outdir / "job_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    results: list[dict[str, object]] = []
    prewarm_records: list[dict[str, object]] = []
    for method in prewarm_methods:
        try:
            rec = _prewarm_method(method)
            rec["status"] = "ok"
        except Exception as exc:
            rec = {
                "method": str(method),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        prewarm_records.append(rec)
    if prewarm_records:
        (outdir / "prewarm_summary.json").write_text(json.dumps(prewarm_records, indent=2, ensure_ascii=False), encoding="utf-8")

    for job in jobs:
        result_path = outdir / f"{job['setting_id']}__{job['method']}__r{int(args.replicate)}.json"
        if result_path.exists():
            try:
                existing = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception:
                existing = None
            if isinstance(existing, dict) and str(existing.get("status", "")) == "ok":
                payload = {
                    "setting_id": str(job["setting_id"]),
                    "method": str(job["method"]),
                    "exit_code": 0,
                    "result_path": str(result_path),
                    "result": existing,
                    "skipped_existing_ok": True,
                }
                results.append(payload)
                (outdir / "run_progress.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
                continue

        if bool(args.isolated):
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "run_highdim_single_case_isolated.py"),
                "--config",
                str(args.config),
                "--setting-id",
                str(job["setting_id"]),
                "--method",
                str(job["method"]),
                "--replicate",
                str(int(args.replicate)),
                "--outdir",
                str(args.outdir),
                "--timeout-seconds",
                str(int(args.timeout_seconds)),
            ]
            completed = subprocess.run(
                cmd,
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            stdout_tail = completed.stdout[-4000:] if completed.stdout else ""
            stderr_tail = completed.stderr[-4000:] if completed.stderr else ""
            payload: dict[str, object] = {
                "setting_id": str(job["setting_id"]),
                "method": str(job["method"]),
                "exit_code": int(completed.returncode),
                "result_path": str(result_path),
            }
            if result_path.exists():
                try:
                    payload["result"] = json.loads(result_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    payload["result_load_error"] = f"{type(exc).__name__}: {exc}"
            else:
                payload["stdout_tail"] = stdout_tail[-4000:] if stdout_tail else ""
                payload["stderr_tail"] = stderr_tail[-4000:] if stderr_tail else ""
        else:
            try:
                result = run_case(
                    config=str(args.config),
                    setting_id=str(job["setting_id"]),
                    method=str(job["method"]),
                    replicate=int(args.replicate),
                    outdir=str(args.outdir),
                )
                payload = {
                    "setting_id": str(job["setting_id"]),
                    "method": str(job["method"]),
                    "exit_code": 0,
                    "result_path": str(result_path),
                    "result": result,
                }
            except Exception as exc:
                payload = {
                    "setting_id": str(job["setting_id"]),
                    "method": str(job["method"]),
                    "exit_code": 1,
                    "result_path": str(result_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
        results.append(payload)
        (outdir / "run_progress.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "config": str(args.config),
        "replicate": int(args.replicate),
        "n_jobs": int(len(jobs)),
        "prewarm": prewarm_records,
        "prewarm_recommended": bool(args.method_first and requested_methods == {"GR_RHS"} and not prewarm_methods),
        "results": results,
    }
    (outdir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
