from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation_second.src.config import load_benchmark_config


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
    args = parser.parse_args()

    cfg = load_benchmark_config(args.config)
    outdir = ROOT / str(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, str]] = []
    for setting in cfg.settings:
        methods = list(setting.methods or cfg.methods.roster)
        for method in methods:
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
        "jobs": jobs,
    }
    (outdir / "job_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    results: list[dict[str, object]] = []
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
        else:
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "run_highdim_single_case_benchmark.py"),
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
            ]
            rc, stdout_tail, stderr_tail = _run_direct_case(cmd, timeout_seconds=int(args.timeout_seconds))
            completed = subprocess.CompletedProcess(cmd, rc, stdout_tail, stderr_tail)
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
        results.append(payload)
        (outdir / "run_progress.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "config": str(args.config),
        "replicate": int(args.replicate),
        "n_jobs": int(len(jobs)),
        "results": results,
    }
    (outdir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
