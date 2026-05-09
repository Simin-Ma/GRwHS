from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one high-dimensional single-case benchmark in an isolated subprocess.")
    parser.add_argument("--config", default="Simulation_highdimension/config/highdimension.yaml")
    parser.add_argument("--setting-id", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--outdir", default="tmp/highdim_single_case_runs")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_highdim_single_case_benchmark.py"),
        "--config",
        str(args.config),
        "--setting-id",
        str(args.setting_id),
        "--method",
        str(args.method),
        "--replicate",
        str(int(args.replicate)),
        "--outdir",
        str(args.outdir),
    ]

    creationflags = 0
    popen_kwargs: dict[str, object] = {
        "cwd": str(ROOT),
        "env": env,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
    }
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        popen_kwargs["creationflags"] = creationflags
    else:
        popen_kwargs["start_new_session"] = True

    start = time.perf_counter()
    proc = subprocess.Popen(cmd, **popen_kwargs)
    try:
        out, _ = proc.communicate(timeout=max(1, int(args.timeout_seconds)))
    except subprocess.TimeoutExpired:
        _kill_process_tree(proc.pid)
        out, _ = proc.communicate()
        elapsed = time.perf_counter() - start
        print(out or "", end="")
        print(
            f"\n[isolated-runner] timed out after {elapsed:.1f}s; killed process tree rooted at PID {proc.pid}",
            file=sys.stderr,
        )
        return 124

    elapsed = time.perf_counter() - start
    print(out or "", end="")
    print(f"[isolated-runner] completed in {elapsed:.1f}s with exit code {proc.returncode}", file=sys.stderr)
    return int(proc.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
