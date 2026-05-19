from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "Simulation_highdimension" / "config" / "highdimension.yaml"
OUTDIR = ROOT / "tmp" / "highdim_four_grrhs_variants_r1"
LOG = OUTDIR / "run.log"
SUMMARY = OUTDIR / "progress.jsonl"

SETTINGS = [
    "hd_setting_1_classical_anchor",
    "hd_setting_2_classical_high",
    "hd_setting_3_dense_single_mode",
    "hd_setting_4_single_mode_unequal",
    "hd_setting_5_multimode_showcase",
    "hd_setting_6_sparse_boundary",
]
METHODS = ["GR_RHS_B01", "GR_RHS_B04", "GR_RHS_B08", "GR_RHS_Adaptive"]


def expected_path(setting: str, method: str) -> Path:
    return OUTDIR / f"{setting}__{method}__r1.json"


def log_line(message: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}\n")


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    total = len(SETTINGS) * len(METHODS)
    done = 0
    for setting in SETTINGS:
        for method in METHODS:
            done += 1
            out_path = expected_path(setting, method)
            if out_path.exists():
                log_line(f"SKIP {done}/{total} {setting} {method}")
                continue
            log_line(f"START {done}/{total} {setting} {method}")
            t0 = time.perf_counter()
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "run_highdim_single_case_benchmark.py"),
                "--config",
                str(CONFIG),
                "--setting-id",
                setting,
                "--method",
                method,
                "--replicate",
                "1",
                "--outdir",
                str(OUTDIR),
            ]
            proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
            elapsed = time.perf_counter() - t0
            payload = {
                "setting_id": setting,
                "method": method,
                "elapsed_seconds": elapsed,
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-4000:],
                "stderr_tail": proc.stderr[-4000:],
                "out_path": str(out_path),
                "out_exists": out_path.exists(),
            }
            with SUMMARY.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            status = "OK" if proc.returncode == 0 and out_path.exists() else "FAIL"
            log_line(f"{status} {done}/{total} {setting} {method} elapsed={elapsed:.1f}s")
            if proc.returncode != 0:
                return proc.returncode
    log_line("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
