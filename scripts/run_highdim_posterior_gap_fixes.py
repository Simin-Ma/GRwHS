from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SETTINGS = [
    "hd_setting_1_classical_anchor",
    "hd_setting_2_single_mode",
    "hd_setting_3_multimode_showcase",
]


def _run(cmd: list[str], *, timeout_seconds: int) -> dict[str, object]:
    t0 = time.perf_counter()
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=int(timeout_seconds),
        check=False,
    )
    return {
        "cmd": cmd,
        "exit_code": int(completed.returncode),
        "seconds": float(time.perf_counter() - t0),
        "output_tail": (completed.stdout or "")[-6000:],
    }


def _json_ok(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return False
    return bool(payload.get("status") == "ok" and payload.get("converged") is True)


def _run_standard_case(
    *,
    method: str,
    setting: str,
    rep: int,
    outdir: Path,
    timeout_seconds: int,
    force: bool,
) -> dict[str, object]:
    out_path = outdir / f"{setting}__{method}__r{rep}.json"
    if out_path.exists() and not force:
        return {"status": "skipped_existing", "path": str(out_path), "converged": _json_ok(out_path)}
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_highdim_single_case_benchmark.py"),
        "--setting-id",
        setting,
        "--method",
        method,
        "--replicate",
        str(rep),
        "--outdir",
        str(outdir.relative_to(ROOT)),
    ]
    rec = _run(cmd, timeout_seconds=timeout_seconds)
    rec["path"] = str(out_path)
    rec["converged"] = _json_ok(out_path)
    return rec


def _run_rhs_strict_case(
    *,
    setting: str,
    rep: int,
    outdir: Path,
    timeout_seconds: int,
    force: bool,
    seed_offset: int,
) -> dict[str, object]:
    out_path = outdir / f"{setting}__RHS__r{rep}__w2500_d5000_s{seed_offset}.json"
    if out_path.exists() and not force:
        return {"status": "skipped_existing", "path": str(out_path), "converged": _json_ok(out_path)}
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_highdim_rhs_setting2_retry.py"),
        "--setting-id",
        setting,
        "--replicate",
        str(rep),
        "--outdir",
        str(outdir.relative_to(ROOT)),
        "--warmup",
        "2500",
        "--draws",
        "5000",
        "--adapt-delta",
        "0.995",
        "--max-treedepth",
        "15",
        "--seed-offset",
        str(seed_offset),
    ]
    rec = _run(cmd, timeout_seconds=timeout_seconds)
    rec["path"] = str(out_path)
    rec["converged"] = _json_ok(out_path)
    return rec


def main() -> int:
    parser = argparse.ArgumentParser(description="Rerun high-dimensional posterior diagnostics gaps.")
    parser.add_argument("--outroot", default="tmp/highdim_posterior_gap_fixes")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rhs-timeout", type=int, default=7200)
    parser.add_argument("--gigg-timeout", type=int, default=7200)
    parser.add_argument("--methods", default="RHS,GIGG_MMLE")
    parser.add_argument("--settings", default=",".join(SETTINGS))
    parser.add_argument("--replicates", default="1,2,3,4,5")
    args = parser.parse_args()

    outroot = ROOT / str(args.outroot)
    rhs_dir = outroot / "rhs"
    gigg_dir = outroot / "gigg"
    rhs_strict_dir = outroot / "rhs_strict"
    for folder in [rhs_dir, gigg_dir, rhs_strict_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    methods = {m.strip() for m in str(args.methods).split(",") if m.strip()}
    settings = [s.strip() for s in str(args.settings).split(",") if s.strip()]
    reps = [int(r.strip()) for r in str(args.replicates).split(",") if r.strip()]
    log_path = outroot / "run_log.json"
    if log_path.exists() and not args.force:
        try:
            log = json.loads(log_path.read_text(encoding="utf-8"))
            if not isinstance(log, list):
                log = []
        except Exception:
            log = []
    else:
        log = []

    for setting in settings:
        for rep in reps:
            if "RHS" in methods:
                rec = _run_standard_case(
                    method="RHS",
                    setting=setting,
                    rep=rep,
                    outdir=rhs_dir,
                    timeout_seconds=int(args.rhs_timeout),
                    force=bool(args.force),
                )
                rec.update({"setting_id": setting, "replicate": rep, "method": "RHS", "stage": "rhs_standard"})
                log.append(rec)
                log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
                if not bool(rec.get("converged")):
                    retry = _run_rhs_strict_case(
                        setting=setting,
                        rep=rep,
                        outdir=rhs_strict_dir,
                        timeout_seconds=int(args.rhs_timeout),
                        force=bool(args.force),
                        seed_offset=70 + rep,
                    )
                    retry.update({"setting_id": setting, "replicate": rep, "method": "RHS", "stage": "rhs_strict"})
                    log.append(retry)
                    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

            if "GIGG_MMLE" in methods:
                rec = _run_standard_case(
                    method="GIGG_MMLE",
                    setting=setting,
                    rep=rep,
                    outdir=gigg_dir,
                    timeout_seconds=int(args.gigg_timeout),
                    force=bool(args.force),
                )
                rec.update({"setting_id": setting, "replicate": rep, "method": "GIGG_MMLE", "stage": "gigg_standard"})
                log.append(rec)
                log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"outroot": str(outroot), "n_log": len(log)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
