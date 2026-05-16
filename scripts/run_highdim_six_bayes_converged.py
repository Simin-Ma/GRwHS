from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation_second.src.config import load_benchmark_config


METHODS = ["GR_RHS", "RHS", "GIGG_MMLE", "GHS_plus_NUTS"]


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
        "output_tail": (completed.stdout or "")[-5000:],
    }


def _read_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _is_converged(path: Path) -> bool:
    payload = _read_json(path)
    return bool(isinstance(payload, dict) and payload.get("status") == "ok" and payload.get("converged") is True)


def _latest_json(folder: Path, pattern: str) -> Path | None:
    paths = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime)
    return paths[-1] if paths else None


def _run_standard_case(
    *,
    config: str,
    setting: str,
    method: str,
    rep: int,
    outdir: str,
    timeout_seconds: int,
    force: bool,
) -> dict[str, object]:
    out_path = ROOT / outdir / f"{setting}__{method}__r{rep}.json"
    if out_path.exists() and (not force):
        return {"status": "skipped_existing", "path": str(out_path), "converged": _is_converged(out_path)}
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_highdim_single_case_benchmark.py"),
        "--config",
        config,
        "--setting-id",
        setting,
        "--method",
        method,
        "--replicate",
        str(rep),
        "--outdir",
        outdir,
    ]
    rec = _run(cmd, timeout_seconds=timeout_seconds)
    rec["path"] = str(out_path)
    rec["converged"] = _is_converged(out_path)
    return rec


def _run_gigg_case(
    *,
    config: str,
    setting: str,
    rep: int,
    outdir: str,
    timeout_seconds: int,
    force: bool,
    rounds: int,
    draws_per_round: int,
) -> dict[str, object]:
    folder = ROOT / outdir
    pattern = f"{setting}__GIGG_MMLE__r{rep}__rounds*_dpr*_s*.json"
    existing = _latest_json(folder, pattern)
    if existing is not None and (not force) and _is_converged(existing):
        return {"status": "skipped_existing", "path": str(existing), "converged": True}
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_highdim_gigg_retry_case.py"),
        "--config",
        config,
        "--setting-id",
        setting,
        "--replicate",
        str(rep),
        "--outdir",
        outdir,
        "--rounds",
        str(rounds),
        "--draws-per-round",
        str(draws_per_round),
        "--seed-offset",
        "31",
    ]
    rec = _run(cmd, timeout_seconds=timeout_seconds)
    latest = _latest_json(folder, pattern)
    rec["path"] = str(latest) if latest is not None else ""
    rec["converged"] = bool(latest is not None and _is_converged(latest))
    return rec


def _run_ghs_nuts_case(
    *,
    config: str,
    setting: str,
    rep: int,
    outdir: str,
    timeout_seconds: int,
    force: bool,
    budgets: list[tuple[int, int, float, int]],
) -> dict[str, object]:
    folder = ROOT / outdir
    last: dict[str, object] = {}
    for warmup, draws, target_accept, max_tree_depth in budgets:
        out_path = folder / f"{setting}__GHS_plus_NUTS__r{rep}_w{warmup}_d{draws}.json"
        if out_path.exists() and (not force) and _is_converged(out_path):
            return {"status": "skipped_existing", "path": str(out_path), "converged": True}
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_highdim_ghs_plus_nuts_probe.py"),
            "--config",
            config,
            "--setting-id",
            setting,
            "--replicate",
            str(rep),
            "--outdir",
            outdir,
            "--warmup",
            str(warmup),
            "--draws",
            str(draws),
            "--chains",
            "4",
            "--max-tree-depth",
            str(max_tree_depth),
            "--target-accept",
            str(target_accept),
        ]
        last = _run(cmd, timeout_seconds=timeout_seconds)
        last["path"] = str(out_path)
        last["converged"] = _is_converged(out_path)
        if bool(last["converged"]):
            return last
    return last


def _scalar(value: object) -> float | str:
    if value is None:
        return math.nan
    try:
        val = float(value)
    except Exception:
        return str(value)
    return val if math.isfinite(val) else math.nan


def _row(path: Path, *, method_override: str | None = None) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    method = str(method_override or payload.get("method", ""))
    return {
        "setting_id": str(payload.get("setting_id", "")),
        "method": method,
        "replicate": int(payload.get("replicate", 0)),
        "status": str(payload.get("status", "")),
        "converged": bool(payload.get("converged", False)),
        "rhat_max": _scalar(payload.get("rhat_max")),
        "ess_min": _scalar(payload.get("ess_min")),
        "divergence_ratio": _scalar(payload.get("divergence_ratio")),
        "mse_overall": _scalar(payload.get("mse_overall")),
        "mse_signal": _scalar(payload.get("mse_signal")),
        "mse_null": _scalar(payload.get("mse_null")),
        "coverage_95": _scalar(payload.get("coverage_95")),
        "lpd_test": _scalar(payload.get("lpd_test")),
        "wall_seconds": _scalar(payload.get("wall_seconds")),
        "source_file": str(path.relative_to(ROOT)),
    }


def _collect(outroot: Path, settings: list[str], rep: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sub in ["standard", "gigg", "ghs_plus_nuts"]:
        folder = outroot / sub
        if not folder.exists():
            continue
        for path in folder.glob("*.json"):
            method_override = "GHS_plus_NUTS" if "GHS_plus_NUTS" in path.name else None
            rows.append(_row(path, method_override=method_override))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df[df["setting_id"].isin(settings) & df["method"].isin(METHODS) & (df["replicate"] == int(rep))].copy()
    if df.empty:
        return df
    df["_conv"] = df["converged"].astype(int)
    df["_rhat"] = pd.to_numeric(df["rhat_max"], errors="coerce").fillna(float("inf"))
    df["_ess"] = pd.to_numeric(df["ess_min"], errors="coerce").fillna(float("-inf"))
    df = df.sort_values(
        ["setting_id", "method", "replicate", "_conv", "_rhat", "_ess"],
        ascending=[True, True, True, False, True, False],
    )
    df = df.groupby(["setting_id", "method", "replicate"], sort=False, as_index=False).head(1)
    return df.drop(columns=[c for c in df.columns if c.startswith("_")]).sort_values(["setting_id", "method"])


def _write_report(df: pd.DataFrame, outroot: Path, settings: list[str]) -> None:
    outroot.mkdir(parents=True, exist_ok=True)
    df.to_csv(outroot / "six_setting_bayes_raw_best.csv", index=False)
    if df.empty:
        (outroot / "six_setting_bayes_report.md").write_text("No results collected.\n", encoding="utf-8")
        return
    summary = (
        df.groupby(["setting_id", "method"], as_index=False)
        .agg(
            converged=("converged", "max"),
            rhat_max=("rhat_max", "min"),
            ess_min=("ess_min", "max"),
            mse_overall=("mse_overall", "mean"),
            mse_signal=("mse_signal", "mean"),
            mse_null=("mse_null", "mean"),
            lpd_test=("lpd_test", "mean"),
            wall_seconds=("wall_seconds", "mean"),
        )
        .sort_values(["setting_id", "mse_overall"], kind="stable")
    )
    summary.to_csv(outroot / "six_setting_bayes_summary.csv", index=False)
    lines = [
        "# Six-Setting High-Dimensional Bayesian Comparison",
        "",
        "Only converged runs should be used for final quality claims.",
        "",
    ]
    for setting in settings:
        chunk = summary[summary["setting_id"] == setting].copy()
        if chunk.empty:
            continue
        lines.append(f"## {setting}")
        lines.append("")
        lines.append("```text")
        lines.append(chunk.to_string(index=False))
        lines.append("```")
        lines.append("")
    (outroot / "six_setting_bayes_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run converged Bayesian comparison for Simulation_highdimension HD1-HD6.")
    parser.add_argument("--config", default="Simulation_highdimension/config/highdimension.yaml")
    parser.add_argument("--outroot", default="tmp/highdim_six_bayes_converged")
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--settings", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=METHODS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout-standard", type=int, default=1200)
    parser.add_argument("--timeout-ghs", type=int, default=1500)
    parser.add_argument("--timeout-gigg", type=int, default=1200)
    parser.add_argument("--gigg-rounds", type=int, default=260)
    parser.add_argument("--gigg-draws-per-round", type=int, default=5)
    args = parser.parse_args()

    cfg = load_benchmark_config(args.config)
    settings = [s.setting_id for s in cfg.settings]
    if args.settings:
        wanted = set(str(x) for x in args.settings)
        settings = [s for s in settings if s in wanted]
    methods = [m for m in METHODS if m in set(str(x) for x in args.methods)]
    outroot = ROOT / str(args.outroot)
    log: list[dict[str, object]] = []

    for setting in settings:
        for method in methods:
            print(f"[run] setting={setting} method={method} rep={args.replicate}", flush=True)
            if method in {"GR_RHS", "RHS"}:
                rec = _run_standard_case(
                    config=str(args.config),
                    setting=setting,
                    method=method,
                    rep=int(args.replicate),
                    outdir=str(Path(args.outroot) / "standard"),
                    timeout_seconds=int(args.timeout_standard),
                    force=bool(args.force),
                )
            elif method == "GIGG_MMLE":
                rec = _run_gigg_case(
                    config=str(args.config),
                    setting=setting,
                    rep=int(args.replicate),
                    outdir=str(Path(args.outroot) / "gigg"),
                    timeout_seconds=int(args.timeout_gigg),
                    force=bool(args.force),
                    rounds=int(args.gigg_rounds),
                    draws_per_round=int(args.gigg_draws_per_round),
                )
            elif method == "GHS_plus_NUTS":
                rec = _run_ghs_nuts_case(
                    config=str(args.config),
                    setting=setting,
                    rep=int(args.replicate),
                    outdir=str(Path(args.outroot) / "ghs_plus_nuts"),
                    timeout_seconds=int(args.timeout_ghs),
                    force=bool(args.force),
                    budgets=[(3000, 3000, 0.95, 10), (6000, 6000, 0.97, 11)],
                )
            else:
                continue
            rec.update({"setting_id": setting, "method": method, "replicate": int(args.replicate)})
            log.append(rec)
            (outroot / "run_log.json").parent.mkdir(parents=True, exist_ok=True)
            (outroot / "run_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
            df = _collect(outroot, settings, int(args.replicate))
            _write_report(df, outroot, settings)
            print(f"[done] converged={rec.get('converged')} path={rec.get('path', '')}", flush=True)

    df = _collect(outroot, settings, int(args.replicate))
    _write_report(df, outroot, settings)
    print(f"Artifacts saved in: {outroot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
