from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SETTINGS = [
    "hd_setting_1_classical_anchor",
    "hd_setting_2_single_mode",
    "hd_setting_3_multimode_showcase",
]
METHODS = ["GR_RHS", "RHS", "GIGG_MMLE", "GHS_plus_NUTS"]


def _run(cmd: list[str], *, timeout_seconds: int) -> dict[str, object]:
    t0 = time.perf_counter()
    env = None
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=int(timeout_seconds),
        check=False,
        env=env,
    )
    return {
        "cmd": cmd,
        "exit_code": int(completed.returncode),
        "seconds": float(time.perf_counter() - t0),
        "output_tail": (completed.stdout or "")[-6000:],
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
    method: str,
    setting: str,
    rep: int,
    outdir: str,
    timeout_seconds: int,
    force: bool,
) -> dict[str, object]:
    out_path = ROOT / outdir / f"{setting}__{method}__r{rep}.json"
    if not force and out_path.exists():
        return {"status": "skipped_existing", "path": str(out_path), "converged": _is_converged(out_path)}
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
        outdir,
    ]
    rec = _run(cmd, timeout_seconds=timeout_seconds)
    rec["path"] = str(out_path)
    rec["converged"] = _is_converged(out_path)
    return rec


def _run_rhs_retry(
    *,
    setting: str,
    rep: int,
    outdir: str,
    timeout_seconds: int,
    force: bool,
) -> dict[str, object]:
    pattern = f"{setting}__RHS__r{rep}__w4000_d8000_s*.json"
    existing = _latest_json(ROOT / outdir, pattern)
    if existing is not None and (not force) and _is_converged(existing):
        return {"status": "skipped_existing_retry", "path": str(existing), "converged": True}
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_highdim_rhs_setting2_retry.py"),
        "--setting-id",
        setting,
        "--replicate",
        str(rep),
        "--outdir",
        outdir,
        "--warmup",
        "4000",
        "--draws",
        "8000",
        "--adapt-delta",
        "0.995",
        "--max-treedepth",
        "15",
        "--seed-offset",
        "11",
    ]
    rec = _run(cmd, timeout_seconds=timeout_seconds)
    retry_path = _latest_json(ROOT / outdir, pattern)
    rec["path"] = str(retry_path) if retry_path is not None else ""
    rec["converged"] = bool(retry_path is not None and _is_converged(retry_path))
    return rec


def _run_ghs_case(
    *,
    setting: str,
    rep: int,
    outdir: str,
    timeout_seconds: int,
    force: bool,
    warmup: int,
    draws: int,
    target_accept: float,
) -> dict[str, object]:
    out_path = ROOT / outdir / f"{setting}__GHS_plus_NUTS__r{rep}_w{warmup}_d{draws}.json"
    if not force and out_path.exists():
        return {"status": "skipped_existing", "path": str(out_path), "converged": _is_converged(out_path)}
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_highdim_ghs_plus_nuts_probe.py"),
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
        "12",
        "--target-accept",
        str(target_accept),
    ]
    rec = _run(cmd, timeout_seconds=timeout_seconds)
    rec["path"] = str(out_path)
    rec["converged"] = _is_converged(out_path)
    return rec


def _scalar(value):
    if value is None:
        return math.nan
    try:
        val = float(value)
    except Exception:
        return value
    return val if math.isfinite(val) else math.nan


def _rep_from_name(path: Path) -> int:
    m = re.search(r"__r(\d+)", path.name)
    return int(m.group(1)) if m else 0


def _row(path: Path, *, method: str | None = None) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    budget = payload.get("retry_budget", {})
    return {
        "setting_id": str(payload.get("setting_id", "")),
        "method": str(method or payload.get("method", "")),
        "replicate": int(payload.get("replicate", _rep_from_name(path))),
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
        "runtime_seconds": _scalar(payload.get("runtime_seconds")),
        "wall_seconds": _scalar(payload.get("wall_seconds")),
        "warmup": int(budget.get("warmup", 0)) if isinstance(budget, dict) and budget.get("warmup") else math.nan,
        "draws": int(budget.get("draws", 0)) if isinstance(budget, dict) and budget.get("draws") else math.nan,
        "source_file": str(path.relative_to(ROOT)),
    }


def _collect_rows(outroot: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    standard = outroot / "standard_cases"
    rhs_retry = outroot / "rhs_retries"
    ghs = outroot / "ghs_plus_nuts"
    for path in standard.glob("*.json"):
        rows.append(_row(path))
    for path in rhs_retry.glob("*.json"):
        rows.append(_row(path))
    for path in ghs.glob("*.json"):
        rows.append(_row(path, method="GHS_plus_NUTS"))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["setting_id"].isin(SETTINGS) & df["method"].isin(METHODS)].copy()
    df["_conv"] = df["converged"].astype(int)
    df["_rhat"] = pd.to_numeric(df["rhat_max"], errors="coerce").fillna(float("inf"))
    df["_ess"] = pd.to_numeric(df["ess_min"], errors="coerce").fillna(float("-inf"))
    df["_draws"] = pd.to_numeric(df["draws"], errors="coerce").fillna(0.0)
    df = df.sort_values(
        ["setting_id", "method", "replicate", "_conv", "_rhat", "_ess", "_draws"],
        ascending=[True, True, True, False, True, False, False],
    )
    df = df.groupby(["setting_id", "method", "replicate"], sort=False, as_index=False).head(1)
    return df.drop(columns=[c for c in df.columns if c.startswith("_")]).sort_values(["setting_id", "method", "replicate"])


def _summaries(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for (setting, method), g in raw.groupby(["setting_id", "method"], sort=False):
        eligible = int(g["converged"].sum()) == 5 and int(g.shape[0]) == 5
        use = g[g["converged"] == True] if eligible else g.iloc[0:0]
        row = {
            "setting_id": setting,
            "method": method,
            "n_runs": int(g.shape[0]),
            "n_converged": int(g["converged"].sum()),
            "eligible_5of5": bool(eligible),
        }
        for metric in [
            "mse_overall",
            "mse_signal",
            "mse_null",
            "coverage_95",
            "lpd_test",
            "rhat_max",
            "ess_min",
            "runtime_seconds",
            "wall_seconds",
        ]:
            vals = pd.to_numeric(use[metric], errors="coerce") if metric in use else pd.Series(dtype=float)
            row[f"{metric}_mean"] = float(vals.mean()) if vals.notna().any() else math.nan
            row[f"{metric}_sd"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else math.nan
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["setting_id", "method"])
    ranking = summary[summary["eligible_5of5"] == True].copy()
    ranking = ranking.sort_values(["setting_id", "mse_overall_mean", "method"])
    ranking["rank_mse_overall"] = ranking.groupby("setting_id").cumcount() + 1
    ranking = ranking[
        [
            "setting_id",
            "rank_mse_overall",
            "method",
            "mse_overall_mean",
            "mse_signal_mean",
            "mse_null_mean",
            "coverage_95_mean",
            "rhat_max_mean",
            "ess_min_mean",
            "wall_seconds_mean",
        ]
    ]
    return summary, ranking


def _write_report(outroot: Path, summary: pd.DataFrame, ranking: pd.DataFrame) -> None:
    lines = [
        "# 2026-05-12 高维四种贝叶斯方法重新运行比较",
        "",
        "本轮只使用 `tmp/highdim_bayes_rerun_20260512_full` 中新生成的结果。",
        "准入规则：每个 setting 下每个方法 5 次重复都完整后验收敛，才进入该 setting 的质量排名。",
        "",
        "## 收敛准入",
        "",
        "| Setting | 方法 | 收敛次数 | 可比较 |",
        "|---|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['setting_id']} | {row['method']} | {int(row['n_converged'])}/{int(row['n_runs'])} | "
            f"{'是' if bool(row['eligible_5of5']) else '否'} |"
        )
    lines.extend(["", "## 按 MSE overall 排名", "", "| Setting | 排名 | 方法 | MSE overall | MSE signal | MSE null | Coverage | 平均 Rhat | 平均 ESS-min | 平均墙钟秒 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for _, row in ranking.iterrows():
        lines.append(
            f"| {row['setting_id']} | {int(row['rank_mse_overall'])} | {row['method']} | "
            f"{row['mse_overall_mean']:.6g} | {row['mse_signal_mean']:.6g} | {row['mse_null_mean']:.6g} | "
            f"{row['coverage_95_mean']:.4f} | {row['rhat_max_mean']:.6g} | {row['ess_min_mean']:.2f} | "
            f"{row['wall_seconds_mean']:.2f} |"
        )
    (outroot / "rerun_report_cn.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fresh rerun for high-dimensional four Bayesian methods.")
    parser.add_argument("--outroot", default="tmp/highdim_bayes_rerun_20260512_full")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--standard-timeout", type=int, default=7200)
    parser.add_argument("--rhs-retry-timeout", type=int, default=7200)
    parser.add_argument("--ghs-timeout", type=int, default=3600)
    args = parser.parse_args()

    outroot = ROOT / str(args.outroot)
    standard_dir = outroot / "standard_cases"
    rhs_retry_dir = outroot / "rhs_retries"
    ghs_dir = outroot / "ghs_plus_nuts"
    for folder in [standard_dir, rhs_retry_dir, ghs_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    log: list[dict[str, object]] = []
    for setting in SETTINGS:
        for rep in range(1, 6):
            for method in ["GR_RHS", "RHS", "GIGG_MMLE"]:
                rec = _run_standard_case(
                    method=method,
                    setting=setting,
                    rep=rep,
                    outdir=str(standard_dir.relative_to(ROOT)),
                    timeout_seconds=int(args.standard_timeout),
                    force=bool(args.force),
                )
                rec.update({"setting_id": setting, "replicate": rep, "method": method, "stage": "standard"})
                log.append(rec)
                (outroot / "run_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
                if method == "RHS" and not bool(rec.get("converged")):
                    retry = _run_rhs_retry(
                        setting=setting,
                        rep=rep,
                        outdir=str(rhs_retry_dir.relative_to(ROOT)),
                        timeout_seconds=int(args.rhs_retry_timeout),
                        force=bool(args.force),
                    )
                    retry.update({"setting_id": setting, "replicate": rep, "method": method, "stage": "rhs_retry"})
                    log.append(retry)
                    (outroot / "run_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

            ghs_rec = _run_ghs_case(
                setting=setting,
                rep=rep,
                outdir=str(ghs_dir.relative_to(ROOT)),
                timeout_seconds=int(args.ghs_timeout),
                force=bool(args.force),
                warmup=3000,
                draws=3000,
                target_accept=0.97,
            )
            ghs_rec.update({"setting_id": setting, "replicate": rep, "method": "GHS_plus_NUTS", "stage": "ghs_3000"})
            log.append(ghs_rec)
            (outroot / "run_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
            if not bool(ghs_rec.get("converged")):
                retry = _run_ghs_case(
                    setting=setting,
                    rep=rep,
                    outdir=str(ghs_dir.relative_to(ROOT)),
                    timeout_seconds=int(args.ghs_timeout),
                    force=bool(args.force),
                    warmup=5000,
                    draws=5000,
                    target_accept=0.98,
                )
                retry.update({"setting_id": setting, "replicate": rep, "method": "GHS_plus_NUTS", "stage": "ghs_5000"})
                log.append(retry)
                (outroot / "run_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

    raw = _collect_rows(outroot)
    summary, ranking = _summaries(raw)
    raw.to_csv(outroot / "rerun_raw.csv", index=False)
    summary.to_csv(outroot / "rerun_summary.csv", index=False)
    ranking.to_csv(outroot / "rerun_rankings.csv", index=False)
    _write_report(outroot, summary, ranking)
    print(summary[["setting_id", "method", "n_converged", "n_runs", "eligible_5of5"]].to_string(index=False))
    print(ranking.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
