from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd


SETTINGS = [
    "hd_setting_1_classical_anchor",
    "hd_setting_2_single_mode",
    "hd_setting_3_multimode_showcase",
]
METHODS = ["GR_RHS", "RHS", "GIGG_MMLE", "GHS_plus_NUTS"]


def _scalar(value):
    if value is None:
        return math.nan
    try:
        val = float(value)
    except Exception:
        return value
    return val if math.isfinite(val) else math.nan


def _rep_from_name(path: Path) -> int | None:
    m = re.search(r"__r(\d+)", path.name)
    if not m:
        return None
    return int(m.group(1))


def _row_from_json(path: Path, *, method: str | None = None) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = {
        "setting_id": str(payload.get("setting_id", "")),
        "method": str(method or payload.get("method", "")),
        "replicate": int(payload.get("replicate", _rep_from_name(path) or 0)),
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
        "source_file": str(path.relative_to(ROOT)),
    }
    retry_budget = payload.get("retry_budget", {})
    if isinstance(retry_budget, dict):
        row["nuts_warmup"] = retry_budget.get("warmup", math.nan)
        row["nuts_draws"] = retry_budget.get("draws", math.nan)
    return row


def _load_candidates() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base = ROOT / "tmp" / "highdim_bayes_all4_r1to5"
    for setting in SETTINGS:
        for method in ["GR_RHS", "RHS", "GIGG_MMLE"]:
            for rep in range(1, 6):
                path = base / f"{setting}__{method}__r{rep}.json"
                if path.exists():
                    rows.append(_row_from_json(path))

    rhs_retry = ROOT / "tmp" / "highdim_rhs_setting2_retry_full"
    for path in rhs_retry.glob("hd_setting_2_single_mode__RHS__r*.json"):
        rows.append(_row_from_json(path))

    for folder in ["highdim_ghs_plus_nuts_formal", "highdim_ghs_plus_nuts_retry"]:
        for path in (ROOT / "tmp" / folder).glob("*__GHS_plus_NUTS__r*.json"):
            rows.append(_row_from_json(path, method="GHS_plus_NUTS"))
    return rows


def _select_best(df: pd.DataFrame) -> pd.DataFrame:
    # Deterministic priority: converged > lower rhat > higher ESS > larger draw budget.
    work = df.copy()
    for col in ["nuts_warmup", "nuts_draws"]:
        if col not in work.columns:
            work[col] = math.nan
    work["_conv_rank"] = work["converged"].astype(int)
    work["_rhat_rank"] = pd.to_numeric(work["rhat_max"], errors="coerce").fillna(float("inf"))
    work["_ess_rank"] = pd.to_numeric(work["ess_min"], errors="coerce").fillna(float("-inf"))
    work["_draw_rank"] = pd.to_numeric(work["nuts_draws"], errors="coerce").fillna(0.0)
    work = work.sort_values(
        ["setting_id", "method", "replicate", "_conv_rank", "_rhat_rank", "_ess_rank", "_draw_rank"],
        ascending=[True, True, True, False, True, False, False],
    )
    picked = work.groupby(["setting_id", "method", "replicate"], as_index=False, sort=False).head(1)
    return picked.drop(columns=[c for c in picked.columns if c.startswith("_")])


def _summarize(raw: pd.DataFrame) -> pd.DataFrame:
    grouped = raw.groupby(["setting_id", "method"], sort=False)
    rows: list[dict[str, object]] = []
    for (setting, method), g in grouped:
        conv = g[g["converged"] == True]
        eligible = int(conv.shape[0]) == 5 and int(g.shape[0]) == 5
        row = {
            "setting_id": setting,
            "method": method,
            "n_runs": int(g.shape[0]),
            "n_converged": int(conv.shape[0]),
            "eligible_5of5": bool(eligible),
        }
        metrics = [
            "mse_overall",
            "mse_signal",
            "mse_null",
            "coverage_95",
            "lpd_test",
            "rhat_max",
            "ess_min",
            "runtime_seconds",
            "wall_seconds",
        ]
        src = conv if eligible else g[g["converged"] == True]
        for metric in metrics:
            values = pd.to_numeric(src[metric], errors="coerce") if metric in src else pd.Series(dtype=float)
            row[f"{metric}_mean"] = float(values.mean()) if values.notna().any() and eligible else math.nan
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if values.notna().sum() > 1 and eligible else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _rank(summary: pd.DataFrame) -> pd.DataFrame:
    eligible = summary[summary["eligible_5of5"] == True].copy()
    eligible = eligible.sort_values(["setting_id", "mse_overall_mean", "method"])
    eligible["rank_mse_overall"] = eligible.groupby("setting_id").cumcount() + 1
    return eligible[
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
            "runtime_seconds_mean",
            "wall_seconds_mean",
        ]
    ]


def _write_reports(outdir: Path, summary: pd.DataFrame, rankings: pd.DataFrame) -> None:
    lines = [
        "# High-dimensional four-method full-converged comparison",
        "",
        "Rule: each method enters a setting-level comparison only if all 5 replicates pass full posterior convergence diagnostics.",
        "GHS+ uses the NumPyro NUTS implementation of the original random-global-scale posterior; the earlier Gibbs/light runs are not used for quality ranking.",
        "",
        "## Eligibility",
        "",
        "| Setting | Method | Converged | Eligible |",
        "|---|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['setting_id']} | {row['method']} | {int(row['n_converged'])}/{int(row['n_runs'])} | "
            f"{'yes' if bool(row['eligible_5of5']) else 'no'} |"
        )
    lines.extend(["", "## Ranking by MSE overall", "", "| Setting | Rank | Method | MSE overall | MSE signal | MSE null | Coverage | Rhat mean | ESS-min mean | Wall sec mean |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for _, row in rankings.iterrows():
        lines.append(
            f"| {row['setting_id']} | {int(row['rank_mse_overall'])} | {row['method']} | "
            f"{row['mse_overall_mean']:.6g} | {row['mse_signal_mean']:.6g} | {row['mse_null_mean']:.6g} | "
            f"{row['coverage_95_mean']:.4f} | {row['rhat_max_mean']:.6g} | {row['ess_min_mean']:.2f} | "
            f"{row['wall_seconds_mean']:.2f} |"
        )
    (outdir / "four_method_full_converged_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    cn = [
        "# 高维四种贝叶斯方法完整后验收敛比较",
        "",
        "准入规则：某个 setting 中，某方法 5 次重复都通过完整后验诊断，才进入该 setting 的质量比较。",
        "GHS+ 使用原始随机全局尺度后验的 NumPyro NUTS 实现；早先 Gibbs/light 结果不用于质量排名。",
        "",
        "## 收敛准入",
        "",
        "| Setting | 方法 | 收敛次数 | 可比较 |",
        "|---|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        cn.append(
            f"| {row['setting_id']} | {row['method']} | {int(row['n_converged'])}/{int(row['n_runs'])} | "
            f"{'是' if bool(row['eligible_5of5']) else '否'} |"
        )
    cn.extend(["", "## 按 MSE overall 排名", "", "| Setting | 排名 | 方法 | MSE overall | MSE signal | MSE null | Coverage | 平均 Rhat | 平均 ESS-min | 平均墙钟秒 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for _, row in rankings.iterrows():
        cn.append(
            f"| {row['setting_id']} | {int(row['rank_mse_overall'])} | {row['method']} | "
            f"{row['mse_overall_mean']:.6g} | {row['mse_signal_mean']:.6g} | {row['mse_null_mean']:.6g} | "
            f"{row['coverage_95_mean']:.4f} | {row['rhat_max_mean']:.6g} | {row['ess_min_mean']:.2f} | "
            f"{row['wall_seconds_mean']:.2f} |"
        )
    cn.extend(
        [
            "",
            "## 说明",
            "",
            "- RHS 在 `hd_setting_2_single_mode` 的第 3、4 次重复已用更高 Stan 预算重跑并完整收敛。",
            "- 所有三个 setting 现在都是四方法 5/5 完整后验收敛后比较。",
            "- `runtime_seconds` 对 JAX/NUTS 不一定包含编译时间；公平看耗时建议同时参考 `wall_seconds_mean`。",
        ]
    )
    (outdir / "four_method_full_converged_report_cn.md").write_text("\n".join(cn) + "\n", encoding="utf-8-sig")


def main() -> int:
    outdir = ROOT / "tmp" / "highdim_bayes_four_method_full_converged_nuts"
    outdir.mkdir(parents=True, exist_ok=True)
    candidates = pd.DataFrame(_load_candidates())
    raw = _select_best(candidates)
    raw = raw[raw["setting_id"].isin(SETTINGS) & raw["method"].isin(METHODS)].copy()
    raw = raw.sort_values(["setting_id", "method", "replicate"])
    summary = _summarize(raw)
    summary = summary.sort_values(["setting_id", "method"])
    rankings = _rank(summary)

    raw.to_csv(outdir / "four_method_raw.csv", index=False)
    summary.to_csv(outdir / "four_method_summary.csv", index=False)
    rankings.to_csv(outdir / "four_method_rankings.csv", index=False)
    _write_reports(outdir, summary, rankings)
    print(f"Wrote {outdir}")
    print(summary[["setting_id", "method", "n_converged", "n_runs", "eligible_5of5"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
