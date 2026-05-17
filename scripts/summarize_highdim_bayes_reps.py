from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


METHODS = ["GR_RHS", "RHS", "GIGG_MMLE", "GHS_plus_NUTS"]


def _scalar(value: object) -> float | str:
    if value is None:
        return math.nan
    try:
        val = float(value)
    except Exception:
        return str(value)
    return val if math.isfinite(val) else math.nan


def _row(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    method = str(payload.get("method", ""))
    if "GHS_plus_NUTS" in path.name:
        method = "GHS_plus_NUTS"
    if method not in METHODS:
        return None
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
        "source_file": str(path),
    }


def collect(outroot: Path, reps: list[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sub in ["standard", "grrhs_retries", "rhs_retries", "gigg", "ghs_plus_nuts"]:
        folder = outroot / sub
        if not folder.exists():
            continue
        for path in folder.glob("*.json"):
            row = _row(path)
            if row is not None:
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df[df["replicate"].isin(reps)].copy()
    if df.empty:
        return df
    df["_conv"] = df["converged"].astype(int)
    df["_rhat"] = pd.to_numeric(df["rhat_max"], errors="coerce").fillna(float("inf"))
    df["_ess"] = pd.to_numeric(df["ess_min"], errors="coerce").fillna(float("-inf"))
    df = df.sort_values(
        ["setting_id", "method", "replicate", "_conv", "_rhat", "_ess"],
        ascending=[True, True, True, False, True, False],
    )
    return (
        df.groupby(["setting_id", "method", "replicate"], as_index=False, sort=False)
        .head(1)
        .drop(columns=["_conv", "_rhat", "_ess"])
        .sort_values(["setting_id", "replicate", "method"])
    )


def _se(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) <= 1:
        return math.nan
    return float(vals.std(ddof=1) / math.sqrt(len(vals)))


def write_outputs(best: pd.DataFrame, outroot: Path, reps: list[int]) -> None:
    outroot.mkdir(parents=True, exist_ok=True)
    best.to_csv(outroot / "bayes_reps_best_raw.csv", index=False)
    converged = best[best["converged"]].copy()
    summary = (
        best.groupby(["setting_id", "method"], as_index=False)
        .agg(
            n_runs=("replicate", "nunique"),
            n_converged=("converged", "sum"),
            rhat_max_best=("rhat_max", "min"),
            ess_min_best=("ess_min", "max"),
            mean_wall_seconds=("wall_seconds", "mean"),
        )
        .sort_values(["setting_id", "method"])
    )
    if not converged.empty:
        metric_summary = converged.groupby(["setting_id", "method"], as_index=False).agg(
            mse_overall_mean=("mse_overall", "mean"),
            mse_overall_se=("mse_overall", _se),
            mse_signal_mean=("mse_signal", "mean"),
            mse_signal_se=("mse_signal", _se),
            mse_null_mean=("mse_null", "mean"),
            mse_null_se=("mse_null", _se),
            lpd_test_mean=("lpd_test", "mean"),
            coverage_95_mean=("coverage_95", "mean"),
        )
        summary = summary.merge(metric_summary, on=["setting_id", "method"], how="left")
    summary.to_csv(outroot / "bayes_reps_summary.csv", index=False)

    wins = []
    for metric in ["mse_overall", "mse_signal", "mse_null"]:
        for (setting, rep), chunk in converged.groupby(["setting_id", "replicate"]):
            if set(METHODS).issubset(set(chunk["method"])):
                row = chunk.sort_values(metric, kind="stable").iloc[0]
                wins.append(
                    {
                        "setting_id": setting,
                        "replicate": int(rep),
                        "metric": metric,
                        "winner": row["method"],
                        "winning_value": row[metric],
                    }
                )
    wins_df = pd.DataFrame(wins)
    wins_df.to_csv(outroot / "bayes_reps_wins_by_rep.csv", index=False)
    if not wins_df.empty:
        win_summary = (
            wins_df.groupby(["setting_id", "metric", "winner"], as_index=False)
            .agg(wins=("replicate", "nunique"))
            .sort_values(["setting_id", "metric", "wins"], ascending=[True, True, False])
        )
    else:
        win_summary = pd.DataFrame()
    win_summary.to_csv(outroot / "bayes_reps_win_summary.csv", index=False)

    missing = []
    settings = sorted(best["setting_id"].unique())
    for setting in settings:
        for method in METHODS:
            for rep in reps:
                hit = best[
                    (best["setting_id"] == setting)
                    & (best["method"] == method)
                    & (best["replicate"] == rep)
                ]
                if hit.empty:
                    missing.append({"setting_id": setting, "method": method, "replicate": rep, "gap": "missing"})
                elif not bool(hit.iloc[0]["converged"]):
                    missing.append({"setting_id": setting, "method": method, "replicate": rep, "gap": "not_converged"})
    missing_df = pd.DataFrame(missing)
    missing_df.to_csv(outroot / "bayes_reps_missing_or_unconverged.csv", index=False)

    lines = [
        "# High-Dimensional Bayesian Repeats Summary",
        "",
        f"Replicates requested: {', '.join(str(r) for r in reps)}",
        "",
        "Only converged rows are used for MSE means and win counts.",
        "",
        "## Convergence Counts",
        "",
        "```text",
        summary[["setting_id", "method", "n_runs", "n_converged"]].to_string(index=False),
        "```",
        "",
        "## Mean MSE Among Converged Runs",
        "",
        "```text",
        summary[
            [
                "setting_id",
                "method",
                "mse_overall_mean",
                "mse_overall_se",
                "mse_signal_mean",
                "mse_signal_se",
                "mse_null_mean",
                "mse_null_se",
            ]
        ].to_string(index=False),
        "```",
    ]
    if not win_summary.empty:
        lines.extend(["", "## Win Counts", "", "```text", win_summary.to_string(index=False), "```"])
    if not missing_df.empty:
        lines.extend(["", "## Missing Or Unconverged", "", "```text", missing_df.to_string(index=False), "```"])
    (outroot / "bayes_reps_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize HD Bayesian JSON outputs across replicates.")
    parser.add_argument("--outroot", default="tmp/highdim_six_bayes_converged_reps5")
    parser.add_argument("--reps", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    outroot = Path(args.outroot)
    best = collect(outroot, list(args.reps))
    write_outputs(best, outroot, list(args.reps))
    print(f"rows={len(best)} converged={int(best['converged'].sum()) if not best.empty else 0}")
    print(f"Artifacts saved in: {outroot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
