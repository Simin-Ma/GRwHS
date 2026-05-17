from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METHOD_ORDER = ["GR_RHS", "RHS", "GHS_plus_NUTS", "GIGG_MMLE"]
METRICS = ["mse_overall", "mse_signal", "mse_null"]


def _fmt_mean_se(mean: float, se: float) -> str:
    if pd.isna(mean):
        return ""
    if pd.isna(se):
        return f"{mean:.4g}"
    return f"{mean:.4g} ({se:.2g})"


def _se(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    if len(vals) <= 1:
        return float("nan")
    return float(vals.std(ddof=1) / (len(vals) ** 0.5))


def _method_rank(method: str) -> int:
    return METHOD_ORDER.index(method) if method in METHOD_ORDER else 999


def build_tables(outroot: Path) -> None:
    raw_path = outroot / "bayes_reps_best_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    raw = pd.read_csv(raw_path)
    raw = raw[raw["converged"]].copy()
    raw["method_rank"] = raw["method"].map(_method_rank)

    summary = (
        raw.groupby(["setting_id", "method"], as_index=False)
        .agg(
            n=("replicate", "nunique"),
            rhat_max_mean=("rhat_max", "mean"),
            ess_min_mean=("ess_min", "mean"),
            wall_seconds_mean=("wall_seconds", "mean"),
            mse_overall_mean=("mse_overall", "mean"),
            mse_overall_se=("mse_overall", _se),
            mse_signal_mean=("mse_signal", "mean"),
            mse_signal_se=("mse_signal", _se),
            mse_null_mean=("mse_null", "mean"),
            mse_null_se=("mse_null", _se),
            lpd_test_mean=("lpd_test", "mean"),
        )
        .sort_values(["setting_id", "method"], key=lambda s: s.map(_method_rank) if s.name == "method" else s)
    )
    summary.to_csv(outroot / "paper_table_method_means.csv", index=False)

    rows = []
    for setting, chunk in summary.groupby("setting_id", sort=True):
        row: dict[str, object] = {"setting_id": setting}
        for method in METHOD_ORDER:
            m = chunk[chunk["method"] == method]
            if m.empty:
                continue
            rec = m.iloc[0]
            row[f"{method}_overall"] = _fmt_mean_se(rec["mse_overall_mean"], rec["mse_overall_se"])
            row[f"{method}_signal"] = _fmt_mean_se(rec["mse_signal_mean"], rec["mse_signal_se"])
            row[f"{method}_null"] = _fmt_mean_se(rec["mse_null_mean"], rec["mse_null_se"])
        rows.append(row)
    wide = pd.DataFrame(rows)
    wide.to_csv(outroot / "paper_table_wide_mean_se.csv", index=False)

    paired_rows = []
    for setting, s_chunk in raw.groupby("setting_id", sort=True):
        for metric in METRICS:
            pivot = s_chunk.pivot_table(index="replicate", columns="method", values=metric, aggfunc="first")
            for competitor in [m for m in METHOD_ORDER if m != "GR_RHS"]:
                if "GR_RHS" not in pivot.columns or competitor not in pivot.columns:
                    continue
                paired = pivot[["GR_RHS", competitor]].dropna()
                if paired.empty:
                    continue
                delta = paired["GR_RHS"] - paired[competitor]
                pct = 100.0 * (paired[competitor] - paired["GR_RHS"]) / paired[competitor]
                paired_rows.append(
                    {
                        "setting_id": setting,
                        "metric": metric,
                        "competitor": competitor,
                        "paired_n": int(len(paired)),
                        "grrhs_mean": float(paired["GR_RHS"].mean()),
                        "competitor_mean": float(paired[competitor].mean()),
                        "delta_grrhs_minus_competitor_mean": float(delta.mean()),
                        "delta_se": _se(delta),
                        "percent_reduction_mean": float(pct.mean()),
                        "grrhs_wins": int((paired["GR_RHS"] < paired[competitor]).sum()),
                    }
                )
    paired_summary = pd.DataFrame(paired_rows)
    paired_summary.to_csv(outroot / "paper_table_paired_deltas.csv", index=False)

    leaders = []
    for setting, s_chunk in raw.groupby("setting_id", sort=True):
        for metric in METRICS:
            means = s_chunk.groupby("method")[metric].mean().sort_values()
            leaders.append(
                {
                    "setting_id": setting,
                    "metric": metric,
                    "best_method": means.index[0],
                    "best_mean": float(means.iloc[0]),
                    "second_method": means.index[1] if len(means) > 1 else "",
                    "second_mean": float(means.iloc[1]) if len(means) > 1 else float("nan"),
                }
            )
    leader_df = pd.DataFrame(leaders)
    leader_df.to_csv(outroot / "paper_table_metric_leaders.csv", index=False)

    lines = [
        "# Paper-Ready High-Dimensional Bayesian Tables",
        "",
        "All rows use converged posterior runs only.",
        "",
        "## Metric Leaders By Mean",
        "",
        "```text",
        leader_df.to_string(index=False),
        "```",
        "",
        "## Paired GR-RHS Deltas",
        "",
        "Negative delta means GR-RHS has lower MSE than the competitor. Positive percent reduction means GR-RHS improves over the competitor.",
        "",
        "```text",
        paired_summary.to_string(index=False),
        "```",
    ]
    (outroot / "paper_tables_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paper-ready HD Bayesian comparison tables.")
    parser.add_argument("--outroot", default="tmp/highdim_six_bayes_converged_reps5")
    args = parser.parse_args()
    build_tables(Path(args.outroot))
    print(f"Paper tables saved in: {args.outroot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
