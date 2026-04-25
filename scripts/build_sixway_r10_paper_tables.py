from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = ROOT / "outputs" / "grrhs_sixway_stable_r10" / "results" / "exp3_grrhs_sixway_stable_r10"
OUT_DIR = ROOT / "outputs" / "grrhs_sixway_stable_r10" / "paper_tables"
STABLE_SRC = ROOT / "outputs" / "grrhs_sixway_region_scan_combined" / "stable_sixway_dominance_points.csv"

METHODS = ["GR_RHS", "RHS", "GHS_plus", "LASSO_CV", "GIGG_MMLE", "OLS"]
LABELS = {
    "GR_RHS": "GR-RHS",
    "RHS": "RHS",
    "GHS_plus": "GHS+",
    "LASSO_CV": "Lasso-CV",
    "GIGG_MMLE": "GIGG-MMLE",
    "OLS": "OLS",
}
KEYS = [
    "gigg_mode",
    "group_config",
    "signal",
    "setting_block",
    "env_id",
    "design_type",
    "rho_within",
    "rho_between",
    "target_snr",
    "n_train",
    "boundary_xi_ratio",
]


def _fmt(x: float, nd: int = 3) -> str:
    if pd.isna(x):
        return "--"
    return f"{float(x):.{nd}f}"


def _fmt_pm(mean: float, se: float, nd: int = 3) -> str:
    if pd.isna(mean):
        return "--"
    if pd.isna(se):
        return f"{float(mean):.{nd}f}"
    return f"{float(mean):.{nd}f} ± {float(se):.{nd}f}"


def _setting_label(row: pd.Series) -> str:
    return (
        f"rw={float(row['rho_within']):.2f}, "
        f"rb={float(row['rho_between']):.2f}, "
        f"snr={float(row['target_snr']):.1f}, "
        f"n={int(row['n_train'])}"
    )


def _paired_subset(raw: pd.DataFrame) -> pd.DataFrame:
    req_ok = raw["status"].astype(str).eq("ok") & raw["converged"].fillna(False).astype(bool)
    req_ok &= raw["mse_overall"].notna() & raw["mse_signal"].notna() & raw["mse_null"].notna()
    work = raw.loc[req_ok].copy()
    keep_reps = []
    for setting_vals, sub in work.groupby(KEYS, dropna=False):
        counts = sub.groupby("replicate_id")["method"].nunique()
        valid_rep_ids = counts.loc[counts.eq(len(METHODS))].index.tolist()
        if not valid_rep_ids:
            continue
        sel = sub["replicate_id"].isin(valid_rep_ids)
        keep_reps.append(sub.loc[sel])
    if not keep_reps:
        return work.iloc[0:0].copy()
    return pd.concat(keep_reps, ignore_index=True)


def _se(series: pd.Series) -> float:
    arr = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size <= 1:
        return float("nan")
    return float(arr.std(ddof=1) / np.sqrt(arr.size))


def build_method_table(paired: pd.DataFrame) -> pd.DataFrame:
    agg = (
        paired.groupby(KEYS + ["method"], as_index=False)
        .agg(
            n_paired=("replicate_id", "nunique"),
            mse_overall_mean=("mse_overall", "mean"),
            mse_signal_mean=("mse_signal", "mean"),
            mse_null_mean=("mse_null", "mean"),
            coverage_mean=("coverage_95", "mean"),
            runtime_mean=("runtime_seconds", "mean"),
        )
    )
    se = (
        paired.groupby(KEYS + ["method"], as_index=False)
        .agg(
            mse_overall_se=("mse_overall", _se),
            mse_signal_se=("mse_signal", _se),
            mse_null_se=("mse_null", _se),
            coverage_se=("coverage_95", _se),
            runtime_se=("runtime_seconds", _se),
        )
    )
    out = agg.merge(se, on=KEYS + ["method"], how="left")
    out["method_order"] = out["method"].map({m: i for i, m in enumerate(METHODS)})
    out = out.sort_values(KEYS + ["method_order"]).drop(columns=["method_order"]).reset_index(drop=True)
    return out


def build_winloss_table(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for setting_vals, sub in paired.groupby(KEYS, dropna=False):
        wide_overall = sub.pivot(index="replicate_id", columns="method", values="mse_overall")
        wide_signal = sub.pivot(index="replicate_id", columns="method", values="mse_signal")
        gr_o = wide_overall["GR_RHS"]
        gr_s = wide_signal["GR_RHS"]
        base = {k: v for k, v in zip(KEYS, setting_vals)}
        base["n_paired"] = int(wide_overall.shape[0])
        for method in METHODS:
            if method == "GR_RHS":
                continue
            diff_o = wide_overall[method] - gr_o
            diff_s = wide_signal[method] - gr_s
            rows.append(
                {
                    **base,
                    "method": method,
                    "gr_wins_overall": int((diff_o > 0).sum()),
                    "method_wins_overall": int((diff_o < 0).sum()),
                    "ties_overall": int((np.isclose(diff_o, 0.0)).sum()),
                    "gr_wins_signal": int((diff_s > 0).sum()),
                    "method_wins_signal": int((diff_s < 0).sum()),
                    "ties_signal": int((np.isclose(diff_s, 0.0)).sum()),
                }
            )
    return pd.DataFrame(rows)


def build_main_table(method_df: pd.DataFrame, winloss_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for setting_vals, sub in method_df.groupby(KEYS, dropna=False):
        sub = sub.sort_values(["mse_overall_mean", "mse_signal_mean", "runtime_mean"])
        gr = sub.loc[sub["method"] == "GR_RHS"].iloc[0]
        runner = sub.loc[sub["method"] != "GR_RHS"].sort_values(["mse_overall_mean", "mse_signal_mean"]).iloc[0]
        w = winloss_df.loc[
            (winloss_df["env_id"] == gr["env_id"]) &
            (winloss_df["method"] == runner["method"])
        ].iloc[0]
        row = {k: v for k, v in zip(KEYS, setting_vals)}
        row.update(
            {
                "setting": _setting_label(gr),
                "n_paired": int(gr["n_paired"]),
                "gr_mse_overall_mean": float(gr["mse_overall_mean"]),
                "gr_mse_overall_se": float(gr["mse_overall_se"]),
                "gr_mse_signal_mean": float(gr["mse_signal_mean"]),
                "gr_mse_signal_se": float(gr["mse_signal_se"]),
                "gr_coverage_mean": float(gr["coverage_mean"]),
                "gr_runtime_mean": float(gr["runtime_mean"]),
                "runner_up": LABELS[str(runner["method"])],
                "runner_mse_overall_mean": float(runner["mse_overall_mean"]),
                "runner_mse_overall_se": float(runner["mse_overall_se"]),
                "runner_mse_signal_mean": float(runner["mse_signal_mean"]),
                "runner_mse_signal_se": float(runner["mse_signal_se"]),
                "gigg_mse_overall_mean": float(sub.loc[sub["method"] == "GIGG_MMLE", "mse_overall_mean"].iloc[0]),
                "gigg_mse_overall_se": float(sub.loc[sub["method"] == "GIGG_MMLE", "mse_overall_se"].iloc[0]),
                "gigg_coverage_mean": float(sub.loc[sub["method"] == "GIGG_MMLE", "coverage_mean"].iloc[0]),
                "paired_overall_gr_vs_runner": f"{int(w['gr_wins_overall'])}-{int(w['method_wins_overall'])}-{int(w['ties_overall'])}",
                "paired_signal_gr_vs_runner": f"{int(w['gr_wins_signal'])}-{int(w['method_wins_signal'])}-{int(w['ties_signal'])}",
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    out = out.sort_values(["gr_mse_overall_mean", "gr_mse_signal_mean"]).reset_index(drop=True)
    return out


def build_appendix_table(method_df: pd.DataFrame, winloss_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in method_df.iterrows():
        item = row.to_dict()
        item["setting"] = _setting_label(row)
        item["method_label"] = LABELS[str(row["method"])]
        item["bayes_converged"] = "yes" if str(row["method"]) in {"GR_RHS", "RHS", "GHS_plus", "GIGG_MMLE"} else "n/a"
        if str(row["method"]) != "GR_RHS":
            w = winloss_df.loc[
                (winloss_df["env_id"] == row["env_id"]) &
                (winloss_df["method"] == row["method"])
            ].iloc[0]
            item["paired_overall_vs_gr"] = f"{int(w['gr_wins_overall'])}-{int(w['method_wins_overall'])}-{int(w['ties_overall'])}"
            item["paired_signal_vs_gr"] = f"{int(w['gr_wins_signal'])}-{int(w['method_wins_signal'])}-{int(w['ties_signal'])}"
        else:
            item["paired_overall_vs_gr"] = "--"
            item["paired_signal_vs_gr"] = "--"
        rows.append(item)
    out = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(METHODS)}
    out["method_order"] = out["method"].map(order)
    out = out.sort_values(["rho_within", "rho_between", "n_paired", "n_train", "method_order"]).drop(columns=["method_order"]).reset_index(drop=True)
    return out


def write_md_main(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "| Setting | n paired | GR-RHS Overall | GR-RHS Signal | GR Cov. | Runner-up | Runner Overall | Runner Signal | Paired Overall (GR-Other-Tie) | Paired Signal (GR-Other-Tie) | GIGG Overall | GIGG Cov. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            "| " + " | ".join(
                [
                    str(r["setting"]),
                    str(int(r["n_paired"])),
                    _fmt_pm(r["gr_mse_overall_mean"], r["gr_mse_overall_se"]),
                    _fmt_pm(r["gr_mse_signal_mean"], r["gr_mse_signal_se"]),
                    _fmt(r["gr_coverage_mean"], 2),
                    str(r["runner_up"]),
                    _fmt_pm(r["runner_mse_overall_mean"], r["runner_mse_overall_se"]),
                    _fmt_pm(r["runner_mse_signal_mean"], r["runner_mse_signal_se"]),
                    str(r["paired_overall_gr_vs_runner"]),
                    str(r["paired_signal_gr_vs_runner"]),
                    _fmt_pm(r["gigg_mse_overall_mean"], r["gigg_mse_overall_se"]),
                    _fmt(r["gigg_coverage_mean"], 2),
                ]
            ) + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_md_appendix(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "| Setting | Method | n paired | Overall | Signal | Null | Coverage | Runtime | Paired Overall vs GR | Paired Signal vs GR | Bayesian Converged |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            "| " + " | ".join(
                [
                    str(r["setting"]),
                    str(r["method_label"]),
                    str(int(r["n_paired"])),
                    _fmt_pm(r["mse_overall_mean"], r["mse_overall_se"]),
                    _fmt_pm(r["mse_signal_mean"], r["mse_signal_se"]),
                    _fmt_pm(r["mse_null_mean"], r["mse_null_se"]),
                    _fmt_pm(r["coverage_mean"], r["coverage_se"], 2),
                    _fmt_pm(r["runtime_mean"], r["runtime_se"], 2),
                    str(r["paired_overall_vs_gr"]),
                    str(r["paired_signal_vs_gr"]),
                    str(r["bayes_converged"]),
                ]
            ) + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex_main(df: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{lccccccccccc}",
        r"\toprule",
        r"Setting & $n$ & GR Overall & GR Signal & GR Cov. & Runner-up & Runner Overall & Runner Signal & Pair O & Pair S & GIGG Overall & GIGG Cov. \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['setting']} & {int(r['n_paired'])} & "
            f"{_fmt(r['gr_mse_overall_mean'])} $\\pm$ {_fmt(r['gr_mse_overall_se'])} & "
            f"{_fmt(r['gr_mse_signal_mean'])} $\\pm$ {_fmt(r['gr_mse_signal_se'])} & "
            f"{_fmt(r['gr_coverage_mean'], 2)} & {r['runner_up']} & "
            f"{_fmt(r['runner_mse_overall_mean'])} $\\pm$ {_fmt(r['runner_mse_overall_se'])} & "
            f"{_fmt(r['runner_mse_signal_mean'])} $\\pm$ {_fmt(r['runner_mse_signal_se'])} & "
            f"{r['paired_overall_gr_vs_runner']} & {r['paired_signal_gr_vs_runner']} & "
            f"{_fmt(r['gigg_mse_overall_mean'])} $\\pm$ {_fmt(r['gigg_mse_overall_se'])} & {_fmt(r['gigg_coverage_mean'], 2)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex_appendix(df: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{longtable}{llrccccccc}",
        r"\toprule",
        r"Setting & Method & $n$ & Overall & Signal & Null & Cov. & Time & Pair O & Pair S & Conv. \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Setting & Method & $n$ & Overall & Signal & Null & Cov. & Time & Pair O & Pair S & Conv. \\",
        r"\midrule",
        r"\endhead",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['setting']} & {r['method_label']} & {int(r['n_paired'])} & "
            f"{_fmt(r['mse_overall_mean'])} $\\pm$ {_fmt(r['mse_overall_se'])} & "
            f"{_fmt(r['mse_signal_mean'])} $\\pm$ {_fmt(r['mse_signal_se'])} & "
            f"{_fmt(r['mse_null_mean'])} $\\pm$ {_fmt(r['mse_null_se'])} & "
            f"{_fmt(r['coverage_mean'], 2)} & {_fmt(r['runtime_mean'], 2)} & "
            f"{r['paired_overall_vs_gr']} & {r['paired_signal_vs_gr']} & {r['bayes_converged']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(BASE / "raw_results.csv")
    summary_paired = pd.read_csv(BASE / "summary_paired.csv")
    stable_src = pd.read_csv(STABLE_SRC)[["env_id", "n_train"]].drop_duplicates()
    raw = raw.merge(stable_src, on="env_id", how="left")
    summary_paired = summary_paired.merge(stable_src, on="env_id", how="left")
    paired = _paired_subset(raw)

    method_df = build_method_table(paired)
    winloss_df = build_winloss_table(paired)
    main_df = build_main_table(method_df, winloss_df)
    appendix_df = build_appendix_table(method_df, winloss_df)

    summary_paired.to_csv(OUT_DIR / "summary_paired_copy.csv", index=False)
    method_df.to_csv(OUT_DIR / "paper_table_method_means_se.csv", index=False)
    winloss_df.to_csv(OUT_DIR / "paper_table_paired_winloss.csv", index=False)
    main_df.to_csv(OUT_DIR / "paper_table_main.csv", index=False)
    appendix_df.to_csv(OUT_DIR / "paper_table_appendix_full.csv", index=False)

    write_md_main(main_df, OUT_DIR / "paper_table_main.md")
    write_md_appendix(appendix_df, OUT_DIR / "paper_table_appendix_full.md")
    write_tex_main(main_df, OUT_DIR / "paper_table_main.tex")
    write_tex_appendix(appendix_df, OUT_DIR / "paper_table_appendix_full.tex")

    print(OUT_DIR / "paper_table_main.md")
    print(OUT_DIR / "paper_table_appendix_full.md")


if __name__ == "__main__":
    main()
