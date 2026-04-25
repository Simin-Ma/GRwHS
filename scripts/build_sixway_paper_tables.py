from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


INPUT = ROOT / "outputs" / "grrhs_sixway_region_scan_combined" / "stable_points_all_methods_full.csv"
OUT_DIR = ROOT / "outputs" / "grrhs_sixway_region_scan_combined" / "paper_tables"

METHOD_ORDER = ["GR_RHS", "RHS", "GHS_plus", "LASSO_CV", "GIGG_MMLE", "OLS"]
METHOD_LABEL = {
    "GR_RHS": "GR-RHS",
    "RHS": "RHS",
    "GHS_plus": "GHS+",
    "LASSO_CV": "Lasso-CV",
    "GIGG_MMLE": "GIGG-MMLE",
    "OLS": "OLS",
}


def _fmt(x: float, nd: int = 3) -> str:
    if pd.isna(x):
        return "--"
    return f"{float(x):.{nd}f}"


def _setting_label(row: pd.Series) -> str:
    return (
        f"rw={float(row['rho_within']):.2f}, "
        f"rb={float(row['rho_between']):.2f}, "
        f"snr={float(row['target_snr']):.1f}, "
        f"n={int(row['n_train'])}"
    )


def build_main_table(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["env_id", "rho_within", "rho_between", "target_snr", "n_train"]
    rows: list[dict[str, object]] = []
    for _, sub in df.groupby(keys, sort=False):
        sub = sub.sort_values(["rank_mse_overall", "rank_mse_signal", "runtime_mean"])
        winner = sub.iloc[0]
        runner_up = sub.iloc[1]
        gigg = sub.loc[sub["method"] == "GIGG_MMLE"].iloc[0]
        rows.append(
            {
                "setting": _setting_label(winner),
                "winner": METHOD_LABEL[str(winner["method"])],
                "winner_mse_overall": float(winner["mse_overall"]),
                "winner_mse_signal": float(winner["mse_signal"]),
                "winner_coverage_95": float(winner["coverage_95"]),
                "winner_runtime_s": float(winner["runtime_mean"]),
                "runner_up": METHOD_LABEL[str(runner_up["method"])],
                "runner_up_mse_overall": float(runner_up["mse_overall"]),
                "runner_up_mse_signal": float(runner_up["mse_signal"]),
                "delta_overall_vs_runner_up": float(runner_up["mse_overall"]) - float(winner["mse_overall"]),
                "delta_signal_vs_runner_up": float(runner_up["mse_signal"]) - float(winner["mse_signal"]),
                "gigg_mse_overall": float(gigg["mse_overall"]),
                "gigg_coverage_95": float(gigg["coverage_95"]),
                "gigg_runtime_s": float(gigg["runtime_mean"]),
            }
        )
    out = pd.DataFrame(rows)
    out = out.sort_values(["winner_mse_overall", "delta_overall_vs_runner_up"], ascending=[True, False]).reset_index(drop=True)
    return out


def build_appendix_table(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["setting"] = work.apply(_setting_label, axis=1)
    work["method_label"] = work["method"].map(METHOD_LABEL)
    work["bayes_conv"] = work.apply(
        lambda r: "yes" if str(r["method"]) in {"GR_RHS", "RHS", "GHS_plus", "GIGG_MMLE"} and int(r["n_converged"]) == int(r["n_ok"]) == 1 else ("n/a" if str(r["method"]) in {"OLS", "LASSO_CV"} else "no"),
        axis=1,
    )
    cols = [
        "setting",
        "method_label",
        "rank_mse_overall",
        "rank_mse_signal",
        "mse_overall",
        "mse_signal",
        "mse_null",
        "coverage_95",
        "runtime_mean",
        "bayes_conv",
    ]
    work = work.loc[:, cols].sort_values(["setting", "rank_mse_overall", "rank_mse_signal", "runtime_mean"]).reset_index(drop=True)
    return work


def write_markdown_main(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "| Setting | Winner | MSE (Overall) | MSE (Signal) | Coverage | Runtime (s) | Runner-up | Delta Overall | Delta Signal | GIGG MSE | GIGG Cov |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["setting"]),
                    str(r["winner"]),
                    _fmt(r["winner_mse_overall"]),
                    _fmt(r["winner_mse_signal"]),
                    _fmt(r["winner_coverage_95"], 2),
                    _fmt(r["winner_runtime_s"], 2),
                    str(r["runner_up"]),
                    _fmt(r["delta_overall_vs_runner_up"]),
                    _fmt(r["delta_signal_vs_runner_up"]),
                    _fmt(r["gigg_mse_overall"]),
                    _fmt(r["gigg_coverage_95"], 2),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_appendix(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "| Setting | Method | Rank Overall | Rank Signal | MSE Overall | MSE Signal | MSE Null | Coverage | Runtime (s) | Bayesian Converged |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["setting"]),
                    str(r["method_label"]),
                    str(int(r["rank_mse_overall"])),
                    str(int(r["rank_mse_signal"])),
                    _fmt(r["mse_overall"]),
                    _fmt(r["mse_signal"]),
                    _fmt(r["mse_null"]),
                    _fmt(r["coverage_95"], 2),
                    _fmt(r["runtime_mean"], 2),
                    str(r["bayes_conv"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_main(df: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{lcccccccccc}",
        r"\toprule",
        r"Setting & Winner & Overall & Signal & Cov. & Time & Runner-up & $\Delta$ Overall & $\Delta$ Signal & GIGG Overall & GIGG Cov. \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['setting']} & {r['winner']} & {_fmt(r['winner_mse_overall'])} & {_fmt(r['winner_mse_signal'])} & "
            f"{_fmt(r['winner_coverage_95'], 2)} & {_fmt(r['winner_runtime_s'], 2)} & {r['runner_up']} & "
            f"{_fmt(r['delta_overall_vs_runner_up'])} & {_fmt(r['delta_signal_vs_runner_up'])} & "
            f"{_fmt(r['gigg_mse_overall'])} & {_fmt(r['gigg_coverage_95'], 2)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_appendix(df: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{longtable}{llrrcccccl}",
        r"\toprule",
        r"Setting & Method & Rank O. & Rank S. & Overall & Signal & Null & Cov. & Time & Conv. \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Setting & Method & Rank O. & Rank S. & Overall & Signal & Null & Cov. & Time & Conv. \\",
        r"\midrule",
        r"\endhead",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['setting']} & {r['method_label']} & {int(r['rank_mse_overall'])} & {int(r['rank_mse_signal'])} & "
            f"{_fmt(r['mse_overall'])} & {_fmt(r['mse_signal'])} & {_fmt(r['mse_null'])} & {_fmt(r['coverage_95'], 2)} & "
            f"{_fmt(r['runtime_mean'], 2)} & {r['bayes_conv']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT)
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    df = df.sort_values(
        ["rho_within", "rho_between", "n_train", "method"],
        ascending=[True, True, True, True],
    )

    main_df = build_main_table(df)
    appendix_df = build_appendix_table(df)

    main_df.to_csv(OUT_DIR / "paper_table_main.csv", index=False)
    appendix_df.to_csv(OUT_DIR / "paper_table_appendix_full.csv", index=False)
    write_markdown_main(main_df, OUT_DIR / "paper_table_main.md")
    write_markdown_appendix(appendix_df, OUT_DIR / "paper_table_appendix_full.md")
    write_latex_main(main_df, OUT_DIR / "paper_table_main.tex")
    write_latex_appendix(appendix_df, OUT_DIR / "paper_table_appendix_full.tex")

    print(OUT_DIR / "paper_table_main.md")
    print(OUT_DIR / "paper_table_appendix_full.md")


if __name__ == "__main__":
    main()
