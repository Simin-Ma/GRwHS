from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from simulation_project.src.utils import load_pandas, method_display_name

from .reporting import (
    build_group_selection_frequency,
    build_paired_summary,
    build_selection_stability,
    default_dataset_group_cols,
)
from .utils import resolve_history_results_dir


DEFAULT_NDIGITS = 6
CSV_FLOAT_FORMAT = "%.8f"
MAIN_TABLE_PRIORITY = (
    "rmse_test_mean",
    "mae_test_mean",
    "lpd_test_mean",
    "r2_test_mean",
    "runtime_mean",
    "group_selected_count_mean",
    "mean_pairwise_group_jaccard",
)
FULL_TABLE_BASE_COLS = (
    "dataset_id",
    "dataset_label",
    "method",
    "method_label",
    "n_runs",
    "n_ok",
    "n_converged",
    "n_paired",
    "n_total_replicates",
    "n_common_replicates",
    "common_rate",
)
INTEGER_LIKE_COLS = {
    "n_runs",
    "n_ok",
    "n_converged",
    "n_paired",
    "n_total_replicates",
    "n_common_replicates",
}


def _fmt(value: float, ndigits: int = DEFAULT_NDIGITS) -> str:
    pd = load_pandas()
    if pd.isna(value):
        return "--"
    return f"{float(value):.{ndigits}f}"


def _fmt_pm(mean: float, se: float, ndigits: int = DEFAULT_NDIGITS) -> str:
    pd = load_pandas()
    if pd.isna(mean):
        return "--"
    if pd.isna(se):
        return f"{float(mean):.{ndigits}f}"
    return f"{float(mean):.{ndigits}f} +/- {float(se):.{ndigits}f}"


def _se(series) -> float:
    pd = load_pandas()
    arr = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size <= 1:
        return float("nan")
    return float(arr.std(ddof=1) / np.sqrt(arr.size))


def _latex_escape(value: Any) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def _metric_highlight_rule(metric: str) -> tuple[str | None, float | None]:
    name = str(metric)
    if name in {"n_ok", "n_converged", "n_paired", "common_rate", "mean_pairwise_group_jaccard", "modal_top_group_rate"}:
        return "max", None
    if name.startswith("lpd") or name.startswith("r2") or name.startswith("kappa"):
        return "max", None
    if any(token in name for token in ("rmse", "mae", "runtime", "attempt", "rhat", "divergence")):
        return "min", None
    return None, None


def _ordered_full_metric_cols(df, *, excluded: set[str]) -> list[str]:
    pd = load_pandas()
    out: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            out.append(str(col))
    priority = [col for col in MAIN_TABLE_PRIORITY if col in out]
    extras = sorted(col for col in out if col not in set(priority))
    return priority + extras


def _best_metric_index_map(df, *, group_col: str, metric_cols: Sequence[str]) -> dict[tuple[Any, str], set[int]]:
    pd = load_pandas()
    best: dict[tuple[Any, str], set[int]] = {}
    if df.empty or group_col not in df.columns:
        return best
    for group_value, sub in df.groupby(group_col, dropna=False, sort=False):
        for metric in metric_cols:
            if metric not in sub.columns:
                continue
            mode, target = _metric_highlight_rule(metric)
            if mode is None:
                continue
            vals = pd.to_numeric(sub[metric], errors="coerce")
            valid = vals.notna()
            if not bool(valid.any()):
                continue
            compare = vals.loc[valid]
            if mode == "max":
                best_value = float(compare.max())
                mask = np.isclose(compare.to_numpy(dtype=float), best_value, rtol=1e-10, atol=1e-12)
            elif mode == "min":
                best_value = float(compare.min())
                mask = np.isclose(compare.to_numpy(dtype=float), best_value, rtol=1e-10, atol=1e-12)
            else:
                distances = np.abs(compare.to_numpy(dtype=float) - float(target))
                best_distance = float(np.min(distances))
                mask = np.isclose(distances, best_distance, rtol=1e-10, atol=1e-12)
            best[(group_value, str(metric))] = set(compare.index[np.asarray(mask, dtype=bool)].tolist())
    return best


def _format_full_cell(value: Any, *, col: str, bold: bool, latex: bool) -> str:
    pd = load_pandas()
    if pd.isna(value):
        text = "--"
    elif str(col) in INTEGER_LIKE_COLS:
        text = str(int(round(float(value))))
    elif isinstance(value, (int, np.integer)):
        text = str(int(value))
    elif isinstance(value, (float, np.floating)):
        text = f"{float(value):.{DEFAULT_NDIGITS}f}"
    else:
        text = str(value)
    if latex:
        text = _latex_escape(text)
        return rf"\textbf{{{text}}}" if bold and text != "--" else text
    return f"**{text}**" if bold and text != "--" else text


def build_method_table(paired_raw, *, group_cols: Sequence[str], method_order: Sequence[str]):
    pd = load_pandas()
    if paired_raw.empty:
        return pd.DataFrame()

    agg = paired_raw.groupby(list(group_cols) + ["method"], as_index=False).agg(
        n_paired=("replicate_id", "nunique"),
        rmse_test_mean=("rmse_test", "mean"),
        mae_test_mean=("mae_test", "mean"),
        lpd_test_mean=("lpd_test", "mean"),
        r2_test_mean=("r2_test", "mean"),
        runtime_mean=("runtime_seconds", "mean"),
        group_selected_count_mean=("group_selected_count", "mean"),
    )
    se_df = paired_raw.groupby(list(group_cols) + ["method"], as_index=False).agg(
        rmse_test_se=("rmse_test", _se),
        mae_test_se=("mae_test", _se),
        lpd_test_se=("lpd_test", _se),
        r2_test_se=("r2_test", _se),
        runtime_se=("runtime_seconds", _se),
        group_selected_count_se=("group_selected_count", _se),
    )
    out = agg.merge(se_df, on=list(group_cols) + ["method"], how="left")
    out["method_label"] = out["method"].map(method_display_name)
    order_map = {str(method): idx for idx, method in enumerate(method_order)}
    out["method_order"] = out["method"].map(lambda x: order_map.get(str(x), len(order_map)))
    out = out.sort_values(list(group_cols) + ["method_order", "method"], kind="stable").drop(columns=["method_order"])
    return out.reset_index(drop=True)


def build_main_table(
    method_df,
    stability_df,
    *,
    group_cols: Sequence[str],
    method_order: Sequence[str],
):
    pd = load_pandas()
    if method_df.empty:
        return pd.DataFrame()
    stability_use = stability_df.copy()
    if "method_label" in stability_use.columns and "method_label" in method_df.columns:
        stability_use = stability_use.drop(columns=["method_label"])
    merge_cols = [col for col in list(group_cols) + ["method"] if col in method_df.columns and col in stability_use.columns]
    out = method_df.merge(stability_use, on=merge_cols, how="left") if merge_cols else method_df.copy()
    order_map = {str(method): idx for idx, method in enumerate(method_order)}
    out["method_order"] = out["method"].map(lambda x: order_map.get(str(x), len(order_map)))
    sort_cols = [col for col in ["dataset_id", "rmse_test_mean", "method_order", "method"] if col in out.columns]
    if sort_cols:
        ascending = [True if col != "method_order" else True for col in sort_cols]
        out = out.sort_values(sort_cols, ascending=ascending, kind="stable")
    return out.drop(columns=["method_order"], errors="ignore").reset_index(drop=True)


def build_full_appendix_table(summary_paired, stability_df, *, method_order: Sequence[str]):
    pd = load_pandas()
    if summary_paired.empty:
        return pd.DataFrame()
    stability_use = stability_df.copy()
    if "method_label" in stability_use.columns and "method_label" in summary_paired.columns:
        stability_use = stability_use.drop(columns=["method_label"])
    merge_cols = [col for col in ["dataset_id", "dataset_label", "target_label", "covariate_mode", "n_train", "n_test", "feature_count", "group_count", "notes", "method"] if col in summary_paired.columns and col in stability_use.columns]
    out = summary_paired.merge(stability_use, on=merge_cols, how="left") if merge_cols else summary_paired.copy()
    base_cols = [col for col in FULL_TABLE_BASE_COLS if col in out.columns]
    excluded = set(base_cols) | {"summary_scope", "rank_rmse_test", "rank_lpd_test", "rank_r2_test", "methods_required", "methods_list"}
    metric_cols = _ordered_full_metric_cols(out, excluded=excluded)
    out = out.loc[:, base_cols + metric_cols].copy()
    order_map = {str(method): idx for idx, method in enumerate(method_order)}
    out["_method_order"] = out["method"].map(lambda x: order_map.get(str(x), len(order_map))) if "method" in out.columns else 0
    sort_cols = [col for col in ["dataset_id", "_method_order", "method"] if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="stable")
    return out.drop(columns=["_method_order"], errors="ignore").reset_index(drop=True)


def write_markdown_main(df, path: Path | str) -> None:
    path_obj = Path(path)
    lines = [
        "| Dataset | Method | n paired | RMSE test | MAE test | LPD test | R2 test | Runtime | Selected groups | Stability Jaccard | Modal top group |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in df.iterrows():
        lines.append(
            "| " + " | ".join(
                [
                    str(row["dataset_label"]),
                    str(row["method_label"]),
                    str(int(row["n_paired"])),
                    _fmt_pm(row["rmse_test_mean"], row["rmse_test_se"]),
                    _fmt_pm(row["mae_test_mean"], row["mae_test_se"]),
                    _fmt_pm(row["lpd_test_mean"], row["lpd_test_se"]),
                    _fmt_pm(row["r2_test_mean"], row["r2_test_se"]),
                    _fmt_pm(row["runtime_mean"], row["runtime_se"]),
                    _fmt_pm(row["group_selected_count_mean"], row["group_selected_count_se"]),
                    _fmt(row.get("mean_pairwise_group_jaccard", float("nan"))),
                    str(row.get("modal_top_group_label", "")),
                ]
            ) + " |"
        )
    path_obj.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_full_table(df, path: Path | str, *, group_col: str = "dataset_id") -> None:
    pd = load_pandas()
    path_obj = Path(path)
    if df.empty:
        path_obj.write_text("_No rows._\n", encoding="utf-8")
        return
    cols = list(df.columns)
    metric_cols = [col for col in cols if pd.api.types.is_numeric_dtype(df[col])]
    best_map = _best_metric_index_map(df, group_col=group_col, metric_cols=metric_cols)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for idx, row in df.iterrows():
        group_value = row[group_col] if group_col in row.index else None
        cells = []
        for col in cols:
            bold = idx in best_map.get((group_value, str(col)), set())
            cells.append(_format_full_cell(row[col], col=str(col), bold=bold, latex=False))
        lines.append("| " + " | ".join(cells) + " |")
    path_obj.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_main(df, path: Path | str) -> None:
    path_obj = Path(path)
    lines = [
        r"\begin{tabular}{llcccccccl}",
        r"\toprule",
        r"Dataset & Method & $n$ & RMSE & MAE & LPD & $R^2$ & Runtime & Sel. groups & Stability \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{_latex_escape(row['dataset_label'])} & {_latex_escape(row['method_label'])} & {int(row['n_paired'])} & "
            f"{_fmt(row['rmse_test_mean'])} $\\pm$ {_fmt(row['rmse_test_se'])} & "
            f"{_fmt(row['mae_test_mean'])} $\\pm$ {_fmt(row['mae_test_se'])} & "
            f"{_fmt(row['lpd_test_mean'])} $\\pm$ {_fmt(row['lpd_test_se'])} & "
            f"{_fmt(row['r2_test_mean'])} $\\pm$ {_fmt(row['r2_test_se'])} & "
            f"{_fmt(row['runtime_mean'])} $\\pm$ {_fmt(row['runtime_se'])} & "
            f"{_fmt(row['group_selected_count_mean'])} $\\pm$ {_fmt(row['group_selected_count_se'])} & "
            f"{_fmt(row.get('mean_pairwise_group_jaccard', float('nan')))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path_obj.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_full_table(df, path: Path | str, *, group_col: str = "dataset_id") -> None:
    pd = load_pandas()
    path_obj = Path(path)
    if df.empty:
        path_obj.write_text("% No rows.\n", encoding="utf-8")
        return
    cols = list(df.columns)
    metric_cols = [col for col in cols if pd.api.types.is_numeric_dtype(df[col])]
    best_map = _best_metric_index_map(df, group_col=group_col, metric_cols=metric_cols)
    align = "".join("r" if pd.api.types.is_numeric_dtype(df[col]) else "l" for col in cols)
    header = " & ".join(_latex_escape(col) for col in cols) + r" \\"
    lines = [
        rf"\begin{{longtable}}{{{align}}}",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
    ]
    for idx, row in df.iterrows():
        group_value = row[group_col] if group_col in row.index else None
        cells = []
        for col in cols:
            bold = idx in best_map.get((group_value, str(col)), set())
            cells.append(_format_full_cell(row[col], col=str(col), bold=bold, latex=True))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    path_obj.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_paper_tables(
    raw,
    *,
    out_dir: Path | str,
    method_order: Sequence[str],
    group_cols: Sequence[str] | None = None,
    required_metric_cols: Sequence[str] = ("rmse_test", "mae_test", "lpd_test", "r2_test"),
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    use_group_cols = list(group_cols or default_dataset_group_cols(raw))
    paired_raw, _, summary_paired = build_paired_summary(
        raw,
        group_cols=use_group_cols,
        method_levels=method_order,
        required_metric_cols=required_metric_cols,
        method_order=method_order,
    )
    stability_df = build_selection_stability(
        paired_raw,
        group_cols=use_group_cols,
        required_metric_cols=required_metric_cols,
    )
    group_freq_df = build_group_selection_frequency(
        paired_raw,
        group_cols=use_group_cols,
        required_metric_cols=required_metric_cols,
    )
    method_df = build_method_table(paired_raw, group_cols=use_group_cols, method_order=method_order)
    main_df = build_main_table(method_df, stability_df, group_cols=use_group_cols, method_order=method_order)
    appendix_df = build_full_appendix_table(summary_paired, stability_df, method_order=method_order)

    method_csv = out_path / "paper_table_method_means_se.csv"
    stability_csv = out_path / "paper_table_selection_stability.csv"
    group_freq_csv = out_path / "paper_table_group_selection_frequency.csv"
    main_csv = out_path / "paper_table_main.csv"
    appendix_csv = out_path / "paper_table_appendix_full.csv"
    main_md = out_path / "paper_table_main.md"
    appendix_md = out_path / "paper_table_appendix_full.md"
    main_tex = out_path / "paper_table_main.tex"
    appendix_tex = out_path / "paper_table_appendix_full.tex"

    method_df.to_csv(method_csv, index=False, float_format=CSV_FLOAT_FORMAT)
    stability_df.to_csv(stability_csv, index=False, float_format=CSV_FLOAT_FORMAT)
    group_freq_df.to_csv(group_freq_csv, index=False, float_format=CSV_FLOAT_FORMAT)
    main_df.to_csv(main_csv, index=False, float_format=CSV_FLOAT_FORMAT)
    appendix_df.to_csv(appendix_csv, index=False, float_format=CSV_FLOAT_FORMAT)
    write_markdown_main(main_df, main_md)
    write_markdown_full_table(appendix_df, appendix_md, group_col="dataset_id")
    write_latex_main(main_df, main_tex)
    write_latex_full_table(appendix_df, appendix_tex, group_col="dataset_id")

    return {
        "paper_table_method_means_se": str(method_csv),
        "paper_table_selection_stability": str(stability_csv),
        "paper_table_group_selection_frequency": str(group_freq_csv),
        "paper_table_main_csv": str(main_csv),
        "paper_table_appendix_csv": str(appendix_csv),
        "paper_table_main_md": str(main_md),
        "paper_table_appendix_md": str(appendix_md),
        "paper_table_main_tex": str(main_tex),
        "paper_table_appendix_tex": str(appendix_tex),
    }


def build_paper_tables_from_results_dir(
    results_dir: Path | str,
    *,
    method_order: Sequence[str],
    group_cols: Sequence[str] | None = None,
    required_metric_cols: Sequence[str] = ("rmse_test", "mae_test", "lpd_test", "r2_test"),
):
    pd = load_pandas()
    base = resolve_history_results_dir(results_dir, required_files=("raw_results.csv",))
    raw = pd.read_csv(base / "raw_results.csv")
    return build_paper_tables(
        raw,
        out_dir=base / "paper_tables",
        method_order=method_order,
        group_cols=group_cols,
        required_metric_cols=required_metric_cols,
    )
