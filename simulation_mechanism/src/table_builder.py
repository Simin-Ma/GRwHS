from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from simulation_project.src.utils import load_pandas

from .utils import ensure_dir, mechanism_method_label, resolve_history_results_dir


DEFAULT_NDIGITS = 6
CSV_FLOAT_FORMAT = "%.8f"
FULL_TABLE_BASE_COLS = (
    "experiment_id",
    "experiment_label",
    "setting_id",
    "setting_label",
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
FULL_TABLE_PRIORITY = (
    "group_auroc",
    "kappa_gap",
    "kappa_signal_mean",
    "kappa_null_mean",
    "kappa_signal_prob_gt_0_5",
    "kappa_null_prob_gt_0_5",
    "kappa_decoy_mean",
    "kappa_decoy_prob_gt_0_5",
    "null_group_mse",
    "signal_group_mse",
    "mse_overall",
    "mse_signal",
    "mse_null",
    "coverage_95",
    "avg_ci_length",
    "lpd_test",
    "runtime_mean",
    "runtime_max",
    "fit_attempts_mean",
    "rhat_max_mean",
    "bulk_ess_min_mean",
    "divergence_ratio_mean",
    "tau_post_mean",
    "tau_ratio_to_oracle",
    "bridge_ratio_mean",
    "bridge_ratio_min",
    "bridge_ratio_max",
    "bridge_ratio_p95",
    "bridge_ratio_violations",
    "bridge_ratio_signal_mean",
    "bridge_ratio_null_mean",
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


def _latex_escape(value: Any) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def _weighted_mean(sub, metric: str) -> float:
    pd = load_pandas()
    vals = pd.to_numeric(sub[metric], errors="coerce")
    if vals.notna().sum() == 0:
        return float("nan")
    if "n_paired" in sub.columns:
        weights = pd.to_numeric(sub["n_paired"], errors="coerce").fillna(0.0)
        if float(weights.sum()) > 0.0:
            mask = vals.notna() & weights.gt(0.0)
            if bool(mask.any()):
                return float(np.average(vals.loc[mask], weights=weights.loc[mask]))
    return float(vals.mean())


def _metric_direction(metric: str) -> str:
    if str(metric) in {"group_auroc", "kappa_gap"} or str(metric).startswith("lpd"):
        return "larger_is_better"
    return "smaller_is_better"


def _takeaway(metric: str, baseline_value: float, comparator_value: float, baseline_label: str, comparator_label: str) -> str:
    if not np.isfinite(baseline_value) or not np.isfinite(comparator_value):
        return "Insufficient common-converged evidence."
    direction = _metric_direction(metric)
    if direction == "larger_is_better":
        if baseline_value > comparator_value:
            return f"{baseline_label} retains the stronger mechanism signal."
        if baseline_value < comparator_value:
            return f"{comparator_label} is larger on the headline metric in this slice."
        return "The headline metric is effectively tied."
    if baseline_value < comparator_value:
        return f"{baseline_label} retains the lower error."
    if baseline_value > comparator_value:
        return f"{comparator_label} retains the lower error."
    return "The headline metric is effectively tied."


def _metric_highlight_rule(metric: str) -> tuple[str | None, float | None]:
    name = str(metric)
    if name == "coverage_95":
        return "target", 0.95
    if name == "tau_ratio_to_oracle":
        return "target", 1.0
    if name in {"n_ok", "n_converged", "n_paired", "common_rate"}:
        return "max", None
    if name.startswith("lpd") or name.endswith("auroc") or name.endswith("_gap") or name.startswith("bulk_ess"):
        return "max", None
    if "kappa" in name:
        if "null" in name or "decoy" in name:
            return "min", None
        return "max", None
    if "bridge_ratio" in name:
        if "signal" in name:
            return "max", None
        return "min", None
    if any(token in name for token in ("mse", "runtime", "attempt", "rhat", "divergence", "ci_length", "violations")):
        return "min", None
    return None, None


def _ordered_full_metric_cols(df, *, excluded: set[str]) -> list[str]:
    pd = load_pandas()
    out: list[str] = []
    for col in FULL_TABLE_PRIORITY:
        if col in df.columns and col not in excluded and pd.api.types.is_numeric_dtype(df[col]):
            out.append(col)
    extras = [
        str(col)
        for col in df.columns
        if col not in excluded and col not in set(out) and pd.api.types.is_numeric_dtype(df[col])
    ]
    return out + sorted(extras)


def _best_metric_index_map(df, *, group_col: str, metric_cols: list[str]) -> dict[tuple[Any, str], set[int]]:
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


def build_compact_mechanism_table(summary_paired):
    pd = load_pandas()
    if summary_paired.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for experiment_id in ["M1", "M2", "M3", "M4"]:
        sub = summary_paired.loc[summary_paired["experiment_id"] == experiment_id].copy()
        if sub.empty:
            continue
        if experiment_id == "M2" and "within_group_pattern" in sub.columns:
            main = sub.loc[sub["within_group_pattern"].astype(str).eq("mixed_decoy")]
            if not main.empty:
                sub = main
        primary_metric = str(sub["primary_metric"].iloc[0]) if "primary_metric" in sub.columns else "kappa_gap"
        baseline_method = "GR_RHS"
        comparator_method = "RHS"
        if experiment_id == "M4":
            comparator_candidates = ["GR_RHS_no_kappa", "GR_RHS_shared_kappa", "GR_RHS_no_local_scales", "RHS_oracle"]
            present = [name for name in comparator_candidates if name in set(sub["method"].astype(str))]
            comparator_method = present[0] if present else "RHS_oracle"
        baseline = sub.loc[sub["method"] == baseline_method]
        comparator = sub.loc[sub["method"] == comparator_method]
        if baseline.empty or comparator.empty:
            continue
        baseline_value = _weighted_mean(baseline, primary_metric)
        comparator_value = _weighted_mean(comparator, primary_metric)
        n_effective = int(pd.to_numeric(baseline["n_paired"], errors="coerce").fillna(0).sum())
        rows.append(
            {
                "experiment_id": experiment_id,
                "experiment_label": str(sub["experiment_label"].iloc[0]),
                "scientific_question": str(sub["scientific_question"].iloc[0]),
                "primary_metric": primary_metric,
                "baseline_method": baseline_method,
                "baseline_label": mechanism_method_label(baseline_method),
                "baseline_value": float(baseline_value),
                "comparator_method": comparator_method,
                "comparator_label": mechanism_method_label(comparator_method),
                "comparator_value": float(comparator_value),
                "baseline_minus_comparator": float(baseline_value - comparator_value),
                "n_effective": int(n_effective),
                "takeaway": _takeaway(
                    primary_metric,
                    float(baseline_value),
                    float(comparator_value),
                    mechanism_method_label(baseline_method),
                    mechanism_method_label(comparator_method),
                ),
            }
        )
    return pd.DataFrame(rows)


def build_full_mechanism_table(summary_paired):
    pd = load_pandas()
    if summary_paired.empty:
        return pd.DataFrame()

    base_cols = [col for col in FULL_TABLE_BASE_COLS if col in summary_paired.columns]
    excluded = set(base_cols) | {
        "summary_scope",
        "rank_mse_overall",
        "rank_kappa_gap",
        "methods_required",
        "methods_list",
    }
    metric_cols = _ordered_full_metric_cols(summary_paired, excluded=excluded)
    out = summary_paired.loc[:, base_cols + metric_cols].copy()
    sort_cols = [col for col in ["experiment_id", "setting_id"] if col in out.columns]
    if "rank_mse_overall" in summary_paired.columns:
        out["_rank_mse_overall"] = summary_paired.loc[out.index, "rank_mse_overall"]
        sort_cols.append("_rank_mse_overall")
    if "method_label" in out.columns:
        sort_cols.append("method_label")
    elif "method" in out.columns:
        sort_cols.append("method")
    if sort_cols:
        out = out.sort_values(sort_cols, kind="stable")
    return out.drop(columns=[col for col in ["_rank_mse_overall"] if col in out.columns]).reset_index(drop=True)


def write_markdown_compact_mechanism(df, path: Path | str) -> None:
    path_obj = Path(path)
    lines = [
        "| Experiment | Question | Metric | GR-RHS | Comparator | n effective | Takeaway |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for _, row in df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["experiment_label"]),
                    str(row["scientific_question"]),
                    str(row["primary_metric"]),
                    _fmt(row["baseline_value"]),
                    _fmt(row["comparator_value"]),
                    str(int(row["n_effective"])),
                    str(row["takeaway"]),
                ]
            )
            + " |"
        )
    path_obj.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_full_table(df, path: Path | str, *, group_col: str = "setting_id") -> None:
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


def write_latex_full_table(df, path: Path | str, *, group_col: str = "setting_id") -> None:
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


def build_figure2_data(summary_paired):
    if summary_paired.empty:
        return summary_paired
    cols = [col for col in ["experiment_id", "setting_id", "method", "method_label", "group_auroc", "kappa_gap", "mse_overall", "n_paired"] if col in summary_paired.columns]
    return summary_paired.loc[summary_paired["experiment_id"] == "M1", cols].copy()


def build_figure3_data(summary_paired, paired_deltas):
    pd = load_pandas()
    if summary_paired.empty:
        return pd.DataFrame()
    gr = summary_paired.loc[
        (summary_paired["experiment_id"] == "M2") & (summary_paired["method"] == "GR_RHS"),
        [col for col in ["setting_id", "rho_within", "within_group_pattern", "kappa_gap", "n_paired"] if col in summary_paired.columns],
    ].copy()
    delta = paired_deltas.loc[
        (paired_deltas["experiment_id"] == "M2")
        & (paired_deltas["method"] == "GR_RHS")
        & (paired_deltas["metric"] == "mse_overall"),
        [col for col in ["setting_id", "rho_within", "within_group_pattern", "mean_diff", "n_effective_pairs"] if col in paired_deltas.columns],
    ].copy()
    if delta.empty:
        return gr
    out = gr.merge(delta, on=[col for col in ["setting_id", "rho_within", "within_group_pattern"] if col in gr.columns and col in delta.columns], how="left")
    out = out.rename(columns={"mean_diff": "gr_minus_rhs_mse_overall"})
    return out


def build_figure4_representative_profile(paired_raw, per_group_kappa):
    pd = load_pandas()
    if paired_raw.empty or per_group_kappa.empty:
        return pd.DataFrame()
    sub = paired_raw.loc[
        (paired_raw["experiment_id"] == "M2")
        & (paired_raw["within_group_pattern"].astype(str) == "mixed_decoy")
        & (paired_raw["method"] == "GR_RHS")
    ].copy()
    if sub.empty:
        return pd.DataFrame()
    max_rho = float(pd.to_numeric(sub["rho_within"], errors="coerce").max())
    sub = sub.loc[pd.to_numeric(sub["rho_within"], errors="coerce").eq(max_rho)].copy()
    median_gap = float(pd.to_numeric(sub["kappa_gap"], errors="coerce").median())
    sub["distance_to_median"] = (pd.to_numeric(sub["kappa_gap"], errors="coerce") - median_gap).abs()
    pick = sub.sort_values(["distance_to_median", "replicate_id"], kind="stable").iloc[0]
    mask = (
        per_group_kappa["setting_id"].astype(str).eq(str(pick["setting_id"]))
        & per_group_kappa["replicate_id"].astype(int).eq(int(pick["replicate_id"]))
        & per_group_kappa["method"].astype(str).isin(["GR_RHS", "RHS"])
    )
    if "paired_common_converged" in per_group_kappa.columns:
        mask &= per_group_kappa["paired_common_converged"].fillna(False).astype(bool)
    out = per_group_kappa.loc[mask].copy()
    if not out.empty:
        out = out.sort_values(["method", "group_id"], kind="stable").reset_index(drop=True)
    return out


def build_figure5_data(summary_paired):
    if summary_paired.empty:
        return summary_paired
    cols = [col for col in ["setting_id", "complexity_pattern", "within_group_pattern", "method", "method_label", "kappa_gap", "mse_overall", "n_paired"] if col in summary_paired.columns]
    return summary_paired.loc[summary_paired["experiment_id"] == "M3", cols].copy()


def build_figure6_data(summary_paired):
    if summary_paired.empty:
        return summary_paired
    cols = [col for col in ["setting_id", "total_active_coeff", "method", "method_label", "kappa_gap", "mse_overall", "mse_signal", "tau_ratio_to_oracle", "n_paired"] if col in summary_paired.columns]
    return summary_paired.loc[summary_paired["experiment_id"] == "M4", cols].copy()


def build_figure6_delta_data(paired_deltas):
    if paired_deltas.empty:
        return paired_deltas
    keep_metrics = {"kappa_gap", "mse_overall", "mse_signal"}
    cols = [
        col
        for col in [
            "experiment_id",
            "setting_id",
            "total_active_coeff",
            "method",
            "method_label",
            "baseline_method",
            "baseline_method_label",
            "metric",
            "metric_direction",
            "n_effective_pairs",
            "mean_diff",
            "std_diff",
            "se_diff",
            "ci95_lo",
            "ci95_hi",
            "wins_vs_baseline",
            "losses_vs_baseline",
            "ties_vs_baseline",
        ]
        if col in paired_deltas.columns
    ]
    return paired_deltas.loc[
        (paired_deltas["experiment_id"] == "M4")
        & paired_deltas["metric"].astype(str).isin(keep_metrics),
        cols,
    ].copy()


def build_paper_tables(
    *,
    summary_paired,
    paired_deltas,
    paired_raw,
    per_group_kappa,
    out_dir: Path | str,
) -> dict[str, str]:
    root = ensure_dir(out_dir)
    figure_dir = ensure_dir(root / "figure_data")

    full_df = build_full_mechanism_table(summary_paired)
    compact = build_compact_mechanism_table(summary_paired)

    full_csv = root / "paper_table_mechanism.csv"
    full_md = root / "paper_table_mechanism.md"
    full_tex = root / "paper_table_mechanism.tex"
    compact_csv = root / "paper_table_mechanism_compact.csv"
    compact_md = root / "paper_table_mechanism_compact.md"

    full_df.to_csv(full_csv, index=False, float_format=CSV_FLOAT_FORMAT)
    write_markdown_full_table(full_df, full_md, group_col="setting_id")
    write_latex_full_table(full_df, full_tex, group_col="setting_id")
    compact.to_csv(compact_csv, index=False, float_format=CSV_FLOAT_FORMAT)
    write_markdown_compact_mechanism(compact, compact_md)

    fig2 = build_figure2_data(summary_paired)
    fig3 = build_figure3_data(summary_paired, paired_deltas)
    fig4 = build_figure4_representative_profile(paired_raw, per_group_kappa)
    fig5 = build_figure5_data(summary_paired)
    fig6 = build_figure6_data(summary_paired)
    fig6_delta = build_figure6_delta_data(paired_deltas)

    fig2_path = figure_dir / "figure2_group_separation.csv"
    fig3_path = figure_dir / "figure3_correlation_ambiguity.csv"
    fig4_path = figure_dir / "figure4_representative_profile.csv"
    fig5_path = figure_dir / "figure5_complexity_unit.csv"
    fig6_path = figure_dir / "figure6_ablation.csv"
    fig6_delta_path = figure_dir / "figure6_ablation_deltas.csv"
    fig2.to_csv(fig2_path, index=False, float_format=CSV_FLOAT_FORMAT)
    fig3.to_csv(fig3_path, index=False, float_format=CSV_FLOAT_FORMAT)
    fig4.to_csv(fig4_path, index=False, float_format=CSV_FLOAT_FORMAT)
    fig5.to_csv(fig5_path, index=False, float_format=CSV_FLOAT_FORMAT)
    fig6.to_csv(fig6_path, index=False, float_format=CSV_FLOAT_FORMAT)
    fig6_delta.to_csv(fig6_delta_path, index=False, float_format=CSV_FLOAT_FORMAT)

    return {
        "paper_table_mechanism_csv": str(full_csv),
        "paper_table_mechanism_md": str(full_md),
        "paper_table_mechanism_tex": str(full_tex),
        "paper_table_mechanism_compact_csv": str(compact_csv),
        "paper_table_mechanism_compact_md": str(compact_md),
        "figure2_group_separation": str(fig2_path),
        "figure3_correlation_ambiguity": str(fig3_path),
        "figure4_representative_profile": str(fig4_path),
        "figure5_complexity_unit": str(fig5_path),
        "figure6_ablation": str(fig6_path),
        "figure6_ablation_deltas": str(fig6_delta_path),
    }


def build_paper_tables_from_results_dir(results_dir: Path | str) -> dict[str, str]:
    pd = load_pandas()
    root = resolve_history_results_dir(
        results_dir,
        required_files=("summary_paired.csv", "summary_paired_deltas.csv", "raw_results_paired.csv", "per_group_kappa.csv"),
    )
    summary_paired = pd.read_csv(root / "summary_paired.csv")
    paired_deltas = pd.read_csv(root / "summary_paired_deltas.csv")
    paired_raw = pd.read_csv(root / "raw_results_paired.csv")
    per_group_kappa = pd.read_csv(root / "per_group_kappa.csv")
    return build_paper_tables(
        summary_paired=summary_paired,
        paired_deltas=paired_deltas,
        paired_raw=paired_raw,
        per_group_kappa=per_group_kappa,
        out_dir=root / "paper_tables",
    )
