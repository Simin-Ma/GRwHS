from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from simulation_project.src.utils import load_pandas

from .utils import ensure_dir, mechanism_method_label


def _fmt(value: float, ndigits: int = 3) -> str:
    pd = load_pandas()
    if pd.isna(value):
        return "--"
    return f"{float(value):.{ndigits}f}"


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

    compact = build_compact_mechanism_table(summary_paired)
    compact_csv = root / "paper_table_mechanism.csv"
    compact_md = root / "paper_table_mechanism.md"
    compact.to_csv(compact_csv, index=False)
    write_markdown_compact_mechanism(compact, compact_md)

    fig2 = build_figure2_data(summary_paired)
    fig3 = build_figure3_data(summary_paired, paired_deltas)
    fig4 = build_figure4_representative_profile(paired_raw, per_group_kappa)
    fig5 = build_figure5_data(summary_paired)
    fig6 = build_figure6_data(summary_paired)

    fig2_path = figure_dir / "figure2_group_separation.csv"
    fig3_path = figure_dir / "figure3_correlation_ambiguity.csv"
    fig4_path = figure_dir / "figure4_representative_profile.csv"
    fig5_path = figure_dir / "figure5_complexity_unit.csv"
    fig6_path = figure_dir / "figure6_ablation.csv"
    fig2.to_csv(fig2_path, index=False)
    fig3.to_csv(fig3_path, index=False)
    fig4.to_csv(fig4_path, index=False)
    fig5.to_csv(fig5_path, index=False)
    fig6.to_csv(fig6_path, index=False)

    return {
        "paper_table_mechanism_csv": str(compact_csv),
        "paper_table_mechanism_md": str(compact_md),
        "figure2_group_separation": str(fig2_path),
        "figure3_correlation_ambiguity": str(fig3_path),
        "figure4_representative_profile": str(fig4_path),
        "figure5_complexity_unit": str(fig5_path),
        "figure6_ablation": str(fig6_path),
    }


def build_paper_tables_from_results_dir(results_dir: Path | str) -> dict[str, str]:
    pd = load_pandas()
    root = Path(results_dir)
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
