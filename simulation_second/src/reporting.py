from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from simulation_project.src.experiments.reporting import _paired_converged_subset
from simulation_project.src.utils import load_pandas, method_display_name


DEFAULT_SETTING_GROUP_COLS = (
    "setting_id",
    "setting_label",
    "family",
    "group_config",
    "group_sizes",
    "active_groups",
    "n_train",
    "n_test",
    "rho_within",
    "rho_between",
    "target_r2",
    "suite",
    "role",
    "notes",
)

DEFAULT_REQUIRED_METRICS = ("mse_null", "mse_signal", "mse_overall", "lpd_test")
DEFAULT_DELTA_METRICS = ("mse_null", "mse_signal", "mse_overall", "lpd_test")


def default_setting_group_cols(raw=None) -> list[str]:
    cols = list(DEFAULT_SETTING_GROUP_COLS)
    if raw is None:
        return cols
    return [col for col in cols if col in raw.columns]


def _status_is_ok(series) -> Any:
    return series.astype(str).str.strip().str.lower().eq("ok")


def _filter_valid_rows(raw, *, required_metric_cols: Sequence[str]) -> Any:
    if raw.empty or "status" not in raw.columns or "converged" not in raw.columns:
        return raw.iloc[0:0].copy()
    valid = _status_is_ok(raw["status"]) & raw["converged"].fillna(False).astype(bool)
    for col in required_metric_cols:
        if col in raw.columns:
            valid &= raw[col].notna()
    return raw.loc[valid].copy()


def _aggregate_counts(raw, *, group_cols: Sequence[str]):
    pd = load_pandas()
    if raw.empty:
        return pd.DataFrame(columns=list(group_cols) + ["method", "n_runs", "n_ok", "n_converged"])
    return raw.groupby(list(group_cols) + ["method"], as_index=False).agg(
        n_runs=("replicate_id", "count"),
        n_ok=("status", lambda s: int(_status_is_ok(s).sum())),
        n_converged=("converged", lambda s: int(s.fillna(False).astype(bool).sum())),
    )


def _aggregate_metrics(valid, *, group_cols: Sequence[str], count_name: str):
    pd = load_pandas()
    if valid.empty:
        return pd.DataFrame(columns=list(group_cols) + ["method", count_name])

    agg_spec: dict[str, tuple[str, str]] = {
        count_name: ("replicate_id", "nunique"),
        "mse_null": ("mse_null", "mean"),
        "mse_signal": ("mse_signal", "mean"),
        "mse_overall": ("mse_overall", "mean"),
        "coverage_95": ("coverage_95", "mean"),
        "avg_ci_length": ("avg_ci_length", "mean"),
        "lpd_test": ("lpd_test", "mean"),
        "runtime_mean": ("runtime_seconds", "mean"),
        "runtime_max": ("runtime_seconds", "max"),
        "fit_attempts_mean": ("fit_attempts", "mean"),
        "rhat_max_mean": ("rhat_max", "mean"),
        "bulk_ess_min_mean": ("bulk_ess_min", "mean"),
        "divergence_ratio_mean": ("divergence_ratio", "mean"),
        "kappa_signal_mean": ("kappa_signal_mean", "mean"),
        "kappa_null_mean": ("kappa_null_mean", "mean"),
        "kappa_signal_prob_gt_0_5": ("kappa_signal_prob_gt_0_5", "mean"),
        "kappa_null_prob_gt_0_5": ("kappa_null_prob_gt_0_5", "mean"),
        "bridge_ratio_mean": ("bridge_ratio_mean", "mean"),
        "bridge_ratio_min": ("bridge_ratio_min", "mean"),
        "bridge_ratio_max": ("bridge_ratio_max", "mean"),
        "bridge_ratio_p95": ("bridge_ratio_p95", "mean"),
        "bridge_ratio_violations": ("bridge_ratio_violations", "mean"),
        "bridge_ratio_signal_mean": ("bridge_ratio_signal_mean", "mean"),
        "bridge_ratio_null_mean": ("bridge_ratio_null_mean", "mean"),
    }
    present = {
        name: value
        for name, value in agg_spec.items()
        if value[0] in valid.columns
    }
    return valid.groupby(list(group_cols) + ["method"], as_index=False).agg(**present)


def _add_metric_ranks(summary, *, group_cols: Sequence[str], method_order: Sequence[str]) -> Any:
    pd = load_pandas()
    if summary.empty:
        out = summary.copy()
        out["rank_mse_overall"] = pd.Series(dtype="float64")
        out["rank_mse_signal"] = pd.Series(dtype="float64")
        return out

    order_map = {str(method): idx for idx, method in enumerate(method_order)}
    parts = []
    for _, sub in summary.groupby(list(group_cols), dropna=False, sort=False):
        block = sub.copy()
        block["rank_mse_overall"] = np.nan
        block["rank_mse_signal"] = np.nan
        block["_method_order"] = block["method"].map(lambda x: order_map.get(str(x), len(order_map)))

        overall_valid = block["mse_overall"].notna()
        if bool(overall_valid.any()):
            ordered = block.loc[overall_valid].sort_values(
                ["mse_overall", "mse_signal", "runtime_mean", "_method_order", "method"],
                kind="stable",
            )
            block.loc[ordered.index, "rank_mse_overall"] = np.arange(1, len(ordered) + 1, dtype=float)

        signal_valid = block["mse_signal"].notna()
        if bool(signal_valid.any()):
            ordered = block.loc[signal_valid].sort_values(
                ["mse_signal", "mse_overall", "runtime_mean", "_method_order", "method"],
                kind="stable",
            )
            block.loc[ordered.index, "rank_mse_signal"] = np.arange(1, len(ordered) + 1, dtype=float)

        parts.append(block.drop(columns=["_method_order"]))
    return pd.concat(parts, ignore_index=True)


def build_summary(
    raw,
    *,
    group_cols: Sequence[str],
    method_order: Sequence[str],
    required_metric_cols: Sequence[str] = DEFAULT_REQUIRED_METRICS,
):
    counts = _aggregate_counts(raw, group_cols=group_cols)
    valid = _filter_valid_rows(raw, required_metric_cols=required_metric_cols)
    metrics = _aggregate_metrics(valid, group_cols=group_cols, count_name="n_summary_reps")
    summary = counts.merge(metrics, on=list(group_cols) + ["method"], how="left")
    summary["method_label"] = summary["method"].map(method_display_name)
    summary["summary_scope"] = "marginal_valid"
    return _add_metric_ranks(summary, group_cols=group_cols, method_order=method_order)


def build_paired_summary(
    raw,
    *,
    group_cols: Sequence[str],
    method_levels: Sequence[str],
    required_metric_cols: Sequence[str],
    method_order: Sequence[str],
):
    paired_raw, paired_stats = _paired_converged_subset(
        raw,
        group_cols=group_cols,
        method_col="method",
        replicate_col="replicate_id",
        converged_col="converged",
        required_cols=required_metric_cols,
        method_levels=method_levels,
        status_col="status",
        status_ok_values=("ok",),
    )
    counts = _aggregate_counts(raw, group_cols=group_cols)
    metrics = _aggregate_metrics(paired_raw, group_cols=group_cols, count_name="n_paired")
    summary = counts.merge(metrics, on=list(group_cols) + ["method"], how="left")
    if group_cols:
        summary = summary.merge(paired_stats, on=list(group_cols), how="left")
    else:
        for col in paired_stats.columns:
            if col in summary.columns:
                continue
            summary[col] = paired_stats[col].iloc[0] if not paired_stats.empty else np.nan
    summary["method_label"] = summary["method"].map(method_display_name)
    summary["summary_scope"] = "common_converged_paired"
    summary = _add_metric_ranks(summary, group_cols=group_cols, method_order=method_order)
    return paired_raw, paired_stats, summary


def _metric_direction(metric: str) -> str:
    if str(metric).startswith("lpd"):
        return "larger_is_better"
    return "smaller_is_better"


def build_paired_deltas(
    paired_raw,
    *,
    group_cols: Sequence[str],
    baseline_method: str,
    metrics: Sequence[str] = DEFAULT_DELTA_METRICS,
):
    pd = load_pandas()
    rows: list[dict[str, Any]] = []
    if paired_raw.empty:
        return pd.DataFrame(
            columns=list(group_cols)
            + [
                "method",
                "baseline_method",
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
        )

    for setting_vals, sub in paired_raw.groupby(list(group_cols), dropna=False, sort=False):
        wide_setting = sub.pivot_table(index="replicate_id", columns="method", values=list(metrics), aggfunc="mean")
        if baseline_method not in wide_setting.columns.get_level_values(1):
            continue
        vals = setting_vals if isinstance(setting_vals, tuple) else (setting_vals,)
        base = {key: value for key, value in zip(group_cols, vals)}
        for metric in metrics:
            if metric not in wide_setting.columns.get_level_values(0):
                continue
            wide = wide_setting[metric]
            if baseline_method not in wide.columns:
                continue
            base_vec = wide[baseline_method]
            direction = _metric_direction(metric)
            for method in [col for col in wide.columns if str(col) != str(baseline_method)]:
                diff = (wide[method] - base_vec).dropna()
                n_eff = int(diff.shape[0])
                if n_eff == 0:
                    continue
                mean_v = float(diff.mean())
                sd_v = float(diff.std(ddof=1)) if n_eff > 1 else float("nan")
                se_v = float(sd_v / np.sqrt(n_eff)) if n_eff > 1 else float("nan")
                if direction == "smaller_is_better":
                    wins = int((diff < 0.0).sum())
                    losses = int((diff > 0.0).sum())
                else:
                    wins = int((diff > 0.0).sum())
                    losses = int((diff < 0.0).sum())
                rows.append(
                    {
                        **base,
                        "method": str(method),
                        "baseline_method": str(baseline_method),
                        "metric": str(metric),
                        "metric_direction": direction,
                        "n_effective_pairs": n_eff,
                        "mean_diff": mean_v,
                        "std_diff": sd_v,
                        "se_diff": se_v,
                        "ci95_lo": float(mean_v - 1.96 * se_v) if np.isfinite(se_v) else float("nan"),
                        "ci95_hi": float(mean_v + 1.96 * se_v) if np.isfinite(se_v) else float("nan"),
                        "wins_vs_baseline": wins,
                        "losses_vs_baseline": losses,
                        "ties_vs_baseline": int(n_eff - wins - losses),
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["method_label"] = out["method"].map(method_display_name)
        out["baseline_method_label"] = out["baseline_method"].map(method_display_name)
    return out


def write_json_manifest(payload: dict[str, Any], path: Path | str) -> Path:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    import json

    path_obj.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path_obj
