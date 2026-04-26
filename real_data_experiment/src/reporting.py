from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from simulation_project.src.experiments.reporting import _paired_converged_subset
from simulation_project.src.utils import load_pandas, method_display_name


DEFAULT_DATASET_GROUP_COLS = (
    "dataset_id",
    "dataset_label",
    "target_label",
    "covariate_mode",
    "n_train",
    "n_test",
    "feature_count",
    "group_count",
    "notes",
)
DEFAULT_REQUIRED_METRICS = ("rmse_test", "mae_test", "lpd_test", "r2_test")
DEFAULT_DELTA_METRICS = ("rmse_test", "mae_test", "lpd_test", "r2_test", "group_selected_count")


def default_dataset_group_cols(raw=None) -> list[str]:
    cols = list(DEFAULT_DATASET_GROUP_COLS)
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
        "rmse_train": ("rmse_train", "mean"),
        "mae_train": ("mae_train", "mean"),
        "r2_train": ("r2_train", "mean"),
        "rmse_test": ("rmse_test", "mean"),
        "mae_test": ("mae_test", "mean"),
        "r2_test": ("r2_test", "mean"),
        "lpd_test": ("lpd_test", "mean"),
        "runtime_mean": ("runtime_seconds", "mean"),
        "runtime_max": ("runtime_seconds", "max"),
        "fit_attempts_mean": ("fit_attempts", "mean"),
        "rhat_max_mean": ("rhat_max", "mean"),
        "bulk_ess_min_mean": ("bulk_ess_min", "mean"),
        "divergence_ratio_mean": ("divergence_ratio", "mean"),
        "sigma2_hat_train_mean": ("sigma2_hat_train", "mean"),
        "coef_l1_norm_mean": ("coef_l1_norm", "mean"),
        "coef_l2_norm_mean": ("coef_l2_norm", "mean"),
        "coef_nonzero_count_mean": ("coef_nonzero_count", "mean"),
        "coef_rel_1pct_count_mean": ("coef_rel_1pct_count", "mean"),
        "group_selected_count_mean": ("group_selected_count", "mean"),
        "group_selected_fraction_mean": ("group_selected_fraction", "mean"),
        "group_norm_entropy_mean": ("group_norm_entropy", "mean"),
        "top_group_score_mean": ("top_group_score", "mean"),
        "kappa_mean_overall": ("kappa_mean_overall", "mean"),
        "kappa_prob_gt_0_5_overall": ("kappa_prob_gt_0_5_overall", "mean"),
        "bridge_ratio_mean": ("bridge_ratio_mean", "mean"),
        "bridge_ratio_min": ("bridge_ratio_min", "mean"),
        "bridge_ratio_max": ("bridge_ratio_max", "mean"),
        "bridge_ratio_p95": ("bridge_ratio_p95", "mean"),
        "bridge_ratio_violations": ("bridge_ratio_violations", "mean"),
    }
    present = {name: value for name, value in agg_spec.items() if value[0] in valid.columns}
    return valid.groupby(list(group_cols) + ["method"], as_index=False).agg(**present)


def _add_metric_ranks(summary, *, group_cols: Sequence[str], method_order: Sequence[str]) -> Any:
    pd = load_pandas()
    if summary.empty:
        out = summary.copy()
        out["rank_rmse_test"] = pd.Series(dtype="float64")
        out["rank_lpd_test"] = pd.Series(dtype="float64")
        out["rank_r2_test"] = pd.Series(dtype="float64")
        return out

    order_map = {str(method): idx for idx, method in enumerate(method_order)}
    parts = []
    for _, sub in summary.groupby(list(group_cols), dropna=False, sort=False):
        block = sub.copy()
        block["rank_rmse_test"] = np.nan
        block["rank_lpd_test"] = np.nan
        block["rank_r2_test"] = np.nan
        block["_method_order"] = block["method"].map(lambda x: order_map.get(str(x), len(order_map)))

        valid_rmse = block["rmse_test"].notna()
        if bool(valid_rmse.any()):
            ordered = block.loc[valid_rmse].sort_values(
                ["rmse_test", "mae_test", "runtime_mean", "_method_order", "method"],
                kind="stable",
            )
            block.loc[ordered.index, "rank_rmse_test"] = np.arange(1, len(ordered) + 1, dtype=float)

        valid_lpd = block["lpd_test"].notna()
        if bool(valid_lpd.any()):
            ordered = block.loc[valid_lpd].sort_values(
                ["lpd_test", "r2_test", "_method_order", "method"],
                ascending=[False, False, True, True],
                kind="stable",
            )
            block.loc[ordered.index, "rank_lpd_test"] = np.arange(1, len(ordered) + 1, dtype=float)

        valid_r2 = block["r2_test"].notna()
        if bool(valid_r2.any()):
            ordered = block.loc[valid_r2].sort_values(
                ["r2_test", "rmse_test", "_method_order", "method"],
                ascending=[False, True, True, True],
                kind="stable",
            )
            block.loc[ordered.index, "rank_r2_test"] = np.arange(1, len(ordered) + 1, dtype=float)

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
    name = str(metric)
    if name.startswith("lpd") or name.startswith("r2"):
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


def _parse_json_list(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, str) or not payload.strip():
        return []
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return []
    return list(data) if isinstance(data, list) else []


def _pairwise_jaccard(masks: list[np.ndarray]) -> float:
    if len(masks) <= 1:
        return float("nan")
    scores: list[float] = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            left = np.asarray(masks[i], dtype=bool)
            right = np.asarray(masks[j], dtype=bool)
            denom = int(np.sum(left | right))
            if denom == 0:
                scores.append(1.0)
            else:
                scores.append(float(np.sum(left & right) / denom))
    return float(np.mean(scores)) if scores else float("nan")


def build_selection_stability(
    raw,
    *,
    group_cols: Sequence[str],
    required_metric_cols: Sequence[str] = DEFAULT_REQUIRED_METRICS,
):
    pd = load_pandas()
    valid = _filter_valid_rows(raw, required_metric_cols=required_metric_cols)
    rows: list[dict[str, Any]] = []
    if valid.empty:
        return pd.DataFrame(
            columns=list(group_cols)
            + [
                "method",
                "method_label",
                "n_valid_reps",
                "mean_pairwise_group_jaccard",
                "modal_top_group_label",
                "modal_top_group_rate",
                "selected_group_count_mean",
                "selected_group_count_sd",
            ]
        )

    for key, sub in valid.groupby(list(group_cols) + ["method"], dropna=False, sort=False):
        values = key if isinstance(key, tuple) else (key,)
        base = {name: values[idx] for idx, name in enumerate(list(group_cols) + ["method"])}
        masks = [np.asarray(_parse_json_list(item), dtype=bool) for item in sub["group_selected_json"].tolist()]
        top_labels = [str(item) for item in sub["top_group_label"].fillna("").astype(str).tolist() if str(item)]
        selected_counts = [int(np.sum(mask)) for mask in masks if mask.size]
        if top_labels:
            label_counts: dict[str, int] = {}
            for label in top_labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            modal_top_group_label = max(label_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
            modal_top_group_rate = float(max(label_counts.values()) / max(len(top_labels), 1))
        else:
            modal_top_group_label = ""
            modal_top_group_rate = float("nan")
        rows.append(
            {
                **base,
                "method_label": method_display_name(str(base["method"])),
                "n_valid_reps": int(sub["replicate_id"].nunique()),
                "mean_pairwise_group_jaccard": float(_pairwise_jaccard(masks)),
                "modal_top_group_label": modal_top_group_label,
                "modal_top_group_rate": modal_top_group_rate,
                "selected_group_count_mean": float(np.mean(selected_counts)) if selected_counts else float("nan"),
                "selected_group_count_sd": float(np.std(selected_counts, ddof=1)) if len(selected_counts) > 1 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_group_selection_frequency(
    raw,
    *,
    group_cols: Sequence[str],
    required_metric_cols: Sequence[str] = DEFAULT_REQUIRED_METRICS,
):
    pd = load_pandas()
    valid = _filter_valid_rows(raw, required_metric_cols=required_metric_cols)
    rows: list[dict[str, Any]] = []
    if valid.empty:
        return pd.DataFrame(
            columns=list(group_cols) + ["method", "method_label", "group_id", "group_label", "selection_rate", "mean_group_score"]
        )

    for key, sub in valid.groupby(list(group_cols) + ["method"], dropna=False, sort=False):
        values = key if isinstance(key, tuple) else (key,)
        base = {name: values[idx] for idx, name in enumerate(list(group_cols) + ["method"])}
        group_labels = _parse_json_list(sub["group_labels_json"].iloc[0]) if "group_labels_json" in sub.columns else []
        selection_masks = [np.asarray(_parse_json_list(item), dtype=bool) for item in sub["group_selected_json"].tolist()]
        group_scores = [np.asarray(_parse_json_list(item), dtype=float) for item in sub["group_scores_json"].tolist()]
        if not group_scores:
            continue
        n_groups = int(max(arr.size for arr in group_scores))
        if not group_labels:
            group_labels = [f"group_{gid + 1}" for gid in range(n_groups)]
        for gid in range(n_groups):
            selected = [bool(mask[gid]) for mask in selection_masks if mask.size > gid]
            scores = [float(arr[gid]) for arr in group_scores if arr.size > gid and np.isfinite(arr[gid])]
            rows.append(
                {
                    **base,
                    "method_label": method_display_name(str(base["method"])),
                    "group_id": int(gid),
                    "group_label": str(group_labels[gid]),
                    "selection_rate": float(np.mean(selected)) if selected else float("nan"),
                    "mean_group_score": float(np.mean(scores)) if scores else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def write_json_manifest(payload: dict[str, Any], path: Path | str) -> Path:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path_obj
