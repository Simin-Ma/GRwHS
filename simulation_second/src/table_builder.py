from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from simulation_project.src.utils import load_pandas, method_display_name

from .utils import ensure_dir, resolve_history_results_dir
from .reporting import build_paired_summary, default_setting_group_cols


DEFAULT_NDIGITS = 6
CSV_FLOAT_FORMAT = "%.8f"
FULL_TABLE_BASE_COLS = (
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
    "kappa_signal_mean",
    "kappa_null_mean",
    "kappa_signal_prob_gt_0_5",
    "kappa_null_prob_gt_0_5",
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


def _fmt_pm(mean: float, se: float, ndigits: int = DEFAULT_NDIGITS) -> str:
    pd = load_pandas()
    if pd.isna(mean):
        return "--"
    if pd.isna(se):
        return f"{float(mean):.{ndigits}f}"
    return f"{float(mean):.{ndigits}f} +/- {float(se):.{ndigits}f}"


def _setting_display(row) -> str:
    return (
        f"{row['setting_id']} "
        f"(n={int(row['n_train'])}, rw={float(row['rho_within']):.2f}, rb={float(row['rho_between']):.2f})"
    )


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

    agg_spec = {
        "n_paired": ("replicate_id", "nunique"),
        "mse_overall_mean": ("mse_overall", "mean"),
        "mse_signal_mean": ("mse_signal", "mean"),
        "mse_null_mean": ("mse_null", "mean"),
        "coverage_mean": ("coverage_95", "mean"),
        "avg_ci_length_mean": ("avg_ci_length", "mean"),
        "lpd_test_mean": ("lpd_test", "mean"),
        "runtime_mean": ("runtime_seconds", "mean"),
    }
    agg_present = {
        out_name: spec
        for out_name, spec in agg_spec.items()
        if spec[0] in paired_raw.columns
    }
    agg = paired_raw.groupby(list(group_cols) + ["method"], as_index=False).agg(**agg_present)

    se_spec = {
        "mse_overall_se": ("mse_overall", _se),
        "mse_signal_se": ("mse_signal", _se),
        "mse_null_se": ("mse_null", _se),
        "coverage_se": ("coverage_95", _se),
        "avg_ci_length_se": ("avg_ci_length", _se),
        "lpd_test_se": ("lpd_test", _se),
        "runtime_se": ("runtime_seconds", _se),
    }
    se_present = {
        out_name: spec
        for out_name, spec in se_spec.items()
        if spec[0] in paired_raw.columns
    }
    se_df = paired_raw.groupby(list(group_cols) + ["method"], as_index=False).agg(**se_present)
    out = agg.merge(se_df, on=list(group_cols) + ["method"], how="left")
    for col in [
        "mse_overall_mean",
        "mse_signal_mean",
        "mse_null_mean",
        "coverage_mean",
        "avg_ci_length_mean",
        "lpd_test_mean",
        "runtime_mean",
        "mse_overall_se",
        "mse_signal_se",
        "mse_null_se",
        "coverage_se",
        "avg_ci_length_se",
        "lpd_test_se",
        "runtime_se",
    ]:
        if col not in out.columns:
            out[col] = np.nan
    out["method_label"] = out["method"].map(method_display_name)
    order_map = {str(method): idx for idx, method in enumerate(method_order)}
    out["method_order"] = out["method"].map(lambda x: order_map.get(str(x), len(order_map)))
    out = out.sort_values(list(group_cols) + ["method_order", "method"], kind="stable").drop(columns=["method_order"])
    return out.reset_index(drop=True)


def build_winloss_table(paired_raw, *, group_cols: Sequence[str], focal_method: str = "GR_RHS"):
    pd = load_pandas()
    rows: list[dict[str, Any]] = []
    if paired_raw.empty:
        return pd.DataFrame()

    for setting_vals, sub in paired_raw.groupby(list(group_cols), dropna=False, sort=False):
        wide_overall = sub.pivot(index="replicate_id", columns="method", values="mse_overall")
        wide_signal = sub.pivot(index="replicate_id", columns="method", values="mse_signal")
        if focal_method not in wide_overall.columns or focal_method not in wide_signal.columns:
            continue
        focal_overall = wide_overall[focal_method]
        focal_signal = wide_signal[focal_method]
        vals = setting_vals if isinstance(setting_vals, tuple) else (setting_vals,)
        base = {key: value for key, value in zip(group_cols, vals)}
        base["n_paired"] = int(wide_overall.shape[0])
        for method in sorted(set(sub["method"].astype(str).tolist())):
            if method == focal_method:
                continue
            diff_overall = wide_overall[method] - focal_overall
            diff_signal = wide_signal[method] - focal_signal
            rows.append(
                {
                    **base,
                    "method": str(method),
                    "focal_method": str(focal_method),
                    "gr_wins_overall": int((diff_overall > 0.0).sum()),
                    "method_wins_overall": int((diff_overall < 0.0).sum()),
                    "ties_overall": int(np.isclose(diff_overall, 0.0).sum()),
                    "gr_wins_signal": int((diff_signal > 0.0).sum()),
                    "method_wins_signal": int((diff_signal < 0.0).sum()),
                    "ties_signal": int(np.isclose(diff_signal, 0.0).sum()),
                }
            )
    return pd.DataFrame(rows)


def build_main_table(method_df, winloss_df, *, group_cols: Sequence[str], focal_method: str = "GR_RHS", gigg_method: str = "GIGG_MMLE"):
    pd = load_pandas()
    rows: list[dict[str, Any]] = []
    if method_df.empty:
        return pd.DataFrame()

    for _, sub in method_df.groupby(list(group_cols), dropna=False, sort=False):
        if focal_method not in set(sub["method"].astype(str)):
            continue
        focal = sub.loc[sub["method"] == focal_method].iloc[0]
        runner_pool = sub.loc[sub["method"] != focal_method].sort_values(
            ["mse_overall_mean", "mse_signal_mean", "runtime_mean"],
            kind="stable",
        )
        if runner_pool.empty:
            continue
        runner = runner_pool.iloc[0]
        base = {key: focal[key] for key in group_cols if key in focal.index}
        row = {
            **base,
            "setting": _setting_display(focal),
            "n_paired": int(focal["n_paired"]),
            "gr_mse_overall_mean": float(focal["mse_overall_mean"]),
            "gr_mse_overall_se": float(focal["mse_overall_se"]),
            "gr_mse_signal_mean": float(focal["mse_signal_mean"]),
            "gr_mse_signal_se": float(focal["mse_signal_se"]),
            "gr_coverage_mean": float(focal["coverage_mean"]),
            "runner_up": str(runner["method_label"]),
            "runner_mse_overall_mean": float(runner["mse_overall_mean"]),
            "runner_mse_overall_se": float(runner["mse_overall_se"]),
            "runner_mse_signal_mean": float(runner["mse_signal_mean"]),
            "runner_mse_signal_se": float(runner["mse_signal_se"]),
        }

        if gigg_method in set(sub["method"].astype(str)):
            gigg = sub.loc[sub["method"] == gigg_method].iloc[0]
            row["gigg_mse_overall_mean"] = float(gigg["mse_overall_mean"])
            row["gigg_mse_overall_se"] = float(gigg["mse_overall_se"])
            row["gigg_coverage_mean"] = float(gigg["coverage_mean"])
        else:
            row["gigg_mse_overall_mean"] = float("nan")
            row["gigg_mse_overall_se"] = float("nan")
            row["gigg_coverage_mean"] = float("nan")

        if not winloss_df.empty:
            match = winloss_df.loc[
                (winloss_df["setting_id"] == focal["setting_id"]) &
                (winloss_df["method"] == runner["method"])
            ]
            if not match.empty:
                w = match.iloc[0]
                row["paired_overall_gr_vs_runner"] = (
                    f"{int(w['gr_wins_overall'])}-{int(w['method_wins_overall'])}-{int(w['ties_overall'])}"
                )
                row["paired_signal_gr_vs_runner"] = (
                    f"{int(w['gr_wins_signal'])}-{int(w['method_wins_signal'])}-{int(w['ties_signal'])}"
                )
            else:
                row["paired_overall_gr_vs_runner"] = "--"
                row["paired_signal_gr_vs_runner"] = "--"
        else:
            row["paired_overall_gr_vs_runner"] = "--"
            row["paired_signal_gr_vs_runner"] = "--"
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["setting_id"], kind="stable").reset_index(drop=True)
    return out


def build_full_appendix_table(summary_paired, *, method_order: Sequence[str]):
    pd = load_pandas()
    if summary_paired.empty:
        return pd.DataFrame()

    base_cols = [col for col in FULL_TABLE_BASE_COLS if col in summary_paired.columns]
    excluded = set(base_cols) | {"summary_scope", "rank_mse_overall", "rank_mse_signal", "methods_required", "methods_list"}
    metric_cols = _ordered_full_metric_cols(summary_paired, excluded=excluded)
    out = summary_paired.loc[:, base_cols + metric_cols].copy()

    order_map = {str(method): idx for idx, method in enumerate(method_order)}
    out["_method_order"] = out["method"].map(lambda x: order_map.get(str(x), len(order_map))) if "method" in out.columns else 0
    sort_cols = [col for col in ["setting_id"] if col in out.columns]
    if "rank_mse_overall" in summary_paired.columns and "setting_id" in out.columns:
        out["_rank_mse_overall"] = summary_paired.loc[out.index, "rank_mse_overall"]
        sort_cols.extend(["_rank_mse_overall", "_method_order", "method"])
    elif "method" in out.columns:
        sort_cols.extend(["_method_order", "method"])
    if sort_cols:
        out = out.sort_values(sort_cols, kind="stable")
    return out.drop(columns=[col for col in ["_method_order", "_rank_mse_overall"] if col in out.columns]).reset_index(drop=True)


def build_figure1_coefficient_recovery_profile_data(coefficient_estimates, *, focal_method: str = "GR_RHS"):
    pd = load_pandas()
    if coefficient_estimates is None or coefficient_estimates.empty:
        return pd.DataFrame()

    frame = coefficient_estimates.copy()
    needed = {
        "setting_id",
        "replicate_id",
        "method",
        "coefficient_index",
        "group_id",
        "true_beta",
        "estimated_beta",
        "is_active_coefficient",
    }
    if not needed.issubset(set(frame.columns)):
        return pd.DataFrame()
    if "paired_common_converged" in frame.columns:
        frame = frame.loc[frame["paired_common_converged"].fillna(False).astype(bool)].copy()
    if frame.empty:
        return pd.DataFrame()

    if "role" in frame.columns:
        main_showcase = frame.loc[
            frame["role"].astype(str).str.contains("main showcase", case=False, na=False),
            "setting_id",
        ].drop_duplicates()
        setting_candidates = [str(item) for item in main_showcase.tolist()]
    else:
        setting_candidates = []
    if not setting_candidates and "setting_5_multimode_equal" in set(frame["setting_id"].astype(str)):
        setting_candidates = ["setting_5_multimode_equal"]
    if not setting_candidates:
        setting_candidates = sorted(frame["setting_id"].astype(str).drop_duplicates().tolist())
    if not setting_candidates:
        return pd.DataFrame()

    focal_setting_id = str(setting_candidates[0])
    setting_frame = frame.loc[frame["setting_id"].astype(str).eq(focal_setting_id)].copy()
    if setting_frame.empty:
        return pd.DataFrame()

    rep_frame = setting_frame.loc[setting_frame["method"].astype(str).eq(str(focal_method))].copy()
    if rep_frame.empty:
        rep_frame = setting_frame.copy()

    rep_rows: list[dict[str, Any]] = []
    for replicate_id, sub in rep_frame.groupby("replicate_id", dropna=False, sort=False):
        sq = pd.to_numeric(sub["sq_error"], errors="coerce") if "sq_error" in sub.columns else (
            pd.to_numeric(sub["estimated_beta"], errors="coerce") - pd.to_numeric(sub["true_beta"], errors="coerce")
        ) ** 2
        active_mask = sub["is_active_coefficient"].fillna(False).astype(bool)
        signal_vals = sq.loc[active_mask] if bool(active_mask.any()) else sq.iloc[0:0]
        if bool(signal_vals.notna().any()):
            signal_mse = float(signal_vals.mean())
        elif bool(sq.notna().any()):
            signal_mse = float(sq.mean())
        else:
            signal_mse = float("nan")
        rep_rows.append(
            {
                "replicate_id": replicate_id,
                "signal_mse": signal_mse,
            }
        )
    rep_stats = pd.DataFrame(rep_rows)
    if rep_stats.empty:
        return pd.DataFrame()
    rep_stats["distance_to_median"] = (
        pd.to_numeric(rep_stats["signal_mse"], errors="coerce") - float(pd.to_numeric(rep_stats["signal_mse"], errors="coerce").median())
    ).abs()
    rep_pick = rep_stats.sort_values(["distance_to_median", "replicate_id"], kind="stable").iloc[0]
    focal_replicate_id = rep_pick["replicate_id"]

    profile = setting_frame.loc[setting_frame["replicate_id"].eq(focal_replicate_id)].copy()
    if profile.empty:
        return pd.DataFrame()

    truth = profile.drop_duplicates(subset=["coefficient_index"]).copy()
    truth["abs_true_beta"] = pd.to_numeric(truth["abs_true_beta"], errors="coerce")
    truth["group_id"] = pd.to_numeric(truth["group_id"], errors="coerce")
    truth["coefficient_index"] = pd.to_numeric(truth["coefficient_index"], errors="coerce")
    truth["group_sort_active"] = truth.groupby("group_id")["is_active_coefficient"].transform(lambda s: bool(s.fillna(False).astype(bool).any()))
    truth = truth.sort_values(
        ["group_id", "is_active_coefficient", "abs_true_beta", "coefficient_index"],
        ascending=[True, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    truth["plot_order"] = np.arange(truth.shape[0], dtype=int)
    truth["group_rank_within_plot"] = truth.groupby("group_id", sort=False).cumcount()

    order_map = truth.set_index("coefficient_index")["plot_order"].to_dict()
    rank_map = truth.set_index("coefficient_index")["group_rank_within_plot"].to_dict()
    profile["plot_order"] = profile["coefficient_index"].map(order_map)
    profile["group_rank_within_plot"] = profile["coefficient_index"].map(rank_map)

    group_info = (
        truth.groupby("group_id", as_index=False)
        .agg(
            group_plot_lo=("plot_order", "min"),
            group_plot_hi=("plot_order", "max"),
            group_size=("group_size", "first"),
            is_active_group=("group_sort_active", "first"),
        )
        .sort_values(["group_id"], kind="stable")
        .reset_index(drop=True)
    )
    group_info["group_plot_center"] = 0.5 * (
        pd.to_numeric(group_info["group_plot_lo"], errors="coerce")
        + pd.to_numeric(group_info["group_plot_hi"], errors="coerce")
    )
    profile = profile.merge(group_info, on="group_id", how="left")

    method_stats = []
    for method, sub in profile.groupby("method", dropna=False, sort=False):
        sq = pd.to_numeric(sub["sq_error"], errors="coerce")
        active_mask = sub["is_active_coefficient"].fillna(False).astype(bool)
        signal_vals = sq.loc[active_mask]
        method_stats.append(
            {
                "method": method,
                "method_signal_rmse": float(np.sqrt(signal_vals.mean())) if bool(signal_vals.notna().any()) else float("nan"),
                "method_overall_rmse": float(np.sqrt(sq.mean())) if bool(sq.notna().any()) else float("nan"),
            }
        )
    method_stats_df = pd.DataFrame(method_stats)
    profile = profile.merge(method_stats_df, on="method", how="left")

    if "method_order" not in profile.columns:
        order_map_method = {
            "GR_RHS": 0,
            "RHS": 1,
            "GHS_plus": 2,
            "GIGG_MMLE": 3,
            "LASSO_CV": 4,
            "OLS": 5,
        }
        profile["method_order"] = profile["method"].map(lambda x: order_map_method.get(str(x), len(order_map_method)))

    profile["representative_setting_id"] = str(focal_setting_id)
    profile["representative_replicate_id"] = int(focal_replicate_id)
    profile["representative_selector_method"] = str(focal_method)
    profile["representative_selector_signal_mse"] = float(rep_pick["signal_mse"]) if pd.notna(rep_pick["signal_mse"]) else float("nan")
    profile["representative_selector_distance_to_median"] = float(rep_pick["distance_to_median"]) if pd.notna(rep_pick["distance_to_median"]) else float("nan")

    out_cols = [
        col
        for col in [
            "setting_id",
            "setting_label",
            "family",
            "suite",
            "role",
            "replicate_id",
            "seed",
            "method",
            "method_label",
            "method_type",
            "method_order",
            "coefficient_index",
            "group_id",
            "group_size",
            "within_group_index",
            "group_rank_within_plot",
            "plot_order",
            "group_plot_lo",
            "group_plot_hi",
            "group_plot_center",
            "is_active_group",
            "is_active_coefficient",
            "true_beta",
            "estimated_beta",
            "error",
            "sq_error",
            "abs_error",
            "method_signal_rmse",
            "method_overall_rmse",
            "representative_setting_id",
            "representative_replicate_id",
            "representative_selector_method",
            "representative_selector_signal_mse",
            "representative_selector_distance_to_median",
        ]
        if col in profile.columns
    ]
    return profile.loc[:, out_cols].sort_values(
        ["method_order", "method", "plot_order"],
        kind="stable",
    ).reset_index(drop=True)


def write_markdown_main(df, path: Path | str) -> None:
    path_obj = Path(path)
    lines = [
        "| Setting | n paired | GR-RHS Overall | GR-RHS Signal | GR Cov. | Runner-up | Runner Overall | Runner Signal | Paired Overall (GR-Other-Tie) | Paired Signal (GR-Other-Tie) | GIGG Overall | GIGG Cov. |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            "| " + " | ".join(
                [
                    str(row["setting"]),
                    str(int(row["n_paired"])),
                    _fmt_pm(row["gr_mse_overall_mean"], row["gr_mse_overall_se"]),
                    _fmt_pm(row["gr_mse_signal_mean"], row["gr_mse_signal_se"]),
                    _fmt(row["gr_coverage_mean"]),
                    str(row["runner_up"]),
                    _fmt_pm(row["runner_mse_overall_mean"], row["runner_mse_overall_se"]),
                    _fmt_pm(row["runner_mse_signal_mean"], row["runner_mse_signal_se"]),
                    str(row["paired_overall_gr_vs_runner"]),
                    str(row["paired_signal_gr_vs_runner"]),
                    _fmt_pm(row["gigg_mse_overall_mean"], row["gigg_mse_overall_se"]),
                    _fmt(row["gigg_coverage_mean"]),
                ]
            ) + " |"
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


def write_latex_main(df, path: Path | str) -> None:
    path_obj = Path(path)
    lines = [
        r"\begin{tabular}{lccccccccccc}",
        r"\toprule",
        r"Setting & $n$ & GR Overall & GR Signal & GR Cov. & Runner-up & Runner Overall & Runner Signal & Pair O & Pair S & GIGG Overall & GIGG Cov. \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{_latex_escape(row['setting'])} & {int(row['n_paired'])} & "
            f"{_fmt(row['gr_mse_overall_mean'])} $\\pm$ {_fmt(row['gr_mse_overall_se'])} & "
            f"{_fmt(row['gr_mse_signal_mean'])} $\\pm$ {_fmt(row['gr_mse_signal_se'])} & "
            f"{_fmt(row['gr_coverage_mean'])} & {_latex_escape(row['runner_up'])} & "
            f"{_fmt(row['runner_mse_overall_mean'])} $\\pm$ {_fmt(row['runner_mse_overall_se'])} & "
            f"{_fmt(row['runner_mse_signal_mean'])} $\\pm$ {_fmt(row['runner_mse_signal_se'])} & "
            f"{_latex_escape(row['paired_overall_gr_vs_runner'])} & {_latex_escape(row['paired_signal_gr_vs_runner'])} & "
            f"{_fmt(row['gigg_mse_overall_mean'])} $\\pm$ {_fmt(row['gigg_mse_overall_se'])} & "
            f"{_fmt(row['gigg_coverage_mean'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
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


def build_paper_tables(
    raw,
    *,
    out_dir: Path | str,
    method_order: Sequence[str],
    group_cols: Sequence[str] | None = None,
    required_metric_cols: Sequence[str] = ("mse_null", "mse_signal", "mse_overall", "lpd_test"),
    coefficient_estimates=None,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    figure_dir = ensure_dir(out_path / "figure_data")
    use_group_cols = list(group_cols or default_setting_group_cols(raw))
    paired_raw, _, summary_paired = build_paired_summary(
        raw,
        group_cols=use_group_cols,
        method_levels=method_order,
        required_metric_cols=required_metric_cols,
        method_order=method_order,
    )
    method_df = build_method_table(paired_raw, group_cols=use_group_cols, method_order=method_order)
    winloss_df = build_winloss_table(paired_raw, group_cols=use_group_cols)
    main_df = build_main_table(method_df, winloss_df, group_cols=use_group_cols)
    appendix_df = build_full_appendix_table(summary_paired, method_order=method_order)

    method_csv = out_path / "paper_table_method_means_se.csv"
    winloss_csv = out_path / "paper_table_paired_winloss.csv"
    main_csv = out_path / "paper_table_main.csv"
    appendix_csv = out_path / "paper_table_appendix_full.csv"
    main_md = out_path / "paper_table_main.md"
    appendix_md = out_path / "paper_table_appendix_full.md"
    main_tex = out_path / "paper_table_main.tex"
    appendix_tex = out_path / "paper_table_appendix_full.tex"
    figure1_profile_csv = figure_dir / "figure1_coefficient_recovery_profile.csv"

    method_df.to_csv(method_csv, index=False, float_format=CSV_FLOAT_FORMAT)
    winloss_df.to_csv(winloss_csv, index=False, float_format=CSV_FLOAT_FORMAT)
    main_df.to_csv(main_csv, index=False, float_format=CSV_FLOAT_FORMAT)
    appendix_df.to_csv(appendix_csv, index=False, float_format=CSV_FLOAT_FORMAT)
    write_markdown_main(main_df, main_md)
    write_markdown_full_table(appendix_df, appendix_md, group_col="setting_id")
    write_latex_main(main_df, main_tex)
    write_latex_full_table(appendix_df, appendix_tex, group_col="setting_id")

    fig1_profile = build_figure1_coefficient_recovery_profile_data(coefficient_estimates)
    fig1_profile.to_csv(figure1_profile_csv, index=False, float_format=CSV_FLOAT_FORMAT)

    return {
        "paper_table_method_means_se": str(method_csv),
        "paper_table_paired_winloss": str(winloss_csv),
        "paper_table_main_csv": str(main_csv),
        "paper_table_appendix_csv": str(appendix_csv),
        "paper_table_main_md": str(main_md),
        "paper_table_appendix_md": str(appendix_md),
        "paper_table_main_tex": str(main_tex),
        "paper_table_appendix_tex": str(appendix_tex),
        "figure1_coefficient_recovery_profile": str(figure1_profile_csv),
    }


def build_paper_tables_from_results_dir(
    results_dir: Path | str,
    *,
    method_order: Sequence[str],
    group_cols: Sequence[str] | None = None,
    required_metric_cols: Sequence[str] = ("mse_null", "mse_signal", "mse_overall", "lpd_test"),
):
    pd = load_pandas()
    base = resolve_history_results_dir(results_dir, required_files=("raw_results.csv",))
    raw = pd.read_csv(base / "raw_results.csv")
    coefficient_estimates_path = base / "coefficient_estimates.csv"
    if coefficient_estimates_path.exists():
        try:
            coefficient_estimates = pd.read_csv(coefficient_estimates_path)
        except pd.errors.EmptyDataError:
            coefficient_estimates = pd.DataFrame()
    else:
        coefficient_estimates = pd.DataFrame()
    return build_paper_tables(
        raw,
        out_dir=base / "paper_tables",
        method_order=method_order,
        group_cols=group_cols,
        required_metric_cols=required_metric_cols,
        coefficient_estimates=coefficient_estimates,
    )
