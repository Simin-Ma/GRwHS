from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_COL = "trust_experts"
PERIOD_COL = "period"
CATEGORY_COLUMNS = ["age", "gender", "raceethnicity"]
SPLINE_COLUMNS = ["cli", "hh_cmnty_cli"]

AGE_ORDER = ["18-24", "25-44", "45-64", "65plus", "NotReported"]
GENDER_ORDER = ["Female", "Male", "Other", "NotReported"]
RACE_ORDER = [
    "NonHispanicAsian",
    "NonHispanicWhite",
    "NonHispanicBlackAfricanAmerican",
    "Hispanic",
    "NonHispanicAmericanIndianAlaskaNative",
    "NonHispanicMultipleOther",
    "NonHispanicNativeHawaiianPacificIslander",
    "NotReported",
]

DISPLAY_LABELS: Dict[str, str] = {
    "trust_experts": "Trust in Experts",
    "period": "Survey period",
    "cli": "Reported COVID-like illness (%)",
    "hh_cmnty_cli": "Reported community & household CLI (%)",
    "age": "Age",
    "gender": "Gender",
    "raceethnicity": "Race / ethnicity",
}

FAMILY_COLORS: Dict[str, str] = {
    "age": "#0f6b50",
    "gender": "#1769aa",
    "raceethnicity": "#b42318",
}

SHORT_LEVEL_LABELS: Dict[str, str] = {
    "NonHispanicAsian": "NH Asian",
    "NonHispanicWhite": "NH White",
    "NonHispanicBlackAfricanAmerican": "NH Black",
    "Hispanic": "Hispanic",
    "NonHispanicAmericanIndianAlaskaNative": "NH AI/AN",
    "NonHispanicMultipleOther": "NH Multi/Other",
    "NonHispanicNativeHawaiianPacificIslander": "NH NH/PI",
    "NotReported": "Not reported",
}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_serializable(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def _load_frame(processed_dir: Path) -> pd.DataFrame:
    csv_path = processed_dir / "analysis_bundle" / "trust_experts_raw.csv"
    frame = pd.read_csv(csv_path)
    frame[PERIOD_COL] = pd.to_datetime(frame[PERIOD_COL])
    return frame


def _ordered_categories(column: str) -> List[str]:
    if column == "age":
        return AGE_ORDER
    if column == "gender":
        return GENDER_ORDER
    if column == "raceethnicity":
        return RACE_ORDER
    return []


def _short_level_label(level: str) -> str:
    return SHORT_LEVEL_LABELS.get(level, level)


def _categorical_summary(frame: pd.DataFrame) -> pd.DataFrame:
    overall_mean = float(frame[TARGET_COL].mean())
    rows: List[Dict[str, Any]] = []
    for column in CATEGORY_COLUMNS:
        stats = frame.groupby(column)[TARGET_COL].agg(["mean", "std", "count"]).reset_index()
        order = _ordered_categories(column)
        if order:
            stats[column] = pd.Categorical(stats[column], categories=order, ordered=True)
            stats = stats.sort_values(column)
        for _, row in stats.iterrows():
            level = str(row[column])
            count = int(row["count"])
            std = float(row["std"]) if pd.notna(row["std"]) else 0.0
            se = std / np.sqrt(max(count, 1))
            rows.append(
                {
                    "family": column,
                    "family_label": DISPLAY_LABELS[column],
                    "level": level,
                    "count": count,
                    "mean": float(row["mean"]),
                    "mean_deviation": float(row["mean"] - overall_mean),
                    "lower_95": float(row["mean"] - 1.96 * se),
                    "upper_95": float(row["mean"] + 1.96 * se),
                }
            )
    return pd.DataFrame(rows)


def _region_summary(frame: pd.DataFrame) -> pd.DataFrame:
    overall_mean = float(frame[TARGET_COL].mean())
    stats = frame.groupby("region")[TARGET_COL].agg(["mean", "std", "count"]).reset_index()
    stats["mean_deviation"] = stats["mean"] - overall_mean
    stats["se"] = stats["std"].fillna(0.0) / np.sqrt(np.maximum(stats["count"], 1))
    stats["lower_95"] = stats["mean"] - 1.96 * stats["se"]
    stats["upper_95"] = stats["mean"] + 1.96 * stats["se"]
    stats.sort_values("mean_deviation", inplace=True)
    stats.reset_index(drop=True, inplace=True)
    return stats


def _period_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = frame.groupby(PERIOD_COL)[TARGET_COL].agg(["mean", "median", "count"]).reset_index()
    q = frame.groupby(PERIOD_COL)[TARGET_COL].quantile([0.25, 0.75]).unstack().reset_index()
    q.columns = [PERIOD_COL, "q25", "q75"]
    summary = summary.merge(q, on=PERIOD_COL, how="left")
    return summary


def _binned_curve(frame: pd.DataFrame, x_col: str, y_col: str, n_bins: int = 16) -> pd.DataFrame:
    work = frame[[x_col, y_col]].dropna().copy()
    work["bin"] = pd.qcut(work[x_col], q=min(n_bins, work.shape[0]), duplicates="drop")
    summary = work.groupby("bin").agg(
        x_mid=(x_col, "median"),
        y_mean=(y_col, "mean"),
        y_q25=(y_col, lambda s: float(np.quantile(s, 0.25))),
        y_q75=(y_col, lambda s: float(np.quantile(s, 0.75))),
        count=(y_col, "size"),
    )
    return summary.reset_index(drop=True)


def _style_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#222222")
    ax.set_facecolor("white")


def _plot_target_distribution(ax: plt.Axes, frame: pd.DataFrame) -> None:
    _style_axis(ax)
    values = frame[TARGET_COL].to_numpy(dtype=float)
    q25, median, q75 = np.quantile(values, [0.25, 0.50, 0.75])
    mean = float(values.mean())
    ax.hist(values, bins=28, color="#153b50", alpha=0.9, edgecolor="white", linewidth=0.5)
    ax.axvspan(q25, q75, color="#8ecae6", alpha=0.25)
    ax.axvline(mean, color="#b42318", linewidth=2.0, label=f"Mean = {mean:.2f}")
    ax.axvline(median, color="#0f6b50", linewidth=2.0, linestyle="--", label=f"Median = {median:.2f}")
    ax.set_title("Outcome distribution", fontsize=12.5, fontweight="bold")
    ax.set_xlabel(DISPLAY_LABELS[TARGET_COL])
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="upper left")


def _plot_period_trend(ax: plt.Axes, period_summary: pd.DataFrame) -> None:
    _style_axis(ax)
    ax2 = ax.twinx()
    counts = period_summary["count"].to_numpy(dtype=float)
    periods = pd.to_datetime(period_summary[PERIOD_COL])
    ax2.bar(periods, counts, width=24, color="#d9d9d9", alpha=0.45, zorder=0)
    ax.fill_between(
        periods,
        period_summary["q25"].to_numpy(dtype=float),
        period_summary["q75"].to_numpy(dtype=float),
        color="#8ecae6",
        alpha=0.35,
        linewidth=0.0,
        zorder=1,
    )
    ax.plot(periods, period_summary["mean"].to_numpy(dtype=float), color="#153b50", linewidth=2.4, marker="o", zorder=3)
    ax.set_title("Monthly trend in trust", fontsize=12.5, fontweight="bold")
    ax.set_xlabel(DISPLAY_LABELS[PERIOD_COL])
    ax.set_ylabel("Mean trust in experts")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.22)
    ax2.set_ylabel("Monthly sample size")
    ax2.grid(False)
    ax2.spines["right"].set_color("#666666")
    ax.text(
        0.02,
        0.95,
        "Ribbon = monthly IQR\nBars = n per month",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        color="#333333",
    )


def _plot_spline_panel(ax: plt.Axes, frame: pd.DataFrame, spline_meta: Dict[str, Any], x_col: str) -> None:
    _style_axis(ax)
    x = frame[x_col].to_numpy(dtype=float)
    y = frame[TARGET_COL].to_numpy(dtype=float)
    hb = ax.hexbin(x, y, gridsize=34, mincnt=1, cmap="YlGnBu", linewidths=0.0)
    curve = _binned_curve(frame, x_col, TARGET_COL)
    ax.fill_between(
        curve["x_mid"].to_numpy(dtype=float),
        curve["y_q25"].to_numpy(dtype=float),
        curve["y_q75"].to_numpy(dtype=float),
        color="#f4a261",
        alpha=0.26,
        linewidth=0.0,
    )
    ax.plot(curve["x_mid"].to_numpy(dtype=float), curve["y_mean"].to_numpy(dtype=float), color="#b42318", linewidth=2.2)
    for knot in spline_meta.get("inner_knots", []):
        ax.axvline(float(knot), color="#666666", linestyle=":", linewidth=0.9, alpha=0.75)
    ax.set_title(f"{DISPLAY_LABELS[x_col]} vs trust", fontsize=12.5, fontweight="bold")
    ax.set_xlabel(DISPLAY_LABELS[x_col])
    ax.set_ylabel(DISPLAY_LABELS[TARGET_COL])
    ax.grid(alpha=0.18)
    cb = plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.set_ylabel("Count", rotation=90, fontsize=9)
    cb.ax.tick_params(labelsize=8)


def _plot_categorical_deviation(ax: plt.Axes, cat_summary: pd.DataFrame, overall_mean: float) -> None:
    _style_axis(ax)
    order_rows: List[pd.DataFrame] = []
    for family in CATEGORY_COLUMNS:
        sub = cat_summary[cat_summary["family"] == family].copy()
        order = _ordered_categories(family)
        if order:
            sub["level"] = pd.Categorical(sub["level"], categories=order, ordered=True)
            sub = sub.sort_values("level")
        order_rows.append(sub)
    ordered = pd.concat(order_rows, ignore_index=True)
    ordered["y_pos"] = np.arange(ordered.shape[0])[::-1]

    for family in CATEGORY_COLUMNS:
        sub = ordered[ordered["family"] == family]
        color = FAMILY_COLORS[family]
        ax.hlines(
            sub["y_pos"].to_numpy(dtype=float),
            sub["lower_95"].to_numpy(dtype=float) - overall_mean,
            sub["upper_95"].to_numpy(dtype=float) - overall_mean,
            color=color,
            linewidth=2.0,
            alpha=0.85,
        )
        ax.scatter(
            sub["mean_deviation"].to_numpy(dtype=float),
            sub["y_pos"].to_numpy(dtype=float),
            s=np.sqrt(sub["count"].to_numpy(dtype=float)) * 6.0,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.95,
            label=DISPLAY_LABELS[family],
        )
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    ax.set_yticks(ordered["y_pos"].to_numpy(dtype=float))
    ax.set_yticklabels([_short_level_label(str(level)) for level in ordered["level"].astype(str)], fontsize=8.6)
    ax.set_xlabel("Subgroup mean - overall mean")
    ax.set_title("Categorical subgroup deviations", fontsize=12.5, fontweight="bold")
    ax.grid(axis="x", alpha=0.22)
    handles, labels = ax.get_legend_handles_labels()
    dedup: Dict[str, Any] = {}
    for h, label in zip(handles, labels):
        dedup[label] = h
    ax.legend(dedup.values(), dedup.keys(), frameon=False, loc="lower right", fontsize=8.8)


def _plot_region_heterogeneity(ax: plt.Axes, region_summary: pd.DataFrame) -> None:
    _style_axis(ax)
    sub = region_summary.copy()
    sub["rank"] = np.arange(sub.shape[0])
    ax.hlines(
        sub["rank"].to_numpy(dtype=float),
        np.zeros(sub.shape[0]),
        sub["mean_deviation"].to_numpy(dtype=float),
        color="#a6a6a6",
        linewidth=1.0,
        alpha=0.65,
    )
    ax.scatter(
        sub["mean_deviation"].to_numpy(dtype=float),
        sub["rank"].to_numpy(dtype=float),
        s=np.sqrt(sub["count"].to_numpy(dtype=float)) * 8.0,
        color="#153b50",
        edgecolor="white",
        linewidth=0.6,
        alpha=0.95,
    )
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Region mean - overall mean")
    ax.set_title("Region-level heterogeneity", fontsize=12.5, fontweight="bold")
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.22)

    low_extremes = sub.head(3).copy()
    high_extremes = sub.tail(3).copy()
    low_offsets = [-0.7, 0.0, 0.7]
    high_offsets = [-0.7, 0.0, 0.7]
    for (_, row), y_shift in zip(low_extremes.iterrows(), low_offsets):
        ax.text(
            float(row["mean_deviation"]) + (0.18 if row["mean_deviation"] >= 0 else -0.18),
            float(row["rank"]) + y_shift,
            f"{row['region']} (n={int(row['count'])})",
            ha="left" if row["mean_deviation"] >= 0 else "right",
            va="center",
            fontsize=8.4,
            color="#222222",
        )
    for (_, row), y_shift in zip(high_extremes.iterrows(), high_offsets):
        ax.text(
            float(row["mean_deviation"]) + (0.18 if row["mean_deviation"] >= 0 else -0.18),
            float(row["rank"]) + y_shift,
            f"{row['region']} (n={int(row['count'])})",
            ha="left" if row["mean_deviation"] >= 0 else "right",
            va="center",
            fontsize=8.4,
            color="#222222",
        )
    ax.text(
        0.02,
        0.04,
        "Point size scales with state sample size",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
        color="#333333",
    )


def _write_summary_json(
    out_dir: Path,
    frame: pd.DataFrame,
    period_summary: pd.DataFrame,
    cat_summary: pd.DataFrame,
    region_summary: pd.DataFrame,
) -> None:
    overall_mean = float(frame[TARGET_COL].mean())
    summary = {
        "n_rows": int(frame.shape[0]),
        "overall_mean": overall_mean,
        "overall_std": float(frame[TARGET_COL].std(ddof=1)),
        "target_quantiles": {
            "q25": float(frame[TARGET_COL].quantile(0.25)),
            "q50": float(frame[TARGET_COL].quantile(0.50)),
            "q75": float(frame[TARGET_COL].quantile(0.75)),
        },
        "period_min_mean": period_summary.sort_values("mean").head(1).to_dict(orient="records")[0],
        "period_max_mean": period_summary.sort_values("mean", ascending=False).head(1).to_dict(orient="records")[0],
        "largest_positive_category_deviation": cat_summary.sort_values("mean_deviation", ascending=False).head(5).to_dict(orient="records"),
        "largest_negative_category_deviation": cat_summary.sort_values("mean_deviation", ascending=True).head(5).to_dict(orient="records"),
        "largest_positive_region_deviation": region_summary.sort_values("mean_deviation", ascending=False).head(5).to_dict(orient="records"),
        "largest_negative_region_deviation": region_summary.sort_values("mean_deviation", ascending=True).head(5).to_dict(orient="records"),
        "cli_correlation": float(frame[[SPLINE_COLUMNS[0], TARGET_COL]].corr().iloc[0, 1]),
        "hh_cmnty_cli_correlation": float(frame[[SPLINE_COLUMNS[1], TARGET_COL]].corr().iloc[0, 1]),
    }
    (out_dir / "eda_summary.json").write_text(
        json.dumps(_to_serializable(summary), indent=2),
        encoding="utf-8",
    )


def run_eda(processed_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_frame(processed_dir)
    spline_meta = _load_json(processed_dir / "analysis_bundle" / "spline_metadata.json")

    period_summary = _period_summary(frame)
    cat_summary = _categorical_summary(frame)
    region_summary = _region_summary(frame)
    overall_mean = float(frame[TARGET_COL].mean())

    cat_summary.to_csv(out_dir / "covid_categorical_summary.csv", index=False)
    region_summary.to_csv(out_dir / "covid_region_summary.csv", index=False)
    period_summary.to_csv(out_dir / "covid_period_summary.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(16.2, 10.2), constrained_layout=False)
    _plot_target_distribution(axes[0, 0], frame)
    _plot_period_trend(axes[0, 1], period_summary)
    _plot_spline_panel(axes[0, 2], frame, spline_meta["cli"], "cli")
    _plot_spline_panel(axes[1, 0], frame, spline_meta["hh_cmnty_cli"], "hh_cmnty_cli")
    _plot_categorical_deviation(axes[1, 1], cat_summary, overall_mean)
    _plot_region_heterogeneity(axes[1, 2], region_summary)

    fig.suptitle(
        "COVID-19 Trust in Experts: exploratory data overview",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.02,
        0.02,
        f"n = {frame.shape[0]} complete observations. The panels summarize the outcome distribution, temporal drift, "
        "nonlinear illness covariates, subgroup mean differences, and state-level heterogeneity.",
        fontsize=10,
        color="#222222",
    )
    plt.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.08, wspace=0.24, hspace=0.28)
    fig.savefig(out_dir / "covid_eda_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    _write_summary_json(out_dir, frame, period_summary, cat_summary, region_summary)
    print(f"[ok] EDA artifacts written to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EDA artifacts for the COVID trust_experts dataset.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/real/covid19_trust_experts/processed"),
        help="Processed dataset directory containing analysis_bundle and runner_ready outputs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/reports/covid19_trust_experts_eda"),
        help="Destination directory for EDA figures and tables.",
    )
    args = parser.parse_args()
    run_eda(args.processed_dir, args.out_dir)


if __name__ == "__main__":
    main()
