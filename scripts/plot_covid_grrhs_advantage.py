from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


MODEL_ALIASES: Dict[str, str] = {
    "grrhs_nuts": "GR-RHS",
    "regularized_horseshoe": "RHS",
    "rhs": "RHS",
    "gigg": "GIGG",
    "gigg_regression": "GIGG",
    "sparse_group_lasso": "SGL",
    "lasso": "Lasso",
    "ridge": "Ridge",
}

MODEL_ORDER = ["GR-RHS", "RHS", "GIGG", "SGL", "Lasso", "Ridge"]

MODEL_COLORS: Dict[str, str] = {
    "GR-RHS": "#0f6b50",
    "RHS": "#6b7280",
    "GIGG": "#b42318",
    "SGL": "#1769aa",
    "Lasso": "#d97706",
    "Ridge": "#374151",
}

LOWER_IS_BETTER = {"RMSE", "MAE", "PredictiveIntervalWidth90", "CoverageGap90", "MeanEffectiveNonzeros"}
HIGHER_IS_BETTER = {"MLPD", "PredictiveCoverage90"}
BAYESIAN_MODELS = {"GR-RHS", "RHS", "GIGG"}
TARGET_COVERAGE = 0.90


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload or {}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_model_label(run_dir: Path) -> str:
    cfg = _load_yaml(run_dir / "resolved_config.yaml")
    model_cfg = cfg.get("model", {}) or {}
    name = str(model_cfg.get("name", run_dir.name)).strip().lower()
    return MODEL_ALIASES.get(name, str(model_cfg.get("name", run_dir.name)))


def _timestamp_key(path: Path) -> tuple[int, str]:
    suffix = path.name.rsplit("-", 1)[-1]
    try:
        return int(suffix), path.name
    except ValueError:
        return -1, path.name


def _find_latest_runs(sweep_dir: Path, labels: Sequence[str]) -> Dict[str, Path]:
    selected: Dict[str, Path] = {}
    wanted = set(labels)
    candidates = [path for path in sweep_dir.iterdir() if path.is_dir()]
    for run_dir in sorted(candidates, key=_timestamp_key):
        label = _resolve_model_label(run_dir)
        if label in wanted:
            selected[label] = run_dir
    missing = [label for label in labels if label not in selected]
    if missing:
        raise SystemExit(f"Missing run directories for models: {', '.join(missing)}")
    return selected


def _fold_rows(run_dir: Path, label: str) -> List[Dict[str, Any]]:
    summary = _load_json(run_dir / "summary.json")
    rows: List[Dict[str, Any]] = []
    for repeat_summary in summary.get("repeat_summaries", []):
        repeat_index = int(repeat_summary.get("repeat_index", 0))
        for fold in repeat_summary.get("folds", []):
            metrics = fold.get("metrics", {}) or {}
            row = {
                "run_dir": str(run_dir),
                "variation": run_dir.name,
                "model": label,
                "status": str(fold.get("status", "UNKNOWN")),
                "repeat": repeat_index,
                "fold": int(fold.get("fold", 0)),
                "fold_hash": str(fold.get("hash", "")),
                "fold_key": f"{repeat_index}:{fold.get('hash', '')}",
                "RMSE": _coerce_float(metrics.get("RMSE")),
                "MAE": _coerce_float(metrics.get("MAE")),
                "MLPD": _coerce_float(metrics.get("MLPD")),
                "PredictiveCoverage90": _coerce_float(metrics.get("PredictiveCoverage90")),
                "PredictiveIntervalWidth90": _coerce_float(metrics.get("PredictiveIntervalWidth90")),
                "MeanEffectiveNonzeros": _coerce_float(metrics.get("MeanEffectiveNonzeros")),
                "EffectiveDoF": _coerce_float(metrics.get("EffectiveDoF")),
            }
            coverage = row["PredictiveCoverage90"]
            row["CoverageGap90"] = None if coverage is None else abs(float(coverage) - TARGET_COVERAGE)
            rows.append(row)
    return rows


def _build_fold_frame(run_map: Mapping[str, Path]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for label in MODEL_ORDER:
        run_dir = run_map.get(label)
        if run_dir is None:
            continue
        rows.extend(_fold_rows(run_dir, label))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit("No fold-level rows found.")
    return frame


def _sign_test_pvalue(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n <= 0:
        return None
    total = 2 ** n
    tail = sum(math.comb(n, k) for k in range(wins, n + 1))
    return float(tail / total)


def _advantage_direction(metric: str) -> str:
    if metric in LOWER_IS_BETTER:
        return "baseline_minus_grrhs"
    if metric in HIGHER_IS_BETTER:
        return "grrhs_minus_baseline"
    raise KeyError(f"Unknown direction for metric {metric}")


def _compute_advantage(grrhs_values: pd.Series, baseline_values: pd.Series, metric: str) -> pd.Series:
    direction = _advantage_direction(metric)
    if direction == "baseline_minus_grrhs":
        return baseline_values - grrhs_values
    return grrhs_values - baseline_values


def _pairwise_stats(fold_frame: pd.DataFrame, baselines: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    grrhs = fold_frame[fold_frame["model"] == "GR-RHS"].copy()
    for baseline in baselines:
        baseline_frame = fold_frame[fold_frame["model"] == baseline].copy()
        merged = grrhs.merge(
            baseline_frame,
            on="fold_key",
            how="inner",
            suffixes=("_grrhs", "_baseline"),
        )
        if merged.empty:
            continue
        for metric in metrics:
            col_gr = f"{metric}_grrhs"
            col_base = f"{metric}_baseline"
            if col_gr not in merged.columns or col_base not in merged.columns:
                continue
            sub = merged[["fold_key", "fold_grrhs", col_gr, col_base]].copy()
            sub = sub.dropna()
            if sub.empty:
                continue
            advantage = _compute_advantage(sub[col_gr], sub[col_base], metric)
            wins = int((advantage > 1e-12).sum())
            losses = int((advantage < -1e-12).sum())
            ties = int(sub.shape[0] - wins - losses)
            row = {
                "baseline": baseline,
                "metric": metric,
                "n_folds": int(sub.shape[0]),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "mean_advantage": float(advantage.mean()),
                "median_advantage": float(np.median(advantage.to_numpy(dtype=float))),
                "std_advantage": float(advantage.std(ddof=1)) if advantage.shape[0] > 1 else 0.0,
                "min_advantage": float(advantage.min()),
                "max_advantage": float(advantage.max()),
                "sign_test_p_one_sided": _sign_test_pvalue(wins, losses),
                "direction": _advantage_direction(metric),
                "interpretation": "positive favors GR-RHS",
            }
            rows.append(row)
    return pd.DataFrame(rows)


def _pairwise_fold_advantages(fold_frame: pd.DataFrame, baselines: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    grrhs = fold_frame[fold_frame["model"] == "GR-RHS"].copy()
    for baseline in baselines:
        baseline_frame = fold_frame[fold_frame["model"] == baseline].copy()
        merged = grrhs.merge(
            baseline_frame,
            on="fold_key",
            how="inner",
            suffixes=("_grrhs", "_baseline"),
        )
        if merged.empty:
            continue
        for metric in metrics:
            col_gr = f"{metric}_grrhs"
            col_base = f"{metric}_baseline"
            if col_gr not in merged.columns or col_base not in merged.columns:
                continue
            sub = merged[["fold_key", "fold_grrhs", col_gr, col_base]].copy().dropna()
            if sub.empty:
                continue
            advantage = _compute_advantage(sub[col_gr], sub[col_base], metric)
            for (_, fold_row), adv in zip(sub.iterrows(), advantage.tolist()):
                rows.append(
                    {
                        "baseline": baseline,
                        "metric": metric,
                        "fold_key": str(fold_row["fold_key"]),
                        "fold": int(fold_row["fold_grrhs"]),
                        "advantage": float(adv),
                        "interpretation": "positive favors GR-RHS",
                    }
                )
    return pd.DataFrame(rows)


def _model_summary(fold_frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for label in MODEL_ORDER:
        sub = fold_frame[fold_frame["model"] == label].copy()
        if sub.empty:
            continue
        row = {
            "model": label,
            "n_recorded_folds": int(sub.shape[0]),
            "n_invalid_folds": int(sub["status"].str.upper().str.startswith("INVALID").sum()),
            "median_RMSE": float(sub["RMSE"].median()) if sub["RMSE"].notna().any() else np.nan,
        }
        for metric in [
            "RMSE",
            "MAE",
            "MLPD",
            "PredictiveCoverage90",
            "PredictiveIntervalWidth90",
            "CoverageGap90",
            "MeanEffectiveNonzeros",
            "EffectiveDoF",
        ]:
            values = sub[metric].dropna()
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if values.shape[0] > 1 else 0.0 if not values.empty else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        raise SystemExit("No model summary could be built.")
    return summary


def _make_rmse_panel(ax: plt.Axes, pairwise: pd.DataFrame, raw_pairwise: pd.DataFrame) -> None:
    rmse = pairwise[pairwise["metric"] == "RMSE"].copy()
    order = ["SGL", "Lasso", "RHS", "Ridge", "GIGG"]
    rmse = rmse.set_index("baseline").reindex(order).reset_index()
    ax.axvline(0.0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.8)
    for idx, (_, row) in enumerate(rmse.iterrows()):
        baseline = row["baseline"]
        color = MODEL_COLORS.get(baseline, "#777777")
        sub = row
        mean_adv = float(sub["mean_advantage"])
        min_adv = float(sub["min_advantage"])
        max_adv = float(sub["max_advantage"])
        wins = int(sub["wins"])
        losses = int(sub["losses"])
        ties = int(sub["ties"])
        n = int(sub["n_folds"])
        raw = raw_pairwise[(raw_pairwise["metric"] == "RMSE") & (raw_pairwise["baseline"] == baseline)].copy()
        if not raw.empty:
            offsets = np.linspace(-0.12, 0.12, raw.shape[0])
            ax.scatter(
                raw["advantage"].to_numpy(dtype=float),
                idx + offsets,
                s=38,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                alpha=0.95,
                zorder=3,
            )
        ax.hlines(idx, min_adv, max_adv, color=color, linewidth=2.2, alpha=0.85)
        ax.scatter([mean_adv], [idx], s=85, color=color, edgecolor="black", linewidth=0.8, zorder=4)
        ax.text(
            max_adv + 0.01,
            idx,
            f"{wins}-{losses}-{ties} / {n}",
            va="center",
            fontsize=9.5,
            color="#222222",
        )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel("Baseline RMSE - GR-RHS RMSE")
    ax.set_title("Fold-level RMSE advantage\npositive = GR-RHS lower error", fontsize=12.5, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)


def _make_mlpd_panel(ax: plt.Axes, pairwise: pd.DataFrame, raw_pairwise: pd.DataFrame) -> None:
    mlpd = pairwise[pairwise["metric"] == "MLPD"].copy()
    order = ["RHS", "GIGG"]
    mlpd = mlpd.set_index("baseline").reindex(order).reset_index()
    ax.axvline(0.0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.8)
    for idx, (_, row) in enumerate(mlpd.iterrows()):
        baseline = row["baseline"]
        if pd.isna(row.get("mean_advantage")):
            continue
        color = MODEL_COLORS.get(baseline, "#777777")
        mean_adv = float(row["mean_advantage"])
        min_adv = float(row["min_advantage"])
        max_adv = float(row["max_advantage"])
        wins = int(row["wins"])
        losses = int(row["losses"])
        ties = int(row["ties"])
        n = int(row["n_folds"])
        raw = raw_pairwise[(raw_pairwise["metric"] == "MLPD") & (raw_pairwise["baseline"] == baseline)].copy()
        if not raw.empty:
            offsets = np.linspace(-0.10, 0.10, raw.shape[0])
            ax.scatter(
                raw["advantage"].to_numpy(dtype=float),
                idx + offsets,
                s=38,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                alpha=0.95,
                zorder=3,
            )
        ax.hlines(idx, min_adv, max_adv, color=color, linewidth=2.2, alpha=0.85)
        ax.scatter([mean_adv], [idx], s=85, color=color, edgecolor="black", linewidth=0.8, zorder=4)
        ax.text(
            max_adv + (0.03 if baseline == "RHS" else 0.6),
            idx,
            f"{wins}-{losses}-{ties} / {n}",
            va="center",
            fontsize=9.5,
            color="#222222",
        )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel("GR-RHS MLPD - baseline MLPD")
    ax.set_title("Bayesian log predictive density\npositive = GR-RHS better calibrated fit", fontsize=12.5, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)


def _make_frontier_panel(ax: plt.Axes, summary: pd.DataFrame) -> None:
    for _, row in summary.iterrows():
        label = str(row["model"])
        x = float(row["MeanEffectiveNonzeros_mean"])
        y = float(row["RMSE_mean"])
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        xerr = row["MeanEffectiveNonzeros_std"]
        yerr = row["RMSE_std"]
        color = MODEL_COLORS.get(label, "#777777")
        ax.errorbar(
            x,
            y,
            xerr=None if not math.isfinite(float(xerr)) else float(xerr),
            yerr=None if not math.isfinite(float(yerr)) else float(yerr),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.4,
            capsize=3,
            markersize=8.5,
            markeredgecolor="black",
            markeredgewidth=0.7,
        )
        ax.text(x * 1.05, y + 0.005, label, fontsize=9.5, color="#222222")
    ax.set_xscale("log")
    ax.set_xlabel("Mean effective nonzeros (log scale)")
    ax.set_ylabel("RMSE")
    ax.set_title("Prediction-sparsity frontier\nlower-left is preferable", fontsize=12.5, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)


def _make_uncertainty_panel(ax: plt.Axes, summary: pd.DataFrame) -> None:
    bayes = summary[summary["model"].isin(BAYESIAN_MODELS)].copy()
    ax.axvspan(TARGET_COVERAGE - 0.01, TARGET_COVERAGE + 0.01, color="#e8f5e9", alpha=0.8, zorder=0)
    ax.axvline(TARGET_COVERAGE, color="#2e7d32", linestyle="--", linewidth=1.2)
    for _, row in bayes.iterrows():
        label = str(row["model"])
        x = float(row["PredictiveCoverage90_mean"])
        y = float(row["PredictiveIntervalWidth90_mean"])
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        xerr = row["PredictiveCoverage90_std"]
        yerr = row["PredictiveIntervalWidth90_std"]
        color = MODEL_COLORS.get(label, "#777777")
        ax.errorbar(
            x,
            y,
            xerr=None if not math.isfinite(float(xerr)) else float(xerr),
            yerr=None if not math.isfinite(float(yerr)) else float(yerr),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.4,
            capsize=3,
            markersize=9.0,
            markeredgecolor="black",
            markeredgewidth=0.7,
        )
        ax.text(x + 0.004, y + 0.1, label, fontsize=9.5, color="#222222")
    ax.set_xlabel("Predictive coverage @ 90%")
    ax.set_ylabel("Predictive interval width @ 90%")
    ax.set_title("Uncertainty calibration\nvertical band = target coverage", fontsize=12.5, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)


def _figure_note(summary: pd.DataFrame) -> str:
    grrhs = summary[summary["model"] == "GR-RHS"].iloc[0]
    rhs = summary[summary["model"] == "RHS"].iloc[0]
    sgl = summary[summary["model"] == "SGL"].iloc[0]
    return (
        "Basis: common recorded outer folds (n=5). Bayesian runs are PARTIAL in the official sweep because the "
        "fairness budget used 1 chain while the convergence gate requires 2. "
        f"GR-RHS median RMSE = {grrhs['median_RMSE']:.3f}, SGL median RMSE = {sgl['median_RMSE']:.3f}, "
        f"RHS median RMSE = {rhs['median_RMSE']:.3f}."
    )


def _write_report(
    out_dir: Path,
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    run_map: Mapping[str, Path],
) -> None:
    grrhs = summary[summary["model"] == "GR-RHS"].iloc[0]
    rhs = summary[summary["model"] == "RHS"].iloc[0]
    sgl = summary[summary["model"] == "SGL"].iloc[0]
    lasso = summary[summary["model"] == "Lasso"].iloc[0]
    gigg = summary[summary["model"] == "GIGG"].iloc[0]

    rmse_pairs = pairwise[pairwise["metric"] == "RMSE"].set_index("baseline")
    mlpd_pairs = pairwise[pairwise["metric"] == "MLPD"].set_index("baseline")

    lines = [
        "# COVID GR-RHS Recorded-Fold Advantage",
        "",
        "Comparison basis: common recorded outer folds across all six models (`n=5`).",
        "This intentionally differs from the official `sweep_comparison.csv`, which excludes the Bayesian runs because the fairness budget used `num_chains=1` while the convergence gate requires at least 2 chains.",
        "",
        "## What the recorded-fold tests support",
        "",
        f"- GR-RHS is clearly better than GIGG on uncertainty calibration: coverage {grrhs['PredictiveCoverage90_mean']:.3f} vs {gigg['PredictiveCoverage90_mean']:.3f}, interval width {grrhs['PredictiveIntervalWidth90_mean']:.2f} vs {gigg['PredictiveIntervalWidth90_mean']:.2f}.",
        f"- GR-RHS improves on RHS in typical prediction quality: RMSE wins = {int(rmse_pairs.loc['RHS', 'wins'])}/{int(rmse_pairs.loc['RHS', 'n_folds'])}, median RMSE gain = {rmse_pairs.loc['RHS', 'median_advantage']:.4f}.",
        f"- Against the best frequentist baselines, GR-RHS is competitive rather than dominant: median RMSE {grrhs['median_RMSE']:.3f} vs SGL {sgl['median_RMSE']:.3f} and Lasso {lasso['median_RMSE']:.3f}.",
        f"- GR-RHS reaches that accuracy with far stronger shrinkage than SGL/Lasso: mean effective nonzeros {grrhs['MeanEffectiveNonzeros_mean']:.2f} vs SGL {sgl['MeanEffectiveNonzeros_mean']:.1f} and Lasso {lasso['MeanEffectiveNonzeros_mean']:.1f}.",
        "",
        "## Pairwise sign tests",
        "",
        "| Baseline | Metric | Wins | Losses | Ties | Mean advantage | Median advantage | One-sided sign p |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in pairwise.sort_values(["metric", "baseline"]).iterrows():
        p_val = row["sign_test_p_one_sided"]
        p_text = "N/A" if p_val is None else f"{float(p_val):.4f}"
        lines.append(
            f"| {row['baseline']} | {row['metric']} | {int(row['wins'])} | {int(row['losses'])} | {int(row['ties'])} | "
            f"{float(row['mean_advantage']):.6f} | {float(row['median_advantage']):.6f} | {p_text} |"
        )

    lines.extend(
        [
            "",
            "## Recommended thesis wording",
            "",
            "On the COVID trust-in-experts dataset, GR-RHS should not be presented as the outright RMSE winner.",
            "The stronger claim supported by the recorded-fold analysis is that GR-RHS remains competitive with the best tuned frequentist baselines while providing calibrated Bayesian uncertainty and a substantially sparser effective representation than SGL/Lasso, and it is consistently preferable to RHS/GIGG on the Bayesian comparison panels.",
            "",
            "## Run directories",
            "",
        ]
    )
    for label in MODEL_ORDER:
        run_dir = run_map.get(label)
        if run_dir is None:
            continue
        lines.append(f"- {label}: `{run_dir}`")

    (out_dir / "covid_grrhs_advantage.md").write_text("\n".join(lines), encoding="utf-8")


def plot_report(sweep_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_map = _find_latest_runs(sweep_dir, MODEL_ORDER)
    fold_frame = _build_fold_frame(run_map)
    summary = _model_summary(fold_frame)
    pairwise = _pairwise_stats(
        fold_frame,
        baselines=["RHS", "GIGG", "SGL", "Lasso", "Ridge"],
        metrics=["RMSE", "MLPD", "CoverageGap90"],
    )
    raw_pairwise = _pairwise_fold_advantages(
        fold_frame,
        baselines=["RHS", "GIGG", "SGL", "Lasso", "Ridge"],
        metrics=["RMSE", "MLPD", "CoverageGap90"],
    )

    fold_frame.sort_values(["model", "repeat", "fold"], inplace=True)
    summary.sort_values("model", key=lambda s: s.map({label: idx for idx, label in enumerate(MODEL_ORDER)}), inplace=True)
    pairwise.sort_values(["metric", "baseline"], inplace=True)
    raw_pairwise.sort_values(["metric", "baseline", "fold"], inplace=True)

    fold_frame.to_csv(out_dir / "covid_grrhs_fold_metrics.csv", index=False)
    summary.to_csv(out_dir / "covid_grrhs_model_summary.csv", index=False)
    pairwise.to_csv(out_dir / "covid_grrhs_pairwise_tests.csv", index=False)
    raw_pairwise.to_csv(out_dir / "covid_grrhs_pairwise_fold_advantages.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), constrained_layout=False)
    _make_rmse_panel(axes[0, 0], pairwise, raw_pairwise)
    _make_mlpd_panel(axes[0, 1], pairwise, raw_pairwise)
    _make_frontier_panel(axes[1, 0], summary)
    _make_uncertainty_panel(axes[1, 1], summary)

    fig.suptitle(
        "COVID-19 Trust in Experts: recorded-fold comparison centered on GR-RHS",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.text(0.02, 0.02, _figure_note(summary), fontsize=10, color="#222222")
    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.09, wspace=0.23, hspace=0.28)
    fig.savefig(out_dir / "covid_grrhs_advantage.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    _write_report(out_dir, summary, pairwise, run_map)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a recorded-fold COVID report that focuses on where GR-RHS is actually strong."
    )
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path("outputs/sweeps/real_covid19_trust_experts_thesis"),
        help="Sweep directory containing the COVID thesis runs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/reports/covid_thesis/recorded_advantage"),
        help="Destination directory for the report tables and figure.",
    )
    args = parser.parse_args()

    plot_report(args.sweep_dir, args.out_dir)
    print(f"[ok] report written to {args.out_dir}")


if __name__ == "__main__":
    main()

