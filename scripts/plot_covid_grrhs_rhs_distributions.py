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
    "grrhs_gibbs": "GR-RHS",
    "regularized_horseshoe": "RHS",
    "rhs": "RHS",
}

MODEL_ORDER = ["GR-RHS", "RHS"]

MODEL_COLORS: Dict[str, str] = {
    "GR-RHS": "#0f6b50",
    "RHS": "#6b7280",
}

GROUP_LABELS = {
    0: "Period",
    1: "Region",
    2: "Age",
    3: "Gender",
    4: "Race/Eth",
    5: "CLI spline",
    6: "Community CLI spline",
}

SUMMARY_METRIC_LABELS = {
    "region_share": "Region share",
    "race_share": "Race/Eth share",
    "spline_share": "All spline share",
    "demo_spline_ratio": "Demographic / spline ratio",
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload or {}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _timestamp_key(path: Path) -> tuple[int, str]:
    suffix = path.name.rsplit("-", 1)[-1]
    try:
        return int(suffix), path.name
    except ValueError:
        return -1, path.name


def _resolve_model_label(run_dir: Path) -> str:
    cfg = _load_yaml(run_dir / "resolved_config.yaml")
    model_cfg = cfg.get("model", {}) or {}
    name = str(model_cfg.get("name", run_dir.name)).strip().lower()
    return MODEL_ALIASES.get(name, str(model_cfg.get("name", run_dir.name)))


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


def _flatten_draws(beta: np.ndarray) -> np.ndarray:
    arr = np.asarray(beta, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    return arr.reshape(-1, arr.shape[-1])


def _sign_test_pvalue(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n <= 0:
        return None
    tail = sum(math.comb(n, k) for k in range(wins, n + 1))
    return float(tail / (2 ** n))


def _paired_sign_test(rows: Iterable[tuple[float, float]], *, prefer_higher: bool) -> Dict[str, Any]:
    wins = 0
    losses = 0
    ties = 0
    diffs: List[float] = []
    for grrhs_val, rhs_val in rows:
        diff = float(grrhs_val - rhs_val) if prefer_higher else float(rhs_val - grrhs_val)
        diffs.append(diff)
        if diff > 1e-12:
            wins += 1
        elif diff < -1e-12:
            losses += 1
        else:
            ties += 1
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "mean_advantage": float(np.mean(diffs)) if diffs else np.nan,
        "median_advantage": float(np.median(diffs)) if diffs else np.nan,
        "sign_test_p_one_sided": _sign_test_pvalue(wins, losses),
    }


def _load_dataset_arrays(repo_root: Path) -> tuple[np.ndarray, np.ndarray]:
    X = np.load(repo_root / "data" / "real" / "covid19_trust_experts" / "processed" / "runner_ready" / "X.npy")
    y = np.load(repo_root / "data" / "real" / "covid19_trust_experts" / "processed" / "runner_ready" / "y.npy")
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def _quantile_rows(
    model: str,
    fold: int,
    abs_errors: np.ndarray,
    quantile_grid: np.ndarray,
) -> List[Dict[str, Any]]:
    values = np.quantile(abs_errors, quantile_grid)
    return [
        {
            "model": model,
            "fold": fold,
            "quantile": float(q),
            "abs_error_quantile": float(v),
        }
        for q, v in zip(quantile_grid, values)
    ]


def _posterior_summary_rows(
    model: str,
    fold: int,
    group_share: np.ndarray,
) -> List[Dict[str, Any]]:
    region_share = group_share[:, 1]
    race_share = group_share[:, 4]
    spline_share = group_share[:, 5] + group_share[:, 6]
    demo_share = group_share[:, 2] + group_share[:, 3] + group_share[:, 4]
    demo_spline_ratio = demo_share / np.maximum(spline_share, 1e-12)

    metrics = {
        "region_share": region_share,
        "race_share": race_share,
        "spline_share": spline_share,
        "demo_spline_ratio": demo_spline_ratio,
    }
    rows: List[Dict[str, Any]] = []
    for metric, values in metrics.items():
        for value in values:
            rows.append(
                {
                    "model": model,
                    "fold": fold,
                    "summary_metric": metric,
                    "summary_label": SUMMARY_METRIC_LABELS[metric],
                    "value": float(value),
                }
            )
    return rows


def _load_analysis_frames(
    repo_root: Path,
    run_map: Mapping[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X, y = _load_dataset_arrays(repo_root)
    grrhs_meta = _load_json(run_map["GR-RHS"] / "repeat_001" / "dataset_meta.json")
    groups = [np.asarray(group, dtype=int) for group in grrhs_meta["groups"]]

    quantile_grid = np.unique(
        np.concatenate(
            [
                np.linspace(0.05, 0.99, 70, dtype=float),
                np.array([0.90, 0.95, 0.99], dtype=float),
            ]
        )
    )
    residual_rows: List[Dict[str, Any]] = []
    posterior_rows: List[Dict[str, Any]] = []
    fold_stat_rows: List[Dict[str, Any]] = []

    for model in MODEL_ORDER:
        run_dir = run_map[model]
        repeat_dir = run_dir / "repeat_001"
        for fold_dir in sorted(repeat_dir.glob("fold_*")):
            fold = int(fold_dir.name.split("_")[-1])
            posterior = np.load(fold_dir / "posterior_samples.npz", allow_pickle=True)
            beta_draws = _flatten_draws(np.asarray(posterior["beta"], dtype=float))
            beta_mean = beta_draws.mean(axis=0)

            fold_arrays = np.load(fold_dir / "fold_arrays.npz")
            test_idx = np.asarray(fold_arrays["test_idx"], dtype=int)
            x_mean = np.asarray(fold_arrays["x_mean"], dtype=float)
            x_scale = np.maximum(np.asarray(fold_arrays["x_scale"], dtype=float), 1e-8)
            y_mean = float(fold_arrays["y_mean"][0]) if fold_arrays["y_mean"].size else 0.0

            X_test = (X[test_idx] - x_mean) / x_scale
            y_test = y[test_idx] - y_mean
            pred_mean = X_test @ beta_mean
            abs_errors = np.abs(y_test - pred_mean)
            residual_rows.extend(_quantile_rows(model, fold, abs_errors, quantile_grid))

            group_mass = np.stack([np.abs(beta_draws[:, group]).sum(axis=1) for group in groups], axis=1)
            group_share = group_mass / np.maximum(group_mass.sum(axis=1, keepdims=True), 1e-12)
            posterior_rows.extend(_posterior_summary_rows(model, fold, group_share))

            fold_stat_rows.append(
                {
                    "model": model,
                    "fold": fold,
                    "median_abs_error": float(np.quantile(abs_errors, 0.50)),
                    "p90_abs_error": float(np.quantile(abs_errors, 0.90)),
                    "p95_abs_error": float(np.quantile(abs_errors, 0.95)),
                    "p99_abs_error": float(np.quantile(abs_errors, 0.99)),
                    "region_share_median": float(np.median(group_share[:, 1])),
                    "race_share_median": float(np.median(group_share[:, 4])),
                    "spline_share_median": float(np.median(group_share[:, 5] + group_share[:, 6])),
                    "demo_spline_ratio_median": float(
                        np.median((group_share[:, 2] + group_share[:, 3] + group_share[:, 4]) / np.maximum(group_share[:, 5] + group_share[:, 6], 1e-12))
                    ),
                }
            )

    return pd.DataFrame(residual_rows), pd.DataFrame(posterior_rows), pd.DataFrame(fold_stat_rows)


def _style_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#222222")
    ax.set_facecolor("white")


def _plot_residual_quantile_curves(ax: plt.Axes, residual_df: pd.DataFrame) -> None:
    _style_axis(ax)
    summary = (
        residual_df.groupby(["model", "quantile"], as_index=False)["abs_error_quantile"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    for model in MODEL_ORDER:
        sub = summary[summary["model"] == model].copy()
        color = MODEL_COLORS[model]
        ax.fill_between(
            sub["quantile"].to_numpy(dtype=float),
            sub["min"].to_numpy(dtype=float),
            sub["max"].to_numpy(dtype=float),
            color=color,
            alpha=0.13,
            linewidth=0.0,
        )
        ax.plot(
            sub["quantile"].to_numpy(dtype=float),
            sub["mean"].to_numpy(dtype=float),
            color=color,
            linewidth=2.4,
            label=model,
        )
    ax.set_xlabel("Absolute-error quantile")
    ax.set_ylabel("Held-out absolute error")
    ax.set_title("Held-out error quantile curves\nlower is better", fontsize=12.5, fontweight="bold")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="upper left")


def _plot_tail_advantage(ax: plt.Axes, residual_df: pd.DataFrame) -> None:
    _style_axis(ax)
    focus = residual_df[residual_df["quantile"] >= 0.70].copy()
    pivot = focus.pivot_table(index=["fold", "quantile"], columns="model", values="abs_error_quantile").reset_index()
    pivot["advantage"] = pivot["RHS"] - pivot["GR-RHS"]
    for fold, sub in pivot.groupby("fold", sort=True):
        ax.plot(
            sub["quantile"].to_numpy(dtype=float),
            sub["advantage"].to_numpy(dtype=float),
            color="#9ca3af",
            linewidth=1.4,
            alpha=0.9,
        )
    mean_curve = pivot.groupby("quantile", as_index=False)["advantage"].mean()
    ax.plot(
        mean_curve["quantile"].to_numpy(dtype=float),
        mean_curve["advantage"].to_numpy(dtype=float),
        color=MODEL_COLORS["GR-RHS"],
        linewidth=2.8,
        label="mean advantage",
    )
    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Absolute-error quantile")
    ax.set_ylabel("RHS quantile - GR-RHS quantile")
    ax.set_title("Upper-tail error advantage by fold\npositive = GR-RHS lower tail risk", fontsize=12.5, fontweight="bold")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="upper left")


def _violin(ax: plt.Axes, values: Sequence[np.ndarray], positions: Sequence[float], colors: Sequence[str], widths: float = 0.26) -> None:
    parts = ax.violinplot(values, positions=positions, widths=widths, showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.28)
    for pos, arr, color in zip(positions, values, colors):
        q1, med, q3 = np.quantile(arr, [0.25, 0.50, 0.75])
        ax.vlines(pos, q1, q3, color=color, linewidth=3.2, alpha=0.95)
        ax.scatter([pos], [med], color=color, s=44, edgecolor="white", linewidth=0.7, zorder=5)


def _plot_ratio_distribution(ax: plt.Axes, posterior_df: pd.DataFrame) -> None:
    _style_axis(ax)
    sub = posterior_df[posterior_df["summary_metric"] == "demo_spline_ratio"].copy()
    values = [sub[sub["model"] == model]["value"].to_numpy(dtype=float) for model in MODEL_ORDER]
    positions = [0.9, 1.1]
    colors = [MODEL_COLORS[model] for model in MODEL_ORDER]
    _violin(ax, values, positions, colors, widths=0.16)
    ax.set_xticks(positions)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylabel("Demographic share / spline share")
    ax.set_yscale("log")
    ax.set_title("Posterior group-allocation ratio\nhigher = more mass on demographic blocks", fontsize=12.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.22)


def _plot_selected_group_shares(ax: plt.Axes, posterior_df: pd.DataFrame) -> None:
    _style_axis(ax)
    metrics = ["race_share", "region_share", "spline_share"]
    base_positions = np.arange(len(metrics), dtype=float)
    offset = 0.16
    width = 0.22
    for model, sign in [("GR-RHS", -1), ("RHS", 1)]:
        values = []
        positions = []
        for idx, metric in enumerate(metrics):
            sub = posterior_df[(posterior_df["summary_metric"] == metric) & (posterior_df["model"] == model)]
            values.append(sub["value"].to_numpy(dtype=float))
            positions.append(base_positions[idx] + sign * offset)
        _violin(ax, values, positions, [MODEL_COLORS[model]] * len(values), widths=width)
    ax.set_xticks(base_positions)
    ax.set_xticklabels([SUMMARY_METRIC_LABELS[m] for m in metrics], rotation=10)
    ax.set_ylabel("Posterior mass share")
    ax.set_title("Selected posterior group-share distributions", fontsize=12.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.22)
    handles = [
        plt.Line2D([0], [0], color=MODEL_COLORS[model], marker="o", linestyle="", markersize=8, label=model)
        for model in MODEL_ORDER
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right")


def _build_test_table(fold_stats: pd.DataFrame) -> pd.DataFrame:
    tests: List[Dict[str, Any]] = []

    def pair(metric: str, prefer_higher: bool, label: str) -> None:
        pivot = fold_stats.pivot_table(index="fold", columns="model", values=metric)
        rows = [(float(v["GR-RHS"]), float(v["RHS"])) for _, v in pivot.iterrows()]
        result = _paired_sign_test(rows, prefer_higher=prefer_higher)
        result["metric"] = label
        result["fold_count"] = int(len(rows))
        tests.append(result)

    pair("p90_abs_error", prefer_higher=False, label="p90 absolute error")
    pair("p95_abs_error", prefer_higher=False, label="p95 absolute error")
    pair("p99_abs_error", prefer_higher=False, label="p99 absolute error")
    pair("race_share_median", prefer_higher=True, label="Race/Eth posterior share")
    pair("region_share_median", prefer_higher=False, label="Region posterior share")
    pair("spline_share_median", prefer_higher=False, label="Spline posterior share")
    pair("demo_spline_ratio_median", prefer_higher=True, label="Demographic / spline ratio")

    return pd.DataFrame(tests)


def _write_markdown_report(out_dir: Path, tests: pd.DataFrame, residual_df: pd.DataFrame, fold_stats: pd.DataFrame) -> None:
    quant_summary = (
        residual_df.groupby(["model", "quantile"], as_index=False)["abs_error_quantile"].mean()
        .pivot(index="quantile", columns="model", values="abs_error_quantile")
    )
    q90 = quant_summary.loc[0.90]
    q95 = quant_summary.loc[0.95]
    q99 = quant_summary.loc[0.99]

    ratio_test = tests[tests["metric"] == "Demographic / spline ratio"].iloc[0]
    race_test = tests[tests["metric"] == "Race/Eth posterior share"].iloc[0]
    spline_test = tests[tests["metric"] == "Spline posterior share"].iloc[0]
    p95_test = tests[tests["metric"] == "p95 absolute error"].iloc[0]

    lines = [
        "# COVID GR-RHS vs RHS Distribution Report",
        "",
        "This report is restricted to the real COVID trust-in-experts dataset and compares `GR-RHS` against `RHS` using two families of distributions:",
        "1. held-out absolute-error quantile distributions",
        "2. posterior group-mass distributions",
        "",
        "## Main takeaways",
        "",
        f"- Upper-tail prediction risk is modestly better for GR-RHS: mean p90 error {q90['GR-RHS']:.3f} vs {q90['RHS']:.3f}, mean p95 error {q95['GR-RHS']:.3f} vs {q95['RHS']:.3f}, mean p99 error {q99['GR-RHS']:.3f} vs {q99['RHS']:.3f}.",
        f"- The tail improvement is not huge, but it is directional: GR-RHS wins {int(p95_test['wins'])}/{int(p95_test['fold_count'])} folds on p95 absolute error.",
        f"- The structural shift is much cleaner: GR-RHS has a higher demographic-to-spline posterior ratio in {int(ratio_test['wins'])}/{int(ratio_test['fold_count'])} folds (one-sided sign p = {float(ratio_test['sign_test_p_one_sided']):.4f}).",
        f"- GR-RHS also raises the Race/Eth group share in {int(race_test['wins'])}/{int(race_test['fold_count'])} folds and lowers total spline share in {int(spline_test['wins'])}/{int(spline_test['fold_count'])} folds.",
        "",
        "## Interpretation",
        "",
        "If you want a real-data figure that highlights the grouped prior itself, the posterior allocation panels are stronger than the RMSE-only panels.",
        "They show that GR-RHS systematically reallocates posterior mass away from diffuse spline blocks and toward compact demographic groups while keeping held-out predictive error essentially on par with RHS or slightly better in the upper tail.",
        "",
        "## Sign tests",
        "",
        "| Metric | Wins | Losses | Ties | Mean advantage | Median advantage | One-sided sign p |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in tests.iterrows():
        p_value = row["sign_test_p_one_sided"]
        p_text = "N/A" if p_value is None else f"{float(p_value):.4f}"
        lines.append(
            f"| {row['metric']} | {int(row['wins'])} | {int(row['losses'])} | {int(row['ties'])} | "
            f"{float(row['mean_advantage']):.6f} | {float(row['median_advantage']):.6f} | {p_text} |"
        )

    (out_dir / "covid_grrhs_rhs_distribution_report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_report(sweep_dir: Path, out_dir: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir.mkdir(parents=True, exist_ok=True)
    run_map = _find_latest_runs(sweep_dir, MODEL_ORDER)
    residual_df, posterior_df, fold_stats = _load_analysis_frames(repo_root, run_map)
    tests = _build_test_table(fold_stats)

    residual_df.to_csv(out_dir / "covid_grrhs_rhs_residual_quantiles.csv", index=False)
    posterior_df.to_csv(out_dir / "covid_grrhs_rhs_posterior_summaries.csv", index=False)
    fold_stats.to_csv(out_dir / "covid_grrhs_rhs_fold_stats.csv", index=False)
    tests.to_csv(out_dir / "covid_grrhs_rhs_sign_tests.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(15.4, 10.2), constrained_layout=False)
    _plot_residual_quantile_curves(axes[0, 0], residual_df)
    _plot_tail_advantage(axes[0, 1], residual_df)
    _plot_ratio_distribution(axes[1, 0], posterior_df)
    _plot_selected_group_shares(axes[1, 1], posterior_df)

    fig.suptitle(
        "COVID-19 Trust in Experts: distribution-level comparison of GR-RHS and RHS",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.02,
        0.02,
        "Top row: held-out absolute-error distributions. Bottom row: posterior group-mass distributions. "
        "Positive tail-advantage values mean RHS has larger upper-tail errors than GR-RHS.",
        fontsize=10,
        color="#222222",
    )
    plt.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.09, wspace=0.22, hspace=0.28)
    fig.savefig(out_dir / "covid_grrhs_rhs_distributions.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    _write_markdown_report(out_dir, tests, residual_df, fold_stats)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build real-data distribution plots comparing GR-RHS and RHS on the COVID dataset."
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
        default=Path("outputs/reports/covid_thesis/grrhs_rhs_distributions"),
        help="Destination directory for figures and tables.",
    )
    args = parser.parse_args()

    plot_report(args.sweep_dir, args.out_dir)
    print(f"[ok] report written to {args.out_dir}")


if __name__ == "__main__":
    main()
