from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from patsy import dmatrix

from data.preprocess import StandardizationConfig, apply_standardization
from grrhs.experiments.runner import (
    _apply_bayesian_sampling_budget,
    _instantiate_model,
    _maybe_calibrate_tau,
    _set_nested_config_value,
    _standardization_from_config,
)


PANEL_VARIABLES: List[Tuple[str, str]] = [
    ("hh_cmnty_cli", "Reported Community & Household COVID-like Illness (%)"),
    ("cli", "Reported COVID-like Illness (%)"),
]

MODEL_COLORS: Dict[str, str] = {
    "GIGG": "#2F55D4",
    "GR-RHS": "#111111",
    "RHS": "#E31A1C",
    "SGL": "#D9271C",
    "Lasso": "#B55D92",
    "Ridge": "#7A7A7A",
}

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

BAYESIAN_MODELS = {"GR-RHS", "RHS", "GIGG"}
REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = REPO_ROOT / "outputs" / "reports" / "covid_thesis" / "_curve_cache"


@dataclass
class RunContext:
    run_dir: Path
    label: str
    config: Dict[str, Any]
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    groups: List[List[int]]
    repeat_dir: Path
    raw_frame: pd.DataFrame
    spline_metadata: Dict[str, Dict[str, Any]]


@dataclass
class CurveSummary:
    label: str
    x_grid: np.ndarray
    mean: np.ndarray
    low: np.ndarray
    high: np.ndarray
    band_source: str


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload or {}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_model_label(run_dir: Path) -> str:
    cfg = _load_yaml(run_dir / "resolved_config.yaml")
    model_cfg = cfg.get("model", {}) or {}
    name = str(model_cfg.get("name", run_dir.name)).strip().lower()
    return MODEL_ALIASES.get(name, str(model_cfg.get("name", run_dir.name)))


def _timestamp_key(path: Path) -> Tuple[int, str]:
    suffix = path.name.rsplit("-", 1)[-1]
    try:
        return int(suffix), path.name
    except ValueError:
        return -1, path.name


def _find_latest_runs(sweep_dir: Path, labels: Sequence[str]) -> List[Path]:
    candidates = [path for path in sweep_dir.iterdir() if path.is_dir()]
    selected: Dict[str, Path] = {}
    wanted = set(labels)
    for run_dir in sorted(candidates, key=_timestamp_key):
        label = _resolve_model_label(run_dir)
        if label not in wanted:
            continue
        selected[label] = run_dir
    missing = [label for label in labels if label not in selected]
    if missing:
        raise SystemExit(f"Missing run directories for models: {', '.join(missing)}")
    return [selected[label] for label in labels]


def _flatten_draws(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=float)
    if data.ndim == 1:
        return data.reshape(1, -1)
    if data.ndim == 2:
        return data
    return data.reshape(-1, data.shape[-1])


def _bspline_basis(values: np.ndarray, metadata: Mapping[str, Any]) -> np.ndarray:
    degree = int(metadata["degree"])
    inner_knots = list(metadata.get("inner_knots", []))
    lower_bound = float(metadata["lower_bound"])
    upper_bound = float(metadata["upper_bound"])
    include_intercept = bool(metadata.get("include_intercept", False))
    design = dmatrix(
        "bs(x, knots=knots, degree=degree, include_intercept=include_intercept, lower_bound=lower_bound, upper_bound=upper_bound) - 1",
        {
            "x": np.asarray(values, dtype=np.float64),
            "knots": inner_knots,
            "degree": degree,
            "include_intercept": include_intercept,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        },
        return_type="dataframe",
    )
    return design.to_numpy(dtype=np.float32, copy=True)


def _load_run_context(run_dir: Path) -> RunContext:
    cfg = _load_yaml(run_dir / "resolved_config.yaml")
    repeat_dirs = sorted(path for path in run_dir.glob("repeat_*") if path.is_dir())
    if not repeat_dirs:
        raise SystemExit(f"No repeat directories found under {run_dir}")
    repeat_dir = repeat_dirs[0]
    meta = _load_json(repeat_dir / "dataset_meta.json")
    paths = meta["metadata"]["paths"]
    path_X = Path(paths["path_X"])
    path_y = Path(paths["path_y"])
    if not path_X.is_absolute():
        path_X = REPO_ROOT / path_X
    if not path_y.is_absolute():
        path_y = REPO_ROOT / path_y
    raw_csv = REPO_ROOT / "data" / "real" / "covid19_trust_experts" / "processed" / "analysis_bundle" / "trust_experts_raw.csv"
    spline_metadata_path = REPO_ROOT / "data" / "real" / "covid19_trust_experts" / "processed" / "analysis_bundle" / "spline_metadata.json"
    X = np.load(path_X).astype(np.float32)
    y = np.load(path_y).astype(np.float32)
    feature_names = list(meta.get("feature_names") or [])
    groups = [list(map(int, group)) for group in (meta.get("model_groups") or meta.get("groups") or [])]
    raw_frame = pd.read_csv(raw_csv)
    return RunContext(
        run_dir=run_dir,
        label=_resolve_model_label(run_dir),
        config=cfg,
        X=X,
        y=y,
        feature_names=feature_names,
        groups=groups,
        repeat_dir=repeat_dir,
        raw_frame=raw_frame,
        spline_metadata=_load_json(spline_metadata_path),
    )


def _feature_indices(feature_names: Sequence[str], variable: str) -> List[int]:
    prefix = f"{variable}_bs_"
    indices = [idx for idx, name in enumerate(feature_names) if str(name).startswith(prefix)]
    if not indices:
        raise SystemExit(f"No spline features found for {variable}")
    return indices


def _grid_for_variable(frame: pd.DataFrame, variable: str, *, points: int, metadata: Mapping[str, Any]) -> np.ndarray:
    observed = np.asarray(frame[variable], dtype=float)
    upper = float(metadata.get("upper_bound", np.nanmax(observed)))
    lower = float(metadata.get("lower_bound", np.nanmin(observed)))
    return np.linspace(lower, upper, points, dtype=float)


def _posterior_from_saved_fold(fold_dir: Path) -> Optional[np.ndarray]:
    posterior_path = fold_dir / "posterior_samples.npz"
    if not posterior_path.exists():
        return None
    posterior = np.load(posterior_path, allow_pickle=True)
    if "beta" not in posterior.files:
        return None
    return _flatten_draws(np.asarray(posterior["beta"], dtype=float))


def _fit_fold_model(
    context: RunContext,
    fold_dir: Path,
    std_cfg: StandardizationConfig,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    cache_dir = CACHE_ROOT / context.run_dir.name / context.repeat_dir.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{fold_dir.name}_coefficients.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        beta_point = np.asarray(cached["beta_point"], dtype=float).reshape(-1)
        if "beta_draws" in cached.files:
            beta_draws = _flatten_draws(np.asarray(cached["beta_draws"], dtype=float))
            return beta_draws, beta_point
        return None, beta_point

    fold_summary = _load_json(fold_dir / "fold_summary.json")
    fold_arrays = np.load(fold_dir / "fold_arrays.npz")
    train_idx = np.asarray(fold_arrays["train_idx"], dtype=int)

    std_train = apply_standardization(context.X[train_idx], context.y[train_idx], std_cfg)
    X_train = std_train.X
    y_train = std_train.y

    cfg = deepcopy(context.config)
    cfg.setdefault("model", {})
    cfg["model"].pop("search", None)
    _apply_bayesian_sampling_budget(cfg)
    for key, value in (fold_summary.get("best_params") or {}).items():
        if "." in str(key):
            _set_nested_config_value(cfg["model"], str(key), value)
        else:
            cfg["model"][key] = value

    _maybe_calibrate_tau(cfg["model"], std_cfg, X_train, y_train, context.groups, "regression")
    model = _instantiate_model(cfg, context.groups, X_train.shape[1])
    try:
        model.fit(X_train, y_train, groups=context.groups)
    except TypeError:
        model.fit(X_train, y_train)

    coef_point = getattr(model, "coef_mean_", None)
    if coef_point is None:
        coef_point = getattr(model, "coef_", None)
    if coef_point is None:
        raise RuntimeError(f"Unable to recover coefficients for {fold_dir}")

    coef_draws = getattr(model, "coef_samples_", None)
    if coef_draws is None:
        np.savez_compressed(cache_path, beta_point=np.asarray(coef_point, dtype=float).reshape(-1))
        return None, np.asarray(coef_point, dtype=float).reshape(-1)
    beta_draws = _flatten_draws(np.asarray(coef_draws, dtype=float))
    beta_point_arr = np.asarray(coef_point, dtype=float).reshape(-1)
    np.savez_compressed(cache_path, beta_draws=beta_draws, beta_point=beta_point_arr)
    return beta_draws, beta_point_arr


def _curves_for_run(
    context: RunContext,
    variable: str,
    x_grid: np.ndarray,
) -> CurveSummary:
    indices = _feature_indices(context.feature_names, variable)
    spline_meta = context.spline_metadata.get(variable)
    if not spline_meta:
        raise RuntimeError(f"Missing spline metadata for {variable}")
    raw_basis_grid = _bspline_basis(x_grid, spline_meta)
    reference_x = np.array([float(spline_meta.get("lower_bound", x_grid[0]))], dtype=np.float32)
    reference_basis = _bspline_basis(reference_x, spline_meta)

    std_cfg = _standardization_from_config(context.config, "regression")

    fold_dir_paths = sorted(path for path in context.repeat_dir.glob("fold_*") if path.is_dir())
    if not fold_dir_paths:
        raise SystemExit(f"No fold directories found under {context.repeat_dir}")

    curve_draws: List[np.ndarray] = []
    curve_points: List[np.ndarray] = []

    for fold_dir in fold_dir_paths:
        fold_arrays = np.load(fold_dir / "fold_arrays.npz")
        x_mean = np.asarray(fold_arrays["x_mean"], dtype=float).reshape(-1)
        x_scale = np.asarray(fold_arrays["x_scale"], dtype=float).reshape(-1)
        if x_mean.size == 0 or x_scale.size == 0:
            raise RuntimeError(f"Missing standardization stats in {fold_dir}")

        design_grid = np.zeros((x_grid.shape[0], len(context.feature_names)), dtype=np.float32)
        design_ref = np.zeros((1, len(context.feature_names)), dtype=np.float32)
        design_grid[:, indices] = raw_basis_grid
        design_ref[:, indices] = reference_basis

        design_grid = (design_grid - x_mean) / np.maximum(x_scale, 1e-8)
        design_ref = (design_ref - x_mean) / np.maximum(x_scale, 1e-8)

        beta_draws = _posterior_from_saved_fold(fold_dir)
        beta_point: Optional[np.ndarray] = None
        if beta_draws is None:
            beta_draws, beta_point = _fit_fold_model(context, fold_dir, std_cfg)
        else:
            beta_point = np.mean(beta_draws, axis=0)

        if beta_draws is not None:
            fold_curves = design_grid @ beta_draws.T
            ref_curve = design_ref @ beta_draws.T
            curve_draws.append((fold_curves - ref_curve).T)
        else:
            ref_value = float((design_ref @ beta_point).reshape(-1)[0])
            curve_points.append((design_grid @ beta_point) - ref_value)

        if beta_point is not None:
            ref_value = float((design_ref @ beta_point).reshape(-1)[0])
            curve_points.append((design_grid @ beta_point) - ref_value)

    if curve_draws:
        stacked = np.vstack(curve_draws)
        mean = stacked.mean(axis=0)
        low = np.quantile(stacked, 0.025, axis=0)
        high = np.quantile(stacked, 0.975, axis=0)
        return CurveSummary(
            label=context.label,
            x_grid=x_grid,
            mean=mean,
            low=low,
            high=high,
            band_source="posterior",
        )

    if not curve_points:
        raise RuntimeError(f"No curves were constructed for {context.label}")

    stacked_points = np.vstack(curve_points)
    mean = stacked_points.mean(axis=0)
    low = np.quantile(stacked_points, 0.025, axis=0)
    high = np.quantile(stacked_points, 0.975, axis=0)
    return CurveSummary(
        label=context.label,
        x_grid=x_grid,
        mean=mean,
        low=low,
        high=high,
        band_source="cross_fold",
    )


def _plot_rug(ax: plt.Axes, values: np.ndarray) -> None:
    ymin, ymax = ax.get_ylim()
    height = ymax - ymin
    rug_top = ymin + 0.03 * height
    ax.vlines(values, ymin, rug_top, color="black", linewidth=0.5, alpha=0.75)


def _panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(0.5, -0.20, text, transform=ax.transAxes, ha="center", va="top", fontsize=16)


def _set_robust_ylim(ax: plt.Axes, summaries: Sequence[CurveSummary]) -> None:
    values: List[float] = [0.0]
    for summary in summaries:
        values.extend(np.asarray(summary.mean, dtype=float).tolist())
        values.extend(np.asarray(summary.low, dtype=float).tolist())
        values.extend(np.asarray(summary.high, dtype=float).tolist())
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    y_lo = float(np.quantile(arr, 0.02))
    y_hi = float(np.quantile(arr, 0.98))
    if not math.isfinite(y_lo) or not math.isfinite(y_hi) or y_hi <= y_lo:
        return
    pad = 0.12 * (y_hi - y_lo)
    ax.set_ylim(y_lo - pad, y_hi + pad)


def _paper_style_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#222222")
    ax.set_facecolor("white")


def plot_covid_nonlinear_effects(
    sweep_dir: Path,
    out_path: Path,
    *,
    models: Sequence[str],
    grid_points: int,
    title: Optional[str],
    add_inset: bool,
) -> None:
    run_dirs = _find_latest_runs(sweep_dir, models)
    contexts = [_load_run_context(path) for path in run_dirs]
    raw_frame = contexts[0].raw_frame

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.6))
    plt.subplots_adjust(bottom=0.24, wspace=0.20)

    for panel_idx, (variable, xlabel) in enumerate(PANEL_VARIABLES):
        ax = axes[panel_idx]
        _paper_style_axes(ax)
        spline_meta = contexts[0].spline_metadata.get(variable)
        if not spline_meta:
            raise RuntimeError(f"Missing spline metadata for {variable}")
        x_grid = _grid_for_variable(raw_frame, variable, points=grid_points, metadata=spline_meta)
        summaries = [_curves_for_run(context, variable, x_grid) for context in contexts]

        for summary in summaries:
            color = MODEL_COLORS.get(summary.label, "#4C78A8")
            show_band = summary.label in BAYESIAN_MODELS
            if show_band:
                alpha = 0.16
                ax.fill_between(summary.x_grid, summary.low, summary.high, color=color, alpha=alpha, linewidth=0.0)
            ax.plot(summary.x_grid, summary.mean, color=color, linewidth=2.1, label=summary.label)

        ax.axhline(0.0, color="#606060", linewidth=0.9, alpha=0.55)
        _set_robust_ylim(ax, summaries)
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel("Predicted Effect", fontsize=15)
        ax.tick_params(labelsize=12)
        ax.grid(axis="y", alpha=0.18, linewidth=0.8)
        _plot_rug(ax, np.asarray(raw_frame[variable], dtype=float))
        _panel_label(ax, "(a)" if panel_idx == 0 else "(b)")

        if panel_idx == 0 and add_inset:
            inset = inset_axes(ax, width="36%", height="48%", loc="upper left", borderpad=1.6)
            for summary in summaries:
                color = MODEL_COLORS.get(summary.label, "#4C78A8")
                inset.fill_between(summary.x_grid, summary.low, summary.high, color=color, alpha=0.10, linewidth=0.0)
                inset.plot(summary.x_grid, summary.mean, color=color, linewidth=1.5)
            x_lo = float(np.quantile(raw_frame[variable], 0.02))
            x_hi = float(np.quantile(raw_frame[variable], 0.35))
            inset.set_xlim(x_lo, x_hi)
            ylim_candidates = []
            for summary in summaries:
                mask = (summary.x_grid >= x_lo) & (summary.x_grid <= x_hi)
                ylim_candidates.extend(summary.low[mask].tolist())
                ylim_candidates.extend(summary.high[mask].tolist())
            if ylim_candidates:
                y_lo = float(np.quantile(np.asarray(ylim_candidates), 0.05))
                y_hi = float(np.quantile(np.asarray(ylim_candidates), 0.95))
                if math.isfinite(y_lo) and math.isfinite(y_hi) and y_hi > y_lo:
                    pad = 0.10 * (y_hi - y_lo)
                    inset.set_ylim(y_lo - pad, y_hi + pad)
            inset.tick_params(labelsize=8.5)
            _paper_style_axes(inset)
            mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="#707070", linestyle="--", linewidth=0.9)

    handles, labels = axes[1].get_legend_handles_labels()
    dedup: Dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        dedup[label] = handle
    axes[1].legend(
        dedup.values(),
        dedup.keys(),
        loc="lower left",
        frameon=False,
        fontsize=12.5,
        handlelength=1.0,
        handletextpad=0.6,
    )

    if title:
        fig.suptitle(title, fontsize=18, y=0.98)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot paper-style nonlinear COVID effect curves from sweep outputs.")
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path("outputs/sweeps/real_covid19_trust_experts"),
        help="Sweep directory containing trust_experts_* run folders.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Destination PNG path.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["GIGG", "GR-RHS", "RHS"],
        help="Model labels to include. Paper-style Bayesian comparison: GIGG GR-RHS RHS.",
    )
    parser.add_argument("--grid-points", type=int, default=220, help="Number of x-grid points per panel.")
    parser.add_argument("--title", type=str, default=None, help="Optional figure title.")
    parser.add_argument(
        "--no-inset",
        action="store_true",
        help="Disable the inset zoom on the hh_cmnty_cli panel.",
    )
    args = parser.parse_args()

    plot_covid_nonlinear_effects(
        args.sweep_dir,
        args.out,
        models=args.models,
        grid_points=max(60, int(args.grid_points)),
        title=args.title,
        add_inset=not args.no_inset,
    )
    print(f"[ok] figure written to {args.out}")


if __name__ == "__main__":
    main()

