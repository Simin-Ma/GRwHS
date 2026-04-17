from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from data.preprocess import StandardizationConfig, apply_standardization
from grrhs.experiments.runner import (
    _instantiate_model,
    _residualize_against_covariates,
    _standardize_covariates_train_test,
)


GROUP_ORDER = [
    "Metals",
    "Phthalates",
    "Organochlorine pesticides",
    "PBDEs",
    "PAHs",
]

TARGET_MODELS = [
    "GR-RHS",
    "RHS",
    "GIGG",
    "Sparse Group Lasso",
    "Lasso",
    "Ridge",
]

MODEL_COLORS: Dict[str, str] = {
    "GR-RHS": "#153B50",
    "RHS": "#2F6690",
    "GIGG": "#3A7D44",
    "Sparse Group Lasso": "#D17A22",
    "Lasso": "#8E5572",
    "Ridge": "#6C757D",
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload or {}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _model_label(run_dir: Path) -> str:
    cfg = _load_yaml(run_dir / "resolved_config.yaml")
    model_cfg = cfg.get("model", {})
    name = str(model_cfg.get("name", run_dir.name))
    aliases = {
        "grrhs_nuts": "GR-RHS",
        "gigg": "GIGG",
        "ridge": "Ridge",
        "lasso": "Lasso",
        "sparse_group_lasso": "Sparse Group Lasso",
    }
    if name == "grrhs_nuts" and model_cfg.get("use_groups") is False:
        return "RHS"
    return aliases.get(name, name)


def _resolve_path(path_str: str, root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _infer_sweep_dir(exposure_csv: Path) -> Path:
    frame = pd.read_csv(exposure_csv)
    if frame.empty:
        raise SystemExit(f"{exposure_csv} is empty.")
    run_dir = Path(str(frame.iloc[0]["run_dir"])).resolve()
    return run_dir.parent


def _set_nested_config_value(payload: Dict[str, Any], dotted_key: str, value: Any) -> None:
    current = payload
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def _deterministic_effect_rows(run_dir: Path, repo_root: Path) -> List[Dict[str, Any]]:
    label = _model_label(run_dir)
    if label not in {"Lasso", "Ridge", "Sparse Group Lasso"}:
        return []

    resolved_cfg = _load_yaml(run_dir / "resolved_config.yaml")
    repeat_dirs = sorted(path for path in run_dir.glob("repeat_*") if path.is_dir())
    if not repeat_dirs:
        raise RuntimeError(f"No repeat directories found under {run_dir}")

    meta = _load_json(repeat_dirs[0] / "dataset_meta.json")
    paths = meta["metadata"]["paths"]
    X = np.load(_resolve_path(paths["path_X"], repo_root)).astype(np.float32)
    y = np.load(_resolve_path(paths["path_y"], repo_root)).astype(np.float32)
    C_path = paths.get("path_C")
    C = np.load(_resolve_path(C_path, repo_root)).astype(np.float32) if C_path else None
    feature_names = list(meta.get("feature_names") or [])
    groups = [list(map(int, members)) for members in (meta.get("model_groups") or meta.get("groups") or [])]
    std_cfg = StandardizationConfig(
        X=str((resolved_cfg.get("standardization", {}) or {}).get("X", "unit_variance")),
        y_center=bool((resolved_cfg.get("standardization", {}) or {}).get("y_center", True)),
    )

    fold_rows: List[Dict[str, Any]] = []
    for repeat_dir in repeat_dirs:
        meta_local = _load_json(repeat_dir / "dataset_meta.json")
        feature_names_local = list(meta_local.get("feature_names") or feature_names)
        groups_local = [list(map(int, members)) for members in (meta_local.get("model_groups") or groups)]
        for fold_dir in sorted(path for path in repeat_dir.glob("fold_*") if path.is_dir()):
            fold_arrays = np.load(fold_dir / "fold_arrays.npz")
            fold_summary = _load_json(fold_dir / "fold_summary.json")

            train_idx = np.asarray(fold_arrays["train_idx"], dtype=int)
            test_idx = np.asarray(fold_arrays["test_idx"], dtype=int)

            std_train = apply_standardization(X[train_idx], y[train_idx], std_cfg)
            X_train_model = std_train.X
            y_train_model = std_train.y

            if C is not None:
                C_train = np.asarray(C[train_idx], dtype=np.float32)
                C_test = np.asarray(C[test_idx], dtype=np.float32)
                C_train_std, C_test_std, _, _, _ = _standardize_covariates_train_test(C_train, C_test)
                X_train_model, _, _ = _residualize_against_covariates(
                    std_train.X,
                    apply_standardization(X[test_idx], y[test_idx], std_cfg).X,
                    C_train_std,
                    C_test_std,
                )
                y_train_model, _, _ = _residualize_against_covariates(
                    std_train.y,
                    y[test_idx] - float(std_train.y_mean or 0.0),
                    C_train_std,
                    C_test_std,
                )

            model_cfg = json.loads(json.dumps(resolved_cfg))
            model_cfg.setdefault("model", {})
            model_cfg["model"].pop("search", None)
            for key, value in (fold_summary.get("best_params") or {}).items():
                if "." in str(key):
                    _set_nested_config_value(model_cfg["model"], str(key), value)
                else:
                    model_cfg["model"][key] = value

            model = _instantiate_model(model_cfg, groups_local, X_train_model.shape[1])
            try:
                model.fit(X_train_model, y_train_model, groups=groups_local)
            except TypeError:
                model.fit(X_train_model, y_train_model)

            beta = getattr(model, "coef_mean_", None)
            if beta is None:
                beta = getattr(model, "coef_", None)
            if beta is None:
                raise RuntimeError(f"Unable to recover coefficients for {run_dir}")

            beta = np.asarray(beta, dtype=float).reshape(-1)
            x_scale = np.asarray(std_train.x_scale, dtype=float).reshape(-1)
            delta = np.log(2.0) / np.maximum(x_scale, 1e-8)
            effect_pct = 100.0 * (np.exp(beta * delta) - 1.0)

            group_name_by_index: Dict[int, str] = {}
            for gid, members in enumerate(groups_local):
                group_label = GROUP_ORDER[gid] if gid < len(GROUP_ORDER) else f"Group {gid + 1}"
                for idx in members:
                    group_name_by_index[int(idx)] = group_label

            for idx, feature_name in enumerate(feature_names_local):
                fold_rows.append(
                    {
                        "run_dir": str(run_dir),
                        "run_name": run_dir.name,
                        "model": label,
                        "feature": feature_name,
                        "label": feature_name,
                        "group": group_name_by_index.get(idx, "Unknown"),
                        "fold": fold_dir.name,
                        "median_percent_change": float(effect_pct[idx]),
                    }
                )

    label_map = (
        pd.read_csv(repo_root / "outputs" / "reports" / "nhanes_effects" / "nhanes_exposure_effects.csv")
        .loc[:, ["feature", "label", "group"]]
        .drop_duplicates()
    )
    fold_frame = pd.DataFrame(fold_rows).merge(label_map, on=["feature", "group"], how="left", suffixes=("", "_pretty"))
    fold_frame["label"] = fold_frame["label_pretty"].fillna(fold_frame["label"])
    fold_frame.drop(columns=["label_pretty"], inplace=True)

    aggregated = (
        fold_frame.groupby(["run_dir", "run_name", "model", "feature", "label", "group"], as_index=False)
        .agg(
            mean_percent_change=("median_percent_change", "mean"),
            median_percent_change=("median_percent_change", "median"),
            ci95_low=("median_percent_change", lambda s: float(np.quantile(np.asarray(s, dtype=float), 0.025))),
            ci95_high=("median_percent_change", lambda s: float(np.quantile(np.asarray(s, dtype=float), 0.975))),
            folds=("fold", "nunique"),
        )
    )
    aggregated["ci95_length"] = aggregated["ci95_high"] - aggregated["ci95_low"]
    aggregated["total_draws"] = aggregated["folds"]
    aggregated["interval_source"] = "cross_fold"
    return aggregated.to_dict(orient="records")


def _prepare_plot_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[frame["model"].isin(TARGET_MODELS)].copy()
    if frame.empty:
        raise SystemExit("No target models found in effect table.")

    summary = (
        frame.groupby(["feature", "label", "group"], as_index=False)
        .agg(sort_key=("median_percent_change", lambda s: float(np.mean(np.abs(s)))))
    )
    summary["group"] = pd.Categorical(summary["group"], categories=GROUP_ORDER, ordered=True)
    summary.sort_values(["group", "sort_key", "label"], ascending=[True, False, True], inplace=True)
    summary["order"] = np.arange(summary.shape[0], dtype=int)
    return frame.merge(summary[["feature", "order"]], on="feature", how="left")


def _plot_forest(frame: pd.DataFrame, out_path: Path, *, title: str) -> None:
    frame = _prepare_plot_frame(frame)
    labels = frame[["feature", "label", "group", "order"]].drop_duplicates().sort_values("order").reset_index(drop=True)

    n = labels.shape[0]
    fig_h = max(10.0, 0.38 * n + 2.8)
    fig, ax = plt.subplots(figsize=(13.8, fig_h))

    offsets = np.linspace(-0.30, 0.30, num=len(TARGET_MODELS))
    offset_map = {model: float(offset) for model, offset in zip(TARGET_MODELS, offsets)}

    for model in TARGET_MODELS:
        model_frame = frame[frame["model"] == model].copy()
        model_frame = labels.merge(
            model_frame[["feature", "median_percent_change", "ci95_low", "ci95_high"]],
            on="feature",
            how="left",
        )
        y = model_frame["order"].to_numpy(dtype=float) + offset_map[model]
        center = model_frame["median_percent_change"].to_numpy(dtype=float)
        low = model_frame["ci95_low"].to_numpy(dtype=float)
        high = model_frame["ci95_high"].to_numpy(dtype=float)
        xerr = np.vstack([center - low, high - center])

        ax.errorbar(
            center,
            y,
            xerr=xerr,
            fmt="o",
            color=MODEL_COLORS[model],
            ecolor=MODEL_COLORS[model],
            elinewidth=1.4,
            capsize=2.3,
            markersize=4.4,
            label=model,
            alpha=0.94,
        )

    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1.2, alpha=0.85)
    ax.set_yticks(labels["order"].to_numpy(dtype=float))
    ax.set_yticklabels(labels["label"].tolist(), fontsize=10.2)
    ax.invert_yaxis()
    ax.set_xlabel("Percent change in GGT for 2x exposure", fontsize=13)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(axis="x", alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)

    group_blocks: List[tuple[str, float, float]] = []
    for group in GROUP_ORDER:
        sub = labels[labels["group"] == group]
        if sub.empty:
            continue
        y0 = float(sub["order"].min()) - 0.7
        y1 = float(sub["order"].max()) + 0.7
        group_blocks.append((group, y0, y1))

    for idx, (group, y0, y1) in enumerate(group_blocks):
        if idx % 2 == 0:
            ax.axhspan(y0, y1, color="#F5F2EB", alpha=0.7, zorder=0)
        if idx > 0:
            ax.axhline(y0, color="#B0B0B0", linewidth=0.8, alpha=0.9)
        ax.text(
            0.995,
            (y0 + y1) / 2.0,
            group,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=11.2,
            fontweight="bold",
            color="#444444",
        )

    ax.legend(loc="lower right", frameon=False, fontsize=10.2, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a multi-model NHANES forest plot including deterministic baselines.")
    parser.add_argument(
        "--exposure-csv",
        type=Path,
        default=Path("outputs/reports/nhanes_effects/nhanes_exposure_effects.csv"),
        help="Exposure summary CSV produced by summarize_nhanes_effects.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/reports/nhanes_effects/nhanes_all_models_forest.png"),
        help="Destination PNG path.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/reports/nhanes_effects/nhanes_exposure_effects_all_models.csv"),
        help="Merged effect summary CSV path.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="NHANES 2003-2004: Exposure Effect Sizes Across Models",
        help="Figure title.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    posterior_frame = pd.read_csv(args.exposure_csv)
    posterior_frame["interval_source"] = "posterior_draws"

    sweep_dir = _infer_sweep_dir(args.exposure_csv)
    deterministic_runs = []
    for run_dir in sorted(path for path in sweep_dir.iterdir() if path.is_dir()):
        label = _model_label(run_dir)
        if label in {"Lasso", "Ridge", "Sparse Group Lasso"}:
            deterministic_runs.append(run_dir)

    deterministic_rows: List[Dict[str, Any]] = []
    for run_dir in deterministic_runs:
        deterministic_rows.extend(_deterministic_effect_rows(run_dir, repo_root))

    merged = pd.concat([posterior_frame, pd.DataFrame(deterministic_rows)], ignore_index=True, sort=False)
    merged = merged[merged["model"].isin(TARGET_MODELS)].copy()
    merged.to_csv(args.out_csv, index=False)

    _plot_forest(merged, args.out, title=args.title)
    print(f"[ok] merged effect table written to {args.out_csv}")
    print(f"[ok] plot written to {args.out}")


if __name__ == "__main__":
    main()

