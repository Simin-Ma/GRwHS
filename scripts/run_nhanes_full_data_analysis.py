from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from data.loaders import load_real_dataset
from data.preprocess import StandardizationConfig, apply_standardization
from grrhs.experiments.registry import build_from_config
from grrhs.experiments.runner import (
    _maybe_calibrate_tau,
    _residualize_against_covariates,
    _standardize_covariates_train_test,
)


PAPER_LABELS: Dict[str, str] = {
    "LBXBPB": "Lead",
    "LBXBCD": "Cadmium",
    "LBXTHG": "Mercury",
    "URXMEP": "Mono-ethyl phthalate",
    "URXMBP": "Mono-n-butyl phthalate",
    "URXMIB": "Mono-isobutyl phthalate",
    "URXMHP": "Mono-(2-ethylhexyl) phthalate",
    "URXMHH": "Mono-(2-ethyl-5-hydroxyhexyl) phthalate",
    "URXMOH": "Mono-(2-ethyl-5-oxohexyl) phthalate",
    "URXMZP": "Mono-benzyl phthalate",
    "LBXPDE": "p,p'-DDE",
    "LBXPDT": "p,p'-DDT",
    "LBXBHC": "Beta-hexachlorocyclohexane",
    "LBXOXY": "Oxychlordane",
    "LBXTNA": "Trans-nonachlor",
    "LBXHPE": "Heptachlor epoxide",
    "LBXMIR": "Mirex",
    "LBXHCB": "Hexachlorobenzene",
    "LBXBR2": "BDE-28",
    "LBXBR3": "BDE-47",
    "LBXBR5": "BDE-99",
    "LBXBR6": "BDE-100",
    "LBXBR7": "BDE-153",
    "LBXBR8": "BDE-154",
    "LBXBR9": "BDE-183",
    "URXP01": "1-hydroxynaphthalene",
    "URXP02": "2-hydroxynaphthalene",
    "URXP03": "2-hydroxyfluorene",
    "URXP04": "3-hydroxyfluorene",
    "URXP05": "1-hydroxyphenanthrene",
    "URXP06": "2-hydroxyphenanthrene",
    "URXP07": "3-hydroxyphenanthrene",
    "URXP10": "9-hydroxyfluorene",
    "URXP17": "1-hydroxypyrene",
    "URXP19": "4-phenanthrene",
}

GROUP_DISPLAY_NAMES: Dict[int, str] = {
    0: "Metals",
    1: "Phthalates",
    2: "Organochlorine pesticides",
    3: "PBDEs",
    4: "PAHs",
}

GROUP_ORDER = list(GROUP_DISPLAY_NAMES.values())
TARGET_MODELS = ["GIGG", "GR-RHS", "RHS"]
MODEL_COLORS: Dict[str, str] = {
    "GIGG": "#3A7D44",
    "GR-RHS": "#153B50",
    "RHS": "#2F6690",
}

DEFAULT_MODEL_SPECS = {
    "gigg": [
        "configs/base.yaml",
        "configs/experiments/real_nhanes_2003_2004_ggt.yaml",
        "configs/methods/gigg.yaml",
        "configs/overrides/nhanes_gigg_paper_exact.yaml",
    ],
    "grrhs": [
        "configs/base.yaml",
        "configs/experiments/real_nhanes_2003_2004_ggt.yaml",
        "configs/methods/grrhs_regression.yaml",
    ],
    "rhs": [
        "configs/base.yaml",
        "configs/experiments/real_nhanes_2003_2004_ggt.yaml",
        "configs/methods/regularized_horseshoe.yaml",
    ],
}


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def _load_merged_config(paths: Sequence[Path]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for path in paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must decode to a mapping.")
        _deep_update(merged, payload)
    return merged


def _model_label(cfg: Mapping[str, Any]) -> str:
    model_cfg = cfg.get("model", {}) or {}
    name = str(model_cfg.get("name", "model"))
    aliases = {
        "gigg": "GIGG",
        "grrhs_gibbs": "GR-RHS",
    }
    if name == "grrhs_gibbs" and model_cfg.get("use_groups") is False:
        return "RHS"
    return aliases.get(name, name)


def _standardization_from_config(cfg: Mapping[str, Any]) -> StandardizationConfig:
    std_cfg = cfg.get("standardization", {}) or {}
    return StandardizationConfig(
        X=str(std_cfg.get("X", "unit_variance")),
        y_center=bool(std_cfg.get("y_center", True)),
    )


def _group_name_lookup(groups: Sequence[Sequence[int]], p: int) -> List[str]:
    lookup = ["Unknown"] * p
    for gid, members in enumerate(groups):
        label = GROUP_DISPLAY_NAMES.get(gid, f"Group {gid + 1}")
        for idx in members:
            lookup[int(idx)] = label
    return lookup


def _prepare_design(cfg: Mapping[str, Any], repo_root: Path) -> Dict[str, Any]:
    data_cfg = cfg.get("data", {}) or {}
    loader_cfg = data_cfg.get("loader", {}) or {}
    loaded = load_real_dataset(loader_cfg, base_dir=repo_root)
    if loaded.y is None:
        raise ValueError("Full-data NHANES analysis requires y.")
    if loaded.groups is None:
        raise ValueError("Full-data NHANES analysis requires exposure groups.")

    std_cfg = _standardization_from_config(cfg)
    std_all = apply_standardization(loaded.X, loaded.y, std_cfg)
    X_model = np.asarray(std_all.X, dtype=np.float32)
    y_model = np.asarray(std_all.y, dtype=np.float32)

    C_model = None
    covariate_alpha_hat = None
    if loaded.C is not None:
        C_train_std, _, cov_mean, cov_scale, cov_binary = _standardize_covariates_train_test(loaded.C, loaded.C)
        X_model, _, _ = _residualize_against_covariates(X_model, X_model, C_train_std, C_train_std)
        y_model, _, covariate_alpha_hat = _residualize_against_covariates(y_model, y_model, C_train_std, C_train_std)
        C_model = C_train_std
    else:
        cov_mean = np.zeros(0, dtype=np.float32)
        cov_scale = np.zeros(0, dtype=np.float32)
        cov_binary = np.zeros(0, dtype=bool)

    return {
        "loaded": loaded,
        "X_model": np.asarray(X_model, dtype=np.float32),
        "y_model": np.asarray(y_model, dtype=np.float32),
        "x_mean": np.asarray(std_all.x_mean, dtype=np.float32),
        "x_scale": np.asarray(std_all.x_scale, dtype=np.float32),
        "y_mean": float(std_all.y_mean or 0.0),
        "C_model": C_model,
        "covariate_alpha_hat": covariate_alpha_hat,
        "covariate_mean": cov_mean,
        "covariate_scale": cov_scale,
        "covariate_binary_mask": cov_binary,
    }


def _collect_posterior_arrays(model: Any) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    attr_map = {
        "coef_samples_": "beta",
        "sigma_samples_": "sigma",
        "sigma2_samples_": "sigma2",
        "tau_samples_": "tau",
        "phi_samples_": "phi",
        "lambda_samples_": "lambda",
        "gamma_samples_": "gamma",
        "b_samples_": "b",
    }
    for attr, key in attr_map.items():
        value = getattr(model, attr, None)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.size == 0:
            continue
        arrays[key] = arr
    return arrays


def _effect_frame(
    *,
    label: str,
    model: Any,
    feature_names: Sequence[str],
    group_names: Sequence[str],
    x_scale: np.ndarray,
) -> pd.DataFrame:
    draws = getattr(model, "coef_samples_", None)
    if draws is None:
        coef = getattr(model, "coef_mean_", None)
        if coef is None:
            coef = getattr(model, "coef_", None)
        if coef is None:
            raise RuntimeError(f"Model {label} does not expose coefficients.")
        draws = np.asarray(coef, dtype=float).reshape(1, -1)
    else:
        draws = np.asarray(draws, dtype=float)

    delta = np.log(2.0) / np.maximum(np.asarray(x_scale, dtype=float), 1e-8)
    pct_draws = 100.0 * (np.exp(draws * delta[np.newaxis, :]) - 1.0)

    rows: List[Dict[str, Any]] = []
    for idx, feature in enumerate(feature_names):
        series = pct_draws[:, idx]
        rows.append(
            {
                "model": label,
                "feature": feature,
                "label": PAPER_LABELS.get(feature, feature),
                "group": group_names[idx],
                "mean_percent_change": float(np.mean(series)),
                "median_percent_change": float(np.quantile(series, 0.5)),
                "ci95_low": float(np.quantile(series, 0.025)),
                "ci95_high": float(np.quantile(series, 0.975)),
                "ci95_length": float(np.quantile(series, 0.975) - np.quantile(series, 0.025)),
                "draws": int(series.size),
            }
        )
    return pd.DataFrame(rows)


def _group_summary(effect_frame: pd.DataFrame) -> pd.DataFrame:
    return (
        effect_frame.groupby(["model", "group"], as_index=False)
        .agg(
            mean_abs_percent_change=("median_percent_change", lambda s: float(np.mean(np.abs(s)))),
            mean_ci95_length=("ci95_length", "mean"),
            median_ci95_length=("ci95_length", "median"),
            exposure_count=("feature", "count"),
        )
        .sort_values(["model", "group"])
    )


def _group_hyper_summary(label: str, model: Any, groups: Sequence[Sequence[int]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    phi_samples = getattr(model, "phi_samples_", None)
    b_samples = getattr(model, "b_samples_", None)
    gamma_samples = getattr(model, "gamma_samples_", None)

    for gid, _members in enumerate(groups):
        group_name = GROUP_DISPLAY_NAMES.get(gid, f"Group {gid + 1}")
        row: Dict[str, Any] = {"model": label, "group": group_name}
        if phi_samples is not None:
            phi = np.asarray(phi_samples, dtype=float)[:, gid]
            row.update(
                {
                    "phi_mean": float(np.mean(phi)),
                    "phi_median": float(np.median(phi)),
                    "phi_q05": float(np.quantile(phi, 0.05)),
                    "phi_q95": float(np.quantile(phi, 0.95)),
                }
            )
        if b_samples is not None:
            b = np.asarray(b_samples, dtype=float)[:, gid]
            row.update(
                {
                    "b_mean": float(np.mean(b)),
                    "b_median": float(np.median(b)),
                    "b_q05": float(np.quantile(b, 0.05)),
                    "b_q95": float(np.quantile(b, 0.95)),
                }
            )
        if gamma_samples is not None:
            gamma = np.asarray(gamma_samples, dtype=float)[:, gid]
            row.update(
                {
                    "gamma_mean": float(np.mean(gamma)),
                    "gamma_median": float(np.median(gamma)),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_forest(effect_frame: pd.DataFrame, out_path: Path, *, title: str) -> None:
    frame = effect_frame[effect_frame["model"].isin(TARGET_MODELS)].copy()
    summary = (
        frame.groupby(["feature", "label", "group"], as_index=False)
        .agg(sort_key=("median_percent_change", lambda s: float(np.mean(np.abs(s)))))
    )
    summary["group"] = pd.Categorical(summary["group"], categories=GROUP_ORDER, ordered=True)
    summary.sort_values(["group", "sort_key", "label"], ascending=[True, False, True], inplace=True)
    summary["order"] = np.arange(summary.shape[0], dtype=int)
    frame = frame.merge(summary[["feature", "order"]], on="feature", how="left")
    labels = frame[["feature", "label", "group", "order"]].drop_duplicates().sort_values("order").reset_index(drop=True)

    fig_h = max(10.0, 0.35 * labels.shape[0] + 2.6)
    fig, ax = plt.subplots(figsize=(12.6, fig_h))
    offsets = {"GIGG": -0.24, "GR-RHS": 0.0, "RHS": 0.24}

    for model_name in TARGET_MODELS:
        model_frame = frame[frame["model"] == model_name].copy()
        model_frame = labels.merge(
            model_frame[["feature", "median_percent_change", "ci95_low", "ci95_high"]],
            on="feature",
            how="left",
        )
        y = model_frame["order"].to_numpy(dtype=float) + offsets[model_name]
        center = model_frame["median_percent_change"].to_numpy(dtype=float)
        low = model_frame["ci95_low"].to_numpy(dtype=float)
        high = model_frame["ci95_high"].to_numpy(dtype=float)
        xerr = np.vstack([center - low, high - center])
        ax.errorbar(
            center,
            y,
            xerr=xerr,
            fmt="o",
            color=MODEL_COLORS[model_name],
            ecolor=MODEL_COLORS[model_name],
            elinewidth=1.7,
            capsize=2.5,
            markersize=4.8,
            label=model_name,
            alpha=0.95,
        )

    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1.1, alpha=0.85)
    ax.set_yticks(labels["order"].to_numpy(dtype=float))
    ax.set_yticklabels(labels["label"].tolist(), fontsize=10.3)
    ax.invert_yaxis()
    ax.set_xlabel("Percent change in GGT for 2x exposure", fontsize=13)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=12)
    ax.grid(axis="x", alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)

    group_blocks: List[tuple[str, float, float]] = []
    for group in GROUP_ORDER:
        sub = labels[labels["group"] == group]
        if sub.empty:
            continue
        group_blocks.append((group, float(sub["order"].min()) - 0.6, float(sub["order"].max()) + 0.6))

    for idx, (group, y0, y1) in enumerate(group_blocks):
        if idx % 2 == 0:
            ax.axhspan(y0, y1, color="#F5F2EB", alpha=0.7, zorder=0)
        if idx > 0:
            ax.axhline(y0, color="#B0B0B0", linewidth=0.8, alpha=0.85)
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

    ax.legend(loc="lower right", frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _predictive_story(cv_csv: Path) -> List[str]:
    if not cv_csv.exists():
        return []
    frame = pd.read_csv(cv_csv)
    if frame.empty:
        return []
    rows = []
    ranked = frame.sort_values("RMSE").reset_index(drop=True)
    best = ranked.iloc[0]
    rows.append(
        f"Best CV predictive model: {best['variation']} ({best['model']}) with RMSE={best['RMSE']:.6f} and MLPD={best['MLPD']:.6f}."
    )
    return rows


def _write_story(
    out_path: Path,
    *,
    cv_csv: Path,
    effect_frame: pd.DataFrame,
    group_hyper_frame: pd.DataFrame,
    group_summary_frame: pd.DataFrame,
) -> None:
    lines = ["# NHANES Full-Data Analysis", ""]
    lines.extend(_predictive_story(cv_csv))
    if lines[-1:] != [""]:
        lines.append("")

    lines.append("## Full-data effect summary")
    lines.append("")
    for model_name in TARGET_MODELS:
        sub = group_summary_frame[group_summary_frame["model"] == model_name]
        if sub.empty:
            continue
        lines.append(f"### {model_name}")
        for row in sub.itertuples(index=False):
            lines.append(
                f"- {row.group}: mean |effect|={row.mean_abs_percent_change:.2f}, median CI length={row.median_ci95_length:.2f}"
            )
        lines.append("")

    grrhs_hyper = group_hyper_frame[group_hyper_frame["model"] == "GR-RHS"]
    if not grrhs_hyper.empty and "phi_median" in grrhs_hyper.columns:
        lines.append("## GR-RHS interpretability")
        lines.append("")
        ordered = grrhs_hyper.sort_values("phi_median", ascending=False)
        for row in ordered.itertuples(index=False):
            lines.append(
                f"- {row.group}: phi median={row.phi_median:.3f}, phi 90% interval=[{row.phi_q05:.3f}, {row.phi_q95:.3f}]"
            )
        lines.append("")

    gigg_hyper = group_hyper_frame[group_hyper_frame["model"] == "GIGG"]
    if not gigg_hyper.empty and "b_median" in gigg_hyper.columns:
        lines.append("## GIGG MMLE group dependence")
        lines.append("")
        ordered = gigg_hyper.sort_values("b_median")
        for row in ordered.itertuples(index=False):
            lines.append(
                f"- {row.group}: b median={row.b_median:.3f}, b 90% interval=[{row.b_q05:.3f}, {row.b_q95:.3f}]"
            )

    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit NHANES full-data models and emit paper-style effect summaries.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/reports/nhanes_full_data_analysis"),
        help="Destination directory for fitted models and plots.",
    )
    parser.add_argument(
        "--cv-summary",
        type=Path,
        default=Path("outputs/sweeps/real_nhanes_2003_2004_ggt/sweep_comparison_20260308-014419.csv"),
        help="Optional CV comparison CSV used in the narrative summary.",
    )
    parser.add_argument("--iters", type=int, default=None, help="Optional override for model.iters across all fitted models.")
    parser.add_argument("--burn-in", type=int, default=None, help="Optional override for inference.gibbs.burn_in.")
    parser.add_argument("--thin", type=int, default=None, help="Optional override for inference.gibbs.thin.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_effects: List[pd.DataFrame] = []
    merged_group_hypers: List[pd.DataFrame] = []
    merged_group_summaries: List[pd.DataFrame] = []
    manifests: Dict[str, Any] = {}

    for key, rel_paths in DEFAULT_MODEL_SPECS.items():
        cfg_paths = [(repo_root / rel_path).resolve() for rel_path in rel_paths]
        cfg = _load_merged_config(cfg_paths)
        if args.iters is not None:
            cfg.setdefault("model", {})["iters"] = int(args.iters)
        if args.burn_in is not None:
            cfg.setdefault("inference", {}).setdefault("gibbs", {})["burn_in"] = int(args.burn_in)
        if args.thin is not None:
            cfg.setdefault("inference", {}).setdefault("gibbs", {})["thin"] = int(args.thin)
        design = _prepare_design(cfg, repo_root)
        loaded = design["loaded"]
        label = _model_label(cfg)

        fit_cfg = deepcopy(cfg)
        fit_cfg.setdefault("data", {})
        fit_cfg["data"]["groups"] = loaded.groups
        fit_cfg["data"]["p"] = int(loaded.X.shape[1])
        fit_cfg["data"]["n"] = int(loaded.X.shape[0])
        _maybe_calibrate_tau(
            fit_cfg.setdefault("model", {}),
            _standardization_from_config(fit_cfg),
            np.asarray(design["X_model"], dtype=float),
            np.asarray(design["y_model"], dtype=float),
            loaded.groups,
            "regression",
        )
        model = build_from_config(fit_cfg)
        model.fit(design["X_model"], design["y_model"], groups=loaded.groups)

        model_dir = out_dir / key
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "resolved_config.yaml").write_text(yaml.safe_dump(fit_cfg, sort_keys=False), encoding="utf-8")
        np.savez_compressed(model_dir / "posterior_samples.npz", **_collect_posterior_arrays(model))

        group_names = _group_name_lookup(loaded.groups, loaded.X.shape[1])
        effect_frame = _effect_frame(
            label=label,
            model=model,
            feature_names=loaded.feature_names or [f"x{j}" for j in range(loaded.X.shape[1])],
            group_names=group_names,
            x_scale=design["x_scale"],
        )
        group_summary = _group_summary(effect_frame)
        group_hyper = _group_hyper_summary(label, model, loaded.groups)

        effect_frame.to_csv(model_dir / "effect_summary.csv", index=False)
        group_summary.to_csv(model_dir / "group_summary.csv", index=False)
        if not group_hyper.empty:
            group_hyper.to_csv(model_dir / "group_hyper_summary.csv", index=False)

        manifests[key] = {
            "label": label,
            "config_paths": [str(path) for path in cfg_paths],
            "output_dir": str(model_dir),
            "n": int(loaded.X.shape[0]),
            "p": int(loaded.X.shape[1]),
        }

        merged_effects.append(effect_frame)
        merged_group_summaries.append(group_summary)
        if not group_hyper.empty:
            merged_group_hypers.append(group_hyper)

    effect_frame_all = pd.concat(merged_effects, ignore_index=True)
    group_summary_all = pd.concat(merged_group_summaries, ignore_index=True)
    group_hyper_all = pd.concat(merged_group_hypers, ignore_index=True) if merged_group_hypers else pd.DataFrame()

    effect_csv = out_dir / "nhanes_full_data_effects.csv"
    group_csv = out_dir / "nhanes_full_data_group_summary.csv"
    hyper_csv = out_dir / "nhanes_full_data_group_hyper_summary.csv"
    forest_png = out_dir / "nhanes_full_data_gigg_grrhs_rhs_forest.png"
    story_md = out_dir / "nhanes_full_data_story.md"
    manifest_json = out_dir / "nhanes_full_data_manifest.json"

    effect_frame_all.to_csv(effect_csv, index=False)
    group_summary_all.to_csv(group_csv, index=False)
    if not group_hyper_all.empty:
        group_hyper_all.to_csv(hyper_csv, index=False)
    _plot_forest(effect_frame_all, forest_png, title="NHANES 2003-2004: Full-data effect sizes")
    _write_story(
        story_md,
        cv_csv=args.cv_summary.resolve(),
        effect_frame=effect_frame_all,
        group_hyper_frame=group_hyper_all,
        group_summary_frame=group_summary_all,
    )
    manifest_json.write_text(json.dumps(manifests, indent=2), encoding="utf-8")

    print(f"[ok] effects -> {effect_csv}")
    print(f"[ok] group summary -> {group_csv}")
    if not group_hyper_all.empty:
        print(f"[ok] group hyper summary -> {hyper_csv}")
    print(f"[ok] forest -> {forest_png}")
    print(f"[ok] story -> {story_md}")


if __name__ == "__main__":
    main()
