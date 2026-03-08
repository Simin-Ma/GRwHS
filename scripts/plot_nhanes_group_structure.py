from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from grrhs.diagnostics.shrinkage import prior_precision, regularized_lambda, shrinkage_kappa


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

GROUP_DISPLAY_NAMES: Dict[str, str] = {
    "metals": "Metals",
    "phthalates": "Phthalates",
    "organochlorines": "Organochlorine pesticides",
    "pbdes": "PBDEs",
    "pahs": "PAHs",
}

GROUP_ORDER = list(GROUP_DISPLAY_NAMES.values())

MODEL_COLORS: Dict[str, str] = {
    "GR-RHS": "#153B50",
    "RHS": "#3C6E71",
    "GIGG": "#3A7D44",
    "Sparse Group Lasso": "#D17A22",
    "Lasso": "#8E5572",
    "Ridge": "#7A7A7A",
}

MAIN_STRUCTURE_MODELS = ["RHS", "GR-RHS"]


@dataclass
class RunContext:
    label: str
    run_dir: Path
    run_payload: Dict[str, Any]
    metrics_mean: Dict[str, float]
    metrics_std: Dict[str, float]
    groups: List[List[int]]
    feature_names: List[str]
    feature_labels: List[str]
    group_labels: List[str]
    slab_width: float
    x_path: Path


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload or {}


def _model_label(run_dir: Path) -> str:
    cfg_path = run_dir / "resolved_config.yaml"
    if not cfg_path.exists():
        return run_dir.name
    cfg = _load_yaml(cfg_path)
    model_cfg = cfg.get("model", {})
    name = str(model_cfg.get("name", run_dir.name))
    aliases = {
        "grrhs_gibbs": "GR-RHS",
        "gigg": "GIGG",
        "ridge": "Ridge",
        "lasso": "Lasso",
        "sparse_group_lasso": "Sparse Group Lasso",
    }
    if name == "grrhs_gibbs" and model_cfg.get("use_groups") is False:
        return "RHS"
    return aliases.get(name, name)


def _ordered_group_labels(groups: Sequence[Sequence[int]]) -> List[str]:
    canonical = list(GROUP_DISPLAY_NAMES.values())
    return [canonical[idx] if idx < len(canonical) else f"Group {idx + 1}" for idx, _ in enumerate(groups)]


def _feature_labels(feature_names: Sequence[str]) -> List[str]:
    return [PAPER_LABELS.get(name, name) for name in feature_names]


def _resolve_path(path_str: str, root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _iter_ok_runs(summary_payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for run in summary_payload.get("runs", []):
        if str(run.get("status", "")).upper() == "OK":
            yield run


def _build_run_context(run_payload: Dict[str, Any], repo_root: Path) -> RunContext:
    run_dir = Path(run_payload["run_dir"]).resolve()
    summary = _load_json(run_dir / "summary.json")
    repeat_dir = sorted(path for path in run_dir.glob("repeat_*") if path.is_dir())[0]
    meta = _load_json(repeat_dir / "dataset_meta.json")
    feature_names = list(meta.get("feature_names") or [])
    groups = [list(map(int, members)) for members in (meta.get("groups") or [])]
    metrics_summary = summary.get("metrics_summary", {})
    metrics_mean = {name: float(stats["mean"]) for name, stats in metrics_summary.items() if isinstance(stats, dict) and "mean" in stats}
    metrics_std = {name: float(stats["std"]) for name, stats in metrics_summary.items() if isinstance(stats, dict) and "std" in stats}
    resolved_cfg = _load_yaml(run_dir / "resolved_config.yaml") if (run_dir / "resolved_config.yaml").exists() else {}
    slab_width = float(resolved_cfg.get("model", {}).get("c", 1.0))
    x_path = _resolve_path(meta["metadata"]["paths"]["path_X"], repo_root)
    return RunContext(
        label=_model_label(run_dir),
        run_dir=run_dir,
        run_payload=run_payload,
        metrics_mean=metrics_mean,
        metrics_std=metrics_std,
        groups=groups,
        feature_names=feature_names,
        feature_labels=_feature_labels(feature_names),
        group_labels=_ordered_group_labels(groups),
        slab_width=slab_width,
        x_path=x_path,
    )


def _standardize_subset(
    X_raw: np.ndarray,
    indices: np.ndarray,
    x_mean: np.ndarray,
    x_scale: np.ndarray,
) -> np.ndarray:
    return (X_raw[indices] - x_mean[None, :]) / np.maximum(x_scale[None, :], 1e-8)


def _group_sums(values: np.ndarray, groups: Sequence[Sequence[int]]) -> np.ndarray:
    return np.asarray([float(np.sum(values[np.asarray(group, dtype=int)])) for group in groups], dtype=float)


def _compute_group_activation_scores(
    X_train_std: np.ndarray,
    sigma_draws: np.ndarray,
    tau_draws: np.ndarray,
    lambda_draws: np.ndarray,
    groups: Sequence[Sequence[int]],
    slab_width: float,
    phi_draws: Optional[np.ndarray] = None,
) -> np.ndarray:
    xtx_diag = np.sum(np.square(X_train_std), axis=0)
    p = xtx_diag.shape[0]
    group_index = np.empty(p, dtype=int)
    for gid, members in enumerate(groups):
        group_index[np.asarray(members, dtype=int)] = gid

    scores = np.zeros(len(groups), dtype=float)
    for draw_idx in range(lambda_draws.shape[0]):
        tau_t = float(max(tau_draws[draw_idx], 1e-12))
        sigma_t = float(max(sigma_draws[draw_idx], 1e-12))
        tilde_sq = regularized_lambda(lambda_draws[draw_idx], tau_t, slab_width)
        if phi_draws is None:
            phi_j = np.ones(p, dtype=float)
        else:
            phi_j = np.asarray(phi_draws[draw_idx], dtype=float)[group_index]
        prior_prec = prior_precision(phi_j, tau_t, tilde_sq, sigma_t)
        kappa = shrinkage_kappa(xtx_diag, sigma_t ** 2, prior_prec)
        scores += _group_sums(1.0 - kappa, groups)
    return scores / float(lambda_draws.shape[0])


def _collect_structure_statistics(context: RunContext, x_cache: Dict[Path, np.ndarray]) -> Dict[str, Any]:
    X_raw = x_cache.setdefault(context.x_path, np.load(context.x_path).astype(float))
    coef_mass_folds: List[np.ndarray] = []
    pred_var_folds: List[np.ndarray] = []
    pred_abs_folds: List[np.ndarray] = []
    kappa_group_folds: List[np.ndarray] = []
    phi_pool: List[np.ndarray] = []

    for repeat_dir in sorted(path for path in context.run_dir.glob("repeat_*") if path.is_dir()):
        for fold_dir in sorted(path for path in repeat_dir.glob("fold_*") if path.is_dir()):
            posterior_path = fold_dir / "posterior_samples.npz"
            fold_arrays_path = fold_dir / "fold_arrays.npz"
            if not posterior_path.exists() or not fold_arrays_path.exists():
                continue

            posterior = np.load(posterior_path)
            beta_draws = np.asarray(posterior["beta"], dtype=float)
            beta_abs_mean = np.mean(np.abs(beta_draws), axis=0)
            beta_mean = np.mean(beta_draws, axis=0)
            coef_mass_folds.append(beta_abs_mean)

            fold_arrays = np.load(fold_arrays_path)
            test_idx = np.asarray(fold_arrays["test_idx"], dtype=int)
            x_mean = np.asarray(fold_arrays["x_mean"], dtype=float)
            x_scale = np.asarray(fold_arrays["x_scale"], dtype=float)
            X_test_std = _standardize_subset(X_raw, test_idx, x_mean, x_scale)

            group_pred_var = []
            group_pred_abs = []
            for group in context.groups:
                group_idx = np.asarray(group, dtype=int)
                contrib = X_test_std[:, group_idx] @ beta_mean[group_idx]
                if contrib.size <= 1:
                    group_pred_var.append(float(np.mean(np.square(contrib))))
                else:
                    group_pred_var.append(float(np.var(contrib, ddof=1)))
                group_pred_abs.append(float(np.mean(np.abs(contrib))))
            pred_var_folds.append(np.asarray(group_pred_var, dtype=float))
            pred_abs_folds.append(np.asarray(group_pred_abs, dtype=float))

            if context.label in {"RHS", "GR-RHS"} and "lambda" in posterior and "tau" in posterior:
                train_idx = np.asarray(fold_arrays["train_idx"], dtype=int)
                X_train_std = _standardize_subset(X_raw, train_idx, x_mean, x_scale)
                tau_draws = np.asarray(posterior["tau"], dtype=float)
                lambda_draws = np.asarray(posterior["lambda"], dtype=float)
                if "sigma2" in posterior:
                    sigma_draws = np.sqrt(np.maximum(np.asarray(posterior["sigma2"], dtype=float), 1e-12))
                else:
                    sigma_draws = np.asarray(posterior["sigma"], dtype=float)
                phi_draws = np.asarray(posterior["phi"], dtype=float) if "phi" in posterior else None
                kappa_group_folds.append(
                    _compute_group_activation_scores(
                        X_train_std=X_train_std,
                        sigma_draws=sigma_draws,
                        tau_draws=tau_draws,
                        lambda_draws=lambda_draws,
                        groups=context.groups,
                        slab_width=context.slab_width,
                        phi_draws=phi_draws,
                    )
                )

            if context.label == "GR-RHS" and "phi" in posterior:
                phi_pool.append(np.asarray(posterior["phi"], dtype=float))

    if not coef_mass_folds:
        raise RuntimeError(f"No posterior fold outputs found for {context.run_dir}")

    pred_var = np.mean(np.vstack(pred_var_folds), axis=0)
    pred_abs = np.mean(np.vstack(pred_abs_folds), axis=0)
    pred_var_share = pred_var / max(float(np.sum(pred_var)), 1e-12)
    pred_abs_share = pred_abs / max(float(np.sum(pred_abs)), 1e-12)

    return {
        "coef_abs_mean": np.mean(np.vstack(coef_mass_folds), axis=0),
        "group_coef_mass": np.mean(np.vstack([_group_sums(row, context.groups) for row in coef_mass_folds]), axis=0),
        "group_pred_var": pred_var,
        "group_pred_abs": pred_abs,
        "group_pred_var_share": pred_var_share,
        "group_pred_abs_share": pred_abs_share,
        "group_kappa_activity": np.mean(np.vstack(kappa_group_folds), axis=0) if kappa_group_folds else None,
        "phi_pool": np.concatenate(phi_pool, axis=0) if phi_pool else None,
    }


def _plot_complexity_tradeoff(
    perf_frame: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    panels = [
        ("EffectiveDoF", "Panel A. RMSE vs Effective DoF"),
        ("MeanEffectiveNonzeros", "Panel B. RMSE vs Mean effective nonzeros"),
    ]

    for ax, (x_col, title) in zip(axes, panels):
        for row in perf_frame.itertuples(index=False):
            color = MODEL_COLORS.get(row.label, "#4C78A8")
            ax.scatter(getattr(row, x_col), row.RMSE, s=90, color=color, edgecolor="white", linewidth=0.8, zorder=3)
            ax.text(getattr(row, x_col), row.RMSE, f"  {row.label}", fontsize=10, va="center")
        ax.set_xlabel("Effective DoF" if x_col == "EffectiveDoF" else "Mean effective nonzeros", fontsize=12)
        ax.set_title(title, fontsize=14, pad=10)
        ax.grid(alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("RMSE", fontsize=12)
    fig.suptitle("Figure 1. Predictive performance vs structural complexity", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_group_concentration(
    contexts: Dict[str, RunContext],
    stats_map: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> pd.DataFrame:
    models = ["RHS", "GR-RHS"]
    reference = stats_map["GR-RHS"]["group_kappa_activity"]
    if reference is None:
        raise RuntimeError("GR-RHS group kappa activity was not available.")
    order = np.argsort(reference)[::-1]
    ordered_groups = [contexts["GR-RHS"].group_labels[idx] for idx in order]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharey=True)
    rows: List[Dict[str, Any]] = []
    ymax = max(float(np.max(stats_map[model]["group_kappa_activity"])) for model in models if stats_map[model]["group_kappa_activity"] is not None)

    for ax, model in zip(axes, models):
        values = np.asarray(stats_map[model]["group_kappa_activity"], dtype=float)[order]
        ax.bar(np.arange(values.size), values, color=MODEL_COLORS.get(model, "#4C78A8"), edgecolor="white", linewidth=0.8)
        ax.set_xticks(np.arange(values.size))
        ax.set_xticklabels(ordered_groups, rotation=25, ha="right")
        ax.set_ylim(0.0, ymax * 1.12)
        ax.set_title(model, fontsize=14)
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)
        for group_name, value in zip(ordered_groups, values):
            rows.append({"model": model, "group": group_name, "group_activity_score": float(value)})

    axes[0].set_ylabel(r"Group activity score $\sum_{j \in g} \mathbb{E}[1-\kappa_j]$", fontsize=12)
    fig.suptitle("Figure 2. Group-level signal concentration", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def _plot_coefficient_heatmap(
    contexts: Dict[str, RunContext],
    stats_map: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> pd.DataFrame:
    reference_context = contexts["GR-RHS"]
    order: List[int] = []
    group_boundaries: List[int] = []
    current = 0
    for group in reference_context.groups:
        group_idx = np.asarray(group, dtype=int)
        group_values = stats_map["GR-RHS"]["coef_abs_mean"][group_idx]
        ordered_group = group_idx[np.argsort(group_values)[::-1]]
        order.extend(ordered_group.tolist())
        current += len(group_idx)
        group_boundaries.append(current)

    heat = np.vstack([stats_map[model]["coef_abs_mean"][order] for model in MAIN_STRUCTURE_MODELS])
    vmax = float(np.quantile(heat, 0.98)) if np.any(heat > 0) else 1.0

    fig, ax = plt.subplots(figsize=(16, 3.8))
    im = ax.imshow(heat, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=vmax)
    ax.set_yticks(np.arange(len(MAIN_STRUCTURE_MODELS)))
    ax.set_yticklabels(MAIN_STRUCTURE_MODELS, fontsize=12)
    ax.set_xticks([])
    ax.set_title("Figure 3. Coefficient heatmap ordered by exposure group", fontsize=16, pad=12)

    for boundary in group_boundaries[:-1]:
        ax.axvline(boundary - 0.5, color="white", linewidth=1.4, alpha=0.9)

    centers = []
    start = 0
    for boundary in group_boundaries:
        centers.append((start + boundary - 1) / 2.0)
        start = boundary

    ax_top = ax.secondary_xaxis("top")
    ax_top.set_xticks(centers)
    ax_top.set_xticklabels(reference_context.group_labels, rotation=20, ha="left", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(r"Posterior mean $|\beta_j|$", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    rows: List[Dict[str, Any]] = []
    ordered_feature_names = [reference_context.feature_names[idx] for idx in order]
    ordered_feature_labels = [reference_context.feature_labels[idx] for idx in order]
    feature_to_group = {}
    for group_name, members in zip(reference_context.group_labels, reference_context.groups):
        for idx in members:
            feature_to_group[int(idx)] = group_name
    for model in MAIN_STRUCTURE_MODELS:
        for idx, feat_name, feat_label in zip(order, ordered_feature_names, ordered_feature_labels):
            rows.append(
                {
                    "model": model,
                    "feature": feat_name,
                    "label": feat_label,
                    "group": feature_to_group[int(idx)],
                    "coef_abs_mean": float(stats_map[model]["coef_abs_mean"][idx]),
                }
            )
    return pd.DataFrame(rows)


def _plot_predictive_contribution(
    contexts: Dict[str, RunContext],
    stats_map: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> pd.DataFrame:
    models = ["RHS", "GR-RHS"]
    order = np.argsort(stats_map["GR-RHS"]["group_pred_var_share"])[::-1]
    ordered_groups = [contexts["GR-RHS"].group_labels[idx] for idx in order]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharey=True)
    rows: List[Dict[str, Any]] = []

    for ax, model in zip(axes, models):
        values = np.asarray(stats_map[model]["group_pred_var_share"], dtype=float)[order]
        ax.hlines(np.arange(values.size), 0.0, values, color=MODEL_COLORS.get(model, "#4C78A8"), linewidth=2.2)
        ax.scatter(values, np.arange(values.size), color=MODEL_COLORS.get(model, "#4C78A8"), s=72, zorder=3)
        ax.set_yticks(np.arange(values.size))
        ax.set_yticklabels(ordered_groups)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25, linewidth=0.8)
        ax.set_title(model, fontsize=14)
        ax.set_xlabel(r"Share of $\mathrm{Var}(f_g(X_{test}))$", fontsize=12)
        for group_name, value in zip(ordered_groups, values):
            rows.append({"model": model, "group": group_name, "predictive_contribution_share": float(value)})

    axes[0].set_ylabel("Exposure group", fontsize=12)
    fig.suptitle("Figure 4. Group-wise predictive contribution", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def _plot_phi_posterior(
    context: RunContext,
    phi_samples: np.ndarray,
    out_path: Path,
) -> pd.DataFrame:
    order = np.argsort(np.median(phi_samples, axis=0))[::-1]
    ordered_groups = [context.group_labels[idx] for idx in order]
    ordered_samples = [phi_samples[:, idx] for idx in order]

    fig, ax = plt.subplots(figsize=(11, 5.6))
    parts = ax.violinplot(ordered_samples, showmeans=False, showmedians=True, widths=0.9)
    for body in parts["bodies"]:
        body.set_facecolor(MODEL_COLORS["GR-RHS"])
        body.set_edgecolor("black")
        body.set_alpha(0.65)
    if "cmedians" in parts:
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.2)
    ax.set_xticks(np.arange(1, len(ordered_groups) + 1))
    ax.set_xticklabels(ordered_groups, rotation=25, ha="right")
    ax.set_ylabel(r"Posterior $\phi_g$ (larger = weaker shrinkage)", fontsize=12)
    ax.set_title("Supplementary Figure. GR-RHS group-shrinkage posterior distributions", fontsize=16, pad=12)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    rows: List[Dict[str, Any]] = []
    for group_name, samples in zip(ordered_groups, ordered_samples):
        rows.append(
            {
                "group": group_name,
                "phi_mean": float(np.mean(samples)),
                "phi_median": float(np.median(samples)),
                "phi_q05": float(np.quantile(samples, 0.05)),
                "phi_q95": float(np.quantile(samples, 0.95)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-style NHANES grouped-shrinkage figures from a completed sweep summary.")
    parser.add_argument("--sweep-summary", type=Path, required=True, help="Path to sweep_summary_*.json")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Destination directory. Defaults to <sweep_dir>/paper_group_figures",
    )
    args = parser.parse_args()

    sweep_summary = args.sweep_summary.expanduser().resolve()
    sweep_payload = _load_json(sweep_summary)
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = (args.out_dir.expanduser().resolve() if args.out_dir else sweep_summary.parent / "paper_group_figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    contexts = [_build_run_context(run, repo_root) for run in _iter_ok_runs(sweep_payload)]
    context_by_label = {ctx.label: ctx for ctx in contexts}

    perf_rows = []
    for ctx in contexts:
        perf_rows.append(
            {
                "label": ctx.label,
                "RMSE": ctx.metrics_mean.get("RMSE", np.nan),
                "MLPD": ctx.metrics_mean.get("MLPD", np.nan),
                "PredictiveLogLikelihood": ctx.metrics_mean.get("PredictiveLogLikelihood", np.nan),
                "EffectiveDoF": ctx.metrics_mean.get("EffectiveDoF", np.nan),
                "MeanEffectiveNonzeros": ctx.metrics_mean.get("MeanEffectiveNonzeros", np.nan),
            }
        )
    perf_frame = pd.DataFrame(perf_rows).sort_values("RMSE")
    perf_frame.to_csv(out_dir / "figure1_tradeoff_data.csv", index=False)
    _plot_complexity_tradeoff(perf_frame, out_dir / "figure1_predictive_complexity_tradeoff.png")

    x_cache: Dict[Path, np.ndarray] = {}
    structure_needed = {label: context_by_label[label] for label in MAIN_STRUCTURE_MODELS if label in context_by_label}
    missing_models = [label for label in MAIN_STRUCTURE_MODELS if label not in structure_needed]
    if missing_models:
        raise RuntimeError(f"Missing required models in sweep summary: {missing_models}")

    structure_stats = {label: _collect_structure_statistics(ctx, x_cache) for label, ctx in structure_needed.items()}

    fig2_data = _plot_group_concentration(structure_needed, structure_stats, out_dir / "figure2_group_signal_concentration.png")
    fig2_data.to_csv(out_dir / "figure2_group_signal_concentration_data.csv", index=False)

    fig3_data = _plot_coefficient_heatmap(structure_needed, structure_stats, out_dir / "figure3_group_ordered_coefficient_heatmap.png")
    fig3_data.to_csv(out_dir / "figure3_group_ordered_coefficient_heatmap_data.csv", index=False)

    fig4_data = _plot_predictive_contribution(structure_needed, structure_stats, out_dir / "figure4_group_predictive_contribution.png")
    fig4_data.to_csv(out_dir / "figure4_group_predictive_contribution_data.csv", index=False)

    grrhs_phi = structure_stats["GR-RHS"]["phi_pool"]
    if grrhs_phi is None:
        raise RuntimeError("GR-RHS phi posterior samples were not found.")
    supp_data = _plot_phi_posterior(structure_needed["GR-RHS"], grrhs_phi, out_dir / "supplementary_group_phi_posterior.png")
    supp_data.to_csv(out_dir / "supplementary_group_phi_posterior_data.csv", index=False)

    manifest = {
        "sweep_summary": str(sweep_summary),
        "output_dir": str(out_dir),
        "generated_files": sorted([path.name for path in out_dir.iterdir() if path.is_file()]),
        "models_in_tradeoff": perf_frame["label"].tolist(),
        "structure_models": list(structure_needed.keys()),
    }
    (out_dir / "paper_group_figures_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[ok] figure set written to {out_dir}")


if __name__ == "__main__":
    main()
