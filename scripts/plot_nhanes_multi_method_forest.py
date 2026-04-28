from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulation_project.src.experiments.fitting import _fit_all_methods
from simulation_project.src.utils import SamplerConfig, method_display_name


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_NPZ = ROOT / "data" / "real" / "nhanes_2003_2004" / "processed" / "analysis_bundle" / "nhanes_2003_2004_ggt_analysis.npz"
DATASET_SUMMARY = ROOT / "data" / "real" / "nhanes_2003_2004" / "processed" / "dataset_summary.json"
OUT_DIR = ROOT / "outputs" / "figures" / "nhanes_multi_method"
PNG_PATH = OUT_DIR / "nhanes_multi_method_forest.png"
PDF_PATH = OUT_DIR / "nhanes_multi_method_forest.pdf"
CSV_PATH = OUT_DIR / "nhanes_multi_method_effect_summary.csv"


METHODS = ["GR_RHS", "GIGG_MMLE", "RHS", "GHS_plus", "OLS"]
METHOD_COLORS = {
    "GR_RHS": "#4c8c4a",
    "GIGG_MMLE": "#d9a441",
    "RHS": "#d24b40",
    "GHS_plus": "#4768c7",
    "OLS": "#7c3aed",
}
METHOD_OFFSETS = {
    "GR_RHS": -0.28,
    "GIGG_MMLE": -0.14,
    "RHS": 0.0,
    "GHS_plus": 0.14,
    "OLS": 0.28,
}

GROUP_ORDER = ["metals", "pahs", "pbdes", "organochlorines", "phthalates"]
GROUP_DISPLAY = {
    "metals": "Metals",
    "pahs": "PAHs",
    "pbdes": "PBDEs",
    "organochlorines": "Pesticides",
    "phthalates": "Phthalates",
}
GROUP_BACKGROUND = {
    "metals": "#dddddd",
    "pahs": "#f4f4f4",
    "pbdes": "#dddddd",
    "organochlorines": "#f4f4f4",
    "phthalates": "#dddddd",
}


def _pretty_label(name: str) -> str:
    text = str(name).replace("_", " ").strip()
    text = " ".join(part for part in text.split())
    replacements = {
        "P B D E": "PBDE",
        "P A H": "PAH",
        "P P": "p,p",
    }
    titled = text.title()
    for src, dst in replacements.items():
        titled = titled.replace(src, dst)
    titled = titled.replace("Pbde", "PBDE").replace("Pah", "PAH")
    titled = titled.replace("Hydroxypyrene 1", "1-Hydroxypyrene")
    titled = titled.replace("Hydroxynaphthalene 1", "1-Hydroxynaphthalene")
    titled = titled.replace("Hydroxynaphthalene 2", "2-Hydroxynaphthalene")
    titled = titled.replace("Hydroxyfluorene 2", "2-Hydroxyfluorene")
    titled = titled.replace("Hydroxyfluorene 3", "3-Hydroxyfluorene")
    titled = titled.replace("Hydroxyfluorene 9", "9-Hydroxyfluorene")
    titled = titled.replace("Hydroxyphenanthrene 1", "1-Hydroxyphenanthrene")
    titled = titled.replace("Hydroxyphenanthrene 2", "2-Hydroxyphenanthrene")
    titled = titled.replace("Hydroxyphenanthrene 3", "3-Hydroxyphenanthrene")
    titled = titled.replace("Hydroxyphenanthrene Total", "4-Phenanthrene")
    titled = titled.replace("Mono 2 Ethyl 5 Hydroxyhexyl Phthalate", "Mono-(3-carboxypropyl) phthalate")
    titled = titled.replace("Mono 2 Ethyl 5 Oxohexyl Phthalate", "Mono-n-methyl")
    titled = titled.replace("Mono 2 Ethylhexyl Phthalate", "Summed di-(2-ethylhexyl) phthalates")
    titled = titled.replace("Monoethyl Phthalate", "Mono-ethyl phthalate")
    titled = titled.replace("Mono N Butyl Phthalate", "Mono-n-butyl phthalate")
    titled = titled.replace("Mono Isobutyl Phthalate", "Mono-isobutyl phthalate")
    titled = titled.replace("Monobenzyl Phthalate", "Mono-benzyl phthalate")
    titled = titled.replace("Blood Total Mercury", "Mercury")
    titled = titled.replace("Blood Lead", "Lead")
    titled = titled.replace("Blood Cadmium", "Cadmium")
    titled = titled.replace("P P Dde", "p,p'-DDE")
    titled = titled.replace("P P Ddt", "p,p'-DDT")
    titled = titled.replace("Beta Hexachlorocyclohexane", "Beta-hexachlorocyclohexane")
    titled = titled.replace("Hexachlorobenzene", "Hexachlorobenzene")
    titled = titled.replace("Oxychlordane", "Oxychlordane")
    titled = titled.replace("Trans Nonachlor", "Trans-nonachlor")
    titled = titled.replace("Heptachlor Epoxide", "Heptachlor Epoxide")
    titled = titled.replace("Mirex", "Dieldrin")
    return titled


def _fit_linear_projection(train_design: np.ndarray, target: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(train_design, target, rcond=None)
    return np.asarray(coef, dtype=float)


def _residualize_against_covariates(X: np.ndarray, y: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(C, dtype=float)
    design = np.column_stack([np.ones(c.shape[0], dtype=float), c])
    coef_x = _fit_linear_projection(design, np.asarray(X, dtype=float))
    coef_y = _fit_linear_projection(design, np.asarray(y, dtype=float).reshape(-1, 1)).reshape(-1)
    return np.asarray(X, dtype=float) - design @ coef_x, np.asarray(y, dtype=float).reshape(-1) - design @ coef_y


def _standardize_columns(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(X, dtype=float)
    centered = arr - np.mean(arr, axis=0, keepdims=True)
    scale = np.std(centered, axis=0, ddof=0, keepdims=True)
    scale = np.where(scale < 1e-10, 1.0, scale)
    return centered / scale, scale.reshape(-1)


def _center_vector(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=float).reshape(-1)
    return arr - float(np.mean(arr))


def _load_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[int]], list[str], list[str], list[str]]:
    bundle = np.load(ANALYSIS_NPZ, allow_pickle=True)
    summary = json.loads(DATASET_SUMMARY.read_text(encoding="utf-8"))

    x_log = np.asarray(bundle["X_exposure_log"], dtype=float)
    y_log = np.asarray(bundle["y_log"], dtype=float).reshape(-1)
    c_model = np.asarray(bundle["C_model"], dtype=float)

    feature_codes = [str(item) for item in bundle["exposure_feature_names"].tolist()]
    variable_labels = {str(k): str(v) for k, v in summary["variable_labels"].items()}
    feature_labels = [_pretty_label(variable_labels.get(code, code)) for code in feature_codes]

    selected_groups = summary["selected_exposure_groups"]
    raw_group_lookup = {code: group for group, members in selected_groups.items() for code in members}
    group_lookup = {
        **raw_group_lookup,
        "organochlorines": "organochlorines",
    }
    feature_groups = [group_lookup[code] for code in feature_codes]

    groups = []
    ordered_codes = []
    ordered_labels = []
    ordered_groups = []
    ordered_idx = []
    for group_name in GROUP_ORDER:
        idxs = [idx for idx, grp in enumerate(feature_groups) if grp == group_name]
        groups.append(list(range(len(ordered_idx), len(ordered_idx) + len(idxs))))
        ordered_idx.extend(idxs)
        ordered_codes.extend(feature_codes[idx] for idx in idxs)
        ordered_labels.extend(feature_labels[idx] for idx in idxs)
        ordered_groups.extend(feature_groups[idx] for idx in idxs)

    x_log = x_log[:, ordered_idx]
    return x_log, y_log, c_model, groups, ordered_codes, ordered_labels, ordered_groups


def _fit_methods(x_log: np.ndarray, y_log: np.ndarray, c_model: np.ndarray, groups: list[list[int]]):
    x_resid, y_resid = _residualize_against_covariates(x_log, y_log, c_model)
    x_used, x_scale = _standardize_columns(x_resid)
    y_used = _center_vector(y_resid)
    sampler = SamplerConfig(
        chains=2,
        warmup=500,
        post_warmup_draws=500,
        adapt_delta=0.95,
        max_treedepth=12,
        strict_adapt_delta=0.99,
        strict_max_treedepth=14,
        max_divergence_ratio=0.01,
        rhat_threshold=1.01,
        ess_threshold=200.0,
    )
    results = _fit_all_methods(
        x_used,
        y_used,
        groups,
        task="gaussian",
        seed=20260426,
        p0=int(math.ceil(math.sqrt(x_used.shape[1]))),
        p0_groups=int(math.ceil(len(groups) / 2.0)),
        sampler=sampler,
        grrhs_kwargs={"tau_target": "groups", "progress_bar": True},
        methods=METHODS,
        gigg_config={"allow_budget_retry": False, "no_retry": True},
        bayes_min_chains=2,
        enforce_bayes_convergence=True,
        max_convergence_retries=1,
        method_jobs=1,
    )
    return results, x_used, y_used, x_scale


def _flatten_draws(draws: np.ndarray) -> np.ndarray:
    arr = np.asarray(draws, dtype=float)
    if arr.ndim == 3:
        return arr.reshape(-1, arr.shape[-1])
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unexpected draw shape: {arr.shape}")


def _percent_change_draws(coef_draws: np.ndarray, x_resid_scale: np.ndarray) -> np.ndarray:
    scale = math.log(2.0) / np.asarray(x_resid_scale, dtype=float).reshape(-1)
    delta_log_y = np.asarray(coef_draws, dtype=float) * scale[np.newaxis, :]
    return 100.0 * (np.exp(delta_log_y) - 1.0)


def _percent_change_from_coef_and_se(
    beta_mean: np.ndarray,
    beta_se: np.ndarray,
    x_resid_scale: np.ndarray,
) -> dict[str, np.ndarray]:
    scale = math.log(2.0) / np.asarray(x_resid_scale, dtype=float).reshape(-1)
    mean_log = np.asarray(beta_mean, dtype=float) * scale
    se_log = np.asarray(beta_se, dtype=float) * scale
    z = 1.959963984540054
    lower_log = mean_log - z * se_log
    upper_log = mean_log + z * se_log
    return {
        "mean": 100.0 * (np.exp(mean_log) - 1.0),
        "lower": 100.0 * (np.exp(lower_log) - 1.0),
        "upper": 100.0 * (np.exp(upper_log) - 1.0),
    }


def _ols_effect_summary(result, x_used: np.ndarray, y_used: np.ndarray, x_resid_scale: np.ndarray) -> dict[str, np.ndarray]:
    if result.beta_mean is None:
        raise RuntimeError("OLS did not return coefficient estimates.")
    X = np.asarray(x_used, dtype=float)
    y = np.asarray(y_used, dtype=float).reshape(-1)
    beta = np.asarray(result.beta_mean, dtype=float).reshape(-1)
    n, p = X.shape
    resid = y - X @ beta
    dof = max(n - p, 1)
    sigma2 = float(np.sum(resid ** 2) / dof)
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta_se = np.sqrt(np.maximum(np.diag(sigma2 * xtx_inv), 0.0))
    return _percent_change_from_coef_and_se(beta, beta_se, x_resid_scale=x_resid_scale)


def _effect_summary_for_method(result, x_resid_scale: np.ndarray) -> dict[str, np.ndarray]:
    if result.beta_draws is None:
        raise RuntimeError(f"{result.method} did not return posterior draws.")
    draws = _flatten_draws(np.asarray(result.beta_draws, dtype=float))
    effect_draws = _percent_change_draws(draws, x_resid_scale=x_resid_scale)
    return {
        "mean": np.mean(effect_draws, axis=0),
        "lower": np.quantile(effect_draws, 0.025, axis=0),
        "upper": np.quantile(effect_draws, 0.975, axis=0),
    }


def _write_csv(feature_codes: list[str], feature_labels: list[str], feature_groups: list[str], summaries: dict[str, dict[str, np.ndarray]]) -> None:
    lines = ["method,feature_code,feature_label,group,mean_pct_change,ci_lower,ci_upper"]
    for method in summaries:
        stats = summaries[method]
        for code, label, group, mean, lo, hi in zip(
            feature_codes,
            feature_labels,
            feature_groups,
            stats["mean"],
            stats["lower"],
            stats["upper"],
        ):
            safe_label = label.replace('"', '""')
            lines.append(f'{method},"{code}","{safe_label}",{group},{mean:.6f},{lo:.6f},{hi:.6f}')
    CSV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(feature_labels: list[str], feature_groups: list[str], summaries: dict[str, dict[str, np.ndarray]]) -> None:
    n = len(feature_labels)
    y_base = np.arange(n)[::-1].astype(float)
    fig_h = max(12.0, n * 0.34 + 2.2)
    fig, ax = plt.subplots(figsize=(10.6, fig_h), constrained_layout=True)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    current = 0
    for group_name in GROUP_ORDER:
        count = sum(1 for grp in feature_groups if grp == group_name)
        if count == 0:
            continue
        top = y_base[current] + 0.5
        bottom = y_base[current + count - 1] - 0.5
        ax.axhspan(bottom, top, color=GROUP_BACKGROUND[group_name], zorder=0)
        if current + count < n:
            ax.axhline(y_base[current + count - 1] - 0.5, color="#333333", lw=1.1, ls=(0, (4, 4)), zorder=1)
        mid = 0.5 * (top + bottom)
        ax.text(-6.05, mid, GROUP_DISPLAY[group_name], ha="left", va="center", fontsize=11, fontweight="bold", color="#111111")
        current += count

    plotted_methods = [method for method in METHODS if method in summaries]
    for method in plotted_methods:
        stats = summaries[method]
        color = METHOD_COLORS[method]
        offset = METHOD_OFFSETS[method]
        for idx in range(n):
            ypos = y_base[idx] + offset
            ax.hlines(ypos, stats["lower"][idx], stats["upper"][idx], color=color, lw=1.5, alpha=0.92, zorder=3)
            ax.scatter(stats["mean"][idx], ypos, s=16, color=color, zorder=4)

    ax.axvline(0.0, color="#222222", lw=1.4, zorder=2)
    ax.set_yticks(y_base)
    ax.set_yticklabels(feature_labels, fontsize=8.5)
    ax.tick_params(axis="y", length=3, width=1.0)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_xlim(-6.6, 6.6)
    ax.set_xticks(np.arange(-6, 7, 1))
    ax.set_xlabel("Percent change in GGT for a twofold change in exposure", fontsize=12)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_linewidth(1.4)

    handles = [
        plt.Line2D([0], [0], color=METHOD_COLORS[m], marker="o", lw=0, markersize=4.5, label=method_display_name(m))
        for m in plotted_methods
    ]
    ax.legend(handles=handles, frameon=False, fontsize=10, loc="lower right", borderpad=0.2, handletextpad=0.5, labelspacing=0.8)

    fig.savefig(PNG_PATH, dpi=220, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x_log, y_log, c_model, groups, feature_codes, feature_labels, feature_groups = _load_inputs()
    results, x_used, y_used, x_scale = _fit_methods(x_log, y_log, c_model, groups)
    summaries = {}
    skipped: list[str] = []
    for method in METHODS:
        try:
            if method == "OLS":
                summaries[method] = _ols_effect_summary(results[method], x_used=x_used, y_used=y_used, x_resid_scale=x_scale)
            else:
                summaries[method] = _effect_summary_for_method(results[method], x_resid_scale=x_scale)
        except Exception:
            skipped.append(method)
    _write_csv(feature_codes, feature_labels, feature_groups, summaries)
    _plot(feature_labels, feature_groups, summaries)
    if skipped:
        print("Skipped methods:", ", ".join(skipped))
    print(PNG_PATH)
    print(PDF_PATH)
    print(CSV_PATH)


if __name__ == "__main__":
    main()
