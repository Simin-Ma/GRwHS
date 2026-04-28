from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulation_project.src.experiments.fitting import _fit_with_convergence_retry
from simulation_project.src.experiments.methods.fit_gr_rhs import fit_gr_rhs
from simulation_project.src.utils import SamplerConfig


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_NPZ = ROOT / "data" / "real" / "nhanes_2003_2004" / "processed" / "analysis_bundle" / "nhanes_2003_2004_ggt_analysis.npz"
DATASET_SUMMARY = ROOT / "data" / "real" / "nhanes_2003_2004" / "processed" / "dataset_summary.json"
OUT_DIR = ROOT / "outputs" / "figures" / "nhanes_grrhs_high_budget"
PNG_PATH = OUT_DIR / "nhanes_grrhs_high_budget_forest.png"
PDF_PATH = OUT_DIR / "nhanes_grrhs_high_budget_forest.pdf"
CSV_PATH = OUT_DIR / "nhanes_grrhs_high_budget_effect_summary.csv"
JSON_PATH = OUT_DIR / "nhanes_grrhs_high_budget_fit_summary.json"


GROUP_ORDER = ["metals", "phthalates", "organochlorines", "pbdes", "pahs"]
GROUP_COLORS = {
    "metals": "#9a3412",
    "phthalates": "#0f766e",
    "organochlorines": "#a16207",
    "pbdes": "#1d4ed8",
    "pahs": "#be123c",
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
    titled = titled.replace("Hydroxyphenanthrene Total", "Total Hydroxyphenanthrene")
    titled = titled.replace("Mono 2 Ethyl 5 Hydroxyhexyl Phthalate", "MEHHP")
    titled = titled.replace("Mono 2 Ethyl 5 Oxohexyl Phthalate", "MEOHP")
    titled = titled.replace("Mono 2 Ethylhexyl Phthalate", "MEHP")
    titled = titled.replace("Monoethyl Phthalate", "MEP")
    titled = titled.replace("Mono N Butyl Phthalate", "MnBP")
    titled = titled.replace("Mono Isobutyl Phthalate", "MiBP")
    titled = titled.replace("Monobenzyl Phthalate", "MBzP")
    titled = titled.replace("Blood Total Mercury", "Blood Mercury")
    titled = titled.replace("P P Dde", "p,p-DDE")
    titled = titled.replace("P P Ddt", "p,p-DDT")
    titled = titled.replace("Beta Hexachlorocyclohexane", "beta-HCH")
    titled = titled.replace("Trans Nonachlor", "trans-Nonachlor")
    return titled


def _fit_linear_projection(train_design: np.ndarray, target: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(train_design, target, rcond=None)
    return np.asarray(coef, dtype=float)


def _residualize_against_covariates(X: np.ndarray, y: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(C, dtype=float)
    design = np.column_stack([np.ones(c.shape[0], dtype=float), c])
    coef_x = _fit_linear_projection(design, np.asarray(X, dtype=float))
    coef_y = _fit_linear_projection(design, np.asarray(y, dtype=float).reshape(-1, 1)).reshape(-1)
    x_hat = design @ coef_x
    y_hat = design @ coef_y
    return np.asarray(X, dtype=float) - x_hat, np.asarray(y, dtype=float).reshape(-1) - y_hat


def _standardize_columns(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(X, dtype=float)
    centered = arr - np.mean(arr, axis=0, keepdims=True)
    scale = np.std(centered, axis=0, ddof=0, keepdims=True)
    scale = np.where(scale < 1e-10, 1.0, scale)
    return centered / scale, scale.reshape(-1)


def _center_vector(y: np.ndarray) -> tuple[np.ndarray, float]:
    arr = np.asarray(y, dtype=float).reshape(-1)
    offset = float(np.mean(arr))
    return arr - offset, offset


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
    group_lookup = {code: group for group, members in selected_groups.items() for code in members}
    feature_groups = [group_lookup[code] for code in feature_codes]

    groups: list[list[int]] = []
    for group_name in GROUP_ORDER:
        idxs = [idx for idx, grp in enumerate(feature_groups) if grp == group_name]
        groups.append(idxs)

    return x_log, y_log, c_model, groups, feature_codes, feature_labels, feature_groups


def _fit_model_high_budget(
    x_log: np.ndarray,
    y_log: np.ndarray,
    c_model: np.ndarray,
    groups: list[list[int]],
):
    x_resid_raw, y_resid_raw = _residualize_against_covariates(x_log, y_log, c_model)
    x_used, x_scale = _standardize_columns(x_resid_raw)
    y_used, _ = _center_vector(y_resid_raw)

    sampler = SamplerConfig(
        chains=4,
        warmup=1000,
        post_warmup_draws=1000,
        adapt_delta=0.97,
        max_treedepth=13,
        strict_adapt_delta=0.995,
        strict_max_treedepth=15,
        max_divergence_ratio=0.01,
        rhat_threshold=1.015,
        ess_threshold=150.0,
    )
    p0_groups = int(math.ceil(len(groups) / 2.0))

    def _fit_once(sampler_try: SamplerConfig, attempt: int, resume_payload=None):
        return fit_gr_rhs(
            x_used,
            y_used,
            groups,
            task="gaussian",
            seed=20260426,
            p0=p0_groups,
            sampler=sampler_try,
            tau_target="groups",
            progress_bar=True,
            retry_resume_payload=resume_payload,
            retry_attempt=attempt,
        )

    result = _fit_with_convergence_retry(
        _fit_once,
        method="GR_RHS",
        sampler=sampler,
        bayes_min_chains=4,
        max_convergence_retries=3,
        enforce_bayes_convergence=True,
        continue_on_retry=True,
    )
    return result, x_used, y_used, x_scale


def _percent_change_draws(coef_draws: np.ndarray, x_resid_scale: np.ndarray) -> np.ndarray:
    scale = math.log(2.0) / np.asarray(x_resid_scale, dtype=float).reshape(-1)
    delta_log_y = np.asarray(coef_draws, dtype=float) * scale[np.newaxis, :]
    return 100.0 * (np.exp(delta_log_y) - 1.0)


def _write_summary_csv(
    *,
    feature_codes: list[str],
    feature_labels: list[str],
    feature_groups: list[str],
    means: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    lines = ["feature_code,feature_label,group,mean_pct_change,ci_lower,ci_upper"]
    for code, label, group, mean, lo, hi in zip(feature_codes, feature_labels, feature_groups, means, lower, upper):
        safe_label = label.replace('"', '""')
        lines.append(f'{code},"{safe_label}",{group},{mean:.6f},{lo:.6f},{hi:.6f}')
    CSV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(
    *,
    feature_labels: list[str],
    feature_groups: list[str],
    means: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    n = len(feature_labels)
    y_pos = np.arange(n)[::-1]
    fig_h = max(12.0, n * 0.34 + 1.8)
    fig, ax = plt.subplots(figsize=(10.8, fig_h), constrained_layout=True)
    fig.patch.set_facecolor("#f8f6f0")
    ax.set_facecolor("#fcfbf7")

    for idx, (_, group, mean, lo, hi) in enumerate(zip(feature_labels, feature_groups, means, lower, upper)):
        ypos = y_pos[idx]
        color = GROUP_COLORS[group]
        ax.hlines(ypos, lo, hi, color=color, lw=2.2, alpha=0.95)
        ax.scatter(mean, ypos, s=40, color=color, edgecolor="#111827", linewidth=0.45, zorder=3)

    ax.axvline(0.0, color="#111827", lw=1.2, alpha=0.8)

    current = 0
    for group_name in GROUP_ORDER:
        count = sum(1 for grp in feature_groups if grp == group_name)
        if count == 0:
            continue
        top = y_pos[current]
        bottom = y_pos[current + count - 1]
        mid = 0.5 * (top + bottom)
        ax.text(
            ax.get_xlim()[0] if ax.has_data() else -5.0,
            mid,
            group_name.upper(),
            color=GROUP_COLORS[group_name],
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="center",
            bbox={"facecolor": "#fcfbf7", "edgecolor": "none", "pad": 1.2},
        )
        if current + count < n:
            ax.axhline(y_pos[current + count - 1] - 0.5, color="#d6d3d1", lw=0.8, ls="--", zorder=0)
        current += count

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_labels, fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="x", color="#e7e5e4", lw=0.8)
    ax.set_axisbelow(True)

    xmin = float(np.min(lower))
    xmax = float(np.max(upper))
    pad = max(1.0, 0.08 * (xmax - xmin))
    ax.set_xlim(xmin - pad, xmax + pad)

    ax.set_xlabel("Estimated % change in GGT for a twofold increase in exposure", fontsize=11)
    ax.set_title("NHANES 2003-2004: GR-RHS Exposure Effects (High Budget)", fontsize=14, pad=12, fontweight="bold")
    ax.text(
        0.0,
        1.01,
        "Posterior mean with 95% credible interval; high-budget NHANES figure run with relaxed gate",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#44403c",
        ha="left",
    )

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#78716c")

    fig.savefig(PNG_PATH, dpi=220, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x_log, y_log, c_model, groups, feature_codes, feature_labels, feature_groups = _load_inputs()
    result, x_used, _y_used, x_scale = _fit_model_high_budget(x_log, y_log, c_model, groups)
    if result.beta_draws is None:
        raise RuntimeError(f"GR_RHS high-budget fit did not return coefficient draws: {result.error}")

    coef_draws = np.asarray(result.beta_draws, dtype=float)
    if coef_draws.ndim == 3:
        coef_draws = coef_draws.reshape(-1, coef_draws.shape[-1])
    effect_draws = _percent_change_draws(coef_draws, x_resid_scale=x_scale)

    means = np.mean(effect_draws, axis=0)
    lower = np.quantile(effect_draws, 0.025, axis=0)
    upper = np.quantile(effect_draws, 0.975, axis=0)

    _write_summary_csv(
        feature_codes=feature_codes,
        feature_labels=feature_labels,
        feature_groups=feature_groups,
        means=means,
        lower=lower,
        upper=upper,
    )
    _plot(
        feature_labels=feature_labels,
        feature_groups=feature_groups,
        means=means,
        lower=lower,
        upper=upper,
    )

    fit_summary = {
        "method": "GR_RHS",
        "status": str(result.status),
        "converged": bool(result.converged),
        "runtime_seconds": float(result.runtime_seconds),
        "rhat_max": float(result.rhat_max),
        "bulk_ess_min": float(result.bulk_ess_min),
        "divergence_ratio": float(result.divergence_ratio),
        "n": int(x_used.shape[0]),
        "p": int(x_used.shape[1]),
        "group_sizes": [int(len(group)) for group in groups],
        "sampler_budget": {
            "chains": 4,
            "warmup": 1000,
            "post_warmup_draws": 1000,
            "adapt_delta": 0.97,
            "max_treedepth": 13,
            "strict_adapt_delta": 0.995,
            "strict_max_treedepth": 15,
            "max_convergence_retries": 3,
        },
        "gate": {
            "rhat_threshold": 1.015,
            "ess_threshold": 150.0,
            "max_divergence_ratio": 0.01,
        },
        "preprocessing": {
            "y": "log then residualize on covariates then center",
            "X": "log then residualize on covariates then standardize",
            "covariates": "age, sex, BMI, poverty-to-income ratio, ethnicity dummies, log urinary creatinine",
        },
        "effect_scale_note": "Effect conversion uses log(2) divided by the residualized log-exposure standard deviation for each feature.",
        "diagnostics": result.diagnostics,
    }
    JSON_PATH.write_text(json.dumps(fit_summary, indent=2, default=str), encoding="utf-8")

    print(PNG_PATH)
    print(PDF_PATH)
    print(CSV_PATH)
    print(JSON_PATH)


if __name__ == "__main__":
    main()
