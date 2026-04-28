from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulation_project.src.core.models.gigg_regression import GIGGRegression


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_NPZ = ROOT / "data" / "real" / "nhanes_2003_2004" / "processed" / "analysis_bundle" / "nhanes_2003_2004_ggt_analysis.npz"
DATASET_SUMMARY = ROOT / "data" / "real" / "nhanes_2003_2004" / "processed" / "dataset_summary.json"
OUT_DIR = ROOT / "outputs" / "figures" / "nhanes_gigg_mmle_paper_style"
PNG_PATH = OUT_DIR / "nhanes_gigg_mmle_paper_style_forest.png"
PDF_PATH = OUT_DIR / "nhanes_gigg_mmle_paper_style_forest.pdf"
CSV_PATH = OUT_DIR / "nhanes_gigg_mmle_paper_style_effect_summary.csv"
JSON_PATH = OUT_DIR / "nhanes_gigg_mmle_paper_style_fit_summary.json"


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


def _load_inputs() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[list[int]],
    list[str],
    list[str],
    list[str],
    list[str],
]:
    bundle = np.load(ANALYSIS_NPZ, allow_pickle=True)
    summary = json.loads(DATASET_SUMMARY.read_text(encoding="utf-8"))

    x_scaled = np.asarray(bundle["X_exposure_scaled"], dtype=float)
    x_log = np.asarray(bundle["X_exposure_log"], dtype=float)
    y_z = np.asarray(bundle["y_zscore"], dtype=float).reshape(-1)
    y_log = np.asarray(bundle["y_log"], dtype=float).reshape(-1)
    c_model = np.asarray(bundle["C_model"], dtype=float)
    covariate_names = [str(item) for item in bundle["covariate_feature_names"].tolist()]

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

    return x_scaled, x_log, y_z, y_log, c_model, groups, feature_codes, feature_labels, feature_groups, covariate_names


def _fit_model(
    x_scaled: np.ndarray,
    y_z: np.ndarray,
    c_model: np.ndarray,
    groups: list[list[int]],
) -> GIGGRegression:
    model = GIGGRegression(
        method="mmle",
        n_burn_in=10000,
        n_samples=10000,
        n_thin=1,
        seed=20260426,
        num_chains=1,
        fit_intercept=False,
        store_lambda=True,
        tau_sq_init=1.0,
        btrick=False,
        stable_solve=True,
        mmle_burnin_only=False,
        init_strategy="zero",
        init_ridge=1.0,
        init_scale_blend=0.0,
        randomize_group_order=False,
        lambda_vectorized_update=False,
        extra_beta_refresh_prob=0.0,
        mmle_step_size=1.0,
        mmle_update_every=1,
        mmle_window=1,
        lambda_constraint_mode="none",
        q_constraint_mode="hard",
        progress_bar=True,
    )
    return model.fit(x_scaled, y_z, groups=groups, C=c_model, method="mmle")


def _percent_change_draws(
    coef_draws: np.ndarray,
    y_log: np.ndarray,
    x_log: np.ndarray,
) -> np.ndarray:
    sd_y = np.std(np.asarray(y_log, dtype=float), ddof=0)
    sd_x = np.std(np.asarray(x_log, dtype=float), axis=0, ddof=0)
    scale = math.log(2.0) * sd_y / sd_x
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


def _write_fit_summary(
    *,
    x_scaled: np.ndarray,
    y_z: np.ndarray,
    c_model: np.ndarray,
    groups: list[list[int]],
    covariate_names: list[str],
    means: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    payload = {
        "method": "GIGG_MMLE",
        "route": "nhanes_paper_style",
        "n": int(x_scaled.shape[0]),
        "p_exposure": int(x_scaled.shape[1]),
        "p_covariate": int(c_model.shape[1]),
        "group_sizes": [int(len(g)) for g in groups],
        "preprocessing": {
            "y": "log-transform then standardize",
            "X": "log-transform then standardize",
            "covariates": "included directly in model via C",
            "external_residualization": False,
        },
        "sampler_budget": {
            "burn_in": 10000,
            "posterior_draws": 10000,
            "thin": 1,
            "num_chains": 1,
            "seed": 20260426,
        },
        "effect_scale_note": "Percent change corresponds to a twofold increase in exposure using log(2) * sd(log GGT) / sd(log exposure).",
        "covariate_feature_names": covariate_names,
        "effect_range": {
            "mean_min": float(np.min(means)),
            "mean_max": float(np.max(means)),
            "ci_lower_min": float(np.min(lower)),
            "ci_upper_max": float(np.max(upper)),
        },
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    ax.set_title("NHANES 2003-2004: GIGG-MMLE Paper-Style Effects", fontsize=14, pad=12, fontweight="bold")
    ax.text(
        0.0,
        1.01,
        "95% credible intervals; log-transform + standardize; covariates included directly in the model",
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
    x_scaled, x_log, y_z, y_log, c_model, groups, feature_codes, feature_labels, feature_groups, covariate_names = _load_inputs()
    model = _fit_model(x_scaled, y_z, c_model, groups)
    if model.coef_samples_ is None:
        raise RuntimeError("GIGG_MMLE fit did not return coefficient draws.")

    coef_draws = np.asarray(model.coef_samples_, dtype=float)
    if coef_draws.ndim == 3:
        coef_draws = coef_draws.reshape(-1, coef_draws.shape[-1])
    effect_draws = _percent_change_draws(coef_draws, y_log=y_log, x_log=x_log)

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
    _write_fit_summary(
        x_scaled=x_scaled,
        y_z=y_z,
        c_model=c_model,
        groups=groups,
        covariate_names=covariate_names,
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

    print(PNG_PATH)
    print(PDF_PATH)
    print(CSV_PATH)
    print(JSON_PATH)


if __name__ == "__main__":
    main()
