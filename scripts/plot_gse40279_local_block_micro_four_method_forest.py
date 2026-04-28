from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = (
    ROOT
    / "outputs"
    / "history"
    / "real_data_experiment"
    / "gse40279_micro_local_block_gap1000_six_methods"
    / "20260427_085416_815954"
)
OUT_DIR = ROOT / "outputs" / "figures" / "gse40279_local_block_micro_four_methods"
ANALYSIS_DIR = (
    ROOT
    / "data"
    / "real"
    / "gse40279_methylation_age"
    / "processed"
    / "analysis_bundle"
)
DATASET_ID = "gse40279_age_local_block_micro_gap1000"
REPLICATE_ID = "rep_001"

METHODS = ["OLS", "GHS_plus", "RHS", "GR_RHS"]
METHOD_DISPLAY = {
    "OLS": "OLS",
    "GHS_plus": "Grouped Horseshoe+",
    "RHS": "RHS",
    "GR_RHS": "GR-RHS",
}
METHOD_COLORS = {
    "OLS": "#7c3aed",
    "GHS_plus": "#4768c7",
    "RHS": "#d24b40",
    "GR_RHS": "#4c8c4a",
}
METHOD_OFFSETS = {
    "OLS": -0.27,
    "GHS_plus": -0.09,
    "RHS": 0.09,
    "GR_RHS": 0.27,
}
GROUP_COLORS = {
    0: "#8b5e3c",
    1: "#2f6b7c",
    2: "#4d6a2f",
    3: "#8c3b3b",
    4: "#5a4fcf",
    5: "#9a7a13",
}


def _fit_dir(method: str) -> Path:
    return RUN_DIR / "fit_details" / DATASET_ID / REPLICATE_ID / method / "fit"


def _split_dir() -> Path:
    return RUN_DIR / "splits" / DATASET_ID / REPLICATE_ID


def _load_feature_metadata() -> dict[str, dict[str, object]]:
    meta_path = ANALYSIS_DIR / "micro_local_block_gap1000_selected_features.tsv"
    out: dict[str, dict[str, object]] = {}
    with meta_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            probe_id = str(row["probe_id"])
            out[probe_id] = {
                "variance": float(row["variance"]),
                "chromosome": str(row["chromosome"]),
                "position": int(row["position"]),
                "group_label": str(row["group_label"]),
                "anchor_gene": str(row["anchor_gene"]),
                "group_id": int(row["group_id"]),
            }
    return out


def _load_split_metadata() -> dict[str, object]:
    path = _split_dir() / "split_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _display_label(feature_name: str, meta: dict[str, object]) -> str:
    gene = str(meta.get("anchor_gene", "")).strip()
    chrom = str(meta.get("chromosome", "")).strip()
    pos = int(meta.get("position", 0))
    if gene:
        return f"{feature_name} | {gene} | chr{chrom}:{pos}"
    return f"{feature_name} | chr{chrom}:{pos}"


def _flatten_draws(draws: np.ndarray) -> np.ndarray:
    arr = np.asarray(draws, dtype=float)
    if arr.ndim == 3:
        return arr.reshape(-1, arr.shape[-1])
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unexpected draw shape: {arr.shape}")


def _bayes_summary(method: str) -> dict[str, np.ndarray]:
    draws_path = _fit_dir(method) / "posterior_draws.npz"
    draws_npz = np.load(draws_path, allow_pickle=True)
    beta_draws = _flatten_draws(np.asarray(draws_npz["beta_draws"], dtype=float))
    return {
        "mean": np.mean(beta_draws, axis=0),
        "lower": np.quantile(beta_draws, 0.025, axis=0),
        "upper": np.quantile(beta_draws, 0.975, axis=0),
        "sd": np.std(beta_draws, axis=0, ddof=1),
    }


def _ols_summary() -> dict[str, np.ndarray]:
    split_dir = _split_dir()
    X_train = np.load(split_dir / "X_train_used.npy")
    y_train = np.load(split_dir / "y_train_used.npy")
    beta = np.load(_fit_dir("OLS") / "beta_mean.npy").reshape(-1)
    resid = np.asarray(y_train, dtype=float).reshape(-1) - np.asarray(X_train, dtype=float) @ beta
    n, p = X_train.shape
    dof = max(n - p, 1)
    sigma2 = float(np.sum(resid ** 2) / dof)
    xtx_inv = np.linalg.pinv(np.asarray(X_train, dtype=float).T @ np.asarray(X_train, dtype=float))
    beta_se = np.sqrt(np.maximum(np.diag(sigma2 * xtx_inv), 0.0))
    z = 1.959963984540054
    return {
        "mean": beta,
        "lower": beta - z * beta_se,
        "upper": beta + z * beta_se,
        "sd": beta_se,
    }


def _load_method_summaries() -> dict[str, dict[str, np.ndarray]]:
    out = {"OLS": _ols_summary()}
    for method in ("GHS_plus", "RHS", "GR_RHS"):
        out[method] = _bayes_summary(method)
    return out


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_rows(top_k: int = 30) -> tuple[list[dict[str, object]], list[str]]:
    split_meta = _load_split_metadata()
    feature_names = [str(name) for name in split_meta["feature_names"]]
    feature_meta = _load_feature_metadata()
    summaries = _load_method_summaries()

    grrhs_order = np.argsort(np.abs(summaries["GR_RHS"]["mean"]))[::-1][:top_k]
    selected_features = [feature_names[int(idx)] for idx in grrhs_order]
    rows: list[dict[str, object]] = []

    for rank, idx in enumerate(grrhs_order, start=1):
        feature_name = feature_names[int(idx)]
        meta = feature_meta[feature_name]
        for method in METHODS:
            rows.append(
                {
                    "feature_rank_grrhs_abs_mean": rank,
                    "method": method,
                    "method_label": METHOD_DISPLAY[method],
                    "feature_index": int(idx),
                    "feature_name": feature_name,
                    "display_label": _display_label(feature_name, meta),
                    "estimate": float(summaries[method]["mean"][idx]),
                    "ci_lower_95": float(summaries[method]["lower"][idx]),
                    "ci_upper_95": float(summaries[method]["upper"][idx]),
                    "estimate_sd_or_se": float(summaries[method]["sd"][idx]),
                    "group_id": int(meta["group_id"]),
                    "group_label": str(meta["group_label"]),
                    "anchor_gene": str(meta["anchor_gene"]),
                    "chromosome": str(meta["chromosome"]),
                    "position": int(meta["position"]),
                    "variance": float(meta["variance"]),
                }
            )
    return rows, selected_features


def _plot(rows: list[dict[str, object]], *, png_path: Path, pdf_path: Path) -> None:
    ordered = sorted(
        {row["feature_rank_grrhs_abs_mean"]: row for row in rows}.values(),
        key=lambda item: int(item["feature_rank_grrhs_abs_mean"]),
    )
    ordered = list(reversed(ordered))
    labels = [str(item["display_label"]) for item in ordered]
    feature_names = [str(item["feature_name"]) for item in ordered]
    y_base = np.arange(len(ordered), dtype=float)

    fig_h = max(10.0, len(ordered) * 0.42 + 2.2)
    fig, ax = plt.subplots(figsize=(13.2, fig_h), constrained_layout=True)
    fig.patch.set_facecolor("#fbfaf6")
    ax.set_facecolor("#fffdfa")

    row_lookup = {(str(row["feature_name"]), str(row["method"])): row for row in rows}
    for i, feature_name in enumerate(feature_names):
        gid = int(row_lookup[(feature_name, "GR_RHS")]["group_id"])
        ax.axhspan(i - 0.5, i + 0.5, color=GROUP_COLORS.get(gid, "#e7e5e4"), alpha=0.08, zorder=0)
        for method in METHODS:
            row = row_lookup[(feature_name, method)]
            ypos = y_base[i] + METHOD_OFFSETS[method]
            ax.hlines(
                ypos,
                float(row["ci_lower_95"]),
                float(row["ci_upper_95"]),
                color=METHOD_COLORS[method],
                lw=1.8,
                alpha=0.94,
                zorder=2,
            )
            ax.scatter(
                float(row["estimate"]),
                ypos,
                s=24,
                color=METHOD_COLORS[method],
                edgecolor="#111827",
                linewidth=0.35,
                zorder=3,
            )

    ax.axvline(0.0, color="#111827", lw=1.2, alpha=0.9, zorder=1)
    ax.grid(axis="x", color="#e7e5e4", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_yticks(y_base)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_xlabel("Age effect (years) per +1 SD CpG methylation", fontsize=11)
    ax.set_title("GSE40279 Local-Block Micro: OLS vs Grouped Horseshoe+ vs RHS vs GR-RHS", fontsize=14, pad=12)
    ax.text(
        0.0,
        1.01,
        "Top 30 CpGs ranked by |GR-RHS posterior mean| on the common-converged replicate; 95% intervals for all four methods",
        transform=ax.transAxes,
        fontsize=9.2,
        color="#44403c",
        ha="left",
    )

    all_lowers = np.asarray([float(row["ci_lower_95"]) for row in rows], dtype=float)
    all_uppers = np.asarray([float(row["ci_upper_95"]) for row in rows], dtype=float)
    xmin = float(np.min(all_lowers))
    xmax = float(np.max(all_uppers))
    pad = max(0.35, 0.08 * (xmax - xmin if xmax > xmin else 1.0))
    ax.set_xlim(xmin - pad, xmax + pad)

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            marker="o",
            lw=1.6,
            markersize=5.0,
            label=METHOD_DISPLAY[method],
        )
        for method in METHODS
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9.8, loc="lower right")

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#78716c")

    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, _ = _build_rows(top_k=30)

    csv_path = OUT_DIR / "gse40279_local_block_micro_four_methods_effect_summary.csv"
    png_path = OUT_DIR / "gse40279_local_block_micro_four_methods_forest.png"
    pdf_path = OUT_DIR / "gse40279_local_block_micro_four_methods_forest.pdf"
    json_path = OUT_DIR / "gse40279_local_block_micro_four_methods_fit_summary.json"

    _write_csv(csv_path, rows)
    _plot(rows, png_path=png_path, pdf_path=pdf_path)

    summary = {
        "dataset_id": DATASET_ID,
        "run_dir": str(RUN_DIR),
        "replicate_id": REPLICATE_ID,
        "methods": METHODS,
        "method_labels": {k: v for k, v in METHOD_DISPLAY.items()},
        "top_k": 30,
        "ranking_rule": "top CpGs by absolute GR-RHS posterior mean on the common-converged replicate",
        "effect_scale": "age years per +1 SD CpG methylation",
        "artifacts": {
            "forest_png": str(png_path),
            "forest_pdf": str(pdf_path),
            "effect_summary_csv": str(csv_path),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(str(png_path))
    print(str(pdf_path))
    print(str(csv_path))
    print(str(json_path))


if __name__ == "__main__":
    main()
