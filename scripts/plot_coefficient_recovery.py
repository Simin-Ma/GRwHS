"""Coefficient-level recovery plots comparing GRRHS vs RHS."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402


TAG_COLORS = {
    "strong": "#d62728",
    "medium": "#ff7f0e",
    "weak": "#9467bd",
    "null": "#7f7f7f",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Produce coefficient-level recovery plots.")
    parser.add_argument("--grrhs-dir", required=True, type=Path, help="GRRHS run directory.")
    parser.add_argument("--rhs-dir", required=True, type=Path, help="RHS run directory.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Destination directory for figures.")
    parser.add_argument("--title", type=str, default="Coefficient recovery", help="Figure title prefix.")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k coefficients (by |β_true|) to highlight.")
    return parser.parse_args()


def _load_resolved_config(run_dir: Path) -> Mapping[str, object]:
    cfg_path = run_dir / "resolved_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"resolved_config.yaml not found under {run_dir}")
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def _derive_context(grrhs_dir: Path, resolved_cfg: Mapping[str, object]) -> tuple[str, str, float]:
    parts = list(grrhs_dir.parts)
    scenario = "scenario"
    if "sweeps" in parts:
        try:
            idx = parts.index("sweeps")
            scenario = parts[idx + 1]
        except Exception:
            pass
    snr_val = float(resolved_cfg.get("data", {}).get("snr", 0.0))
    token = ("%0.1f" % snr_val).replace(".", "p")
    return scenario, token, snr_val


def _regenerate_beta(resolved_cfg: Mapping[str, object]) -> np.ndarray:
    from data.generators import generate_synthetic, synthetic_config_from_dict

    data_cfg = resolved_cfg.get("data")
    if data_cfg is None:
        raise ValueError("Resolved config missing 'data' block.")
    seed = data_cfg.get("seed")
    seeds_block = resolved_cfg.get("seeds") or {}
    seed = seed or seeds_block.get("data_generation")
    synthetic_cfg = synthetic_config_from_dict(
        data_cfg,
        seed=seed,
        name=resolved_cfg.get("name"),
        task=resolved_cfg.get("task"),
    )
    dataset = generate_synthetic(synthetic_cfg)
    return np.asarray(dataset.beta, dtype=float)


def _load_groups_and_meta(run_dir: Path) -> Tuple[List[List[int]], Mapping[str, object]]:
    for repeat_dir in sorted(run_dir.glob("repeat_*")):
        meta_path = repeat_dir / "dataset_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            groups = meta.get("groups")
            if not groups:
                raise RuntimeError(f"No 'groups' entry in {meta_path}")
            return [list(map(int, grp)) for grp in groups], meta
    raise RuntimeError(f"No dataset_meta.json found under {run_dir}")


def _infer_group_tags(meta: Mapping[str, object], num_groups: int) -> List[str]:
    tags = ["null"] * num_groups
    info = (meta.get("metadata") or {}).get("info") or {}
    blueprint = info.get("signal_blueprint") or {}
    assignments = blueprint.get("assignments") or []
    for assignment in assignments:
        gid = assignment.get("group")
        if gid is None or not (0 <= int(gid) < num_groups):
            continue
        entry_label = str(
            assignment.get("entry") or assignment.get("label") or assignment.get("component") or ""
        ).lower()
        tag = "null"
        for candidate in ("strong", "medium", "weak"):
            if candidate in entry_label:
                tag = candidate
                break
        tags[int(gid)] = tag
    return tags


def _load_x_scale(run_dir: Path) -> np.ndarray:
    for repeat_dir in sorted(run_dir.glob("repeat_*")):
        for fold_dir in sorted(repeat_dir.glob("fold_*")):
            arr_path = fold_dir / "fold_arrays.npz"
            if arr_path.exists():
                arrays = np.load(arr_path)
                if "x_scale" not in arrays:
                    raise RuntimeError(f"x_scale missing in {arr_path}")
                return np.asarray(arrays["x_scale"], dtype=float)
    raise RuntimeError(f"No fold_arrays.npz with x_scale under {run_dir}")


def _load_beta_hat(run_dir: Path) -> np.ndarray:
    summary_path = run_dir / "posterior_summary.parquet"
    if not summary_path.exists():
        raise FileNotFoundError(f"posterior_summary not found under {run_dir}")
    df = pd.read_parquet(summary_path)
    beta_df = df[df["parameter"] == "beta"].sort_values("index")
    if beta_df.empty:
        raise RuntimeError("posterior_summary contains no beta rows.")
    return beta_df["mean"].to_numpy(dtype=float)


def _build_dataframe(
    *,
    beta_true_std: np.ndarray,
    beta_true_raw: np.ndarray,
    beta_hat_grrhs_std: np.ndarray,
    beta_hat_rhs_std: np.ndarray,
    beta_hat_grrhs_raw: np.ndarray,
    beta_hat_rhs_raw: np.ndarray,
    group_tags: Sequence[str],
    groups: Sequence[Sequence[int]],
) -> pd.DataFrame:
    p = beta_true_std.shape[0]
    group_index = np.zeros(p, dtype=int)
    for gid, idxs in enumerate(groups):
        group_index[np.asarray(idxs, dtype=int)] = gid
    tags = [group_tags[g] for g in group_index]
    df = pd.DataFrame(
        {
            "coef": np.arange(p),
            "group": group_index,
            "tag": tags,
            "beta_true": beta_true_std,
            "beta_true_raw": beta_true_raw,
            "beta_grrhs": beta_hat_grrhs_std,
            "beta_rhs": beta_hat_rhs_std,
            "beta_grrhs_raw": beta_hat_grrhs_raw,
            "beta_rhs_raw": beta_hat_rhs_raw,
        }
    )
    df["abs_true"] = df["beta_true"].abs()
    df["abs_true_raw"] = df["beta_true_raw"].abs()
    return df


def _scatter_plot(
    df: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    true_col: str,
    grrhs_col: str,
    rhs_col: str,
    xlabel: str,
    include_indices: Sequence[int] | None,
) -> None:
    if include_indices is not None:
        df = df.loc[include_indices]
    colors = df["tag"].map(TAG_COLORS).fillna("#aaaaaa")
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.axline((0, 0), slope=1.0, linestyle="--", color="#666666", linewidth=1.0)
    ax.scatter(df[true_col], df[grrhs_col], s=35, marker="o", facecolors=colors, edgecolors="black", label="GRRHS")
    ax.scatter(df[true_col], df[rhs_col], s=35, marker="^", facecolors=colors, edgecolors="black", alpha=0.85, label="RHS")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Estimated coefficient β̂_j")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    tag_handles = [plt.Line2D([0], [0], color="white", marker="o", markerfacecolor=col, markeredgecolor="black", linestyle="", label=tag.title()) for tag, col in TAG_COLORS.items()]
    legend1 = ax.legend(loc="upper left")
    ax.add_artist(legend1)
    ax.legend(handles=tag_handles, loc="lower right", title="Group tag")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _bar_plot_topk(df: pd.DataFrame, out_path: Path, title: str, k: int) -> None:
    top = df.nlargest(k, "abs_true").copy()
    top = top.sort_values("abs_true", ascending=False)
    idx = np.arange(top.shape[0])
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(6.5, 0.35 * k + 2), 4.5))
    ax.bar(idx - width, top["beta_true"], width, color="#9edae5", label="Truth")
    ax.bar(idx, top["beta_grrhs"], width, color="#1f77b4", label="GRRHS")
    ax.bar(idx + width, top["beta_rhs"], width, color="#7f7f7f", label="RHS")
    ax.set_xticks(idx, [f"β{c}" for c in top["coef"]], rotation=45, ha="right")
    ax.set_ylabel("Coefficient value (standardized)")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _stacked_mass_plot(df: pd.DataFrame, out_path: Path) -> None:
    def share(series: pd.Series) -> Dict[str, float]:
        total = float(series.abs().sum())
        if total == 0:
            return {tag: 0.0 for tag in TAG_COLORS}
        return {
            tag: float(series[df["tag"] == tag].abs().sum() / total)
            for tag in TAG_COLORS
        }

    rows = [
        ("Truth", share(df["beta_true"])),
        ("GRRHS", share(df["beta_grrhs"])),
        ("RHS", share(df["beta_rhs"])),
    ]
    labels = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    bottom = np.zeros(len(rows))
    for tag, color in TAG_COLORS.items():
        heights = [r[1][tag] for r in rows]
        ax.bar(labels, heights, bottom=bottom, color=color, edgecolor="black", label=tag.title())
        bottom += heights
    ax.set_ylabel("Share of |β| mass")
    ax.set_ylim(0, 1.0)
    ax.set_title("Coefficient mass by group type")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    grrhs_dir = args.grrhs_dir.expanduser().resolve()
    rhs_dir = args.rhs_dir.expanduser().resolve()
    if not grrhs_dir.exists() or not rhs_dir.exists():
        raise FileNotFoundError("Run directories not found.")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    groups, meta = _load_groups_and_meta(grrhs_dir)
    group_tags = _infer_group_tags(meta, len(groups))

    resolved_cfg = _load_resolved_config(grrhs_dir)
    beta_true = _regenerate_beta(resolved_cfg)
    x_scale = _load_x_scale(grrhs_dir)
    safe_scale = np.where(x_scale == 0, 1.0, x_scale)
    beta_true_std = beta_true * x_scale
    beta_true_raw = beta_true.copy()

    beta_hat_grrhs = _load_beta_hat(grrhs_dir)
    beta_hat_rhs = _load_beta_hat(rhs_dir)

    if beta_hat_grrhs.shape[0] != beta_true_std.shape[0]:
        raise ValueError("β dimension mismatch for GRRHS run.")
    if beta_hat_rhs.shape[0] != beta_true_std.shape[0]:
        raise ValueError("β dimension mismatch for RHS run.")

    df = _build_dataframe(
        beta_true_std=beta_true_std,
        beta_true_raw=beta_true_raw,
        beta_hat_grrhs_std=beta_hat_grrhs,
        beta_hat_rhs_std=beta_hat_rhs,
        beta_hat_grrhs_raw=beta_hat_grrhs / safe_scale,
        beta_hat_rhs_raw=beta_hat_rhs / safe_scale,
        group_tags=group_tags,
        groups=groups,
    )
    df_sorted_std = df.sort_values("abs_true", ascending=False)
    top_std_idx = df_sorted_std.head(args.top_k).index
    df_sorted_raw = df.sort_values("abs_true_raw", ascending=False)
    top_raw_idx = df_sorted_raw.head(args.top_k).index

    scenario_slug, snr_token, _ = _derive_context(grrhs_dir, resolved_cfg)
    prefix = f"{scenario_slug}_snr{snr_token}"

    _scatter_plot(
        df,
        out_path=output_dir / f"{prefix}_coeff_scatter_all_std.png",
        title=f"{args.title}: all coefficients (standardized)",
        true_col="beta_true",
        grrhs_col="beta_grrhs",
        rhs_col="beta_rhs",
        xlabel="True coefficient β_j (standardized)",
        include_indices=None,
    )
    _scatter_plot(
        df,
        out_path=output_dir / f"{prefix}_coeff_scatter_topk_std.png",
        title=f"{args.title}: top-{args.top_k} (standardized)",
        true_col="beta_true",
        grrhs_col="beta_grrhs",
        rhs_col="beta_rhs",
        xlabel="True coefficient β_j (standardized)",
        include_indices=top_std_idx,
    )
    _scatter_plot(
        df,
        out_path=output_dir / f"{prefix}_coeff_scatter_all_raw.png",
        title=f"{args.title}: all coefficients (raw units)",
        true_col="beta_true_raw",
        grrhs_col="beta_grrhs_raw",
        rhs_col="beta_rhs_raw",
        xlabel="True coefficient β_j (raw units)",
        include_indices=None,
    )
    _scatter_plot(
        df,
        out_path=output_dir / f"{prefix}_coeff_scatter_topk_raw.png",
        title=f"{args.title}: top-{args.top_k} (raw units)",
        true_col="beta_true_raw",
        grrhs_col="beta_grrhs_raw",
        rhs_col="beta_rhs_raw",
        xlabel="True coefficient β_j (raw units)",
        include_indices=top_raw_idx,
    )
    _bar_plot_topk(
        df,
        out_path=output_dir / f"{prefix}_coeff_bar_topk.png",
        title=f"{args.title}: top-{args.top_k} coefficients",
        k=args.top_k,
    )
    _stacked_mass_plot(df, output_dir / f"{prefix}_coeff_mass_stacked.png")

    df.to_csv(output_dir / "coefficients_summary.csv", index=False)
    print(f"[OK] Coefficient plots written to {output_dir}")


if __name__ == "__main__":
    main()
