from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SWEEP_DIR = Path("outputs/sweeps/real_covid19_trust_experts_thesis")
TIMESTAMP = "20260309-025417"
RUNS = {
    "grrhs": ("trust_experts_grrhs", "GR-RHS", "#0f6b50"),
    "rhs": ("trust_experts_rhs", "RHS", "#6b7280"),
    "gigg": ("trust_experts_gigg", "GIGG", "#b42318"),
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


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_posterior_means(run_dir: Path) -> list[np.ndarray]:
    repeat_dir = run_dir / "repeat_001"
    fold_dirs = sorted(p for p in repeat_dir.iterdir() if p.is_dir() and p.name.startswith("fold_"))
    means: list[np.ndarray] = []
    for fold_dir in fold_dirs:
        posterior_path = fold_dir / "posterior_samples.npz"
        if not posterior_path.exists():
            continue
        arr = np.load(posterior_path, allow_pickle=True)
        beta = np.asarray(arr["beta"], dtype=float)
        beta_mean = beta.mean(axis=0) if beta.ndim > 1 else beta.reshape(-1)
        means.append(beta_mean.reshape(-1))
    return means


def _group_rms(beta: np.ndarray, groups: list[list[int]]) -> np.ndarray:
    out = []
    for g in groups:
        idx = np.asarray(g, dtype=int)
        out.append(float(np.linalg.norm(beta[idx]) / np.sqrt(len(idx))))
    return np.asarray(out, dtype=float)


def _group_centers(groups: list[list[int]]) -> list[float]:
    return [float((g[0] + g[-1]) / 2.0) for g in groups]


def _group_boundaries(groups: list[list[int]]) -> list[int]:
    bounds = []
    cursor = 0
    for g in groups[:-1]:
        cursor += len(g)
        bounds.append(cursor)
    return bounds


def _build_group_table(groups: list[list[int]], feature_names: list[str], run_data: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for gid, members in enumerate(groups):
        row = {
            "group_id": gid,
            "group_label": GROUP_LABELS.get(gid, f"Group {gid+1}"),
            "group_size": len(members),
            "feature_start": feature_names[members[0]],
            "feature_end": feature_names[members[-1]],
        }
        for key, payload in run_data.items():
            group_rms_mean = np.asarray(payload["group_rms_mean"], dtype=float)
            row[f"{key}_group_rms"] = float(group_rms_mean[gid])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sweep_dir = repo_root / SWEEP_DIR
    out_dir = repo_root / "outputs" / "reports" / "covid_latest"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_meta = _load_json(
        sweep_dir / f"trust_experts_grrhs-{TIMESTAMP}" / "repeat_001" / "dataset_meta.json"
    )
    groups = dataset_meta["groups"]
    feature_names = dataset_meta["feature_names"]

    run_data: dict[str, dict[str, object]] = {}
    for key, (run_name, label, color) in RUNS.items():
        run_dir = sweep_dir / f"{run_name}-{TIMESTAMP}"
        beta_means = _collect_posterior_means(run_dir)
        beta_means_arr = np.asarray(beta_means, dtype=float)
        run_data[key] = {
            "label": label,
            "color": color,
            "beta_abs_mean": np.mean(np.abs(beta_means_arr), axis=0),
            "beta_signed_mean": np.mean(beta_means_arr, axis=0),
            "group_rms_mean": np.mean([_group_rms(beta, groups) for beta in beta_means_arr], axis=0),
        }

    centers = _group_centers(groups)
    boundaries = _group_boundaries(groups)
    n_features = len(feature_names)

    fig = plt.figure(figsize=(15.5, 8.4), constrained_layout=True)
    gs = fig.add_gridspec(5, 1, height_ratios=[1.6, 0.85, 0.85, 0.85, 0.12])

    ax_top = fig.add_subplot(gs[0, 0])
    x = np.arange(len(groups))
    width = 0.22
    offsets = [-width, 0.0, width]
    for offset, (key, payload) in zip(offsets, run_data.items()):
        ax_top.bar(
            x + offset,
            payload["group_rms_mean"],
            width=width,
            color=payload["color"],
            alpha=0.88,
            label=payload["label"],
        )
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([GROUP_LABELS.get(i, f"G{i+1}") for i in range(len(groups))], fontsize=10)
    ax_top.set_ylabel("Group RMS coefficient")
    ax_top.set_title("COVID trust survey: grouped effect strength by model", loc="left", fontsize=14, fontweight="bold")
    ax_top.grid(axis="y", alpha=0.25)
    ax_top.legend(loc="upper left", frameon=False, ncol=3)

    vmax = float(
        np.max(
            np.concatenate([np.asarray(payload["beta_abs_mean"], dtype=float) for payload in run_data.values()])
        )
    )
    axes = []
    for idx, (key, payload) in enumerate(run_data.items(), start=1):
        ax = fig.add_subplot(gs[idx, 0])
        axes.append(ax)
        image = np.asarray(payload["beta_abs_mean"], dtype=float).reshape(1, -1)
        ax.imshow(image, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=vmax if vmax > 0 else 1.0)
        for b in boundaries:
            ax.axvline(b - 0.5, color="white", linewidth=1.5, alpha=0.9)
        ax.set_yticks([])
        ax.set_ylabel(payload["label"], rotation=0, labelpad=28, va="center", color=payload["color"], fontsize=10, fontweight="bold")
        if idx == len(run_data):
            ax.set_xticks(centers)
            ax.set_xticklabels([GROUP_LABELS.get(i, f"G{i+1}") for i in range(len(groups))], fontsize=10)
            ax.set_xlabel("Features ordered by group")
        else:
            ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    cax = fig.add_subplot(gs[4, 0])
    cax.axis("off")
    fig.suptitle(
        "COVID latest sweep: grouped posterior structure (average across outer folds)",
        fontsize=16,
        y=1.02,
        fontweight="bold",
    )

    out_path = out_dir / "covid_latest_group_structure.png"
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    group_table = _build_group_table(groups, feature_names, run_data)
    group_table.to_csv(out_dir / "covid_latest_group_strength_table.csv", index=False)

    notes = [
        "# COVID Latest Group Structure",
        "",
        "This figure is the real-data analogue of the synthetic group-recovery plots.",
        "There is no ground-truth beta on real data, so the chart compares group-level and feature-level posterior strength across Bayesian grouped models.",
        "",
        "Top panel:",
        "- Group RMS coefficient by model, averaged across outer folds.",
        "- Larger values indicate the model allocates more total signal to that group after normalizing for group size.",
        "",
        "Bottom panels:",
        "- Feature-level mean absolute posterior coefficients ordered by group.",
        "- Useful for seeing whether a model spreads signal broadly within a group or concentrates it on a subset of features.",
    ]
    (out_dir / "README.md").write_text("\n".join(notes), encoding="utf-8")


if __name__ == "__main__":
    main()
