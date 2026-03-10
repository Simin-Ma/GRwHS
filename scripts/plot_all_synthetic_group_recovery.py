from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from data.generators import generate_synthetic, synthetic_config_from_dict


SWEEPS = {
    "sim_s1": ("20260309-115403", "S1 Sparse-Strong"),
    "sim_s2": ("20260309-120416", "S2 Dense-Weak"),
    "sim_s3": ("20260309-121319", "S3 Mixed"),
    "sim_s4": ("20260309-122319", "S4 Half-Dense"),
}

SNR_KEYS = ["0p1", "0p5", "1p0", "3p0"]
MODEL_MAP = {
    "grrhs": ("grrhs", "GR-RHS", "#0f6b50"),
    "rhs": ("rhs", "RHS", "#6b7280"),
    "gigg": ("gigg", "GIGG", "#b42318"),
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _posterior_beta_mean(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    beta = np.asarray(arr["beta"], dtype=float)
    if beta.ndim == 1:
        return beta.reshape(-1)
    return beta.mean(axis=0).reshape(-1)


def _group_norms(beta: np.ndarray, groups: list[list[int]]) -> np.ndarray:
    return np.asarray([np.linalg.norm(beta[np.asarray(g, dtype=int)]) for g in groups], dtype=float)


def _group_boundaries(groups: list[list[int]]) -> list[int]:
    bounds = []
    cursor = 0
    for g in groups[:-1]:
        cursor += len(g)
        bounds.append(cursor)
    return bounds


def _regen_true_beta(resolved_config_path: Path, repeat_meta_path: Path) -> tuple[np.ndarray, list[list[int]]]:
    cfg = _load_json(repeat_meta_path)
    data_seed = int(cfg["metadata"]["seed"])
    # resolved config is YAML but JSON parser cannot read it
    import yaml

    resolved_cfg = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8"))
    data_cfg = dict(resolved_cfg.get("data", {}) or {})
    data_cfg["seed"] = data_seed
    synth_cfg = synthetic_config_from_dict(data_cfg)
    dataset = generate_synthetic(synth_cfg)
    return np.asarray(dataset.beta, dtype=float).reshape(-1), dataset.groups


def _plot_scenario(
    scenario_label: str,
    snr_label: str,
    true_beta: np.ndarray,
    groups: list[list[int]],
    estimates: list[tuple[str, str, str, np.ndarray]],
    out_path: Path,
) -> None:
    true_norms = _group_norms(true_beta, groups)
    est_norms = {key: _group_norms(beta, groups) for key, _label, _color, beta in estimates}

    fig = plt.figure(figsize=(15, 8.8), constrained_layout=True)
    gs = fig.add_gridspec(5, 1, height_ratios=[1.5, 0.75, 0.75, 0.75, 0.75])

    ax_top = fig.add_subplot(gs[0, 0])
    x = np.arange(len(groups))
    width = 0.18
    ax_top.bar(x - 1.5 * width, true_norms, width=width, color="#111827", alpha=0.9, label="True")
    for offset, (key, label, color, _beta) in zip([-0.5 * width, 0.5 * width, 1.5 * width], estimates):
        ax_top.bar(x + offset, est_norms[key], width=width, color=color, alpha=0.85, label=label)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([f"G{i+1}" for i in range(len(groups))], fontsize=9)
    ax_top.set_ylabel("Group norm")
    ax_top.set_title(f"{scenario_label} | SNR={snr_label} | Group norm recovery", loc="left", fontsize=13, fontweight="bold")
    ax_top.grid(axis="y", alpha=0.25)
    ax_top.legend(frameon=False, ncol=4, loc="upper left")

    heat_rows = [("True", "#111827", true_beta)] + [(label, color, beta) for _key, label, color, beta in estimates]
    vmax = float(np.max(np.abs(np.concatenate([np.abs(true_beta)] + [np.abs(beta) for *_rest, beta in estimates]))))
    norm = Normalize(vmin=0.0, vmax=vmax if vmax > 0 else 1.0)
    bounds = _group_boundaries(groups)

    for idx, (label, color, beta_vec) in enumerate(heat_rows, start=1):
        ax = fig.add_subplot(gs[idx, 0])
        image = np.abs(beta_vec).reshape(1, -1)
        ax.imshow(image, aspect="auto", cmap="YlOrRd", norm=norm)
        for b in bounds:
            ax.axvline(b - 0.5, color="white", linewidth=1.4, alpha=0.9)
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, labelpad=28, va="center", color=color, fontsize=10, fontweight="bold")
        if idx == len(heat_rows):
            centers = [np.mean(g) for g in groups]
            ax.set_xticks(centers)
            ax.set_xticklabels([f"G{i+1}" for i in range(len(groups))], fontsize=9)
            ax.set_xlabel("Features ordered by group")
        else:
            ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        f"{scenario_label} | SNR={snr_label} | Within-group coefficient recovery (|beta| heatmaps)",
        fontsize=15,
        y=1.02,
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "outputs" / "reports" / "all_group_recovery"
    out_dir.mkdir(parents=True, exist_ok=True)

    index_lines = ["# Synthetic Group Recovery Figures", ""]

    for sim_key, (timestamp, scenario_label) in SWEEPS.items():
        sweep_dir = repo_root / "outputs" / "sweeps" / sim_key
        for snr_key in SNR_KEYS:
            run_stem = f"snr{snr_key}"
            grrhs_dir = sweep_dir / f"{run_stem}_grrhs-{timestamp}"
            rhs_dir = sweep_dir / f"{run_stem}_rhs-{timestamp}"
            gigg_dir = sweep_dir / f"{run_stem}_gigg-{timestamp}"
            repeat_meta = grrhs_dir / "repeat_001" / "dataset_meta.json"
            resolved_cfg = grrhs_dir / "resolved_config.yaml"
            true_beta, groups = _regen_true_beta(resolved_cfg, repeat_meta)

            estimates = []
            for model_name, run_dir in [("grrhs", grrhs_dir), ("rhs", rhs_dir), ("gigg", gigg_dir)]:
                beta_path = run_dir / "repeat_001" / "fold_01" / "posterior_samples.npz"
                if not beta_path.exists():
                    continue
                key, label, color = MODEL_MAP[model_name]
                estimates.append((key, label, color, _posterior_beta_mean(beta_path)))

            snr_label = snr_key.replace("p", ".")
            out_path = out_dir / f"{sim_key}_{run_stem}_group_recovery.png"
            _plot_scenario(scenario_label, snr_label, true_beta, groups, estimates, out_path)
            index_lines.append(f"- `{scenario_label} | SNR={snr_label}`: `{out_path.name}`")

    (out_dir / "INDEX.md").write_text("\n".join(index_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
