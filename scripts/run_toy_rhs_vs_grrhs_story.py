from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from grrhs.models.baselines import RegularizedHorseshoeRegression
from grrhs.models.grrhs_gibbs import GRRHS_Gibbs


def _import_toy_fn(repo_root: Path):
    mod_path = repo_root / "scripts" / "run_tiny_sanity_check.py"
    spec = importlib.util.spec_from_file_location("tiny_sanity_mod", str(mod_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import toy generator from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    fn = getattr(mod, "_toy_mixed_signal_dataset", None)
    if fn is None:
        raise RuntimeError("_toy_mixed_signal_dataset not found")
    return fn


def _rmse(vec: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(vec))))


def _group_rank(beta_hat: np.ndarray, groups: List[List[int]]) -> Dict[int, int]:
    scores = np.array([float(np.mean(np.abs(beta_hat[np.array(g, dtype=int)]))) for g in groups], dtype=float)
    order = np.argsort(-scores)
    return {int(gid): int(np.where(order == gid)[0][0]) + 1 for gid in range(len(groups))}


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    outdir = repo_root / "outputs" / "toy_rhs_vs_grrhs_story"
    outdir.mkdir(parents=True, exist_ok=True)

    toy_fn = _import_toy_fn(repo_root)

    seed = 196
    n = 60
    sigma_noise = 1.0
    a_values = [1.0, 2.0, 3.0, 4.0, 5.0]

    # Tuned GR-RHS setup found for the RHS-vs-GR-RHS toy mechanism narrative.
    grrhs_cfg = {
        "c": 4.0,
        "tau0_multiplier": 1.3,
        "eta": 1.0,
        "s0": 1.0,
        "iters": 1200,
        "burnin": 600,
        "tau_slice_w": 0.25,
        "tau_slice_m": 180,
    }
    rhs_cfg = {
        "scale_global": 0.3,
        "num_warmup": 300,
        "num_samples": 300,
        "num_chains": 1,
        "thinning": 1,
        "target_accept_prob": 0.9,
    }

    idx_g2 = [10]
    idx_g3 = [20, 21, 22, 23, 24, 25]
    idx_g4 = [30, 31, 32, 33, 34, 35]
    idx_g5 = [40, 41, 42, 43]

    per_a: List[Dict[str, Any]] = []
    coeff_store: Dict[str, Dict[str, List[float]]] = {}

    for a in a_values:
        payload = toy_fn(n=n, sigma_noise=sigma_noise, seed=seed, A=float(a), plus_corr=False)
        X_raw = np.asarray(payload["X"], dtype=float)
        y_raw = np.asarray(payload["y"], dtype=float)
        beta_true = np.asarray(payload["beta"], dtype=float)
        groups = [list(map(int, g)) for g in payload["groups"]]
        signal_meta = dict(payload["signal_meta"])

        scaler = StandardScaler(with_mean=True, with_std=True)
        X = scaler.fit_transform(X_raw).astype(float)
        y = (y_raw - float(np.mean(y_raw))).astype(float)
        beta_true_std = beta_true * scaler.scale_

        tags = signal_meta.get("tags", {}) if isinstance(signal_meta, dict) else {}
        p = X.shape[1]
        p0 = sum(len(tags.get(k, [])) for k in ("strong", "weak", "medium"))
        p0 = int(max(1, min(p - 1, p0)))
        tau0_auto = (p0 / (p - p0)) * sigma_noise / math.sqrt(max(n, 1))

        rhs = RegularizedHorseshoeRegression(
            scale_global=float(rhs_cfg["scale_global"]),
            num_warmup=int(rhs_cfg["num_warmup"]),
            num_samples=int(rhs_cfg["num_samples"]),
            num_chains=int(rhs_cfg["num_chains"]),
            thinning=int(rhs_cfg["thinning"]),
            target_accept_prob=float(rhs_cfg["target_accept_prob"]),
            progress_bar=False,
            seed=seed + 11,
        ).fit(X, y)

        gr = GRRHS_Gibbs(
            c=float(grrhs_cfg["c"]),
            tau0=float(tau0_auto * grrhs_cfg["tau0_multiplier"]),
            eta=float(grrhs_cfg["eta"]),
            s0=float(grrhs_cfg["s0"]),
            use_groups=True,
            iters=int(grrhs_cfg["iters"]),
            burnin=int(grrhs_cfg["burnin"]),
            thin=1,
            seed=seed + 23,
            num_chains=1,
            tau_slice_w=float(grrhs_cfg["tau_slice_w"]),
            tau_slice_m=int(grrhs_cfg["tau_slice_m"]),
        ).fit(X, y, groups=groups)

        beta_rhs = np.asarray(rhs.coef_, dtype=float)
        beta_gr = np.asarray(gr.coef_mean_, dtype=float)

        coeff_store[f"A{a:g}"] = {
            "True": beta_true_std.tolist(),
            "RHS": beta_rhs.tolist(),
            "GR-RHS": beta_gr.tolist(),
        }

        rhs_rank = _group_rank(beta_rhs, groups)
        gr_rank = _group_rank(beta_gr, groups)

        row = {
            "A": float(a),
            "group2_strong_abs_error": {
                "RHS": float(abs(beta_rhs[idx_g2[0]] - beta_true_std[idx_g2[0]])),
                "GR-RHS": float(abs(beta_gr[idx_g2[0]] - beta_true_std[idx_g2[0]])),
            },
            "group3_weak_rmse": {
                "RHS": _rmse(beta_rhs[np.array(idx_g3)] - beta_true_std[np.array(idx_g3)]),
                "GR-RHS": _rmse(beta_gr[np.array(idx_g3)] - beta_true_std[np.array(idx_g3)]),
            },
            "group4_mixed_rmse": {
                "RHS": _rmse(beta_rhs[np.array(idx_g4)] - beta_true_std[np.array(idx_g4)]),
                "GR-RHS": _rmse(beta_gr[np.array(idx_g4)] - beta_true_std[np.array(idx_g4)]),
            },
            "group5_nearzero_abs_mean": {
                "RHS": float(np.mean(np.abs(beta_rhs[np.array(idx_g5)]))),
                "GR-RHS": float(np.mean(np.abs(beta_gr[np.array(idx_g5)]))),
            },
            "group_rank_mean_abs": {
                "RHS": {"group3": rhs_rank[2], "group4": rhs_rank[3], "group5": rhs_rank[4]},
                "GR-RHS": {"group3": gr_rank[2], "group4": gr_rank[3], "group5": gr_rank[4]},
            },
        }
        per_a.append(row)

    checks = {
        "group2_not_worse_than_rhs": all(r["group2_strong_abs_error"]["GR-RHS"] <= r["group2_strong_abs_error"]["RHS"] for r in per_a),
        "group3_weak_better_than_rhs": all(r["group3_weak_rmse"]["GR-RHS"] < r["group3_weak_rmse"]["RHS"] for r in per_a),
        "group4_mixed_better_than_rhs": all(r["group4_mixed_rmse"]["GR-RHS"] < r["group4_mixed_rmse"]["RHS"] for r in per_a),
        "group5_not_activated": all(
            r["group_rank_mean_abs"]["GR-RHS"]["group5"] == 6 and r["group_rank_mean_abs"]["RHS"]["group5"] == 6 for r in per_a
        ),
        "group_ranking_focus": all(
            r["group_rank_mean_abs"]["GR-RHS"]["group4"] <= 2 and r["group_rank_mean_abs"]["GR-RHS"]["group3"] <= 3 for r in per_a
        ),
    }

    summary = {
        "narrative_core": "This toy example isolates the added value of the group layer by comparing GR-RHS directly with its non-group counterpart (RHS).",
        "grrhs_config": grrhs_cfg,
        "rhs_config": rhs_cfg,
        "per_A_metrics": per_a,
        "criteria_checks": checks,
        "all_passed": bool(all(checks.values())),
    }
    (outdir / "toy_rhs_vs_grrhs_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (outdir / "toy_rhs_vs_grrhs_coefficients.json").write_text(json.dumps(coeff_store, indent=2), encoding="utf-8")

    # Plot 1: key mechanism metrics vs A (RHS vs GR-RHS)
    fig1, axes = plt.subplots(1, 3, figsize=(14.6, 4.5), squeeze=False)
    axes = axes[0]
    for name, key in [
        ("Group2 strong abs error", "group2_strong_abs_error"),
        ("Group3 weak RMSE", "group3_weak_rmse"),
        ("Group4 mixed RMSE", "group4_mixed_rmse"),
    ]:
        ax = axes[["group2_strong_abs_error", "group3_weak_rmse", "group4_mixed_rmse"].index(key)]
        rhs_y = [r[key]["RHS"] for r in per_a]
        gr_y = [r[key]["GR-RHS"] for r in per_a]
        ax.plot(a_values, rhs_y, marker="o", linewidth=2.0, color="#6b7280", label="RHS")
        ax.plot(a_values, gr_y, marker="o", linewidth=2.0, color="#111827", label="GR-RHS")
        ax.set_title(name)
        ax.set_xlabel("A")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Error")
    axes[2].legend(frameon=False)
    fig1.tight_layout()
    fig1.savefig(outdir / "rhs_vs_grrhs_mechanism_metrics_vs_A.png", dpi=180)
    plt.close(fig1)

    # Plot 2: Group ranking for (Group3, Group4, Group5) by A.
    fig2, ax2 = plt.subplots(figsize=(8.5, 5.0))
    g3_rhs = [r["group_rank_mean_abs"]["RHS"]["group3"] for r in per_a]
    g4_rhs = [r["group_rank_mean_abs"]["RHS"]["group4"] for r in per_a]
    g5_rhs = [r["group_rank_mean_abs"]["RHS"]["group5"] for r in per_a]
    g3_gr = [r["group_rank_mean_abs"]["GR-RHS"]["group3"] for r in per_a]
    g4_gr = [r["group_rank_mean_abs"]["GR-RHS"]["group4"] for r in per_a]
    g5_gr = [r["group_rank_mean_abs"]["GR-RHS"]["group5"] for r in per_a]
    ax2.plot(a_values, g3_rhs, marker="o", color="#9ca3af", linewidth=1.8, label="RHS G3 rank")
    ax2.plot(a_values, g4_rhs, marker="o", color="#6b7280", linewidth=1.8, label="RHS G4 rank")
    ax2.plot(a_values, g5_rhs, marker="o", color="#4b5563", linewidth=1.8, label="RHS G5 rank")
    ax2.plot(a_values, g3_gr, marker="s", color="#374151", linewidth=2.0, label="GR-RHS G3 rank")
    ax2.plot(a_values, g4_gr, marker="s", color="#111827", linewidth=2.0, label="GR-RHS G4 rank")
    ax2.plot(a_values, g5_gr, marker="s", color="#1f2937", linewidth=2.0, label="GR-RHS G5 rank")
    ax2.set_xlabel("A")
    ax2.set_ylabel("Rank (1 is highest)")
    ax2.set_yticks([1, 2, 3, 4, 5, 6])
    ax2.invert_yaxis()
    ax2.grid(alpha=0.22)
    ax2.legend(frameon=False, ncol=2)
    ax2.set_title("RHS vs GR-RHS: Group ranking focus (G3/G4/G5)")
    fig2.tight_layout()
    fig2.savefig(outdir / "rhs_vs_grrhs_group_ranks_vs_A.png", dpi=180)
    plt.close(fig2)

    # Simple markdown report for manuscript copy-paste.
    lines = [
        "# Toy Example: RHS vs GR-RHS",
        "",
        "This toy example isolates the added value of the group layer by comparing GR-RHS directly with its non-group counterpart (RHS).",
        "",
        "## Criteria status",
    ]
    for k, v in checks.items():
        lines.append(f"- `{k}`: {'PASS' if v else 'FAIL'}")
    lines.extend(
        [
            "",
            "## Outputs",
            "- `toy_rhs_vs_grrhs_summary.json`",
            "- `toy_rhs_vs_grrhs_coefficients.json`",
            "- `rhs_vs_grrhs_mechanism_metrics_vs_A.png`",
            "- `rhs_vs_grrhs_group_ranks_vs_A.png`",
        ]
    )
    (outdir / "TOY_RHS_GRRHS_STORY.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved story outputs to: {outdir}")
    print(f"All criteria passed: {all(checks.values())}")


if __name__ == "__main__":
    main()
