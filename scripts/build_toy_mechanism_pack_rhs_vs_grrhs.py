from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from grrhs.diagnostics.postprocess import compute_diagnostics_from_samples
from grrhs.models.baselines import RegularizedHorseshoeRegression
from grrhs.models.grrhs_gibbs import GRRHS_Gibbs


def _import_toy_fn(repo_root: Path):
    script_path = repo_root / "scripts" / "run_tiny_sanity_check.py"
    spec = importlib.util.spec_from_file_location("toy_mod", str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    fn = getattr(mod, "_toy_mixed_signal_dataset", None)
    if fn is None:
        raise RuntimeError("_toy_mixed_signal_dataset not found")
    return fn


def _rmse(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(arr))))


def _group_scores(beta_hat: np.ndarray, groups: List[List[int]]) -> np.ndarray:
    return np.array(
        [
            float(np.sum(np.abs(beta_hat[np.array(g, dtype=int)])) / np.sqrt(max(len(g), 1)))
            for g in groups
        ],
        dtype=float,
    )


def _group_rank(scores: np.ndarray) -> Dict[int, int]:
    order = np.argsort(-scores)
    return {int(gid): int(np.where(order == gid)[0][0]) + 1 for gid in range(scores.size)}


def _flatten_groups(groups: List[List[int]], p: int) -> np.ndarray:
    gidx = np.zeros(p, dtype=int)
    for gid, g in enumerate(groups):
        for j in g:
            gidx[int(j)] = gid
    return gidx


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    outdir = repo_root / "outputs" / "toy_rhs_grrhs_mechanism_pack"
    outdir.mkdir(parents=True, exist_ok=True)

    toy_fn = _import_toy_fn(repo_root)

    seed = 196
    n = 60
    sigma_noise = 1.0
    a_values = [1.0, 2.0, 3.0, 4.0, 5.0]

    # Tuned for mechanism story.
    rhs_cfg = dict(scale_global=0.3, num_warmup=300, num_samples=300, num_chains=1, thinning=1, target_accept_prob=0.9)
    gr_cfg = dict(c=4.0, tau0_multiplier=1.3, eta=1.0, s0=1.0, iters=1200, burnin=600, tau_slice_w=0.25, tau_slice_m=180)

    rows: List[Dict[str, Any]] = []
    coeff_by_a: Dict[str, Dict[str, List[float]]] = {}
    groups_ref: List[List[int]] | None = None

    omega_snapshot: Dict[str, Any] | None = None
    # Use the middle A on the path as the reference snapshot for omega interpretation.
    omega_a_ref = a_values[len(a_values) // 2]

    for a in a_values:
        payload = toy_fn(n=n, sigma_noise=sigma_noise, seed=seed, A=float(a), plus_corr=False)
        X_raw = np.asarray(payload["X"], dtype=float)
        y_raw = np.asarray(payload["y"], dtype=float)
        beta_true = np.asarray(payload["beta"], dtype=float)
        groups = [list(map(int, g)) for g in payload["groups"]]
        if groups_ref is None:
            groups_ref = [list(g) for g in groups]
        signal_meta = dict(payload["signal_meta"])

        scaler = StandardScaler(with_mean=True, with_std=True)
        X = scaler.fit_transform(X_raw).astype(float)
        y = (y_raw - float(np.mean(y_raw))).astype(float)
        beta_true_std = beta_true * scaler.scale_

        tags = signal_meta.get("tags", {}) if isinstance(signal_meta, dict) else {}
        strong_idx = np.array(tags.get("strong", []), dtype=int)
        weak_idx = np.array(tags.get("weak", []), dtype=int)
        near_zero_idx = np.array(tags.get("near_zero", []), dtype=int)
        null_idx = np.array(tags.get("null", []), dtype=int)

        # group2/group3/group4/group5 explicit subsets
        g2 = np.array(groups[1], dtype=int)
        g3 = np.array(groups[2], dtype=int)
        g4 = np.array(groups[3], dtype=int)
        g5 = np.array(groups[4], dtype=int)

        strong_focus = strong_idx[np.isin(strong_idx, np.concatenate([g2, g4]))]
        weak_focus = weak_idx[np.isin(weak_idx, np.concatenate([g3, g4]))]
        null_near_focus = np.concatenate([near_zero_idx[np.isin(near_zero_idx, g5)], null_idx[np.isin(null_idx, g5)]])

        p = X.shape[1]
        p0 = int(sum(len(tags.get(k, [])) for k in ("strong", "weak", "medium")))
        p0 = max(1, min(p - 1, p0))
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
        beta_rhs = np.asarray(rhs.coef_, dtype=float)

        gr = GRRHS_Gibbs(
            c=float(gr_cfg["c"]),
            tau0=float(tau0_auto * float(gr_cfg["tau0_multiplier"])),
            eta=float(gr_cfg["eta"]),
            s0=float(gr_cfg["s0"]),
            use_groups=True,
            iters=int(gr_cfg["iters"]),
            burnin=int(gr_cfg["burnin"]),
            thin=1,
            seed=seed + 23,
            num_chains=1,
            tau_slice_w=float(gr_cfg["tau_slice_w"]),
            tau_slice_m=int(gr_cfg["tau_slice_m"]),
        ).fit(X, y, groups=groups)
        beta_gr = np.asarray(gr.coef_mean_, dtype=float)

        coeff_by_a[f"A{a:g}"] = {"True": beta_true_std.tolist(), "RHS": beta_rhs.tolist(), "GR-RHS": beta_gr.tolist()}

        # four core metrics
        strong_rmse_rhs = _rmse(beta_rhs[strong_focus] - beta_true_std[strong_focus]) if strong_focus.size else float("nan")
        strong_rmse_gr = _rmse(beta_gr[strong_focus] - beta_true_std[strong_focus]) if strong_focus.size else float("nan")
        weak_rmse_rhs = _rmse(beta_rhs[weak_focus] - beta_true_std[weak_focus]) if weak_focus.size else float("nan")
        weak_rmse_gr = _rmse(beta_gr[weak_focus] - beta_true_std[weak_focus]) if weak_focus.size else float("nan")
        null_rmse_rhs = _rmse(beta_rhs[null_near_focus] - beta_true_std[null_near_focus]) if null_near_focus.size else float("nan")
        null_rmse_gr = _rmse(beta_gr[null_near_focus] - beta_true_std[null_near_focus]) if null_near_focus.size else float("nan")

        active_feature = np.zeros(p, dtype=bool)
        active_feature[np.concatenate([strong_idx, weak_idx])] = True
        active_group = np.zeros(len(groups), dtype=bool)
        for gid, g in enumerate(groups):
            if np.any(active_feature[np.array(g, dtype=int)]):
                active_group[gid] = True

        score_rhs = _group_scores(beta_rhs, groups)
        score_gr = _group_scores(beta_gr, groups)
        auprc_rhs = float(average_precision_score(active_group.astype(int), score_rhs))
        auprc_gr = float(average_precision_score(active_group.astype(int), score_gr))
        rank_rhs = _group_rank(score_rhs)
        rank_gr = _group_rank(score_gr)

        rows.append(
            {
                "A": float(a),
                "strong_rmse": {"RHS": strong_rmse_rhs, "GR-RHS": strong_rmse_gr},
                "weak_rmse": {"RHS": weak_rmse_rhs, "GR-RHS": weak_rmse_gr},
                "null_near_rmse": {"RHS": null_rmse_rhs, "GR-RHS": null_rmse_gr},
                "group_auprc": {"RHS": auprc_rhs, "GR-RHS": auprc_gr},
                "group_ranking": {
                    "RHS": {"group3": rank_rhs[2], "group4": rank_rhs[3], "group5": rank_rhs[4]},
                    "GR-RHS": {"group3": rank_gr[2], "group4": rank_gr[3], "group5": rank_gr[4]},
                },
                "group_scores": {
                    "RHS": score_rhs.tolist(),
                    "GR-RHS": score_gr.tolist(),
                },
            }
        )

        # omega explanation snapshot at one reference A (middle point of the integer path)
        if abs(a - omega_a_ref) < 1e-12:
            gidx = _flatten_groups(groups, p)
            diag = compute_diagnostics_from_samples(
                X=X,
                group_index=gidx,
                c=float(gr_cfg["c"]),
                eps=1e-8,
                lambda_=np.asarray(gr.lambda_samples_, dtype=float),
                tau=np.asarray(gr.tau_samples_, dtype=float).reshape(-1),
                phi=np.asarray(gr.phi_samples_, dtype=float),
                sigma=np.sqrt(np.maximum(np.asarray(gr.sigma2_samples_, dtype=float).reshape(-1), 1e-12)),
            )
            per = diag.per_coeff

            strong_rep = int(strong_idx[np.isin(strong_idx, g2)][0]) if np.any(np.isin(strong_idx, g2)) else int(g2[0])
            weak_rep = int(weak_idx[np.isin(weak_idx, g3)][0]) if np.any(np.isin(weak_idx, g3)) else int(g3[0])
            near_rep = int(near_zero_idx[np.isin(near_zero_idx, g5)][0]) if np.any(np.isin(near_zero_idx, g5)) else int(g5[0])

            reps = [
                ("Group2 strong", strong_rep),
                ("Group3 weak", weak_rep),
                ("Group5 near-zero", near_rep),
            ]
            omega_snapshot = {
                "A_ref": float(a),
                "variables": [
                    {
                        "label": label,
                        "index": int(j),
                        "omega_group": float(per["omega_group"][j]),
                        "omega_tau": float(per["omega_tau"][j]),
                        "omega_lambda": float(per["omega_lambda"][j]),
                    }
                    for label, j in reps
                ],
            }

    summary = {
        "narrative": "This toy example isolates the added value of the group layer by comparing GR-RHS directly with its non-group counterpart (RHS).",
        "rhs_config": rhs_cfg,
        "grrhs_config": gr_cfg,
        "metrics_by_A": rows,
        "omega_snapshot": omega_snapshot,
    }
    (outdir / "toy_mechanism_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (outdir / "toy_mechanism_coefficients.json").write_text(json.dumps(coeff_by_a, indent=2), encoding="utf-8")

    # ------- Figure 1: readable comparison (3-bar top + group RMSE bottom) -------
    ncols = len(a_values)
    coeff_cache: Dict[float, Dict[str, np.ndarray]] = {}
    all_top_vals: List[np.ndarray] = []
    all_bottom_vals: List[np.ndarray] = []
    for a in a_values:
        key = f"A{a:g}"
        true_beta = np.asarray(coeff_by_a[key]["True"], dtype=float)
        rhs_beta = np.asarray(coeff_by_a[key]["RHS"], dtype=float)
        gr_beta = np.asarray(coeff_by_a[key]["GR-RHS"], dtype=float)
        coeff_cache[a] = {"true": true_beta, "rhs": rhs_beta, "gr": gr_beta}
        all_top_vals.extend([true_beta, rhs_beta, gr_beta])

        if groups_ref is not None:
            g_rmse_rhs = np.array([_rmse(rhs_beta[np.array(g, dtype=int)] - true_beta[np.array(g, dtype=int)]) for g in groups_ref], dtype=float)
            g_rmse_gr = np.array([_rmse(gr_beta[np.array(g, dtype=int)] - true_beta[np.array(g, dtype=int)]) for g in groups_ref], dtype=float)
            all_bottom_vals.extend([g_rmse_rhs, g_rmse_gr])

    bottom_max = float(max(np.max(v) for v in all_bottom_vals)) if all_bottom_vals else 1.0

    fig1, axes1 = plt.subplots(
        nrows=2,
        ncols=ncols,
        figsize=(5.0 * ncols, 6.8),
        squeeze=False,
        gridspec_kw={"height_ratios": [3.0, 1.8]},
    )
    for c, a in enumerate(a_values):
        cache = coeff_cache[a]
        true_beta = cache["true"]
        rhs_beta = cache["rhs"]
        gr_beta = cache["gr"]
        xs = np.arange(true_beta.size)

        ax_top = axes1[0][c]
        ax_bot = axes1[1][c]

        w = 0.27
        ax_top.bar(xs - w, true_beta, width=w, color="#f3d1cf", edgecolor="#8e1b1b", linewidth=0.4, label="True")
        ax_top.bar(xs, rhs_beta, width=w, color="#9ca3af", label="RHS")
        ax_top.bar(xs + w, gr_beta, width=w, color="#111827", label="GR-RHS")
        ax_top.set_title(f"A={a:g}")
        top_local_min = float(min(np.min(true_beta), np.min(rhs_beta), np.min(gr_beta)))
        top_local_max = float(max(np.max(true_beta), np.max(rhs_beta), np.max(gr_beta)))
        top_local_pad = max(0.12, 0.1 * max(abs(top_local_min), abs(top_local_max)))
        ax_top.set_ylim(top_local_min - top_local_pad, top_local_max + top_local_pad)
        ax_top.grid(axis="y", alpha=0.22)

        if groups_ref is not None:
            for gid, g in enumerate(groups_ref):
                idx = np.array(g, dtype=int)
                if idx.size:
                    xline = float(np.max(idx)) + 0.5
                    ax_top.axvline(xline, color="#d1d5db", linewidth=0.7)
                    if gid in (2, 3):
                        x0, x1 = float(np.min(idx)) - 0.5, float(np.max(idx)) + 0.5
                        ax_top.axvspan(x0, x1, color="#eef2ff" if gid == 3 else "#fef3c7", alpha=0.35, zorder=0)

            g_rmse_rhs = np.array([_rmse(rhs_beta[np.array(g, dtype=int)] - true_beta[np.array(g, dtype=int)]) for g in groups_ref], dtype=float)
            g_rmse_gr = np.array([_rmse(gr_beta[np.array(g, dtype=int)] - true_beta[np.array(g, dtype=int)]) for g in groups_ref], dtype=float)
            gids = np.arange(len(groups_ref))
            bw = 0.36
            ax_bot.bar(gids - bw / 2, g_rmse_rhs, width=bw, color="#9ca3af", label="RHS")
            ax_bot.bar(gids + bw / 2, g_rmse_gr, width=bw, color="#111827", label="GR-RHS")
            ax_bot.set_xticks(gids)
            ax_bot.set_xticklabels([f"G{i+1}" for i in gids.tolist()])
            ax_bot.set_ylim(0.0, 1.12 * bottom_max)
            ax_bot.grid(axis="y", alpha=0.22)

        if c == 0:
            ax_top.set_ylabel("Coefficient")
            ax_bot.set_ylabel("Group RMSE")
        ax_top.set_xlabel("Feature index")
        ax_bot.set_xlabel("Group")

    handles, labels = axes1[0][0].get_legend_handles_labels()
    fig1.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig1.tight_layout()
    fig1.savefig(outdir / "fig1_true_rhs_grrhs_bars.png", dpi=180)
    plt.close(fig1)

    # ------- Figure 2: grouped RMSE tri-panel -------
    fig2, axes2 = plt.subplots(1, 3, figsize=(14.8, 4.8), squeeze=False)
    axes2 = axes2[0]
    metric_triplet = [("strong_rmse", "Strong-signal RMSE"), ("weak_rmse", "Weak-signal RMSE"), ("null_near_rmse", "Null/Near-zero RMSE")]
    for i, (key, title) in enumerate(metric_triplet):
        ax = axes2[i]
        rhs_y = [r[key]["RHS"] for r in rows]
        gr_y = [r[key]["GR-RHS"] for r in rows]
        ax.plot(a_values, rhs_y, marker="o", linewidth=2.0, color="#6b7280", label="RHS")
        ax.plot(a_values, gr_y, marker="o", linewidth=2.0, color="#111827", label="GR-RHS")
        ax.set_title(title)
        ax.set_xlabel("A")
        ax.grid(alpha=0.25)
    axes2[0].set_ylabel("RMSE")
    axes2[2].legend(frameon=False)
    fig2.tight_layout()
    fig2.savefig(outdir / "fig2_grouped_rmse_triplet.png", dpi=180)
    plt.close(fig2)

    # ------- Figure 3: group score ranking bars at reference A -------
    a_ref = a_values[len(a_values) // 2]
    ref_row = next(r for r in rows if abs(r["A"] - a_ref) < 1e-12)
    rhs_scores = np.asarray(ref_row["group_scores"]["RHS"], dtype=float)
    gr_scores = np.asarray(ref_row["group_scores"]["GR-RHS"], dtype=float)
    gid = np.arange(rhs_scores.size)
    w = 0.38
    fig3, ax3 = plt.subplots(figsize=(9.0, 5.0))
    ax3.bar(gid - w / 2, rhs_scores, width=w, color="#6b7280", label="RHS")
    ax3.bar(gid + w / 2, gr_scores, width=w, color="#111827", label="GR-RHS")
    ax3.set_xticks(gid)
    ax3.set_xticklabels([f"G{i+1}" for i in gid.tolist()])
    ax3.set_ylabel(r"score$_g$ = $\sum_{j\in g} |\hat{\beta}_j| / \sqrt{p_g}$")
    ax3.set_title(f"Group score ranking at A={a_ref:g}")
    ax3.grid(axis="y", alpha=0.22)
    ax3.legend(frameon=False)
    fig3.tight_layout()
    fig3.savefig(outdir / "fig3_group_score_ranking.png", dpi=180)
    plt.close(fig3)

    # ------- Figure 4: weak RMSE + group AUPRC vs A -------
    fig4, ax4 = plt.subplots(1, 2, figsize=(12.8, 4.8), squeeze=False)
    axL, axR = ax4[0]
    axL.plot(a_values, [r["weak_rmse"]["RHS"] for r in rows], marker="o", linewidth=2.0, color="#6b7280", label="RHS")
    axL.plot(a_values, [r["weak_rmse"]["GR-RHS"] for r in rows], marker="o", linewidth=2.0, color="#111827", label="GR-RHS")
    axL.set_title("Weak-signal RMSE vs A")
    axL.set_xlabel("A")
    axL.set_ylabel("RMSE")
    axL.grid(alpha=0.25)
    axL.legend(frameon=False)

    axR.plot(a_values, [r["group_auprc"]["RHS"] for r in rows], marker="o", linewidth=2.0, color="#6b7280", label="RHS")
    axR.plot(a_values, [r["group_auprc"]["GR-RHS"] for r in rows], marker="o", linewidth=2.0, color="#111827", label="GR-RHS")
    axR.set_title("Group AUPRC vs A")
    axR.set_xlabel("A")
    axR.set_ylabel("AUPRC")
    axR.set_ylim(0.0, 1.05)
    axR.grid(alpha=0.25)
    axR.legend(frameon=False)
    fig4.tight_layout()
    fig4.savefig(outdir / "fig4_weak_rmse_and_group_auprc_vs_A.png", dpi=180)
    plt.close(fig4)

    # ------- Figure 5: omega explanation (3 representative variables) -------
    if omega_snapshot is None:
        raise RuntimeError("omega snapshot was not computed")
    labels = [v["label"] for v in omega_snapshot["variables"]]
    og = np.array([v["omega_group"] for v in omega_snapshot["variables"]], dtype=float)
    ot = np.array([v["omega_tau"] for v in omega_snapshot["variables"]], dtype=float)
    ol = np.array([v["omega_lambda"] for v in omega_snapshot["variables"]], dtype=float)

    x = np.arange(len(labels))
    width = 0.25
    fig5, ax5 = plt.subplots(figsize=(9.8, 5.0))
    ax5.bar(x - width, og, width=width, color="#1f2937", label="omega_group")
    ax5.bar(x, ot, width=width, color="#4b5563", label="omega_tau")
    ax5.bar(x + width, ol, width=width, color="#9ca3af", label="omega_lambda")
    ax5.axhline(0.0, color="#111827", linewidth=1.0)
    ax5.set_xticks(x)
    ax5.set_xticklabels(labels)
    ax5.set_ylabel("Omega value")
    ax5.set_title(f"GR-RHS omega explanation at A={omega_snapshot['A_ref']}")
    ax5.grid(axis="y", alpha=0.22)
    ax5.legend(frameon=False)
    fig5.tight_layout()
    fig5.savefig(outdir / "fig5_omega_three_variable_explanation.png", dpi=180)
    plt.close(fig5)

    # narrative report
    lines = [
        "# Toy Mechanism Pack (RHS vs GR-RHS)",
        "",
        "This toy example isolates the added value of the group layer by comparing GR-RHS directly with its non-group counterpart (RHS).",
        "",
        "## Key Files",
        "- `toy_mechanism_summary.json`",
        "- `toy_mechanism_coefficients.json`",
        "- `fig1_true_rhs_grrhs_bars.png`",
        "- `fig2_grouped_rmse_triplet.png`",
        "- `fig3_group_score_ranking.png`",
        "- `fig4_weak_rmse_and_group_auprc_vs_A.png`",
        "- `fig5_omega_three_variable_explanation.png`",
    ]
    (outdir / "README.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved mechanism pack to: {outdir}")


if __name__ == "__main__":
    main()
