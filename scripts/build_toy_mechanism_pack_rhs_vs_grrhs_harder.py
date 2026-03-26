from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, List

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
        [float(np.sum(np.abs(beta_hat[np.array(g, dtype=int)])) / np.sqrt(max(len(g), 1))) for g in groups],
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


def _apply_harder_toy(
    payload: Dict[str, Any],
    *,
    sigma_noise: float,
    seed: int,
    weak_scale: float,
    near_zero_bump: float,
    light_rho: float,
) -> Dict[str, Any]:
    X = np.asarray(payload["X"], dtype=float).copy()
    beta = np.asarray(payload["beta"], dtype=float).copy()
    groups = [list(map(int, g)) for g in payload["groups"]]
    signal_meta = dict(payload["signal_meta"])
    tags = signal_meta.get("tags", {}) if isinstance(signal_meta, dict) else {}

    weak_idx = np.array(tags.get("weak", []), dtype=int)
    near_zero_idx = np.array(tags.get("near_zero", []), dtype=int)

    if weak_idx.size:
        beta[weak_idx] = beta[weak_idx] * float(weak_scale)

    if near_zero_idx.size and near_zero_bump != 0.0:
        signs = np.sign(beta[near_zero_idx])
        signs[signs == 0.0] = 1.0
        beta[near_zero_idx] = beta[near_zero_idx] + float(near_zero_bump) * signs

    if light_rho > 0.0:
        rng_corr = np.random.default_rng(int(seed) + 17)
        rho = float(light_rho)
        for g in groups:
            idx = np.array(g, dtype=int)
            z = rng_corr.standard_normal((X.shape[0], 1))
            X[:, idx] = np.sqrt(max(1.0 - rho, 1e-8)) * X[:, idx] + np.sqrt(rho) * z

    rng_noise = np.random.default_rng(int(seed) + 101)
    y = X @ beta + rng_noise.normal(0.0, float(sigma_noise), size=X.shape[0])

    out = dict(payload)
    out["X"] = X
    out["y"] = y
    out["beta"] = beta
    out["signal_meta"] = signal_meta
    out["harder_profile"] = {
        "weak_scale": float(weak_scale),
        "near_zero_bump": float(near_zero_bump),
        "light_group_rho": float(light_rho),
        "description": "weak signals slightly reduced, near-zero distractors strengthened, light within-group correlation added",
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RHS vs GR-RHS toy mechanism pack under light harder settings.")
    parser.add_argument("--seed", type=int, default=196)
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--sigma-noise", type=float, default=1.0)
    parser.add_argument("--a-values", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0, 5.0])
    parser.add_argument("--weak-scale", type=float, default=0.97)
    parser.add_argument("--near-zero-bump", type=float, default=0.008)
    parser.add_argument("--light-rho", type=float, default=0.03)
    parser.add_argument("--rhs-scale-global", type=float, default=0.3)
    parser.add_argument("--gr-c", type=float, default=4.0)
    parser.add_argument("--gr-tau0-multiplier", type=float, default=1.3)
    parser.add_argument("--outdir", type=str, default="outputs/toy_rhs_grrhs_mechanism_pack_harder")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    outdir = (repo_root / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    toy_fn = _import_toy_fn(repo_root)

    seed = int(args.seed)
    n = int(args.n)
    sigma_noise = float(args.sigma_noise)
    a_values = [float(v) for v in args.a_values]

    rhs_cfg = dict(
        scale_global=float(args.rhs_scale_global),
        num_warmup=300,
        num_samples=300,
        num_chains=1,
        thinning=1,
        target_accept_prob=0.9,
    )
    gr_cfg = dict(
        c=float(args.gr_c),
        tau0_multiplier=float(args.gr_tau0_multiplier),
        eta=1.0,
        s0=1.0,
        iters=1200,
        burnin=600,
        tau_slice_w=0.25,
        tau_slice_m=180,
    )

    rows: List[Dict[str, Any]] = []
    coeff_by_a: Dict[str, Dict[str, List[float]]] = {}
    omega_snapshot: Dict[str, Any] | None = None
    # Use the middle A on the user-provided integer path as omega reference.
    omega_a_ref = a_values[len(a_values) // 2]

    for a in a_values:
        payload_base = toy_fn(n=n, sigma_noise=sigma_noise, seed=seed, A=float(a), plus_corr=False)
        payload = _apply_harder_toy(
            payload_base,
            sigma_noise=sigma_noise,
            seed=seed,
            weak_scale=float(args.weak_scale),
            near_zero_bump=float(args.near_zero_bump),
            light_rho=float(args.light_rho),
        )

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
        strong_idx = np.array(tags.get("strong", []), dtype=int)
        weak_idx = np.array(tags.get("weak", []), dtype=int)
        near_zero_idx = np.array(tags.get("near_zero", []), dtype=int)
        null_idx = np.array(tags.get("null", []), dtype=int)

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
                "group_scores": {"RHS": score_rhs.tolist(), "GR-RHS": score_gr.tolist()},
                "group2_strong_abs_error": {
                    "RHS": float(abs(beta_rhs[g2[0]] - beta_true_std[g2[0]])),
                    "GR-RHS": float(abs(beta_gr[g2[0]] - beta_true_std[g2[0]])),
                },
                "group3_weak_rmse": {
                    "RHS": _rmse(beta_rhs[g3] - beta_true_std[g3]),
                    "GR-RHS": _rmse(beta_gr[g3] - beta_true_std[g3]),
                },
                "group4_mixed_rmse": {
                    "RHS": _rmse(beta_rhs[g4] - beta_true_std[g4]),
                    "GR-RHS": _rmse(beta_gr[g4] - beta_true_std[g4]),
                },
                "group5_nearzero_abs_mean": {
                    "RHS": float(np.mean(np.abs(beta_rhs[g5]))),
                    "GR-RHS": float(np.mean(np.abs(beta_gr[g5]))),
                },
            }
        )

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
            reps = [("Group2 strong", strong_rep), ("Group3 weak", weak_rep), ("Group5 near-zero", near_rep)]
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

    strict_checks = {
        "strong_rmse_not_worse_all_A": all(r["strong_rmse"]["GR-RHS"] <= r["strong_rmse"]["RHS"] for r in rows),
        "weak_rmse_better_all_A": all(r["weak_rmse"]["GR-RHS"] < r["weak_rmse"]["RHS"] for r in rows),
        "null_near_rmse_not_worse_all_A": all(r["null_near_rmse"]["GR-RHS"] <= r["null_near_rmse"]["RHS"] for r in rows),
        "group_auprc_not_worse_all_A": all(r["group_auprc"]["GR-RHS"] >= r["group_auprc"]["RHS"] for r in rows),
        "ranking_g3g4_up_g5_down_all_A": all(
            (r["group_ranking"]["GR-RHS"]["group3"] <= r["group_ranking"]["RHS"]["group3"])
            and (r["group_ranking"]["GR-RHS"]["group4"] <= r["group_ranking"]["RHS"]["group4"])
            and (r["group_ranking"]["GR-RHS"]["group5"] >= r["group_ranking"]["RHS"]["group5"])
            for r in rows
        ),
    }

    story_checks = {
        "group2_not_worse_than_rhs": all(r["group2_strong_abs_error"]["GR-RHS"] <= r["group2_strong_abs_error"]["RHS"] for r in rows),
        "group3_weak_better_than_rhs": all(r["group3_weak_rmse"]["GR-RHS"] < r["group3_weak_rmse"]["RHS"] for r in rows),
        "group4_mixed_better_than_rhs": all(r["group4_mixed_rmse"]["GR-RHS"] < r["group4_mixed_rmse"]["RHS"] for r in rows),
        "group5_nearzero_not_activated": all(r["group5_nearzero_abs_mean"]["GR-RHS"] <= r["group5_nearzero_abs_mean"]["RHS"] for r in rows),
        "group_ranking_focus": all(
            (r["group_ranking"]["GR-RHS"]["group3"] <= 3)
            and (r["group_ranking"]["GR-RHS"]["group4"] <= 2)
            and (r["group_ranking"]["GR-RHS"]["group5"] == 6)
            for r in rows
        ),
    }

    summary = {
        "narrative": "This toy example isolates the added value of the group layer by comparing GR-RHS directly with its non-group counterpart (RHS).",
        "harder_profile": {
            "weak_scale": float(args.weak_scale),
            "near_zero_bump": float(args.near_zero_bump),
            "light_group_rho": float(args.light_rho),
        },
        "rhs_config": rhs_cfg,
        "grrhs_config": gr_cfg,
        "metrics_by_A": rows,
        "strict_checks": strict_checks,
        "strict_all_passed": bool(all(strict_checks.values())),
        "story_checks": story_checks,
        "story_all_passed": bool(all(story_checks.values())),
        "omega_snapshot": omega_snapshot,
    }

    (outdir / "toy_mechanism_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (outdir / "toy_mechanism_coefficients.json").write_text(json.dumps(coeff_by_a, indent=2), encoding="utf-8")

    nrows = 3
    ncols = len(a_values)
    fig1, axes1 = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.8 * ncols, 2.8 * nrows), squeeze=False)
    for c, a in enumerate(a_values):
        key = f"A{a:g}"
        true_beta = np.asarray(coeff_by_a[key]["True"], dtype=float)
        rhs_beta = np.asarray(coeff_by_a[key]["RHS"], dtype=float)
        gr_beta = np.asarray(coeff_by_a[key]["GR-RHS"], dtype=float)
        xs = np.arange(true_beta.size)
        for r, (title, vals, color) in enumerate(
            [("True beta", true_beta, "#8e1b1b"), ("RHS posterior mean", rhs_beta, "#6b7280"), ("GR-RHS posterior mean", gr_beta, "#111827")]
        ):
            ax = axes1[r][c]
            ax.bar(xs, vals, color=color, width=0.85)
            if r == 0:
                ax.set_title(f"A={a:g}")
            if c == 0:
                ax.set_ylabel(title)
            if r == nrows - 1:
                ax.set_xlabel("Feature index")
            ax.grid(axis="y", alpha=0.22)
    fig1.tight_layout()
    fig1.savefig(outdir / "fig1_true_rhs_grrhs_bars.png", dpi=180)
    plt.close(fig1)

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

    report_lines = [
        "# Toy Mechanism Pack (RHS vs GR-RHS, harder toy)",
        "",
        "This toy example isolates the added value of the group layer by comparing GR-RHS directly with its non-group counterpart (RHS).",
        "",
        "## Harder Profile",
        f"- weak_scale = {args.weak_scale}",
        f"- near_zero_bump = {args.near_zero_bump}",
        f"- light_group_rho = {args.light_rho}",
        "",
        "## Strict Checks",
    ]
    for k, v in strict_checks.items():
        report_lines.append(f"- {k}: {'PASS' if v else 'FAIL'}")
    report_lines.append(f"- strict_all_passed: {'PASS' if summary['strict_all_passed'] else 'FAIL'}")
    report_lines.append("")
    report_lines.append("## Story Checks")
    for k, v in story_checks.items():
        report_lines.append(f"- {k}: {'PASS' if v else 'FAIL'}")
    report_lines.append(f"- story_all_passed: {'PASS' if summary['story_all_passed'] else 'FAIL'}")
    report_lines.append("")
    report_lines.append("## Key Files")
    report_lines.extend(
        [
            "- toy_mechanism_summary.json",
            "- toy_mechanism_coefficients.json",
            "- fig1_true_rhs_grrhs_bars.png",
            "- fig2_grouped_rmse_triplet.png",
            "- fig3_group_score_ranking.png",
            "- fig4_weak_rmse_and_group_auprc_vs_A.png",
            "- fig5_omega_three_variable_explanation.png",
        ]
    )
    (outdir / "README.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved harder mechanism pack to: {outdir}")
    print(f"strict_all_passed={summary['strict_all_passed']}")
    print(f"story_all_passed={summary['story_all_passed']}")


if __name__ == "__main__":
    main()
