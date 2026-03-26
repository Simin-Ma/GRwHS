from __future__ import annotations

import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from grrhs.diagnostics.postprocess import compute_diagnostics_from_samples
from grrhs.models.grrhs_gibbs import GRRHS_Gibbs


METHODS = ["Lasso", "Ridge", "Sparse Group Lasso", "Horseshoe", "GR-RHS"]
COLORS = {
    "Lasso": "#2b6cb0",
    "Ridge": "#f2a900",
    "Sparse Group Lasso": "#15803d",
    "Horseshoe": "#7c3aed",
    "GR-RHS": "#111827",
}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_a_dirs(base_dir: Path) -> List[Tuple[float, Path]]:
    out: List[Tuple[float, Path]] = []
    for p in base_dir.glob("tiny_sanity_toyA_path_A*"):
        if not p.is_dir():
            continue
        m = re.search(r"A([0-9]+(?:\.[0-9]+)?)$", p.name)
        if not m:
            continue
        out.append((float(m.group(1)), p))
    out.sort(key=lambda t: t[0])
    return out


def _group_index_from_groups(groups: List[List[int]], p: int) -> np.ndarray:
    gidx = np.zeros(p, dtype=int)
    for gid, g in enumerate(groups):
        for j in g:
            gidx[int(j)] = gid
    return gidx


def _import_toy_dataset_fn(repo_root: Path):
    script_path = repo_root / "scripts" / "run_tiny_sanity_check.py"
    spec = importlib.util.spec_from_file_location("run_tiny_sanity_check_mod", str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    fn = getattr(module, "_toy_mixed_signal_dataset", None)
    if fn is None:
        raise RuntimeError("_toy_mixed_signal_dataset not found in run_tiny_sanity_check.py")
    return fn


def _compute_omega_rows(
    *,
    a_value: float,
    summary: Dict[str, Any],
    toy_dataset_fn,
) -> List[Dict[str, Any]]:
    seed = int(summary["seed"])
    n = int(summary["n"])
    sigma_noise = float(summary["sigma_noise"])
    bayes = summary["bayes_hyperparams"]["grrhs"]

    payload = toy_dataset_fn(
        n=n,
        sigma_noise=sigma_noise,
        seed=seed,
        A=float(a_value),
        plus_corr=False,
    )
    X_raw = np.asarray(payload["X"], dtype=float)
    y_raw = np.asarray(payload["y"], dtype=float)
    groups = [list(map(int, g)) for g in payload["groups"]]
    signal_meta = dict(payload["signal_meta"])

    scaler = StandardScaler(with_mean=True, with_std=True)
    X = scaler.fit_transform(X_raw).astype(float)
    y = (y_raw - float(np.mean(y_raw))).astype(float)

    model = GRRHS_Gibbs(
        c=float(bayes["c"]),
        tau0=float(bayes["tau0"]),
        eta=float(bayes["eta"]),
        s0=float(bayes["s0"]),
        use_groups=True,
        iters=int(bayes["iters"]),
        burnin=int(bayes["burnin"]),
        thin=1,
        seed=seed + 23,
        num_chains=1,
        tau_slice_w=float(bayes["tau_slice_w"]),
        tau_slice_m=int(bayes["tau_slice_m"]),
    ).fit(X, y, groups=groups)

    lam = np.asarray(model.lambda_samples_, dtype=float)
    tau = np.asarray(model.tau_samples_, dtype=float).reshape(-1)
    phi = np.asarray(model.phi_samples_, dtype=float)
    sigma = np.sqrt(np.maximum(np.asarray(model.sigma2_samples_, dtype=float).reshape(-1), 1e-12))

    p = X.shape[1]
    gidx = _group_index_from_groups(groups, p)
    diag = compute_diagnostics_from_samples(
        X=X,
        group_index=gidx,
        c=float(bayes["c"]),
        eps=1e-8,
        lambda_=lam,
        tau=tau,
        phi=phi,
        sigma=sigma,
    )

    mech = signal_meta.get("mechanism_sets", {}) if isinstance(signal_meta, dict) else {}
    sets = {
        "strong": np.asarray(mech.get("strong_idx", []), dtype=int),
        "weak": np.asarray(mech.get("weak_idx", []), dtype=int),
        "null": np.asarray(mech.get("null_idx", []), dtype=int),
    }
    per = diag.per_coeff
    rows: List[Dict[str, Any]] = []
    for set_name, idx in sets.items():
        idx = idx[(idx >= 0) & (idx < p)]
        if idx.size == 0:
            continue
        og = np.asarray(per["omega_group"][idx], dtype=float)
        ot = np.asarray(per["omega_tau"][idx], dtype=float)
        ol = np.asarray(per["omega_lambda"][idx], dtype=float)
        denom = np.abs(og) + np.abs(ot) + np.abs(ol)
        denom = np.maximum(denom, 1e-12)
        rows.append(
            {
                "A": a_value,
                "set": set_name,
                "omega_group_raw_mean": float(np.mean(og)),
                "omega_tau_raw_mean": float(np.mean(ot)),
                "omega_lambda_raw_mean": float(np.mean(ol)),
                "omega_group_share": float(np.mean(np.abs(og) / denom)),
                "omega_tau_share": float(np.mean(np.abs(ot) / denom)),
                "omega_lambda_share": float(np.mean(np.abs(ol) / denom)),
            }
        )
    return rows


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    outputs_dir = repo_root / "outputs"
    outdir = outputs_dir / "toyA_path_five_figures"
    outdir.mkdir(parents=True, exist_ok=True)

    a_dirs = _discover_a_dirs(outputs_dir)
    if not a_dirs:
        raise RuntimeError("No outputs/tiny_sanity_toyA_path_A* directories found.")

    coeff_by_a: Dict[float, Dict[str, Any]] = {}
    summary_by_a: Dict[float, Dict[str, Any]] = {}

    for a, d in a_dirs:
        coeff_by_a[a] = _load_json(d / "tiny_sanity_coefficients.json")
        summary_by_a[a] = _load_json(d / "tiny_sanity_summary.json")

    a_values = [a for a, _ in a_dirs]
    p = len(next(iter(coeff_by_a.values()))["True"])
    x = np.arange(p)

    # Figure 1: True beta bar chart by A.
    ncols = 2
    nrows = math.ceil(len(a_values) / ncols)
    fig1, axes1 = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 3.6 * nrows), squeeze=False)
    for idx, a in enumerate(a_values):
        ax = axes1[idx // ncols][idx % ncols]
        true_beta = np.asarray(coeff_by_a[a]["True"], dtype=float)
        ax.bar(x, true_beta, color="#8e1b1b", width=0.85)
        ax.set_title(f"True beta (A={a:g})")
        ax.set_xlabel("Feature index")
        ax.set_ylabel("Coefficient")
        ax.grid(axis="y", alpha=0.22)
    for idx in range(len(a_values), nrows * ncols):
        axes1[idx // ncols][idx % ncols].axis("off")
    fig1.tight_layout()
    fig1.savefig(outdir / "fig1_true_beta_bar_by_A.png", dpi=180)
    plt.close(fig1)

    # Figure 2: Posterior mean beta-hat bar charts by method and A.
    fig2, axes2 = plt.subplots(
        nrows=len(a_values),
        ncols=len(METHODS),
        figsize=(4.2 * len(METHODS), 2.6 * len(a_values)),
        squeeze=False,
    )
    for r, a in enumerate(a_values):
        true_beta = np.asarray(coeff_by_a[a]["True"], dtype=float)
        for c, method in enumerate(METHODS):
            ax = axes2[r][c]
            beta_hat = np.asarray(coeff_by_a[a][method], dtype=float)
            ax.bar(x, beta_hat, color=COLORS[method], width=0.85, alpha=0.9)
            ax.plot(x, true_beta, color="#8e1b1b", linewidth=1.1, alpha=0.9)
            if r == 0:
                ax.set_title(method)
            if c == 0:
                ax.set_ylabel(f"A={a:g}\ncoef")
            if r == len(a_values) - 1:
                ax.set_xlabel("Feature")
            ax.grid(axis="y", alpha=0.2)
    fig2.tight_layout()
    fig2.savefig(outdir / "fig2_posterior_mean_beta_hat_bars.png", dpi=180)
    plt.close(fig2)

    # Figure 3: weak-signal RMSE vs A.
    fig3, ax3 = plt.subplots(figsize=(8.8, 5.4))
    for method in METHODS:
        ys = []
        for a in a_values:
            rows = summary_by_a[a]["mechanism_rmse"]
            row = next(r for r in rows if r["method"] == method)
            ys.append(float(row["rmse_weak"]))
        ax3.plot(a_values, ys, marker="o", linewidth=2.0, label=method, color=COLORS[method])
    ax3.set_xlabel("A")
    ax3.set_ylabel("Weak-signal RMSE")
    ax3.set_title("Weak-Signal RMSE vs A")
    ax3.grid(alpha=0.25)
    ax3.legend(frameon=False, ncol=2)
    fig3.tight_layout()
    fig3.savefig(outdir / "fig3_weak_signal_rmse_vs_A.png", dpi=180)
    plt.close(fig3)

    # Figure 4: group AUPRC vs A (mean_abs score).
    fig4, ax4 = plt.subplots(figsize=(8.8, 5.4))
    for method in METHODS:
        ys = []
        for a in a_values:
            rows = summary_by_a[a]["group_selection"]
            row = next(r for r in rows if r["method"] == method)
            ys.append(float(row["scores"]["mean_abs"]["AUPRC_group"]))
        ax4.plot(a_values, ys, marker="o", linewidth=2.0, label=method, color=COLORS[method])
    ax4.set_xlabel("A")
    ax4.set_ylabel("Group AUPRC (mean_abs)")
    ax4.set_ylim(0.0, 1.05)
    ax4.set_title("Group AUPRC vs A")
    ax4.grid(alpha=0.25)
    ax4.legend(frameon=False, ncol=2)
    fig4.tight_layout()
    fig4.savefig(outdir / "fig4_group_auprc_vs_A.png", dpi=180)
    plt.close(fig4)

    # Figure 5: GR-RHS omega triplet explanation vs A.
    toy_dataset_fn = _import_toy_dataset_fn(repo_root)
    omega_rows: List[Dict[str, Any]] = []
    for a in a_values:
        omega_rows.extend(_compute_omega_rows(a_value=a, summary=summary_by_a[a], toy_dataset_fn=toy_dataset_fn))

    (outdir / "omega_triplet_summary.json").write_text(json.dumps(omega_rows, indent=2), encoding="utf-8")
    sets = ["strong", "weak", "null"]
    comps = [
        ("omega_group_share", "omega_group_share"),
        ("omega_tau_share", "omega_tau_share"),
        ("omega_lambda_share", "omega_lambda_share"),
    ]
    fig5, axes5 = plt.subplots(nrows=1, ncols=3, figsize=(15.2, 4.8), squeeze=False)
    for i, set_name in enumerate(sets):
        ax = axes5[0][i]
        for comp_key, comp_label in comps:
            ys = []
            for a in a_values:
                row = next(r for r in omega_rows if float(r["A"]) == float(a) and r["set"] == set_name)
                ys.append(float(row[comp_key]))
            ax.plot(a_values, ys, marker="o", linewidth=2.0, label=comp_label)
        ax.set_title(f"{set_name} set")
        ax.set_xlabel("A")
        if i == 0:
            ax.set_ylabel("Mean omega")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)
        if i == 2:
            ax.legend(frameon=False, loc="best")
    fig5.suptitle("GR-RHS omega triplet explanation (relative contribution shares)")
    fig5.tight_layout(rect=[0, 0, 1, 0.95])
    fig5.savefig(outdir / "fig5_grrhs_omega_triplet_explanation.png", dpi=180)
    plt.close(fig5)

    print(f"Saved five figures to: {outdir}")
    print(" - fig1_true_beta_bar_by_A.png")
    print(" - fig2_posterior_mean_beta_hat_bars.png")
    print(" - fig3_weak_signal_rmse_vs_A.png")
    print(" - fig4_group_auprc_vs_A.png")
    print(" - fig5_grrhs_omega_triplet_explanation.png")


if __name__ == "__main__":
    main()
