"""Lambda-layer diagnostics package for Bayesian shrinkage models.

This script focuses on:
1) kappa-transformed diagnostics (trace + ESS/Rhat)
2) log-lambda diagnostics
3) HMC energy/divergence diagnostics (when available; e.g., RHS)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from data.generators import generate_synthetic, synthetic_config_from_dict
from data.preprocess import StandardizationConfig, apply_standardization
from grrhs.diagnostics.convergence import effective_sample_size, split_rhat
from grrhs.diagnostics.shrinkage import prior_precision, regularized_lambda, shrinkage_kappa


def _load_run_context(run_dir: Path) -> Tuple[Mapping[str, object], Dict[str, np.ndarray], Dict[str, object]]:
    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8")) or {}
    fold_dir = run_dir / "repeat_001" / "fold_01"
    post = np.load(fold_dir / "posterior_samples.npz")
    posterior = {k: np.asarray(post[k]) for k in post.files}
    fold_summary = json.loads((fold_dir / "fold_summary.json").read_text(encoding="utf-8"))
    return cfg, posterior, fold_summary


def _reconstruct_standardized_train(cfg: Mapping[str, object], run_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    data_cfg = cfg.get("data") or {}
    seeds = cfg.get("seeds") or {}
    seed = data_cfg.get("seed") or seeds.get("data_generation")
    syn_cfg = synthetic_config_from_dict(
        data_cfg,
        seed=seed,
        name=cfg.get("name"),
        task=cfg.get("task"),
    )
    ds = generate_synthetic(syn_cfg)
    X = np.asarray(ds.X, dtype=np.float32)
    y = np.asarray(ds.y, dtype=np.float32).reshape(-1)
    beta_true = np.asarray(ds.beta, dtype=float).reshape(-1)

    fold_arrays = np.load(run_dir / "repeat_001" / "fold_01" / "fold_arrays.npz")
    train_idx = np.asarray(fold_arrays["train_idx"], dtype=int)
    std_cfg_raw = cfg.get("standardization") or {}
    std_cfg = StandardizationConfig(
        X=std_cfg_raw.get("X", "unit_variance"),
        y_center=bool(std_cfg_raw.get("y_center", True)),
    )
    std_train = apply_standardization(X[train_idx], y[train_idx], std_cfg)
    return np.asarray(std_train.X, dtype=float), beta_true


def _as_chain_draws(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    return arr.reshape(arr.shape[0], arr.shape[1], -1)


def _pick_indices(beta_true: np.ndarray) -> Tuple[int, int]:
    strong = int(np.argmax(np.abs(beta_true)))
    weak = int(np.argmin(np.abs(beta_true)))
    return strong, weak


def _ess_rhat_scalar(x_cd: np.ndarray) -> Dict[str, float]:
    rhat = float(np.asarray(split_rhat(x_cd, scalar_param=True)).reshape(-1)[0])
    ess = float(np.asarray(effective_sample_size(x_cd, scalar_param=True)).reshape(-1)[0])
    sd = float(np.std(x_cd.reshape(-1), ddof=1))
    mcse = float(sd / np.sqrt(max(ess, 1.0)))
    return {
        "rhat": rhat,
        "ess_bulk": ess,
        "mcse_over_sd": float(mcse / max(sd, 1e-12)),
    }


def _compute_kappa_draws(
    model_name: str,
    posterior: Dict[str, np.ndarray],
    X_train_std: np.ndarray,
    group_index: np.ndarray,
) -> np.ndarray:
    xtx_diag = np.sum(np.asarray(X_train_std, dtype=float) ** 2, axis=0)

    lam = _as_chain_draws(posterior["lambda"])  # [C, D, P]
    C, D, P = lam.shape
    tau = _as_chain_draws(posterior["tau"]).reshape(C, D)

    sigma = None
    if "sigma" in posterior:
        sigma = _as_chain_draws(posterior["sigma"]).reshape(C, D)
    elif "sigma2" in posterior:
        sigma = np.sqrt(np.maximum(_as_chain_draws(posterior["sigma2"]).reshape(C, D), 1e-12))
    else:
        sigma = np.ones((C, D), dtype=float)

    kappa = np.zeros((C, D, P), dtype=float)
    model_key = str(model_name).lower()
    is_rhs = model_key in {"rhs", "regularized_horseshoe", "regularised_horseshoe", "horseshoe", "hs"}
    is_grrhs = model_key in {"grrhs_gibbs"}
    is_gigg = model_key in {"gigg", "gigg_regression"}

    for c in range(C):
        for d in range(D):
            tau_cd = float(max(tau[c, d], 1e-12))
            sigma_cd = float(max(sigma[c, d], 1e-12))
            lam_cd = np.maximum(lam[c, d], 1e-12)

            if is_rhs:
                c_draw = float(np.asarray(posterior.get("c", np.array([1.0]))).reshape(-1)[0])
                tilde_lam_sq = regularized_lambda(lam_cd, tau=tau_cd, c=max(c_draw, 1e-12))
                # RHS has no group scale phi/gamma in this implementation.
                phi_j = np.ones_like(tilde_lam_sq)
                d_j = prior_precision(phi_j, tau=tau_cd, tilde_lambda_sq=tilde_lam_sq, sigma=sigma_cd)
                kappa[c, d] = shrinkage_kappa(xtx_diag, sigma2=sigma_cd**2, prior_prec=d_j)
            elif is_grrhs:
                phi = _as_chain_draws(posterior["phi"])[c, d]
                phi_j = np.maximum(phi[group_index], 1e-12)
                # GRRHS stores lambda as local scale.
                tilde_lam_sq = np.square(lam_cd)
                d_j = prior_precision(phi_j, tau=tau_cd, tilde_lambda_sq=tilde_lam_sq, sigma=sigma_cd)
                kappa[c, d] = shrinkage_kappa(xtx_diag, sigma2=sigma_cd**2, prior_prec=d_j)
            elif is_gigg:
                gamma = _as_chain_draws(posterior["gamma"])[c, d]
                gamma_j = np.maximum(gamma[group_index], 1e-12)
                # GIGG stores lambda_sq draws in `lambda`.
                lambda_sq = np.maximum(lam_cd, 1e-12)
                # Map to the same precision form: 1 / (sigma^2 * tau^2 * gamma^2 * lambda_sq)
                d_j = 1.0 / np.maximum((sigma_cd**2) * (tau_cd**2) * np.square(gamma_j) * lambda_sq, 1e-12)
                kappa[c, d] = shrinkage_kappa(xtx_diag, sigma2=sigma_cd**2, prior_prec=d_j)
            else:
                raise ValueError(f"Unsupported model for kappa computation: {model_name}")

    return np.clip(kappa, 0.0, 1.0)


def _plot_trace(ax: plt.Axes, x_cd: np.ndarray, title: str) -> None:
    for c in range(x_cd.shape[0]):
        ax.plot(np.arange(x_cd.shape[1]), x_cd[c], lw=0.8, alpha=0.8)
    ax.set_title(title, fontsize=10)


def _plot_density(ax: plt.Axes, values: np.ndarray, title: str) -> None:
    v = np.asarray(values, dtype=float).reshape(-1)
    ax.hist(v, bins=50, density=True, alpha=0.7)
    ax.set_title(title, fontsize=10)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def diagnose_run(run_dir: Path, out_dir: Path) -> None:
    cfg, posterior, fold_summary = _load_run_context(run_dir)
    model_name = str((cfg.get("model") or {}).get("name", "")).lower()
    X_train_std, beta_true = _reconstruct_standardized_train(cfg, run_dir)
    strong_idx, weak_idx = _pick_indices(beta_true)

    groups = ((cfg.get("data") or {}).get("groups")) or []
    if groups:
        gidx = np.zeros(beta_true.size, dtype=int)
        for gid, grp in enumerate(groups):
            gidx[np.asarray(grp, dtype=int)] = gid
    else:
        gidx = np.zeros(beta_true.size, dtype=int)

    lambda_cdP = _as_chain_draws(posterior["lambda"])
    lam_strong = lambda_cdP[:, :, strong_idx]
    lam_weak = lambda_cdP[:, :, weak_idx]
    log_lam_strong = np.log(np.maximum(lam_strong, 1e-12))
    log_lam_weak = np.log(np.maximum(lam_weak, 1e-12))

    kappa_cdP = _compute_kappa_draws(model_name, posterior, X_train_std, gidx)
    kappa_strong = kappa_cdP[:, :, strong_idx]
    kappa_weak = kappa_cdP[:, :, weak_idx]
    kappa_mean = np.mean(kappa_cdP, axis=2)

    stats = {
        "lambda_strong": _ess_rhat_scalar(lam_strong),
        "lambda_weak": _ess_rhat_scalar(lam_weak),
        "log_lambda_strong": _ess_rhat_scalar(log_lam_strong),
        "log_lambda_weak": _ess_rhat_scalar(log_lam_weak),
        "kappa_strong": _ess_rhat_scalar(kappa_strong),
        "kappa_weak": _ess_rhat_scalar(kappa_weak),
        "kappa_mean": _ess_rhat_scalar(kappa_mean),
    }

    hmc_diag = (fold_summary.get("sampler_diagnostics") or {}).get("hmc", {})
    energy_section: Dict[str, object] = {
        "available": bool(hmc_diag),
        "divergences": hmc_diag.get("divergences"),
        "treedepth_hits": hmc_diag.get("treedepth_hits"),
        "ebfmi_per_chain": hmc_diag.get("ebfmi_per_chain"),
        "ebfmi_min": hmc_diag.get("ebfmi_min"),
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    _plot_trace(axes[0, 0], lam_strong, "Trace: lambda(strong)")
    _plot_trace(axes[0, 1], log_lam_strong, "Trace: log(lambda strong)")
    _plot_trace(axes[1, 0], kappa_strong, "Trace: kappa(strong)")
    _plot_trace(axes[1, 1], kappa_mean, "Trace: mean kappa")
    _plot_density(axes[2, 0], log_lam_strong, "Density: log(lambda strong)")
    _plot_density(axes[2, 1], log_lam_weak, "Density: log(lambda weak)")
    fig.suptitle(f"Lambda-layer diagnostics ({run_dir.name})", fontsize=12)
    fig.savefig(out_dir / "lambda_diagnostic_panel.png", dpi=180)
    plt.close(fig)

    if hmc_diag:
        fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
        ebfmi = np.asarray(hmc_diag.get("ebfmi_per_chain") or [], dtype=float)
        if ebfmi.size:
            ax2.bar(np.arange(1, ebfmi.size + 1), ebfmi)
            ax2.axhline(0.3, color="red", linestyle="--", linewidth=1)
            ax2.set_xlabel("chain")
            ax2.set_ylabel("E-BFMI")
            ax2.set_title("Energy diagnostic (E-BFMI)")
            fig2.savefig(out_dir / "energy_ebfmi.png", dpi=180)
        plt.close(fig2)

    summary = {
        "run_dir": str(run_dir),
        "model_name": model_name,
        "selected_indices": {"strong": strong_idx, "weak": weak_idx},
        "stats": stats,
        "energy_diagnostics": energy_section,
        "interpretation": {
            "kappa_improves_lambda": bool(
                stats["kappa_strong"]["rhat"] <= stats["lambda_strong"]["rhat"]
                and stats["kappa_strong"]["ess_bulk"] >= stats["lambda_strong"]["ess_bulk"]
            ),
            "log_lambda_stable": bool(stats["log_lambda_strong"]["rhat"] <= 1.01),
            "hmc_geometry_ok": bool(
                (energy_section["available"] is False)
                or (
                    int(energy_section.get("divergences") or 0) == 0
                    and float(energy_section.get("ebfmi_min") or 1.0) >= 0.3
                )
            ),
        },
    }
    _write_json(out_dir / "lambda_diagnostic_summary.json", summary)
    print(f"[OK] Lambda diagnostics written to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose lambda-layer posterior behavior.")
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    diagnose_run(args.run_dir.expanduser().resolve(), args.out_dir.expanduser().resolve())


if __name__ == "__main__":
    main()

