from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation_project.src.experiments.analysis.metrics import compute_test_lpd, compute_test_lpd_ppd
from simulation_project.src.experiments.evaluation import _evaluate_row
from simulation_project.src.experiments.methods.fit_gr_rhs import fit_gr_rhs
from simulation_project.src.utils import SamplerConfig
from simulation_second.src.config import load_benchmark_config
from simulation_second.src.dataset import generate_grouped_dataset


def _json_scalar(value):
    if value is None:
        return None
    try:
        val = float(value)
    except Exception:
        return value
    return val if math.isfinite(val) else None


def _json_clean(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _json_clean(value.item())
        except Exception:
            pass
    if isinstance(value, np.ndarray):
        return _json_clean(value.tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (int, str, bool)):
        return value
    try:
        val = float(value)
    except Exception:
        return str(value)
    return val if math.isfinite(val) else None


def _p0_for_grrhs(mode: str, beta: np.ndarray, groups: list[list[int]], tau_target: str) -> int:
    mode_use = str(mode).strip().lower()
    target = str(tau_target).strip().lower()
    if mode_use == "truth":
        if target == "groups":
            return int(sum(bool(np.any(np.abs(np.asarray(beta)[np.asarray(g, dtype=int)]) > 1e-12)) for g in groups))
        return int(np.count_nonzero(np.abs(np.asarray(beta)) > 1e-12))
    if target == "groups":
        G = max(len(groups), 1)
        if mode_use == "sqrt_groups":
            return int(max(1, round(math.sqrt(G))))
        if mode_use.startswith("fraction:"):
            frac = float(mode_use.split(":", 1)[1])
            return int(max(1, round(frac * G)))
        return int(max(1, round(math.sqrt(G))))
    p = int(np.asarray(beta).size)
    if mode_use == "sqrt_groups":
        return int(max(1, round(math.sqrt(max(p, 1)))))
    if mode_use.startswith("fraction:"):
        frac = float(mode_use.split(":", 1)[1])
        return int(max(1, round(frac * p)))
    return int(max(1, round(math.sqrt(max(p, 1)))))


def _split_train_validation(n: int, validation_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n_use = int(n)
    if n_use < 8:
        raise ValueError("Need at least 8 training rows for EB validation split.")
    frac = min(max(float(validation_fraction), 0.05), 0.5)
    n_valid = int(max(4, min(n_use - 4, round(frac * n_use))))
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n_use)
    valid_idx = np.sort(perm[:n_valid])
    train_idx = np.sort(perm[n_valid:])
    return train_idx, valid_idx


def _validation_lpd(result, X_fit: np.ndarray, y_fit: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray) -> tuple[float, float, float]:
    if result.beta_mean is None:
        return float("nan"), float("nan"), float("nan")
    train_resid2 = float(np.mean((np.asarray(y_fit) - np.asarray(X_fit) @ result.beta_mean) ** 2))
    lpd_plugin = compute_test_lpd(result.beta_mean, X_valid, y_valid, sigma2_hat=train_resid2)
    lpd_ppd = compute_test_lpd_ppd(result.beta_draws, X_valid, y_valid, sigma2_hat=train_resid2)
    lpd = lpd_ppd if np.isfinite(lpd_ppd) else lpd_plugin
    return float(lpd), float(lpd_ppd), float(lpd_plugin)


def _sampler_from_args(cfg, *, warmup: int, draws: int, adapt_delta: float, max_treedepth: int) -> SamplerConfig:
    return SamplerConfig(
        chains=max(4, int(cfg.convergence_gate.chains)),
        warmup=int(warmup),
        post_warmup_draws=int(draws),
        adapt_delta=float(adapt_delta),
        max_treedepth=int(max_treedepth),
        strict_adapt_delta=max(0.995, float(adapt_delta)),
        strict_max_treedepth=max(int(max_treedepth), 15),
        max_divergence_ratio=float(cfg.convergence_gate.max_divergence_ratio),
        rhat_threshold=float(cfg.convergence_gate.rhat_threshold),
        ess_threshold=float(cfg.convergence_gate.ess_threshold),
    )


def run_eb_case(
    *,
    config: str,
    setting_id: str,
    replicate: int,
    outdir: str,
    beta_grid: list[float],
    alpha_kappa: float,
    validation_fraction: float,
    p0_mode: str,
    selection_rule: str,
    calibration_warmup: int,
    calibration_draws: int,
    final_warmup: int,
    final_draws: int,
    adapt_delta: float,
    max_treedepth: int,
    seed_offset: int,
    label: str | None = None,
) -> dict[str, object]:
    total_t0 = time.perf_counter()
    cfg = load_benchmark_config(config)
    setting = cfg.setting_map()[str(setting_id)]
    ds = generate_grouped_dataset(
        setting,
        replicate_id=int(replicate),
        master_seed=int(cfg.runner.seed),
        family_specs=cfg.families,
    )

    grrhs_kwargs = dict(cfg.methods.grrhs_kwargs)
    grrhs_kwargs.pop("collapsed_hard_min_warmup", None)
    grrhs_kwargs.pop("collapsed_hard_min_draws", None)
    grrhs_kwargs["alpha_kappa"] = float(alpha_kappa)
    grrhs_kwargs.setdefault("tau_target", "groups")
    grrhs_kwargs.setdefault("progress_bar", False)
    tau_target = str(grrhs_kwargs.get("tau_target", "groups"))
    p0_use = _p0_for_grrhs(str(p0_mode), ds.beta, ds.groups, tau_target)

    split_seed = int(cfg.runner.seed) + 991 + 10000 * int(seed_offset) + int(replicate)
    fit_idx, valid_idx = _split_train_validation(ds.X_train.shape[0], validation_fraction, split_seed)
    X_fit = ds.X_train[fit_idx]
    y_fit = ds.y_train[fit_idx]
    X_valid = ds.X_train[valid_idx]
    y_valid = ds.y_train[valid_idx]

    cal_sampler = _sampler_from_args(
        cfg,
        warmup=int(calibration_warmup),
        draws=int(calibration_draws),
        adapt_delta=float(adapt_delta),
        max_treedepth=int(max_treedepth),
    )
    final_sampler = _sampler_from_args(
        cfg,
        warmup=int(final_warmup),
        draws=int(final_draws),
        adapt_delta=float(adapt_delta),
        max_treedepth=int(max_treedepth),
    )

    candidates: list[dict[str, object]] = []
    for i, beta_kappa in enumerate(beta_grid):
        fit_t0 = time.perf_counter()
        kwargs_i = dict(grrhs_kwargs)
        kwargs_i["beta_kappa"] = float(beta_kappa)
        result = fit_gr_rhs(
            X_fit,
            y_fit,
            ds.groups,
            task=str(cfg.runner.task),
            seed=int(cfg.runner.seed) + 7001 + 10000 * int(seed_offset) + 37 * i,
            p0=int(p0_use),
            sampler=cal_sampler,
            method_name="GR_RHS_EB_CAL",
            retry_resume_payload=None,
            **{
                **kwargs_i,
                "retry_attempt": 0,
                "collapsed_hard_min_warmup": int(calibration_warmup),
                "collapsed_hard_min_draws": int(calibration_draws),
            },
        )
        lpd, lpd_ppd, lpd_plugin = _validation_lpd(result, X_fit, y_fit, X_valid, y_valid)
        candidates.append(
            {
                "beta_kappa": float(beta_kappa),
                "alpha_kappa": float(alpha_kappa),
                "status": str(result.status),
                "converged": bool(result.converged),
                "rhat_max": _json_scalar(result.rhat_max),
                "ess_min": _json_scalar(result.bulk_ess_min),
                "divergence_ratio": _json_scalar(result.divergence_ratio),
                "validation_lpd": _json_scalar(lpd),
                "validation_lpd_ppd": _json_scalar(lpd_ppd),
                "validation_lpd_plugin": _json_scalar(lpd_plugin),
                "wall_seconds": float(time.perf_counter() - fit_t0),
                "diagnostics": _json_clean(result.diagnostics),
            }
        )

    rule = str(selection_rule).strip().lower()
    eligible = [c for c in candidates if np.isfinite(float(c.get("validation_lpd") or float("nan")))]
    if rule == "converged_lpd":
        conv = [c for c in eligible if bool(c.get("converged"))]
        if conv:
            eligible = conv
    if not eligible:
        raise RuntimeError("No EB candidate produced a finite validation log predictive density.")
    selected = max(eligible, key=lambda c: float(c["validation_lpd"]))
    selected_beta = float(selected["beta_kappa"])

    final_kwargs = dict(grrhs_kwargs)
    final_kwargs["beta_kappa"] = float(selected_beta)
    final_t0 = time.perf_counter()
    final_result = fit_gr_rhs(
        ds.X_train,
        ds.y_train,
        ds.groups,
        task=str(cfg.runner.task),
        seed=int(cfg.runner.seed) + 7101 + 10000 * int(seed_offset),
        p0=int(p0_use),
        sampler=final_sampler,
        method_name="GR_RHS_EB",
        retry_resume_payload=None,
        **{
            **final_kwargs,
            "retry_attempt": 0,
            "collapsed_hard_min_warmup": int(final_warmup),
            "collapsed_hard_min_draws": int(final_draws),
        },
    )
    metrics = _evaluate_row(
        final_result,
        ds.beta,
        X_train=ds.X_train,
        y_train=ds.y_train,
        X_test=ds.X_test,
        y_test=ds.y_test,
    )

    payload = {
        "replicate": int(replicate),
        "setting_id": str(setting_id),
        "method": "GR_RHS_EB",
        "status": str(final_result.status),
        "converged": bool(final_result.converged),
        "rhat_max": _json_scalar(final_result.rhat_max),
        "ess_min": _json_scalar(final_result.bulk_ess_min),
        "divergence_ratio": _json_scalar(final_result.divergence_ratio),
        "mse_overall": _json_scalar(metrics.get("mse_overall")),
        "mse_signal": _json_scalar(metrics.get("mse_signal")),
        "mse_null": _json_scalar(metrics.get("mse_null")),
        "coverage_95": _json_scalar(metrics.get("coverage_95")),
        "lpd_test": _json_scalar(metrics.get("lpd_test")),
        "wall_seconds": float(time.perf_counter() - total_t0),
        "final_fit_seconds": float(time.perf_counter() - final_t0),
        "selected_alpha_kappa": float(alpha_kappa),
        "selected_beta_kappa": float(selected_beta),
        "selection_rule": str(selection_rule),
        "p0_mode": str(p0_mode),
        "p0_used": int(p0_use),
        "eb_design": {
            "type": "train_internal_validation_empirical_bayes",
            "test_set_used_for_selection": False,
            "validation_fraction": float(validation_fraction),
            "n_fit": int(X_fit.shape[0]),
            "n_validation": int(X_valid.shape[0]),
            "beta_grid": [float(x) for x in beta_grid],
            "calibration_budget": {
                "warmup": int(calibration_warmup),
                "draws": int(calibration_draws),
                "adapt_delta": float(adapt_delta),
                "max_treedepth": int(max_treedepth),
            },
            "final_budget": {
                "warmup": int(final_warmup),
                "draws": int(final_draws),
                "adapt_delta": float(adapt_delta),
                "max_treedepth": int(max_treedepth),
            },
        },
        "candidates": candidates,
        "final_grrhs_parameters": _json_clean(final_kwargs),
        "diagnostics": _json_clean(final_result.diagnostics),
    }

    outdir_path = ROOT / str(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    label_part = ""
    if label:
        safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(label))
        label_part = f"__{safe_label}"
    grid_part = "grid_" + "-".join(str(float(x)).replace(".", "p") for x in beta_grid)
    stem = f"{setting_id}__GR_RHS_EB{label_part}__r{int(replicate)}__{grid_part}__fw{int(final_warmup)}_fd{int(final_draws)}_s{int(seed_offset)}"
    out_path = outdir_path / f"{stem}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"out_path": str(out_path), **payload}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train-internal empirical-Bayes beta_kappa selection for high-dimensional GR-RHS.")
    parser.add_argument("--config", default="Simulation_highdimension/config/highdimension.yaml")
    parser.add_argument("--setting-id", required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--outdir", default="tmp/highdim_grrhs_eb")
    parser.add_argument("--beta-grid", default="1,4,8,12", help="Comma-separated beta_kappa candidates.")
    parser.add_argument("--alpha-kappa", type=float, default=0.5)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--p0-mode", default="sqrt_groups", help="sqrt_groups, truth, or fraction:<x>. EB-fair default avoids true beta.")
    parser.add_argument("--selection-rule", choices=["converged_lpd", "lpd"], default="converged_lpd")
    parser.add_argument("--calibration-warmup", type=int, default=300)
    parser.add_argument("--calibration-draws", type=int, default=500)
    parser.add_argument("--final-warmup", type=int, required=True)
    parser.add_argument("--final-draws", type=int, required=True)
    parser.add_argument("--adapt-delta", type=float, default=0.97)
    parser.add_argument("--max-treedepth", type=int, default=14)
    parser.add_argument("--seed-offset", type=int, default=9201)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()
    beta_grid = [float(x.strip()) for x in str(args.beta_grid).split(",") if x.strip()]
    if not beta_grid:
        raise ValueError("--beta-grid must contain at least one numeric candidate.")
    payload = run_eb_case(
        config=str(args.config),
        setting_id=str(args.setting_id),
        replicate=int(args.replicate),
        outdir=str(args.outdir),
        beta_grid=beta_grid,
        alpha_kappa=float(args.alpha_kappa),
        validation_fraction=float(args.validation_fraction),
        p0_mode=str(args.p0_mode),
        selection_rule=str(args.selection_rule),
        calibration_warmup=int(args.calibration_warmup),
        calibration_draws=int(args.calibration_draws),
        final_warmup=int(args.final_warmup),
        final_draws=int(args.final_draws),
        adapt_delta=float(args.adapt_delta),
        max_treedepth=int(args.max_treedepth),
        seed_offset=int(args.seed_offset),
        label=args.label,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
