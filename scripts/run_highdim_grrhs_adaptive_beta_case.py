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
    if isinstance(value, np.ndarray):
        return _json_clean(value.tolist())
    if hasattr(value, "item"):
        try:
            return _json_clean(value.item())
        except Exception:
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (int, str, bool)):
        return value
    try:
        val = float(value)
    except Exception:
        return str(value)
    return val if math.isfinite(val) else None


def _group_scores(X: np.ndarray, y: np.ndarray, groups: list[list[int]]) -> np.ndarray:
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    y_cent = y_arr - float(np.mean(y_arr))
    n = max(int(X_arr.shape[0]), 1)
    scores = []
    for group in groups:
        idx = np.asarray(group, dtype=int)
        if idx.size == 0:
            scores.append(float("nan"))
            continue
        raw = (X_arr[:, idx].T @ y_cent) / float(n)
        scores.append(float(np.linalg.norm(raw, ord=2) / math.sqrt(max(int(idx.size), 1))))
    return np.asarray(scores, dtype=float)


def estimate_adaptive_beta_kappa(
    X: np.ndarray,
    y: np.ndarray,
    groups: list[list[int]],
    *,
    alpha_kappa: float,
    null_quantile: float,
    n_permutations: int,
    seed: int,
    min_beta_kappa: float | None = None,
    max_beta_kappa: float | None = None,
) -> dict[str, object]:
    groups_use = [[int(i) for i in g] for g in groups]
    G = max(len(groups_use), 1)
    observed = _group_scores(X, y, groups_use)
    perm_scores = np.zeros((int(n_permutations), G), dtype=float)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    y_cent = y_arr - float(np.mean(y_arr))
    rng = np.random.default_rng(int(seed))
    for b in range(int(n_permutations)):
        perm_scores[b, :] = _group_scores(X, rng.permutation(y_cent), groups_use)
    thresholds = np.nanquantile(perm_scores, float(null_quantile), axis=0)
    active_mask = np.asarray(observed > thresholds, dtype=bool)
    n_active = int(np.sum(active_mask))
    pi_raw = float(n_active / float(G))
    pi_min = float(1.0 / float(G))
    pi_hat = float(min(max(pi_raw, pi_min), 0.5))
    beta_raw = float(float(alpha_kappa) * (1.0 - pi_hat) / max(pi_hat, 1e-12))
    beta_hat = float(beta_raw)
    if min_beta_kappa is not None:
        beta_hat = float(max(beta_hat, float(min_beta_kappa)))
    if max_beta_kappa is not None:
        beta_hat = float(min(beta_hat, float(max_beta_kappa)))
    return {
        "alpha_kappa": float(alpha_kappa),
        "null_quantile": float(null_quantile),
        "n_permutations": int(n_permutations),
        "n_groups": int(G),
        "n_screen_active_groups": int(n_active),
        "pi_raw": float(pi_raw),
        "pi_min": float(pi_min),
        "pi_hat": float(pi_hat),
        "beta_kappa_raw": float(beta_raw),
        "beta_kappa_hat": float(beta_hat),
        "min_beta_kappa": None if min_beta_kappa is None else float(min_beta_kappa),
        "max_beta_kappa": None if max_beta_kappa is None else float(max_beta_kappa),
        "observed_group_scores": observed.tolist(),
        "empirical_null_thresholds": thresholds.tolist(),
        "active_group_screen_mask": active_mask.tolist(),
    }


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


def run_adaptive_case(
    *,
    config: str,
    setting_id: str,
    replicate: int,
    outdir: str,
    alpha_kappa: float,
    null_quantile: float,
    n_permutations: int,
    warmup: int,
    draws: int,
    adapt_delta: float,
    max_treedepth: int,
    seed_offset: int,
    p0_mode: str,
    min_beta_kappa: float | None = None,
    max_beta_kappa: float | None = None,
    label: str | None = None,
    calibrate_only: bool = False,
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
    calib_seed = int(cfg.runner.seed) + 8301 + 10000 * int(seed_offset) + int(round(1000 * float(null_quantile)))
    calibration = estimate_adaptive_beta_kappa(
        ds.X_train,
        ds.y_train,
        ds.groups,
        alpha_kappa=float(alpha_kappa),
        null_quantile=float(null_quantile),
        n_permutations=int(n_permutations),
        seed=int(calib_seed),
        min_beta_kappa=min_beta_kappa,
        max_beta_kappa=max_beta_kappa,
    )
    beta_hat = float(calibration["beta_kappa_hat"])
    p0_mode_use = str(p0_mode).strip().lower()
    if p0_mode_use == "screen_active":
        p0_use = int(max(1, calibration["n_screen_active_groups"]))
    elif p0_mode_use == "sqrt_groups":
        p0_use = int(max(1, round(math.sqrt(max(len(ds.groups), 1)))))
    elif p0_mode_use == "truth":
        p0_use = int(sum(bool(np.any(np.abs(ds.beta[np.asarray(g, dtype=int)]) > 1e-12)) for g in ds.groups))
    else:
        raise ValueError("--p0-mode must be screen_active, sqrt_groups, or truth")

    payload: dict[str, object] = {
        "replicate": int(replicate),
        "setting_id": str(setting_id),
        "method": "GR_RHS_adaptive_beta",
        "adaptive_calibration": _json_clean(calibration),
        "selected_alpha_kappa": float(alpha_kappa),
        "selected_beta_kappa": float(beta_hat),
        "p0_mode": str(p0_mode),
        "p0_used": int(p0_use),
        "calibrate_only": bool(calibrate_only),
    }
    if bool(calibrate_only):
        payload.update(
            {
                "status": "calibrate_only",
                "converged": None,
                "wall_seconds": float(time.perf_counter() - total_t0),
            }
        )
    else:
        sampler = _sampler_from_args(
            cfg,
            warmup=int(warmup),
            draws=int(draws),
            adapt_delta=float(adapt_delta),
            max_treedepth=int(max_treedepth),
        )
        grrhs_kwargs = dict(cfg.methods.grrhs_kwargs)
        grrhs_kwargs.pop("collapsed_hard_min_warmup", None)
        grrhs_kwargs.pop("collapsed_hard_min_draws", None)
        grrhs_kwargs["alpha_kappa"] = float(alpha_kappa)
        grrhs_kwargs["beta_kappa"] = float(beta_hat)
        grrhs_kwargs.setdefault("tau_target", "groups")
        grrhs_kwargs.setdefault("progress_bar", False)
        fit_t0 = time.perf_counter()
        result = fit_gr_rhs(
            ds.X_train,
            ds.y_train,
            ds.groups,
            task=str(cfg.runner.task),
            seed=int(cfg.runner.seed) + 8101 + 10000 * int(seed_offset),
            p0=int(p0_use),
            sampler=sampler,
            method_name="GR_RHS_adaptive_beta",
            retry_resume_payload=None,
            **{
                **grrhs_kwargs,
                "retry_attempt": 0,
                "collapsed_hard_min_warmup": int(warmup),
                "collapsed_hard_min_draws": int(draws),
            },
        )
        metrics = _evaluate_row(
            result,
            ds.beta,
            X_train=ds.X_train,
            y_train=ds.y_train,
            X_test=ds.X_test,
            y_test=ds.y_test,
        )
        payload.update(
            {
                "status": str(result.status),
                "converged": bool(result.converged),
                "rhat_max": _json_scalar(result.rhat_max),
                "ess_min": _json_scalar(result.bulk_ess_min),
                "divergence_ratio": _json_scalar(result.divergence_ratio),
                "mse_overall": _json_scalar(metrics.get("mse_overall")),
                "mse_signal": _json_scalar(metrics.get("mse_signal")),
                "mse_null": _json_scalar(metrics.get("mse_null")),
                "coverage_95": _json_scalar(metrics.get("coverage_95")),
                "lpd_test": _json_scalar(metrics.get("lpd_test")),
                "wall_seconds": float(time.perf_counter() - total_t0),
                "fit_seconds": float(time.perf_counter() - fit_t0),
                "grrhs_parameters": _json_clean(grrhs_kwargs),
                "diagnostics": _json_clean(result.diagnostics),
                "probe_budget": {
                    "warmup": int(warmup),
                    "draws": int(draws),
                    "adapt_delta": float(adapt_delta),
                    "max_treedepth": int(max_treedepth),
                    "seed_offset": int(seed_offset),
                },
            }
        )

    outdir_path = ROOT / str(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    label_part = ""
    if label:
        safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(label))
        label_part = f"__{safe_label}"
    q_part = str(float(null_quantile)).replace(".", "p")
    stem = f"{setting_id}__GR_RHS_adaptive_beta{label_part}__q{q_part}__r{int(replicate)}__w{int(warmup)}_d{int(draws)}_s{int(seed_offset)}"
    if bool(calibrate_only):
        stem += "__calibrate_only"
    out_path = outdir_path / f"{stem}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"out_path": str(out_path), **payload}


def main() -> int:
    parser = argparse.ArgumentParser(description="Adaptive empirical calibration of GR-RHS beta_kappa from training-only group scores.")
    parser.add_argument("--config", default="Simulation_highdimension/config/highdimension.yaml")
    parser.add_argument("--setting-id", required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--outdir", default="tmp/highdim_grrhs_adaptive_beta")
    parser.add_argument("--alpha-kappa", type=float, default=0.5)
    parser.add_argument("--null-quantile", type=float, required=True)
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=700)
    parser.add_argument("--draws", type=int, default=1600)
    parser.add_argument("--adapt-delta", type=float, default=0.97)
    parser.add_argument("--max-treedepth", type=int, default=14)
    parser.add_argument("--seed-offset", type=int, default=9701)
    parser.add_argument("--p0-mode", choices=["screen_active", "sqrt_groups", "truth"], default="screen_active")
    parser.add_argument("--min-beta-kappa", type=float, default=None)
    parser.add_argument("--max-beta-kappa", type=float, default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--calibrate-only", action="store_true")
    args = parser.parse_args()
    payload = run_adaptive_case(
        config=str(args.config),
        setting_id=str(args.setting_id),
        replicate=int(args.replicate),
        outdir=str(args.outdir),
        alpha_kappa=float(args.alpha_kappa),
        null_quantile=float(args.null_quantile),
        n_permutations=int(args.n_permutations),
        warmup=int(args.warmup),
        draws=int(args.draws),
        adapt_delta=float(args.adapt_delta),
        max_treedepth=int(args.max_treedepth),
        seed_offset=int(args.seed_offset),
        p0_mode=str(args.p0_mode),
        min_beta_kappa=args.min_beta_kappa,
        max_beta_kappa=args.max_beta_kappa,
        label=args.label,
        calibrate_only=bool(args.calibrate_only),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
