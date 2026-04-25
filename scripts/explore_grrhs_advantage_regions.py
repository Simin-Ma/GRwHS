from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation_project.src.experiments.analysis.metrics import (
    ci_length_and_coverage,
    group_auroc,
    group_l2_score,
    mse_null_signal_overall,
)
from simulation_project.src.experiments.fitting import _fit_all_methods
from simulation_project.src.utils import MASTER_SEED, SamplerConfig, canonical_groups, ensure_dir, standardize_columns


@dataclass(frozen=True)
class Scenario:
    name: str
    group_sizes: tuple[int, ...]
    family: str
    description: str
    decoy_pairs: tuple[tuple[int, int], ...] = ()


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="hetero_mixed",
        group_sizes=(6, 6, 6, 6, 6, 6),
        family="mixed-occupancy",
        description="Active groups mix spike, partial, and dense weak coefficients.",
    ),
    Scenario(
        name="paired_decoy",
        group_sizes=(6, 6, 6, 6, 6, 6),
        family="decoy-correlation",
        description="One null group is coupled to an active group by an extra latent factor.",
        decoy_pairs=((2, 3),),
    ),
    Scenario(
        name="size_imbalance",
        group_sizes=(3, 3, 6, 6, 12, 12),
        family="unequal-group-size",
        description="Small strong groups compete against larger mostly-null groups.",
    ),
)


def _build_beta(scenario: Scenario) -> tuple[np.ndarray, np.ndarray]:
    groups = canonical_groups(list(scenario.group_sizes))
    p = int(sum(scenario.group_sizes))
    beta = np.zeros(p, dtype=float)
    active_group = np.zeros(len(groups), dtype=int)

    if scenario.name == "hetero_mixed":
        beta[np.asarray(groups[0], dtype=int)[0]] = 1.45
        beta[np.asarray(groups[1], dtype=int)[:3]] = np.asarray([0.82, -0.74, 0.58], dtype=float)
        beta[np.asarray(groups[2], dtype=int)] = np.asarray([0.34, -0.31, 0.28, -0.25, 0.22, -0.19], dtype=float)
        active_group[[0, 1, 2]] = 1
    elif scenario.name == "paired_decoy":
        beta[np.asarray(groups[0], dtype=int)[:2]] = np.asarray([1.1, -0.92], dtype=float)
        beta[np.asarray(groups[1], dtype=int)[:4]] = np.asarray([0.62, -0.55, 0.48, -0.43], dtype=float)
        beta[np.asarray(groups[2], dtype=int)] = np.asarray([0.30, 0.27, -0.25, 0.22, -0.20, 0.18], dtype=float)
        active_group[[0, 1, 2]] = 1
    elif scenario.name == "size_imbalance":
        beta[np.asarray(groups[0], dtype=int)[:2]] = np.asarray([1.18, -0.88], dtype=float)
        beta[np.asarray(groups[2], dtype=int)[:3]] = np.asarray([0.52, -0.44, 0.36], dtype=float)
        beta[np.asarray(groups[4], dtype=int)[:4]] = np.asarray([0.24, 0.22, -0.19, 0.16], dtype=float)
        active_group[[0, 2, 4]] = 1
    else:
        raise ValueError(f"unknown scenario: {scenario.name}")

    return beta, active_group


def _sample_custom_design(
    *,
    n: int,
    scenario: Scenario,
    rho_within: float,
    rho_between: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    groups = canonical_groups(list(scenario.group_sizes))
    p = int(sum(scenario.group_sizes))

    rho_w = float(rho_within)
    rho_b = float(rho_between)
    if not (0.0 <= rho_b < rho_w < 1.0):
        raise ValueError("expected 0 <= rho_between < rho_within < 1")

    z_global = rng.normal(size=(int(n), 1))
    z_groups = rng.normal(size=(int(n), len(groups)))
    z_pair = rng.normal(size=(int(n), len(scenario.decoy_pairs))) if scenario.decoy_pairs else np.zeros((int(n), 0))
    noise = rng.normal(size=(int(n), p))
    X = np.zeros((int(n), p), dtype=float)

    pair_strength = min(rho_w - rho_b, 0.14) if scenario.decoy_pairs else 0.0
    within_resid = max(rho_w - rho_b - pair_strength, 1e-6)
    noise_scale = max(1.0 - rho_b - within_resid - pair_strength, 1e-6)

    pair_lookup: dict[int, list[int]] = {}
    for pair_id, (ga, gb) in enumerate(scenario.decoy_pairs):
        pair_lookup.setdefault(int(ga), []).append(pair_id)
        pair_lookup.setdefault(int(gb), []).append(pair_id)

    col = 0
    for gid, group in enumerate(groups):
        pair_cols = pair_lookup.get(int(gid), [])
        pair_signal = np.sum(z_pair[:, pair_cols], axis=1, keepdims=True) if pair_cols else 0.0
        base = (
            math.sqrt(rho_b) * z_global
            + math.sqrt(within_resid) * z_groups[:, [gid]]
        )
        if pair_cols:
            base = base + math.sqrt(pair_strength / max(len(pair_cols), 1)) * pair_signal
        width = len(group)
        X[:, col : col + width] = base + math.sqrt(noise_scale) * noise[:, col : col + width]
        col += width

    X = standardize_columns(X)
    cov_x = np.cov(X, rowvar=False, ddof=0)
    return X, np.asarray(cov_x, dtype=float)


def _sigma2_for_target_snr(beta: np.ndarray, cov_x: np.ndarray, target_snr: float) -> float:
    signal_var = float(beta.T @ cov_x @ beta)
    return max(signal_var / max(float(target_snr), 1e-8), 1e-8)


def _count_active_groups(beta: np.ndarray, group_sizes: tuple[int, ...]) -> int:
    groups = canonical_groups(list(group_sizes))
    return int(sum(1 for g in groups if np.any(np.abs(beta[np.asarray(g, dtype=int)]) > 1e-12)))


def _evaluate_fit(
    *,
    method: str,
    res: Any,
    beta_true: np.ndarray,
    groups: list[list[int]],
    active_group: np.ndarray,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "method": str(method),
        "status": str(res.status),
        "converged": bool(res.converged),
        "runtime_seconds": float(res.runtime_seconds),
        "mse_overall": float("nan"),
        "mse_signal": float("nan"),
        "mse_null": float("nan"),
        "coverage_95": float("nan"),
        "avg_ci_length": float("nan"),
        "group_auroc": float("nan"),
        "nonzero_hat": float("nan"),
        "error": str(getattr(res, "error", "") or ""),
    }
    if res.beta_mean is None:
        return out

    beta_hat = np.asarray(res.beta_mean, dtype=float).reshape(-1)
    out.update(mse_null_signal_overall(beta_hat, beta_true))
    ci_len, cover = ci_length_and_coverage(beta_true, res.beta_draws)
    out["coverage_95"] = float(cover)
    out["avg_ci_length"] = float(ci_len)
    out["group_auroc"] = float(group_auroc(group_l2_score(beta_hat, groups), active_group))
    out["nonzero_hat"] = int(np.sum(np.abs(beta_hat) > 0.05))

    diag = getattr(res, "diagnostics", None)
    if isinstance(diag, dict):
        mmle = diag.get("mmle_estimate")
        if isinstance(mmle, dict):
            q_est = mmle.get("q_estimate")
            if isinstance(q_est, list):
                out["mmle_q_estimate_json"] = json.dumps([float(v) for v in q_est])
    return out


def _run_one_setting(
    *,
    scenario: Scenario,
    rho_within: float,
    rho_between: float,
    target_snr: float,
    n: int,
    n_test: int,
    replicate_id: int,
    seed: int,
    methods: list[str],
    sampler: SamplerConfig,
    grrhs_kwargs: dict[str, Any],
    gigg_config: dict[str, Any],
    enforce_bayes_convergence: bool,
    max_convergence_retries: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    beta_true, active_group = _build_beta(scenario)
    groups = canonical_groups(list(scenario.group_sizes))

    X_train, cov_x = _sample_custom_design(
        n=int(n),
        scenario=scenario,
        rho_within=float(rho_within),
        rho_between=float(rho_between),
        seed=int(seed),
    )
    sigma2 = _sigma2_for_target_snr(beta_true, cov_x, target_snr=float(target_snr))
    y_train = X_train @ beta_true + rng.normal(0.0, math.sqrt(float(sigma2)), size=int(n))

    results = _fit_all_methods(
        X_train,
        y_train,
        groups,
        task="gaussian",
        seed=int(seed),
        p0=int(np.sum(np.abs(beta_true) > 1e-12)),
        p0_groups=_count_active_groups(beta_true, scenario.group_sizes),
        sampler=sampler,
        grrhs_kwargs=dict(grrhs_kwargs),
        methods=list(methods),
        gigg_config=dict(gigg_config),
        enforce_bayes_convergence=bool(enforce_bayes_convergence),
        max_convergence_retries=int(max_convergence_retries),
        method_jobs=1,
    )

    X_test, _ = _sample_custom_design(
        n=int(n_test),
        scenario=scenario,
        rho_within=float(rho_within),
        rho_between=float(rho_between),
        seed=int(seed) + 999,
    )
    y_test = X_test @ beta_true + rng.normal(0.0, math.sqrt(float(sigma2)), size=int(n_test))
    test_signal_norm = float(np.mean((X_test @ beta_true) ** 2))

    rows: list[dict[str, Any]] = []
    for method, res in results.items():
        row = _evaluate_fit(
            method=method,
            res=res,
            beta_true=beta_true,
            groups=groups,
            active_group=active_group,
        )
        row.update(
            {
                "scenario": scenario.name,
                "scenario_family": scenario.family,
                "scenario_description": scenario.description,
                "rho_within": float(rho_within),
                "rho_between": float(rho_between),
                "target_snr": float(target_snr),
                "n_train": int(n),
                "n_test": int(n_test),
                "replicate_id": int(replicate_id),
                "seed": int(seed),
                "sigma2": float(sigma2),
                "test_signal_norm": float(test_signal_norm),
                "group_sizes_json": json.dumps(list(scenario.group_sizes)),
            }
        )
        rows.append(row)
    return rows


def _aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(["scenario", "rho_within", "target_snr", "method"], as_index=False)
        .agg(
            n_runs=("method", "count"),
            n_ok=("status", lambda s: int(np.sum(np.asarray(s) == "ok"))),
            n_converged=("converged", lambda s: int(np.sum(np.asarray(s, dtype=bool)))),
            mse_overall_mean=("mse_overall", "mean"),
            mse_signal_mean=("mse_signal", "mean"),
            mse_null_mean=("mse_null", "mean"),
            coverage_95_mean=("coverage_95", "mean"),
            avg_ci_length_mean=("avg_ci_length", "mean"),
            group_auroc_mean=("group_auroc", "mean"),
            runtime_mean=("runtime_seconds", "mean"),
        )
        .sort_values(["scenario", "rho_within", "target_snr", "mse_overall_mean"], ascending=[True, True, True, True])
    )


def _compare_gr_vs_gigg(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["scenario", "rho_within", "target_snr"]
    for _, chunk in summary.groupby(keys, as_index=False):
        gr = chunk.loc[chunk["method"] == "GR_RHS"]
        gigg = chunk.loc[chunk["method"] == "GIGG_MMLE"]
        if gr.empty or gigg.empty:
            continue
        gr0 = gr.iloc[0]
        gi0 = gigg.iloc[0]
        gr_mse = float(gr0["mse_overall_mean"])
        gi_mse = float(gi0["mse_overall_mean"])
        ratio = float(gi_mse / gr_mse) if np.isfinite(gr_mse) and gr_mse > 0 and np.isfinite(gi_mse) else float("nan")
        rows.append(
            {
                "scenario": str(gr0["scenario"]),
                "rho_within": float(gr0["rho_within"]),
                "target_snr": float(gr0["target_snr"]),
                "gr_mse": gr_mse,
                "gigg_mse": gi_mse,
                "gigg_over_gr_mse_ratio": ratio,
                "gr_coverage": float(gr0["coverage_95_mean"]),
                "gigg_coverage": float(gi0["coverage_95_mean"]),
                "coverage_gap_gr_minus_gigg": float(gr0["coverage_95_mean"]) - float(gi0["coverage_95_mean"]),
                "gr_group_auroc": float(gr0["group_auroc_mean"]),
                "gigg_group_auroc": float(gi0["group_auroc_mean"]),
                "group_auroc_gap_gr_minus_gigg": float(gr0["group_auroc_mean"]) - float(gi0["group_auroc_mean"]),
                "gr_runtime": float(gr0["runtime_mean"]),
                "gigg_runtime": float(gi0["runtime_mean"]),
                "advantage_score": (
                    (math.log(ratio) if np.isfinite(ratio) and ratio > 0 else -10.0)
                    + 2.5 * (float(gr0["coverage_95_mean"]) - float(gi0["coverage_95_mean"]))
                    + 1.5 * (float(gr0["group_auroc_mean"]) - float(gi0["group_auroc_mean"]))
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("advantage_score", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search new synthetic regions where GR-RHS has structural advantages.")
    parser.add_argument("--save-dir", type=str, default="outputs/grrhs_advantage_region_search")
    parser.add_argument("--scenarios", type=str, default="hetero_mixed,paired_decoy,size_imbalance")
    parser.add_argument("--methods", type=str, default="GR_RHS,GIGG_MMLE,RHS,LASSO_CV")
    parser.add_argument("--scan-repeats", type=int, default=1)
    parser.add_argument("--confirm-repeats", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--n-train", type=int, default=180)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--rho-between", type=float, default=0.15)
    parser.add_argument("--rho-list", type=str, default="0.75,0.90")
    parser.add_argument("--snr-list", type=str, default="0.35,1.0")
    parser.add_argument("--seed", type=int, default=MASTER_SEED + 32000)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=180)
    parser.add_argument("--draws", type=int, default=180)
    parser.add_argument("--ess-threshold", type=float, default=60.0)
    parser.add_argument("--max-convergence-retries", type=int, default=1)
    parser.add_argument("--gigg-iter-floor", type=int, default=400)
    parser.add_argument("--gigg-iter-cap", type=int, default=400)
    parser.add_argument("--no-enforce-bayes-convergence", action="store_true")
    args = parser.parse_args()

    scenario_names = [s.strip() for s in str(args.scenarios).split(",") if s.strip()]
    scenario_map = {s.name: s for s in SCENARIOS}
    scenarios = [scenario_map[name] for name in scenario_names]
    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    rho_list = [float(x) for x in str(args.rho_list).split(",") if str(x).strip()]
    snr_list = [float(x) for x in str(args.snr_list).split(",") if str(x).strip()]
    sampler = SamplerConfig(
        chains=int(args.chains),
        warmup=int(args.warmup),
        post_warmup_draws=int(args.draws),
        ess_threshold=float(args.ess_threshold),
    )
    grrhs_kwargs = {
        "tau_target": "groups",
        "sampler_backend": "gibbs_staged",
        "progress_bar": False,
    }
    gigg_config = {
        "iter_mult": 1,
        "iter_floor": int(args.gigg_iter_floor),
        "iter_cap": int(args.gigg_iter_cap),
        "progress_bar": False,
        "allow_budget_retry": True,
        "extra_retry": 1,
        "retry_cap": 1,
    }

    out_dir = ensure_dir(Path(args.save_dir))

    scan_rows: list[dict[str, Any]] = []
    setting_id = 0
    for scenario in scenarios:
        for rho_within in rho_list:
            for target_snr in snr_list:
                for rep in range(int(args.scan_repeats)):
                    setting_seed = int(args.seed) + 10000 * setting_id + rep
                    scan_rows.extend(
                        _run_one_setting(
                            scenario=scenario,
                            rho_within=float(rho_within),
                            rho_between=float(args.rho_between),
                            target_snr=float(target_snr),
                            n=int(args.n_train),
                            n_test=int(args.n_test),
                            replicate_id=int(rep),
                            seed=int(setting_seed),
                            methods=methods,
                            sampler=sampler,
                            grrhs_kwargs=grrhs_kwargs,
                            gigg_config=gigg_config,
                            enforce_bayes_convergence=not bool(args.no_enforce_bayes_convergence),
                            max_convergence_retries=int(args.max_convergence_retries),
                        )
                    )
                setting_id += 1

    raw_scan = pd.DataFrame(scan_rows)
    scan_summary = _aggregate(raw_scan)
    top_regions = _compare_gr_vs_gigg(scan_summary)

    raw_scan.to_csv(out_dir / "scan_raw.csv", index=False)
    scan_summary.to_csv(out_dir / "scan_summary.csv", index=False)
    top_regions.to_csv(out_dir / "top_regions_scan.csv", index=False)

    confirm_rows: list[dict[str, Any]] = []
    if not top_regions.empty and int(args.confirm_repeats) > 0:
        top_take = min(int(args.top_k), int(top_regions.shape[0]))
        top_candidates = top_regions.head(top_take).to_dict(orient="records")
        for idx, rec in enumerate(top_candidates):
            scenario = scenario_map[str(rec["scenario"])]
            for rep in range(int(args.confirm_repeats)):
                seed = int(args.seed) + 500000 + 10000 * idx + rep
                confirm_rows.extend(
                    _run_one_setting(
                        scenario=scenario,
                        rho_within=float(rec["rho_within"]),
                        rho_between=float(args.rho_between),
                        target_snr=float(rec["target_snr"]),
                        n=int(args.n_train),
                        n_test=int(args.n_test),
                        replicate_id=int(rep),
                        seed=int(seed),
                        methods=methods,
                        sampler=sampler,
                        grrhs_kwargs=grrhs_kwargs,
                        gigg_config=gigg_config,
                        enforce_bayes_convergence=not bool(args.no_enforce_bayes_convergence),
                        max_convergence_retries=int(args.max_convergence_retries),
                    )
                )

    raw_confirm = pd.DataFrame(confirm_rows)
    if not raw_confirm.empty:
        confirm_summary = _aggregate(raw_confirm)
        confirm_regions = _compare_gr_vs_gigg(confirm_summary)
        raw_confirm.to_csv(out_dir / "confirm_raw.csv", index=False)
        confirm_summary.to_csv(out_dir / "confirm_summary.csv", index=False)
        confirm_regions.to_csv(out_dir / "top_regions_confirm.csv", index=False)
    else:
        confirm_summary = pd.DataFrame()
        confirm_regions = pd.DataFrame()

    meta = {
        "seed": int(args.seed),
        "methods": methods,
        "scan_repeats": int(args.scan_repeats),
        "confirm_repeats": int(args.confirm_repeats),
        "top_k": int(args.top_k),
        "n_train": int(args.n_train),
        "n_test": int(args.n_test),
        "rho_between": float(args.rho_between),
        "rho_list": rho_list,
        "snr_list": snr_list,
        "scenarios": [
            {
                "name": s.name,
                "family": s.family,
                "description": s.description,
                "group_sizes": list(s.group_sizes),
                "decoy_pairs": [list(x) for x in s.decoy_pairs],
            }
            for s in scenarios
        ],
        "sampler": {
            "chains": int(args.chains),
            "warmup": int(args.warmup),
            "draws": int(args.draws),
            "ess_threshold": float(args.ess_threshold),
        },
        "grrhs_kwargs": grrhs_kwargs,
        "gigg_config": gigg_config,
        "enforce_bayes_convergence": not bool(args.no_enforce_bayes_convergence),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Top scan regions (GR-RHS vs GIGG_MMLE):")
    if top_regions.empty:
        print("  none")
    else:
        print(top_regions.head(int(args.top_k)).to_string(index=False))
    if not confirm_regions.empty:
        print("\nConfirmed regions:")
        print(confirm_regions.head(int(args.top_k)).to_string(index=False))
    print(f"\nArtifacts saved in: {out_dir}")


if __name__ == "__main__":
    main()
