from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation_project.src.experiments.exp3 import _exp3_worker
from simulation_project.src.utils import MASTER_SEED, SamplerConfig, ensure_dir


DEFAULT_SETTINGS: list[dict[str, Any]] = [
    {"env_id": "RW070_RB020_SNR10_N200", "rho_within": 0.70, "rho_between": 0.20, "target_snr": 1.0, "n_train": 200},
    {"env_id": "RW070_RB020_SNR10_N300", "rho_within": 0.70, "rho_between": 0.20, "target_snr": 1.0, "n_train": 300},
    {"env_id": "RW075_RB020_SNR10_N300", "rho_within": 0.75, "rho_between": 0.20, "target_snr": 1.0, "n_train": 300},
    {"env_id": "RW080_RB020_SNR10_N200", "rho_within": 0.80, "rho_between": 0.20, "target_snr": 1.0, "n_train": 200},
    {"env_id": "RW080_RB020_SNR10_N300", "rho_within": 0.80, "rho_between": 0.20, "target_snr": 1.0, "n_train": 300},
    {"env_id": "RW080_RB020_SNR10_N400", "rho_within": 0.80, "rho_between": 0.20, "target_snr": 1.0, "n_train": 400},
    {"env_id": "RW085_RB020_SNR10_N300", "rho_within": 0.85, "rho_between": 0.20, "target_snr": 1.0, "n_train": 300},
    {"env_id": "RW090_RB020_SNR10_N300", "rho_within": 0.90, "rho_between": 0.20, "target_snr": 1.0, "n_train": 300},
    {"env_id": "RW090_RB020_SNR10_N400", "rho_within": 0.90, "rho_between": 0.20, "target_snr": 1.0, "n_train": 400},
    {"env_id": "RW080_RB010_SNR07_N300", "rho_within": 0.80, "rho_between": 0.10, "target_snr": 0.7, "n_train": 300},
    {"env_id": "RW085_RB015_SNR07_N300", "rho_within": 0.85, "rho_between": 0.15, "target_snr": 0.7, "n_train": 300},
    {"env_id": "RW090_RB015_SNR07_N400", "rho_within": 0.90, "rho_between": 0.15, "target_snr": 0.7, "n_train": 400},
]

METHODS = ["GR_RHS", "RHS", "GIGG_MMLE", "GHS_plus", "OLS", "LASSO_CV"]
GROUP_CFG = {
    "name": "G10x5",
    "group_sizes": [10, 10, 10, 10, 10],
    "active_groups": [0, 1],
    "allowed_signals": ["within_group_mixed"],
}


def _task_for_setting(
    setting: dict[str, Any],
    *,
    replicate_id: int,
    seed: int,
    sampler: SamplerConfig,
    max_convergence_retries: int,
    method_jobs: int,
) -> dict[str, Any]:
    return {
        "setting_id": int(setting["setting_id"]),
        "signal": "within_group_mixed",
        "group_cfg": dict(GROUP_CFG),
        "setting_block": "grrhs_sixway_region_scan",
        "env_id": str(setting["env_id"]),
        "design_type": "correlated",
        "rho_within": float(setting["rho_within"]),
        "rho_between": float(setting["rho_between"]),
        "target_snr": float(setting["target_snr"]),
        "boundary_xi_ratio": 1.2,
        "replicate_id": int(replicate_id),
        "seed_base": int(seed),
        "n_train": int(setting["n_train"]),
        "n_test": 100,
        "sampler": sampler,
        "methods": list(METHODS),
        "gigg_config": {
            "progress_bar": False,
        },
        "gigg_mode": "paper_ref",
        "bayes_min_chains": 2,
        "method_jobs": int(method_jobs),
        "enforce_bayes_convergence": True,
        "max_convergence_retries": int(max_convergence_retries),
        "grrhs_kwargs": {
            "tau_target": "groups",
            "sampler_backend": "gibbs_staged",
            "progress_bar": False,
        },
        "log_path": "",
    }


def _summarize(raw: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "env_id",
        "rho_within",
        "rho_between",
        "target_snr",
        "n_train",
        "method",
    ]
    out = (
        raw.groupby(keys, as_index=False)
        .agg(
            n_runs=("replicate_id", "count"),
            n_ok=("status", lambda s: int((pd.Series(s).astype(str) == "ok").sum())),
            n_converged=("converged", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
            mse_overall=("mse_overall", "mean"),
            mse_signal=("mse_signal", "mean"),
            mse_null=("mse_null", "mean"),
            coverage_95=("coverage_95", "mean"),
            avg_ci_length=("avg_ci_length", "mean"),
            lpd_test=("lpd_test", "mean"),
            runtime_mean=("runtime_seconds", "mean"),
        )
        .sort_values(["env_id", "mse_overall", "mse_signal"], ascending=[True, True, True])
    )
    return out


def _dominance_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_keys = ["env_id", "rho_within", "rho_between", "target_snr", "n_train"]
    for _, sub in summary.groupby(base_keys, as_index=False):
        by_method = {str(r["method"]): r for _, r in sub.iterrows()}
        if any(m not in by_method for m in METHODS):
            continue
        gr = by_method["GR_RHS"]
        competitors = [m for m in METHODS if m != "GR_RHS"]
        mse_overall_win = all(float(gr["mse_overall"]) < float(by_method[m]["mse_overall"]) for m in competitors)
        mse_signal_win = all(float(gr["mse_signal"]) < float(by_method[m]["mse_signal"]) for m in competitors)
        coverage_ok = np.isfinite(float(gr["coverage_95"])) and float(gr["coverage_95"]) >= 0.95
        fully_paired = int(min(int(by_method[m]["n_converged"]) for m in METHODS))
        best_overall_method = str(sub.sort_values("mse_overall").iloc[0]["method"])
        best_signal_method = str(sub.sort_values("mse_signal").iloc[0]["method"])
        row = {
            **{k: gr[k] for k in base_keys},
            "n_common_min_converged": fully_paired,
            "gr_mse_overall": float(gr["mse_overall"]),
            "gr_mse_signal": float(gr["mse_signal"]),
            "gr_coverage_95": float(gr["coverage_95"]),
            "best_overall_method": best_overall_method,
            "best_signal_method": best_signal_method,
            "gr_beats_all5_overall": bool(mse_overall_win),
            "gr_beats_all5_signal": bool(mse_signal_win),
            "gr_coverage_ge_095": bool(coverage_ok),
            "stable_sixway_dominance": bool(mse_overall_win and mse_signal_win and coverage_ok and fully_paired > 0),
        }
        for method in competitors:
            row[f"delta_overall_vs_{method}"] = float(by_method[method]["mse_overall"]) - float(gr["mse_overall"])
            row[f"delta_signal_vs_{method}"] = float(by_method[method]["mse_signal"]) - float(gr["mse_signal"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["stable_sixway_dominance", "gr_mse_overall"],
        ascending=[False, True],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan six-way GR-RHS dominance points using Exp3 within_group_mixed.")
    parser.add_argument("--save-dir", default="outputs/grrhs_sixway_region_scan")
    parser.add_argument("--settings-json", default="")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=MASTER_SEED + 52000)
    parser.add_argument("--max-convergence-retries", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--method-jobs", type=int, default=2)
    args = parser.parse_args()

    out_dir = ensure_dir(Path(args.save_dir))
    sampler = SamplerConfig(
        chains=2,
        warmup=250,
        post_warmup_draws=250,
        adapt_delta=0.97,
        max_treedepth=12,
        ess_threshold=80.0,
    )

    settings_src = DEFAULT_SETTINGS
    if str(args.settings_json).strip():
        settings_src = json.loads(Path(args.settings_json).read_text(encoding="utf-8"))

    settings = []
    for idx, setting in enumerate(settings_src, start=1):
        item = dict(setting)
        item["setting_id"] = idx
        settings.append(item)

    tasks: list[dict[str, Any]] = []
    for setting in settings:
        for rep in range(1, int(args.repeats) + 1):
            tasks.append(
                _task_for_setting(
                    setting,
                    replicate_id=rep,
                    seed=int(args.seed),
                    sampler=sampler,
                    max_convergence_retries=int(args.max_convergence_retries),
                    method_jobs=int(args.method_jobs),
                )
            )

    rows: list[dict[str, Any]] = []
    workers = max(1, int(args.jobs))
    if workers == 1:
        for task in tasks:
            task_rows = _exp3_worker(task)
            for row in task_rows:
                row["n_train"] = int(task["n_train"])
            rows.extend(task_rows)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_exp3_worker, task): task for task in tasks}
            for fut in as_completed(futures):
                task = futures[fut]
                task_rows = fut.result()
                for row in task_rows:
                    row["n_train"] = int(task["n_train"])
                rows.extend(task_rows)

    raw = pd.DataFrame(rows)
    summary = _summarize(raw)
    dominance = _dominance_table(summary)

    raw.to_csv(out_dir / "raw_results.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    dominance.to_csv(out_dir / "dominance_summary.csv", index=False)
    (out_dir / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    print(f"saved raw -> {out_dir / 'raw_results.csv'}")
    print(f"saved summary -> {out_dir / 'summary.csv'}")
    print(f"saved dominance -> {out_dir / 'dominance_summary.csv'}")
    if not dominance.empty:
        cols = [
            "env_id",
            "rho_within",
            "rho_between",
            "target_snr",
            "n_train",
            "n_common_min_converged",
            "best_overall_method",
            "best_signal_method",
            "stable_sixway_dominance",
        ]
        print(dominance.loc[:, cols].to_string(index=False))


if __name__ == "__main__":
    main()
