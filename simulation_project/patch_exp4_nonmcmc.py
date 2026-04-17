"""
Patch exp4 raw_results.csv: re-run only GIGG_MMLE and GHS_plus (fast Gibbs methods)
using the same seeds/DGP as the original exp4, then regenerate summary and figures.

Run from repo root:
    python simulation_project/patch_exp4_nonmcmc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from simulation_project.src.dgp_grouped_linear import build_linear_beta, generate_grouped_linear_dataset
from simulation_project.src.fit_gigg import fit_gigg_mmle
from simulation_project.src.fit_ghs_plus import fit_ghs_plus
from simulation_project.src.run_experiment import _evaluate_method_row
from simulation_project.src.utils import (
    MASTER_SEED,
    SamplerConfig,
    canonical_groups,
    experiment_seed,
)
from simulation_project.src.plotting import plot_exp4_mse_partition, plot_exp4_overall_mse

# ── Same settings as run_exp4_benchmark_linear ────────────────────────────────
SETTINGS = {
    "L0": {"group_sizes": [10, 10, 10, 10, 10], "rho_within": 0.0, "rho_between": 0.0,  "design_type": "orthonormal"},
    "L1": {"group_sizes": [10, 10, 10, 10, 10], "rho_within": 0.3, "rho_between": 0.10, "design_type": "correlated"},
    "L2": {"group_sizes": [10, 10, 10, 10, 10], "rho_within": 0.3, "rho_between": 0.10, "design_type": "correlated"},
    "L3": {"group_sizes": [10, 10, 10, 10, 10], "rho_within": 0.8, "rho_between": 0.10, "design_type": "correlated"},
    "L4": {"group_sizes": [10, 10, 10, 10, 10], "rho_within": 0.8, "rho_between": 0.10, "design_type": "correlated"},
    "L5": {"group_sizes": [30, 10, 5, 3, 2],    "rho_within": 0.3, "rho_between": 0.10, "design_type": "correlated"},
}

BASE = Path(__file__).parent
RAW_CSV   = BASE / "results" / "exp4_benchmark_linear" / "raw_results.csv"
FIG_DIR   = BASE / "figures"
TAB_DIR   = BASE / "tables"


def main() -> None:
    raw = pd.read_csv(RAW_CSV)
    design_meta_path = BASE / "results" / "exp4_benchmark_linear" / "exp4_design_meta.json"
    import json
    repeats = json.loads(design_meta_path.read_text())["repeats"]

    sampler = SamplerConfig()
    patch_rows: list[dict] = []

    for sid, (setting, spec) in enumerate(SETTINGS.items(), start=1):
        beta_shape = build_linear_beta(setting, spec["group_sizes"])
        groups = canonical_groups(spec["group_sizes"])

        for r in range(1, repeats + 1):
            s = experiment_seed(4, sid, r, master_seed=MASTER_SEED)
            ds = generate_grouped_linear_dataset(
                n=500,
                group_sizes=spec["group_sizes"],
                rho_within=spec["rho_within"],
                rho_between=spec["rho_between"],
                beta_shape=beta_shape,
                seed=s,
                target_snr=0.70,
                design_type=str(spec.get("design_type", "correlated")),
            )

            for method_name, fit_fn, seed_offset in [
                ("GIGG_MMLE", fit_gigg_mmle, 3),
                ("GHS_plus",  fit_ghs_plus,  4),
            ]:
                result = fit_fn(
                    ds["X"], ds["y"], groups,
                    task="gaussian",
                    seed=s + seed_offset,
                    sampler=sampler,
                    **({} if method_name == "GIGG_MMLE" else {
                        "p0": int(np.sum(np.abs(ds["beta0"]) > 0.0))
                    }),
                )
                metrics = _evaluate_method_row(result, ds["beta0"])
                patch_rows.append({
                    "setting": setting,
                    "replicate_id": r,
                    "method": method_name,
                    "status": result.status,
                    "converged": result.converged,
                    "runtime_seconds": result.runtime_seconds,
                    "rhat_max": result.rhat_max,
                    "bulk_ess_min": result.bulk_ess_min,
                    "divergence_ratio": result.divergence_ratio,
                    "error": result.error,
                    **metrics,
                })
            print(f"  {setting} rep {r:2d}", end="\r", flush=True)

    patch_df = pd.DataFrame(patch_rows)
    print(f"\nPatch rows: {len(patch_df)}")
    print("MSE null non-null counts:")
    print(patch_df.groupby("method")["mse_overall"].count())

    # Replace old GIGG/GHS rows in raw
    keep = raw[~raw["method"].isin(["GIGG_MMLE", "GHS_plus"])].copy()
    new_raw = pd.concat([keep, patch_df], ignore_index=True)
    # Sort to original ordering
    method_order = {"GR_RHS": 0, "RHS": 1, "GIGG_MMLE": 2, "GHS_plus": 3}
    setting_order = {s: i for i, s in enumerate(SETTINGS)}
    new_raw["_mo"] = new_raw["method"].map(method_order)
    new_raw["_so"] = new_raw["setting"].map(setting_order)
    new_raw = new_raw.sort_values(["replicate_id", "_so", "_mo"]).drop(columns=["_mo", "_so"])
    new_raw.to_csv(RAW_CSV, index=False)
    print(f"Saved updated raw CSV → {RAW_CSV}")

    # Recompute summary
    summary = (
        new_raw.groupby(["setting", "method"], as_index=False)
        .agg(
            mse_null=("mse_null", "mean"),
            mse_signal=("mse_signal", "mean"),
            mse_overall=("mse_overall", "mean"),
            avg_ci_length=("avg_ci_length", "mean"),
            coverage_95=("coverage_95", "mean"),
            n_effective=("converged", "sum"),
        )
    )
    summary_path = BASE / "results" / "exp4_benchmark_linear" / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary → {summary_path}")

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TAB_DIR / "table_benchmark_linear.csv", index=False)

    # Regenerate figures
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_exp4_overall_mse(summary,    out_path=FIG_DIR / "fig4_benchmark_overall_mse.png")
    plot_exp4_mse_partition(summary,  out_path=FIG_DIR / "fig4_benchmark_mse_partition.png")
    print("Figures regenerated.")


if __name__ == "__main__":
    main()
