from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation_project.src.experiments.exp3 import run_exp3_linear_benchmark


def main() -> None:
    stable_path = ROOT / "outputs" / "grrhs_sixway_region_scan_combined" / "stable_sixway_dominance_points.csv"
    save_dir = ROOT / "outputs" / "grrhs_sixway_stable_r10"
    stable = pd.read_csv(stable_path)
    stable = stable.sort_values(["rho_within", "rho_between", "n_train", "env_id"]).reset_index(drop=True)

    env_points = []
    for _, row in stable.iterrows():
        env_points.append(
            {
                "env_id": str(row["env_id"]),
                "setting_block": "grrhs_sixway_stable_r10",
                "rho_within": float(row["rho_within"]),
                "rho_between": float(row["rho_between"]),
                "target_snr": float(row["target_snr"]),
                "signals": ["within_group_mixed"],
            }
        )

    group_configs = [
        {
            "name": "G10x5",
            "group_sizes": [10, 10, 10, 10, 10],
            "active_groups": [0, 1],
            "allowed_signals": ["within_group_mixed"],
        }
    ]

    produced = run_exp3_linear_benchmark(
        save_dir=str(save_dir),
        repeats=10,
        seed=20260425,
        n_jobs=2,
        method_jobs=2,
        skip_run_analysis=False,
        archive_artifacts=False,
        signal_types=["within_group_mixed"],
        env_points=env_points,
        group_configs=group_configs,
        methods=["GR_RHS", "RHS", "GIGG_MMLE", "GHS_plus", "OLS", "LASSO_CV"],
        enforce_bayes_convergence=True,
        max_convergence_retries=5,
        until_bayes_converged=False,
        n_train=100,
        n_test=100,
        grrhs_extra_kwargs={"tau_target": "groups", "sampler_backend": "gibbs_staged", "progress_bar": False},
        sampler_overrides={
            "chains": 2,
            "warmup": 250,
            "post_warmup_draws": 250,
            "adapt_delta": 0.97,
            "max_treedepth": 12,
            "ess_threshold": 80,
        },
        gigg_mode="paper_ref",
        result_dir_name="exp3_grrhs_sixway_stable_r10",
        exp_key="exp3_grrhs_sixway_stable_r10",
    )
    meta_path = save_dir / "stable_points_manifest.json"
    meta_path.write_text(
        json.dumps(
            {
                "stable_points_source": str(stable_path),
                "n_points": int(stable.shape[0]),
                "repeats": 10,
                "env_points": env_points,
                "produced": produced,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(produced, indent=2))


if __name__ == "__main__":
    main()
