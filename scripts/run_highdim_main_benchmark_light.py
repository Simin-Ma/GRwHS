from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation_project.src.experiments.exp3 import run_exp3_linear_benchmark


def main() -> None:
    out_dir = ROOT / "outputs" / "history" / "Simulation_highdimension" / "benchmark_main"
    produced = run_exp3_linear_benchmark(
        save_dir=str(out_dir),
        repeats=1,
        seed=20260428,
        n_jobs=1,
        method_jobs=1,
        skip_run_analysis=False,
        archive_artifacts=False,
        signal_types=["concentrated", "distributed", "boundary"],
        methods=["GR_RHS", "RHS", "GIGG_MMLE", "LASSO_CV"],
        enforce_bayes_convergence=True,
        max_convergence_retries=1,
        until_bayes_converged=False,
        n_train=200,
        n_test=200,
        grrhs_extra_kwargs={"tau_target": "groups", "sampler_backend": "collapsed_profile", "progress_bar": False},
        sampler_overrides={
            "chains": 4,
            "warmup": 250,
            "post_warmup_draws": 250,
            "adapt_delta": 0.90,
            "max_treedepth": 12,
            "strict_adapt_delta": 0.95,
            "strict_max_treedepth": 14,
            "ess_threshold": 200,
        },
        gigg_mode="paper_ref",
        result_dir_name="highdim_main_light_r1",
        exp_key="Simulation_highdimension_main_light_r1",
    )
    print(json.dumps(produced, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
