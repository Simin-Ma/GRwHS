from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation_project.src.run_experiment import run_exp3_linear_benchmark

if __name__ == '__main__':
    save_dir = f"ab_runs/exp3_core30_grrhs_r1_explore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out = run_exp3_linear_benchmark(
        n_jobs=2,
        seed=20260420,
        repeats=1,
        save_dir=save_dir,
        profile='laptop',
        methods=['GR_RHS'],
        bayes_min_chains=2,
        max_convergence_retries=0,
        sampler_backend='collapsed',
        grrhs_extra_kwargs={'progress_bar': True},
    )
    print('SAVE_DIR=', save_dir)
    for k, v in out.items():
        print(f"{k}={v}")
