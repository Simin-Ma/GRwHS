from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation_project.src.run_experiment import run_exp3_linear_benchmark


def main() -> None:
    save_dir = f"ab_runs/exp3_gigg_mmle_core30_r1_tuned4_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out = run_exp3_linear_benchmark(
        n_jobs=4,
        seed=20260420,
        repeats=1,
        save_dir=save_dir,
        profile="laptop",
        methods=["GIGG_MMLE"],
        sampler_backend="nuts",
    )
    print("SAVE_DIR=", save_dir)
    for k, v in out.items():
        print(f"{k}={v}")


if __name__ == "__main__":
    main()
