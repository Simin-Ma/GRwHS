import sys
from pathlib import Path
from multiprocessing import freeze_support

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation_project.src.run_experiment import run_exp4_variant_ablation


def main() -> None:
    paths = run_exp4_variant_ablation(
        save_dir="simulation_project",
        profile="full",
        repeats=20,
        p0_list=[5, 30],
        include_oracle=True,
        max_convergence_retries=0,
        sampler_backend="collapsed",
        n_jobs=6,
    )
    print(paths)


if __name__ == "__main__":
    freeze_support()
    main()
