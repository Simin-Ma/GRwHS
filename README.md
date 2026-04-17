# Simulation Project Repository

This repository has been fully consolidated to a **single active system**:

- `simulation_project/`

The old multi-stack workflow (`grrhs`, `configs/sweeps/sim_s*`, `scene*`) has been removed.

## Run

```bash
python -m simulation_project.src.run_experiment --help
python scripts/run_simulation.py --help
```

Run all experiments:

```bash
python -m simulation_project.src.run_experiment --experiment all --save-dir simulation_project --n-jobs 2
```

## Active Experiments

Current code exposes 5 experiments:

1. `exp1` kappa profile regimes
2. `exp2` group separation
3. `exp3` linear benchmark
4. `exp4` tau variant ablation
5. `exp5` beta-prior sensitivity

## Structure

- `simulation_project/`: active pipeline
- `simulation_project/src/core/`: internal model/inference runtime used by the pipeline
- `scripts/run_simulation.py`: launcher wrapper
- `docs/`: updated protocol and planning docs for the unified pipeline

## Notes

- No legacy sweep/scene entry points remain.
- No `grrhs/` package remains in the active tree.
