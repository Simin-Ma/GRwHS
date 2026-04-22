# Simulation Project Repository

This repository has been fully consolidated to a **single active system**:

- `simulation_project/`

The old multi-stack workflow (`grrhs`, `configs/sweeps/sim_s*`, `scene*`) has been removed.

## Run

```bash
python -m simulation_project.src.run_experiment --help
python scripts/run_simulation.py --help
```

Windows parallel safety:

- On Windows, process-pool parallelism is disabled by default to avoid `spawn` failures in interactive launch contexts.
- You can explicitly re-enable it when running from a spawn-safe script entrypoint:
  `SIM_ALLOW_WINDOWS_PROCESS_POOL=1`.

Detailed command guide:

- `docs/simulation_cli_guide.md`

Run all experiments:

```bash
python -m simulation_project.src.run_experiment --experiment all --save-dir simulation_project --n-jobs 2
```

Run analysis only:

```bash
python -m simulation_project.src.run_experiment --experiment analysis --save-dir simulation_project
```

## Active Experiments

Current code exposes 7 runnable experiment targets:

1. `exp1` kappa profile regimes
2. `exp2` group separation
3. `exp3` linear benchmark (aggregated view)
4. `exp3a` main benchmark
5. `exp3b` boundary stress benchmark
6. `exp4` tau variant ablation
7. `exp5` beta-prior sensitivity

## Structure

- `simulation_project/`: active pipeline
- `simulation_project/src/experiments/`: experiment runners, orchestration, runtime/fitting/evaluation/reporting helpers, schemas
- `simulation_project/src/cli/`: CLI entrypoints
- `simulation_project/src/core/`: internal model/inference runtime used by the pipeline
- `scripts/run_simulation.py`: launcher wrapper
- `docs/`: updated protocol and planning docs for the unified pipeline

## Notes

- No legacy sweep/scene entry points remain.
- No `grrhs/` package remains in the active tree.
