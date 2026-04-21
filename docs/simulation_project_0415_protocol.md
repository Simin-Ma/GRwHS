# Simulation Project Protocol (Current)

This protocol applies to the unified pipeline in:

- `simulation_project/src/run_experiment.py`

## Scope

The active code path includes 5 experiments (`exp1`-`exp5`).

## Reproducibility

- master seed: `20260415`
- derived per task by experiment/setting/replicate indexing
- method comparisons are paired on shared dataset draws within replicate

## Compute Profiles

- `full`
- `laptop`

## Convergence Controls

- `--no-enforce-bayes-convergence`
- `--max-convergence-retries <k>`
- `--until-bayes-converged`
- default retry budget by profile when not specified: `full=2`, `laptop=1`
- convergence thresholds are fixed across retries (`R-hat` / `ESS` / divergence); retries only increase sampling budget

## Commands

```bash
python -m simulation_project.src.run_experiment --help
python -m simulation_project.src.run_experiment --experiment all --save-dir simulation_project --n-jobs 2
python -m simulation_project.src.run_experiment --experiment 3a --save-dir simulation_project --profile laptop --repeats 5 --n-jobs 2 --max-convergence-retries 1 --sampler nuts
python -m simulation_project.src.run_experiment --experiment 3b --save-dir simulation_project --profile laptop --repeats 5 --n-jobs 2 --max-convergence-retries 1 --sampler nuts
python -m simulation_project.src.run_experiment --experiment analysis --save-dir simulation_project
```

## Output Layout

- `simulation_project/results/`
- `simulation_project/figures/`
- `simulation_project/tables/`
- `simulation_project/logs/`

## Repository Status

Legacy `grrhs + sweep/scene` stacks are removed from the active repository structure.
