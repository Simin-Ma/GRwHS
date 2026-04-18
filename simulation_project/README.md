# simulation_project

`simulation_project` is the only active experimental framework in this repository.

## Entrypoint

```bash
python -m simulation_project.src.run_experiment --help
```

Or:

```bash
python scripts/run_simulation.py --help
```

Detailed CLI playbook:

- `docs/simulation_cli_guide.md`

Sweep entrypoint:

```bash
python -m simulation_project.src.run_sweep --list
python scripts/run_sweep.py --list
```

Default sweep config:

- `simulation_project/config/sweeps.yaml`

## Experiment IDs

1. `run_exp1_kappa_profile_regimes`
2. `run_exp2_group_separation`
3. `run_exp3_linear_benchmark`
4. `run_exp4_variant_ablation`
5. `run_exp5_prior_sensitivity`

## Typical Commands

Run all:

```bash
python -m simulation_project.src.run_experiment --experiment all --save-dir simulation_project --n-jobs 2
```

Run one experiment:

```bash
python -m simulation_project.src.run_experiment --experiment 3 --save-dir simulation_project --repeats 20 --n-jobs 2
```

Laptop profile:

```bash
python -m simulation_project.src.run_experiment --experiment all --save-dir simulation_project --profile laptop --n-jobs 2
```

## Internal Runtime

Simulation methods now use in-tree runtime modules under:

- `simulation_project/src/core/models/`
- `simulation_project/src/core/inference/`
- `simulation_project/src/core/diagnostics/`
- `simulation_project/src/core/utils/`

No external `grrhs` package directory is required.
