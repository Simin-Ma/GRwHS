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

Windows parallel safety:

- Process-pool parallelism is disabled by default on Windows to prevent interactive `spawn` failures.
- To force-enable process pools from a spawn-safe script entrypoint, set:
  `SIM_ALLOW_WINDOWS_PROCESS_POOL=1`.

Sweep entrypoint:

```bash
python -m simulation_project.src.run_sweep --list
python scripts/run_sweep.py --list
```

Unified Exp1-Exp5 sweep:

```bash
python -m simulation_project.src.run_sweep --sweep exp1_to_exp5
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
python -m simulation_project.src.run_experiment --experiment 3 --save-dir simulation_project --profile laptop --repeats 5 --n-jobs 2 --max-convergence-retries 1 --sampler nuts
```

Run analysis only:

```bash
python -m simulation_project.src.run_experiment --experiment analysis --save-dir simulation_project
```

Laptop profile:

```bash
python -m simulation_project.src.run_experiment --experiment all --save-dir simulation_project --profile laptop --n-jobs 2
```

Default sampler rule:

- For Exp2-Exp5, Bayesian minimum chains are profile-dependent by default:
  - `profile=laptop`: `2`
  - `profile=full`: `4`

## Internal Runtime

Simulation methods now use in-tree runtime modules under:

- `simulation_project/src/core/models/`
- `simulation_project/src/core/inference/`
- `simulation_project/src/core/diagnostics/`
- `simulation_project/src/core/utils/`

No external `grrhs` package directory is required.
