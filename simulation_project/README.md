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
3. `run_exp3_linear_benchmark` (legacy combined entry)
4. `run_exp3a_main_benchmark` (concentrated + distributed)
5. `run_exp3b_boundary_stress` (boundary-only)
6. `run_exp4_variant_ablation`
7. `run_exp5_prior_sensitivity`

## Typical Commands

Run all:

```bash
python -m simulation_project.src.run_experiment --experiment all --n-jobs 2
```

Run one experiment:

```bash
python -m simulation_project.src.run_experiment --experiment 3a --profile laptop --repeats 5 --n-jobs 2 --max-convergence-retries 1 --sampler nuts
```

Run Exp3a / Exp3b separately:

```bash
python -m simulation_project.src.run_experiment --experiment 3a --profile laptop --repeats 5 --n-jobs 2 --max-convergence-retries 1 --sampler nuts
python -m simulation_project.src.run_experiment --experiment 3b --profile laptop --repeats 5 --n-jobs 2 --max-convergence-retries 1 --sampler nuts
```

Legacy combined entry is still available:

```bash
python -m simulation_project.src.run_experiment --experiment 3 --profile laptop --repeats 5 --n-jobs 2 --max-convergence-retries 1 --sampler nuts
```

Run analysis only:

```bash
python -m simulation_project.src.run_experiment --experiment analysis
```

Laptop profile:

```bash
python -m simulation_project.src.run_experiment --experiment all --profile laptop --n-jobs 2
```

## Unified Output Layout

By default, each CLI run gets an isolated session directory:

`outputs/simulation_project/sessions/<timestamp>_cli_<experiment>/`

Quick pointers:

- Latest run pointer: `outputs/simulation_project/latest_session.txt`
- Latest run metadata: `outputs/simulation_project/latest_session.json`
- Session index: `outputs/simulation_project/session_index.jsonl`

Custom paths still work with `--save-dir`, but relative paths are normalized under the workspace to keep outputs centralized.

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
- `simulation_project/src/experiments/` (Exp1-Exp5 + orchestration/runtime/fitting/evaluation/reporting/schemas)
- `simulation_project/src/experiments/methods/` (fitters)
- `simulation_project/src/experiments/dgp/` (DGP modules)
- `simulation_project/src/experiments/analysis/` (metrics/plots/reporting)
- `simulation_project/src/cli/` (CLI entrypoint)

No external `grrhs` package directory is required.


