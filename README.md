# GR-RHS Simulation Project

Simulation-first grouped shrinkage experimentation toolkit for comparing GR-RHS against baseline Bayesian and frequentist methods.

## Setup

```bash
pip install -e ".[dev]"
```

## Active System

All experiments run through the single full pipeline in `simulation_project/`.

Quick start:

```bash
python -m simulation_project.src.run_experiment --help
```

Run all experiments:

```bash
python -m simulation_project.src.run_experiment --experiment all --n-jobs 2 --method-jobs 2 --all-parallel-jobs 2
```

Default output layout is session-based under:

`outputs/simulation_project/sessions/<timestamp>_cli_<experiment>/`

Ordinary experiment outputs are local artifacts only and should not be committed.

## Documentation

| Document | Purpose |
|---|---|
| `simulation_project/README.md` | Full CLI reference and typical commands |
| `docs/simulation_cli_guide.md` | Per-experiment commands and output layout |
| `docs/fair_benchmark_protocol.md` | Fairness rules for benchmark comparisons |
| `docs/bayesian_four_models_priors_posteriors_sampling_convergence.md` | Bayesian backend module locations |

## Method Notes

`GHS_plus` refers to the Xu et al. (2016) HBGHS baseline with paper-aligned Gaussian Gibbs defaults.

## Active Experiments

| ID | Name |
|---|---|
| `1` | kappa profile regimes |
| `2` | group separation |
| `3a` | main benchmark (concentrated + distributed) |
| `3b` | boundary stress benchmark |
| `3c` | optional high-dimensional paper random-coefficient stress run |
| `3d` | optional legacy boundary stress run |
| `4` | tau variant ablation |
| `5` | beta-prior sensitivity |

`Exp3a` and `Exp3c` are the paper-aligned fixed-coefficient and random-coefficient paths. `Exp3d` is kept only as a legacy boundary-stress line; its historical path name still contains `within_group_mixed`, but that is no longer the active signal definition.
