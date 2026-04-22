# GR-RHS Simulation Project

Simulation-first grouped shrinkage experimentation toolkit for comparing GR-RHS against
baseline Bayesian and frequentist methods.

## Setup

```bash
pip install -e ".[dev]"
```

## Active System

All experiments run through the single active pipeline: `simulation_project/`.

Quick start:

```bash
python -m simulation_project.src.run_experiment --help
```

Run all experiments:

```bash
python -m simulation_project.src.run_experiment --experiment all --n-jobs 2
```

Default output layout is now session-based and centralized under:

`outputs/simulation_project/sessions/<timestamp>_cli_<experiment>/`

## Documentation

| Document | Purpose |
|---|---|
| `simulation_project/README.md` | Full CLI reference and typical commands |
| `docs/simulation_cli_guide.md` | Per-experiment command presets and output layout |
| `docs/fair_benchmark_protocol.md` | Fairness rules for benchmark comparisons |
| `docs/bayesian_four_models_priors_posteriors_sampling_convergence.md` | Bayesian backend module locations |

## Active Experiments

| ID | Name |
|---|---|
| `1` | kappa profile regimes |
| `2` | group separation |
| `3a` | main benchmark (concentrated + distributed) |
| `3b` | boundary stress benchmark |
| `4` | tau variant ablation |
| `5` | beta-prior sensitivity |
