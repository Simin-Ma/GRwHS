# Simulation CLI Guide

This guide matches the current single-default simulation protocol (no laptop/full split).

## 1. Entry Points

```bash
python -m simulation_project.src.run_experiment --help
python scripts/run_simulation.py --help
```

Common CLI args:

- `--experiment {all,1,2,3,3a,3b,3c,3d,4,5,analysis}`
- `--workspace simulation_project`
- `--save-dir <path>`
- `--seed <int>`
- `--repeats <int>`
- `--n-jobs <int>`
- `--method-jobs <int>`
- `--all-parallel-jobs <int>` (used when `--experiment all`)
- `--no-enforce-bayes-convergence`
- `--max-convergence-retries <int>`
- `--until-bayes-converged`
- `--exp3-gigg-mode {paper_ref,stable}`
- `--sampler {nuts,collapsed,gibbs}`

`--profile` is intentionally unsupported.
Default sampler backend is `nuts`.

## 2. Default Protocol

Default run:

```bash
python -m simulation_project.src.run_experiment --experiment all --n-jobs 2 --method-jobs 2 --all-parallel-jobs 2
```

Default experiment order:

1. `exp1`
2. `exp2`
3. `exp3a`
4. `exp3b`
5. `exp4`
6. `exp5`
7. `analysis`

Default repeats:

- `exp1=500`
- `exp2=100`
- `exp3a=100`
- `exp3b=100`
- `exp4=30`
- `exp5=20`

`exp3c=30` and `exp3d=100` remain available via explicit `--experiment 3c/3d` runs.

## 3. Scientific Credibility Rules

Main conclusions are evaluated under strict convergence:

- Exp2-Exp5 Bayesian rows must satisfy `converged=True && status=ok`.
- Analysis report includes `Strict Convergence Gate` and fails when any key block violates this.

Main summary tables use paired-converged-and-ok subsets where applicable:

- Exp2: `summary_paired.csv`, `paired_deltas.csv`
- Exp3a/3b/3c/3d: `summary_paired.csv`, `summary_paired_deltas.csv`
- Exp5: `summary_paired.csv`, `prior_pairwise_delta.csv`

Diagnostics side table is always exported:

- `results/diagnostics_runtime_table.csv`
- Columns: runtime median/p95, ESS median, Rhat p95, divergence mean, convergence rate.

## 4. Default Design by Experiment

### Exp2 (`group_separation`)

- `group_sizes=[10,10,10,10,10]` (aligned to Exp3 `G10x5`, `p=50`)
- `rho_ref=0.8`
- `xi_ratios=[0.0,1.0,2.0,5.0,10.0]`
- `n_train=100`, `n_test=30`
- `rho_within=0.8`, `rho_between=0.2`
- methods: `GR_RHS`, `RHS`

Run:

```bash
python -m simulation_project.src.run_experiment --experiment 2 --n-jobs 2 --method-jobs 2
```

### Exp3a (`main_benchmark`)

- signals: `concentrated`, `distributed`
- correlation axis: `rho_within=[0.8]`, `rho_between=0.2`, enforced `rw>rb`
- SNR axis: `[0.2,1.0,5.0]`
- methods: `GR_RHS,RHS,GIGG_MMLE,GHS_plus,OLS,LASSO_CV`

Run:

```bash
python -m simulation_project.src.run_experiment --experiment 3a --n-jobs 2 --method-jobs 2
```

### Exp3b (`boundary_stress`)

- signal: `boundary`
- same correlation/SNR axes as Exp3a
- boundary `xi/xi_crit` grid via `boundary_xi_ratio_list` (default boundary stress grid)
- same default methods as Exp3a

Run:

```bash
python -m simulation_project.src.run_experiment --experiment 3b --n-jobs 2 --method-jobs 2
```

### Exp3c (`highdim_stress`)

- `n_train=200`, `n_test=100`, `p=500`, `group_sizes=[50]*10`
- signals: `concentrated`, `distributed`
- correlation axis: `rho_within=[0.8]`, `rho_between=0.2`, enforced `rw>rb`
- SNR axis: `[0.2,1.0,5.0]`
- same default methods as Exp3a

Run:

```bash
python -m simulation_project.src.run_experiment --experiment 3c --n-jobs 2 --method-jobs 2
```

### Exp3d (`within_group_mixed`)

- signal: `boundary`
- default group configs: `G10x5`, `CL`, `CS`
- correlation axis: `rho_within=[0.8]`, `rho_between=0.2`, enforced `rw>rb`
- SNR axis: `[0.2,1.0,5.0]`
- same default methods as Exp3a

Run:

```bash
python -m simulation_project.src.run_experiment --experiment 3d --n-jobs 2 --method-jobs 2
```

### Exp4 (`variant_ablation`)

- `p0_list=[5,15,30]`
- `include_oracle=True`
- DGP default correlation: `rho_within=0.8`, `rho_between=0.2`
- sampler routing default: `p0=5 -> collapsed`; other `p0` use `--sampler` (default `nuts`)

Run:

```bash
python -m simulation_project.src.run_experiment --experiment 4 --n-jobs 2 --method-jobs 2
```

### Exp5 (`prior_sensitivity`)

- paired prior comparisons on the same replicate
- main delta file: `prior_pairwise_delta.csv`
- default contrast baseline: prior `(0.5,1.0)`
- default prior grid starts from full sensitivity, then runs a lightweight screening stage before the full paired run
- default convergence retry budget is `max_convergence_retries=5` for Exp5
- retry attempts continue from previous sampler state (no cold restart)
- `summary_partial.csv` is exported even when strict paired summary is empty

Run:

```bash
python -m simulation_project.src.run_experiment --experiment 5 --n-jobs 2 --method-jobs 2
```

## 5. Analysis Only

```bash
python -m simulation_project.src.run_experiment --experiment analysis
```

## 6. Sweep Runner

```bash
python -m simulation_project.src.run_sweep --list
python -m simulation_project.src.run_sweep --sweep exp1_to_exp5
python -m simulation_project.src.run_sweep --sweep exp1_to_exp5 --set n_jobs=2 --set method_jobs=2 --set all_parallel_jobs=2 --set max_convergence_retries=2
```

## 7. Optional Quick Utility

`scripts/run_laptop_best_2h.py` remains available as a quick smoke/main acceptance helper.
It is not the default protocol and does not define the paper-level reference configuration.
