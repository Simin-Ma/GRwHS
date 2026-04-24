# Simulation CLI Guide

This guide matches the single full simulation protocol used for paper analysis.

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
- `--all-parallel-jobs <int>`
- `--skip-analysis`
- `--no-archive-artifacts`
- `--no-enforce-bayes-convergence`
- `--max-convergence-retries <int>`
- `--until-bayes-converged`
- `--exp3-gigg-mode {paper_ref}`

Retry semantics:
- If `--max-convergence-retries` is set to a nonnegative integer, that exact retry budget is used.
- If `--until-bayes-converged` is enabled and `--max-convergence-retries` is omitted, the runtime uses a negative sentinel that activates capped "retry until converged" mode.
- Shared hard cap for until-converged mode is 12 retries (13 total attempts); `Exp5` keeps a smaller practical cap of 5 retries.

`--profile` is intentionally unsupported.

## 2. Full Protocol

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
- `exp4=10`
- `exp5=20`

Optional explicit runs:

- `exp3c=30`
- `exp3d=100`

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

- `group_sizes=[10,10,10,10,10]`
- `rho_ref=0.8`
- `xi_ratios=[0.0,1.0,2.0,5.0,10.0]`
- `n_train=100`, `n_test=30`
- `rho_within=0.8`, `rho_between=0.2`
- methods: `GR_RHS`, `RHS`

Run:

```bash
python -m simulation_project.src.run_experiment --experiment 2
```

### Exp3a (`main_benchmark`)

- paper fixed-coefficient settings:
  `C10H,D10H,C10M,D10M,C5,D5,C25,D25,CL,DL,CS,DS`
- `p=50`; project-default `n_train` / `n_test` are unchanged
- `rho_between=0.2`; `rho_within` follows the paper label (`0.8` for `H`, `0.6` for `M`)
- explained-variance target fixed at `beta'Sigma beta / (beta'Sigma beta + sigma2) = 0.7`
- `concentrated` now means within-group sparse; `distributed` now means within-group dense
- methods: `GR_RHS,RHS,GIGG_MMLE,GHS_plus,OLS,LASSO_CV`

Run:

```bash
python -m simulation_project.src.run_experiment --experiment 3a
```

### Exp3b (`boundary_stress`)

- signal: `boundary`
- same correlation/SNR axes as Exp3a
- boundary `xi/xi_crit` grid via `boundary_xi_ratio_list`
- same default methods as Exp3a

Run:

```bash
python -m simulation_project.src.run_experiment --experiment 3b
```

### Exp3c (`highdim_stress`)

- `n_train=200`, `n_test=100`, `p=500`
- group structure: `50` groups of size `10`
- signal mechanism: paper random-coefficient design
- first group is active with even-probability concentrated/distributed assignment
- remaining groups are sampled as concentrated/distributed/null with probabilities `0.2/0.2/0.6`
- `rho_within=0.8`, `rho_between=0.2`
- explained-variance target fixed at `beta'Sigma beta / (beta'Sigma beta + sigma2) = 0.7`

Run:

```bash
python -m simulation_project.src.run_experiment --experiment 3c
```

### Exp3d (`within_group_mixed`)

- legacy boundary-focused stress variant
- `within_group_mixed` is a historical experiment key/path, not the current signal definition
- signal: `boundary`
- default group configs: `G10x5`, `CL`, `CS`
- correlation axis: `rho_within=[0.8]`, `rho_between=0.2`
- SNR axis: `[0.2,1.0,5.0]`
- this experiment is not part of the paper-aligned fixed/random coefficient redesign

Run:

```bash
python -m simulation_project.src.run_experiment --experiment 3d
```

### Exp4 (`variant_ablation`)

- `p0_list=[5,15,30]`
- `include_oracle=True`
- DGP default correlation: `rho_within=0.8`, `rho_between=0.2`

Run:

```bash
python -m simulation_project.src.run_experiment --experiment 4
```

### Exp5 (`prior_sensitivity`)

- paired prior comparisons on the same replicate
- main delta file: `prior_pairwise_delta.csv`
- default contrast baseline: prior `(0.5,1.0)`
- default prior grid starts from full sensitivity, then runs a lightweight screening stage before the full paired run
- default convergence retry budget is `max_convergence_retries=5` for Exp5
- retry attempts continue from previous sampler state
- `summary_partial.csv` is exported even when strict paired summary is empty

Run:

```bash
python -m simulation_project.src.run_experiment --experiment 5
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
