# GR-RHS Validation Experiment Framework

This document defines an experiment framework for validating the scientific claim of `GR-RHS` relative to `RHS`, using the modules that already exist in this repository whenever possible.

The goal is not to argue that `GR-RHS` must uniformly dominate `RHS` on a single scalar metric such as overall MSE. Instead, the goal is to show that `GR-RHS` changes the **unit of regularization** from the coefficient level to a **group-aware, two-level shrinkage structure**, and that this matters when the data have meaningful group structure, high within-group correlation, and heterogeneous signal allocation across groups.

## Core Scientific Claim

`RHS` solves the problem that large coefficients under the original horseshoe should still receive finite slab regularization.

`GR-RHS` addresses a different remaining limitation: `RHS` is still group-blind. In grouped designs, `GR-RHS` introduces a group-level gate `kappa_g`, while retaining within-group local adaptation through coefficient-level scales. As a result, the main expected advantages of `GR-RHS` are:

- better signal-group vs null-group separation
- more appropriate group-level complexity control
- more structure-matched posterior shrinkage under high within-group correlation
- improved posterior stability in difficult grouped designs

## Design Principles

The experiment suite should answer the following questions in order:

1. Does `GR-RHS` actually learn group-level shrinkage?
2. Does group-level shrinkage become more valuable as within-group correlation increases?
3. Are the gains truly caused by the `group -> coefficient` hierarchy rather than by generic tuning?
4. Is group-level complexity calibration a meaningful modeling device?
5. Under matched coefficient sparsity but different group sparsity, does `GR-RHS` behave differently from `RHS` in the intended way?

This leads to a grouped validation suite with three implemented `group_aware_v2` runners and one broader legacy correlation/stress family in `Exp3`.

## Existing Project Modules To Reuse

The current repository already contains most of the needed infrastructure.

### Experiment runners

- `simulation_project/src/experiments/exp2.py`
- `simulation_project/src/experiments/exp3.py`
- `simulation_project/src/experiments/exp4.py`
- `simulation_project/src/experiments/exp5.py`

### DGP utilities

- `simulation_project/src/experiments/dgp/grouped_linear.py`
- `simulation_project/src/experiments/dgp/grouped_logistic.py`
- `simulation_project/src/experiments/dgp/normal_means.py`

### Method wrappers

- `simulation_project/src/experiments/methods/fit_gr_rhs.py`
- `simulation_project/src/experiments/methods/fit_rhs.py`
- `simulation_project/src/experiments/fitting.py`
- `simulation_project/src/experiments/method_registry.py`

### Evaluation and metrics

- `simulation_project/src/experiments/evaluation.py`
- `simulation_project/src/experiments/analysis/metrics.py`
- `simulation_project/src/experiments/analysis/plotting.py`

### Config and CLI docs

- `simulation_project/config/experiments.yaml`
- `docs/simulation_cli_guide.md`

## Main Validation Suite

## Experiment A: Group Separation

### Scientific purpose

This experiment is the cleanest direct test that `kappa_g` is doing real group-level work rather than being a redundant latent layer.

### Existing implementation

Use the existing `Exp2` implementation:

- `simulation_project/src/experiments/exp2.py`

### Existing DGP

Use:

- `generate_heterogeneity_dataset()` from `simulation_project/src/experiments/dgp/grouped_linear.py`

Default design already matches the intended use case well:

- `group_sizes = [10, 10, 10, 10, 10]`
- `rho_within = 0.8`
- `rho_between = 0.2`
- `n_train = 100`
- `n_test = 30`
- methods restricted to `GR_RHS` and `RHS`
- `GR_RHS` configured with `tau_target="groups"`

### Main estimands

- `group_auroc`
- `null_group_mse`
- `signal_group_mse`
- posterior mean `kappa_g`
- posterior `P(kappa_g > threshold)`
- `bridge_ratio_signal_mean`
- `bridge_ratio_null_mean`

### Expected claim

`GR-RHS` should show stronger group separation than `RHS`, especially at the level of posterior group activity, even when overall coefficient-level MSE differences are moderate.

### Execution

```bash
python -m simulation_project.src.run_experiment --experiment 2
```

## Experiment B: Correlation and Structural Stress Benchmark

### Scientific purpose

This experiment tests whether group-aware shrinkage becomes more useful as grouped structure becomes harder, especially when:

- within-group correlation is high
- signal is concentrated in a few coefficients inside a group
- signal is distributed densely within active groups
- the signal lies close to a boundary or threshold regime

### Existing implementation

Use the existing `Exp3` family:

- `simulation_project/src/experiments/exp3.py`

Recommended emphasis:

- `Exp3a` as the main structured benchmark
- `Exp3b` as the boundary/high-stress benchmark
- optionally `Exp3c` as the high-dimensional extension

### Existing DGP support

Already implemented in `exp3.py`:

- `concentrated`
- `distributed`
- `boundary`
- `random_coefficient`

and group settings such as:

- `G10x5`
- `CL`, `CS`
- paper-aligned fixed-coefficient settings like `C10H`, `D10H`, `C25`, `D25`, etc.

### Main estimands

- `group_auroc`
- `mse_null`
- `mse_signal`
- `mse_overall`
- `lpd_test`
- `group_norm_entropy`
- `bridge_ratio_*`
- convergence rate
- ESS / R-hat / divergence / runtime

### Expected claim

As within-group correlation rises and group structure matters more, `GR-RHS` should produce more appropriate group-level shrinkage patterns than `RHS`. This advantage may appear in mechanism and stability metrics even when global predictive differences are modest.

### Execution

```bash
python -m simulation_project.src.run_experiment --experiment 3a
python -m simulation_project.src.run_experiment --experiment 3b
python -m simulation_project.src.run_experiment --experiment 3c
```

## Experiment B2: Group-Aware Correlation Stress

### Scientific purpose

This experiment isolates the question that is most specific to the `group-aware validation` view:

- if the active-group structure is held fixed,
- and only within-group correlation is increased,
- does `GR-RHS` show a more structure-matched shrinkage response than `RHS`?

### Current implementation

This is implemented in the separate `group_aware_v2` suite as:

- `simulation_project/src/experiments/exp_ga_v2_correlation_stress.py`
- CLI alias: `ga_v2c`
- results path: `results/group_aware_v2/ga_v2_correlation_stress`

### Recommended estimands

- `group_auroc`
- `kappa_gap`
- `null_group_mse`
- `signal_group_mse`
- `mse_overall`
- runtime / convergence metrics

### Execution

```bash
python -m simulation_project.src.run_experiment --experiment ga_v2c
```

## Experiment C: Mechanism Ablation

### Scientific purpose

This experiment asks which part of `GR-RHS` is actually responsible for the observed gains.

The key point is to show that the advantage is not merely due to a better tuned prior scale, but to the specific hierarchy:

- group-level gate `kappa_g`
- coefficient-level local scales `lambda_j`
- optional group-level complexity calibration through `tau_target="groups"`

### Existing implementation

Use the existing `Exp4` implementation:

- `simulation_project/src/experiments/exp4.py`

### Existing variants already present or naturally supported

- calibrated `GR_RHS`
- misspecified `GR_RHS` (`fixed_10x`)
- `RHS_oracle`
- oracle `GR_RHS`

The project also already has naming support for useful ablations in `simulation_project/src/utils.py`:

- `GR_RHS_no_local_scales`
- `GR_RHS_shared_kappa`
- `GR_RHS_no_kappa`

If these variants are not yet fully surfaced in `Exp4`, they should be added there first rather than creating a separate experiment runner.

### Main estimands

- `kappa_gap`
- `tau_ratio_to_oracle`
- `mse_rel_rhs_oracle`
- `bridge_ratio_*`
- convergence / runtime metrics

### Expected claim

Removing or weakening the group-level gate should reduce the group-separation advantage. Removing local scales should reduce within-group adaptivity. Together, these results identify the actual mechanism behind `GR-RHS`.

### Execution

```bash
python -m simulation_project.src.run_experiment --experiment 4
```

## Experiment D: Prior and Complexity Calibration Sensitivity

### Scientific purpose

This experiment evaluates whether group-level complexity calibration is a meaningful modeling choice and whether the `GR-RHS` mechanism is robust across reasonable prior settings.

This should not be interpreted as “guessing the exact number of active groups.” Instead, it is a robustness experiment around the idea that group-level complexity is a legitimate prior unit.

### Existing implementation

Use the existing `Exp5` implementation:

- `simulation_project/src/experiments/exp5.py`

### Existing design

Current `Exp5` already provides:

- paired evaluation on the same DGP replicate
- prior grid over `(alpha_kappa, beta_kappa)`
- `tau_target="groups"`
- paired summaries and prior deltas

### Main estimands

- `group_auroc`
- `kappa_null_mean`
- `kappa_signal_mean`
- `kappa_null_prob_gt_0_1`
- `mse_null`
- `mse_signal`
- convergence rate

### Expected claim

Reasonable group-level priors should preserve the main group-separation mechanism. Some settings may trade off aggressiveness versus robustness, but the group-aware structure itself should remain beneficial.

### Execution

```bash
python -m simulation_project.src.run_experiment --experiment 5
```

## Implemented Extension: Complexity-Mismatch Experiment

### Scientific purpose

This experiment is now represented in the separate `group_aware_v2` suite as `GA-V2-B`, and it is the most direct way to validate the claim that `GR-RHS` changes the effective unit of complexity.

The design should hold the total number of active coefficients fixed while varying how those active coefficients are distributed across groups.

This distinguishes:

- coefficient sparsity
- group sparsity

### Why this matters

If two datasets have the same total number of nonzero coefficients but very different numbers of active groups, `RHS` mainly “sees” coefficient complexity, whereas `GR-RHS` should react to group complexity as well.

### Recommended design

Fixed:

- `p = 50`
- `group_sizes = [10, 10, 10, 10, 10]`
- `n = 100`
- `rho_within = 0.8`
- `rho_between = 0.2`
- total active coefficient count fixed, for example `p0 = 10`

Compare at least two DGPs:

- `few_groups`: active coefficients concentrated in 1 active group
- `many_groups`: active coefficients spread across many active groups while retaining at least one null group

Optional within each:

- `concentrated` within-group pattern
- `distributed` within-group pattern

### Recommended methods

- `GR_RHS`
- `RHS`
- `RHS_oracle`
- optional `GR_RHS_no_kappa`

### Recommended estimands

- `group_auroc`
- `kappa_gap`
- posterior `tau`
- active-group recovery proxy
- `mse_signal`
- `mse_null`
- convergence / runtime

### Current implementation

The current implementation uses:

- `simulation_project/src/experiments/exp_ga_v2_complexity_mismatch.py`
- CLI alias: `ga_v2b`
- results path: `results/group_aware_v2/ga_v2_complexity_mismatch`

It reuses existing infrastructure:

- DGP builders in `grouped_linear.py`
- fitting via `_fit_all_methods()` in `fitting.py`
- metrics from `evaluation.py` and `analysis/metrics.py`

## Primary Reporting Structure

To keep the scientific story aligned with the method claim, the paper/report should organize results into three result families rather than a single headline MSE table.

### 1. Mechanism table

Recommended fields:

- `group_auroc`
- `kappa_gap`
- `kappa_signal_mean`
- `kappa_null_mean`
- `bridge_ratio_signal_mean`
- `bridge_ratio_null_mean`

### 2. Estimation / prediction table

Recommended fields:

- `mse_null`
- `mse_signal`
- `mse_overall`
- `lpd_test`

### 3. Stability table

Recommended fields:

- convergence rate
- median runtime
- ESS
- R-hat
- divergence ratio

## Recommended Interpretation Rules

The experiments should be interpreted using the following logic.

### What counts as strong support for `GR-RHS`

- stronger signal-group vs null-group separation than `RHS`
- better alignment between posterior shrinkage pattern and known group structure
- improved behavior in high-correlation grouped designs
- improved robustness under weak identification or near-boundary settings
- ablation evidence showing that the gains disappear when group-level gating is removed

### What should not be required

The framework does not require `GR-RHS` to win every benchmark in overall MSE. Since the claimed contribution is structural, the decisive evidence should come from:

- group-level recovery
- shrinkage allocation
- posterior behavior
- stability diagnostics

## Execution Checklist

For a standard validation pass:

```bash
python -m simulation_project.src.run_experiment --experiment 2
python -m simulation_project.src.run_experiment --experiment 3a
python -m simulation_project.src.run_experiment --experiment 3b
python -m simulation_project.src.run_experiment --experiment 4
python -m simulation_project.src.run_experiment --experiment 5
python -m simulation_project.src.run_experiment --experiment analysis
```

For a broader full run:

```bash
python -m simulation_project.src.run_experiment --experiment all --n-jobs 2 --method-jobs 2 --all-parallel-jobs 2
```

See also:

- `docs/simulation_cli_guide.md`
- `simulation_project/config/experiments.yaml`

## Future Maintenance Notes

When new experiments are added, they should be evaluated against the same scientific standard:

- Does the experiment isolate a claim that is specific to `GR-RHS`?
- Does it test group-aware shrinkage rather than generic predictive performance only?
- Does it reuse the common fitting and evaluation stack?
- Does it report at least one mechanism metric, one estimation metric, and one stability metric?

If the answer is no, the experiment should not be treated as a main scientific validation experiment.
