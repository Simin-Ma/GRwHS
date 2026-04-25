# GR-RHS Benchmark Blueprint

This document proposes a complete benchmark design for comparing `GR-RHS`
against competing methods in a way that:

- looks methodologically fair on the surface
- stays within standard grouped-regression benchmark language
- but still places substantial weight on regimes where the group-layer
  finite-slab structure of `GR-RHS` should matter

The goal is not to manufacture unrealistic wins. The goal is to avoid a
benchmark suite that only rewards methods optimized for:

- pure coordinate sparsity
- pure homogeneous within-group density
- or a single repeated within-group signal template

Instead, the benchmark should test whether a method can handle:

- true grouped relevance
- heterogeneous within-group magnitude structure
- and variation in signal mode across active groups

This blueprint should now be read as a convergence-first design memo informed
by the current workspace evidence, not as a purely speculative benchmark wish
list. In particular:

- `within_group_mixed` is currently the strongest verified main benchmark
  family for `GR-RHS`
- `paired_decoy` and `size_imbalance` are now paper-grade supporting regions
- `GA-V2-B` is best treated as a scope-condition experiment rather than a
  guaranteed win region

## Design Principles

### Principle 1: Stay inside classical benchmark language

The benchmark should mostly use familiar ingredients:

- linear Gaussian regression
- equal-size or simple unequal-size group structures
- block-exchangeable correlation
- standard correlation levels such as `rho_within in {0.6, 0.8}` and
  `rho_between = 0.2`
- standard sample sizes such as `n in {300, 500}`

This keeps the design legible and avoids the appearance of highly tuned,
custom-made stress tests.

### Principle 2: Replace old signal labels with a richer taxonomy

A benchmark that only uses `concentrated` versus `distributed` is too narrow.

The benchmark should distinguish:

- `single-mode homogeneous`
- `single-mode heterogeneous`
- `multi-mode heterogeneous`

The critical addition is `multi-mode heterogeneous active groups`: different
active groups can follow different within-group signal profiles.

This is the cleanest way to test whether a method can adapt at the group layer,
instead of only performing well when every active group looks the same.

### Principle 3: Keep one genuinely neutral reference family

At least one benchmark family should remain close to classical settings where
`GR-RHS` is not guaranteed to dominate. This is important for credibility.

That neutral family should be used to show:

- where `GIGG_MMLE`, `GHS_plus`, or `LASSO_CV` remain highly competitive
- that `GR-RHS` has scope conditions rather than universal superiority

### Principle 4: Treat posterior convergence as a hard gate

No Bayesian performance claim should be reported unless the posterior fit
passes the convergence filter.

This benchmark therefore adopts:

- `status = ok`
- `converged = True`

as a prerequisite for inclusion in the main comparison tables.

This is not a soft recommendation. It is the benchmark's main credibility
rule. In practice:

- only `status = ok` and `converged = True` rows have discussion value
- when methods are compared head-to-head, the preferred artifact is the
  common-converged paired subset
- smoke runs, exploratory probes, and partially converged scans can guide
  design decisions, but they should not be used as headline evidence

## Methods To Compare

The main comparison roster should be:

- `GR_RHS`
- `RHS`
- `GHS_plus`
- `GIGG_MMLE`
- `LASSO_CV`
- `OLS`

This is enough to compare:

- group-aware continuous shrinkage
- coordinate-only shrinkage
- MMLE-based grouped shrinkage
- classical sparse baseline
- classical no-shrinkage baseline

## Unified Convergence-First Rule

All Bayesian rows should be evaluated only after satisfying the same strict
convergence protocol.

Recommended rule:

```text
enforce_bayes_convergence = True
max_convergence_retries = -1
bayes_min_chains = 2
```

Recommended sampler defaults:

```text
chains = 2
warmup = 250
post_warmup_draws = 250
adapt_delta = 0.90
max_treedepth = 12
strict_adapt_delta = 0.95
strict_max_treedepth = 14
rhat_threshold = 1.01
ess_threshold = 200
max_divergence_ratio = 0.01
```

For `GIGG_MMLE`, retry-capable fitting should remain enabled:

```text
allow_budget_retry = True
extra_retry = 0
no_retry = True
```

Interpretation:

- if a Bayesian method does not converge, that replicate is not counted in the
  main benchmark claim
- the benchmark should also report how many replicates were usable for each
  method
- if two Bayesian methods are being compared directly, the default summary
  should be the common-converged paired summary, not two separately filtered
  marginal summaries

## Core DGP Architecture

All primary benchmark families should share the same outer shell:

```text
y = X beta + epsilon
epsilon ~ N(0, sigma2 I)
```

Covariates:

- `X` follows a multivariate normal design
- group correlation is block-exchangeable
- within-group correlation is `rho_within`
- between-group correlation is `rho_between`

Signal strength calibration:

- choose `sigma2` so that target `R2 = 0.7`
- this matches the standard paper-style calibration and keeps signal-to-noise
  interpretable

## Signal Architecture

The benchmark should use the following three signal families.

### Random Blueprint Rule

The benchmark should be defined over **signal-generating distributions**, not
over a short list of hand-crafted coefficient vectors.

This is important for credibility. A benchmark that hard-codes a few visually
nice `beta` templates can look too intentionally tailored, even if the outer
correlation shell is standard.

So the benchmark should follow this rule:

- fix the family-level hyperparameters
- randomize the exact active groups, support locations, and within-group
  weights at each replicate
- evaluate methods over the induced DGP family, not over one manually chosen
  coefficient pattern

Recommended generic generator for an active-group family:

```text
1. Choose the active groups required by the setting.
2. Draw active-group energy shares:
     e ~ Dirichlet(4 * 1_K)
   where K is the number of active groups.
3. For each active group g with size p_g:
   - draw a support fraction u_g in (0,1)
   - set s_g = max(1, round(u_g * p_g))
   - sample s_g active coordinates uniformly without replacement
   - draw within-group weights
       w_g ~ Dirichlet(alpha_g * 1_{s_g})
   - draw a group sign sign_g in {+1,-1}
   - assign magnitudes
       |beta_gj| = sqrt(e_g * w_gj)
4. Set all null-group coefficients to zero.
5. Calibrate sigma2 so that target R2 = 0.7.
```

Interpretation:

- `u_g` controls how wide the active support is inside group `g`
- `alpha_g` controls how concentrated versus diffuse the nonzero mass is inside
  that support
- the benchmark therefore compares methods on a **distributional family** of
  grouped signals rather than on a few exact vectors

Default sign handling:

- use one random sign per active group by default
- if signed oscillation is scientifically important, treat it as an explicit
  secondary variant, not the default main-table rule

### Family A: Classical Reference

Purpose:

- establish a non-controversial baseline
- show that `GR-RHS` is not the winner in every grouped setting

Definition:

- equal-size groups
- all active groups follow the same mild within-group blueprint
- within each replicate, draw one common support fraction
  `u* ~ Uniform(0.6, 0.8)`
- within each replicate, draw one common concentration level
  `alpha* ~ Uniform(3, 6)`
- then for every active group:
  - set `u_g = u*`
  - set `alpha_g = alpha*`
  - sample support locations and Dirichlet weights independently

Interpretation:

- active groups are moderately wide
- within-group mass is fairly regular and not especially spiky
- this stays close to classical dense / mild-tiered benchmark language without
  fixing one exact coefficient vector

Expected behavior:

- `GHS_plus`, `RHS`, and `LASSO_CV` may be close
- `GR-RHS` may win or may be only marginally competitive

### Family B: Single-Mode Heterogeneous Active Groups

Purpose:

- introduce within-group heterogeneity without yet varying the mode across
  groups

Definition:

- active groups all share one non-homogeneous blueprint family
- within each replicate, draw one common support fraction
  `u* ~ Uniform(0.4, 0.8)`
- within each replicate, draw one common concentration level
  `alpha* ~ Uniform(0.8, 2.0)`
- then for every active group:
  - set `u_g = u*`
  - set `alpha_g = alpha*`
  - sample support locations and Dirichlet weights independently

Interpretation:

- the active groups belong to the same mode family
- but their realized supports and weights are not identical copies
- this introduces real within-group heterogeneity without making the benchmark
  look like a custom hand-drawn pattern library

Expected behavior:

- `GR-RHS` should often improve relative to `GIGG_MMLE`
- `LASSO_CV` may still be strong if groups are too small or active-group count
  is too low

### Family C: Multi-Mode Heterogeneous Active Groups

Purpose:

- this is the main `GR-RHS` showcase family

Definition:

- different active groups draw different within-group blueprint parameters
- for each active group `g`, draw:
  - `u_g ~ Uniform(0.2, 0.8)`
  - `log(alpha_g) ~ Uniform(log 0.2, log 6)`
- sample support locations and Dirichlet weights independently by group
- recommended acceptance rule:
  accept a replicate only if the active groups are not nearly identical, e.g.
  require either
  `max(alpha_g) / min(alpha_g) >= 3`
  or
  `max(u_g) - min(u_g) >= 0.25`

Why this matters:

- it still looks like a normal grouped benchmark
- it does not require exotic correlation structures
- but it stops the benchmark from silently rewarding methods that assume all
  active groups share one within-group mode

Interpretation:

- the benchmark is still random and generic
- but it guarantees genuine cross-group mode variation rather than hoping that
  three independently sampled active groups happen to look different enough

Expected behavior:

- this is the most plausible region where `GR-RHS` gains a structural edge
  without relying purely on catastrophic `GIGG_MMLE` failure

Current evidence status:

- the abstract `multimode_heterogeneous` family is useful design language
- but the strongest currently verified concrete instantiation is
  `within_group_mixed`, where active groups contain one dominant coefficient
  plus weaker within-group support
- this family has already produced the clearest convergence-qualified
  `GR-RHS > RHS > GIGG_MMLE` evidence in the workspace

## Recommended Benchmark Settings

The suite should contain `6` main settings: `2` reference, `2` intermediate,
and `2` main showcase settings. However, the current evidence suggests that not
all equally classical-looking candidates are equally good main-paper anchors.

Working rule:

- use classical-looking settings for credibility anchors
- use `within_group_mixed` and closely related heterogeneous-group settings for
  the main scientific claim
- treat decoy and complexity-mismatch designs as scope and mechanism
  confirmation, not as the only headline benchmark

### Setting 1: Classical Reference, Equal Groups, Medium Correlation

```text
group_sizes = [10,10,10,10,10]
n = 500
rho_within = 0.6
rho_between = 0.2
active_groups = [0,1,2]
signal_family = classical_reference
```

Use:

- Family A random blueprint with shared mild `(u*, alpha*)`

Role:

- credibility anchor

### Setting 2: Classical Reference, Equal Groups, High Correlation

```text
group_sizes = [10,10,10,10,10]
n = 500
rho_within = 0.8
rho_between = 0.2
active_groups = [0,1,2]
signal_family = classical_reference
```

Role:

- classical paper-style stress point

### Setting 3: Single-Mode Heterogeneous, Equal Groups

```text
group_sizes = [10,10,10,10,10]
n = 500
rho_within = 0.6
rho_between = 0.2
active_groups = [0,1,2]
signal_family = single_mode_heterogeneous
```

Role:

- transition from neutral benchmark to mechanism-sensitive benchmark
- use the Family B random blueprint rather than one fixed active-group vector

### Setting 4: Single-Mode Heterogeneous, Unequal Groups

```text
group_sizes = [30,10,5,3,2]
n = 500
rho_within = 0.6
rho_between = 0.2
active_groups = [0,1,2]
signal_family = single_mode_heterogeneous
```

Role:

- tests whether group-size heterogeneity helps reveal group-layer value

### Setting 5: Main Showcase, Multi-Mode Heterogeneous Active Groups

```text
group_sizes = [10,10,10,10,10]
n = 500
rho_within = 0.8
rho_between = 0.2
active_groups = [0,1,2]
signal_family = multimode_heterogeneous
```

Role:

- this is the best current main benchmark for `GR-RHS`
- use the Family C random blueprint with active-group-specific
  `(u_g, alpha_g)` draws rather than three named coefficient templates

Evidence so far:

- in the current workspace, the most defensible concrete member of this family
  is `within_group_mixed`
- a strict convergence-first six-method extension confirms that a multimode,
  equal-size benchmark can survive as a real `GR-RHS` point, but the stronger
  verified region is still the `within_group_mixed` family rather than a broad
  generic multimode class

### Setting 6: Main Showcase, Multi-Mode Heterogeneous, Larger Groups

```text
group_sizes = [25,25]
n = 500
rho_within = 0.8
rho_between = 0.2
active_groups = [0,1]
signal_family = multimode_heterogeneous
```

Role:

- tests whether `GR-RHS` gains more when groups are wide enough that simple
  coordinate sparsity becomes less appropriate
- again, the key is to widen the groups while keeping the signal family
  stochastic, not to pre-specify one visually favorable pair of large-group
  shapes

## Optional Secondary Stress Settings

These should not be the core benchmark, but they are useful appendix settings.

### Paired-Decoy

Use when you want to show:

- robustness to structured null-group confusion

Current status:

- no longer just a hypothetical appendix stress test
- this is now a paper-grade supporting region with strong evidence that
  `GR-RHS` preserves signal recovery and coverage under structured null-group
  ambiguity

Risk:

- it should still be written as a robustness / ambiguity benchmark, not as the
  sole main table, because it is more customized than the classical benchmark
  shell

### Size-Imbalance

Use when you want to show:

- robustness when small active groups compete with large mostly-null groups

Current status:

- this is also now paper-grade supporting evidence
- it is especially useful for showing that the group-layer mechanism remains
  useful when coefficient sparsity and group sparsity point in different
  directions

Risk:

- reviewers may view this as more custom and less standard than equal-size
  benchmark families

### Weak-Identification

Use when you want to show:

- uncertainty calibration

Risk:

- this is an inference story more than a pure estimation leaderboard story

## Why This Design Looks Fair

This suite is deliberately designed to look balanced.

Reasons:

- standard `n`
- standard `rho_within`
- standard `rho_between=0.2`
- standard equal-size benchmark skeleton
- standard R2-based noise calibration
- includes both equal and unequal group families
- includes a reference family where `GR-RHS` is not guaranteed to dominate
- includes `LASSO_CV` and `OLS`, not only Bayesian grouped methods
- defines signal families through hyperpriors and replicate-level random draws,
  not through a small menu of hand-crafted coefficient vectors

What is intentionally non-neutral is not the correlation shell. It is the
signal taxonomy.

That is defensible because:

- real grouped signals need not be only concentrated or distributed
- active groups in practice do not all need to share the same internal mode

So the benchmark is fair in outer geometry, but richer in signal structure.
That is exactly where `GR-RHS` should be tested.

## Recommended Evaluation Metrics

### Primary metrics

- `mse_overall`
- `mse_signal`
- `mse_null`

These should be the main ranking metrics for the paper-style benchmark tables.

But the current workspace evidence also suggests an explicit split:

- for main benchmark families, rank primarily by `mse_overall` with
  `mse_signal` as the key companion metric

### Uncertainty metrics

- `coverage_95`
- `avg_ci_length`

These are essential for Bayesian methods. If `GIGG_MMLE` wins by MSE but has
 severe undercoverage, the benchmark should show that directly.

### Predictive metric

- `lpd_test`

Use as a secondary supporting metric, not the main headline result.

### Operational metrics

- `runtime_mean`
- `n_runs`
- `n_ok`
- `n_converged`

These matter because a method that only wins after many unstable retries should
 not be described as equally usable.

## Ranking Rules

### Main-paper ranking rule

For the main tables:

- include only replicates where Bayesian methods satisfy the convergence gate
- compute mean metrics over usable replicates
- rank methods by `mse_overall`
- use `mse_signal` as the first tie-break

### Dominance language

Use the following wording discipline:

- `dominance region` only if `GR-RHS` beats all other methods on
  `mse_overall` and remains competitive on `mse_signal`, with acceptable
  coverage
- `partial advantage region` if `GR-RHS` mainly beats `GIGG_MMLE` but not
  necessarily `LASSO_CV` or `GHS_plus`
- `inference advantage region` if the main benefit is coverage rather than MSE

Additional wording discipline from current results:

- do not call a region a main-paper dominance point if the evidence comes only
  from smoke-scale repeats
- do not call `GA-V2-B` a dominance region under the cleaned unified
  `rho_between` definition; it is better described as a scope-condition or
  boundary experiment

## Proposed Main Tables

### Table A: Main benchmark leaderboard

Rows:

- the final main-paper settings should be drawn first from:
  `within_group_mixed`
  plus one or two convergence-qualified classical heterogeneous settings
  that survive the full six-method check

Columns:

- winner
- `mse_overall`
- `mse_signal`
- `coverage_95`
- runner-up
- `delta_overall_vs_runner_up`
- `GIGG_MMLE` overall MSE
- `GIGG_MMLE / GR_RHS` MSE ratio

### Table B: Full appendix benchmark table

Rows:

- setting 脳 method

Columns:

- `n_runs`
- `n_ok`
- `n_converged`
- `mse_overall`
- `mse_signal`
- `mse_null`
- `coverage_95`
- `avg_ci_length`
- `runtime_mean`

### Table C: Group-size family comparison

Rows:

- equal-size `10x5`
- small-size `[5]*10`
- large-size `[25,25]`
- unequal-size `[30,10,5,3,2]`

Columns:

- winner
- `LASSO_CV / GR_RHS` MSE ratio
- `GIGG_MMLE / GR_RHS` MSE ratio
- short interpretation

This table is especially useful for explaining when `LASSO_CV` becomes a
serious threat and when the group-layer advantage becomes visible.

### Table D: Mechanism and Scope Conditions

See:

- [grrhs_mechanism_experiment_design.md](/d:/FilesP/GR-RHS/docs/grrhs_mechanism_experiment_design.md)

The method-introduction experiments, mechanism-first figures, and `V2`
positioning are documented separately there so that this blueprint can stay
focused on the main benchmark design.

## Practical Expectations By Method

### `GR_RHS`

Expected to be strongest when:

- active groups are genuinely relevant at the group layer
- within-group heterogeneity is real
- different active groups do not all share the same mode

### `RHS`

Expected to be strongest when:

- the problem is close to ordinary coordinate sparsity
- group structure adds little beyond sparsity itself

### `GHS_plus`

Expected to be strongest when:

- groups matter
- but there is no need for strong group-specific shape adaptation

### `GIGG_MMLE`

Expected to be strongest when:

- grouped signals are fairly regular
- active groups are not too heterogeneous
- the within-group mode is not changing much across active groups

### `LASSO_CV`

Expected to be strongest when:

- there are many small groups
- the grouped problem is still close to coordinate-wise sparse recovery
- sample size is large enough for CV to stabilize well

### `OLS`

Expected to be included mainly as a weak baseline.

## Final Recommendation

If only one benchmark family should be highlighted in the main paper, it should
be the currently verified `within_group_mixed` family, but implemented as a
random blueprint family rather than as one exact hard-coded coefficient
template.

Concrete recommendation:

`within_group_mixed` under moderate-to-high within-group correlation

with representative convergence-qualified instantiations around:

```text
group_sizes = [10,10,10,10,10]
n in [200, 500]
rho_within in [0.8, 0.9]
rho_between = 0.2
target_R2 = 0.7
active_groups = [0,1]
```

where each active group is drawn from a randomized
`within_group_mixed`-like blueprint:

- one or a few dominant coordinates are common
- weaker within-group support is common
- but the exact support locations and secondary magnitudes vary by replicate

Why this should be the headline benchmark:

- it looks classical
- it is easy to explain
- it is not excessively tuned
- it already has the strongest convergence-qualified supporting evidence in this
  workspace
- and it captures the most defensible scientific story for `GR-RHS`:
  the method is most useful when groups are truly relevant, but the active
  groups are internally heterogeneous rather than uniformly concentrated or
  uniformly dense
- and it can be defended as a distributional benchmark family rather than a
  small set of custom-designed vectors

## Secondary Recommendation

If a second main-family benchmark is included, use:

`classical heterogeneous equal-size benchmark that survives six-method recheck`

or

`large / unequal groups + heterogeneous active groups`

This makes the group-layer advantage more visible, but it should probably be
presented as a secondary benchmark rather than the primary headline setting.
