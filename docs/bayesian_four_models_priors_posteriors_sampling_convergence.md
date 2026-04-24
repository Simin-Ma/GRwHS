# Bayesian Model Notes (Unified Runtime)

This note summarizes the Bayesian backends now used by the consolidated simulation pipeline.

## Runtime Modules

- GR-RHS staged Gibbs: `simulation_project/src/core/models/grrhs_nuts.py`
- RHS baseline: `simulation_project/src/core/models/baselines/models.py`
- GIGG: `simulation_project/src/core/models/gigg_regression.py`
- Grouped Horseshoe+: `simulation_project/src/core/models/baselines/grouped_horseshoe.py`

RHS is now a single Stan/HMC implementation aligned with `rstanarm::hs()`.
The active Stan files are:

- `simulation_project/src/core/models/baselines/stan/rhs_gaussian_regression.stan`
- `simulation_project/src/core/models/baselines/stan/rhs_logistic_regression.stan`

There is no separate NumPyro fallback implementation for RHS in the baseline path.
Defaults are aligned to the `rstanarm` regularized horseshoe interface:

- `global_df = 1`
- `local_df = 1`
- `slab_df = 4`
- `slab_scale = 2.5`

Grouped Horseshoe+ in this repository corresponds to the Xu et al. (2016)
hierarchical Bayesian grouped horseshoe (HBGHS) Gaussian Gibbs sampler.
The active wrapper uses paper-aligned defaults:

- `tau ~ C+(0, 1)`
- `lambda_g ~ C+(0, 1)`
- `delta_j ~ C+(0, 1)`

Wrapper entrypoint:

- `simulation_project/src/experiments/methods/fit_ghs_plus.py`

GIGG is ported from the CRAN R package. The active implementation lives at
`simulation_project/src/core/models/gigg_regression.py`, replacing all prior `grrhs`-path
references. Pipeline wrapper: `simulation_project/src/fit_gigg.py` (returns unified
`FitResult` objects for experiment runners).

Sanity check:

```python
from simulation_project.src.core.models.gigg_regression import GIGGRegression
```

## Diagnostics

Convergence diagnostics are provided by:

- `simulation_project/src/core/diagnostics/convergence.py`

and consumed by `simulation_project/src/utils.py`.

## Repository Policy

All old multi-stack experiment paths are removed; only the unified simulation path is supported.
