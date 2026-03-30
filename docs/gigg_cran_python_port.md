# Python Port of CRAN `gigg` (v0.2.1): API + Gibbs Sampling Logic (Repo-Aligned)

> Purpose: Provide a Python interface that is intentionally *very close* to the CRAN package manual you pasted (`gigg` v0.2.1), while still being aligned with this repository’s implementation and runners.
>
> Code-as-truth: `grrhs/models/gigg_cran.py` (pure NumPy port; no R dependency).

---

## Package / module identity

- Upstream reference: CRAN package `gigg` v0.2.1 (published 2021-03-09; manual date 2025-07-22 in your excerpt).
- Python implementation in this repo: `grrhs.models.gigg_cran`
- Sampling route: **Gibbs sampler** (fixed-hyper and MMLE variants), matching the CRAN design.

---

## Contents (Python)

Analogues of the CRAN manual entries (names preserved where feasible):

- `chol_solve(M, V)` – SPD solve via Cholesky (CRAN `chol_solve`)
- `quick_solve(XtX_inv, D_pos, vec_draw)` – rank-update solve (CRAN `quick_solve`)
- `digamma_inv(y, precision=1e-8)` – inverse digamma (CRAN `digamma_inv`)
- `rgig_cpp(chi, psi, lambda)` – scalar GIG sampler (CRAN `rgig_cpp`; rejection)
- `gigg_fixed_gibbs_sampler(...)` – fixed-hyper Gibbs sampler (CRAN `gigg_fixed_gibbs_sampler`)
- `gigg_mmle_gibbs_sampler(...)` – MMLE-hyper Gibbs sampler (CRAN `gigg_mmle_gibbs_sampler`)
- `gigg(X, C, Y, method=..., grp_idx=..., ...)` – CRAN-like high-level wrapper (CRAN `gigg`)
- `GIGGRegressionCRAN` – runner-friendly `fit/predict` wrapper around `gigg()`

---

## `gigg(...)` (CRAN-like wrapper)

Implementation: `grrhs/models/gigg_cran.py:gigg`.

### Usage (Python)

```python
from grrhs.models.gigg_cran import gigg

out = gigg(
    X, C, Y,
    method="fixed",        # or "mmle"
    grp_idx=grp_idx,       # 1..G, contiguous, sorted by group (CRAN requirement)
    n_burn_in=500,
    n_samples=1000,
    n_thin=1,
    verbose=True,
    btrick=False,
    stable_solve=True,
    seed=0,
)
```

### Arguments (Python ↔ CRAN)

- `X` (n×p): covariates with GIGG shrinkage.
- `C` (n×k): covariates with **no shrinkage** (intercept/adjustments).
- `Y` (n,): response.
- `method`: `"fixed"` or `"mmle"`.
- `grp_idx` (length p): **1-based**, contiguous group ids with **no gaps**, and **sorted / contiguous blocks** (CRAN sampler assumption).
- `alpha_inits`, `beta_inits`: initial regression coefficients (defaults to zeros).
- `a`, `b`: prior shape parameters (defaults to 0.5 like CRAN examples).
- `sigma_sq_init`, `tau_sq_init`: initial values.
- `n_burn_in`, `n_samples`, `n_thin`: Gibbs budget knobs.
- `verbose`, `btrick`, `stable_solve`: kept for API fidelity.
- `seed`: NumPy RNG seed.

### Value / return

Returns a dict intended to mirror the CRAN list shape:

- `draws`: dict of arrays for `alphas`, `betas`, `lambda_sqs`, `gamma_sqs`, `tau_sqs`, `sigma_sqs`, plus dataset + sampler metadata.
- Posterior summaries (computed from draws, matching CRAN naming):
  - `beta.hat`, `beta.lcl.95`, `beta.ucl.95`
  - `alpha.hat`, `alpha.lcl.95`, `alpha.ucl.95`
  - `sigma_sq.hat`, `sigma_sq.lcl.95`, `sigma_sq.ucl.95`

---

## Gibbs sampler entry points (lower level)

Implementation: `grrhs/models/gigg_cran.py:gigg_fixed_gibbs_sampler` and `grrhs/models/gigg_cran.py:gigg_mmle_gibbs_sampler`.

- Both functions implement a **block Gibbs** scheme for:
  - regression coefficients (`alpha`, `beta`) (Gaussian conditionals),
  - shrinkage parameters (`lambda_sq`, `gamma_sq`, `tau_sq`),
  - residual variance (`sigma_sq`),
  - augmentation variable(s) (e.g. `nu`), as in CRAN.
- These are direct ports of the CRAN routines; for exact parameter names and update order, read the corresponding function bodies in `grrhs/models/gigg_cran.py`.

---

## Using this backend inside the experiment runner

This repo’s main GIGG model name is still `gigg` / `gigg_regression`, but you can select the CRAN-compatible backend via config:

```yaml
model:
  name: gigg
  backend: cran
```

This builds `grrhs.models.gigg_cran.GIGGRegressionCRAN` (see `grrhs/experiments/registry.py:_build_gigg`).

