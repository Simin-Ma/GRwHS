from __future__ import annotations

import numpy as np

from simulation_project.src.core.diagnostics.convergence import summarize_convergence
from pathlib import Path

from simulation_project.src.core.models.baselines import GroupedHorseshoePlus, RegularizedHorseshoeRegression
from simulation_project.src.core.models.gigg_regression import GIGGRegression
from simulation_project.src.core.models.grrhs_nuts import GRRHS_NUTS, _normalize_init_params, _should_use_median_init


def test_core_models_importable() -> None:
    assert GRRHS_NUTS is not None
    assert GIGGRegression is not None
    assert RegularizedHorseshoeRegression is not None
    assert GroupedHorseshoePlus is not None


def test_rhs_defaults_match_stan_only_rstanarm_aligned_path() -> None:
    model = RegularizedHorseshoeRegression()
    assert model.backend == "stan"
    assert model.scale_global == 0.01
    assert model.nu_global == 1.0
    assert model.nu_local == 1.0
    assert model.slab_df == 4.0
    assert model.slab_scale == 2.5
    assert Path(model._default_stan_file()).exists()


def test_nuts_resume_disables_init_to_median_when_init_params_present() -> None:
    init_params = _normalize_init_params({"sigma": np.asarray([1.0], dtype=np.float32)}, num_chains=1)
    assert init_params is not None
    assert _should_use_median_init(use_init_to_median=True, init_params=init_params) is False
    assert _should_use_median_init(use_init_to_median=True, init_params=None) is True
    assert _should_use_median_init(use_init_to_median=False, init_params=None) is False


def test_convergence_summary_handles_nonfinite_draws_without_crashing() -> None:
    samples = {
        "sigma": np.asarray(
            [
                [1.0, 1.1, np.nan, 1.2],
                [0.9, 1.0, 1.1, np.inf],
            ],
            dtype=float,
        )
    }
    out = summarize_convergence(samples)
    sigma = out["sigma"]
    assert sigma["diagnostic_valid"] is True
    assert np.isnan(float(sigma["rhat_max"]))
    assert np.isnan(float(sigma["ess_min"]))
