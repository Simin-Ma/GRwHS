from __future__ import annotations

import numpy as np

from simulation_second.src.bayes_kernel.experiments.methods import fit_gr_rhs_adaptive as adaptive_mod
from simulation_second.src.bayes_kernel.utils import FitResult, SamplerConfig


def test_regularized_posterior_eb_falls_back_to_prior_center_when_pilot_has_no_kappa(monkeypatch) -> None:
    def _fake_fit_gr_rhs(*args, **kwargs):
        return FitResult(
            method=str(kwargs.get("method_name", "GR_RHS")),
            status="ok",
            beta_mean=np.zeros(2, dtype=float),
            beta_draws=np.zeros((2, 2), dtype=float),
            kappa_draws=None,
            group_scale_draws=None,
            runtime_seconds=0.0,
            rhat_max=1.0,
            bulk_ess_min=100.0,
            divergence_ratio=0.0,
            converged=True,
            diagnostics={},
        )

    monkeypatch.setattr(adaptive_mod, "fit_gr_rhs", _fake_fit_gr_rhs)
    calib = adaptive_mod.calibrate_grrhs_beta_regularized_posterior_eb(
        np.eye(4, 2),
        np.ones(4, dtype=float),
        [[0], [1]],
        task="gaussian",
        seed=11,
        p0=1,
        sampler=SamplerConfig(chains=1, warmup=5, post_warmup_draws=5),
        grrhs_kwargs={},
        beta_kappa=8.0,
        prior_center=4.0,
        min_beta_kappa=1.0,
        max_beta_kappa=12.0,
    )

    assert float(calib.beta_kappa) == 4.0
    assert calib.details["fallback_reason"] == "pilot_missing_kappa_draws"
    assert calib.details["fallback_stage"] == "prior_center"
