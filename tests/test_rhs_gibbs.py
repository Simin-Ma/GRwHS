from __future__ import annotations

import numpy as np

from simulation_project.src.core.models.baselines import RegularizedHorseshoeGibbs
from simulation_project.src.experiments.fitting import _fit_all_methods
from simulation_project.src.experiments.methods.fit_rhs_gibbs import fit_rhs_gibbs
from simulation_project.src.experiments.runtime import _resolve_method_list
from simulation_project.src.utils import SamplerConfig, method_display_name, method_result_label


def _tiny_gaussian_problem(seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[list[int]], np.ndarray]:
    rng = np.random.default_rng(int(seed))
    n = 36
    p = 24
    X = rng.normal(size=(n, p))
    beta_true = np.zeros(p, dtype=float)
    beta_true[:5] = np.asarray([1.75, -1.25, 0.9, 0.0, -0.6], dtype=float)
    y = X @ beta_true + rng.normal(scale=0.8, size=n)
    groups = [[j] for j in range(p)]
    return X, y, groups, beta_true


def test_rhs_gibbs_model_importable_and_defaults_valid() -> None:
    model = RegularizedHorseshoeGibbs()
    assert model.scale_global > 0.0
    assert model.slab_scale == 2.5
    assert model.slab_df == 4.0
    assert model.num_warmup > 0
    assert model.num_samples > 0
    assert model.lambda_warmup_full_refresh is True
    assert model.tau_refresh_after_local is True
    assert model.beta_refresh_after_hyper is True


def test_fit_rhs_gibbs_gaussian_runs_and_returns_draws() -> None:
    X, y, groups, beta_true = _tiny_gaussian_problem(seed=11)
    sampler = SamplerConfig(chains=1, warmup=20, post_warmup_draws=20, ess_threshold=5.0)
    out = fit_rhs_gibbs(
        X,
        y,
        groups,
        task="gaussian",
        seed=11,
        p0=5,
        sampler=sampler,
        progress_bar=False,
    )

    assert out.status == "ok"
    assert out.method == "RHS_Gibbs"
    assert out.beta_mean is not None
    assert out.beta_draws is not None
    assert out.tau_draws is not None
    assert np.asarray(out.beta_draws).shape == (20, X.shape[1])
    assert np.asarray(out.tau_draws).shape == (20,)
    mse = float(np.mean((np.asarray(out.beta_mean, dtype=float) - beta_true) ** 2))
    assert np.isfinite(mse)
    diag = dict(out.diagnostics or {})
    sampler_diag = dict(diag.get("sampler_diagnostics") or {})
    assert sampler_diag.get("backend") == "rhs_gibbs_woodbury"
    lambda_refresh = dict(sampler_diag.get("lambda_refresh") or {})
    assert float(lambda_refresh.get("mean_lambda_updates_per_iter_per_chain", 0.0)) > 0.0
    assert float(lambda_refresh.get("mean_lambda_update_fraction_per_iter", 0.0)) > 0.0
    assert bool(lambda_refresh.get("warmup_full_refresh")) is True


def test_rhs_gibbs_active_lambda_refresh_reduces_average_updates_on_wider_problem() -> None:
    rng = np.random.default_rng(21)
    n = 32
    p = 80
    X = rng.normal(size=(n, p))
    beta_true = np.zeros(p, dtype=float)
    beta_true[:6] = np.asarray([1.6, -1.0, 0.8, -0.5, 0.3, 0.2], dtype=float)
    y = X @ beta_true + rng.normal(scale=0.9, size=n)

    model = RegularizedHorseshoeGibbs(
        num_warmup=12,
        num_samples=12,
        num_chains=1,
        progress_bar=False,
        seed=21,
        lambda_active_fraction=0.25,
        lambda_active_min=8,
        lambda_full_refresh_every=6,
    )
    model.fit(X, y)

    diag = dict(model.sampler_diagnostics_)
    refresh = dict(diag.get("lambda_refresh") or {})
    frac = float(refresh.get("mean_lambda_update_fraction_per_iter", 1.0))
    assert frac > 0.0
    assert frac < 1.0
    assert int(refresh.get("active_refresh_count", 0)) > 0
    assert int(refresh.get("full_refresh_count", 0)) > 0


def test_rhs_gibbs_coupled_refresh_options_do_not_break_fit() -> None:
    X, y, _groups, _beta_true = _tiny_gaussian_problem(seed=31)
    model = RegularizedHorseshoeGibbs(
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        progress_bar=False,
        seed=31,
        lambda_warmup_full_refresh=False,
        tau_refresh_after_local=False,
        beta_refresh_after_hyper=False,
    )
    model.fit(X, y)
    diag = dict(model.sampler_diagnostics_)
    param = dict(diag.get("parameterization") or {})
    refresh = dict(diag.get("lambda_refresh") or {})
    assert param.get("tau_refresh_after_local") is False
    assert param.get("beta_refresh_after_hyper") is False
    assert refresh.get("warmup_full_refresh") is False


def test_fit_rhs_gibbs_logistic_returns_explicit_error() -> None:
    X, _y, groups, _beta_true = _tiny_gaussian_problem(seed=12)
    rng = np.random.default_rng(12)
    y = rng.binomial(1, 0.5, size=X.shape[0]).astype(float)
    sampler = SamplerConfig(chains=1, warmup=10, post_warmup_draws=10, ess_threshold=5.0)
    out = fit_rhs_gibbs(
        X,
        y,
        groups,
        task="logistic",
        seed=12,
        p0=4,
        sampler=sampler,
        progress_bar=False,
    )

    assert out.status == "error"
    assert "gaussian likelihood only" in str(out.error).lower()


def test_rhs_gibbs_selectable_via_experiment_entrypoint() -> None:
    X, y, groups, _beta_true = _tiny_gaussian_problem(seed=13)
    sampler = SamplerConfig(chains=1, warmup=12, post_warmup_draws=12, ess_threshold=5.0)
    out = _fit_all_methods(
        X,
        y,
        groups,
        task="gaussian",
        seed=13,
        p0=5,
        sampler=sampler,
        methods=["RHS_Gibbs"],
        enforce_bayes_convergence=False,
    )

    assert list(out.keys()) == ["RHS_Gibbs"]
    assert out["RHS_Gibbs"].status == "ok"
    assert out["RHS_Gibbs"].beta_mean is not None


def test_rhs_gibbs_method_name_helpers_and_resolution() -> None:
    assert _resolve_method_list(["RHS_Gibbs"]) == ["RHS_Gibbs"]
    assert _resolve_method_list(["RHS", "RHS_Gibbs"]) == ["RHS", "RHS_Gibbs"]
    assert method_display_name("RHS_Gibbs") == "RHS-Gibbs"
    assert method_result_label("RHS_Gibbs") == "RHS-Gibbs [woodbury_slice]"
