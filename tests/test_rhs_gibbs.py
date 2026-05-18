from __future__ import annotations

import numpy as np

from simulation_project.src.core.models.baselines import RegularizedHorseshoeGibbs
from simulation_project.src.experiments.fitting import _fit_all_methods
from simulation_project.src.experiments.methods.fit_rhs_gibbs import fit_rhs_gibbs
from simulation_project.src.experiments.runtime import _resolve_method_list
from simulation_project.src.utils import SamplerConfig, method_display_name, method_result_label
from simulation_second.src.fitting import _rhs_sampler_strategy_for_package


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
    assert out.method == "RHS_HighDim"
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


def test_rhs_gibbs_collapsed_hyperparameter_updates_stabilize_high_dimensional_fit() -> None:
    rng = np.random.default_rng(41)
    n = 28
    p = 96
    X = rng.normal(size=(n, p))
    beta_true = np.zeros(p, dtype=float)
    beta_true[:8] = np.asarray([1.5, -1.2, 0.9, -0.8, 0.6, 0.5, -0.4, 0.3], dtype=float)
    y = X @ beta_true + rng.normal(scale=0.85, size=n)

    model = RegularizedHorseshoeGibbs(
        num_warmup=18,
        num_samples=18,
        num_chains=1,
        progress_bar=False,
        seed=41,
        lambda_active_fraction=0.2,
        lambda_active_min=12,
        lambda_full_refresh_every=4,
    )
    model.fit(X, y)

    diag = dict(model.sampler_diagnostics_)
    param = dict(diag.get("parameterization") or {})
    assert param.get("tau_refresh_after_local") is True
    assert param.get("beta_refresh_after_hyper") is True
    assert np.all(np.isfinite(np.asarray(model.coef_mean_, dtype=float)))
    assert np.isfinite(float(model.tau_mean_))
    assert np.isfinite(float(model.sigma_mean_))


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


def test_rhs_highdim_explicit_name_selectable_via_experiment_entrypoint() -> None:
    X, y, groups, _beta_true = _tiny_gaussian_problem(seed=17)
    sampler = SamplerConfig(chains=1, warmup=12, post_warmup_draws=12, ess_threshold=5.0)
    out = _fit_all_methods(
        X,
        y,
        groups,
        task="gaussian",
        seed=17,
        p0=5,
        sampler=sampler,
        methods=["RHS_HighDim"],
        enforce_bayes_convergence=False,
        rhs_sampler_strategy="high_dim",
    )

    assert list(out.keys()) == ["RHS_HighDim"]
    assert out["RHS_HighDim"].status == "ok"
    assert out["RHS_HighDim"].method == "RHS_HighDim"


def test_rhs_auto_alias_routes_to_highdim_sampler_when_requested() -> None:
    X, y, groups, _beta_true = _tiny_gaussian_problem(seed=18)
    sampler = SamplerConfig(chains=1, warmup=12, post_warmup_draws=12, ess_threshold=5.0)
    out = _fit_all_methods(
        X,
        y,
        groups,
        task="gaussian",
        seed=18,
        p0=5,
        sampler=sampler,
        methods=["RHS"],
        enforce_bayes_convergence=False,
        rhs_sampler_strategy="high_dim",
    )

    res = out["RHS"]
    diag = dict(res.diagnostics or {})
    sampler_diag = dict(diag.get("sampler_diagnostics") or {})
    assert res.method == "RHS"
    assert diag.get("rhs_sampler_name") == "RHS"
    assert diag.get("rhs_sampler_strategy") == "high_dim"
    assert diag.get("rhs_highdim_route") == "stan_exact"
    protocol = dict(diag.get("computational_protocol") or {})
    assert protocol.get("method_family") == "RHS"
    assert protocol.get("protocol") == "high_dim"
    assert protocol.get("sampler_backend") == "stan_exact"
    assert protocol.get("posterior_target") == "same_model_family"
    assert isinstance(protocol.get("sampler_budget"), dict)


def test_rhs_package_strategy_helper_distinguishes_low_vs_high_dimension_suites() -> None:
    assert _rhs_sampler_strategy_for_package("simulation_second") == "low_dim"
    assert _rhs_sampler_strategy_for_package("Simulation_highdimension") == "high_dim"


def test_method_registry_records_unified_protocol_for_gigg_and_ghs_monkeypatched(monkeypatch) -> None:
    from simulation_project.src.experiments import method_registry as registry_mod
    from simulation_project.src.utils import FitResult

    X, y, groups, _beta_true = _tiny_gaussian_problem(seed=19)
    sampler = SamplerConfig(chains=1, warmup=5, post_warmup_draws=5, ess_threshold=1.0)

    def _fake_fit(*args, method_label: str = "GIGG_MMLE", **kwargs) -> FitResult:
        return FitResult(
            method=str(method_label),
            status="ok",
            beta_mean=np.zeros(X.shape[1]),
            beta_draws=np.zeros((1, 5, X.shape[1])),
            kappa_draws=None,
            group_scale_draws=None,
            runtime_seconds=0.0,
            rhat_max=1.0,
            bulk_ess_min=10.0,
            divergence_ratio=0.0,
            converged=True,
            diagnostics={},
        )

    monkeypatch.setattr(
        "simulation_project.src.experiments.methods.fit_gigg.fit_gigg_mmle",
        _fake_fit,
    )
    import importlib

    ghs_module = importlib.import_module("simulation_project.src.experiments.methods.fit_ghs_plus")
    monkeypatch.setattr(ghs_module, "fit_ghs_plus", lambda *args, **kwargs: _fake_fit(method_label="GHS_plus"))

    reg = registry_mod.build_default_method_registry()
    ctx = registry_mod.MethodContext(
        X=X,
        y=y,
        groups=groups,
        task="gaussian",
        seed=19,
        p0=5,
        grrhs_p0=5,
        n=X.shape[0],
        sampler=sampler,
        rhs_sampler_strategy="high_dim",
        rhs_kwargs={},
        grrhs_kwargs={},
        gigg_mmle_kwargs={},
        gigg_fixed_kwargs={},
    )

    gigg = reg.run("GIGG_MMLE", ctx)
    ghs = reg.run("GHS_plus", ctx)
    gigg_protocol = dict((gigg.diagnostics or {}).get("computational_protocol") or {})
    ghs_protocol = dict((ghs.diagnostics or {}).get("computational_protocol") or {})
    assert gigg_protocol.get("method_family") == "GIGG_MMLE"
    assert gigg_protocol.get("protocol") == "high_dim"
    assert gigg_protocol.get("sampler_backend") == "mmle_btrick"
    assert ghs_protocol.get("method_family") == "GHS_plus"
    assert ghs_protocol.get("protocol") == "high_dim"
    assert ghs_protocol.get("sampler_backend") == "gibbs_light_exact"
