from __future__ import annotations

import numpy as np

from simulation_project.src.core.models.grrhs_nuts import GRRHS_Gibbs_Staged
from simulation_project.src.experiments.runtime import _invalidate_unconverged_result
from simulation_project.src.experiments.fitting import _fit_all_methods
from simulation_project.src.experiments.methods.fit_gr_rhs import fit_gr_rhs
from simulation_project.src.utils import FitResult, SamplerConfig
from simulation_second.src.fitting import _grrhs_defaults_for_package


def _tiny_gaussian_problem(seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[list[int]], np.ndarray]:
    rng = np.random.default_rng(int(seed))
    n = 40
    p = 8
    X = rng.normal(size=(n, p))
    beta_true = np.asarray([2.0, 1.5, 0.0, 0.0, -1.25, 0.0, 0.0, 0.5], dtype=float)
    y = X @ beta_true + rng.normal(scale=0.5, size=n)
    groups = [[0, 1], [2, 3], [4, 5], [6, 7]]
    return X, y, groups, beta_true


def test_grrhs_gibbs_staged_phase_diagnostics_present() -> None:
    X, y, groups, _beta_true = _tiny_gaussian_problem(seed=1)
    model = GRRHS_Gibbs_Staged(
        num_chains=2,
        iters=80,
        burnin=20,
        thin=1,
        seed=123,
        progress_bar=False,
        phase_a_max_iters=20,
        phase_b_max_iters=20,
        min_phase_a_iters=5,
        min_phase_b_iters=5,
        geometry_window=4,
        transition_window=4,
    )
    model.fit(X, y, groups=groups)

    diag = dict(model.sampler_diagnostics_)
    assert diag.get("backend") == "simcore_gibbs_staged"
    assert diag.get("adaptive_burnin") is True
    infos = diag.get("chain_phase_infos")
    assert isinstance(infos, list) and len(infos) == 2
    for info in infos:
        assert "phase_a_iters" in info
        assert "phase_b_iters" in info
        assert "actual_burnin" in info
        assert "resume_no_burnin_used" in info
        assert "structure_mode" in info
        assert "block_refresh_count" in info
        assert "block_refresh_steps" in info
        assert "block_refresh_group_histogram" in info


def test_grrhs_lambda_active_refresh_uses_random_coverage() -> None:
    model = GRRHS_Gibbs_Staged(
        lambda_active_fraction=0.25,
        lambda_active_min=10,
        lambda_full_refresh_every=99,
        lambda_warmup_full_refresh=False,
        lambda_random_fraction=0.5,
        progress_bar=False,
    )
    beta = np.linspace(100.0, 1.0, 100)
    rng = np.random.default_rng(123)
    idx = model._select_lambda_update_indices(beta, iteration=1, in_warmup=False, rng=rng)
    assert idx.size == 25
    top_deterministic = set(range(13))
    assert top_deterministic.issubset(set(map(int, idx)))
    assert any(int(j) >= 25 for j in idx)


def test_unconverged_bayesian_result_retains_draws_for_evaluation() -> None:
    beta = np.asarray([1.0, 0.0], dtype=float)
    res = FitResult(
        method="GR_RHS_HighDim",
        status="ok",
        beta_mean=beta.copy(),
        beta_draws=beta.reshape(1, 1, 2),
        kappa_draws=np.ones((1, 1, 1)),
        group_scale_draws=None,
        runtime_seconds=1.0,
        rhat_max=1.2,
        bulk_ess_min=5.0,
        divergence_ratio=float("nan"),
        converged=False,
        tau_draws=np.ones((1, 1)),
        diagnostics={},
    )
    out = _invalidate_unconverged_result(res, method="GR_RHS_HighDim", attempts=2)
    assert out.status == "warning"
    assert out.beta_mean is not None
    assert out.beta_draws is not None
    assert out.kappa_draws is not None
    assert out.tau_draws is not None
    assert "convergence_warning" in dict(out.diagnostics or {})


def test_grrhs_gibbs_staged_resume_no_burnin_skips_phases() -> None:
    X, y, groups, _beta_true = _tiny_gaussian_problem(seed=2)
    base = GRRHS_Gibbs_Staged(
        num_chains=1,
        iters=70,
        burnin=20,
        thin=1,
        seed=321,
        progress_bar=False,
        phase_a_max_iters=20,
        phase_b_max_iters=20,
        min_phase_a_iters=5,
        min_phase_b_iters=5,
        geometry_window=4,
        transition_window=4,
    )
    base.fit(X, y, groups=groups)
    state = dict(base.chain_last_states_[0])

    resumed = GRRHS_Gibbs_Staged(
        num_chains=1,
        iters=40,
        burnin=0,
        thin=1,
        seed=322,
        progress_bar=False,
        initial_chain_states=[state],
        resume_no_burnin=True,
        phase_a_max_iters=20,
        phase_b_max_iters=20,
        min_phase_a_iters=5,
        min_phase_b_iters=5,
        geometry_window=4,
        transition_window=4,
    )
    resumed.fit(X, y, groups=groups)
    info = resumed.sampler_diagnostics_["chain_phase_infos"][0]
    assert bool(info["resume_no_burnin_used"]) is True
    assert int(info["phase_a_iters"]) == 0
    assert int(info["phase_b_iters"]) == 0
    assert int(info["actual_burnin"]) == 0


def test_fit_gr_rhs_gaussian_defaults_to_staged_gibbs_and_tracks_resume() -> None:
    X, y, groups, _beta_true = _tiny_gaussian_problem(seed=3)
    sampler = SamplerConfig(chains=2, warmup=30, post_warmup_draws=30)
    out = fit_gr_rhs(
        X,
        y,
        groups,
        task="gaussian",
        seed=55,
        p0=2,
        sampler=sampler,
        progress_bar=False,
    )
    assert out.status == "ok"
    assert out.beta_mean is not None
    diag = dict(out.diagnostics or {})
    strat = dict(diag.get("sampling_strategy") or {})
    assert strat.get("backend") == "gibbs_staged"
    payload = diag.get("retry_resume_payload")
    assert isinstance(payload, dict)
    assert payload.get("backend") == "gibbs_staged"


def test_fit_gr_rhs_staged_gibbs_accepts_tau_refresh_overrides() -> None:
    X, y, groups, _beta_true = _tiny_gaussian_problem(seed=35)
    sampler = SamplerConfig(chains=1, warmup=20, post_warmup_draws=20, ess_threshold=1.0)
    out = fit_gr_rhs(
        X,
        y,
        groups,
        task="gaussian",
        seed=135,
        p0=2,
        sampler=sampler,
        progress_bar=False,
        sampler_backend="gibbs_staged",
        tau_target="groups",
        grouped_tau_refresh_repeats=2,
        grouped_sigma_tau_block_repeats=4,
    )

    assert out.status == "ok"
    sampler_diag = dict((out.diagnostics or {}).get("sampler_diagnostics") or {})
    assert int(sampler_diag.get("grouped_tau_refresh_repeats")) == 2
    assert int(sampler_diag.get("grouped_sigma_tau_block_repeats")) == 4


def test_fit_gr_rhs_can_return_explicit_lowdim_method_name() -> None:
    X, y, groups, _beta_true = _tiny_gaussian_problem(seed=33)
    sampler = SamplerConfig(chains=2, warmup=20, post_warmup_draws=20)
    out = fit_gr_rhs(
        X,
        y,
        groups,
        task="gaussian",
        seed=133,
        p0=2,
        sampler=sampler,
        progress_bar=False,
        sampler_backend="gibbs_staged",
        method_name="GR_RHS_LowDim",
    )
    assert out.status == "ok"
    assert out.method == "GR_RHS_LowDim"
    diag = dict(out.diagnostics or {})
    assert diag.get("grrhs_sampler_name") == "GR_RHS_LowDim"
    assert diag.get("grrhs_sampler_strategy") == "low_dim"


def test_fit_gr_rhs_gaussian_staged_gibbs_small_problem_not_clearly_worse_than_nuts() -> None:
    X, y, groups, beta_true = _tiny_gaussian_problem(seed=4)
    sampler = SamplerConfig(chains=2, warmup=40, post_warmup_draws=40)

    staged = fit_gr_rhs(
        X,
        y,
        groups,
        task="gaussian",
        seed=77,
        p0=2,
        sampler=sampler,
        progress_bar=False,
        sampler_backend="gibbs_staged",
    )
    nuts = fit_gr_rhs(
        X,
        y,
        groups,
        task="gaussian",
        seed=77,
        p0=2,
        sampler=sampler,
        progress_bar=False,
        sampler_backend="nuts",
    )

    assert staged.status == "ok"
    assert nuts.status == "ok"
    assert staged.beta_mean is not None
    assert nuts.beta_mean is not None

    mse_staged = float(np.mean((np.asarray(staged.beta_mean, dtype=float) - beta_true) ** 2))
    mse_nuts = float(np.mean((np.asarray(nuts.beta_mean, dtype=float) - beta_true) ** 2))
    assert np.isfinite(mse_staged)
    assert np.isfinite(mse_nuts)
    assert mse_staged <= max(0.75, 2.5 * mse_nuts + 1e-8)


def test_fit_gr_rhs_gaussian_collapsed_profile_backend_runs() -> None:
    X, y, groups, _beta_true = _tiny_gaussian_problem(seed=6)
    sampler = SamplerConfig(chains=2, warmup=15, post_warmup_draws=15, ess_threshold=1.0)

    out = fit_gr_rhs(
        X,
        y,
        groups,
        task="gaussian",
        seed=88,
        p0=2,
        sampler=sampler,
        progress_bar=False,
        sampler_backend="collapsed_profile",
        use_local_scale=False,
        tau_target="groups",
    )

    assert out.status == "ok"
    assert out.beta_mean is not None
    diag = dict(out.diagnostics or {})
    strat = dict(diag.get("sampling_strategy") or {})
    sampler_diag = dict(diag.get("sampler_diagnostics") or {})
    assert strat.get("backend") == "collapsed_profile"
    assert sampler_diag.get("backend") == "simcore_collapsed_nuts"
    assert sampler_diag.get("profile_mode") is True
    assert sampler_diag.get("nuts_dim") == len(groups) + 2
    assert "hmc" in sampler_diag
    assert "posterior_quality" in sampler_diag
    assert out.beta_draws is not None
    assert np.asarray(out.beta_draws).ndim == 3


def test_grrhs_auto_alias_routes_to_highdim_backend_when_requested() -> None:
    X, y, groups, _beta_true = _tiny_gaussian_problem(seed=34)
    sampler = SamplerConfig(chains=1, warmup=10, post_warmup_draws=10, ess_threshold=1.0)
    out = _fit_all_methods(
        X,
        y,
        groups,
        task="gaussian",
        seed=134,
        p0=2,
        sampler=sampler,
        methods=["GR_RHS"],
        enforce_bayes_convergence=False,
        rhs_sampler_strategy="high_dim",
    )
    res = out["GR_RHS"]
    diag = dict(res.diagnostics or {})
    strat = dict(diag.get("sampling_strategy") or {})
    assert res.method == "GR_RHS"
    assert diag.get("grrhs_sampler_name") == "GR_RHS"
    assert diag.get("grrhs_sampler_strategy") == "high_dim"
    assert strat.get("backend") == "collapsed_profile"


def test_grrhs_package_defaults_distinguish_low_vs_high_dimension_suites() -> None:
    low = _grrhs_defaults_for_package("simulation_second")
    high = _grrhs_defaults_for_package("Simulation_highdimension")
    assert low.get("sampler_backend") == "gibbs_staged"
    assert high.get("sampler_backend") == "collapsed_profile"
    assert bool(high.get("use_local_scale")) is False


def test_fit_gr_rhs_logistic_does_not_silently_route_to_nuts() -> None:
    X, _y, groups, _beta_true = _tiny_gaussian_problem(seed=5)
    rng = np.random.default_rng(5)
    y = rng.binomial(1, 0.5, size=X.shape[0]).astype(float)
    sampler = SamplerConfig(chains=2, warmup=20, post_warmup_draws=20)

    out = fit_gr_rhs(
        X,
        y,
        groups,
        task="logistic",
        seed=101,
        p0=2,
        sampler=sampler,
        progress_bar=False,
    )

    assert out.status == "error"
    assert "logistic gibbs" in str(out.error).lower() or "gaussian likelihood only" in str(out.error).lower()
