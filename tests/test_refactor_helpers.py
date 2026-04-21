from __future__ import annotations

from simulation_project.src.experiment_aliases import cli_choice_to_key, normalize_sweep_experiment
from simulation_project.src.fit_helpers import as_int_groups, fit_error_result, scaled_iteration_budget
from simulation_project.src.utils import SamplerConfig


def test_normalize_sweep_experiment_aliases() -> None:
    assert normalize_sweep_experiment("exp1_to_exp5") == "all"
    assert normalize_sweep_experiment("EXP3B") == "exp3b"
    assert normalize_sweep_experiment("4") == "exp4"


def test_cli_choice_to_key() -> None:
    assert cli_choice_to_key("1") == "exp1"
    assert cli_choice_to_key("analysis") == "analysis"


def test_scaled_iteration_budget_clamps_floor_and_cap() -> None:
    sampler = SamplerConfig(warmup=100, post_warmup_draws=150)
    burnin, draws = scaled_iteration_budget(
        sampler,
        iter_mult=2,
        iter_floor=500,
        iter_cap=600,
    )
    assert burnin == 500
    assert draws == 500

    burnin2, draws2 = scaled_iteration_budget(
        sampler,
        iter_mult=10,
        iter_floor=100,
        iter_cap=700,
    )
    assert burnin2 == 700
    assert draws2 == 700


def test_fit_error_result_shape() -> None:
    res = fit_error_result("RHS", "boom")
    assert res.method == "RHS"
    assert res.status == "error"
    assert res.error == "boom"
    assert res.beta_mean is None
    assert res.converged is False


def test_as_int_groups() -> None:
    groups = as_int_groups([(0, 1), [2, 3]])
    assert groups == [[0, 1], [2, 3]]
