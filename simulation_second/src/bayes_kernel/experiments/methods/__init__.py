from __future__ import annotations

from importlib import import_module
from typing import Any

from .helpers import as_int_groups, fit_error_result, scaled_iteration_budget

__all__ = [
    "as_int_groups",
    "fit_error_result",
    "scaled_iteration_budget",
    "fit_ols",
    "fit_lasso_cv",
    "fit_ghs_plus",
    "fit_gigg_mmle",
    "fit_gigg_fixed",
    "fit_gr_rhs",
    "fit_gr_rhs_adaptive_beta",
    "fit_rhs_gibbs",
    "fit_rhs",
]

_LAZY_EXPORTS = {
    "fit_ols": ("simulation_second.src.bayes_kernel.experiments.methods.fit_classical", "fit_ols"),
    "fit_lasso_cv": ("simulation_second.src.bayes_kernel.experiments.methods.fit_classical", "fit_lasso_cv"),
    "fit_ghs_plus": ("simulation_second.src.bayes_kernel.experiments.methods.fit_ghs_plus", "fit_ghs_plus"),
    "fit_gigg_mmle": ("simulation_second.src.bayes_kernel.experiments.methods.fit_gigg", "fit_gigg_mmle"),
    "fit_gigg_fixed": ("simulation_second.src.bayes_kernel.experiments.methods.fit_gigg", "fit_gigg_fixed"),
    "fit_gr_rhs": ("simulation_second.src.bayes_kernel.experiments.methods.fit_gr_rhs", "fit_gr_rhs"),
    "fit_gr_rhs_adaptive_beta": ("simulation_second.src.bayes_kernel.experiments.methods.fit_gr_rhs_adaptive", "fit_gr_rhs_adaptive_beta"),
    "fit_rhs_gibbs": ("simulation_second.src.bayes_kernel.experiments.methods.fit_rhs_gibbs", "fit_rhs_gibbs"),
    "fit_rhs": ("simulation_second.src.bayes_kernel.experiments.methods.fit_rhs", "fit_rhs"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(str(name))
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

