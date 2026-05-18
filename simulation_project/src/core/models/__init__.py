from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "GRRHS_Gibbs_Staged",
    "GIGGRegression",
]

_LAZY_EXPORTS = {
    "GRRHS_Gibbs_Staged": ("simulation_project.src.core.models.grrhs_nuts", "GRRHS_Gibbs_Staged"),
    "GIGGRegression": ("simulation_project.src.core.models.gigg_regression", "GIGGRegression"),
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
