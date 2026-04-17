from __future__ import annotations

from simulation_project.src.core.models.baselines import GroupedHorseshoePlus, RegularizedHorseshoeRegression
from simulation_project.src.core.models.gigg_regression import GIGGRegression
from simulation_project.src.core.models.grrhs_nuts import GRRHS_NUTS


def test_core_models_importable() -> None:
    assert GRRHS_NUTS is not None
    assert GIGGRegression is not None
    assert RegularizedHorseshoeRegression is not None
    assert GroupedHorseshoePlus is not None
