# GIGG CRAN Port Notes (Current Location)

The GIGG implementation used by the active pipeline is now located at:

- `simulation_project/src/core/models/gigg_regression.py`

This replaces older path references that pointed to removed `grrhs` directories.

## Usage in Pipeline

`simulation_project/src/fit_gigg.py` wraps this model and returns unified `FitResult` objects for experiment runners.

## Sanity Check

```python
from simulation_project.src.core.models.gigg_regression import GIGGRegression
```
