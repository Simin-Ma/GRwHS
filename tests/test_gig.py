from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from scipy.stats import geninvgauss

from grwhs.inference.gig import sample_gig


def test_sample_gig_vectorized_matches_scipy_rvs():
    lam = -0.3
    chi = 0.7
    psi = 1.4
    size = (2, 3)

    rng_custom = np.random.default_rng(123)
    samples_custom = sample_gig(lam, chi, psi, size=size, rng=rng_custom)

    rng_reference = np.random.default_rng(123)
    expected = geninvgauss.rvs(
        p=lam,
        b=np.sqrt(chi * psi),
        size=size,
        scale=np.sqrt(chi / psi),
        random_state=rng_reference,
    )
    npt.assert_allclose(samples_custom, expected, rtol=1e-10, atol=1e-12)


def test_sample_gig_invalid_parameters_raise():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sample_gig(lambda_param=0.5, chi=0.0, psi=1.0, rng=rng)
    with pytest.raises(ValueError):
        sample_gig(lambda_param=0.5, chi=1.0, psi=-1.0, rng=rng)
