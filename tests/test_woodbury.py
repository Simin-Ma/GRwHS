from __future__ import annotations

import numpy as np
import numpy.testing as npt

from grrhs.inference.woodbury import woodbury_inverse


def test_woodbury_inverse_matches_direct_inversion():
    rng = np.random.default_rng(0)
    n, k = 4, 2
    A_raw = rng.normal(size=(n, n))
    A = A_raw @ A_raw.T + np.eye(n)
    U = rng.normal(size=(n, k))
    C_raw = rng.normal(size=(k, k))
    C = C_raw @ C_raw.T + np.eye(k)
    V = U.T

    direct = np.linalg.inv(A + U @ C @ V)
    via_identity = woodbury_inverse(A, U, np.linalg.inv(C), V)
    npt.assert_allclose(via_identity, direct, rtol=1e-10, atol=1e-10)


def test_woodbury_inverse_handles_diagonal_special_case():
    A = np.diag([2.0, 3.0])
    U = np.array([[1.0], [0.5]])
    C = np.array([[4.0]])
    V = U.T

    direct = np.linalg.inv(A + U @ C @ V)
    via_identity = woodbury_inverse(A, U, np.linalg.inv(C), V)
    npt.assert_allclose(via_identity, direct, rtol=1e-12, atol=1e-12)
