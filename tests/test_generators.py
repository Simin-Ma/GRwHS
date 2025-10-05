from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from data.generators import (
    GeneratorError,
    SyntheticConfig,
    generate_synthetic,
    make_groups,
    synthetic_config_from_dict,
)


def test_make_groups_equal_partition():
    groups = make_groups(p=6, G=3, group_sizes="equal")
    assert len(groups) == 3
    assert all(len(g) == 2 for g in groups)
    assert sorted(sum(groups, [])) == list(range(6))


def test_make_groups_custom_sequence_and_validation():
    groups = make_groups(p=5, G=2, group_sizes=[2, 3])
    assert groups == [[0, 1], [2, 3, 4]]

    with pytest.raises(GeneratorError):
        make_groups(p=5, G=2, group_sizes=[2, 2])

    with pytest.raises(GeneratorError):
        make_groups(p=5, G=2, group_sizes="unknown")


def test_generate_synthetic_respects_signal_configuration():
    cfg = SyntheticConfig(
        n=40,
        p=6,
        G=3,
        group_sizes="equal",
        signal={
            "sparsity": 0.5,
            "strong_frac": 0.4,
            "beta_scale_strong": 2.0,
            "beta_scale_weak": 0.4,
            "sign_mix": "positive",
            "group_sparsity": 0.5,
        },
        noise_sigma=0.2,
        seed=123,
        name="unit-test",
    )

    data = generate_synthetic(cfg)

    assert data.X.shape == (40, 6)
    assert data.y.shape == (40,)
    assert data.beta.shape == (6,)
    assert len(data.groups) == 3
    assert data.noise_sigma == pytest.approx(0.2)

    column_means = data.X.mean(axis=0)
    npt.assert_allclose(column_means, np.zeros(6), atol=1e-7)

    active = data.info["active_idx"]
    strong = data.info["strong_idx"]
    weak = data.info["weak_idx"]
    assert np.all(np.isin(strong, active))
    assert np.all(np.isin(weak, active))
    assert np.all(data.beta[active] != 0.0)
    assert np.all(data.beta[np.setdiff1d(np.arange(6), active)] == 0.0)

    # Positive sign convention enforced by the configuration
    assert np.all(data.beta[active] >= 0.0)


def test_synthetic_config_from_dict_defaults_and_overrides():
    cfg_dict = {
        "n": 20,
        "p": 5,
        "G": 5,
        "group_sizes": None,
        "signal": {"sparsity": 0.2},
    }

    cfg = synthetic_config_from_dict(cfg_dict, seed=999, name="scenario")
    assert cfg.n == 20
    assert cfg.p == 5
    assert cfg.G == 5
    assert cfg.group_sizes is None
    assert cfg.signal["sparsity"] == 0.2
    assert cfg.seed == 999
    assert cfg.name == "scenario"
