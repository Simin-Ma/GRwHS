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

    primary_group = data.info["primary_group"]
    assert isinstance(primary_group, np.ndarray)
    assert primary_group.shape == (cfg.p,)
    assert np.all(primary_group >= 0)


def test_generate_synthetic_classification_outputs_binary():
    cfg = SyntheticConfig(
        n=32,
        p=6,
        G=3,
        group_sizes="equal",
        signal={
            "sparsity": 0.5,
            "strong_frac": 0.5,
            "beta_scale_strong": 1.5,
            "beta_scale_weak": 0.3,
        },
        noise_sigma=0.0,
        seed=321,
        task="classification",
        response={"scale": 0.8, "bias": 0.1, "noise_std": 0.05},
        name="classification-unit",
    )

    data = generate_synthetic(cfg)

    assert data.X.shape == (32, 6)
    assert set(np.unique(data.y)).issubset({0.0, 1.0})
    assert data.info["task"] == "classification"
    assert 0.0 <= data.info["mean_probability"] <= 1.0
    assert data.noise_sigma == pytest.approx(0.0)


def test_synthetic_config_from_dict_defaults_and_overrides():
    cfg_dict = {
        "n": 20,
        "p": 5,
        "G": 5,
        "group_sizes": None,
        "signal": {"sparsity": 0.2},
        "overlap": {"fraction": 0.2, "max_memberships": 2},
    }

    cfg = synthetic_config_from_dict(cfg_dict, seed=999, name="scenario")
    assert cfg.n == 20
    assert cfg.p == 5
    assert cfg.G == 5
    assert cfg.group_sizes is None
    assert cfg.signal["sparsity"] == 0.2
    assert cfg.seed == 999
    assert cfg.name == "scenario"
    assert cfg.task == "regression"
    assert cfg.response.get("type") == "regression"
    assert cfg.overlap["fraction"] == 0.2


def test_synthetic_config_from_dict_classification_override():
    cfg_dict = {
        "n": 10,
        "p": 4,
        "G": 2,
        "group_sizes": "equal",
        "signal": {"sparsity": 0.5},
        "response": {"type": "classification", "scale": 0.75, "bias": -0.2},
    }

    cfg = synthetic_config_from_dict(cfg_dict, seed=111, name="cls")
    assert cfg.task == "classification"
    assert cfg.response["scale"] == 0.75
    assert cfg.response["bias"] == -0.2


def test_generate_synthetic_with_overlap_metadata():
    cfg = SyntheticConfig(
        n=60,
        p=12,
        G=4,
        group_sizes="equal",
        signal={"sparsity": 0.3},
        overlap={"fraction": 0.5, "max_memberships": 3},
        seed=314,
    )
    data = generate_synthetic(cfg)
    info = data.info.get("overlap")
    assert info
    feature_ids = info["feature_ids"]
    membership_counts = info["membership_counts"]
    assert feature_ids.size > 0
    assert membership_counts.shape == feature_ids.shape
    for fid, expected in zip(feature_ids, membership_counts):
        hits = sum(1 for grp in data.groups if int(fid) in grp)
        assert hits == expected
        assert hits >= 2


def test_signal_blueprint_assigns_requested_structure():
    cfg = SyntheticConfig(
        n=40,
        p=12,
        G=3,
        group_sizes=[4, 4, 4],
        signal={
            "blueprint": [
                {
                    "label": "strong",
                    "groups": [0],
                    "components": [
                        {"distribution": "constant", "count": 2, "value": 2.0, "sign": "positive", "tag": "strong"},
                        {"distribution": "uniform", "count": 1, "low": 0.4, "high": 0.5, "sign": "positive", "tag": "medium"},
                    ],
                },
                {
                    "label": "weak",
                    "groups": [1],
                    "components": [
                        {"distribution": "uniform", "fraction": 0.5, "low": 0.1, "high": 0.2, "tag": "weak"},
                    ],
                },
            ]
        },
        seed=2024,
    )

    data = generate_synthetic(cfg)

    strong_idx = data.info["strong_idx"]
    weak_idx = data.info["weak_idx"]
    blueprint_meta = data.info.get("signal_blueprint")

    assert strong_idx.size == 2
    assert np.allclose(data.beta[strong_idx], 2.0)
    assert weak_idx.size == 2  # 50% of a 4-feature group
    assert np.all(np.abs(data.beta[weak_idx]) >= 0.1 - 1e-8)
    assert np.all(np.abs(data.beta[weak_idx]) <= 0.2 + 1e-8)
    assert blueprint_meta is not None
    assert len(blueprint_meta["assignments"]) >= 2
