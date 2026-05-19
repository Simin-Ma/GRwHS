from __future__ import annotations

from real_data_experiment.src.dataset import _replicate_seed as real_data_replicate_seed
from simulation_mechanism.src.utils import setting_replicate_seed as mechanism_setting_replicate_seed
from simulation_second.src.utils import setting_replicate_seed as benchmark_setting_replicate_seed


def test_setting_replicate_seeds_are_stable_distinct_and_backend_safe() -> None:
    ids = [
        "setting_1_classical_equal_medium",
        "setting_2_classical_equal_high",
        "hd_setting_5_multimode_showcase",
        "m2_mixed_decoy_rw080",
    ]
    seeds = [benchmark_setting_replicate_seed(setting_id, 1, master_seed=20260425) for setting_id in ids]
    assert seeds == [benchmark_setting_replicate_seed(setting_id, 1, master_seed=20260425) for setting_id in ids]
    assert len(set(seeds)) == len(seeds)
    assert all(0 < seed < 2_147_000_000 for seed in seeds)
    assert benchmark_setting_replicate_seed(ids[0], 1) != benchmark_setting_replicate_seed(ids[0], 2)


def test_mechanism_and_real_data_seeds_use_independent_namespaces() -> None:
    setting_seed = mechanism_setting_replicate_seed("m2_mixed_decoy_rw080", 1, master_seed=20260425)
    data_seed = real_data_replicate_seed(20260425, "m2_mixed_decoy_rw080", 1)
    assert setting_seed != data_seed
    assert 0 < setting_seed < 2_147_000_000
    assert 0 < data_seed < 2_147_000_000
