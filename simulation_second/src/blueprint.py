from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence

import numpy as np

from .schemas import FamilySpec, SettingSpec, SignalDraw
from .utils import canonical_groups


FAMILY_SPECS: Dict[str, FamilySpec] = {
    "classical_reference": FamilySpec(
        name="classical_reference",
        support_fraction_range=(0.6, 0.8),
        concentration_range=(3.0, 6.0),
        share_hyperparameters=True,
        description="Shared mild within-group blueprint close to classical dense or tiered settings.",
    ),
    "single_mode_heterogeneous": FamilySpec(
        name="single_mode_heterogeneous",
        support_fraction_range=(0.4, 0.8),
        concentration_range=(0.8, 2.0),
        share_hyperparameters=True,
        description="Shared heterogeneous blueprint family with independent support realization per active group.",
    ),
    "multimode_heterogeneous": FamilySpec(
        name="multimode_heterogeneous",
        support_fraction_range=(0.2, 0.8),
        concentration_range=(0.2, 6.0),
        share_hyperparameters=False,
        log_uniform_concentration=True,
        acceptance_alpha_ratio_min=3.0,
        acceptance_support_gap_min=0.25,
        description="Active groups draw distinct within-group blueprint parameters and must not collapse to the same mode.",
    ),
}


def family_spec_for_name(name: str, *, family_specs: Mapping[str, FamilySpec] | None = None) -> FamilySpec:
    key = str(name).strip().lower()
    lookup = FAMILY_SPECS if family_specs is None else {str(k).strip().lower(): v for k, v in family_specs.items()}
    if key not in lookup:
        raise KeyError(f"Unknown signal family: {name!r}")
    return lookup[key]


def _draw_uniform(rng: np.random.Generator, bounds: Sequence[float]) -> float:
    low = float(bounds[0])
    high = float(bounds[1])
    if high < low:
        low, high = high, low
    return float(rng.uniform(low, high))


def _draw_concentration(spec: FamilySpec, rng: np.random.Generator) -> float:
    if spec.log_uniform_concentration:
        low = math.log(float(spec.concentration_range[0]))
        high = math.log(float(spec.concentration_range[1]))
        return float(math.exp(rng.uniform(low, high)))
    return _draw_uniform(rng, spec.concentration_range)


def _accept_draw(
    spec: FamilySpec,
    support_fractions: Dict[int, float],
    concentrations: Dict[int, float],
) -> bool:
    if not support_fractions:
        return True
    if spec.acceptance_alpha_ratio_min is None and spec.acceptance_support_gap_min is None:
        return True

    alpha_ok = False
    support_ok = False

    if spec.acceptance_alpha_ratio_min is not None:
        vals = np.asarray(list(concentrations.values()), dtype=float)
        alpha_ok = bool(vals.size <= 1 or (np.max(vals) / max(np.min(vals), 1e-12)) >= float(spec.acceptance_alpha_ratio_min))

    if spec.acceptance_support_gap_min is not None:
        vals = np.asarray(list(support_fractions.values()), dtype=float)
        support_ok = bool(vals.size <= 1 or (np.max(vals) - np.min(vals)) >= float(spec.acceptance_support_gap_min))

    return alpha_ok or support_ok


def sample_signal_blueprint(
    setting: SettingSpec,
    rng: np.random.Generator,
    *,
    family_specs: Mapping[str, FamilySpec] | None = None,
    max_attempts: int = 256,
) -> SignalDraw:
    spec = family_spec_for_name(setting.family, family_specs=family_specs)
    groups = canonical_groups(setting.group_sizes)
    total_p = int(sum(setting.group_sizes))
    active_groups = tuple(int(g) for g in setting.active_groups)

    for attempt in range(max_attempts):
        if spec.share_hyperparameters:
            shared_support = _draw_uniform(rng, spec.support_fraction_range)
            shared_concentration = _draw_concentration(spec, rng)
            support_fractions = {gid: float(shared_support) for gid in active_groups}
            concentrations = {gid: float(shared_concentration) for gid in active_groups}
        else:
            support_fractions = {gid: _draw_uniform(rng, spec.support_fraction_range) for gid in active_groups}
            concentrations = {gid: _draw_concentration(spec, rng) for gid in active_groups}

        if not _accept_draw(spec, support_fractions, concentrations):
            continue

        energy = rng.dirichlet(np.full(len(active_groups), 4.0, dtype=float))
        energy_shares = {gid: float(share) for gid, share in zip(active_groups, energy)}
        beta = np.zeros(total_p, dtype=float)
        group_signs: Dict[int, int] = {}
        support_indices: Dict[int, list[int]] = {}

        for gid in active_groups:
            group = np.asarray(groups[gid], dtype=int)
            support_fraction = float(support_fractions[gid])
            support_size = int(max(1, min(group.size, round(support_fraction * group.size))))
            chosen = np.sort(rng.choice(group, size=support_size, replace=False)).astype(int)
            weights = rng.dirichlet(np.full(support_size, float(concentrations[gid]), dtype=float))
            sign = int(rng.choice([-1, 1]))
            beta[chosen] = sign * np.sqrt(float(energy_shares[gid]) * weights)
            group_signs[gid] = sign
            support_indices[gid] = [int(idx) for idx in chosen.tolist()]

        return SignalDraw(
            family=setting.family,
            beta=beta,
            active_groups=active_groups,
            energy_shares=energy_shares,
            support_fractions=support_fractions,
            concentrations=concentrations,
            group_signs=group_signs,
            support_indices=support_indices,
            acceptance_restarts=int(attempt),
        )

    raise RuntimeError(
        f"Failed to sample an acceptable blueprint for family {setting.family!r} "
        f"after {max_attempts} attempts."
    )
