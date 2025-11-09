"""Synthetic data generators for GRwHS experiments."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union

import math
import numpy as np

__all__ = [
    "SyntheticConfig",
    "SyntheticDataset",
    "make_groups",
    "generate_synthetic",
    "synthetic_config_from_dict",
    "register_generator",
    "SCENARIO_GENERATORS",
]


class GeneratorError(ValueError):
    """Raised when an invalid synthetic configuration is provided."""


@dataclass
class SyntheticConfig:
    """Container capturing all parameters for synthetic scenario generation."""

    n: int
    p: int
    G: Optional[int] = None
    group_sizes: Union[str, Sequence[int], None] = "equal"
    correlation: Mapping[str, object] = field(default_factory=dict)
    signal: Mapping[str, object] = field(default_factory=dict)
    noise_sigma: float = 1.0
    task: str = "regression"
    response: Mapping[str, object] = field(default_factory=dict)
    seed: Optional[int] = None
    overlap: Mapping[str, object] = field(default_factory=dict)
    name: Optional[str] = None


@dataclass
class SyntheticDataset:
    """Generated synthetic dataset together with metadata."""

    X: np.ndarray
    y: np.ndarray
    beta: np.ndarray
    groups: List[List[int]]
    noise_sigma: float
    info: Dict[str, object] = field(default_factory=dict)


SCENARIO_GENERATORS: Dict[str, Callable[[SyntheticConfig], SyntheticDataset]] = {}


def register_generator(name: str) -> Callable[[Callable[[SyntheticConfig], SyntheticDataset]], Callable[[SyntheticConfig], SyntheticDataset]]:
    """Decorator used by downstream modules to register named scenarios."""

    key = name.strip().lower()

    def decorator(func: Callable[[SyntheticConfig], SyntheticDataset]) -> Callable[[SyntheticConfig], SyntheticDataset]:
        SCENARIO_GENERATORS[key] = func
        return func

    return decorator


def make_groups(p: int, G: Optional[int], group_sizes: Union[str, Sequence[int], None]) -> List[List[int]]:
    """Construct contiguous feature groups according to configuration."""

    if group_sizes is None:
        return [[j] for j in range(p)]

    if isinstance(group_sizes, str):
        label = group_sizes.lower()
        if label == "equal":
            if G is None or G <= 0:
                raise GeneratorError("Equal group sizing requires a positive G.")
            base = p // G
            remainder = p % G
            sizes = [base + (1 if g < remainder else 0) for g in range(G)]
        elif label == "variable":
            if G is None or G <= 0:
                raise GeneratorError("Variable group sizing requires a positive G.")
            base = max(1, p // G)
            if base >= 12:
                min_size = max(4, min(int(math.floor(0.5 * base)), 8))
            else:
                min_size = max(2, int(math.floor(0.5 * base)))
            if base <= 20:
                max_size = max(min_size + 1, 20)
            else:
                max_size = max(min_size + 1, int(math.ceil(1.25 * base)))
            pattern = (-2, 1, 0, 2, -1, 3, -3)
            sizes = []
            for idx in range(G):
                size = base + pattern[idx % len(pattern)]
                size = max(min_size, min(max_size, size))
                sizes.append(size)
            adjust = p - sum(sizes)
            cursor = 0
            safeguard = G * 10 + abs(adjust)
            while adjust != 0 and safeguard > 0:
                j = cursor % G
                if adjust > 0 and sizes[j] < max_size:
                    sizes[j] += 1
                    adjust -= 1
                elif adjust < 0 and sizes[j] > min_size:
                    sizes[j] -= 1
                    adjust += 1
                cursor += 1
                safeguard -= 1
            if adjust != 0:
                raise GeneratorError("Failed to construct variable group sizes matching p.")
        else:
            raise GeneratorError(f"Unsupported group_sizes specifier '{group_sizes}'.")
    else:
        sizes = [int(s) for s in group_sizes]
        if any(s <= 0 for s in sizes):
            raise GeneratorError("Group sizes must all be positive integers.")
        if sum(sizes) != p:
            raise GeneratorError("Sum of group sizes must equal p.")
        if G is not None and len(sizes) != G:
            raise GeneratorError("Length of group_sizes does not match G.")

    groups: List[List[int]] = []
    cursor = 0
    for size in sizes:
        end = cursor + size
        groups.append(list(range(cursor, end)))
        cursor = end

    if cursor != p:
        raise GeneratorError("Group construction did not cover all features.")
    return groups


def _primary_group_index(groups: Sequence[Sequence[int]], p: int) -> np.ndarray:
    primary = np.full(p, -1, dtype=int)
    for gid, members in enumerate(groups):
        for feat in members:
            idx = int(feat)
            if idx < 0 or idx >= p:
                raise GeneratorError(f"Group index {idx} outside valid feature range [0, {p}).")
            if primary[idx] == -1:
                primary[idx] = gid
    if np.any(primary < 0):
        missing = np.nonzero(primary < 0)[0]
        raise GeneratorError(f"Some features lack a primary group assignment: {missing.tolist()}.")
    return primary


def _inject_group_overlap(
    groups: List[List[int]],
    p: int,
    overlap_cfg: Mapping[str, object],
    rng: np.random.Generator,
) -> Dict[str, object]:
    fraction = float(overlap_cfg.get("fraction", overlap_cfg.get("share", 0.0)))
    fraction = min(max(fraction, 0.0), 1.0)
    if fraction <= 0.0 or not groups:
        return {}

    max_memberships = overlap_cfg.get("max_memberships", overlap_cfg.get("copies", 2))
    try:
        max_memberships_int = int(max_memberships)
    except (TypeError, ValueError) as exc:
        raise GeneratorError("overlap.max_memberships must be an integer.") from exc
    max_memberships_int = max(2, max_memberships_int)

    eligible = np.arange(p, dtype=int)
    count_overlap = int(round(fraction * p))
    count_overlap = min(max(count_overlap, 0), p)
    if count_overlap == 0:
        return {}

    chosen = rng.choice(eligible, size=count_overlap, replace=False)
    membership_counts = np.zeros(p, dtype=int)
    for gid, members in enumerate(groups):
        for feat in members:
            membership_counts[int(feat)] += 1

    group_ids = list(range(len(groups)))
    for feat in chosen:
        current_count = int(membership_counts[feat])
        if current_count >= max_memberships_int:
            continue
        available = [g for g in group_ids if feat not in groups[g]]
        if not available:
            continue
        additional = max_memberships_int - current_count
        additional = min(additional, len(available))
        if additional <= 0:
            continue
        targets = rng.choice(available, size=additional, replace=False)
        for gid in targets:
            groups[gid].append(int(feat))
            membership_counts[feat] += 1

    overlap_features = np.nonzero(membership_counts > 1)[0]
    if overlap_features.size == 0:
        return {}
    memberships = membership_counts[overlap_features]
    return {
        "fraction": float(overlap_features.size / p),
        "feature_ids": overlap_features.astype(int),
        "membership_counts": memberships.astype(int),
    }


def _draw_design(
    rng: np.random.Generator,
    n: int,
    p: int,
    corr_cfg: Mapping[str, object],
    *,
    primary_group: Optional[np.ndarray] = None,
    group_count: Optional[int] = None,
) -> np.ndarray:
    corr_type = str(corr_cfg.get("type", "independent")).lower()
    rho = float(corr_cfg.get("rho", 0.0))

    if corr_type in {"independent", "none"} or abs(rho) < 1e-12:
        return rng.standard_normal((n, p))

    if corr_type == "block":
        block_size = corr_cfg.get("block_size")
        if block_size is None:
            raise GeneratorError("Block correlation requires 'block_size'.")
        block = int(block_size)
        if block <= 0:
            raise GeneratorError("block_size must be positive.")
        if not (0.0 <= rho < 1.0):
            raise GeneratorError("Block correlation requires rho in [0, 1).")
        design = np.empty((n, p), dtype=float)
        start = 0
        while start < p:
            end = min(start + block, p)
            width = end - start
            shared = rng.standard_normal((n, 1))
            noise = rng.standard_normal((n, width))
            design[:, start:end] = math.sqrt(rho) * shared + math.sqrt(1.0 - rho) * noise
            start = end
        return design

    if corr_type == "ar1":
        if not (-0.999 <= rho <= 0.999):
            raise GeneratorError("AR1 correlation requires rho in [-0.999, 0.999].")
        eps = rng.standard_normal((n, p))
        design = np.empty((n, p), dtype=float)
        design[:, 0] = eps[:, 0]
        scale = math.sqrt(max(1.0 - rho * rho, 1e-8))
        for j in range(1, p):
            design[:, j] = rho * design[:, j - 1] + scale * eps[:, j]
        return design

    if corr_type in {"cs", "compound_symmetry"}:
        if not (0.0 <= rho < 1.0):
            raise GeneratorError("Compound symmetry requires rho in [0, 1).")
        shared = rng.standard_normal((n, 1))
        noise = rng.standard_normal((n, p))
        return math.sqrt(rho) * shared + math.sqrt(1.0 - rho) * noise

    if corr_type in {"group", "group_block", "grouped"}:
        if primary_group is None or group_count is None:
            raise GeneratorError("Group-block correlation requires provided group assignments.")
        rho_in = float(corr_cfg.get("rho_in", rho))
        rho_out = float(corr_cfg.get("rho_out", 0.0))
        if not (0.0 <= rho_out < 1.0):
            raise GeneratorError("group_block correlation requires rho_out in [0, 1).")
        if not (0.0 <= rho_in < 1.0):
            raise GeneratorError("group_block correlation requires rho_in in [0, 1).")
        if rho_in < rho_out:
            raise GeneratorError("group_block correlation requires rho_in >= rho_out.")
        global_component = rng.standard_normal((n, 1)) if rho_out > 0 else None
        effective_in = max(rho_in - rho_out, 0.0)
        if effective_in > 0 and group_count <= 0:
            raise GeneratorError("group_block correlation needs positive group_count when rho_in > rho_out.")
        group_components = (
            rng.standard_normal((n, group_count))
            if (effective_in > 0 and group_count and group_count > 0)
            else None
        )
        noise = rng.standard_normal((n, p))
        design = np.empty((n, p), dtype=float)
        for j in range(p):
            base = 0.0
            if global_component is not None:
                base += math.sqrt(rho_out) * global_component[:, 0]
            if group_components is not None:
                gidx = int(primary_group[j])
                if gidx < 0 or gidx >= group_count:
                    raise GeneratorError(f"Primary group index {gidx} out of bounds for feature {j}.")
                base += math.sqrt(effective_in) * group_components[:, gidx]
            scale_noise = math.sqrt(max(1.0 - rho_in, 1e-8))
            design[:, j] = base + scale_noise * noise[:, j]
        return design

    raise GeneratorError(f"Unsupported correlation type '{corr_type}'.")


def _soft_sign(values: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    if mode == "positive":
        return np.abs(values)
    if mode == "negative":
        return -np.abs(values)
    signs = rng.choice([-1.0, 1.0], size=values.shape)
    return np.abs(values) * signs


def _blueprint_count(
    component: Mapping[str, object],
    available: int,
    rng: np.random.Generator,
) -> int:
    """Determine how many coefficients to draw for a blueprint component."""

    if available <= 0:
        return 0

    if "count_range" in component:
        low, high = component["count_range"]
        low = max(0, int(math.floor(float(low))))
        high = max(low, int(math.floor(float(high))))
        return int(min(available, rng.integers(low, high + 1)))

    if "count" in component:
        return int(min(available, max(0, int(component["count"]))))

    if "fraction_range" in component:
        low, high = component["fraction_range"]
        frac = rng.uniform(float(low), float(high))
        count = int(round(frac * available))
        return int(min(available, max(0, count)))

    if "fraction" in component:
        frac = float(component["fraction"])
        count = int(round(frac * available))
        return int(min(available, max(0, count)))

    raise GeneratorError("Signal blueprint component must define count/count_range/fraction.")


def _blueprint_draw_values(
    component: Mapping[str, object],
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw magnitude values for blueprint-assigned coefficients."""

    if count <= 0:
        return np.zeros(0, dtype=float)

    distribution = str(component.get("distribution", "") or ("constant" if "value" in component else "uniform")).lower()

    if distribution in {"constant", "fixed"}:
        value = float(component.get("value", component.get("magnitude", 1.0)))
        base = np.full(count, abs(value), dtype=float)
    elif distribution in {"uniform", "range"}:
        low = float(component.get("low", component.get("min", 0.0)))
        high = float(component.get("high", component.get("max", low)))
        if high < low:
            low, high = high, low
        base = rng.uniform(low, high, size=count)
    elif distribution in {"normal", "gaussian"}:
        mean = float(component.get("mean", 0.0))
        scale = float(component.get("scale", component.get("std", 1.0)))
        scale = max(scale, 1e-12)
        base = np.abs(rng.normal(mean, scale, size=count))
    else:
        raise GeneratorError(f"Unsupported blueprint distribution '{distribution}'.")

    sign_mode = str(component.get("sign", "mixed")).lower()
    return _soft_sign(base, sign_mode, rng)


def _apply_signal_blueprint(
    beta: np.ndarray,
    groups: Sequence[Sequence[int]],
    blueprint_cfg: object,
    rng: np.random.Generator,
) -> Dict[str, object]:
    """Deterministically assign coefficients according to a blueprint specification."""

    if isinstance(blueprint_cfg, Mapping):
        entries = [dict(blueprint_cfg)]
    elif isinstance(blueprint_cfg, Sequence):
        entries = [dict(entry) for entry in blueprint_cfg]  # type: ignore[arg-type]
    else:  # pragma: no cover - guarded via config validation
        raise GeneratorError("signal.blueprint must be a mapping or a list of mappings.")

    tag_records: Dict[str, List[int]] = defaultdict(list)
    assignment_records: List[Dict[str, object]] = []

    total_groups = len(groups)

    for entry_idx, entry in enumerate(entries):
        group_ids_raw = entry.get("groups")
        if not group_ids_raw:
            raise GeneratorError("Each signal.blueprint entry requires a non-empty 'groups' specification.")
        group_ids = [int(gid) for gid in group_ids_raw]
        for gid in group_ids:
            if gid < 0 or gid >= total_groups:
                raise GeneratorError(f"Blueprint references invalid group id {gid} (total groups: {total_groups}).")

        components = entry.get("components")
        if not components:
            continue

        for gid in group_ids:
            available = list(int(idx) for idx in groups[gid])
            for comp_idx, comp in enumerate(components):
                comp = dict(comp)
                count = _blueprint_count(comp, len(available), rng)
                if count <= 0:
                    continue
                candidates = np.array(available, dtype=int)
                chosen = rng.choice(candidates, size=count, replace=False)
                available = [idx for idx in available if idx not in chosen]
                values = _blueprint_draw_values(comp, count, rng)
                beta[chosen] = values

                tag = comp.get("tag")
                if tag:
                    tag_records[str(tag)].extend(int(idx) for idx in chosen)

                assignment_records.append(
                    {
                        "entry": str(entry.get("label", f"entry_{entry_idx}")),
                        "group": int(gid),
                        "component": str(comp.get("name", f"component_{comp_idx}")),
                        "indices": [int(i) for i in np.sort(chosen)],
                    }
                )

    active_idx = np.sort(np.flatnonzero(beta)).astype(int)
    summary_tags = {label: sorted({int(idx) for idx in indices}) for label, indices in tag_records.items()}
    return {
        "assignments": assignment_records,
        "active_idx": active_idx,
        "tags": summary_tags,
    }


def _choose_active_indices(
    rng: np.random.Generator,
    eligible: np.ndarray,
    p: int,
    signal_cfg: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    sparsity = float(signal_cfg.get("sparsity", 0.0))
    sparsity = min(max(sparsity, 0.0), 1.0)
    strong_frac = float(signal_cfg.get("strong_frac", 0.0))
    strong_frac = min(max(strong_frac, 0.0), 1.0)

    if eligible.size == 0 or sparsity <= 0.0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    num_active = int(round(sparsity * p))
    if num_active == 0:
        num_active = 1
    num_active = min(num_active, eligible.size)

    active = rng.choice(eligible, size=num_active, replace=False)
    if num_active == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    strong_count = int(round(strong_frac * num_active))
    strong_count = min(strong_count, num_active)
    if strong_count == 0:
        return np.empty(0, dtype=int), np.sort(active)

    strong_idx = rng.choice(active, size=strong_count, replace=False)
    weak_idx = np.setdiff1d(active, strong_idx, assume_unique=False)
    return np.sort(strong_idx), np.sort(weak_idx)


def generate_synthetic(config: SyntheticConfig, *, rng: Optional[np.random.Generator] = None) -> SyntheticDataset:
    if config.n <= 0 or config.p <= 0:
        raise GeneratorError("Both n and p must be positive.")

    local_rng = rng or np.random.default_rng(config.seed)
    groups = make_groups(config.p, config.G, config.group_sizes)
    overlap_info: Dict[str, object] = {}
    if config.overlap:
        groups = [list(block) for block in groups]
        overlap_info = _inject_group_overlap(groups, config.p, config.overlap, local_rng)

    primary_group = _primary_group_index(groups, config.p)

    X = _draw_design(
        local_rng,
        config.n,
        config.p,
        config.correlation,
        primary_group=primary_group,
        group_count=len(groups),
    )
    X -= X.mean(axis=0, keepdims=True)

    signal_cfg = config.signal
    blueprint_cfg = signal_cfg.get("blueprint")
    beta = np.zeros(config.p, dtype=float)
    blueprint_info: Optional[Dict[str, object]] = None

    if blueprint_cfg:
        blueprint_info = _apply_signal_blueprint(beta, groups, blueprint_cfg, local_rng)
        tags = blueprint_info.get("tags", {}) if blueprint_info else {}
        strong_idx = np.array(sorted({int(idx) for idx in tags.get("strong", [])}), dtype=int) if isinstance(tags, Mapping) and "strong" in tags else np.zeros(0, dtype=int)
        weak_idx = np.array(sorted({int(idx) for idx in tags.get("weak", [])}), dtype=int) if isinstance(tags, Mapping) and "weak" in tags else np.zeros(0, dtype=int)
        active_idx = np.asarray(blueprint_info.get("active_idx", np.zeros(0, dtype=int)), dtype=int) if blueprint_info else np.zeros(0, dtype=int)
    else:
        group_sparsity = signal_cfg.get("group_sparsity")
        if group_sparsity is not None:
            frac = min(max(float(group_sparsity), 0.0), 1.0)
            if groups:
                g_total = len(groups)
                g_keep = max(1, min(g_total, int(round(frac * g_total))))
                idx_groups = local_rng.choice(g_total, size=g_keep, replace=False)
                eligible = np.unique(np.concatenate([groups[g] for g in idx_groups]))
            else:
                eligible = np.arange(config.p)
        else:
            eligible = np.arange(config.p)

        strong_idx, weak_idx = _choose_active_indices(local_rng, eligible, config.p, signal_cfg)

        sign_mode = str(signal_cfg.get("sign_mix", "random")).lower()
        scale_strong = float(signal_cfg.get("beta_scale_strong", 1.0))
        scale_weak = float(signal_cfg.get("beta_scale_weak", 0.5))

        if strong_idx.size:
            vals = local_rng.normal(0.0, scale_strong, size=strong_idx.size)
            beta[strong_idx] = _soft_sign(vals, sign_mode, local_rng)
        if weak_idx.size:
            vals = local_rng.normal(0.0, scale_weak, size=weak_idx.size)
            beta[weak_idx] = _soft_sign(vals, sign_mode, local_rng)

        active_idx = np.sort(np.concatenate([strong_idx, weak_idx])) if (strong_idx.size or weak_idx.size) else np.zeros(0, dtype=int)

    linear = X @ beta

    response_cfg = dict(config.response) if config.response else {}
    response_type = str(response_cfg.get("type", config.task)).lower()

    if response_type not in {"regression", "classification"}:
        raise GeneratorError(f"Unsupported response type '{response_type}'.")

    if response_type == "classification":
        scale = float(response_cfg.get("scale", 1.0))
        bias = float(response_cfg.get("bias", 0.0))
        noise_std = float(response_cfg.get("noise_std", 0.0))

        logits = scale * linear + bias
        if noise_std > 0.0:
            logits = logits + local_rng.normal(0.0, noise_std, size=config.n)

        logits = np.clip(logits, -60.0, 60.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = local_rng.binomial(1, probs).astype(np.float32, copy=False)
        effective_noise_sigma = 0.0
    else:
        noise = local_rng.normal(0.0, float(config.noise_sigma), size=config.n)
        y = linear + noise
        probs = None
        effective_noise_sigma = float(config.noise_sigma)

    info: Dict[str, object] = {
        "active_idx": active_idx,
        "strong_idx": strong_idx,
        "weak_idx": weak_idx,
        "seed": config.seed,
        "name": config.name,
        "task": response_type,
        "primary_group": primary_group,
    }
    if probs is not None:
        info["mean_probability"] = float(np.mean(probs))
    if overlap_info:
        info["overlap"] = overlap_info
    if blueprint_info:
        info["signal_blueprint"] = blueprint_info

    return SyntheticDataset(
        X=X.astype(np.float32, copy=False),
        y=y.astype(np.float32, copy=False),
        beta=beta.astype(np.float32, copy=False),
        groups=groups,
        noise_sigma=effective_noise_sigma,
        info=info,
    )


def synthetic_config_from_dict(
    data_cfg: Mapping[str, object],
    *,
    seed: Optional[int] = None,
    name: Optional[str] = None,
    task: Optional[str] = None,
    response_override: Optional[Mapping[str, object]] = None,
) -> SyntheticConfig:
    if "n" not in data_cfg or "p" not in data_cfg:
        raise KeyError("data configuration requires 'n' and 'p'.")

    group_sizes = data_cfg.get("group_sizes", "equal")
    correlation = data_cfg.get("correlation", {})
    signal = data_cfg.get("signal", {})
    noise_sigma = data_cfg.get("noise_sigma", 1.0)
    cfg_seed = data_cfg.get("seed", seed)
    G = data_cfg.get("G")
    response_cfg_raw = data_cfg.get("response", {})
    if isinstance(response_cfg_raw, Mapping):
        response_cfg = dict(response_cfg_raw)
    else:
        response_cfg = {}
    if response_override is not None:
        response_cfg.update(dict(response_override))

    task_spec = task or response_cfg.get("type") or data_cfg.get("task")
    if task_spec is None:
        task_label = "regression"
    else:
        task_label = str(task_spec).lower()
    response_cfg.setdefault("type", task_label)

    return SyntheticConfig(
        n=int(data_cfg["n"]),
        p=int(data_cfg["p"]),
        G=None if G is None else int(G),
        group_sizes=group_sizes,
        correlation=dict(correlation),
        signal=dict(signal),
        noise_sigma=float(noise_sigma),
        task=task_label,
        response=response_cfg,
        seed=None if cfg_seed is None else int(cfg_seed),
        overlap=dict(data_cfg.get("overlap", {})),
        name=name or data_cfg.get("name"),
    )
