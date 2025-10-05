"""Synthetic data generators for GRwHS experiments."""
from __future__ import annotations

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
    seed: Optional[int] = None
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
        if label != "equal":
            raise GeneratorError(f"Unsupported group_sizes specifier '{group_sizes}'.")
        if G is None or G <= 0:
            raise GeneratorError("Equal group sizing requires a positive G.")
        base = p // G
        remainder = p % G
        sizes = [base + (1 if g < remainder else 0) for g in range(G)]
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


def _draw_design(rng: np.random.Generator, n: int, p: int, corr_cfg: Mapping[str, object]) -> np.ndarray:
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

    raise GeneratorError(f"Unsupported correlation type '{corr_type}'.")


def _soft_sign(values: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    if mode == "positive":
        return np.abs(values)
    if mode == "negative":
        return -np.abs(values)
    signs = rng.choice([-1.0, 1.0], size=values.shape)
    return np.abs(values) * signs


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

    X = _draw_design(local_rng, config.n, config.p, config.correlation)
    X -= X.mean(axis=0, keepdims=True)

    signal_cfg = config.signal
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

    beta = np.zeros(config.p, dtype=float)
    sign_mode = str(signal_cfg.get("sign_mix", "random")).lower()
    scale_strong = float(signal_cfg.get("beta_scale_strong", 1.0))
    scale_weak = float(signal_cfg.get("beta_scale_weak", 0.5))

    if strong_idx.size:
        vals = local_rng.normal(0.0, scale_strong, size=strong_idx.size)
        beta[strong_idx] = _soft_sign(vals, sign_mode, local_rng)
    if weak_idx.size:
        vals = local_rng.normal(0.0, scale_weak, size=weak_idx.size)
        beta[weak_idx] = _soft_sign(vals, sign_mode, local_rng)

    noise = local_rng.normal(0.0, float(config.noise_sigma), size=config.n)
    y = X @ beta + noise

    info: Dict[str, object] = {
        "active_idx": np.sort(np.concatenate([strong_idx, weak_idx])) if (strong_idx.size or weak_idx.size) else np.empty(0, dtype=int),
        "strong_idx": strong_idx,
        "weak_idx": weak_idx,
        "seed": config.seed,
        "name": config.name,
    }

    return SyntheticDataset(
        X=X.astype(np.float32, copy=False),
        y=y.astype(np.float32, copy=False),
        beta=beta.astype(np.float32, copy=False),
        groups=groups,
        noise_sigma=float(config.noise_sigma),
        info=info,
    )


def synthetic_config_from_dict(
    data_cfg: Mapping[str, object],
    *,
    seed: Optional[int] = None,
    name: Optional[str] = None,
) -> SyntheticConfig:
    if "n" not in data_cfg or "p" not in data_cfg:
        raise KeyError("data configuration requires 'n' and 'p'.")

    group_sizes = data_cfg.get("group_sizes", "equal")
    correlation = data_cfg.get("correlation", {})
    signal = data_cfg.get("signal", {})
    noise_sigma = data_cfg.get("noise_sigma", 1.0)
    cfg_seed = data_cfg.get("seed", seed)
    G = data_cfg.get("G")

    return SyntheticConfig(
        n=int(data_cfg["n"]),
        p=int(data_cfg["p"]),
        G=None if G is None else int(G),
        group_sizes=group_sizes,
        correlation=dict(correlation),
        signal=dict(signal),
        noise_sigma=float(noise_sigma),
        seed=None if cfg_seed is None else int(cfg_seed),
        name=name or data_cfg.get("name"),
    )
