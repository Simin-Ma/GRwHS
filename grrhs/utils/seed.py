# grrhs/utils/seed.py
from __future__ import annotations

import os
import random as _random
from typing import Optional, Tuple

import numpy as np

_HAS_JAX = False
try:
    from jax import random as jrandom
    _HAS_JAX = True
except Exception:
    jrandom = None  # type: ignore

_HAS_TORCH = False
try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore


def seed_everything(seed: int = 42, deterministic_torch: bool = True) -> None:
    """Set numpy/python/jax/torch random seeds in a unified way."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    _random.seed(seed)
    np.random.seed(seed)

    if _HAS_JAX:
        # JAX requires managing PRNGKey at the call site; this is just a reminder
        pass

    if _HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False     # type: ignore[attr-defined]


def jax_keypair(seed: int = 42) -> Tuple["jrandom.PRNGKey", "jrandom.PRNGKey"]:
    """Create two JAX PRNGKeys to make splitting convenient."""
    if not _HAS_JAX:
        raise RuntimeError("JAX not available.")
    key = jrandom.PRNGKey(seed)
    return key, jrandom.split(key, 1)[0]


def jax_split(key: "jrandom.PRNGKey", num: int = 2):
    if not _HAS_JAX:
        raise RuntimeError("JAX not available.")
    return jrandom.split(key, num)
