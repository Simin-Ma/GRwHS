"""Dataset splitting helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.model_selection import train_test_split

__all__ = [
    "SplitResult",
    "train_val_test_split",
    "repeated_splits",
]


@dataclass
class SplitResult:
    """Indices for train/validation/test splits."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"train": self.train, "val": self.val, "test": self.test}


def train_val_test_split(
    n: int,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: Optional[int] = None,
) -> SplitResult:
    """Return shuffled indices for train/val/test subsets using sklearn helpers."""

    if n <= 0:
        raise ValueError("n must be positive.")
    if not (0.0 <= val_ratio < 1.0) or not (0.0 <= test_ratio < 1.0):
        raise ValueError("val_ratio and test_ratio must lie in [0, 1).")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("Validation + test ratios must sum to < 1.")

    indices = np.arange(n, dtype=int)

    test_size = int(round(test_ratio * n))
    test_size = min(max(test_size, 0), n)
    val_size = int(round(val_ratio * n))
    val_size = min(max(val_size, 0), n - test_size)

    if test_size == 0:
        train_val = indices
        test_idx = np.empty(0, dtype=int)
    else:
        train_val, test_idx = train_test_split(
            indices,
            test_size=test_size,
            shuffle=True,
            random_state=seed,
        )

    if val_size == 0:
        val_idx = np.empty(0, dtype=int)
        train_idx = np.asarray(train_val, dtype=int)
    else:
        if train_val.shape[0] <= val_size:
            raise ValueError("Validation split is empty; adjust ratios.")
        train_idx, val_idx = train_test_split(
            np.asarray(train_val, dtype=int),
            test_size=val_size,
            shuffle=True,
            random_state=None if seed is None else seed + 1,
        )

    if np.asarray(train_idx).size == 0:
        raise ValueError("Train split is empty; adjust ratios.")

    return SplitResult(
        train=np.asarray(train_idx, dtype=int),
        val=np.asarray(val_idx, dtype=int),
        test=np.asarray(test_idx, dtype=int),
    )


def repeated_splits(
    n: int,
    repeats: int,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: Optional[int] = None,
) -> List[SplitResult]:
    """Generate multiple independent train/val/test splits."""

    if repeats <= 0:
        raise ValueError("repeats must be positive.")

    rng = np.random.default_rng(seed)
    splits: List[SplitResult] = []
    for _ in range(repeats):
        split_seed = None if seed is None else int(rng.integers(0, 2**32 - 1))
        splits.append(
            train_val_test_split(
                n=n,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=split_seed,
            )
        )
    return splits
