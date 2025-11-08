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
    "OuterFold",
    "outer_kfold_splits",
]


@dataclass
class SplitResult:
    """Indices for train/validation/test splits."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"train": self.train, "val": self.val, "test": self.test}


@dataclass(frozen=True)
class OuterFold:
    """Indices for an outer CV fold along with bookkeeping metadata."""

    repeat: int
    fold: int
    train: np.ndarray
    test: np.ndarray
    hash: str
    seed: Optional[int] = None

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"train": self.train, "test": self.test}


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


def _fold_hash(train_idx: np.ndarray, test_idx: np.ndarray) -> str:
    """Stable hash representing a split configuration."""
    import hashlib

    delimiter = np.array([-1], dtype=int)
    payload = np.concatenate([np.sort(train_idx), delimiter, np.sort(test_idx)])
    return hashlib.sha1(payload.tobytes()).hexdigest()


def outer_kfold_splits(
    n: int,
    *,
    y: Optional[np.ndarray] = None,
    task: str = "regression",
    n_splits: int = 5,
    n_repeats: int = 1,
    shuffle: bool = True,
    seed: Optional[int] = None,
    stratify: Optional[bool] = None,
) -> List[OuterFold]:
    """
    Generate outer cross-validation folds with optional stratification.

    Args:
        n: Total number of samples.
        y: Optional target array used for stratification.
        task: "regression" or "classification" (controls default stratification).
        n_splits: Number of folds in each repetition.
        n_repeats: Number of independent repetitions.
        shuffle: Whether to shuffle before splitting (recommended).
        seed: Random seed controlling shuffles/repetitions.
        stratify: Force or disable stratification (defaults to classification=True).

    Returns:
        List of OuterFold objects covering all outer folds.
    """

    if n <= 0:
        raise ValueError("n must be positive for outer_kfold_splits.")
    if n_splits < 2:
        raise ValueError("outer_kfold_splits requires n_splits >= 2.")
    if n_splits > n:
        raise ValueError("n_splits cannot exceed number of samples.")
    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive.")

    task_label = str(task).lower()
    if stratify is None:
        stratify_flag = task_label == "classification"
    else:
        stratify_flag = bool(stratify)

    y_array: Optional[np.ndarray]
    if stratify_flag:
        if y is None:
            raise ValueError("Stratified outer splits require target array 'y'.")
        y_array = np.asarray(y, dtype=int)
        unique, counts = np.unique(y_array, return_counts=True)
        if unique.size < 2:
            stratify_flag = False
        elif np.any(counts < 2):
            stratify_flag = False
    else:
        y_array = None

    splits: List[OuterFold] = []
    rng = np.random.default_rng(seed)

    for repeat_idx in range(n_repeats):
        if shuffle:
            split_seed = None if seed is None else int(rng.integers(0, 2**32 - 1))
        else:
            split_seed = None if seed is None else seed + repeat_idx

        if stratify_flag and y_array is not None:
            from sklearn.model_selection import StratifiedKFold

            splitter = StratifiedKFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=split_seed,
            )
            iterator = splitter.split(np.zeros(n, dtype=int), y_array)
        else:
            from sklearn.model_selection import KFold

            splitter = KFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=split_seed,
            )
            iterator = splitter.split(np.zeros(n, dtype=int))

        for fold_idx, (train_idx, test_idx) in enumerate(iterator, start=1):
            train_arr = np.asarray(train_idx, dtype=int)
            test_arr = np.asarray(test_idx, dtype=int)
            if train_arr.size == 0 or test_arr.size == 0:
                raise ValueError("Outer CV fold produced empty train/test splits.")
            fold = OuterFold(
                repeat=repeat_idx + 1,
                fold=fold_idx,
                train=train_arr,
                test=test_arr,
                hash=_fold_hash(train_arr, test_arr),
                seed=None if split_seed is None else int(split_seed),
            )
            splits.append(fold)

    return splits
