from __future__ import annotations

import numpy as np
import pytest

from data.splits import OuterFold, outer_kfold_splits


def test_outer_kfold_splits_regression_basic():
    n = 30
    folds = outer_kfold_splits(
        n,
        n_splits=5,
        n_repeats=2,
        shuffle=True,
        seed=123,
    )
    assert len(folds) == 10
    hashes = {fold.hash for fold in folds}
    assert len(hashes) == len(folds)
    for fold in folds:
        assert isinstance(fold, OuterFold)
        assert fold.train.size + fold.test.size == n
        assert fold.train.size > 0 and fold.test.size > 0


def test_outer_kfold_splits_classification_stratified():
    n = 40
    y = np.array([0] * 18 + [1] * 12 + [2] * 10)
    folds = outer_kfold_splits(
        n,
        y=y,
        task="classification",
        n_splits=4,
        n_repeats=1,
        shuffle=True,
        seed=987,
    )
    assert len(folds) == 4
    for fold in folds:
        y_train = y[fold.train]
        y_test = y[fold.test]
        assert set(np.unique(y_train)) == {0, 1, 2}
        assert set(np.unique(y_test)) == {0, 1, 2}


def test_outer_kfold_invalid_split_raises():
    with pytest.raises(ValueError):
        outer_kfold_splits(3, n_splits=5)
