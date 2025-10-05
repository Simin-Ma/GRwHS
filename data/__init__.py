"""Data subpackage: synthetic generators, preprocessing, splits, and loaders."""

from .generators import generate_synthetic, SyntheticConfig, SyntheticDataset, make_groups
from .preprocess import (
    center_y,
    standardize_X,
    apply_standardization,
    StandardizationConfig,
    StandardizeResult,
)
from .splits import train_val_test_split, repeated_splits, SplitResult
from .loaders import load_real_dataset, LoadedDataset, GroupMap
