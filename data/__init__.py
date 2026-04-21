"""Data subpackage: synthetic generators, preprocessing, splits, and loaders."""

from .generators import (
    SyntheticConfig as SyntheticConfig,
    SyntheticDataset as SyntheticDataset,
    generate_synthetic as generate_synthetic,
    make_groups as make_groups,
)
from .preprocess import (
    StandardizationConfig as StandardizationConfig,
    StandardizeResult as StandardizeResult,
    apply_standardization as apply_standardization,
    center_y as center_y,
    standardize_X as standardize_X,
)
from .splits import (
    SplitResult as SplitResult,
    repeated_splits as repeated_splits,
    train_val_test_split as train_val_test_split,
)
from .loaders import GroupMap as GroupMap, LoadedDataset as LoadedDataset, load_real_dataset as load_real_dataset
