"""Core exports for the second-generation benchmark framework."""

from .blueprint import FAMILY_SPECS, family_spec_for_name, sample_signal_blueprint
from .dataset import generate_grouped_dataset, save_grouped_dataset
from .suite import build_main_suite, build_suite_manifest, get_setting_by_id

__all__ = [
    "FAMILY_SPECS",
    "build_main_suite",
    "build_suite_manifest",
    "family_spec_for_name",
    "generate_grouped_dataset",
    "get_setting_by_id",
    "sample_signal_blueprint",
    "save_grouped_dataset",
]

