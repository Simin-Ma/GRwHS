"""Simulation suites for GR-RHS paper-style experiments."""

from .revision_suite import (
    DEFAULT_METHODS,
    PAPER_GAUSSIAN_METHODS,
    RevisionQuickProfile,
    profile_kappa_posterior_summary,
    run_revision_suite,
)

__all__ = [
    "DEFAULT_METHODS",
    "PAPER_GAUSSIAN_METHODS",
    "RevisionQuickProfile",
    "profile_kappa_posterior_summary",
    "run_revision_suite",
]
