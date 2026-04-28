from __future__ import annotations

from typing import TYPE_CHECKING

from .config import load_mechanism_config

if TYPE_CHECKING:
    from .runner import run_mechanism

__all__ = ["load_mechanism_config", "run_mechanism"]


def __getattr__(name: str):
    if name == "run_mechanism":
        from .runner import run_mechanism

        return run_mechanism
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
