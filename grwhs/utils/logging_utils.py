# grwhs/utils/logging_utils.py
from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

_HAS_RICH = False
try:  # Optional colored logging
    from rich.console import Console
    from rich.logging import RichHandler
    _HAS_RICH = True
except Exception:
    Console = None  # type: ignore

_HAS_TQDM = False
try:
    from tqdm import tqdm as _tqdm  # type: ignore
    _HAS_TQDM = True
except Exception:
    _tqdm = None  # type: ignore


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """Configure global logging; support rich formatting and optional file output."""
    logger = logging.getLogger("grwhs")
    logger.setLevel(level)
    logger.propagate = False  # Avoid duplicate handlers

    # Clear existing handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # Console handler
    if _HAS_RICH:
        rh = RichHandler(console=Console(), show_time=True, show_path=False, markup=True)
        rh.setLevel(level)
        logger.addHandler(rh)
    else:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s",
                                          datefmt="%H:%M:%S"))
        logger.addHandler(ch)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(fh)

    logger.debug("Logger initialized.")
    return logger


@dataclass
class Timer:
    """Context timer for measuring code block durations."""
    name: str = "task"
    logger: Optional[logging.Logger] = None
    start: float = 0.0
    elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.time()
        if self.logger:
            self.logger.debug(f"[{self.name}] started.")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed = time.time() - self.start
        if self.logger:
            if exc_type is None:
                self.logger.info(f"[{self.name}] finished in {self.elapsed:.3f}s.")
            else:
                self.logger.warning(f"[{self.name}] errored after {self.elapsed:.3f}s.")


def progress(iterable: Iterable, total: Optional[int] = None, desc: Optional[str] = None):
    """Wrap an iterable with a progress bar (prefer tqdm, otherwise return the iterable)."""
    if _HAS_TQDM:
        kwargs = {}
        if total is not None:
            kwargs["total"] = total
        if desc:
            kwargs["desc"] = desc
        return _tqdm(iterable, **kwargs)
    return iterable


def log_config(logger: logging.Logger, cfg: dict) -> None:
    """Log configuration entries in a hierarchical layout."""
    def _walk(d: dict, prefix=""):
        for k in sorted(d.keys()):
            v = d[k]
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                logger.info(f"{key}:")
                _walk(v, key)
            else:
                logger.info(f"{key}: {v!r}")
    logger.info("=== Effective Config ===")
    _walk(cfg)
    logger.info("========================")
