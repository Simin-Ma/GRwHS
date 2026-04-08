"""MBSGS::BGLSS wrapper for benchmark integration.

This wrapper calls an R script via ``Rscript`` and exposes a sklearn-like API:
``fit`` / ``predict`` / ``coef_`` / ``intercept_``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Optional, Sequence

import numpy as np


GroupsLike = Sequence[Sequence[int]]


def _normalize_groups(groups: GroupsLike) -> list[list[int]]:
    if not groups:
        raise ValueError("At least one group must be specified.")
    normalized: list[list[int]] = []
    for idx, group in enumerate(groups):
        try:
            block = [int(member) for member in group]  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError(f"Group {idx} must be an iterable of indices.") from exc
        if not block:
            raise ValueError(f"Group {idx} is empty.")
        normalized.append(block)
    return normalized


def _check_groups_partition(groups: list[list[int]], n_features: int) -> None:
    seen = np.zeros(n_features, dtype=int)
    for g in groups:
        idx = np.asarray(g, dtype=int)
        if np.any(idx < 0) or np.any(idx >= n_features):
            raise ValueError("Group index out of bounds.")
        seen[idx] += 1
    if np.any(seen != 1):
        bad = np.where(seen != 1)[0].tolist()
        raise ValueError(f"Groups must form a partition of features; problematic indices: {bad[:20]}")


@dataclass
class MBSGSBGLSSRegression:
    groups: GroupsLike
    niter: int = 3000
    burnin: int = 1000
    seed: int = 2025
    fit_intercept: bool = False
    save_posterior_samples: bool = False
    rscript: str = "Rscript"
    timeout_sec: int = 1800
    verbose: bool = False

    def __post_init__(self) -> None:
        self.groups = _normalize_groups(self.groups)
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.beta_samples_: Optional[np.ndarray] = None

    def fit(self, X: Any, y: Any, **fit_kwargs: Any) -> "MBSGSBGLSSRegression":
        del fit_kwargs
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2D.")
        if y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("X and y shapes are inconsistent.")

        n, p = X_arr.shape
        _check_groups_partition(self.groups, p)
        # Reorder columns to contiguous group blocks for BGLSS(group_size=...).
        perm = np.concatenate([np.asarray(g, dtype=int) for g in self.groups], axis=0)
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(p)
        Xp = X_arr[:, perm]
        group_sizes = [len(g) for g in self.groups]

        if self.fit_intercept:
            x_mean = Xp.mean(axis=0)
            y_mean = float(y_arr.mean())
            X_work = Xp - x_mean
            y_work = y_arr - y_mean
        else:
            x_mean = np.zeros(p, dtype=float)
            y_mean = 0.0
            X_work = Xp
            y_work = y_arr

        rscript_bin = shutil.which(self.rscript) or self.rscript
        if shutil.which(rscript_bin) is None and not Path(rscript_bin).exists():
            raise RuntimeError(
                "Rscript executable was not found. Install R and ensure Rscript is in PATH, "
                "or set model.rscript to the absolute Rscript path."
            )

        driver = Path(__file__).with_name("r_bglss_driver.R")
        if not driver.exists():
            raise RuntimeError(f"Missing R driver script: {driver}")

        with tempfile.TemporaryDirectory(prefix="bglss_mbsgs_") as td:
            tdir = Path(td)
            x_path = tdir / "X.csv"
            y_path = tdir / "y.csv"
            g_path = tdir / "group_sizes.csv"
            coef_path = tdir / "coef.csv"
            samples_path = tdir / "beta_samples.csv"

            np.savetxt(x_path, X_work, delimiter=",")
            np.savetxt(y_path, y_work.reshape(-1, 1), delimiter=",")
            np.savetxt(g_path, np.asarray(group_sizes, dtype=int), delimiter=",", fmt="%d")

            cmd = [
                str(rscript_bin),
                str(driver),
                str(x_path),
                str(y_path),
                str(g_path),
                str(coef_path),
                str(samples_path),
                str(int(self.niter)),
                str(int(self.burnin)),
                str(int(self.seed)),
                "1" if self.save_posterior_samples else "0",
            ]
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=int(self.timeout_sec),
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    "MBSGS::BGLSS call failed.\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"stdout:\n{proc.stdout}\n"
                    f"stderr:\n{proc.stderr}"
                )

            coef_perm = np.loadtxt(coef_path, delimiter=",")
            coef_perm = np.asarray(coef_perm, dtype=float).reshape(-1)
            if coef_perm.shape[0] != p:
                raise RuntimeError(f"BGLSS returned coefficient length {coef_perm.shape[0]} but p={p}.")
            coef_orig = coef_perm[inv_perm]
            self.coef_ = coef_orig.copy()

            if self.fit_intercept:
                self.intercept_ = float(y_mean - x_mean @ coef_perm)
            else:
                self.intercept_ = 0.0

            if self.save_posterior_samples and samples_path.exists():
                draws = np.loadtxt(samples_path, delimiter=",")
                draws = np.atleast_2d(draws)
                # Convert back to original feature order.
                self.beta_samples_ = draws[:, inv_perm]
            else:
                self.beta_samples_ = None

        return self

    def predict(self, X: Any, **predict_kwargs: Any) -> np.ndarray:
        del predict_kwargs
        if self.coef_ is None:
            raise RuntimeError("Model must be fitted before prediction.")
        X_arr = np.asarray(X, dtype=float)
        return X_arr @ self.coef_ + float(self.intercept_)

    def get_posterior_summaries(self) -> dict[str, np.ndarray | float]:
        if self.coef_ is None:
            raise RuntimeError("Model must be fitted before requesting summaries.")
        out: dict[str, np.ndarray | float] = {
            "coef": self.coef_.copy(),
            "intercept": float(self.intercept_),
        }
        if self.beta_samples_ is not None:
            out["coef_sd"] = self.beta_samples_.std(axis=0)
        return out

