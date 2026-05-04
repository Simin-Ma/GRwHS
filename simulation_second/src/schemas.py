from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np


DEFAULT_METHOD_ROSTER: Tuple[str, ...] = (
    "GR_RHS",
    "RHS",
    "GHS_plus",
    "GIGG_MMLE",
    "LASSO_CV",
    "OLS",
)

PRIMARY_METRICS: Tuple[str, ...] = ("mse_overall", "mse_signal", "mse_null")
UNCERTAINTY_METRICS: Tuple[str, ...] = ("coverage_95", "avg_ci_length")
PREDICTIVE_METRICS: Tuple[str, ...] = ("lpd_test",)
OPERATIONAL_METRICS: Tuple[str, ...] = ("runtime_mean", "n_runs", "n_ok", "n_converged")


@dataclass(frozen=True)
class ConvergenceGateSpec:
    enforce_bayes_convergence: bool = True
    max_convergence_retries: int = -1
    bayes_min_chains: int = 4
    chains: int = 4
    warmup: int = 250
    post_warmup_draws: int = 250
    adapt_delta: float = 0.90
    max_treedepth: int = 12
    strict_adapt_delta: float = 0.95
    strict_max_treedepth: int = 14
    rhat_threshold: float = 1.01
    ess_threshold: float = 200.0
    max_divergence_ratio: float = 0.01

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class FamilySpec:
    name: str
    support_fraction_range: Tuple[float, float]
    concentration_range: Tuple[float, float]
    share_hyperparameters: bool
    log_uniform_concentration: bool = False
    acceptance_alpha_ratio_min: float | None = None
    acceptance_support_gap_min: float | None = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class SettingSpec:
    setting_id: str
    label: str
    family: str
    group_sizes: Tuple[int, ...]
    active_groups: Tuple[int, ...]
    n_train: int
    n_test: int = 100
    rho_within: float = 0.8
    rho_between: float = 0.2
    target_r2: float = 0.7
    role: str = ""
    notes: str = ""
    suite: str = "main"
    methods: Tuple[str, ...] = DEFAULT_METHOD_ROSTER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "setting_id": self.setting_id,
            "label": self.label,
            "family": self.family,
            "group_sizes": list(self.group_sizes),
            "active_groups": list(self.active_groups),
            "n_train": int(self.n_train),
            "n_test": int(self.n_test),
            "rho_within": float(self.rho_within),
            "rho_between": float(self.rho_between),
            "target_r2": float(self.target_r2),
            "role": self.role,
            "notes": self.notes,
            "suite": self.suite,
            "methods": list(self.methods),
        }


@dataclass
class SignalDraw:
    family: str
    beta: np.ndarray
    active_groups: Tuple[int, ...]
    energy_shares: Dict[int, float]
    support_fractions: Dict[int, float]
    concentrations: Dict[int, float]
    group_signs: Dict[int, int]
    support_indices: Dict[int, List[int]]
    acceptance_restarts: int = 0

    def metadata(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "active_groups": [int(g) for g in self.active_groups],
            "energy_shares": {str(k): float(v) for k, v in self.energy_shares.items()},
            "support_fractions": {str(k): float(v) for k, v in self.support_fractions.items()},
            "concentrations": {str(k): float(v) for k, v in self.concentrations.items()},
            "group_signs": {str(k): int(v) for k, v in self.group_signs.items()},
            "support_indices": {str(k): [int(i) for i in v] for k, v in self.support_indices.items()},
            "acceptance_restarts": int(self.acceptance_restarts),
        }


@dataclass
class GroupedRegressionDataset:
    setting: SettingSpec
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    beta: np.ndarray
    sigma2: float
    cov_x: np.ndarray
    groups: List[List[int]]
    signal_draw: SignalDraw
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        out = dict(self.metadata)
        out["setting"] = self.setting.to_dict()
        out["groups"] = [[int(i) for i in group] for group in self.groups]
        out["sigma2"] = float(self.sigma2)
        out["beta_nonzero"] = int(np.count_nonzero(self.beta))
        out["signal_draw"] = self.signal_draw.metadata()
        out["train_shape"] = [int(x) for x in self.X_train.shape]
        out["test_shape"] = [int(x) for x in self.X_test.shape]
        return out
