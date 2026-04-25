from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

import numpy as np


DEFAULT_STANDARD_METHODS: Tuple[str, ...] = ("GR_RHS", "RHS")
DEFAULT_ABLATION_VARIANTS: Tuple[str, ...] = (
    "GR_RHS",
    "GR_RHS_fixed_10x",
    "RHS_oracle",
    "GR_RHS_oracle",
    "GR_RHS_no_local_scales",
    "GR_RHS_shared_kappa",
    "GR_RHS_no_kappa",
)


@dataclass(frozen=True)
class ConvergenceGateSpec:
    enforce_bayes_convergence: bool = True
    max_convergence_retries: int = -1
    bayes_min_chains: int = 2
    chains: int = 2
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
class MechanismSettingSpec:
    setting_id: str
    setting_label: str
    experiment_id: str
    experiment_label: str
    experiment_kind: str
    line_id: str
    line_label: str
    scientific_question: str
    primary_metric: str
    group_sizes: Tuple[int, ...]
    active_groups: Tuple[int, ...] = ()
    n_train: int = 100
    n_test: int = 30
    rho_within: float = 0.8
    rho_between: float = 0.2
    target_snr: float = 1.0
    sigma2: float | None = None
    within_group_pattern: str = ""
    complexity_pattern: str = ""
    total_active_coeff: int = 0
    mu: Tuple[float, ...] = ()
    suite: str = "mechanism"
    role: str = ""
    notes: str = ""
    methods: Tuple[str, ...] = DEFAULT_STANDARD_METHODS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "setting_id": self.setting_id,
            "setting_label": self.setting_label,
            "experiment_id": self.experiment_id,
            "experiment_label": self.experiment_label,
            "experiment_kind": self.experiment_kind,
            "line_id": self.line_id,
            "line_label": self.line_label,
            "scientific_question": self.scientific_question,
            "primary_metric": self.primary_metric,
            "group_sizes": list(self.group_sizes),
            "active_groups": list(self.active_groups),
            "n_train": int(self.n_train),
            "n_test": int(self.n_test),
            "rho_within": float(self.rho_within),
            "rho_between": float(self.rho_between),
            "target_snr": float(self.target_snr),
            "sigma2": None if self.sigma2 is None else float(self.sigma2),
            "within_group_pattern": self.within_group_pattern,
            "complexity_pattern": self.complexity_pattern,
            "total_active_coeff": int(self.total_active_coeff),
            "mu": list(self.mu),
            "suite": self.suite,
            "role": self.role,
            "notes": self.notes,
            "methods": list(self.methods),
        }


@dataclass
class MechanismDataset:
    setting: MechanismSettingSpec
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    beta: np.ndarray
    sigma2: float
    cov_x: np.ndarray
    groups: list[list[int]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        out = dict(self.metadata)
        out["setting"] = self.setting.to_dict()
        out["groups"] = [[int(i) for i in group] for group in self.groups]
        out["sigma2"] = float(self.sigma2)
        out["beta_nonzero"] = int(np.count_nonzero(self.beta))
        out["train_shape"] = [int(x) for x in self.X_train.shape]
        out["test_shape"] = [int(x) for x in self.X_test.shape]
        return out


def active_group_mask(beta: np.ndarray, groups: Tuple[Tuple[int, ...], ...] | list[list[int]]) -> np.ndarray:
    beta_arr = np.asarray(beta, dtype=float).reshape(-1)
    mask = []
    for group in groups:
        idx = np.asarray(group, dtype=int)
        mask.append(bool(np.any(np.abs(beta_arr[idx]) > 1e-12)))
    return np.asarray(mask, dtype=bool)


def experiment_primary_metric_map(settings: Tuple[MechanismSettingSpec, ...]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for setting in settings:
        out[str(setting.experiment_id)] = str(setting.primary_metric)
    return out


def setting_question_map(settings: Tuple[MechanismSettingSpec, ...]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for setting in settings:
        out[str(setting.experiment_id)] = str(setting.scientific_question)
    return out
