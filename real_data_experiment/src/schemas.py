from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from simulation_project.src.utils import FitResult
from simulation_second.src.schemas import ConvergenceGateSpec, DEFAULT_METHOD_ROSTER


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    label: str
    description: str = ""
    loader: Dict[str, Any] = field(default_factory=dict)
    task: str = "gaussian"
    methods: Tuple[str, ...] = DEFAULT_METHOD_ROSTER
    group_labels: Tuple[str, ...] = ()
    target_label: str = ""
    target_transform: str = "none"
    response_standardization: str = "train_center"
    covariate_mode: str = "none"
    p0_strategy: str = "sqrt_p"
    p0_override: int | None = None
    p0_groups_strategy: str = "half_groups"
    p0_groups_override: int | None = None
    train_size: int | None = None
    test_size: int | None = None
    test_fraction: float = 0.2
    repeats: int = 10
    shuffle: bool = True
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "label": self.label,
            "description": self.description,
            "loader": dict(self.loader),
            "task": self.task,
            "methods": list(self.methods),
            "group_labels": list(self.group_labels),
            "target_label": self.target_label,
            "target_transform": self.target_transform,
            "response_standardization": self.response_standardization,
            "covariate_mode": self.covariate_mode,
            "p0_strategy": self.p0_strategy,
            "p0_override": self.p0_override,
            "p0_groups_strategy": self.p0_groups_strategy,
            "p0_groups_override": self.p0_groups_override,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "test_fraction": self.test_fraction,
            "repeats": self.repeats,
            "shuffle": self.shuffle,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class MethodRuntimeConfig:
    roster: Tuple[str, ...] = DEFAULT_METHOD_ROSTER
    grrhs_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "tau_target": "groups",
            "progress_bar": False,
        }
    )
    gigg_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "allow_budget_retry": True,
            "extra_retry": 0,
            "no_retry": True,
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "roster": list(self.roster),
            "grrhs_kwargs": dict(self.grrhs_kwargs),
            "gigg_config": dict(self.gigg_config),
        }


@dataclass(frozen=True)
class RunnerConfig:
    output_dir: str = "outputs/history/real_data_experiment/main"
    seed: int = 20260426
    n_jobs: int = 1
    method_jobs: int = 1
    build_tables: bool = True
    save_splits: bool = True
    baseline_method: str = "RHS"
    required_metrics_for_pairing: Tuple[str, ...] = ("rmse_test", "mae_test", "lpd_test", "r2_test")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "seed": self.seed,
            "n_jobs": self.n_jobs,
            "method_jobs": self.method_jobs,
            "build_tables": self.build_tables,
            "save_splits": self.save_splits,
            "baseline_method": self.baseline_method,
            "required_metrics_for_pairing": list(self.required_metrics_for_pairing),
        }


@dataclass(frozen=True)
class RealDataConfig:
    package: str
    description: str
    convergence_gate: ConvergenceGateSpec
    methods: MethodRuntimeConfig
    runner: RunnerConfig
    datasets: Tuple[DatasetSpec, ...]

    def dataset_map(self) -> Dict[str, DatasetSpec]:
        return {item.dataset_id: item for item in self.datasets}

    def to_manifest(self) -> Dict[str, Any]:
        return {
            "package": self.package,
            "description": self.description,
            "convergence_gate": self.convergence_gate.to_dict(),
            "methods": self.methods.to_dict(),
            "runner": self.runner.to_dict(),
            "datasets": [item.to_dict() for item in self.datasets],
        }


@dataclass
class PreparedRealDataset:
    dataset_spec: DatasetSpec
    dataset_id: str
    label: str
    X: np.ndarray
    y: np.ndarray
    groups: List[List[int]]
    group_labels: List[str]
    feature_names: List[str]
    covariates: np.ndarray | None = None
    covariate_feature_names: List[str] | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_summary(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "label": self.label,
            "shape_X": [int(v) for v in self.X.shape],
            "shape_C": None if self.covariates is None else [int(v) for v in self.covariates.shape],
            "group_sizes": [int(len(group)) for group in self.groups],
            "group_labels": list(self.group_labels),
            "feature_count": int(self.X.shape[1]),
            "sample_count": int(self.X.shape[0]),
            "metadata": dict(self.metadata),
        }


@dataclass
class PreparedSplit:
    dataset: PreparedRealDataset
    replicate_id: int
    seed: int
    split_hash: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    groups: List[List[int]]
    covariates_train: np.ndarray | None = None
    covariates_test: np.ndarray | None = None
    X_train_used: np.ndarray | None = None
    X_test_used: np.ndarray | None = None
    y_train_used: np.ndarray | None = None
    y_test_used: np.ndarray | None = None
    prediction_offset_train: np.ndarray | None = None
    prediction_offset_test: np.ndarray | None = None
    x_center: np.ndarray | None = None
    x_scale: np.ndarray | None = None
    y_offset: float = 0.0
    y_scale: float = 1.0
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RealDataMethodEvaluation:
    row: Mapping[str, Any]
    fit_result: FitResult
