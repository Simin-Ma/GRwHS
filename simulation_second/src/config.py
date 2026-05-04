from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from .blueprint import FAMILY_SPECS
from .schemas import (
    DEFAULT_METHOD_ROSTER,
    ConvergenceGateSpec,
    FamilySpec,
    SettingSpec,
)
from .suite import OPTIONAL_STRESS_LINES, TABLE_SPECS, build_main_suite
from .utils import MASTER_SEED


DEFAULT_PAIRING_METRICS = (
    "mse_null",
    "mse_signal",
    "mse_overall",
    "lpd_test",
)


def force_until_converged_gate(gate: ConvergenceGateSpec) -> ConvergenceGateSpec:
    """
    Package-level policy for simulation_second:
    keep Bayesian convergence enforcement enabled, but preserve any explicit
    retry budget chosen by the calling benchmark config.
    """
    return replace(
        gate,
        enforce_bayes_convergence=True,
    )


@dataclass(frozen=True)
class MethodRuntimeConfig:
    roster: tuple[str, ...] = DEFAULT_METHOD_ROSTER
    grrhs_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "tau_target": "groups",
            "progress_bar": False,
        }
    )
    gigg_config: dict[str, Any] = field(
        default_factory=lambda: {
            "allow_budget_retry": True,
            "extra_retry": 0,
            "no_retry": True,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "roster": list(self.roster),
            "grrhs_kwargs": dict(self.grrhs_kwargs),
            "gigg_config": dict(self.gigg_config),
        }


@dataclass(frozen=True)
class RunnerConfig:
    task: str = "gaussian"
    repeats: int = 10
    seed: int = MASTER_SEED
    n_jobs: int = 1
    method_jobs: int = 1
    output_dir: str = "outputs/history/simulation_second/benchmark_main"
    save_datasets: bool = True
    build_tables: bool = True
    baseline_method: str = "RHS"
    required_metrics_for_pairing: tuple[str, ...] = DEFAULT_PAIRING_METRICS

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": str(self.task),
            "repeats": int(self.repeats),
            "seed": int(self.seed),
            "n_jobs": int(self.n_jobs),
            "method_jobs": int(self.method_jobs),
            "output_dir": str(self.output_dir),
            "save_datasets": bool(self.save_datasets),
            "build_tables": bool(self.build_tables),
            "baseline_method": str(self.baseline_method),
            "required_metrics_for_pairing": list(self.required_metrics_for_pairing),
        }


@dataclass(frozen=True)
class BenchmarkConfig:
    package: str
    description: str
    convergence_gate: ConvergenceGateSpec
    families: dict[str, FamilySpec]
    methods: MethodRuntimeConfig
    runner: RunnerConfig
    tables: dict[str, dict[str, list[str]]]
    optional_stress_lines: list[dict[str, Any]]
    settings: tuple[SettingSpec, ...]

    def setting_map(self) -> dict[str, SettingSpec]:
        return {setting.setting_id: setting for setting in self.settings}

    def to_manifest(self) -> dict[str, Any]:
        return {
            "package": str(self.package),
            "description": str(self.description),
            "methods": list(self.methods.roster),
            "method_config": self.methods.to_dict(),
            "convergence_gate": self.convergence_gate.to_dict(),
            "signal_families": {name: spec.to_dict() for name, spec in self.families.items()},
            "runner": self.runner.to_dict(),
            "tables": {
                str(name): {
                    "rows": list(spec.get("rows", [])),
                    "columns": list(spec.get("columns", [])),
                }
                for name, spec in self.tables.items()
            },
            "optional_stress_lines": [dict(item) for item in self.optional_stress_lines],
            "settings": [setting.to_dict() for setting in self.settings],
        }


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "benchmark.yaml"


def family_spec_from_dict(name: str, payload: Mapping[str, Any]) -> FamilySpec:
    return FamilySpec(
        name=str(payload.get("name", name)),
        support_fraction_range=tuple(float(x) for x in payload["support_fraction_range"]),
        concentration_range=tuple(float(x) for x in payload["concentration_range"]),
        share_hyperparameters=bool(payload["share_hyperparameters"]),
        log_uniform_concentration=bool(payload.get("log_uniform_concentration", False)),
        acceptance_alpha_ratio_min=(
            None
            if payload.get("acceptance_alpha_ratio_min") is None
            else float(payload.get("acceptance_alpha_ratio_min"))
        ),
        acceptance_support_gap_min=(
            None
            if payload.get("acceptance_support_gap_min") is None
            else float(payload.get("acceptance_support_gap_min"))
        ),
        description=str(payload.get("description", "")),
    )


def setting_spec_from_dict(payload: Mapping[str, Any], *, default_methods: tuple[str, ...]) -> SettingSpec:
    methods = payload.get("methods", list(default_methods))
    return SettingSpec(
        setting_id=str(payload["setting_id"]),
        label=str(payload["label"]),
        family=str(payload["family"]),
        group_sizes=tuple(int(x) for x in payload["group_sizes"]),
        active_groups=tuple(int(x) for x in payload["active_groups"]),
        n_train=int(payload["n_train"]),
        n_test=int(payload.get("n_test", 100)),
        rho_within=float(payload.get("rho_within", 0.8)),
        rho_between=float(payload.get("rho_between", 0.2)),
        target_r2=float(payload.get("target_r2", 0.7)),
        role=str(payload.get("role", "")),
        notes=str(payload.get("notes", "")),
        suite=str(payload.get("suite", "main")),
        methods=tuple(str(item) for item in methods),
    )


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, Mapping):
        merged = {str(k): v for k, v in base.items()}
        for key, value in override.items():
            key_use = str(key)
            if key_use in merged:
                merged[key_use] = _deep_merge(merged[key_use], value)
            else:
                merged[key_use] = value
        return merged
    return override


def build_default_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        package="simulation_second",
        description="Second-generation benchmark suite built from the GR-RHS blueprint.",
        convergence_gate=force_until_converged_gate(ConvergenceGateSpec()),
        families={name: spec for name, spec in FAMILY_SPECS.items()},
        methods=MethodRuntimeConfig(),
        runner=RunnerConfig(),
        tables={name: {"rows": list(spec["rows"]), "columns": list(spec["columns"])} for name, spec in TABLE_SPECS.items()},
        optional_stress_lines=[dict(item) for item in OPTIONAL_STRESS_LINES],
        settings=tuple(build_main_suite()),
    )


def build_default_config_payload() -> dict[str, Any]:
    return build_default_config().to_manifest()


def benchmark_config_from_payload(payload: Mapping[str, Any]) -> BenchmarkConfig:
    methods_payload = dict(payload.get("method_config", {}))
    if isinstance(payload.get("methods"), list):
        methods_payload["roster"] = payload.get("methods", [])
    default_methods = tuple(str(item) for item in methods_payload.get("roster", list(DEFAULT_METHOD_ROSTER)))

    families_raw = payload.get("signal_families", payload.get("families", {}))
    families: dict[str, FamilySpec] = {}
    for name, spec_payload in families_raw.items():
        families[str(name)] = family_spec_from_dict(str(name), spec_payload)

    gate_payload = payload.get("convergence_gate", {})
    methods_cfg = MethodRuntimeConfig(
        roster=default_methods,
        grrhs_kwargs=dict(methods_payload.get("grrhs_kwargs", {"tau_target": "groups", "progress_bar": False})),
        gigg_config=dict(methods_payload.get("gigg_config", {"allow_budget_retry": True, "extra_retry": 0, "no_retry": True})),
    )
    runner_payload = payload.get("runner", {})
    settings_payload = payload.get("settings", [])

    return BenchmarkConfig(
        package=str(payload.get("package", "simulation_second")),
        description=str(payload.get("description", "")),
        convergence_gate=force_until_converged_gate(
            ConvergenceGateSpec(
                enforce_bayes_convergence=bool(
                    gate_payload.get("enforce_bayes_convergence", True)
                ),
                max_convergence_retries=int(gate_payload.get("max_convergence_retries", -1)),
                bayes_min_chains=int(gate_payload.get("bayes_min_chains", 4)),
                chains=int(gate_payload.get("chains", 4)),
                warmup=int(gate_payload.get("warmup", 250)),
                post_warmup_draws=int(gate_payload.get("post_warmup_draws", 250)),
                adapt_delta=float(gate_payload.get("adapt_delta", 0.90)),
                max_treedepth=int(gate_payload.get("max_treedepth", 12)),
                strict_adapt_delta=float(gate_payload.get("strict_adapt_delta", 0.95)),
                strict_max_treedepth=int(gate_payload.get("strict_max_treedepth", 14)),
                rhat_threshold=float(gate_payload.get("rhat_threshold", 1.01)),
                ess_threshold=float(gate_payload.get("ess_threshold", 200.0)),
                max_divergence_ratio=float(gate_payload.get("max_divergence_ratio", 0.01)),
            )
        ),
        families=families,
        methods=methods_cfg,
        runner=RunnerConfig(
            task=str(runner_payload.get("task", "gaussian")),
            repeats=int(runner_payload.get("repeats", 10)),
            seed=int(runner_payload.get("seed", MASTER_SEED)),
            n_jobs=int(runner_payload.get("n_jobs", 1)),
            method_jobs=int(runner_payload.get("method_jobs", 1)),
            output_dir=str(runner_payload.get("output_dir", "outputs/history/simulation_second/benchmark_main")),
            save_datasets=bool(runner_payload.get("save_datasets", True)),
            build_tables=bool(runner_payload.get("build_tables", True)),
            baseline_method=str(runner_payload.get("baseline_method", "RHS")),
            required_metrics_for_pairing=tuple(
                str(item) for item in runner_payload.get("required_metrics_for_pairing", list(DEFAULT_PAIRING_METRICS))
            ),
        ),
        tables={
            str(name): {
                "rows": [str(item) for item in spec.get("rows", [])],
                "columns": [str(item) for item in spec.get("columns", [])],
            }
            for name, spec in payload.get("tables", {}).items()
        },
        optional_stress_lines=[dict(item) for item in payload.get("optional_stress_lines", [])],
        settings=tuple(
            setting_spec_from_dict(item, default_methods=default_methods)
            for item in settings_payload
        ),
    )


def load_benchmark_config(path: str | Path | None = None) -> BenchmarkConfig:
    merged = build_default_config_payload()
    config_path: Path | None = None
    if path is None:
        maybe_default = default_config_path()
        if maybe_default.exists():
            config_path = maybe_default
    else:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Benchmark config not found: {config_path}")

    if config_path is not None:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        merged = _deep_merge(merged, payload)
    return benchmark_config_from_payload(merged)
