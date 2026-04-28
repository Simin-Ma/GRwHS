from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from .schemas import (
    DEFAULT_ABLATION_VARIANTS,
    DEFAULT_STANDARD_METHODS,
    ConvergenceGateSpec,
    MechanismSettingSpec,
)
from .suite import build_mechanism_suite
from .utils import MASTER_SEED


DEFAULT_PAIRING_METRICS = (
    "group_auroc",
    "mse_null",
    "mse_signal",
    "mse_overall",
    "lpd_test",
)


def force_until_converged_gate(gate: ConvergenceGateSpec) -> ConvergenceGateSpec:
    """
    Package-level policy for simulation_mechanism:
    all Bayesian methods are always run with convergence enforcement enabled and
    the retry budget set to the "until converged" mode used by the legacy
    runtime (`max_convergence_retries = -1` sentinel).
    """
    return replace(
        gate,
        enforce_bayes_convergence=True,
        max_convergence_retries=-1,
    )


@dataclass(frozen=True)
class MethodRuntimeConfig:
    standard_methods: tuple[str, ...] = DEFAULT_STANDARD_METHODS
    ablation_variants: tuple[str, ...] = DEFAULT_ABLATION_VARIANTS
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
    ablation_variant_specs: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "GR_RHS": {
                "method": "GR_RHS",
                "tau_mode": "auto",
                "use_local_scale": True,
                "shared_kappa": False,
            },
            "GR_RHS_fixed_10x": {
                "method": "GR_RHS",
                "tau_mode": "oracle_x10",
                "use_local_scale": True,
                "shared_kappa": False,
            },
            "RHS_oracle": {
                "method": "RHS",
            },
            "GR_RHS_oracle": {
                "method": "GR_RHS",
                "tau_mode": "oracle",
                "use_local_scale": True,
                "shared_kappa": False,
            },
            "GR_RHS_no_local_scales": {
                "method": "GR_RHS",
                "tau_mode": "auto",
                "use_local_scale": False,
                "shared_kappa": False,
            },
            "GR_RHS_shared_kappa": {
                "method": "GR_RHS",
                "tau_mode": "auto",
                "use_local_scale": True,
                "shared_kappa": True,
            },
            "GR_RHS_no_kappa": {
                "method": "GR_RHS",
                "tau_mode": "auto",
                "use_local_scale": True,
                "shared_kappa": True,
                "alpha_kappa": 500.0,
                "beta_kappa": 500.0,
                "note": "approx_fixed_neutral_shared_kappa",
            },
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "standard_methods": list(self.standard_methods),
            "ablation_variants": list(self.ablation_variants),
            "grrhs_kwargs": dict(self.grrhs_kwargs),
            "gigg_config": dict(self.gigg_config),
            "ablation_variant_specs": {
                str(name): dict(spec) for name, spec in self.ablation_variant_specs.items()
            },
        }


@dataclass(frozen=True)
class RunnerConfig:
    task: str = "gaussian"
    repeats: int = 10
    seed: int = MASTER_SEED
    n_jobs: int = 1
    method_jobs: int = 1
    output_dir: str = "outputs/history/simulation_mechanism/mechanism_main"
    save_datasets: bool = True
    build_tables: bool = True
    baseline_method: str = "RHS"
    ablation_baseline_method: str = "GR_RHS"
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
            "ablation_baseline_method": str(self.ablation_baseline_method),
            "required_metrics_for_pairing": list(self.required_metrics_for_pairing),
        }


@dataclass(frozen=True)
class MechanismConfig:
    package: str
    description: str
    convergence_gate: ConvergenceGateSpec
    methods: MethodRuntimeConfig
    runner: RunnerConfig
    settings: tuple[MechanismSettingSpec, ...]
    include_dense_ablation: bool = False

    def setting_map(self) -> dict[str, MechanismSettingSpec]:
        return {setting.setting_id: setting for setting in self.settings}

    def to_manifest(self) -> dict[str, Any]:
        return {
            "package": str(self.package),
            "description": str(self.description),
            "convergence_gate": self.convergence_gate.to_dict(),
            "methods": self.methods.to_dict(),
            "runner": self.runner.to_dict(),
            "include_dense_ablation": bool(self.include_dense_ablation),
            "settings": [setting.to_dict() for setting in self.settings],
        }


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "mechanism.yaml"


def setting_spec_from_dict(
    payload: Mapping[str, Any],
    *,
    default_methods: tuple[str, ...],
    default_ablation_variants: tuple[str, ...],
) -> MechanismSettingSpec:
    experiment_kind = str(payload.get("experiment_kind", "group_separation"))
    default_method_list = default_ablation_variants if experiment_kind == "ablation" else default_methods
    methods = payload.get("methods", list(default_method_list))
    sigma2_raw = payload.get("sigma2")
    return MechanismSettingSpec(
        setting_id=str(payload["setting_id"]),
        setting_label=str(payload.get("setting_label", payload["setting_id"])),
        experiment_id=str(payload["experiment_id"]),
        experiment_label=str(payload["experiment_label"]),
        experiment_kind=experiment_kind,
        line_id=str(payload["line_id"]),
        line_label=str(payload["line_label"]),
        scientific_question=str(payload.get("scientific_question", "")),
        primary_metric=str(payload.get("primary_metric", "kappa_gap")),
        group_sizes=tuple(int(x) for x in payload["group_sizes"]),
        active_groups=tuple(int(x) for x in payload.get("active_groups", [])),
        n_train=int(payload.get("n_train", 100)),
        n_test=int(payload.get("n_test", 30)),
        rho_within=float(payload.get("rho_within", 0.8)),
        rho_between=float(payload.get("rho_between", 0.2)),
        target_snr=float(payload.get("target_snr", 1.0)),
        sigma2=None if sigma2_raw is None else float(sigma2_raw),
        within_group_pattern=str(payload.get("within_group_pattern", "")),
        complexity_pattern=str(payload.get("complexity_pattern", "")),
        total_active_coeff=int(payload.get("total_active_coeff", 0)),
        mu=tuple(float(x) for x in payload.get("mu", [])),
        suite=str(payload.get("suite", "mechanism")),
        role=str(payload.get("role", "")),
        notes=str(payload.get("notes", "")),
        include_in_paper_table=bool(payload.get("include_in_paper_table", True)),
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


def build_default_config() -> MechanismConfig:
    methods = MethodRuntimeConfig()
    return MechanismConfig(
        package="simulation_mechanism",
        description="Mechanism-first GR-RHS suite built from docs/grrhs_mechanism_experiment_design.md.",
        convergence_gate=force_until_converged_gate(ConvergenceGateSpec()),
        methods=methods,
        runner=RunnerConfig(),
        settings=build_mechanism_suite(
            standard_methods=methods.standard_methods,
            ablation_variants=methods.ablation_variants,
            include_dense_ablation=False,
        ),
        include_dense_ablation=False,
    )


def build_default_config_payload() -> dict[str, Any]:
    payload = build_default_config().to_manifest()
    payload.pop("settings", None)
    return payload


def mechanism_config_from_payload(payload: Mapping[str, Any]) -> MechanismConfig:
    include_dense_ablation = bool(payload.get("include_dense_ablation", False))
    methods_payload = dict(payload.get("methods", {}))
    methods_cfg = MethodRuntimeConfig(
        standard_methods=tuple(
            str(item) for item in methods_payload.get("standard_methods", list(DEFAULT_STANDARD_METHODS))
        ),
        ablation_variants=tuple(
            str(item) for item in methods_payload.get("ablation_variants", list(DEFAULT_ABLATION_VARIANTS))
        ),
        grrhs_kwargs=dict(methods_payload.get("grrhs_kwargs", {"tau_target": "groups", "progress_bar": False})),
        gigg_config=dict(
            methods_payload.get(
                "gigg_config",
                {"allow_budget_retry": True, "extra_retry": 0, "no_retry": True},
            )
        ),
        ablation_variant_specs={
            str(name): dict(spec)
            for name, spec in methods_payload.get("ablation_variant_specs", MethodRuntimeConfig().ablation_variant_specs).items()
        },
    )

    gate_payload = dict(payload.get("convergence_gate", {}))
    runner_payload = dict(payload.get("runner", {}))
    settings_payload = payload.get("settings")
    if settings_payload:
        settings = tuple(
            setting_spec_from_dict(
                item,
                default_methods=methods_cfg.standard_methods,
                default_ablation_variants=methods_cfg.ablation_variants,
            )
            for item in settings_payload
        )
    else:
        settings = build_mechanism_suite(
            standard_methods=methods_cfg.standard_methods,
            ablation_variants=methods_cfg.ablation_variants,
            include_dense_ablation=include_dense_ablation,
        )

    return MechanismConfig(
        package=str(payload.get("package", "simulation_mechanism")),
        description=str(payload.get("description", "")),
        convergence_gate=force_until_converged_gate(
            ConvergenceGateSpec(
                enforce_bayes_convergence=bool(gate_payload.get("enforce_bayes_convergence", True)),
                max_convergence_retries=int(gate_payload.get("max_convergence_retries", -1)),
                bayes_min_chains=int(gate_payload.get("bayes_min_chains", 2)),
                chains=int(gate_payload.get("chains", 2)),
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
        methods=methods_cfg,
        runner=RunnerConfig(
            task=str(runner_payload.get("task", "gaussian")),
            repeats=int(runner_payload.get("repeats", 10)),
            seed=int(runner_payload.get("seed", MASTER_SEED)),
            n_jobs=int(runner_payload.get("n_jobs", 1)),
            method_jobs=int(runner_payload.get("method_jobs", 1)),
            output_dir=str(
                runner_payload.get("output_dir", "outputs/history/simulation_mechanism/mechanism_main")
            ),
            save_datasets=bool(runner_payload.get("save_datasets", True)),
            build_tables=bool(runner_payload.get("build_tables", True)),
            baseline_method=str(runner_payload.get("baseline_method", "RHS")),
            ablation_baseline_method=str(runner_payload.get("ablation_baseline_method", "GR_RHS")),
            required_metrics_for_pairing=tuple(
                str(item)
                for item in runner_payload.get(
                    "required_metrics_for_pairing",
                    list(DEFAULT_PAIRING_METRICS),
                )
            ),
        ),
        settings=settings,
        include_dense_ablation=include_dense_ablation,
    )


def load_mechanism_config(
    path: str | Path | None = None,
    *,
    include_dense_ablation: bool | None = None,
) -> MechanismConfig:
    merged = build_default_config_payload()
    config_path: Path | None = None
    if path is None:
        maybe_default = default_config_path()
        if maybe_default.exists():
            config_path = maybe_default
    else:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Mechanism config not found: {config_path}")

    if config_path is not None:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        merged = _deep_merge(merged, payload)
    if include_dense_ablation is not None:
        merged["include_dense_ablation"] = bool(include_dense_ablation)
    return mechanism_config_from_payload(merged)
