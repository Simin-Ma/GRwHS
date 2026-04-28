from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from simulation_second.src.schemas import ConvergenceGateSpec

from .schemas import DatasetSpec, MethodRuntimeConfig, RealDataConfig, RunnerConfig
from .utils import MASTER_SEED


def force_until_converged_gate(gate: ConvergenceGateSpec) -> ConvergenceGateSpec:
    """
    Package-level policy for real_data_experiment:
    all Bayesian methods are always run with convergence enforcement enabled and
    the retry budget set to the legacy "until converged" mode
    (`max_convergence_retries = -1` sentinel).
    """
    return replace(
        gate,
        enforce_bayes_convergence=True,
        max_convergence_retries=-1,
    )


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "real_data.yaml"


def dataset_spec_from_dict(payload: Mapping[str, Any], *, default_methods: tuple[str, ...]) -> DatasetSpec:
    return DatasetSpec(
        dataset_id=str(payload["dataset_id"]),
        label=str(payload.get("label", payload["dataset_id"])),
        description=str(payload.get("description", "")),
        loader=dict(payload.get("loader", {})),
        task=str(payload.get("task", "gaussian")),
        methods=tuple(str(item) for item in payload.get("methods", list(default_methods))),
        group_labels=tuple(str(item) for item in payload.get("group_labels", [])),
        target_label=str(payload.get("target_label", "")),
        target_transform=str(payload.get("target_transform", "none")),
        response_standardization=str(payload.get("response_standardization", "train_center")),
        covariate_mode=str(payload.get("covariate_mode", "none")),
        p0_strategy=str(payload.get("p0_strategy", "sqrt_p")),
        p0_override=None if payload.get("p0_override") is None else int(payload.get("p0_override")),
        p0_groups_strategy=str(payload.get("p0_groups_strategy", "half_groups")),
        p0_groups_override=(
            None if payload.get("p0_groups_override") is None else int(payload.get("p0_groups_override"))
        ),
        train_size=None if payload.get("train_size") is None else int(payload.get("train_size")),
        test_size=None if payload.get("test_size") is None else int(payload.get("test_size")),
        test_fraction=float(payload.get("test_fraction", 0.2)),
        repeats=int(payload.get("repeats", 10)),
        shuffle=bool(payload.get("shuffle", True)),
        notes=str(payload.get("notes", "")),
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


def build_default_config() -> RealDataConfig:
    methods = MethodRuntimeConfig()
    datasets = (
        DatasetSpec(
            dataset_id="nhanes_2003_2004",
            label="NHANES 2003-2004 GGT",
            description="Environmental exposure regression on NHANES complete-case adults.",
            loader={
                "path_X": "data/real/nhanes_2003_2004/processed/runner_ready/X.npy",
                "path_C": "data/real/nhanes_2003_2004/processed/runner_ready/C.npy",
                "path_y": "data/real/nhanes_2003_2004/processed/runner_ready/y.npy",
                "path_feature_names": "data/real/nhanes_2003_2004/processed/runner_ready/feature_names.txt",
                "path_covariate_feature_names": "data/real/nhanes_2003_2004/processed/runner_ready/covariate_feature_names.txt",
                "path_group_map": "data/real/nhanes_2003_2004/processed/runner_ready/group_map.json",
            },
            group_labels=("metals", "phthalates", "organochlorines", "pbdes", "pahs"),
            target_label="gamma_glutamyl_transferase",
            covariate_mode="residualize",
            p0_strategy="sqrt_p",
            p0_groups_strategy="half_groups",
            test_fraction=0.2,
            repeats=10,
            methods=methods.roster,
            notes="Uses covariate residualization so shrinkage comparisons focus on grouped exposures.",
        ),
        DatasetSpec(
            dataset_id="covid19_trust_experts",
            label="COVID-19 Trust Experts",
            description="Regression on the sparsegl trust_experts real-data design matrix.",
            loader={
                "path_X": "data/real/covid19_trust_experts/processed/runner_ready/X.npy",
                "path_y": "data/real/covid19_trust_experts/processed/runner_ready/y.npy",
                "path_feature_names": "data/real/covid19_trust_experts/processed/runner_ready/feature_names.txt",
                "path_group_map": "data/real/covid19_trust_experts/processed/runner_ready/group_map.json",
            },
            group_labels=(
                "period",
                "region",
                "age",
                "gender",
                "raceethnicity",
                "cli_spline",
                "hh_cmnty_cli_spline",
            ),
            target_label="trust_experts",
            covariate_mode="none",
            p0_strategy="sqrt_p",
            p0_groups_strategy="half_groups",
            test_fraction=0.2,
            repeats=10,
            methods=methods.roster,
            notes="Direct grouped-design comparison without extra covariates.",
        ),
        DatasetSpec(
            dataset_id="gse40279_age_gene_groups_smoke",
            label="GSE40279 Methylation Age (smoke)",
            description="Human methylation-age regression with disjoint single-gene proxy groups from 450k CpGs.",
            loader={
                "path_X": "data/real/gse40279_methylation_age/processed/runner_ready_smoke/X.npy",
                "path_y": "data/real/gse40279_methylation_age/processed/runner_ready_smoke/y.npy",
                "path_feature_names": "data/real/gse40279_methylation_age/processed/runner_ready_smoke/feature_names.txt",
                "path_group_map": "data/real/gse40279_methylation_age/processed/runner_ready_smoke/group_map.json",
                "path_group_labels": "data/real/gse40279_methylation_age/processed/runner_ready_smoke/group_labels.txt",
            },
            task="gaussian",
            methods=("GR_RHS", "RHS"),
            target_label="chronological_age",
            covariate_mode="none",
            response_standardization="train_center",
            p0_strategy="sqrt_p",
            p0_groups_strategy="half_groups",
            test_fraction=0.2,
            repeats=2,
            notes=(
                "Smoke runner uses top-variance CpGs grouped by a single-gene proxy derived from the "
                "Illumina 450k UCSC_RefGene_Name annotation."
            ),
        ),
        DatasetSpec(
            dataset_id="gse40279_age_gene_groups_micro",
            label="GSE40279 Methylation Age (micro)",
            description="Micro runner-verification subset from GSE40279 with the same grouping rule but 200 CpGs.",
            loader={
                "path_X": "data/real/gse40279_methylation_age/processed/runner_ready_micro/X.npy",
                "path_y": "data/real/gse40279_methylation_age/processed/runner_ready_micro/y.npy",
                "path_feature_names": "data/real/gse40279_methylation_age/processed/runner_ready_micro/feature_names.txt",
                "path_group_map": "data/real/gse40279_methylation_age/processed/runner_ready_micro/group_map.json",
                "path_group_labels": "data/real/gse40279_methylation_age/processed/runner_ready_micro/group_labels.txt",
            },
            task="gaussian",
            methods=("GR_RHS", "RHS"),
            target_label="chronological_age",
            covariate_mode="none",
            response_standardization="train_center",
            p0_strategy="sqrt_p",
            p0_groups_strategy="half_groups",
            test_fraction=0.2,
            repeats=1,
            notes=(
                "Micro runner is a lightweight real-data closure check derived from the same GSE40279 "
                "preprocessing pipeline as the 2000-feature smoke asset."
            ),
        ),
    )
    return RealDataConfig(
        package="real_data_experiment",
        description="Real-data comparison suite for GR-RHS and baseline methods.",
        convergence_gate=force_until_converged_gate(ConvergenceGateSpec()),
        methods=methods,
        runner=RunnerConfig(seed=MASTER_SEED),
        datasets=datasets,
    )


def build_default_config_payload() -> dict[str, Any]:
    return build_default_config().to_manifest()


def real_data_config_from_payload(payload: Mapping[str, Any]) -> RealDataConfig:
    methods_payload = dict(payload.get("methods", {}))
    default_methods = tuple(str(item) for item in methods_payload.get("roster", list(MethodRuntimeConfig().roster)))
    gate_payload = dict(payload.get("convergence_gate", {}))
    runner_payload = dict(payload.get("runner", {}))
    return RealDataConfig(
        package=str(payload.get("package", "real_data_experiment")),
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
        methods=MethodRuntimeConfig(
            roster=default_methods,
            grrhs_kwargs=dict(methods_payload.get("grrhs_kwargs", {"tau_target": "groups", "progress_bar": False})),
            gigg_config=dict(methods_payload.get("gigg_config", {"allow_budget_retry": True, "extra_retry": 0, "no_retry": True})),
        ),
        runner=RunnerConfig(
            output_dir=str(runner_payload.get("output_dir", "outputs/history/real_data_experiment/main")),
            seed=int(runner_payload.get("seed", MASTER_SEED)),
            n_jobs=int(runner_payload.get("n_jobs", 1)),
            method_jobs=int(runner_payload.get("method_jobs", 1)),
            build_tables=bool(runner_payload.get("build_tables", True)),
            save_splits=bool(runner_payload.get("save_splits", True)),
            baseline_method=str(runner_payload.get("baseline_method", "RHS")),
            required_metrics_for_pairing=tuple(
                str(item)
                for item in runner_payload.get(
                    "required_metrics_for_pairing",
                    ["rmse_test", "mae_test", "lpd_test", "r2_test"],
                )
            ),
        ),
        datasets=tuple(
            dataset_spec_from_dict(item, default_methods=default_methods)
            for item in payload.get("datasets", [])
        ),
    )


def load_real_data_config(path: str | Path | None = None) -> RealDataConfig:
    merged = build_default_config_payload()
    config_path: Path | None = None
    if path is None:
        maybe_default = default_config_path()
        if maybe_default.exists():
            config_path = maybe_default
    else:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Real-data config not found: {config_path}")
    if config_path is not None:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        merged = _deep_merge(merged, payload)
    return real_data_config_from_payload(merged)
