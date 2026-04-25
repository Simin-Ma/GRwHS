from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np

from simulation_project.src.experiments.fitting import (
    _fit_all_methods as legacy_fit_all_methods,
    _fit_with_convergence_retry,
)
from simulation_project.src.experiments.methods.fit_gr_rhs import fit_gr_rhs
from simulation_project.src.experiments.methods.fit_rhs import fit_rhs
from simulation_project.src.utils import FitResult, SamplerConfig, rhs_style_tau0

from .schemas import ConvergenceGateSpec, MechanismDataset, MechanismSettingSpec
from .utils import mechanism_method_family


def sampler_config_from_gate(gate: ConvergenceGateSpec) -> SamplerConfig:
    return SamplerConfig(
        chains=int(gate.chains),
        warmup=int(gate.warmup),
        post_warmup_draws=int(gate.post_warmup_draws),
        adapt_delta=float(gate.adapt_delta),
        max_treedepth=int(gate.max_treedepth),
        strict_adapt_delta=float(gate.strict_adapt_delta),
        strict_max_treedepth=int(gate.strict_max_treedepth),
        max_divergence_ratio=float(gate.max_divergence_ratio),
        rhat_threshold=float(gate.rhat_threshold),
        ess_threshold=float(gate.ess_threshold),
    )


def _active_group_count(beta: np.ndarray, groups: list[list[int]]) -> int:
    beta_arr = np.asarray(beta, dtype=float).reshape(-1)
    return int(
        sum(
            np.any(np.abs(beta_arr[np.asarray(group, dtype=int)]) > 1e-12)
            for group in groups
        )
    )


def oracle_tau0_for_method(
    method_name: str,
    *,
    dataset: MechanismDataset,
    grrhs_kwargs: dict[str, Any] | None,
    ablation_variant_specs: dict[str, dict[str, Any]] | None = None,
) -> float:
    p0_true = int(np.count_nonzero(np.asarray(dataset.beta, dtype=float)))
    p0_groups = int(_active_group_count(dataset.beta, dataset.groups))
    n = int(dataset.X_train.shape[0])
    p = int(dataset.X_train.shape[1])
    family = mechanism_method_family(method_name)
    if family == "RHS":
        return float(rhs_style_tau0(n=n, p=p, p0=max(p0_true, 1)))

    tau_target = str((grrhs_kwargs or {}).get("tau_target", "groups")).strip().lower()
    dim = len(dataset.groups) if tau_target == "groups" else p
    active_dim = p0_groups if tau_target == "groups" else p0_true
    return float(rhs_style_tau0(n=n, p=max(int(dim), 1), p0=max(int(active_dim), 1)))


def fit_setting_methods(
    dataset: MechanismDataset,
    setting: MechanismSettingSpec,
    *,
    task: str,
    gate: ConvergenceGateSpec,
    grrhs_kwargs: dict[str, Any] | None = None,
    gigg_config: dict[str, Any] | None = None,
    ablation_variant_specs: dict[str, dict[str, Any]] | None = None,
    method_jobs: int = 1,
) -> dict[str, FitResult]:
    methods = [str(item) for item in setting.methods]
    p0_true = int(np.count_nonzero(np.asarray(dataset.beta, dtype=float)))
    p0_groups = int(_active_group_count(dataset.beta, dataset.groups))
    sampler = sampler_config_from_gate(gate)

    if str(setting.experiment_kind).strip().lower() != "ablation":
        return legacy_fit_all_methods(
            dataset.X_train,
            dataset.y_train,
            dataset.groups,
            task=str(task),
            seed=int(dataset.metadata.get("seed", 0)) + 17,
            p0=p0_true,
            p0_groups=p0_groups,
            sampler=sampler,
            grrhs_kwargs=dict(grrhs_kwargs or {}),
            methods=methods,
            gigg_config=dict(gigg_config or {}),
            bayes_min_chains=int(gate.bayes_min_chains),
            enforce_bayes_convergence=bool(gate.enforce_bayes_convergence),
            max_convergence_retries=int(gate.max_convergence_retries),
            method_jobs=int(method_jobs),
        )

    specs = {str(name): dict(spec) for name, spec in (ablation_variant_specs or {}).items()}
    tau_target = str((grrhs_kwargs or {}).get("tau_target", "groups")).strip().lower()
    grrhs_p0 = p0_groups if tau_target == "groups" else p0_true

    def _fit_one(method_name: str) -> tuple[str, FitResult]:
        spec = dict(specs.get(method_name, {}))
        family = str(spec.get("method", mechanism_method_family(method_name)))
        if family == "RHS":
            result = _fit_with_convergence_retry(
                lambda st, att, _resume=None: fit_rhs(
                    dataset.X_train,
                    dataset.y_train,
                    dataset.groups,
                    task=str(task),
                    seed=int(dataset.metadata.get("seed", 0)) + 31 + 100 * int(att),
                    p0=int(p0_true),
                    sampler=st,
                ),
                method="RHS",
                sampler=sampler,
                bayes_min_chains=int(gate.bayes_min_chains),
                max_convergence_retries=int(gate.max_convergence_retries),
                enforce_bayes_convergence=bool(gate.enforce_bayes_convergence),
            )
            return method_name, result

        tau_mode = str(spec.get("tau_mode", "auto")).strip().lower()
        tau0_oracle = oracle_tau0_for_method(
            method_name,
            dataset=dataset,
            grrhs_kwargs=grrhs_kwargs,
            ablation_variant_specs=ablation_variant_specs,
        )
        tau0_use: float | None = None
        auto_calibrate = True
        if tau_mode == "oracle":
            auto_calibrate = False
            tau0_use = float(tau0_oracle)
        elif tau_mode == "oracle_x10":
            auto_calibrate = False
            tau0_use = float(tau0_oracle) * 10.0
        elif tau_mode == "fixed":
            auto_calibrate = False
            tau0_use = None if spec.get("tau0") is None else float(spec["tau0"])

        result = _fit_with_convergence_retry(
            lambda st, att, _resume=None: fit_gr_rhs(
                dataset.X_train,
                dataset.y_train,
                dataset.groups,
                task=str(task),
                seed=int(dataset.metadata.get("seed", 0)) + 41 + 100 * int(att),
                p0=int(grrhs_p0),
                sampler=st,
                alpha_kappa=float(spec.get("alpha_kappa", 0.5)),
                beta_kappa=float(spec.get("beta_kappa", 1.0)),
                use_local_scale=bool(spec.get("use_local_scale", True)),
                shared_kappa=bool(spec.get("shared_kappa", False)),
                auto_calibrate_tau=bool(auto_calibrate),
                tau0=tau0_use,
                tau_target=tau_target,
                progress_bar=bool((grrhs_kwargs or {}).get("progress_bar", False)),
                retry_resume_payload=_resume,
                retry_attempt=int(att),
            ),
            method="GR_RHS",
            sampler=sampler,
            bayes_min_chains=int(gate.bayes_min_chains),
            max_convergence_retries=int(gate.max_convergence_retries),
            enforce_bayes_convergence=bool(gate.enforce_bayes_convergence),
            continue_on_retry=True,
        )
        return method_name, result

    if int(method_jobs) <= 1 or len(methods) <= 1:
        return {name: _fit_one(name)[1] for name in methods}

    done: dict[str, FitResult] = {}
    workers = max(1, min(int(method_jobs), len(methods)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_map = {ex.submit(_fit_one, name): name for name in methods}
        for fut in as_completed(fut_map):
            key, result = fut.result()
            done[str(key)] = result
    return {name: done[name] for name in methods}
