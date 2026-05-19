from __future__ import annotations

from typing import Sequence

import numpy as np

from simulation_second.src.bayes_kernel.core.models.baselines import RegularizedHorseshoeGibbs
from simulation_second.src.bayes_kernel.core.diagnostics.convergence import summarize_convergence

from .helpers import fit_error_result
from ...utils import FitResult, SamplerConfig, diagnostics_summary_for_method, rhs_style_tau0, timed_call


def _clone_numeric_dict(obj: dict[str, object] | None) -> dict[str, np.ndarray] | None:
    if not isinstance(obj, dict) or not obj:
        return None
    out: dict[str, np.ndarray] = {}
    for k, v in obj.items():
        if isinstance(v, np.ndarray):
            out[str(k)] = np.asarray(v, dtype=float).copy()
        elif isinstance(v, (list, tuple)):
            out[str(k)] = np.asarray(v, dtype=float).copy()
        elif isinstance(v, (float, int, np.floating, np.integer)):
            out[str(k)] = np.asarray(float(v), dtype=float)
    return out or None


def _resume_chain_states(retry_resume_payload: dict[str, object] | None) -> list[dict[str, object]] | None:
    if not isinstance(retry_resume_payload, dict):
        return None
    states = retry_resume_payload.get("chain_last_states")
    if not isinstance(states, list) or not states:
        return None
    out: list[dict[str, object]] = []
    for item in states:
        cloned = _clone_numeric_dict(item if isinstance(item, dict) else None)
        if cloned:
            out.append(cloned)
    return out or None


def _extract_retry_resume_payload(*, model: RegularizedHorseshoeGibbs) -> dict[str, object] | None:
    chain_last_states = getattr(model, "chain_last_states_", None)
    if not isinstance(chain_last_states, list) or not chain_last_states:
        return None
    payload_states: list[dict[str, object]] = []
    for st in chain_last_states:
        cloned = _clone_numeric_dict(st if isinstance(st, dict) else None)
        if cloned:
            payload_states.append(cloned)
    if not payload_states:
        return None
    return {"backend": "rhs_gibbs_woodbury", "chain_last_states": payload_states}


def _extract_draw_diag(beta_chains: np.ndarray) -> tuple[float, float, dict[str, object]]:
    conv = summarize_convergence({"beta": np.asarray(beta_chains, dtype=float)})
    beta_diag = dict(conv.get("beta", {}))
    return (
        float(beta_diag.get("rhat_max", float("nan"))),
        float(beta_diag.get("ess_min", float("nan"))),
        {"convergence_detail": conv},
    )


def _mean_within_abs_corr(X: np.ndarray, groups: Sequence[Sequence[int]]) -> float:
    X_arr = np.asarray(X, dtype=float)
    vals: list[float] = []
    for members in groups:
        idx = np.asarray(list(members), dtype=int).reshape(-1)
        if idx.size <= 1:
            continue
        block = X_arr[:, idx]
        corr = np.corrcoef(block, rowvar=False)
        corr = np.asarray(corr, dtype=float)
        if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
            continue
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        if not np.any(mask):
            continue
        vals.extend(np.abs(corr[mask]).reshape(-1).tolist())
    if not vals:
        return float("nan")
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _rhs_highdim_auto_continuation_schedule(
    X: np.ndarray,
    groups: Sequence[Sequence[int]],
) -> tuple[int, int, int]:
    within_corr = _mean_within_abs_corr(np.asarray(X, dtype=float), groups)
    if np.isfinite(within_corr) and within_corr >= 0.75:
        return 3, 250, 250
    return 3, 200, 200


def _run_rhs_highdim_continuation(
    *,
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Sequence[int]],
    seed: int,
    p0: int,
    sampler: SamplerConfig,
    progress_bar: bool,
    stage_kwargs: dict[str, object],
    rounds: int,
    warmup: int,
    draws: int,
    select_best_round: bool,
) -> tuple[np.ndarray, float, dict[str, object]]:
    chains = int(max(1, sampler.chains))
    rounds_use = int(max(1, rounds))
    warmup_use = int(max(0, warmup))
    draws_use = int(max(1, draws))
    payload: dict[str, object] | None = None
    chain_beta_rounds: list[list[np.ndarray]] = [[] for _ in range(chains)]
    history: list[dict[str, object]] = []
    total_runtime = 0.0
    best_round = 1
    best_rhat = float("inf")

    for round_idx in range(1, rounds_use + 1):
        stage_sampler = SamplerConfig(
            chains=int(sampler.chains),
            warmup=int(warmup_use if round_idx == 1 else 0),
            post_warmup_draws=int(draws_use),
            adapt_delta=float(sampler.adapt_delta),
            max_treedepth=int(sampler.max_treedepth),
            strict_adapt_delta=float(sampler.strict_adapt_delta),
            strict_max_treedepth=int(sampler.strict_max_treedepth),
            max_divergence_ratio=float(sampler.max_divergence_ratio),
            rhat_threshold=float(sampler.rhat_threshold),
            ess_threshold=float(sampler.ess_threshold),
        )
        res = fit_rhs_gibbs(
            X,
            y,
            groups,
            task="gaussian",
            seed=int(seed),
            p0=int(p0),
            sampler=stage_sampler,
            progress_bar=bool(progress_bar),
            method_name="RHS_HighDim",
            retry_resume_payload=payload,
            highdim_continuation_rounds=0,
            highdim_continuation_warmup=None,
            highdim_continuation_draws=None,
            highdim_select_best_round=False,
            enable_highdim_auto_continuation=False,
            **stage_kwargs,
        )
        if str(res.status).lower() != "ok" or res.beta_draws is None:
            raise RuntimeError(str(res.error or "RHS continuation round failed."))
        total_runtime += float(res.runtime_seconds)
        payload = dict(res.diagnostics or {}).get("retry_resume_payload") if isinstance(res.diagnostics, dict) else None
        draws_arr = np.asarray(res.beta_draws, dtype=float)
        if draws_arr.ndim != 3 or int(draws_arr.shape[0]) != chains:
            raise RuntimeError("Unexpected RHS continuation draw shape.")
        round_rec: dict[str, object] = {
            "round": int(round_idx),
            "runtime_seconds": float(res.runtime_seconds),
        }
        for chain_idx in range(chains):
            chain_beta_rounds[chain_idx].append(np.asarray(draws_arr[chain_idx], dtype=float))
        aligned = [np.concatenate(chain_beta_rounds[idx], axis=0) for idx in range(chains)]
        min_draws = min(arr.shape[0] for arr in aligned)
        beta_chains = np.stack([arr[:min_draws] for arr in aligned], axis=0)
        rhat_now, ess_now, detail_now = _extract_draw_diag(beta_chains)
        round_rec["merged_beta_diag"] = dict(detail_now.get("convergence_detail", {}).get("beta", {}))
        round_rec["merged_shape"] = [int(x) for x in beta_chains.shape]
        history.append(round_rec)
        if np.isfinite(rhat_now) and rhat_now < best_rhat:
            best_rhat = float(rhat_now)
            best_round = int(round_idx)

    rounds_keep = int(best_round if bool(select_best_round) else rounds_use)
    aligned = [np.concatenate(chain_beta_rounds[idx][:rounds_keep], axis=0) for idx in range(chains)]
    min_draws = min(arr.shape[0] for arr in aligned)
    beta_chains = np.stack([arr[:min_draws] for arr in aligned], axis=0)
    _, _, detail = _extract_draw_diag(beta_chains)
    return beta_chains, float(total_runtime), {
        "continuation_history": history,
        "best_round": int(best_round),
        "rounds_executed": int(rounds_use),
        "rounds_used_for_final_artifact": int(rounds_keep),
        "continuation_warmup": int(warmup_use),
        "continuation_draws": int(draws_use),
        **detail,
    }


def _build_rhs_gibbs(
    *,
    n: int,
    p: int,
    p0: int,
    sampler: SamplerConfig,
    progress_bar: bool,
    seed: int,
    lambda_active_fraction: float | None = None,
    lambda_active_min: int | None = None,
    lambda_full_refresh_every: int | None = None,
    lambda_selection_mode: str | None = None,
    lambda_random_fraction: float | None = None,
    lambda_warmup_full_refresh: bool | None = None,
    tau_refresh_after_local: bool | None = None,
    beta_refresh_after_hyper: bool | None = None,
    extra_beta_refreshes: int | None = None,
    extra_lambda_sweeps: int | None = None,
    slice_max_steps: int | None = None,
    init_dispersion: float | None = None,
    group_block_refresh_every: int | None = None,
    slice_width_log_sigma: float | None = None,
    slice_width_log_tau: float | None = None,
    slice_width_log_lambda: float | None = None,
    slice_width_log_caux: float | None = None,
    initial_chain_states: list[dict[str, object]] | None = None,
    resume_no_burnin: bool = False,
) -> RegularizedHorseshoeGibbs:
    highdim = bool(int(p) > int(n) and int(p) >= 150)
    active_fraction_use = (
        float(lambda_active_fraction)
        if lambda_active_fraction is not None
        else (1.0 if highdim else 0.25)
    )
    active_min_use = (
        int(lambda_active_min)
        if lambda_active_min is not None
        else (int(p) if highdim else 32)
    )
    full_refresh_use = (
        int(lambda_full_refresh_every)
        if lambda_full_refresh_every is not None
        else (1 if highdim else 8)
    )
    selection_mode_use = (
        str(lambda_selection_mode).strip().lower()
        if lambda_selection_mode is not None
        else "magnitude"
    )
    random_fraction_use = (
        float(lambda_random_fraction)
        if lambda_random_fraction is not None
        else 0.0
    )
    warmup_full_refresh_use = (
        bool(lambda_warmup_full_refresh)
        if lambda_warmup_full_refresh is not None
        else True
    )
    tau_refresh_use = (
        bool(tau_refresh_after_local)
        if tau_refresh_after_local is not None
        else (False if highdim else True)
    )
    beta_refresh_use = (
        bool(beta_refresh_after_hyper)
        if beta_refresh_after_hyper is not None
        else True
    )
    extra_beta_refreshes_use = (
        int(extra_beta_refreshes)
        if extra_beta_refreshes is not None
        else 0
    )
    extra_lambda_sweeps_use = (
        int(extra_lambda_sweeps)
        if extra_lambda_sweeps is not None
        else 0
    )
    slice_steps_use = (
        int(slice_max_steps)
        if slice_max_steps is not None
        else (160 if highdim else 200)
    )
    init_dispersion_use = (
        float(init_dispersion)
        if init_dispersion is not None
        else 0.0
    )
    group_block_refresh_use = (
        int(group_block_refresh_every)
        if group_block_refresh_every is not None
        else 0
    )
    width_log_sigma_use = (
        float(slice_width_log_sigma)
        if slice_width_log_sigma is not None
        else 0.35
    )
    width_log_tau_use = (
        float(slice_width_log_tau)
        if slice_width_log_tau is not None
        else 0.35
    )
    width_log_lambda_use = (
        float(slice_width_log_lambda)
        if slice_width_log_lambda is not None
        else 0.45
    )
    width_log_caux_use = (
        float(slice_width_log_caux)
        if slice_width_log_caux is not None
        else 0.45
    )
    return RegularizedHorseshoeGibbs(
        scale_global=float(rhs_style_tau0(n=int(n), p=int(p), p0=int(max(p0, 1)))),
        num_warmup=int(sampler.warmup),
        num_samples=int(sampler.post_warmup_draws),
        num_chains=int(sampler.chains),
        thinning=1,
        seed=int(seed),
        progress_bar=bool(progress_bar),
        slice_width_log_sigma=float(width_log_sigma_use),
        slice_width_log_tau=float(width_log_tau_use),
        slice_width_log_lambda=float(width_log_lambda_use),
        slice_width_log_caux=float(width_log_caux_use),
        lambda_active_fraction=float(active_fraction_use),
        lambda_active_min=int(active_min_use),
        lambda_full_refresh_every=int(full_refresh_use),
        lambda_selection_mode=str(selection_mode_use),
        lambda_random_fraction=float(random_fraction_use),
        lambda_warmup_full_refresh=bool(warmup_full_refresh_use),
        tau_refresh_after_local=bool(tau_refresh_use),
        beta_refresh_after_hyper=bool(beta_refresh_use),
        extra_beta_refreshes=int(extra_beta_refreshes_use),
        extra_lambda_sweeps=int(extra_lambda_sweeps_use),
        slice_max_steps=int(slice_steps_use),
        init_dispersion=float(init_dispersion_use),
        group_block_refresh_every=int(group_block_refresh_use),
        initial_chain_states=initial_chain_states,
        resume_no_burnin=bool(resume_no_burnin),
    )


def fit_rhs_gibbs(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    task: str,
    seed: int,
    p0: int,
    sampler: SamplerConfig,
    progress_bar: bool = True,
    method_name: str = "RHS_HighDim",
    lambda_active_fraction: float | None = None,
    lambda_active_min: int | None = None,
    lambda_full_refresh_every: int | None = None,
    lambda_selection_mode: str | None = None,
    lambda_random_fraction: float | None = None,
    lambda_warmup_full_refresh: bool | None = None,
    tau_refresh_after_local: bool | None = None,
    beta_refresh_after_hyper: bool | None = None,
    extra_beta_refreshes: int | None = None,
    extra_lambda_sweeps: int | None = None,
    slice_max_steps: int | None = None,
    init_dispersion: float | None = None,
    group_block_refresh_every: int | None = None,
    slice_width_log_sigma: float | None = None,
    slice_width_log_tau: float | None = None,
    slice_width_log_lambda: float | None = None,
    slice_width_log_caux: float | None = None,
    retry_resume_payload: dict[str, object] | None = None,
    highdim_continuation_rounds: int = 0,
    highdim_continuation_warmup: int | None = None,
    highdim_continuation_draws: int | None = None,
    highdim_select_best_round: bool = True,
    enable_highdim_auto_continuation: bool = True,
) -> FitResult:
    if str(task).strip().lower() != "gaussian":
        return fit_error_result(str(method_name), "NotImplementedError: RHS_HighDim currently supports Gaussian likelihood only.")

    tracked = ["beta", "tau", "lambda", "c", "sigma"]
    n, p = int(X.shape[0]), int(X.shape[1])
    highdim = bool(int(p) > int(n) and int(p) >= 150)

    try:
        stage_kwargs = {
            "lambda_active_fraction": lambda_active_fraction,
            "lambda_active_min": lambda_active_min,
            "lambda_full_refresh_every": lambda_full_refresh_every,
            "lambda_selection_mode": lambda_selection_mode,
            "lambda_random_fraction": lambda_random_fraction,
            "lambda_warmup_full_refresh": lambda_warmup_full_refresh,
            "tau_refresh_after_local": tau_refresh_after_local,
            "beta_refresh_after_hyper": beta_refresh_after_hyper,
            "extra_beta_refreshes": extra_beta_refreshes,
            "extra_lambda_sweeps": extra_lambda_sweeps,
            "slice_max_steps": slice_max_steps,
            "init_dispersion": init_dispersion,
            "group_block_refresh_every": group_block_refresh_every,
            "slice_width_log_sigma": slice_width_log_sigma,
            "slice_width_log_tau": slice_width_log_tau,
            "slice_width_log_lambda": slice_width_log_lambda,
            "slice_width_log_caux": slice_width_log_caux,
        }
        stage_kwargs = {k: v for k, v in stage_kwargs.items() if v is not None}
        rounds_auto = int(highdim_continuation_rounds)
        warmup_auto = highdim_continuation_warmup
        draws_auto = highdim_continuation_draws
        if bool(enable_highdim_auto_continuation) and highdim and int(rounds_auto) <= 0 and retry_resume_payload is None:
            rounds_auto, warm_auto_resolved, draws_auto_resolved = _rhs_highdim_auto_continuation_schedule(
                np.asarray(X, dtype=float),
                groups,
            )
            if warmup_auto is None:
                warmup_auto = int(warm_auto_resolved)
            if draws_auto is None:
                draws_auto = int(draws_auto_resolved)
        if highdim and int(rounds_auto) > 0 and retry_resume_payload is None:
            beta_chains, runtime, cont_info = _run_rhs_highdim_continuation(
                X=np.asarray(X, dtype=float),
                y=np.asarray(y, dtype=float),
                groups=groups,
                seed=int(seed),
                p0=int(p0),
                sampler=sampler,
                progress_bar=bool(progress_bar),
                stage_kwargs=stage_kwargs,
                rounds=int(rounds_auto),
                warmup=int(sampler.warmup if warmup_auto is None else warmup_auto),
                draws=int(sampler.post_warmup_draws if draws_auto is None else draws_auto),
                select_best_round=bool(highdim_select_best_round),
            )
            rhat_max, ess_min, detail = _extract_draw_diag(beta_chains)
            diagnostics = dict(detail)
            diagnostics["rhs_impl"] = "rhs_gibbs_woodbury"
            diagnostics["rhs_sampler_name"] = str(method_name)
            diagnostics["rhs_sampler_strategy"] = "high_dim"
            diagnostics["rhs_defaults"] = dict(stage_kwargs)
            diagnostics["rhs_defaults"]["highdim_continuation_rounds"] = int(rounds_auto)
            diagnostics["rhs_defaults"]["highdim_continuation_warmup"] = int(sampler.warmup if warmup_auto is None else warmup_auto)
            diagnostics["rhs_defaults"]["highdim_continuation_draws"] = int(sampler.post_warmup_draws if draws_auto is None else draws_auto)
            diagnostics["rhs_defaults"]["highdim_select_best_round"] = bool(highdim_select_best_round)
            diagnostics["staged_runtime"] = {"highdim_continuation": cont_info}
            return FitResult(
                method=str(method_name),
                status="ok",
                beta_mean=np.asarray(beta_chains, dtype=float).reshape(-1, beta_chains.shape[-1]).mean(axis=0),
                beta_draws=np.asarray(beta_chains, dtype=float),
                kappa_draws=None,
                group_scale_draws=None,
                tau_draws=None,
                runtime_seconds=float(runtime),
                rhat_max=float(rhat_max),
                bulk_ess_min=float(ess_min),
                divergence_ratio=float("nan"),
                converged=bool(np.isfinite(rhat_max) and rhat_max <= float(sampler.rhat_threshold) and np.isfinite(ess_min) and ess_min >= float(sampler.ess_threshold)),
                diagnostics=diagnostics,
            )

        initial_chain_states = _resume_chain_states(retry_resume_payload)
        model = _build_rhs_gibbs(
            n=n,
            p=p,
            p0=p0,
            sampler=sampler,
            progress_bar=bool(progress_bar),
            seed=seed,
            lambda_active_fraction=lambda_active_fraction,
            lambda_active_min=lambda_active_min,
            lambda_full_refresh_every=lambda_full_refresh_every,
            lambda_selection_mode=lambda_selection_mode,
            lambda_random_fraction=lambda_random_fraction,
            lambda_warmup_full_refresh=lambda_warmup_full_refresh,
            tau_refresh_after_local=tau_refresh_after_local,
            beta_refresh_after_hyper=beta_refresh_after_hyper,
            extra_beta_refreshes=extra_beta_refreshes,
            extra_lambda_sweeps=extra_lambda_sweeps,
            slice_max_steps=slice_max_steps,
            init_dispersion=init_dispersion,
            group_block_refresh_every=group_block_refresh_every,
            slice_width_log_sigma=slice_width_log_sigma,
            slice_width_log_tau=slice_width_log_tau,
            slice_width_log_lambda=slice_width_log_lambda,
            slice_width_log_caux=slice_width_log_caux,
            initial_chain_states=initial_chain_states,
            resume_no_burnin=bool(initial_chain_states),
        )
        model, runtime = timed_call(model.fit, X, y, groups=groups)
        beta_draws = getattr(model, "coef_samples_", None)
        beta_mean = getattr(model, "coef_", None)
        resume_payload_out = _extract_retry_resume_payload(model=model)

        rhat_max, ess_min, div_ratio, converged, details = diagnostics_summary_for_method(
            model=model,
            tracked_params=tracked,
            beta_draws=beta_draws,
            config=sampler,
        )
        details = dict(details or {})
        details["rhs_impl"] = "rhs_gibbs_woodbury"
        details["rhs_sampler_name"] = str(method_name)
        details["rhs_sampler_strategy"] = "high_dim"
        if resume_payload_out is not None:
            details["retry_resume_payload"] = resume_payload_out
        details["rhs_defaults"] = {
            "slab_df": float(model.slab_df),
            "slab_scale": float(model.slab_scale),
            "global_scale": float(model.scale_global),
            "lambda_active_fraction": float(model.lambda_active_fraction),
            "lambda_active_min": int(model.lambda_active_min),
            "lambda_full_refresh_every": int(model.lambda_full_refresh_every),
            "lambda_selection_mode": str(model.lambda_selection_mode),
            "lambda_random_fraction": float(model.lambda_random_fraction),
            "lambda_warmup_full_refresh": bool(model.lambda_warmup_full_refresh),
            "tau_refresh_after_local": bool(model.tau_refresh_after_local),
            "beta_refresh_after_hyper": bool(model.beta_refresh_after_hyper),
            "extra_beta_refreshes": int(model.extra_beta_refreshes),
            "extra_lambda_sweeps": int(model.extra_lambda_sweeps),
            "slice_max_steps": int(model.slice_max_steps),
            "slice_width_log_sigma": float(model.slice_width_log_sigma),
            "slice_width_log_tau": float(model.slice_width_log_tau),
            "slice_width_log_lambda": float(model.slice_width_log_lambda),
            "slice_width_log_caux": float(model.slice_width_log_caux),
            "init_dispersion": float(model.init_dispersion),
            "group_block_refresh_every": int(model.group_block_refresh_every),
        }

        return FitResult(
            method=str(method_name),
            status="ok",
            beta_mean=None if beta_mean is None else np.asarray(beta_mean, dtype=float),
            beta_draws=None if beta_draws is None else np.asarray(beta_draws, dtype=float),
            kappa_draws=None,
            group_scale_draws=None,
            tau_draws=None if getattr(model, "tau_samples_", None) is None else np.asarray(model.tau_samples_, dtype=float),
            runtime_seconds=float(runtime),
            rhat_max=float(rhat_max),
            bulk_ess_min=float(ess_min),
            divergence_ratio=float(div_ratio),
            converged=bool(converged),
            diagnostics=details,
        )
    except Exception as exc:
        return fit_error_result(str(method_name), f"{type(exc).__name__}: {exc}")

