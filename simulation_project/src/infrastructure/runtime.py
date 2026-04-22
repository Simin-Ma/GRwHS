from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Sequence

from tqdm.auto import tqdm

from ..core.utils.parallel_runtime import can_use_process_pool
from ..utils import FitResult, SamplerConfig

# ---------------------------------------------------------------------------
# Method lists
# ---------------------------------------------------------------------------
METHODS = ["GR_RHS", "RHS", "GIGG_MMLE", "GIGG_b_small", "GIGG_GHS", "GIGG_b_large", "GHS_plus", "OLS", "LASSO_CV"]
LAPTOP_METHODS = ["GR_RHS", "RHS", "GIGG_MMLE", "GHS_plus", "OLS", "LASSO_CV"]
COMPUTE_PROFILES = ("full", "laptop")
EXP3_GIGG_MODES = ("paper_ref", "stable")

_BAYESIAN_METHODS = {"GR_RHS", "RHS", "GIGG_MMLE", "GIGG_b_small", "GIGG_GHS", "GIGG_b_large", "GHS_plus"}
_BAYESIAN_DEFAULT_CHAINS = 4
_UNTIL_CONVERGED_RETRY_HARD_CAP = 12
_RETRY_MAX_WARMUP = 8000
_RETRY_MAX_POST_DRAWS = 8000
_RETRY_MAX_GIGG_ITER = 50000
_GHS_PLUS_DEFAULT_CHAINS = 4
_GHS_PLUS_DEFAULT_WARMUP = 2500
_GHS_PLUS_DEFAULT_POST_DRAWS = 2500
_GHS_PLUS_DEFAULT_RHAT_THRESHOLD = 1.01
_GHS_PLUS_DEFAULT_ESS_THRESHOLD = 400.0
_EXP4_DEFAULT_BACKEND = "gibbs"
_EXP4_DEFAULT_MAX_CONV_RETRIES = 3

# ---------------------------------------------------------------------------
# Compute-profile helpers
# ---------------------------------------------------------------------------

def _normalize_compute_profile(profile: str) -> str:
    p = str(profile).strip().lower()
    if p not in COMPUTE_PROFILES:
        raise ValueError(f"unknown compute profile: {profile!r}; expected one of {COMPUTE_PROFILES}")
    return p


def _normalize_exp3_gigg_mode(gigg_mode: str) -> str:
    mode = str(gigg_mode).strip().lower()
    if mode not in EXP3_GIGG_MODES:
        raise ValueError(f"unknown exp3 gigg_mode: {gigg_mode!r}; expected one of {EXP3_GIGG_MODES}")
    return mode


def _exp3_gigg_config_for_mode(base_cfg: dict[str, Any], *, gigg_mode: str) -> dict[str, Any]:
    mode = _normalize_exp3_gigg_mode(gigg_mode)
    out = dict(base_cfg)
    if mode == "paper_ref":
        # Strict reference mode: keep original MMLE trajectory and disable
        # Exp3-only stabilization/rescue behavior.
        out["mmle_step_size"] = 1.0
        out["randomize_group_order"] = False
        out["lambda_vectorized_update"] = False
        out["extra_beta_refresh_prob"] = 0.0
        out["init_scale_blend"] = 0.5
        out["extra_retry"] = 0
        out.pop("retry_cap", None)
        out["no_retry"] = True
    else:
        out.setdefault("extra_retry", 0)
    return out


def _resolve_method_list(methods: Sequence[str] | None, *, profile: str) -> list[str]:
    if methods is None:
        base = METHODS if _normalize_compute_profile(profile) == "full" else LAPTOP_METHODS
        return list(base)
    requested = [str(m).strip() for m in methods]
    unknown = sorted(set(requested) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")
    return [m for m in METHODS if m in set(requested)]


def _sampler_for_profile(profile: str, *, experiment: str = "") -> SamplerConfig:
    p = _normalize_compute_profile(profile)
    if p == "full":
        return SamplerConfig()
    return SamplerConfig(
        chains=1,
        warmup=250,
        post_warmup_draws=250,
        adapt_delta=0.92,
        max_treedepth=10,
        strict_adapt_delta=0.97,
        strict_max_treedepth=12,
        max_divergence_ratio=0.01,
        rhat_threshold=1.03,
        ess_threshold=120.0,
    )


def _gigg_config_for_profile(profile: str) -> dict[str, Any]:
    p = _normalize_compute_profile(profile)
    if p == "full":
        # Boss et al. (2024) Section 5.2: 10 000 burn-in + 10 000 posterior draws.
        # floor=cap=10000 pins both burnin and draws to exactly 10k regardless of
        # the HMC sampler budget, matching the published computational baseline.
        # no_retry=True: if 10k+10k is not enough, report non-convergence rather
        # than inflating the budget beyond what the paper used.
        return {"iter_mult": 4, "iter_floor": 10000, "iter_cap": 10000, "btrick": False, "mmle_burnin_only": True, "no_retry": True}
    # Laptop profile: keep GIGG light by default; difficult settings get extra retry.
    # This avoids slowing the full grid while still rescuing problematic CL/G10x5 cases.
    return {
        "iter_mult": 2,
        "iter_floor": 600,
        "iter_cap": 1800,
        "btrick": False,
        "mmle_burnin_only": True,
        "randomize_group_order": False,
        "extra_beta_refresh_prob": 0.0,
        "progress_bar": False,
        "extra_retry": 0,
    }


def _sampler_for_exp5(base: SamplerConfig, *, profile: str) -> SamplerConfig:
    # DGP aligned with Exp3 (n=100, p=50); standard Exp3-level sampler budget suffices.
    p = _normalize_compute_profile(profile)
    if p == "full":
        return SamplerConfig(
            chains=max(4, int(base.chains)),
            warmup=max(800, int(base.warmup)),
            post_warmup_draws=max(800, int(base.post_warmup_draws)),
            adapt_delta=max(0.95, float(base.adapt_delta)),
            max_treedepth=max(12, int(base.max_treedepth)),
            strict_adapt_delta=max(0.99, float(base.strict_adapt_delta)),
            strict_max_treedepth=max(14, int(base.strict_max_treedepth)),
            max_divergence_ratio=min(0.015, float(base.max_divergence_ratio)),
            rhat_threshold=min(1.03, float(base.rhat_threshold)),
            ess_threshold=max(400.0, float(base.ess_threshold)),
        )
    return SamplerConfig(
        chains=max(2, int(base.chains)),
        warmup=max(500, int(base.warmup)),
        post_warmup_draws=max(500, int(base.post_warmup_draws)),
        adapt_delta=max(0.93, float(base.adapt_delta)),
        max_treedepth=max(11, int(base.max_treedepth)),
        strict_adapt_delta=max(0.98, float(base.strict_adapt_delta)),
        strict_max_treedepth=max(13, int(base.strict_max_treedepth)),
        max_divergence_ratio=min(0.02, float(base.max_divergence_ratio)),
        rhat_threshold=min(1.05, float(base.rhat_threshold)),
        ess_threshold=max(200.0, float(base.ess_threshold)),
    )


def _sampler_for_ghs_plus_default(base: SamplerConfig) -> SamplerConfig:
    """Method-specific default budget for Grouped Horseshoe+."""
    return SamplerConfig(
        chains=max(_GHS_PLUS_DEFAULT_CHAINS, int(base.chains)),
        warmup=max(_GHS_PLUS_DEFAULT_WARMUP, int(base.warmup)),
        post_warmup_draws=max(_GHS_PLUS_DEFAULT_POST_DRAWS, int(base.post_warmup_draws)),
        adapt_delta=max(0.95, float(base.adapt_delta)),
        max_treedepth=max(12, int(base.max_treedepth)),
        strict_adapt_delta=max(0.99, float(base.strict_adapt_delta)),
        strict_max_treedepth=max(14, int(base.strict_max_treedepth)),
        max_divergence_ratio=min(0.005, float(base.max_divergence_ratio)),
        rhat_threshold=min(_GHS_PLUS_DEFAULT_RHAT_THRESHOLD, float(base.rhat_threshold)),
        ess_threshold=max(_GHS_PLUS_DEFAULT_ESS_THRESHOLD, float(base.ess_threshold)),
    )


def _sampler_for_bayesian_default(base: SamplerConfig, *, min_chains: int | None = None) -> SamplerConfig:
    """Unified Bayesian default: enforce a minimum chain count for Bayesian methods."""
    mc = int(_BAYESIAN_DEFAULT_CHAINS if min_chains is None else min_chains)
    mc = max(1, mc)
    return SamplerConfig(
        chains=max(mc, int(base.chains)),
        warmup=int(base.warmup),
        post_warmup_draws=int(base.post_warmup_draws),
        adapt_delta=float(base.adapt_delta),
        max_treedepth=int(base.max_treedepth),
        strict_adapt_delta=float(base.strict_adapt_delta),
        strict_max_treedepth=int(base.strict_max_treedepth),
        max_divergence_ratio=float(base.max_divergence_ratio),
        rhat_threshold=float(base.rhat_threshold),
        ess_threshold=float(base.ess_threshold),
    )


def _default_repeats(exp: str, profile: str) -> int:
    p = _normalize_compute_profile(profile)
    full = {"exp1": 500, "exp2": 30, "exp3": 20, "exp4": 30, "exp5": 30}
    laptop = {"exp1": 200, "exp2": 10, "exp3": 5, "exp4": 15, "exp5": 15}
    table = full if p == "full" else laptop
    if str(exp).lower() not in table:
        raise ValueError(f"unknown experiment: {exp!r}")
    return int(table[str(exp).lower()])

# ---------------------------------------------------------------------------
# Convergence-retry infrastructure
# ---------------------------------------------------------------------------

def _is_bayesian_method(method: str) -> bool:
    return str(method) in _BAYESIAN_METHODS


def _default_convergence_retries(profile: str) -> int:
    return 2 if _normalize_compute_profile(profile) == "full" else 1


def _resolve_convergence_retry_limit(
    profile: str,
    max_convergence_retries: int | None,
    *,
    until_bayes_converged: bool,
) -> int:
    if max_convergence_retries is not None:
        return int(max_convergence_retries)
    # Default retries are profile-bounded to avoid runaway wall-time.
    # If truly unbounded-until-converged behavior is desired, pass
    # max_convergence_retries=-1 explicitly.
    if bool(until_bayes_converged):
        return _default_convergence_retries(profile)
    return _default_convergence_retries(profile)


def _resolve_sampler_backend_for_experiment(exp: str, sampler_backend: str) -> str:
    exp_key = str(exp).strip().lower()
    backend = str(sampler_backend).strip().lower()
    exp4_aliases = {"4", "exp4", "exp4_variant_ablation"}
    if exp_key in exp4_aliases and backend == "nuts":
        return str(_EXP4_DEFAULT_BACKEND)
    return backend


def _retry_budget_from_limit(max_convergence_retries: int) -> tuple[int, bool]:
    retry_raw = int(max_convergence_retries)
    if retry_raw >= 0:
        return retry_raw, False
    return int(_UNTIL_CONVERGED_RETRY_HARD_CAP), True


def _scale_sampler_for_retry(base: SamplerConfig, attempt: int) -> SamplerConfig:
    k = max(0, int(attempt))
    if k == 0:
        return base
    mul = int(2 ** k)
    # Keep convergence criteria fixed across retries to enforce a uniform
    # Bayesian quality standard. Retries only increase sampling budget.
    return SamplerConfig(
        chains=max(1, int(base.chains)),
        warmup=min(_RETRY_MAX_WARMUP, max(50, int(base.warmup) * mul)),
        post_warmup_draws=min(_RETRY_MAX_POST_DRAWS, max(50, int(base.post_warmup_draws) * mul)),
        adapt_delta=min(0.995, float(base.adapt_delta) + 0.02 * k),
        max_treedepth=min(15, int(base.max_treedepth) + k),
        strict_adapt_delta=min(0.999, float(base.strict_adapt_delta) + 0.01 * k),
        strict_max_treedepth=min(16, int(base.strict_max_treedepth) + k),
        max_divergence_ratio=float(base.max_divergence_ratio),
        rhat_threshold=float(base.rhat_threshold),
        ess_threshold=float(base.ess_threshold),
    )


def _scale_gigg_config_for_retry(cfg: dict[str, Any], attempt: int) -> dict[str, Any]:
    k = max(0, int(attempt))
    if k == 0:
        return dict(cfg)
    out = dict(cfg)
    mul = int(2 ** k)
    out["iter_mult"] = max(1, int(out.get("iter_mult", 1)) * mul)
    out["iter_floor"] = min(_RETRY_MAX_GIGG_ITER, max(10, int(out.get("iter_floor", 500)) * mul))
    out["iter_cap"] = min(_RETRY_MAX_GIGG_ITER, max(out["iter_floor"], int(out.get("iter_cap", 1500)) * mul))
    # Retries should improve mixing quality, not only increase iteration counts.
    out["randomize_group_order"] = bool(out.get("randomize_group_order", True))
    out["lambda_vectorized_update"] = bool(out.get("lambda_vectorized_update", True))
    refresh_prev = float(out.get("extra_beta_refresh_prob", 0.0))
    refresh_target = min(0.20, 0.08 + 0.04 * float(k - 1))
    out["extra_beta_refresh_prob"] = max(refresh_prev, refresh_target)
    return out


def _invalidate_unconverged_result(res: FitResult, *, method: str, attempts: int) -> FitResult:
    msg = f"ConvergenceError: {method} did not converge after {attempts} attempt(s)"
    if str(res.error).strip():
        msg = f"{msg}; last_error={res.error}"
    res.status = "error"
    res.error = msg
    res.converged = False
    res.beta_mean = None
    res.beta_draws = None
    res.kappa_draws = None
    res.group_scale_draws = None
    res.tau_draws = None
    return res


def _attach_retry_diagnostics(
    res: FitResult,
    *,
    method: str,
    attempts: int,
    retry_max: int,
    until_mode: bool,
    enforce_bayes_convergence: bool,
) -> FitResult:
    diag = dict(res.diagnostics or {})
    diag["convergence_retry"] = {
        "method": str(method),
        "attempts_used": int(max(1, attempts)),
        "max_attempts": int(max(1, retry_max + 1)),
        "until_converged_mode": bool(until_mode),
        "enforce_bayes_convergence": bool(enforce_bayes_convergence),
        "status": str(res.status),
        "converged": bool(res.converged),
    }
    res.diagnostics = diag
    return res


def _attempts_used(res: FitResult) -> int:
    diag = res.diagnostics if isinstance(res.diagnostics, dict) else {}
    retry = diag.get("convergence_retry", {}) if isinstance(diag, dict) else {}
    try:
        return int(retry.get("attempts_used", 1))
    except Exception:
        return 1


def _result_diag_fields(res: FitResult) -> dict[str, float | str]:
    return {
        "runtime_seconds": float(res.runtime_seconds),
        "rhat_max": float(res.rhat_max),
        "bulk_ess_min": float(res.bulk_ess_min),
        "divergence_ratio": float(res.divergence_ratio),
        "error": str(res.error),
    }


def _resolve_workers(n_jobs: int, n_tasks: int) -> int:
    return max(1, min(int(n_jobs), int(n_tasks)))


def _parallel_rows(
    tasks: list[Any],
    worker,
    n_jobs: int,
    *,
    prefer_process: bool = False,
    process_fallback: str = "thread",
    progress_desc: str | None = None,
) -> list[Any]:
    if len(tasks) == 0:
        return []
    workers = _resolve_workers(n_jobs=n_jobs, n_tasks=len(tasks))
    if workers <= 1:
        return [worker(t) for t in tqdm(tasks, total=len(tasks), desc=progress_desc or "Running", leave=True)]
    out: list[Any] = [None] * len(tasks)
    done: list[bool] = [False] * len(tasks)
    fut_map: dict[Any, int] = {}
    use_process = bool(prefer_process)
    if use_process:
        process_ok, process_reason = can_use_process_pool()
        if not process_ok:
            mode = str(process_fallback).strip().lower()
            if mode == "serial":
                print(f"[WARN] Process pool disabled ({process_reason}). Using serial execution.")
                return [worker(t) for t in tqdm(tasks, total=len(tasks), desc=(progress_desc or "Running") + " [serial]", leave=True)]
            print(f"[WARN] Process pool disabled ({process_reason}). Using thread pool.")
            use_process = False
    executor_cls = ProcessPoolExecutor if use_process else ThreadPoolExecutor
    try:
        with executor_cls(max_workers=workers) as ex:
            fut_map = {ex.submit(worker, tasks[i]): i for i in range(len(tasks))}
            for fut in tqdm(as_completed(fut_map), total=len(tasks), desc=progress_desc or "Running", leave=True):
                idx = fut_map[fut]
                out[idx] = fut.result()
                done[idx] = True
    except Exception as exc:
        if use_process:
            pending_idxs = [i for i, ok in enumerate(done) if not ok]
            pending_tasks = [tasks[i] for i in pending_idxs]
            mode = str(process_fallback).strip().lower()
            if mode == "thread":
                print(f"[WARN] Process pool failed ({type(exc).__name__}: {exc}). Falling back to thread pool.")
                if pending_tasks:
                    with ThreadPoolExecutor(max_workers=min(workers, len(pending_tasks))) as ex:
                        tfut_map = {ex.submit(worker, pending_tasks[i]): i for i in range(len(pending_tasks))}
                        for fut in tqdm(as_completed(tfut_map), total=len(pending_tasks), desc=(progress_desc or "Running") + " [thread]", leave=True):
                            loc = tfut_map[fut]
                            out[pending_idxs[loc]] = fut.result()
                            done[pending_idxs[loc]] = True
            elif mode == "serial":
                print(f"[WARN] Process pool failed ({type(exc).__name__}: {exc}). Falling back to serial execution.")
                if pending_tasks:
                    serial_out = [worker(t) for t in tqdm(pending_tasks, total=len(pending_tasks), desc=(progress_desc or "Running") + " [serial]", leave=True)]
                    for loc, row in enumerate(serial_out):
                        out[pending_idxs[loc]] = row
                        done[pending_idxs[loc]] = True
                return out
            else:
                raise
        else:
            raise
    return out

# ---------------------------------------------------------------------------
# Theory helpers
# ---------------------------------------------------------------------------

def theta_u0_rho(u0: float, rho: float) -> float:
    u = float(u0)
    rho2 = float(rho) ** 2
    den = u + (1.0 - u) * rho2
    return float((u * rho2) / max(den, 1e-12))


def xi_crit_u0_rho(u0: float, rho: float) -> float:
    return 0.5 * theta_u0_rho(u0=u0, rho=rho)


