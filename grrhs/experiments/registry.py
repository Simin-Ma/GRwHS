from __future__ import annotations
"""Model registry and builders for experiments.

This module exposes a simple registry to construct models from config dicts.
Optional models (SVI/Gibbs) are imported in try blocks so that baseline-only
environments can still import this module without errors.
"""

from typing import Callable, Dict, Any, Optional, Sequence

import numpy as np

from data.generators import make_groups

# Optional: GRRHS main models (if available)
try:
    from grrhs.models.grrhs_svi_numpyro import GRRHS_SVI  # type: ignore
except Exception:  # pragma: no cover
    GRRHS_SVI = None  # type: ignore

try:
    from grrhs.models.grrhs_gibbs import GRRHS_Gibbs  # type: ignore
except Exception:  # pragma: no cover
    GRRHS_Gibbs = None  # type: ignore

try:
    from grrhs.models.gigg_regression import GIGGRegression  # type: ignore
except Exception:  # pragma: no cover
    GIGGRegression = None  # type: ignore

try:
    from grrhs.models.gigg_cran import GIGGRegressionCRAN  # type: ignore
except Exception:  # pragma: no cover
    GIGGRegressionCRAN = None  # type: ignore

# Baselines (numpy/skglm implementations)
from grrhs.models.baselines import (
    OLS,
    Ridge,
    Lasso,
    ElasticNet,
    SparseGroupLasso,
    MBSGSBGLSSRegression,
    BGLSSPythonRegression,
    HorseshoeRegression,
    RegularizedHorseshoeRegression,
)

# ------------------------------
# Registry and decorator
# ------------------------------
REGISTRY: Dict[str, Callable[[Dict[str, Any]], Any]] = {}


def register(name: str) -> Callable[[Callable[[Dict[str, Any]], Any]], Callable[[Dict[str, Any]], Any]]:
    """Register a builder via @register('model_name')."""

    def deco(fn: Callable[[Dict[str, Any]], Any]) -> Callable[[Dict[str, Any]], Any]:
        key = name.strip().lower()
        if key in REGISTRY:
            raise ValueError(f"Model '{key}' already registered.")
        REGISTRY[key] = fn
        return fn

    return deco


def _get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Safe getter: _get(cfg, 'model.alpha', 1.0)."""
    cur: Any = cfg
    for seg in path.split("."):
        if not isinstance(cur, dict) or seg not in cur:
            return default
        cur = cur[seg]
    return cur


# ------------------------------
# Helpers
# ------------------------------


def _resolve_group_weight_mode(cfg: Dict[str, Any], groups: Sequence[Sequence[int]]) -> Any:
    """
    Resolve group weights either from explicit values or a declarative mode.

    If `model.group_weights` exists, it takes precedence. Otherwise, interpret
    `model.group_weight_mode` (sqrt|size|uniform) to build a weight vector.
    """

    direct = _get(cfg, "model.group_weights", None)
    if direct is not None:
        return direct

    mode_raw = _get(cfg, "model.group_weight_mode", None)
    if mode_raw is None:
        return None

    mode = str(mode_raw).lower()
    if mode in {"sqrt", "default", "none"}:
        return None
    if mode in {"size", "count"}:
        return [float(len(group)) for group in groups]
    if mode in {"uniform", "ones"}:
        return [1.0 for _ in groups]
    raise ValueError(f"Unsupported group_weight_mode '{mode_raw}'. Use sqrt|size|uniform or explicit weights.")


# ------------------------------
# Helpers
# ------------------------------


def _infer_groups(cfg: Dict[str, Any]) -> Optional[list[list[int]]]:
    """Infer group structure from config when not provided explicitly."""
    groups = _get(cfg, "data.groups", None)
    if groups is not None:
        normalized: list[list[int]] = []
        for group in groups:
            normalized.append([int(idx) for idx in group])
        return normalized

    p = _get(cfg, "data.p", None)
    if p is None:
        return None

    group_sizes = _get(cfg, "data.group_sizes", None)
    if group_sizes is None:
        group_sizes = "equal"

    if isinstance(group_sizes, str):
        G = _get(cfg, "data.G", None)
        if G is None:
            return None
        label = group_sizes.lower()
        if label == "equal":
            if p % G != 0:
                raise ValueError("Cannot build equal-sized groups: 'data.p' not divisible by 'data.G'.")
            size = p // G
            return [list(range(g * size, (g + 1) * size)) for g in range(G)]
        # Delegate to synthetic grouping helper for other presets (e.g., "variable")
        return make_groups(int(p), int(G), group_sizes)

    try:
        sizes = [int(s) for s in group_sizes]
    except TypeError as exc:
        raise TypeError("data.group_sizes must be a string or sequence of ints.") from exc

    total = sum(sizes)
    if total != p:
        raise ValueError("Sum of 'data.group_sizes' must equal 'data.p' to build group indices.")
    groups_out: list[list[int]] = []
    cursor = 0
    for size in sizes:
        if size <= 0:
            raise ValueError("Group sizes must be positive integers.")
        groups_out.append(list(range(cursor, cursor + size)))
        cursor += size
    return groups_out


def _horseshoe_common_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract shared kwargs for Horseshoe baselines."""
    scale_intercept = float(_get(cfg, "model.scale_intercept", 10.0))
    tau0 = _get(cfg, "model.tau0", None)
    scale_global = float(_get(cfg, "model.scale_global", 1.0))
    if tau0 is not None:
        scale_global = float(tau0)
    nu_global = float(_get(cfg, "model.nu_global", 1.0))
    nu_local = float(_get(cfg, "model.nu_local", 1.0))
    sigma_scale = float(_get(cfg, "model.sigma_scale", 1.0))
    num_warmup = int(_get(cfg, "inference.nuts.num_warmup", _get(cfg, "model.num_warmup", 1000)))
    num_samples = int(_get(cfg, "inference.nuts.num_samples", _get(cfg, "model.num_samples", 1000)))
    num_chains = max(1, int(_get(cfg, "inference.nuts.num_chains", _get(cfg, "model.num_chains", 1))))
    thinning = max(1, int(_get(cfg, "inference.nuts.thinning", _get(cfg, "model.thinning", 1))))
    target_accept = _get(
        cfg,
        "inference.nuts.target_accept_prob",
        _get(cfg, "model.target_accept_prob", 0.99),
    )
    max_tree_depth = int(
        _get(
            cfg,
            "inference.nuts.max_tree_depth",
            _get(cfg, "model.max_tree_depth", 10),
        )
    )
    progress_bar = bool(_get(cfg, "model.progress_bar", _get(cfg, "runtime.progress_bar", False)))
    backend = str(_get(cfg, "model.backend", _get(cfg, "model.rhs_backend", "cmdstan"))).strip().lower()
    seed_candidates = [
        "model.seed",
        "inference.nuts.seed",
        "inference.seed",
        "inference.gibbs.seed",
        "inference.svi.seed",
        "runtime.seed",
        "seed",
    ]
    seed_val = None
    for path in seed_candidates:
        seed_val = _get(cfg, path, None)
        if seed_val is not None:
            break

    kwargs: Dict[str, Any] = {
        "scale_intercept": scale_intercept,
        "scale_global": scale_global,
        "nu_global": nu_global,
        "nu_local": nu_local,
        "sigma_scale": sigma_scale,
        "num_warmup": num_warmup,
        "num_samples": num_samples,
        "num_chains": num_chains,
        "thinning": thinning,
        "target_accept_prob": float(0.99 if target_accept is None else target_accept),
        "max_tree_depth": max(1, int(max_tree_depth)),
        "progress_bar": progress_bar,
        "backend": backend,
    }
    if seed_val is not None:
        kwargs["seed"] = int(seed_val)
    kwargs["likelihood"] = "gaussian"
    return kwargs


# ------------------------------
# Baselines
# ------------------------------


@register("ridge")
def _build_ridge(cfg: Dict[str, Any]) -> Any:
    alpha = float(_get(cfg, "model.alpha", 1.0))
    fit_intercept = bool(_get(cfg, "model.fit_intercept", False))
    return Ridge(alpha=alpha, fit_intercept=fit_intercept)


@register("ols")
@register("linear_regression")
@register("linear")
def _build_ols(cfg: Dict[str, Any]) -> Any:
    fit_intercept = bool(_get(cfg, "model.fit_intercept", False))
    positive = bool(_get(cfg, "model.positive", False))
    n_jobs = _get(cfg, "model.n_jobs", None)
    return OLS(
        fit_intercept=fit_intercept,
        positive=positive,
        n_jobs=None if n_jobs is None else int(n_jobs),
    )


@register("lasso")
def _build_lasso(cfg: Dict[str, Any]) -> Any:
    alpha = float(_get(cfg, "model.alpha", 1.0))
    fit_intercept = bool(_get(cfg, "model.fit_intercept", False))
    max_iter = int(_get(cfg, "model.max_iter", 10_000))
    max_epochs = int(_get(cfg, "model.max_epochs", 50_000))
    p0 = int(_get(cfg, "model.p0", 10))
    tol = float(_get(cfg, "model.tol", 1e-6))
    warm_start = bool(_get(cfg, "model.warm_start", True))
    ws_strategy = str(_get(cfg, "model.ws_strategy", "subdiff"))
    verbose = int(_get(cfg, "model.verbose", 0))
    positive = bool(_get(cfg, "model.positive", False))
    return Lasso(
        alpha=alpha,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        max_epochs=max_epochs,
        p0=p0,
        tol=tol,
        warm_start=warm_start,
        ws_strategy=ws_strategy,
        verbose=verbose,
        positive=positive,
    )


@register("elastic_net")
@register("enet")  # alias
def _build_enet(cfg: Dict[str, Any]) -> Any:
    alpha = float(_get(cfg, "model.alpha", 1.0))
    l1_ratio = float(_get(cfg, "model.l1_ratio", 0.5))
    fit_intercept = bool(_get(cfg, "model.fit_intercept", False))
    max_iter = int(_get(cfg, "model.max_iter", 10_000))
    tol = float(_get(cfg, "model.tol", 1e-6))
    warm_start = bool(_get(cfg, "model.warm_start", True))
    return ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        tol=tol,
        warm_start=warm_start,
    )


@register("sparse_group_lasso")
@register("sparsegrouplasso")
@register("sgl")
def _build_sparse_group_lasso(cfg: Dict[str, Any]) -> Any:
    groups = _infer_groups(cfg)
    if groups is None:
        raise ValueError("SparseGroupLasso requires 'data.groups' in config (list of index lists).")
    alpha = float(_get(cfg, "model.alpha", 1.0))
    l1_ratio = float(_get(cfg, "model.l1_ratio", 0.5))
    fit_intercept = bool(_get(cfg, "model.fit_intercept", False))
    max_iter = int(_get(cfg, "model.max_iter", 2_000))
    max_epochs = int(_get(cfg, "model.max_epochs", 50_000))
    p0 = int(_get(cfg, "model.p0", 10))
    tol = float(_get(cfg, "model.tol", 1e-6))
    warm_start = bool(_get(cfg, "model.warm_start", True))
    ws_strategy = str(_get(cfg, "model.ws_strategy", "fixpoint"))
    verbose = int(_get(cfg, "model.verbose", 0))
    group_weights = _resolve_group_weight_mode(cfg, groups)
    feature_weights = _get(cfg, "model.feature_weights", None)
    return SparseGroupLasso(
        groups=groups,
        alpha=alpha,
        l1_ratio=l1_ratio,
        group_weights=group_weights,
        feature_weights=feature_weights,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        max_epochs=max_epochs,
        p0=p0,
        tol=tol,
        warm_start=warm_start,
        ws_strategy=ws_strategy,
        verbose=verbose,
    )

# ------------------------------
# Horseshoe baselines
# ------------------------------


@register("horseshoe")
@register("hs")
def _build_horseshoe(cfg: Dict[str, Any]) -> Any:
    kwargs = _horseshoe_common_kwargs(cfg)
    return HorseshoeRegression(**kwargs)


@register("regularized_horseshoe")
@register("rhs")
@register("regularised_horseshoe")
def _build_regularized_horseshoe(cfg: Dict[str, Any]) -> Any:
    kwargs = _horseshoe_common_kwargs(cfg)
    kwargs["slab_scale"] = float(_get(cfg, "model.slab_scale", 1.0))
    slab_df = _get(cfg, "model.slab_df", None)
    if slab_df is not None:
        kwargs["slab_df"] = float(slab_df)
    return RegularizedHorseshoeRegression(**kwargs)


@register("gigg")
@register("gigg_regression")
def _build_gigg(cfg: Dict[str, Any]) -> Any:
    backend = str(_get(cfg, "model.backend", _get(cfg, "model.gigg_backend", "python"))).strip().lower()
    use_cran = backend in {"cran", "cran_compatible"} or bool(_get(cfg, "model.cran_compatible", False))
    if use_cran:
        if GIGGRegressionCRAN is None:
            raise ImportError("GIGGRegressionCRAN is not available. Ensure grrhs.models.gigg_cran exists.")
    else:
        if GIGGRegression is None:
            raise ImportError("GIGGRegression is not available. Ensure grrhs.models.gigg_regression exists.")
    groups = _infer_groups(cfg)
    if groups is None:
        raise ValueError("GIGGRegression requires 'data.groups' in config (list of index lists).")
    method = str(_get(cfg, "model.method", "mmle")).lower()
    mmle_enabled = _get(cfg, "model.mmle_enabled", None)
    if mmle_enabled is not None and not bool(mmle_enabled):
        method = "fixed"
    burnin = int(_get(cfg, "inference.gibbs.burn_in", _get(cfg, "model.n_burn_in", _get(cfg, "model.burnin", 500))))
    thin = max(1, int(_get(cfg, "inference.gibbs.thin", _get(cfg, "model.n_thin", _get(cfg, "model.thin", 1)))))
    n_samples_raw = _get(cfg, "model.n_samples", None)
    if n_samples_raw is None:
        total_iters = int(_get(cfg, "model.iters", 3000))
        n_samples = max(1, (max(0, total_iters - burnin)) // thin)
    else:
        n_samples = max(1, int(n_samples_raw))
    num_chains = max(1, int(_get(cfg, "inference.gibbs.num_chains", _get(cfg, "model.num_chains", 1))))
    jitter = float(_get(cfg, "model.jitter", 1e-8))
    seed = _get(
        cfg,
        "inference.gibbs.seed",
        _get(cfg, "model.seed", _get(cfg, "seed", 0)),
    )
    b_init = float(_get(cfg, "model.b_init", 0.5))
    b_floor = float(_get(cfg, "model.b_floor", 1e-3))
    b_max = float(_get(cfg, "model.b_max", 4.0))
    tau_sq_init = float(_get(cfg, "model.tau_sq_init", _get(cfg, "model.tau_init", _get(cfg, "model.tau_scale", 1.0))))
    sigma_sq_init = float(_get(cfg, "model.sigma_sq_init", _get(cfg, "model.sigma_init", _get(cfg, "model.sigma_scale", 1.0))))
    store_lambda = bool(_get(cfg, "model.store_lambda", False))
    a_value = _get(cfg, "model.a_value", None)
    share_group_hyper = bool(_get(cfg, "model.share_group_hyper", False))
    mmle_update = str(_get(cfg, "model.mmle_update", "paper_lambda_only"))
    mmle_burnin_only = bool(_get(cfg, "model.mmle_burnin_only", True))
    mmle_samp_size = int(_get(cfg, "model.mmle_samp_size", 1000))
    mmle_tol_scale = float(_get(cfg, "model.mmle_tol_scale", 1e-4))
    mmle_max_iters = int(_get(cfg, "model.mmle_max_iters", 50000))
    fit_intercept = bool(_get(cfg, "model.fit_intercept", True))
    btrick = bool(_get(cfg, "model.btrick", False))
    stable_solve = bool(_get(cfg, "model.stable_solve", True))
    lambda_constraint_mode = str(_get(cfg, "model.lambda_constraint_mode", "hard"))
    lambda_cap = float(_get(cfg, "model.lambda_cap", 1e3))
    lambda_soft_cap = float(_get(cfg, "model.lambda_soft_cap", lambda_cap))
    force_a_1_over_n = bool(_get(cfg, "model.force_a_1_over_n", True))

    if use_cran:
        # CRAN sampler expects group hyperparameter vectors `a` and `b` (length G).
        G = len(groups)
        a_vec = np.full(G, 0.5, dtype=float) if a_value is None else np.full(G, float(a_value), dtype=float)
        b_vec = np.full(G, float(b_init), dtype=float)
        return GIGGRegressionCRAN(
            method=method,
            n_burn_in=burnin,
            n_samples=n_samples,
            n_thin=thin,
            seed=int(seed) if seed is not None else 0,
            num_chains=num_chains,
            btrick=btrick,
            stable_solve=stable_solve,
            fit_intercept=fit_intercept,
            store_lambda=store_lambda,
            a=a_vec,
            b=b_vec,
        )

    return GIGGRegression(
        method=method,
        n_burn_in=burnin,
        n_samples=n_samples,
        n_thin=thin,
        jitter=jitter,
        seed=int(seed) if seed is not None else 0,
        num_chains=num_chains,
        a_value=None if a_value is None else float(a_value),
        b_init=b_init,
        b_floor=b_floor,
        b_max=b_max,
        tau_sq_init=tau_sq_init,
        sigma_sq_init=sigma_sq_init,
        store_lambda=store_lambda,
        share_group_hyper=share_group_hyper,
        mmle_update=mmle_update,
        mmle_burnin_only=mmle_burnin_only,
        force_a_1_over_n=force_a_1_over_n,
        mmle_samp_size=mmle_samp_size,
        mmle_tol_scale=mmle_tol_scale,
        mmle_max_iters=mmle_max_iters,
        fit_intercept=fit_intercept,
        btrick=btrick,
        stable_solve=stable_solve,
        lambda_constraint_mode=lambda_constraint_mode,
        lambda_cap=lambda_cap,
        lambda_soft_cap=lambda_soft_cap,
    )


@register("bglss_mbsgs")
@register("mbsgs_bglss")
@register("bglss")
def _build_bglss_mbsgs(cfg: Dict[str, Any]) -> Any:
    groups = _infer_groups(cfg)
    if groups is None:
        raise ValueError("MBSGSBGLSSRegression requires grouped features in config (data.groups or inferable).")
    niter = int(_get(cfg, "model.niter", _get(cfg, "model.n_samples", 3000)))
    burnin = int(_get(cfg, "model.burnin", _get(cfg, "inference.gibbs.burn_in", 1000)))
    seed = int(
        _get(
            cfg,
            "model.seed",
            _get(
                cfg,
                "inference.gibbs.seed",
                _get(cfg, "seed", 2025),
            ),
        )
    )
    fit_intercept = bool(_get(cfg, "model.fit_intercept", False))
    save_posterior_samples = bool(_get(cfg, "model.save_posterior_samples", False))
    rscript = str(_get(cfg, "model.rscript", "Rscript"))
    timeout_sec = int(_get(cfg, "model.timeout_sec", 1800))
    verbose = bool(_get(cfg, "model.verbose", False))

    return MBSGSBGLSSRegression(
        groups=groups,
        niter=niter,
        burnin=burnin,
        seed=seed,
        fit_intercept=fit_intercept,
        save_posterior_samples=save_posterior_samples,
        rscript=rscript,
        timeout_sec=timeout_sec,
        verbose=verbose,
    )


@register("bglss_python")
@register("bglss_py")
def _build_bglss_python(cfg: Dict[str, Any]) -> Any:
    groups = _infer_groups(cfg)
    if groups is None:
        raise ValueError("BGLSSPythonRegression requires grouped features in config (data.groups or inferable).")

    niter = int(_get(cfg, "model.niter", _get(cfg, "model.n_samples", 6000)))
    burnin = int(_get(cfg, "model.burnin", _get(cfg, "inference.gibbs.burn_in", 2000)))
    seed = int(
        _get(
            cfg,
            "model.seed",
            _get(
                cfg,
                "inference.gibbs.seed",
                _get(cfg, "seed", 2025),
            ),
        )
    )
    fit_intercept = bool(_get(cfg, "model.fit_intercept", False))
    a = float(_get(cfg, "model.a", 1.0))
    b = float(_get(cfg, "model.b", 1.0))
    pi_prior = bool(_get(cfg, "model.pi_prior", True))
    pi_init = float(_get(cfg, "model.pi", 0.5))
    alpha = float(_get(cfg, "model.alpha", 0.1))
    gamma = float(_get(cfg, "model.gamma", 0.1))
    lambda_slab2 = float(_get(cfg, "model.lambda_slab2", _get(cfg, "model.lambda2_slab", 0.5)))
    lambda_spike2 = float(_get(cfg, "model.lambda_spike2", _get(cfg, "model.lambda2_spike", 25.0)))
    update_tau = bool(_get(cfg, "model.update_tau", True))
    num_update = int(_get(cfg, "model.num_update", 100))
    niter_update = int(_get(cfg, "model.niter.update", _get(cfg, "model.niter_update", 100)))
    store_beta_samples = bool(_get(cfg, "model.store_beta_samples", False))
    verbose = bool(_get(cfg, "model.verbose", False))

    return BGLSSPythonRegression(
        groups=groups,
        niter=niter,
        burnin=burnin,
        seed=seed,
        fit_intercept=fit_intercept,
        a=a,
        b=b,
        pi_prior=pi_prior,
        pi_init=pi_init,
        alpha=alpha,
        gamma=gamma,
        lambda_slab2=lambda_slab2,
        lambda_spike2=lambda_spike2,
        update_tau=update_tau,
        num_update=num_update,
        niter_update=niter_update,
        store_beta_samples=store_beta_samples,
        verbose=verbose,
    )


# ------------------------------
# GRRHS main models (if available)
# ------------------------------


@register("grrhs_svi")
def _build_grrhs_svi(cfg: Dict[str, Any]) -> Any:
    if GRRHS_SVI is None:
        raise ImportError("GRRHS_SVI is not available. Ensure grrhs.models.grrhs_svi_numpyro exists.")
    # Key hyperparameters from config with defaults
    c = float(_get(cfg, "model.c", 1.0))
    tau0 = float(_get(cfg, "model.tau0", 0.1))
    eta = float(_get(cfg, "model.eta", 0.5))
    s0 = float(_get(cfg, "model.s0", 1.0))
    alpha_c = float(_get(cfg, "model.alpha_c", 2.0))
    beta_c = float(_get(cfg, "model.beta_c", 2.0))
    svi_kwargs: Dict[str, Any] = {}
    steps = _get(cfg, "inference.svi.steps", None)
    if steps is not None:
        svi_kwargs["num_steps"] = int(steps)
    lr = _get(cfg, "inference.svi.lr", None)
    if lr is not None:
        svi_kwargs["lr"] = float(lr)
    seed = _get(cfg, "inference.svi.seed", _get(cfg, "seed", None))
    if seed is not None:
        svi_kwargs["seed"] = int(seed)
    batch_size = _get(cfg, "inference.svi.batch_size", None)
    if batch_size is not None:
        svi_kwargs["batch_size"] = int(batch_size)
    natural_grad = bool(_get(cfg, "inference.svi.natural_grad", False))
    use_hutch = bool(_get(cfg, "inference.svi.use_hutchinson", natural_grad))
    hutch_samples = int(_get(cfg, "inference.svi.hutchinson_samples", 8))
    cg_tol = float(_get(cfg, "inference.svi.cg_tol", 1e-5))
    cg_max_iters = int(_get(cfg, "inference.svi.cg_maxiter", 50))
    natgrad_damping = float(_get(cfg, "inference.svi.natgrad_damping", 1e-3))
    coupling_clip_raw = _get(cfg, "inference.svi.coupling_clip", 3.0)
    if coupling_clip_raw is None:
        coupling_clip_val = None
    else:
        try:
            coupling_clip_val = float(coupling_clip_raw)
        except (TypeError, ValueError):
            coupling_clip_val = None
    cov_damping = float(_get(cfg, "inference.svi.cov_damping", 0.0))

    svi = GRRHS_SVI(
        c=c,
        tau0=tau0,
        eta=eta,
        s0=s0,
        alpha_c=alpha_c,
        beta_c=beta_c,
        natural_gradient=natural_grad,
        use_hutchinson=use_hutch,
        hutchinson_samples=hutch_samples,
        cg_tol=cg_tol,
        cg_max_iters=cg_max_iters,
        natgrad_damping=natgrad_damping,
        coupling_clip=coupling_clip_val,
        covariance_damping=cov_damping,
        **svi_kwargs,
    )  # type: ignore
    return svi


def _gibbs_runtime_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract optional runtime overrides shared by Gibbs samplers."""
    overrides: Dict[str, Any] = {}
    burnin = _get(cfg, "inference.gibbs.burn_in", None)
    if burnin is not None:
        overrides["burnin"] = int(burnin)
    thin = _get(cfg, "inference.gibbs.thin", None)
    if thin is not None:
        overrides["thin"] = max(1, int(thin))
    slice_w = _get(cfg, "inference.gibbs.slice_w", None)
    if slice_w is not None:
        overrides["slice_w"] = float(slice_w)
    slice_m = _get(cfg, "inference.gibbs.slice_m", None)
    if slice_m is not None:
        overrides["slice_m"] = max(1, int(slice_m))
    tau_slice_w = _get(cfg, "inference.gibbs.tau_slice_w", None)
    if tau_slice_w is not None:
        overrides["tau_slice_w"] = float(tau_slice_w)
    tau_slice_m = _get(cfg, "inference.gibbs.tau_slice_m", None)
    if tau_slice_m is not None:
        overrides["tau_slice_m"] = max(1, int(tau_slice_m))
    jitter = _get(cfg, "inference.gibbs.jitter", None)
    if jitter is not None:
        overrides["jitter"] = float(jitter)
    num_chains = _get(cfg, "inference.gibbs.num_chains", _get(cfg, "model.num_chains", None))
    if num_chains is not None:
        overrides["num_chains"] = max(1, int(num_chains))
    mh_sd_log_tau2 = _get(cfg, "inference.gibbs.mh_sd_log_tau2", None)
    if mh_sd_log_tau2 is not None:
        overrides["mh_sd_log_tau2"] = float(mh_sd_log_tau2)
    mh_sd_log_lambda = _get(cfg, "inference.gibbs.mh_sd_log_lambda", None)
    if mh_sd_log_lambda is not None:
        overrides["mh_sd_log_lambda"] = float(mh_sd_log_lambda)
    mh_sd_log_a = _get(cfg, "inference.gibbs.mh_sd_log_a", None)
    if mh_sd_log_a is not None:
        overrides["mh_sd_log_a"] = float(mh_sd_log_a)
    mh_sd_log_c2 = _get(cfg, "inference.gibbs.mh_sd_log_c2", None)
    if mh_sd_log_c2 is not None:
        overrides["mh_sd_log_c2"] = float(mh_sd_log_c2)
    global_block_sd_u = _get(cfg, "inference.gibbs.global_block_sd_u", None)
    if global_block_sd_u is not None:
        overrides["global_block_sd_u"] = float(global_block_sd_u)
    global_block_sd_alpha = _get(cfg, "inference.gibbs.global_block_sd_alpha", None)
    if global_block_sd_alpha is not None:
        overrides["global_block_sd_alpha"] = float(global_block_sd_alpha)
    global_comp_sd = _get(cfg, "inference.gibbs.global_comp_sd", None)
    if global_comp_sd is not None:
        overrides["global_comp_sd"] = float(global_comp_sd)
    group_ac2_block_sd_alpha = _get(cfg, "inference.gibbs.group_ac2_block_sd_alpha", None)
    if group_ac2_block_sd_alpha is not None:
        overrides["group_ac2_block_sd_alpha"] = float(group_ac2_block_sd_alpha)
    group_ac2_block_sd_xi = _get(cfg, "inference.gibbs.group_ac2_block_sd_xi", None)
    if group_ac2_block_sd_xi is not None:
        overrides["group_ac2_block_sd_xi"] = float(group_ac2_block_sd_xi)
    group_comp_sd = _get(cfg, "inference.gibbs.group_comp_sd", None)
    if group_comp_sd is not None:
        overrides["group_comp_sd"] = float(group_comp_sd)
    use_lambda_slice = _get(cfg, "inference.gibbs.use_lambda_slice", None)
    if use_lambda_slice is not None:
        overrides["use_lambda_slice"] = bool(use_lambda_slice)
    lambda_slice_w = _get(cfg, "inference.gibbs.lambda_slice_w", None)
    if lambda_slice_w is not None:
        overrides["lambda_slice_w"] = float(lambda_slice_w)
    lambda_slice_m = _get(cfg, "inference.gibbs.lambda_slice_m", None)
    if lambda_slice_m is not None:
        overrides["lambda_slice_m"] = int(lambda_slice_m)
    use_collapsed_scale_updates = _get(cfg, "inference.gibbs.use_collapsed_scale_updates", None)
    if use_collapsed_scale_updates is not None:
        overrides["use_collapsed_scale_updates"] = bool(use_collapsed_scale_updates)
    adapt_proposals = _get(cfg, "inference.gibbs.adapt_proposals", None)
    if adapt_proposals is not None:
        overrides["adapt_proposals"] = bool(adapt_proposals)
    adapt_interval = _get(cfg, "inference.gibbs.adapt_interval", None)
    if adapt_interval is not None:
        overrides["adapt_interval"] = max(1, int(adapt_interval))
    adapt_until_frac = _get(cfg, "inference.gibbs.adapt_until_frac", None)
    if adapt_until_frac is not None:
        overrides["adapt_until_frac"] = float(adapt_until_frac)
    adapt_target_accept = _get(cfg, "inference.gibbs.adapt_target_accept", None)
    if adapt_target_accept is not None:
        overrides["adapt_target_accept"] = float(adapt_target_accept)
    adapt_step_size = _get(cfg, "inference.gibbs.adapt_step_size", None)
    if adapt_step_size is not None:
        overrides["adapt_step_size"] = float(adapt_step_size)
    min_proposal_sd = _get(cfg, "inference.gibbs.min_proposal_sd", None)
    if min_proposal_sd is not None:
        overrides["min_proposal_sd"] = float(min_proposal_sd)
    max_proposal_sd = _get(cfg, "inference.gibbs.max_proposal_sd", None)
    if max_proposal_sd is not None:
        overrides["max_proposal_sd"] = float(max_proposal_sd)
    return overrides


@register("grrhs_gibbs")
def _build_grrhs_gibbs(cfg: Dict[str, Any]) -> Any:
    if GRRHS_Gibbs is None:
        raise ImportError("GRRHS_Gibbs is not available. Ensure grrhs.models.grrhs_gibbs exists.")
    c = float(_get(cfg, "model.c", 1.0))
    tau0 = float(_get(cfg, "model.tau0", 0.1))
    eta = float(_get(cfg, "model.eta", 0.5))
    use_groups = bool(_get(cfg, "model.use_groups", True))
    s0 = float(_get(cfg, "model.s0", 1.0))
    alpha_c = float(_get(cfg, "model.alpha_c", 2.0))
    beta_c = float(_get(cfg, "model.beta_c", 2.0))
    iters = int(_get(cfg, "model.iters", 2000))
    use_pcabs_lite = bool(_get(cfg, "model.use_pcabs_lite", True))
    use_collapsed_scale_updates = bool(_get(cfg, "model.use_collapsed_scale_updates", True))
    seed = _get(
        cfg,
        "inference.gibbs.seed",
        _get(cfg, "model.seed", _get(cfg, "seed", 42)),
    )
    runtime_overrides = _gibbs_runtime_overrides(cfg)
    sampler = GRRHS_Gibbs(
        c=c,
        tau0=tau0,
        eta=eta,
        s0=s0,
        alpha_c=alpha_c,
        beta_c=beta_c,
        iters=iters,
        seed=int(seed),
        use_groups=use_groups,
        use_pcabs_lite=use_pcabs_lite,
        use_collapsed_scale_updates=use_collapsed_scale_updates,
        **runtime_overrides,
    )  # type: ignore
    return sampler


# ------------------------------
# Public API
# ------------------------------


def get_available_models() -> Dict[str, Callable[[Dict[str, Any]], Any]]:
    """Return a copy of registered model builder mapping."""
    return dict(REGISTRY)


def get_builder(name: str) -> Callable[[Dict[str, Any]], Any]:
    """Get builder by name (case-insensitive, supports aliases)."""
    key = (name or "").strip().lower()
    if key not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(REGISTRY.keys())}")
    return REGISTRY[key]


def get_model_name_from_config(cfg: Dict[str, Any]) -> str:
    """Support both 'model.name' and 'model.type' keys.
    Examples: ridge / lasso / elastic_net / sparse_group_lasso / grrhs_svi / grrhs_gibbs
    """
    name = _get(cfg, "model.name", None)
    if name is None:
        name = _get(cfg, "model.type", None)
    if name is None:
        raise KeyError("Config requires 'model.name' (or 'model.type').")
    return str(name)


def build_from_config(cfg: Dict[str, Any]) -> Any:
    """Instantiate model from config dict."""
    name = get_model_name_from_config(cfg)
    builder = get_builder(name)
    return builder(cfg)
