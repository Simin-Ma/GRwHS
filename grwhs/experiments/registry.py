from __future__ import annotations
"""Model registry and builders for experiments.

This module exposes a simple registry to construct models from config dicts.
Optional models (SVI/Gibbs) are imported in try blocks so that baseline-only
environments can still import this module without errors.
"""

from typing import Callable, Dict, Any, Optional

from data.generators import make_groups

# Optional: GRwHS main models (if available)
try:
    from grwhs.models.grwhs_svi_numpyro import GRwHS_SVI  # type: ignore
except Exception:  # pragma: no cover
    GRwHS_SVI = None  # type: ignore

try:
    from grwhs.models.grwhs_gibbs import GRwHS_Gibbs  # type: ignore
except Exception:  # pragma: no cover
    GRwHS_Gibbs = None  # type: ignore

# Baselines (numpy/skglm implementations)
from grwhs.models.baselines import (
    Ridge,
    Lasso,
    ElasticNet,
    GroupLasso,
    SparseGroupLasso,
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
    scale_global = float(_get(cfg, "model.scale_global", 1.0))
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
    progress_bar = bool(_get(cfg, "model.progress_bar", _get(cfg, "runtime.progress_bar", False)))
    seed_val = _get(
        cfg,
        "model.seed",
        _get(cfg, "inference.seed", _get(cfg, "runtime.seed", _get(cfg, "seed", None))),
    )

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
        "progress_bar": progress_bar,
    }
    if seed_val is not None:
        kwargs["seed"] = int(seed_val)
    return kwargs


# ------------------------------
# Baselines
# ------------------------------


@register("ridge")
def _build_ridge(cfg: Dict[str, Any]) -> Any:
    alpha = float(_get(cfg, "model.alpha", 1.0))
    fit_intercept = bool(_get(cfg, "model.fit_intercept", False))
    return Ridge(alpha=alpha, fit_intercept=fit_intercept)


@register("lasso")
def _build_lasso(cfg: Dict[str, Any]) -> Any:
    alpha = float(_get(cfg, "model.alpha", 1.0))
    fit_intercept = bool(_get(cfg, "model.fit_intercept", False))
    max_iter = int(_get(cfg, "model.max_iter", 10_000))
    tol = float(_get(cfg, "model.tol", 1e-6))
    warm_start = bool(_get(cfg, "model.warm_start", True))
    return Lasso(
        alpha=alpha,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        tol=tol,
        warm_start=warm_start,
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


@register("group_lasso")
@register("grouplasso")  # alias
def _build_group_lasso(cfg: Dict[str, Any]) -> Any:
    # groups: List[List[int]] required
    groups = _infer_groups(cfg)
    if groups is None:
        raise ValueError("GroupLasso requires 'data.groups' in config (list of index lists).")
    alpha = float(_get(cfg, "model.alpha", 1.0))
    fit_intercept = bool(_get(cfg, "model.fit_intercept", False))
    max_iter = int(_get(cfg, "model.max_iter", 2_000))
    max_epochs = int(_get(cfg, "model.max_epochs", 50_000))
    p0 = int(_get(cfg, "model.p0", 10))
    tol = float(_get(cfg, "model.tol", 1e-6))
    warm_start = bool(_get(cfg, "model.warm_start", True))
    ws_strategy = str(_get(cfg, "model.ws_strategy", "fixpoint"))
    verbose = int(_get(cfg, "model.verbose", 0))
    positive = bool(_get(cfg, "model.positive", False))

    # group_weights (optional), defaults to sqrt(|G_g|)
    gw = _get(cfg, "model.group_weights", None)
    return GroupLasso(
        groups=groups,
        alpha=alpha,
        group_weights=gw,
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
    group_weights = _get(cfg, "model.group_weights", None)
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
    return RegularizedHorseshoeRegression(**kwargs)


# ------------------------------
# GRwHS main models (if available)
# ------------------------------


@register("grwhs_svi")
def _build_grwhs_svi(cfg: Dict[str, Any]) -> Any:
    if GRwHS_SVI is None:
        raise ImportError("GRwHS_SVI is not available. Ensure grwhs.models.grwhs_svi_numpyro exists.")
    # Key hyperparameters from config with defaults
    c = float(_get(cfg, "model.c", 1.0))
    tau0 = float(_get(cfg, "model.tau0", 0.1))
    eta = float(_get(cfg, "model.eta", 0.5))
    s0 = float(_get(cfg, "model.s0", 1.0))
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

    svi = GRwHS_SVI(
        c=c,
        tau0=tau0,
        eta=eta,
        s0=s0,
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


@register("grwhs_gibbs")
def _build_grwhs_gibbs(cfg: Dict[str, Any]) -> Any:
    if GRwHS_Gibbs is None:
        raise ImportError("GRwHS_Gibbs is not available. Ensure grwhs.models.grwhs_gibbs exists.")
    c = float(_get(cfg, "model.c", 1.0))
    tau0 = float(_get(cfg, "model.tau0", 0.1))
    eta = float(_get(cfg, "model.eta", 0.5))
    s0 = float(_get(cfg, "model.s0", 1.0))
    iters = int(_get(cfg, "model.iters", 2000))
    seed = _get(cfg, "runtime.seed", 42)
    sampler = GRwHS_Gibbs(c=c, tau0=tau0, eta=eta, s0=s0, iters=iters, seed=seed)  # type: ignore
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
    Examples: ridge / lasso / elastic_net / group_lasso / sparse_group_lasso / grwhs_svi / grwhs_gibbs
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
