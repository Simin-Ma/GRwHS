from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import logging
import math
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.random import Generator, default_rng
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from grrhs.utils.logging_utils import progress

_EPS = 1e-12
_MIN_POS = 1e-10


def _sample_invgamma(alpha: float, beta: float, rng: Generator) -> float:
    """InvGamma(alpha, beta), shape-scale parameterization."""
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("InvGamma requires alpha > 0 and beta > 0.")
    z = rng.gamma(shape=alpha, scale=1.0 / beta)
    return float(1.0 / max(z, _MIN_POS))


def _fit_grrhs_chain_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    model = GRRHS_Gibbs(
        c=float(payload["c"]),
        tau0=float(payload["tau0"]),
        eta=float(payload["eta"]),
        s0=float(payload["s0"]),
        use_groups=bool(payload["use_groups"]),
        alpha_c=float(payload["alpha_c"]),
        beta_c=float(payload["beta_c"]),
        iters=int(payload["iters"]),
        burnin=int(payload["burnin"]),
        thin=int(payload["thin"]),
        seed=int(payload["seed"]),
        num_chains=1,
        slice_w=float(payload["slice_w"]),
        slice_m=int(payload["slice_m"]),
        tau_slice_w=float(payload["tau_slice_w"]),
        tau_slice_m=int(payload["tau_slice_m"]),
        jitter=float(payload["jitter"]),
        mh_sd_log_tau2=float(payload["mh_sd_log_tau2"]),
        mh_sd_log_lambda=float(payload["mh_sd_log_lambda"]),
        mh_sd_log_a=float(payload["mh_sd_log_a"]),
        mh_sd_log_c2=float(payload["mh_sd_log_c2"]),
        use_pcabs_lite=bool(payload.get("use_pcabs_lite", True)),
        use_collapsed_scale_updates=bool(payload.get("use_collapsed_scale_updates", True)),
        global_block_sd_u=float(payload.get("global_block_sd_u", 0.05)),
        global_block_sd_alpha=float(payload.get("global_block_sd_alpha", 0.1)),
        global_comp_sd=float(payload.get("global_comp_sd", 0.05)),
        group_ac2_block_sd_alpha=float(payload.get("group_ac2_block_sd_alpha", 0.1)),
        group_ac2_block_sd_xi=float(payload.get("group_ac2_block_sd_xi", 0.1)),
        group_comp_sd=float(payload.get("group_comp_sd", 0.08)),
        use_lambda_slice=bool(payload.get("use_lambda_slice", False)),
        lambda_slice_w=payload.get("lambda_slice_w"),
        lambda_slice_m=payload.get("lambda_slice_m"),
        continuation_state=payload.get("continuation_state"),
        adapt_proposals=bool(payload.get("adapt_proposals", True)),
        adapt_interval=int(payload.get("adapt_interval", 50)),
        adapt_until_frac=float(payload.get("adapt_until_frac", 0.8)),
        adapt_target_accept=float(payload.get("adapt_target_accept", 0.30)),
        adapt_target_accept_by_block=payload.get("adapt_target_accept_by_block"),
        adapt_step_size=float(payload.get("adapt_step_size", 0.05)),
        adapt_only_during_burnin=bool(payload.get("adapt_only_during_burnin", True)),
        min_proposal_sd=float(payload.get("min_proposal_sd", 1e-3)),
        max_proposal_sd=float(payload.get("max_proposal_sd", 2.5)),
    )
    fitted = model.fit(
        np.asarray(payload["X"], dtype=float),
        np.asarray(payload["y"], dtype=float),
        groups=payload["groups"],
    )
    return {
        "groups": fitted.groups_,
        "group_id": None if fitted.group_id_ is None else np.asarray(fitted.group_id_, dtype=int),
        "group_sizes": None if fitted.group_sizes_ is None else np.asarray(fitted.group_sizes_, dtype=int),
        "coef_samples": None if fitted.coef_samples_ is None else np.asarray(fitted.coef_samples_),
        "sigma2_samples": None if fitted.sigma2_samples_ is None else np.asarray(fitted.sigma2_samples_),
        "tau_samples": None if fitted.tau_samples_ is None else np.asarray(fitted.tau_samples_),
        "phi_samples": None if fitted.phi_samples_ is None else np.asarray(fitted.phi_samples_),
        "lambda_samples": None if fitted.lambda_samples_ is None else np.asarray(fitted.lambda_samples_),
        "a_samples": None if fitted.a_samples_ is None else np.asarray(fitted.a_samples_),
        "c2_samples": None if fitted.c2_samples_ is None else np.asarray(fitted.c2_samples_),
        "coef_mean": None if fitted.coef_mean_ is None else np.asarray(fitted.coef_mean_),
        "intercept": float(fitted.intercept_),
        "final_state": fitted.get_continuation_state(),
        "sampler_diagnostics": dict(getattr(fitted, "sampler_diagnostics_", {}) or {}),
    }


@dataclass
class GRRHS_Gibbs:
    """
    Group Regularized Horseshoe (GR-RHS) Metropolis-within-Gibbs sampler.

    Hierarchy implemented:
      beta_j | lambda_j, a_g, c_g^2, tau ~ N(0, tau^2 * tilde_lambda_{j,g}^2)
      tilde_lambda_{j,g}^2 = c_g^2 * lambda_j^2 * a_g^2 / (c_g^2 + tau^2 * lambda_j^2 * a_g^2)
      lambda_j ~ HC(0, 1)
      a_g ~ HN(0, s_{a,g}^2), with s_{a,g} = eta / sqrt(p_g)
      c_g^2 ~ IG(alpha_c, beta_c)
      tau ~ HC(0, tau0), via IG augmentation with nu
      p(sigma^2) ∝ 1 / sigma^2
    """

    # Backward-compatible public hyperparameters
    c: float = 1.0
    tau0: float = 0.1
    eta: float = 0.5
    s0: float = 1.0
    use_groups: bool = True

    # New slab prior parameters
    alpha_c: float = 2.0
    beta_c: float = 2.0

    # Sampling controls
    iters: int = 2000
    burnin: Optional[int] = None
    thin: int = 1
    seed: int = 42
    num_chains: int = 1

    # Kept for compatibility with existing configs; mapped to MH step sizes
    slice_w: float = 0.25
    slice_m: int = 100
    tau_slice_w: float = 0.2
    tau_slice_m: int = 200
    jitter: float = 1e-10

    # Explicit MH proposal SDs in log space
    mh_sd_log_tau2: Optional[float] = None
    mh_sd_log_lambda: Optional[float] = None
    mh_sd_log_a: Optional[float] = None
    mh_sd_log_c2: Optional[float] = None

    # PCABS-lite controls (non-collapsed blocked/compensation moves)
    use_pcabs_lite: bool = True
    use_collapsed_scale_updates: bool = True
    global_block_sd_u: Optional[float] = None
    global_block_sd_alpha: Optional[float] = None
    global_comp_sd: Optional[float] = None
    group_ac2_block_sd_alpha: Optional[float] = None
    group_ac2_block_sd_xi: Optional[float] = None
    group_comp_sd: Optional[float] = None
    use_lambda_slice: bool = False
    lambda_slice_w: Optional[float] = None
    lambda_slice_m: Optional[int] = None
    continuation_state: Optional[Mapping[str, Any]] = None
    adapt_proposals: bool = True
    adapt_interval: int = 50
    adapt_until_frac: float = 0.8
    adapt_target_accept: float = 0.30
    adapt_target_accept_by_block: Optional[Mapping[str, float]] = None
    adapt_step_size: float = 0.05
    adapt_only_during_burnin: bool = True
    min_proposal_sd: float = 1e-3
    max_proposal_sd: float = 2.5

    rng: Generator = field(init=False)
    coef_samples_: Optional[np.ndarray] = field(default=None, init=False)
    sigma2_samples_: Optional[np.ndarray] = field(default=None, init=False)
    tau_samples_: Optional[np.ndarray] = field(default=None, init=False)
    phi_samples_: Optional[np.ndarray] = field(default=None, init=False)  # alias of a_samples_
    lambda_samples_: Optional[np.ndarray] = field(default=None, init=False)
    a_samples_: Optional[np.ndarray] = field(default=None, init=False)
    c2_samples_: Optional[np.ndarray] = field(default=None, init=False)

    coef_mean_: Optional[np.ndarray] = field(default=None, init=False)
    intercept_: float = field(default=0.0, init=False)

    groups_: Optional[List[List[int]]] = field(default=None, init=False)
    group_id_: Optional[np.ndarray] = field(default=None, init=False)
    group_sizes_: Optional[np.ndarray] = field(default=None, init=False)
    chain_final_states_: Optional[List[Dict[str, Any]]] = field(default=None, init=False)
    sampler_diagnostics_: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.rng = default_rng(self.seed)
        if self.burnin is None:
            self.burnin = self.iters // 2
        if self.burnin < 0 or self.burnin >= self.iters:
            raise ValueError("burnin must satisfy 0 <= burnin < iters.")
        if self.thin <= 0:
            raise ValueError("thin must be positive.")
        if self.c <= 0.0:
            raise ValueError("c must be positive.")
        if self.tau0 <= 0.0:
            raise ValueError("tau0 must be positive.")
        if self.eta <= 0.0:
            raise ValueError("eta must be positive.")
        if self.alpha_c <= 0.0 or self.beta_c <= 0.0:
            raise ValueError("alpha_c and beta_c must be positive.")
        if self.num_chains <= 0:
            raise ValueError("num_chains must be positive.")
        if self.jitter <= 0.0:
            raise ValueError("jitter must be positive.")
        if self.adapt_interval <= 0:
            raise ValueError("adapt_interval must be positive.")
        if not (0.0 < float(self.adapt_until_frac) <= 1.0):
            raise ValueError("adapt_until_frac must lie in (0, 1].")
        if not (0.0 < float(self.adapt_target_accept) < 1.0):
            raise ValueError("adapt_target_accept must lie in (0, 1).")
        if float(self.adapt_step_size) <= 0.0:
            raise ValueError("adapt_step_size must be positive.")
        if self.adapt_target_accept_by_block is not None:
            for key, value in dict(self.adapt_target_accept_by_block).items():
                v = float(value)
                if not (0.0 < v < 1.0):
                    raise ValueError(f"adapt_target_accept_by_block[{key!r}] must lie in (0, 1).")
        if float(self.min_proposal_sd) <= 0.0 or float(self.max_proposal_sd) <= 0.0:
            raise ValueError("min_proposal_sd/max_proposal_sd must be positive.")
        if float(self.min_proposal_sd) > float(self.max_proposal_sd):
            raise ValueError("min_proposal_sd cannot exceed max_proposal_sd.")

        if self.mh_sd_log_tau2 is None:
            self.mh_sd_log_tau2 = float(max(self.tau_slice_w, 1e-3))
        if self.mh_sd_log_lambda is None:
            self.mh_sd_log_lambda = float(max(self.slice_w, 1e-3))
        if self.mh_sd_log_a is None:
            self.mh_sd_log_a = float(max(self.slice_w, 1e-3))
        if self.mh_sd_log_c2 is None:
            self.mh_sd_log_c2 = float(max(self.slice_w, 1e-3))
        if self.global_block_sd_u is None:
            self.global_block_sd_u = float(max(0.5 * self.mh_sd_log_tau2, 1e-3))
        if self.global_block_sd_alpha is None:
            self.global_block_sd_alpha = float(max(self.mh_sd_log_a, 1e-3))
        if self.global_comp_sd is None:
            self.global_comp_sd = float(max(0.25 * self.mh_sd_log_a, 1e-3))
        if self.group_ac2_block_sd_alpha is None:
            self.group_ac2_block_sd_alpha = float(max(self.mh_sd_log_a, 1e-3))
        if self.group_ac2_block_sd_xi is None:
            self.group_ac2_block_sd_xi = float(max(self.mh_sd_log_c2, 1e-3))
        if self.group_comp_sd is None:
            self.group_comp_sd = float(max(0.5 * self.mh_sd_log_lambda, 1e-3))
        if self.lambda_slice_w is None:
            self.lambda_slice_w = float(max(0.15, 0.5 * self.slice_w))
        if self.lambda_slice_m is None:
            self.lambda_slice_m = int(max(20, min(60, self.slice_m)))
        if self.adapt_target_accept_by_block is None:
            self.adapt_target_accept_by_block = {}
        else:
            self.adapt_target_accept_by_block = {
                str(k): float(v) for k, v in dict(self.adapt_target_accept_by_block).items()
            }

    def _init_mh_stats(self) -> None:
        keys = (
            "tau2",
            "lambda",
            "a",
            "c2",
            "global_block",
            "global_comp",
            "group_ac2_block",
            "group_comp",
        )
        self._mh_stats = {
            key: {"attempted": 0, "accepted": 0, "window_attempted": 0, "window_accepted": 0}
            for key in keys
        }

    def _record_mh(self, key: Optional[str], accepted: bool) -> None:
        if key is None:
            return
        stats = getattr(self, "_mh_stats", None)
        if not isinstance(stats, dict):
            return
        slot = stats.get(str(key))
        if not isinstance(slot, dict):
            return
        slot["attempted"] = int(slot.get("attempted", 0)) + 1
        slot["window_attempted"] = int(slot.get("window_attempted", 0)) + 1
        if accepted:
            slot["accepted"] = int(slot.get("accepted", 0)) + 1
            slot["window_accepted"] = int(slot.get("window_accepted", 0)) + 1

    def _proposal_bindings(self) -> Dict[str, Tuple[str, ...]]:
        return {
            "tau2": ("mh_sd_log_tau2",),
            "lambda": ("mh_sd_log_lambda",),
            "a": ("mh_sd_log_a",),
            "c2": ("mh_sd_log_c2",),
            "global_block": ("global_block_sd_u", "global_block_sd_alpha"),
            "global_comp": ("global_comp_sd",),
            "group_ac2_block": ("group_ac2_block_sd_alpha", "group_ac2_block_sd_xi"),
            "group_comp": ("group_comp_sd",),
        }

    def _adapt_proposals_if_needed(self, *, iteration: int) -> None:
        if not bool(self.adapt_proposals):
            return
        if bool(self.adapt_only_during_burnin) and iteration >= int(self.burnin):
            return
        max_adapt_iter = int(max(1, math.floor(float(self.iters) * float(self.adapt_until_frac))))
        if iteration >= max_adapt_iter:
            return
        if (iteration + 1) % int(self.adapt_interval) != 0:
            return
        stats = getattr(self, "_mh_stats", None)
        if not isinstance(stats, dict):
            return
        default_target = float(self.adapt_target_accept)
        step = float(self.adapt_step_size)
        sd_min = float(self.min_proposal_sd)
        sd_max = float(self.max_proposal_sd)
        target_by_block = dict(getattr(self, "adapt_target_accept_by_block", {}) or {})
        for key, attrs in self._proposal_bindings().items():
            slot = stats.get(key)
            if not isinstance(slot, dict):
                continue
            attempted = int(slot.get("window_attempted", 0))
            accepted = int(slot.get("window_accepted", 0))
            if attempted <= 0:
                continue
            acc_rate = float(accepted / max(attempted, 1))
            target = float(target_by_block.get(str(key), default_target))
            factor = math.exp(step * (acc_rate - target))
            for attr in attrs:
                cur = float(getattr(self, attr))
                new = float(min(max(cur * factor, sd_min), sd_max))
                setattr(self, attr, new)
            slot["window_attempted"] = 0
            slot["window_accepted"] = 0

    def _build_sampler_diagnostics(self, *, runtime_sec: float, kept_draws: int) -> Dict[str, Any]:
        stats = getattr(self, "_mh_stats", {})
        mh_acceptance: Dict[str, Dict[str, float]] = {}
        if isinstance(stats, dict):
            for key, slot in stats.items():
                if not isinstance(slot, dict):
                    continue
                attempted = int(slot.get("attempted", 0))
                accepted = int(slot.get("accepted", 0))
                mh_acceptance[str(key)] = {
                    "attempted": attempted,
                    "accepted": accepted,
                    "rate": float(accepted / attempted) if attempted > 0 else float("nan"),
                }
        proposal_scales = {
            "mh_sd_log_tau2": float(self.mh_sd_log_tau2),
            "mh_sd_log_lambda": float(self.mh_sd_log_lambda),
            "mh_sd_log_a": float(self.mh_sd_log_a),
            "mh_sd_log_c2": float(self.mh_sd_log_c2),
            "global_block_sd_u": float(self.global_block_sd_u),
            "global_block_sd_alpha": float(self.global_block_sd_alpha),
            "global_comp_sd": float(self.global_comp_sd),
            "group_ac2_block_sd_alpha": float(self.group_ac2_block_sd_alpha),
            "group_ac2_block_sd_xi": float(self.group_ac2_block_sd_xi),
            "group_comp_sd": float(self.group_comp_sd),
        }
        return {
            "backend": "grrhs_mwg",
            "runtime_sec": float(max(runtime_sec, 0.0)),
            "kept_draws_per_chain": int(kept_draws),
            "mh_acceptance": mh_acceptance,
            "proposal_scales": proposal_scales,
            "adaptation": {
                "enabled": bool(self.adapt_proposals),
                "interval": int(self.adapt_interval),
                "until_frac": float(self.adapt_until_frac),
                "target_accept": float(self.adapt_target_accept),
                "target_accept_by_block": dict(getattr(self, "adapt_target_accept_by_block", {}) or {}),
                "step_size": float(self.adapt_step_size),
                "only_during_burnin": bool(self.adapt_only_during_burnin),
            },
        }

    def get_continuation_state(self) -> Dict[str, Any]:
        state = getattr(self, "_final_state", None)
        if not isinstance(state, Mapping):
            return {}
        out: Dict[str, Any] = {}
        for key, value in state.items():
            if isinstance(value, np.ndarray):
                out[key] = np.asarray(value, dtype=float).copy()
            else:
                out[key] = value
        return out

    def set_chain_initial_states(self, states: Sequence[Mapping[str, Any]]) -> None:
        self._chain_init_states = [dict(s) for s in states]

    @staticmethod
    def _flatten_scalar_draws(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        data = np.asarray(arr, dtype=float)
        return data.reshape(-1)

    @staticmethod
    def _flatten_param_draws(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        data = np.asarray(arr, dtype=float)
        if data.ndim <= 1:
            return data.reshape(1, -1)
        if data.ndim == 2:
            return data
        return data.reshape(-1, *data.shape[2:])

    @staticmethod
    def _stack_chain_draws(arrays: List[Optional[np.ndarray]]) -> Optional[np.ndarray]:
        if not arrays or arrays[0] is None:
            return None
        return np.stack([np.asarray(arr) for arr in arrays], axis=0)

    def _spawn_single_chain(self, *, seed: int) -> "GRRHS_Gibbs":
        return type(self)(
            c=self.c,
            tau0=self.tau0,
            eta=self.eta,
            s0=self.s0,
            use_groups=self.use_groups,
            alpha_c=self.alpha_c,
            beta_c=self.beta_c,
            iters=self.iters,
            burnin=self.burnin,
            thin=self.thin,
            seed=seed,
            num_chains=1,
            slice_w=self.slice_w,
            slice_m=self.slice_m,
            tau_slice_w=self.tau_slice_w,
            tau_slice_m=self.tau_slice_m,
            jitter=self.jitter,
            mh_sd_log_tau2=self.mh_sd_log_tau2,
            mh_sd_log_lambda=self.mh_sd_log_lambda,
            mh_sd_log_a=self.mh_sd_log_a,
            mh_sd_log_c2=self.mh_sd_log_c2,
            use_pcabs_lite=self.use_pcabs_lite,
            use_collapsed_scale_updates=self.use_collapsed_scale_updates,
            global_block_sd_u=self.global_block_sd_u,
            global_block_sd_alpha=self.global_block_sd_alpha,
            global_comp_sd=self.global_comp_sd,
            group_ac2_block_sd_alpha=self.group_ac2_block_sd_alpha,
            group_ac2_block_sd_xi=self.group_ac2_block_sd_xi,
            group_comp_sd=self.group_comp_sd,
            use_lambda_slice=self.use_lambda_slice,
            lambda_slice_w=self.lambda_slice_w,
            lambda_slice_m=self.lambda_slice_m,
            continuation_state=self.continuation_state,
            adapt_proposals=self.adapt_proposals,
            adapt_interval=self.adapt_interval,
            adapt_until_frac=self.adapt_until_frac,
            adapt_target_accept=self.adapt_target_accept,
            adapt_target_accept_by_block=self.adapt_target_accept_by_block,
            adapt_step_size=self.adapt_step_size,
            adapt_only_during_burnin=self.adapt_only_during_burnin,
            min_proposal_sd=self.min_proposal_sd,
            max_proposal_sd=self.max_proposal_sd,
        )

    def _fit_multichain(self, X: np.ndarray, y: np.ndarray, groups: Optional[List[List[int]]] = None) -> "GRRHS_Gibbs":
        payloads: List[Dict[str, Any]] = []
        groups_payload = None if groups is None else [list(group) for group in groups]
        for chain_idx in range(int(self.num_chains)):
            continuation_state = None
            chain_init_states = getattr(self, "_chain_init_states", None)
            if isinstance(chain_init_states, Sequence) and chain_idx < len(chain_init_states):
                continuation_state = chain_init_states[chain_idx]
            payloads.append(
                {
                    "c": self.c,
                    "tau0": self.tau0,
                    "eta": self.eta,
                    "s0": self.s0,
                    "use_groups": self.use_groups,
                    "alpha_c": self.alpha_c,
                    "beta_c": self.beta_c,
                    "iters": self.iters,
                    "burnin": self.burnin,
                    "thin": self.thin,
                    "seed": int(self.seed) + chain_idx,
                    "slice_w": self.slice_w,
                    "slice_m": self.slice_m,
                    "tau_slice_w": self.tau_slice_w,
                    "tau_slice_m": self.tau_slice_m,
                    "jitter": self.jitter,
                    "mh_sd_log_tau2": self.mh_sd_log_tau2,
                    "mh_sd_log_lambda": self.mh_sd_log_lambda,
                    "mh_sd_log_a": self.mh_sd_log_a,
                    "mh_sd_log_c2": self.mh_sd_log_c2,
                    "use_pcabs_lite": self.use_pcabs_lite,
                    "use_collapsed_scale_updates": self.use_collapsed_scale_updates,
                    "global_block_sd_u": self.global_block_sd_u,
                    "global_block_sd_alpha": self.global_block_sd_alpha,
                    "global_comp_sd": self.global_comp_sd,
                    "group_ac2_block_sd_alpha": self.group_ac2_block_sd_alpha,
                    "group_ac2_block_sd_xi": self.group_ac2_block_sd_xi,
                    "group_comp_sd": self.group_comp_sd,
                    "use_lambda_slice": self.use_lambda_slice,
                    "lambda_slice_w": self.lambda_slice_w,
                    "lambda_slice_m": self.lambda_slice_m,
                    "continuation_state": continuation_state,
                    "adapt_proposals": self.adapt_proposals,
                    "adapt_interval": self.adapt_interval,
                    "adapt_until_frac": self.adapt_until_frac,
                    "adapt_target_accept": self.adapt_target_accept,
                    "adapt_target_accept_by_block": dict(self.adapt_target_accept_by_block or {}),
                    "adapt_step_size": self.adapt_step_size,
                    "adapt_only_during_burnin": self.adapt_only_during_burnin,
                    "min_proposal_sd": self.min_proposal_sd,
                    "max_proposal_sd": self.max_proposal_sd,
                    "X": np.asarray(X, dtype=float),
                    "y": np.asarray(y, dtype=float),
                    "groups": groups_payload,
                }
            )

        try:
            with ProcessPoolExecutor(max_workers=int(self.num_chains)) as executor:
                chain_results = list(executor.map(_fit_grrhs_chain_task, payloads))
        except Exception:
            chain_results = [_fit_grrhs_chain_task(payload) for payload in payloads]

        lead = chain_results[0]
        self.groups_ = lead["groups"]
        self.group_id_ = None if lead["group_id"] is None else np.asarray(lead["group_id"], dtype=int).copy()
        self.group_sizes_ = None if lead["group_sizes"] is None else np.asarray(lead["group_sizes"], dtype=int).copy()
        self.coef_samples_ = self._stack_chain_draws([item["coef_samples"] for item in chain_results])
        self.sigma2_samples_ = self._stack_chain_draws([item["sigma2_samples"] for item in chain_results])
        self.tau_samples_ = self._stack_chain_draws([item["tau_samples"] for item in chain_results])
        self.lambda_samples_ = self._stack_chain_draws([item["lambda_samples"] for item in chain_results])
        self.a_samples_ = self._stack_chain_draws([item["a_samples"] for item in chain_results])
        self.c2_samples_ = self._stack_chain_draws([item["c2_samples"] for item in chain_results])
        self.phi_samples_ = self._stack_chain_draws([item["phi_samples"] for item in chain_results])
        self.chain_final_states_ = [dict(item.get("final_state") or {}) for item in chain_results]
        chain_diags = [dict(item.get("sampler_diagnostics") or {}) for item in chain_results]
        self.sampler_diagnostics_ = self._aggregate_chain_diagnostics(chain_diags)
        coef_draws = self._flatten_param_draws(self.coef_samples_)
        self.coef_mean_ = None if coef_draws is None else coef_draws.mean(axis=0)
        self.intercept_ = float(lead["intercept"])
        return self

    @staticmethod
    def _aggregate_chain_diagnostics(chain_diags: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not chain_diags:
            return {}
        runtime = [float(d.get("runtime_sec", 0.0)) for d in chain_diags if isinstance(d, Mapping)]
        kept = [int(d.get("kept_draws_per_chain", 0)) for d in chain_diags if isinstance(d, Mapping)]
        out: Dict[str, Any] = {
            "backend": "grrhs_mwg_multichain",
            "num_chains": int(len(chain_diags)),
            "runtime_sec_total": float(np.sum(runtime)) if runtime else float("nan"),
            "runtime_sec_mean": float(np.mean(runtime)) if runtime else float("nan"),
            "kept_draws_per_chain": int(np.max(kept)) if kept else 0,
        }
        accept_attempts: Dict[str, int] = {}
        accept_accepted: Dict[str, int] = {}
        for diag in chain_diags:
            mh = diag.get("mh_acceptance")
            if not isinstance(mh, Mapping):
                continue
            for key, stat in mh.items():
                if not isinstance(stat, Mapping):
                    continue
                attempted = int(stat.get("attempted", 0))
                accepted = int(stat.get("accepted", 0))
                accept_attempts[str(key)] = int(accept_attempts.get(str(key), 0) + attempted)
                accept_accepted[str(key)] = int(accept_accepted.get(str(key), 0) + accepted)
        if accept_attempts:
            out["mh_acceptance"] = {
                key: {
                    "attempted": int(accept_attempts[key]),
                    "accepted": int(accept_accepted.get(key, 0)),
                    "rate": (
                        float(accept_accepted.get(key, 0) / accept_attempts[key])
                        if accept_attempts[key] > 0
                        else float("nan")
                    ),
                }
                for key in sorted(accept_attempts.keys())
            }
        proposal_keys = (
            "mh_sd_log_tau2",
            "mh_sd_log_lambda",
            "mh_sd_log_a",
            "mh_sd_log_c2",
            "global_block_sd_u",
            "global_block_sd_alpha",
            "global_comp_sd",
            "group_ac2_block_sd_alpha",
            "group_ac2_block_sd_xi",
            "group_comp_sd",
        )
        proposal_means: Dict[str, float] = {}
        for key in proposal_keys:
            vals: List[float] = []
            for diag in chain_diags:
                prop = diag.get("proposal_scales")
                if isinstance(prop, Mapping) and prop.get(key) is not None:
                    vals.append(float(prop[key]))
            if vals:
                proposal_means[key] = float(np.mean(vals))
        if proposal_means:
            out["proposal_scales_mean"] = proposal_means
        return out

    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[List[List[int]]] = None) -> "GRRHS_Gibbs":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.num_chains > 1:
            return self._fit_multichain(X, y, groups)
        t_start = time.perf_counter()
        self._init_mh_stats()

        n, p = X.shape
        if groups is None:
            groups = [[j] for j in range(p)]
        self.groups_ = groups

        if self.use_groups:
            sampler_groups = [list(g) for g in groups]
        else:
            sampler_groups = [list(range(p))]
        G = len(sampler_groups)

        group_id = np.empty(p, dtype=int)
        group_sizes = np.zeros(G, dtype=int)
        for g, idxs in enumerate(sampler_groups):
            idx_arr = np.asarray(idxs, dtype=int)
            group_id[idx_arr] = g
            group_sizes[g] = len(idx_arr)
        self.group_id_ = group_id
        self.group_sizes_ = group_sizes

        s_a = self.eta / np.sqrt(np.maximum(group_sizes.astype(float), 1.0))

        XtX = X.T @ X
        Xty = X.T @ y

        try:
            beta = np.linalg.solve(XtX + 1e-3 * np.eye(p), Xty)
        except np.linalg.LinAlgError:
            beta = np.zeros(p, dtype=float)

        resid_init = y - X @ beta
        sigma2 = max(float(np.mean(resid_init * resid_init)), self.jitter)
        tau2 = max(self.tau0 * self.tau0, self.jitter)
        nu = max(self.tau0 * self.tau0, self.jitter)

        lam = np.ones(p, dtype=float)
        a = np.maximum(s_a, 1e-3)
        c2 = np.full(G, max(self.c * self.c, 1e-3), dtype=float)

        state = self.continuation_state or {}
        if isinstance(state, Mapping):
            try:
                beta_state = np.asarray(state.get("beta"), dtype=float)
                if beta_state.shape == (p,):
                    beta = beta_state.copy()
            except Exception:
                pass
            for key, arr, shape in (
                ("lam", lam, (p,)),
                ("a", a, (G,)),
                ("c2", c2, (G,)),
            ):
                try:
                    v = np.asarray(state.get(key), dtype=float)
                    if v.shape == shape:
                        arr[:] = np.maximum(v, self.jitter)
                except Exception:
                    pass
            try:
                sigma2 = max(float(state.get("sigma2", sigma2)), self.jitter)
                tau2 = max(float(state.get("tau2", tau2)), self.jitter)
                nu = max(float(state.get("nu", nu)), self.jitter)
            except Exception:
                pass

        kept = max(0, (self.iters - self.burnin + self.thin - 1) // self.thin)
        beta_draws = np.zeros((kept, p), dtype=float)
        sigma2_draws = np.zeros(kept, dtype=float)
        tau_draws = np.zeros(kept, dtype=float)
        lambda_draws = np.zeros((kept, p), dtype=float)
        a_draws = np.zeros((kept, G), dtype=float)
        c2_draws = np.zeros((kept, G), dtype=float)
        keep_i = 0

        tau_trace: List[float] = []
        sigma_trace: List[float] = []

        for it in progress(range(self.iters), total=self.iters, desc="GR-RHS MWG"):
            d = self._prior_precision_vector(lam=lam, a=a, c2=c2, tau2=tau2, group_id=group_id)
            beta = self._sample_beta_conditional(XtX=XtX, Xty=Xty, sigma2=sigma2, prior_prec=d, rng=self.rng)

            resid = y - X @ beta
            rss = float(resid @ resid)
            sigma2 = _sample_invgamma(alpha=0.5 * n, beta=0.5 * max(rss, self.jitter), rng=self.rng)

            nu = _sample_invgamma(alpha=1.0, beta=(1.0 / (self.tau0 * self.tau0)) + (1.0 / max(tau2, self.jitter)), rng=self.rng)

            if self.use_pcabs_lite:
                if self.use_collapsed_scale_updates:
                    tau2, a = self._mh_update_global_u_alpha_block_collapsed(
                        X=X,
                        y=y,
                        sigma2=sigma2,
                        tau2=tau2,
                        a=a,
                        nu=nu,
                        lam=lam,
                        c2=c2,
                        group_id=group_id,
                        s_a=s_a,
                    )
                else:
                    tau2, a = self._mh_update_global_u_alpha_block(
                        tau2=tau2,
                        a=a,
                        nu=nu,
                        beta=beta,
                        lam=lam,
                        c2=c2,
                        group_id=group_id,
                        s_a=s_a,
                    )
            else:
                tau2 = self._mh_update_tau2(
                    tau2=tau2,
                    nu=nu,
                    beta=beta,
                    lam=lam,
                    a=a,
                    c2=c2,
                    group_id=group_id,
                )

            for j in range(p):
                g = group_id[j]
                if self.use_lambda_slice:
                    lam[j] = self._slice_update_lambda_j(
                        lam_j=lam[j],
                        beta_j=beta[j],
                        tau2=tau2,
                        a_g=a[g],
                        c2_g=c2[g],
                    )
                else:
                    lam[j] = self._mh_update_lambda_j(
                        lam_j=lam[j],
                        beta_j=beta[j],
                        tau2=tau2,
                        a_g=a[g],
                        c2_g=c2[g],
                    )

            if self.use_pcabs_lite:
                for g in range(G):
                    idx = np.asarray(sampler_groups[g], dtype=int)
                    a[g], c2[g] = self._mh_update_group_a_c2_block(
                        a_g=a[g],
                        c2_g=c2[g],
                        beta_g=beta[idx],
                        lam_g=lam[idx],
                        tau2=tau2,
                        s_a_g=s_a[g],
                    )
                for g in range(G):
                    idx = np.asarray(sampler_groups[g], dtype=int)
                    a[g], lam[idx] = self._mh_update_group_compensation_a_lambda(
                        a_g=a[g],
                        lam_g=lam[idx],
                        beta_g=beta[idx],
                        tau2=tau2,
                        c2_g=c2[g],
                        s_a_g=s_a[g],
                    )
                if self.use_collapsed_scale_updates:
                    tau2, a = self._mh_update_global_compensation_tau_a_collapsed(
                        X=X,
                        y=y,
                        sigma2=sigma2,
                        tau2=tau2,
                        a=a,
                        nu=nu,
                        lam=lam,
                        c2=c2,
                        group_id=group_id,
                        s_a=s_a,
                    )
                else:
                    tau2, a = self._mh_update_global_compensation_tau_a(
                        tau2=tau2,
                        a=a,
                        nu=nu,
                        beta=beta,
                        lam=lam,
                        c2=c2,
                        group_id=group_id,
                        s_a=s_a,
                    )
            else:
                for g in range(G):
                    idx = np.asarray(sampler_groups[g], dtype=int)
                    a[g] = self._mh_update_a_g(
                        a_g=a[g],
                        beta_g=beta[idx],
                        lam_g=lam[idx],
                        tau2=tau2,
                        c2_g=c2[g],
                        s_a_g=s_a[g],
                    )

                for g in range(G):
                    idx = np.asarray(sampler_groups[g], dtype=int)
                    c2[g] = self._mh_update_c2_g(
                        c2_g=c2[g],
                        beta_g=beta[idx],
                        lam_g=lam[idx],
                        tau2=tau2,
                        a_g=a[g],
                    )

            if self.use_pcabs_lite and self.use_collapsed_scale_updates:
                d = self._prior_precision_vector(lam=lam, a=a, c2=c2, tau2=tau2, group_id=group_id)
                beta = self._sample_beta_conditional(XtX=XtX, Xty=Xty, sigma2=sigma2, prior_prec=d, rng=self.rng)

            tau_trace.append(math.sqrt(max(tau2, self.jitter)))
            sigma_trace.append(math.sqrt(max(sigma2, self.jitter)))

            if it >= self.burnin and ((it - self.burnin) % self.thin == 0):
                beta_draws[keep_i] = beta
                sigma2_draws[keep_i] = sigma2
                tau_draws[keep_i] = math.sqrt(max(tau2, self.jitter))
                lambda_draws[keep_i] = lam
                a_draws[keep_i] = a
                c2_draws[keep_i] = c2
                keep_i += 1
            self._adapt_proposals_if_needed(iteration=it)

        self.coef_samples_ = beta_draws
        self.sigma2_samples_ = sigma2_draws
        self.tau_samples_ = tau_draws
        self.lambda_samples_ = lambda_draws
        self.a_samples_ = a_draws
        self.c2_samples_ = c2_draws
        self.phi_samples_ = a_draws
        self.coef_mean_ = beta_draws.mean(axis=0) if kept > 0 else np.zeros(p, dtype=float)
        self.intercept_ = 0.0

        if tau_trace:
            logger = logging.getLogger(__name__)
            logger.info("tau_trace=%s", np.asarray(tau_trace))
            logger.info("sigma_trace=%s", np.asarray(sigma_trace))
        self._final_state = {
            "beta": np.asarray(beta, dtype=float).copy(),
            "lam": np.asarray(lam, dtype=float).copy(),
            "a": np.asarray(a, dtype=float).copy(),
            "c2": np.asarray(c2, dtype=float).copy(),
            "sigma2": float(sigma2),
            "tau2": float(tau2),
            "nu": float(nu),
        }
        self.chain_final_states_ = [self.get_continuation_state()]
        self.sampler_diagnostics_ = self._build_sampler_diagnostics(
            runtime_sec=float(time.perf_counter() - t_start),
            kept_draws=int(keep_i),
        )

        return self

    def predict(self, X: np.ndarray, use_posterior_mean: bool = True) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if use_posterior_mean:
            if self.coef_mean_ is None:
                raise RuntimeError("Model not fitted.")
            return X @ self.coef_mean_ + self.intercept_
        coef_draws = self._flatten_param_draws(self.coef_samples_)
        if coef_draws is None or coef_draws.shape[0] == 0:
            raise RuntimeError("No posterior samples available.")
        return X @ coef_draws[-1] + self.intercept_

    def get_posterior_summaries(self) -> Dict[str, Any]:
        if self.coef_samples_ is None:
            raise RuntimeError("Not fitted.")
        beta_draws = self._flatten_param_draws(self.coef_samples_)
        sigma2_draws = self._flatten_scalar_draws(self.sigma2_samples_)
        tau_draws = self._flatten_scalar_draws(self.tau_samples_)
        a_draws = self._flatten_param_draws(self.a_samples_)
        c2_draws = self._flatten_param_draws(self.c2_samples_)
        if beta_draws is None:
            raise RuntimeError("Posterior coefficient draws are unavailable.")
        return {
            "beta_mean": beta_draws.mean(axis=0),
            "beta_median": np.median(beta_draws, axis=0),
            "beta_ci95": np.quantile(beta_draws, [0.025, 0.975], axis=0),
            "sigma2_mean": float(sigma2_draws.mean()) if sigma2_draws is not None else None,
            "tau_mean": float(tau_draws.mean()) if tau_draws is not None else None,
            "phi_mean": a_draws.mean(axis=0) if a_draws is not None else None,
            "a_mean": a_draws.mean(axis=0) if a_draws is not None else None,
            "c2_mean": c2_draws.mean(axis=0) if c2_draws is not None else None,
        }

    def _prior_precision_vector(
        self,
        lam: np.ndarray,
        a: np.ndarray,
        c2: np.ndarray,
        tau2: float,
        group_id: np.ndarray,
    ) -> np.ndarray:
        a_map = a[group_id]
        c2_map = c2[group_id]
        tau2_safe = max(float(tau2), self.jitter)
        local = 1.0 / np.maximum(tau2_safe * (lam * lam) * (a_map * a_map), self.jitter)
        slab = 1.0 / np.maximum(c2_map, self.jitter)
        return local + slab

    def _beta_logprior_contrib(self, beta: np.ndarray, d: np.ndarray) -> float:
        return float(0.5 * np.log(np.maximum(d, self.jitter)).sum() - 0.5 * (d * beta * beta).sum())

    def _mh_accept(self, log_target_new: float, log_target_old: float, *, stat_key: Optional[str] = None) -> bool:
        log_alpha = log_target_new - log_target_old
        accepted = False
        if log_alpha >= 0.0:
            accepted = True
        else:
            u = max(self.rng.uniform(), _MIN_POS)
            accepted = bool(math.log(u) < log_alpha)
        self._record_mh(stat_key, accepted)
        return accepted

    def _mh_update_tau2(
        self,
        tau2: float,
        nu: float,
        beta: np.ndarray,
        lam: np.ndarray,
        a: np.ndarray,
        c2: np.ndarray,
        group_id: np.ndarray,
    ) -> float:
        log_tau2 = math.log(max(tau2, self.jitter))
        log_prop = log_tau2 + self.rng.normal(scale=float(self.mh_sd_log_tau2))
        tau2_prop = max(math.exp(log_prop), self.jitter)

        d_old = self._prior_precision_vector(lam=lam, a=a, c2=c2, tau2=tau2, group_id=group_id)
        d_new = self._prior_precision_vector(lam=lam, a=a, c2=c2, tau2=tau2_prop, group_id=group_id)

        lp_old = self._beta_logprior_contrib(beta=beta, d=d_old) + self._log_prior_tau2_given_nu(tau2=tau2, nu=nu) + log_tau2
        lp_new = self._beta_logprior_contrib(beta=beta, d=d_new) + self._log_prior_tau2_given_nu(tau2=tau2_prop, nu=nu) + log_prop
        return tau2_prop if self._mh_accept(lp_new, lp_old, stat_key="tau2") else tau2

    def _scale_block_log_target(
        self,
        *,
        tau2: float,
        a: np.ndarray,
        nu: float,
        beta: np.ndarray,
        lam: np.ndarray,
        c2: np.ndarray,
        group_id: np.ndarray,
        s_a: np.ndarray,
    ) -> float:
        d = self._prior_precision_vector(lam=lam, a=a, c2=c2, tau2=tau2, group_id=group_id)
        lp = self._beta_logprior_contrib(beta=beta, d=d)
        lp += self._log_prior_tau2_given_nu(tau2=tau2, nu=nu)
        lp += float(np.sum([self._log_prior_a(float(a[g]), float(s_a[g])) for g in range(a.size)]))
        # Jacobian terms for u=log(tau) and alpha_g=log(a_g)
        lp += math.log(max(tau2, self.jitter))
        lp += float(np.log(np.maximum(a, self.jitter)).sum())
        return lp

    def _collapsed_log_marginal_y(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
        sigma2: float,
        lam: np.ndarray,
        a: np.ndarray,
        c2: np.ndarray,
        tau2: float,
        group_id: np.ndarray,
    ) -> float:
        sigma2_safe = max(float(sigma2), self.jitter)
        d = self._prior_precision_vector(lam=lam, a=a, c2=c2, tau2=tau2, group_id=group_id)
        v = 1.0 / np.maximum(d, self.jitter)
        Xw = X * np.sqrt(v)[None, :]
        Sigma = sigma2_safe * np.eye(X.shape[0]) + Xw @ Xw.T
        eye_n = np.eye(X.shape[0])
        for jitter_mult in (0.0, 1.0, 10.0, 100.0):
            jitter = jitter_mult * self.jitter
            try:
                chol, lower = cho_factor(Sigma + jitter * eye_n, overwrite_a=False, check_finite=False)
                quad = float(y @ cho_solve((chol, lower), y, check_finite=False))
                logdet = float(2.0 * np.log(np.diag(chol)).sum())
                return -0.5 * (logdet + quad)
            except np.linalg.LinAlgError:
                continue
        return -np.inf

    def _scale_block_log_target_collapsed(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
        sigma2: float,
        tau2: float,
        a: np.ndarray,
        nu: float,
        lam: np.ndarray,
        c2: np.ndarray,
        group_id: np.ndarray,
        s_a: np.ndarray,
    ) -> float:
        lp = self._collapsed_log_marginal_y(
            X=X,
            y=y,
            sigma2=sigma2,
            lam=lam,
            a=a,
            c2=c2,
            tau2=tau2,
            group_id=group_id,
        )
        if not np.isfinite(lp):
            return -np.inf
        lp += self._log_prior_tau2_given_nu(tau2=tau2, nu=nu)
        lp += float(np.sum([self._log_prior_a(float(a[g]), float(s_a[g])) for g in range(a.size)]))
        lp += math.log(max(tau2, self.jitter))
        lp += float(np.log(np.maximum(a, self.jitter)).sum())
        return lp

    def _mh_update_global_u_alpha_block(
        self,
        *,
        tau2: float,
        a: np.ndarray,
        nu: float,
        beta: np.ndarray,
        lam: np.ndarray,
        c2: np.ndarray,
        group_id: np.ndarray,
        s_a: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        u_old = 0.5 * math.log(max(tau2, self.jitter))
        alpha_old = np.log(np.maximum(a, self.jitter))
        u_new = u_old + self.rng.normal(scale=float(self.global_block_sd_u))
        alpha_new = alpha_old + self.rng.normal(scale=float(self.global_block_sd_alpha), size=alpha_old.shape[0])
        tau2_new = max(math.exp(2.0 * u_new), self.jitter)
        a_new = np.maximum(np.exp(alpha_new), self.jitter)

        lp_old = self._scale_block_log_target(
            tau2=tau2,
            a=a,
            nu=nu,
            beta=beta,
            lam=lam,
            c2=c2,
            group_id=group_id,
            s_a=s_a,
        )
        lp_new = self._scale_block_log_target(
            tau2=tau2_new,
            a=a_new,
            nu=nu,
            beta=beta,
            lam=lam,
            c2=c2,
            group_id=group_id,
            s_a=s_a,
        )
        if self._mh_accept(lp_new, lp_old, stat_key="global_block"):
            return tau2_new, a_new
        return tau2, a

    def _mh_update_global_u_alpha_block_collapsed(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
        sigma2: float,
        tau2: float,
        a: np.ndarray,
        nu: float,
        lam: np.ndarray,
        c2: np.ndarray,
        group_id: np.ndarray,
        s_a: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        u_old = 0.5 * math.log(max(tau2, self.jitter))
        alpha_old = np.log(np.maximum(a, self.jitter))
        u_new = u_old + self.rng.normal(scale=float(self.global_block_sd_u))
        alpha_new = alpha_old + self.rng.normal(scale=float(self.global_block_sd_alpha), size=alpha_old.shape[0])
        tau2_new = max(math.exp(2.0 * u_new), self.jitter)
        a_new = np.maximum(np.exp(alpha_new), self.jitter)

        lp_old = self._scale_block_log_target_collapsed(
            X=X,
            y=y,
            sigma2=sigma2,
            tau2=tau2,
            a=a,
            nu=nu,
            lam=lam,
            c2=c2,
            group_id=group_id,
            s_a=s_a,
        )
        lp_new = self._scale_block_log_target_collapsed(
            X=X,
            y=y,
            sigma2=sigma2,
            tau2=tau2_new,
            a=a_new,
            nu=nu,
            lam=lam,
            c2=c2,
            group_id=group_id,
            s_a=s_a,
        )
        if self._mh_accept(lp_new, lp_old, stat_key="global_block"):
            return tau2_new, a_new
        return tau2, a

    def _mh_update_global_compensation_tau_a(
        self,
        *,
        tau2: float,
        a: np.ndarray,
        nu: float,
        beta: np.ndarray,
        lam: np.ndarray,
        c2: np.ndarray,
        group_id: np.ndarray,
        s_a: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        delta = self.rng.normal(scale=float(self.global_comp_sd))
        u_old = 0.5 * math.log(max(tau2, self.jitter))
        alpha_old = np.log(np.maximum(a, self.jitter))
        u_new = u_old + delta
        alpha_new = alpha_old - delta
        tau2_new = max(math.exp(2.0 * u_new), self.jitter)
        a_new = np.maximum(np.exp(alpha_new), self.jitter)

        lp_old = self._scale_block_log_target(
            tau2=tau2,
            a=a,
            nu=nu,
            beta=beta,
            lam=lam,
            c2=c2,
            group_id=group_id,
            s_a=s_a,
        )
        lp_new = self._scale_block_log_target(
            tau2=tau2_new,
            a=a_new,
            nu=nu,
            beta=beta,
            lam=lam,
            c2=c2,
            group_id=group_id,
            s_a=s_a,
        )
        if self._mh_accept(lp_new, lp_old, stat_key="global_comp"):
            return tau2_new, a_new
        return tau2, a

    def _mh_update_global_compensation_tau_a_collapsed(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
        sigma2: float,
        tau2: float,
        a: np.ndarray,
        nu: float,
        lam: np.ndarray,
        c2: np.ndarray,
        group_id: np.ndarray,
        s_a: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        delta = self.rng.normal(scale=float(self.global_comp_sd))
        u_old = 0.5 * math.log(max(tau2, self.jitter))
        alpha_old = np.log(np.maximum(a, self.jitter))
        u_new = u_old + delta
        alpha_new = alpha_old - delta
        tau2_new = max(math.exp(2.0 * u_new), self.jitter)
        a_new = np.maximum(np.exp(alpha_new), self.jitter)

        lp_old = self._scale_block_log_target_collapsed(
            X=X,
            y=y,
            sigma2=sigma2,
            tau2=tau2,
            a=a,
            nu=nu,
            lam=lam,
            c2=c2,
            group_id=group_id,
            s_a=s_a,
        )
        lp_new = self._scale_block_log_target_collapsed(
            X=X,
            y=y,
            sigma2=sigma2,
            tau2=tau2_new,
            a=a_new,
            nu=nu,
            lam=lam,
            c2=c2,
            group_id=group_id,
            s_a=s_a,
        )
        if self._mh_accept(lp_new, lp_old, stat_key="global_comp"):
            return tau2_new, a_new
        return tau2, a

    def _mh_update_lambda_j(
        self,
        lam_j: float,
        beta_j: float,
        tau2: float,
        a_g: float,
        c2_g: float,
    ) -> float:
        log_old = math.log(max(lam_j, self.jitter))
        log_new = log_old + self.rng.normal(scale=float(self.mh_sd_log_lambda))
        lam_new = max(math.exp(log_new), self.jitter)

        d_old = self._single_precision(lam=lam_j, a_g=a_g, c2_g=c2_g, tau2=tau2)
        d_new = self._single_precision(lam=lam_new, a_g=a_g, c2_g=c2_g, tau2=tau2)

        lp_old = 0.5 * math.log(d_old) - 0.5 * d_old * (beta_j * beta_j) + self._log_prior_lambda(lam_j) + log_old
        lp_new = 0.5 * math.log(d_new) - 0.5 * d_new * (beta_j * beta_j) + self._log_prior_lambda(lam_new) + log_new
        return lam_new if self._mh_accept(lp_new, lp_old, stat_key="lambda") else lam_j

    def _slice_update_lambda_j(
        self,
        *,
        lam_j: float,
        beta_j: float,
        tau2: float,
        a_g: float,
        c2_g: float,
    ) -> float:
        log_x0 = math.log(max(lam_j, self.jitter))

        def log_target(log_lam: float) -> float:
            lam_val = max(math.exp(log_lam), self.jitter)
            d_val = self._single_precision(lam=lam_val, a_g=a_g, c2_g=c2_g, tau2=tau2)
            return (
                0.5 * math.log(max(d_val, self.jitter))
                - 0.5 * d_val * (beta_j * beta_j)
                + self._log_prior_lambda(lam_val)
                + log_lam
            )

        return max(
            math.exp(
                self._slice_sample_log_univariate(
                    log_x0=log_x0,
                    logf=log_target,
                    width=float(max(float(self.lambda_slice_w), 1e-3)),
                    max_steps=int(max(int(self.lambda_slice_m), 8)),
                )
            ),
            self.jitter,
        )

    def _slice_sample_log_univariate(
        self,
        *,
        log_x0: float,
        logf: Any,
        width: float,
        max_steps: int,
    ) -> float:
        log_fx0 = float(logf(log_x0))
        if not np.isfinite(log_fx0):
            return log_x0
        log_y = log_fx0 + math.log(max(self.rng.uniform(), _MIN_POS))

        u = self.rng.uniform()
        L = log_x0 - width * u
        R = L + width

        j = int(math.floor(self.rng.uniform() * max_steps))
        k = max_steps - 1 - j

        while j > 0 and np.isfinite(logf(L)) and float(logf(L)) > log_y:
            L -= width
            j -= 1
        while k > 0 and np.isfinite(logf(R)) and float(logf(R)) > log_y:
            R += width
            k -= 1

        for _ in range(max_steps * 10):
            x1 = self.rng.uniform(L, R)
            log_fx1 = float(logf(x1))
            if np.isfinite(log_fx1) and log_fx1 >= log_y:
                return x1
            if x1 < log_x0:
                L = x1
            else:
                R = x1
        return log_x0

    def _mh_update_a_g(
        self,
        a_g: float,
        beta_g: np.ndarray,
        lam_g: np.ndarray,
        tau2: float,
        c2_g: float,
        s_a_g: float,
    ) -> float:
        log_old = math.log(max(a_g, self.jitter))
        log_new = log_old + self.rng.normal(scale=float(self.mh_sd_log_a))
        a_new = max(math.exp(log_new), self.jitter)

        d_old = self._group_precision(lam_g=lam_g, a_g=a_g, c2_g=c2_g, tau2=tau2)
        d_new = self._group_precision(lam_g=lam_g, a_g=a_new, c2_g=c2_g, tau2=tau2)

        lp_old = self._beta_logprior_contrib(beta=beta_g, d=d_old) + self._log_prior_a(a=a_g, s_a=s_a_g) + log_old
        lp_new = self._beta_logprior_contrib(beta=beta_g, d=d_new) + self._log_prior_a(a=a_new, s_a=s_a_g) + log_new
        return a_new if self._mh_accept(lp_new, lp_old, stat_key="a") else a_g

    def _mh_update_c2_g(
        self,
        c2_g: float,
        beta_g: np.ndarray,
        lam_g: np.ndarray,
        tau2: float,
        a_g: float,
    ) -> float:
        log_old = math.log(max(c2_g, self.jitter))
        log_new = log_old + self.rng.normal(scale=float(self.mh_sd_log_c2))
        c2_new = max(math.exp(log_new), self.jitter)

        d_old = self._group_precision(lam_g=lam_g, a_g=a_g, c2_g=c2_g, tau2=tau2)
        d_new = self._group_precision(lam_g=lam_g, a_g=a_g, c2_g=c2_new, tau2=tau2)

        lp_old = self._beta_logprior_contrib(beta=beta_g, d=d_old) + self._log_prior_c2(c2_g) + log_old
        lp_new = self._beta_logprior_contrib(beta=beta_g, d=d_new) + self._log_prior_c2(c2_new) + log_new
        return c2_new if self._mh_accept(lp_new, lp_old, stat_key="c2") else c2_g

    def _mh_update_group_a_c2_block(
        self,
        *,
        a_g: float,
        c2_g: float,
        beta_g: np.ndarray,
        lam_g: np.ndarray,
        tau2: float,
        s_a_g: float,
    ) -> Tuple[float, float]:
        alpha_old = math.log(max(a_g, self.jitter))
        xi_old = math.log(max(c2_g, self.jitter))
        alpha_new = alpha_old + self.rng.normal(scale=float(self.group_ac2_block_sd_alpha))
        xi_new = xi_old + self.rng.normal(scale=float(self.group_ac2_block_sd_xi))
        a_new = max(math.exp(alpha_new), self.jitter)
        c2_new = max(math.exp(xi_new), self.jitter)

        d_old = self._group_precision(lam_g=lam_g, a_g=a_g, c2_g=c2_g, tau2=tau2)
        d_new = self._group_precision(lam_g=lam_g, a_g=a_new, c2_g=c2_new, tau2=tau2)
        lp_old = (
            self._beta_logprior_contrib(beta=beta_g, d=d_old)
            + self._log_prior_a(a=a_g, s_a=s_a_g)
            + self._log_prior_c2(c2_g)
            + alpha_old
            + xi_old
        )
        lp_new = (
            self._beta_logprior_contrib(beta=beta_g, d=d_new)
            + self._log_prior_a(a=a_new, s_a=s_a_g)
            + self._log_prior_c2(c2_new)
            + alpha_new
            + xi_new
        )
        if self._mh_accept(lp_new, lp_old, stat_key="group_ac2_block"):
            return a_new, c2_new
        return a_g, c2_g

    def _mh_update_group_a_c2_block_collapsed(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
        sigma2: float,
        g: int,
        tau2: float,
        lam: np.ndarray,
        a: np.ndarray,
        c2: np.ndarray,
        group_id: np.ndarray,
        s_a: np.ndarray,
    ) -> Tuple[float, float]:
        a_g = float(a[g])
        c2_g = float(c2[g])
        alpha_old = math.log(max(a_g, self.jitter))
        xi_old = math.log(max(c2_g, self.jitter))
        alpha_new = alpha_old + self.rng.normal(scale=float(self.group_ac2_block_sd_alpha))
        xi_new = xi_old + self.rng.normal(scale=float(self.group_ac2_block_sd_xi))
        a_new = max(math.exp(alpha_new), self.jitter)
        c2_new = max(math.exp(xi_new), self.jitter)

        a_prop = a.copy()
        c2_prop = c2.copy()
        a_prop[g] = a_new
        c2_prop[g] = c2_new

        lp_old = self._collapsed_log_marginal_y(
            X=X,
            y=y,
            sigma2=sigma2,
            lam=lam,
            a=a,
            c2=c2,
            tau2=tau2,
            group_id=group_id,
        )
        lp_old += self._log_prior_a(a=a_g, s_a=float(s_a[g])) + self._log_prior_c2(c2_g) + alpha_old + xi_old
        lp_new = self._collapsed_log_marginal_y(
            X=X,
            y=y,
            sigma2=sigma2,
            lam=lam,
            a=a_prop,
            c2=c2_prop,
            tau2=tau2,
            group_id=group_id,
        )
        lp_new += self._log_prior_a(a=a_new, s_a=float(s_a[g])) + self._log_prior_c2(c2_new) + alpha_new + xi_new
        if self._mh_accept(lp_new, lp_old, stat_key="group_ac2_block"):
            return a_new, c2_new
        return a_g, c2_g

    def _mh_update_group_compensation_a_lambda(
        self,
        *,
        a_g: float,
        lam_g: np.ndarray,
        beta_g: np.ndarray,
        tau2: float,
        c2_g: float,
        s_a_g: float,
    ) -> Tuple[float, np.ndarray]:
        p_g = max(int(lam_g.shape[0]), 1)
        delta = self.rng.normal(scale=float(self.group_comp_sd))
        alpha_old = math.log(max(a_g, self.jitter))
        log_lam_old = np.log(np.maximum(lam_g, self.jitter))
        alpha_new = alpha_old + delta
        log_lam_new = log_lam_old - (delta / float(p_g))
        a_new = max(math.exp(alpha_new), self.jitter)
        lam_new = np.maximum(np.exp(log_lam_new), self.jitter)

        d_old = self._group_precision(lam_g=lam_g, a_g=a_g, c2_g=c2_g, tau2=tau2)
        d_new = self._group_precision(lam_g=lam_new, a_g=a_new, c2_g=c2_g, tau2=tau2)
        lp_old = (
            self._beta_logprior_contrib(beta=beta_g, d=d_old)
            + self._log_prior_a(a=a_g, s_a=s_a_g)
            + float(np.sum([self._log_prior_lambda(float(v)) for v in lam_g]))
            + alpha_old
            + float(np.sum(log_lam_old))
        )
        lp_new = (
            self._beta_logprior_contrib(beta=beta_g, d=d_new)
            + self._log_prior_a(a=a_new, s_a=s_a_g)
            + float(np.sum([self._log_prior_lambda(float(v)) for v in lam_new]))
            + alpha_new
            + float(np.sum(log_lam_new))
        )
        if self._mh_accept(lp_new, lp_old, stat_key="group_comp"):
            return a_new, lam_new
        return a_g, lam_g

    def _mh_update_group_compensation_a_lambda_collapsed(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
        sigma2: float,
        g: int,
        tau2: float,
        lam: np.ndarray,
        a: np.ndarray,
        c2: np.ndarray,
        group_id: np.ndarray,
        s_a: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        idx = np.where(group_id == g)[0]
        lam_g = np.asarray(lam[idx], dtype=float)
        p_g = max(int(lam_g.shape[0]), 1)
        a_g = float(a[g])

        delta = self.rng.normal(scale=float(self.group_comp_sd))
        alpha_old = math.log(max(a_g, self.jitter))
        log_lam_old = np.log(np.maximum(lam_g, self.jitter))
        alpha_new = alpha_old + delta
        log_lam_new = log_lam_old - (delta / float(p_g))
        a_new = max(math.exp(alpha_new), self.jitter)
        lam_new = np.maximum(np.exp(log_lam_new), self.jitter)

        a_prop = a.copy()
        lam_prop = lam.copy()
        a_prop[g] = a_new
        lam_prop[idx] = lam_new

        lp_old = self._collapsed_log_marginal_y(
            X=X,
            y=y,
            sigma2=sigma2,
            lam=lam,
            a=a,
            c2=c2,
            tau2=tau2,
            group_id=group_id,
        )
        lp_old += self._log_prior_a(a=a_g, s_a=float(s_a[g]))
        lp_old += float(np.sum([self._log_prior_lambda(float(v)) for v in lam_g]))
        lp_old += alpha_old + float(np.sum(log_lam_old))

        lp_new = self._collapsed_log_marginal_y(
            X=X,
            y=y,
            sigma2=sigma2,
            lam=lam_prop,
            a=a_prop,
            c2=c2,
            tau2=tau2,
            group_id=group_id,
        )
        lp_new += self._log_prior_a(a=a_new, s_a=float(s_a[g]))
        lp_new += float(np.sum([self._log_prior_lambda(float(v)) for v in lam_new]))
        lp_new += alpha_new + float(np.sum(log_lam_new))

        if self._mh_accept(lp_new, lp_old, stat_key="group_comp"):
            return a_new, lam_new
        return a_g, lam_g

    def _log_prior_tau2_given_nu(self, tau2: float, nu: float) -> float:
        t = max(tau2, self.jitter)
        n = max(nu, self.jitter)
        alpha = 0.5
        beta = 1.0 / n
        return alpha * math.log(beta) - math.lgamma(alpha) - (alpha + 1.0) * math.log(t) - beta / t

    @staticmethod
    def _log_prior_lambda(lam: float) -> float:
        return -math.log1p(lam * lam)

    @staticmethod
    def _log_prior_a(a: float, s_a: float) -> float:
        ss = max(s_a, _MIN_POS)
        return -0.5 * (a * a) / (ss * ss)

    def _log_prior_c2(self, c2: float) -> float:
        x = max(c2, self.jitter)
        return (
            self.alpha_c * math.log(self.beta_c)
            - math.lgamma(self.alpha_c)
            - (self.alpha_c + 1.0) * math.log(x)
            - self.beta_c / x
        )

    def _single_precision(self, lam: float, a_g: float, c2_g: float, tau2: float) -> float:
        local = 1.0 / max(tau2 * lam * lam * a_g * a_g, self.jitter)
        slab = 1.0 / max(c2_g, self.jitter)
        return local + slab

    def _group_precision(self, lam_g: np.ndarray, a_g: float, c2_g: float, tau2: float) -> np.ndarray:
        local = 1.0 / np.maximum(tau2 * (lam_g * lam_g) * (a_g * a_g), self.jitter)
        slab = 1.0 / max(c2_g, self.jitter)
        return local + slab

    def _sample_beta_conditional(
        self,
        XtX: np.ndarray,
        Xty: np.ndarray,
        sigma2: float,
        prior_prec: np.ndarray,
        rng: Generator,
    ) -> np.ndarray:
        p = XtX.shape[0]
        sigma2_safe = max(float(sigma2), self.jitter)
        precision = (XtX / sigma2_safe).copy()
        precision.flat[:: p + 1] += np.maximum(prior_prec, self.jitter)

        chol, lower = cho_factor(precision, overwrite_a=False, check_finite=False)
        mean = cho_solve((chol, lower), Xty / sigma2_safe, check_finite=False)

        z = rng.standard_normal(p)
        if lower:
            noise = solve_triangular(chol.T, z, lower=False, check_finite=False)
        else:
            noise = solve_triangular(chol, z, lower=False, check_finite=False)
        return mean + noise
