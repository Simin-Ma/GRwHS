from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation_project.src.experiments.evaluation import _evaluate_row
from simulation_project.src.utils import FitResult, save_fit_result_artifacts
from simulation_second.src.config import load_benchmark_config
from simulation_second.src.dataset import generate_grouped_dataset


@dataclass
class GHSPlusCaseState:
    cfg: object
    ds: object
    X_j: object
    y_j: object
    gid_j: object
    model: object
    chains: int


def _standardize_train(X: np.ndarray, y: np.ndarray):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    x_mean = X.mean(axis=0)
    Xc = X - x_mean
    x_scale = Xc.std(axis=0, ddof=0)
    x_scale = np.where(x_scale < 1e-8, 1.0, x_scale)
    y_mean = float(y.mean())
    return Xc / x_scale, y - y_mean, x_mean, x_scale, y_mean


def _group_id(groups, p: int) -> np.ndarray:
    gid = np.zeros(int(p), dtype=np.int32)
    for i, g in enumerate(groups):
        gid[np.asarray(g, dtype=int)] = int(i)
    return gid


def _ghs_plus_model(X, y, gid, *, x_scale, n_groups: int):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    p_use = X.shape[1]
    sigma = numpyro.sample("sigma", dist.HalfCauchy(1.0))
    tau = numpyro.sample("tau", dist.HalfCauchy(1.0))
    group_lambda = numpyro.sample("group_scale", dist.HalfCauchy(jnp.ones(int(n_groups))).to_event(1))
    local_lambda = numpyro.sample("lambda", dist.HalfCauchy(jnp.ones(p_use)).to_event(1))
    beta_raw = numpyro.sample("beta_raw", dist.Normal(jnp.zeros(p_use), jnp.ones(p_use)).to_event(1))
    beta_std = numpyro.deterministic(
        "beta_std",
        beta_raw * sigma * tau * group_lambda[gid] * local_lambda,
    )
    numpyro.deterministic("beta", beta_std / jnp.asarray(x_scale))
    numpyro.sample("y", dist.Normal(X @ beta_std, sigma), obs=y)


def _json_scalar(value):
    if value is None:
        return None
    try:
        val = float(value)
    except Exception:
        return value
    return val if math.isfinite(val) else None


def _json_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_budget_spec(spec: str) -> dict[str, object]:
    parts = [part.strip() for part in str(spec).split(":")]
    if len(parts) < 2 or len(parts) > 5:
        raise ValueError("budget spec must be warmup:draws[:max_tree_depth[:target_accept[:dense_mass]]]")
    out: dict[str, object] = {
        "warmup": int(parts[0]),
        "draws": int(parts[1]),
        "max_tree_depth": None,
        "target_accept": None,
        "dense_mass": None,
    }
    if len(parts) >= 3 and parts[2]:
        out["max_tree_depth"] = int(parts[2])
    if len(parts) >= 4 and parts[3]:
        out["target_accept"] = float(parts[3])
    if len(parts) >= 5 and parts[4]:
        out["dense_mass"] = _json_bool(parts[4])
    return out


def _prepare_case_state(
    *,
    config: str,
    setting_id: str,
    replicate: int,
    chains: int,
) -> GHSPlusCaseState:
    import jax
    import jax.numpy as jnp
    import numpyro

    numpyro.set_host_device_count(int(chains))
    try:
        jax.config.update("jax_enable_x64", True)
    except Exception:
        pass

    cfg = load_benchmark_config(config)
    setting = cfg.setting_map()[str(setting_id)]
    ds = generate_grouped_dataset(
        setting,
        replicate_id=int(replicate),
        master_seed=int(cfg.runner.seed),
        family_specs=cfg.families,
    )
    Xs, ys, _, x_scale, _ = _standardize_train(ds.X_train, ds.y_train)
    group_id = _group_id(ds.groups, Xs.shape[1])
    model = partial(
        _ghs_plus_model,
        x_scale=jnp.asarray(x_scale),
        n_groups=int(max(group_id) + 1),
    )
    return GHSPlusCaseState(
        cfg=cfg,
        ds=ds,
        X_j=jnp.asarray(Xs),
        y_j=jnp.asarray(ys),
        gid_j=jnp.asarray(group_id),
        model=model,
        chains=int(chains),
    )


def _run_prepared_budget(
    state: GHSPlusCaseState,
    *,
    setting_id: str,
    replicate: int,
    outdir: str,
    warmup: int,
    draws: int,
    max_tree_depth: int,
    target_accept: float,
    dense_mass: bool,
    seed_offset: int,
    save_artifacts: bool,
) -> dict[str, object]:
    import jax
    from numpyro.diagnostics import summary as numpyro_summary
    from numpyro.infer import MCMC, NUTS

    total_t0 = time.perf_counter()
    kernel = NUTS(
        state.model,
        target_accept_prob=float(target_accept),
        max_tree_depth=int(max_tree_depth),
        dense_mass=bool(dense_mass),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=int(warmup),
        num_samples=int(draws),
        num_chains=int(state.chains),
        progress_bar=False,
    )
    fit_t0 = time.perf_counter()
    run_seed = int(state.cfg.runner.seed) + int(replicate) + 10000 * int(seed_offset)
    mcmc.run(jax.random.PRNGKey(run_seed), state.X_j, state.y_j, state.gid_j)
    runtime_seconds = time.perf_counter() - fit_t0
    samples = mcmc.get_samples(group_by_chain=True)
    diag = numpyro_summary(samples, prob=0.9, group_by_chain=True)

    rhat_vals: list[float] = []
    ess_vals: list[float] = []
    for name in ("beta", "tau", "group_scale", "lambda", "sigma"):
        item = diag.get(name)
        if item is None:
            continue
        rhat = np.asarray(item.get("r_hat"), dtype=float).reshape(-1)
        ess = np.asarray(item.get("n_eff"), dtype=float).reshape(-1)
        rhat_vals.extend(rhat[np.isfinite(rhat)].tolist())
        ess_vals.extend(ess[np.isfinite(ess)].tolist())
    rhat_max = float(max(rhat_vals)) if rhat_vals else float("nan")
    ess_min = float(min(ess_vals)) if ess_vals else float("nan")
    converged = bool(np.isfinite(rhat_max) and rhat_max < 1.01 and np.isfinite(ess_min) and ess_min > 400.0)

    beta_draws = np.asarray(samples["beta"], dtype=float)
    beta_mean = beta_draws.reshape(-1, beta_draws.shape[-1]).mean(axis=0)
    result = FitResult(
        method="GHS_plus_NUTS",
        status="ok",
        beta_mean=beta_mean,
        beta_draws=beta_draws,
        kappa_draws=None,
        group_scale_draws=np.asarray(samples["group_scale"], dtype=float),
        tau_draws=np.asarray(samples["tau"], dtype=float),
        runtime_seconds=float(runtime_seconds),
        rhat_max=float(rhat_max),
        bulk_ess_min=float(ess_min),
        divergence_ratio=float("nan"),
        converged=bool(converged),
        diagnostics={
            "backend": "numpyro_nuts",
            "tracked_params": ["beta", "tau", "group_scale", "lambda", "sigma"],
            "warmup": int(warmup),
            "draws": int(draws),
            "chains": int(state.chains),
            "dense_mass": bool(dense_mass),
            "max_tree_depth": int(max_tree_depth),
            "target_accept": float(target_accept),
            "param_diagnostics": {
                name: {
                    "rhat_max": _json_scalar(np.nanmax(np.asarray(item.get("r_hat"), dtype=float))),
                    "ess_min": _json_scalar(np.nanmin(np.asarray(item.get("n_eff"), dtype=float))),
                }
                for name, item in diag.items()
                if name in {"beta", "tau", "group_scale", "lambda", "sigma"}
            },
        },
    )
    metrics = _evaluate_row(
        result,
        state.ds.beta,
        X_train=state.ds.X_train,
        y_train=state.ds.y_train,
        X_test=state.ds.X_test,
        y_test=state.ds.y_test,
    )
    payload = {
        "replicate": int(replicate),
        "setting_id": str(setting_id),
        "method": "GHS_plus_NUTS",
        "status": "ok",
        "converged": bool(converged),
        "rhat_max": _json_scalar(rhat_max),
        "ess_min": _json_scalar(ess_min),
        "runtime_seconds": float(runtime_seconds),
        "wall_seconds": float(time.perf_counter() - total_t0),
        "mse_overall": _json_scalar(metrics.get("mse_overall")),
        "mse_signal": _json_scalar(metrics.get("mse_signal")),
        "mse_null": _json_scalar(metrics.get("mse_null")),
        "coverage_95": _json_scalar(metrics.get("coverage_95")),
        "lpd_test": _json_scalar(metrics.get("lpd_test")),
        "diagnostics": result.diagnostics,
        "probe_budget": {
            "warmup": int(warmup),
            "draws": int(draws),
            "chains": int(state.chains),
            "max_tree_depth": int(max_tree_depth),
            "target_accept": float(target_accept),
            "dense_mass": bool(dense_mass),
            "seed_offset": int(seed_offset),
        },
    }
    outdir_path = ROOT / str(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    seed_part = "" if int(seed_offset) == 0 else f"_s{int(seed_offset)}"
    stem = f"{setting_id}__GHS_plus_NUTS__r{int(replicate)}_w{int(warmup)}_d{int(draws)}{seed_part}"
    out_path = outdir_path / f"{stem}.json"
    if bool(save_artifacts):
        payload["artifacts"] = save_fit_result_artifacts(
            outdir_path / stem,
            result=result,
            run_context={
                "setting_id": str(setting_id),
                "method": "GHS_plus_NUTS",
                "replicate": int(replicate),
                "source_script": "run_highdim_ghs_plus_nuts_probe.py",
            },
            coefficient_truth=state.ds.beta,
            dataset_arrays={
                "X_train": state.ds.X_train,
                "y_train": state.ds.y_train,
                "X_test": state.ds.X_test,
                "y_test": state.ds.y_test,
                "beta": state.ds.beta,
            },
            dataset_metadata={"groups": [[int(i) for i in g] for g in state.ds.groups]},
            save_dataset_bundle=True,
        )
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"out_path": str(out_path), **payload}


def run_case(
    *,
    config: str,
    setting_id: str,
    replicate: int,
    outdir: str,
    warmup: int,
    draws: int,
    chains: int,
    max_tree_depth: int,
    target_accept: float,
    dense_mass: bool,
    seed_offset: int = 0,
    save_artifacts: bool = False,
) -> dict[str, object]:
    state = _prepare_case_state(
        config=config,
        setting_id=setting_id,
        replicate=replicate,
        chains=chains,
    )
    return _run_prepared_budget(
        state,
        setting_id=setting_id,
        replicate=replicate,
        outdir=outdir,
        warmup=warmup,
        draws=draws,
        max_tree_depth=max_tree_depth,
        target_accept=target_accept,
        dense_mass=dense_mass,
        seed_offset=seed_offset,
        save_artifacts=save_artifacts,
    )


def run_scan(
    *,
    config: str,
    setting_id: str,
    replicate: int,
    outdir: str,
    chains: int,
    seed_offset: int,
    save_artifacts: bool,
    budgets: list[dict[str, object]],
) -> dict[str, object]:
    scan_t0 = time.perf_counter()
    state = _prepare_case_state(
        config=config,
        setting_id=setting_id,
        replicate=replicate,
        chains=chains,
    )
    results: list[dict[str, object]] = []
    for idx, budget in enumerate(budgets, start=1):
        payload = _run_prepared_budget(
            state,
            setting_id=setting_id,
            replicate=replicate,
            outdir=outdir,
            warmup=int(budget["warmup"]),
            draws=int(budget["draws"]),
            max_tree_depth=int(budget["max_tree_depth"]),
            target_accept=float(budget["target_accept"]),
            dense_mass=bool(budget["dense_mass"]),
            seed_offset=seed_offset,
            save_artifacts=save_artifacts,
        )
        payload["scan_index"] = int(idx)
        results.append(payload)

    converged = [row for row in results if bool(row.get("converged"))]
    best = None
    if converged:
        row = min(converged, key=lambda item: float(item.get("wall_seconds", float("inf"))))
        best = {
            "warmup": int(row["probe_budget"]["warmup"]),
            "draws": int(row["probe_budget"]["draws"]),
            "max_tree_depth": int(row["probe_budget"]["max_tree_depth"]),
            "target_accept": float(row["probe_budget"]["target_accept"]),
            "dense_mass": bool(row["probe_budget"]["dense_mass"]),
            "wall_seconds": _json_scalar(row.get("wall_seconds")),
            "rhat_max": _json_scalar(row.get("rhat_max")),
            "ess_min": _json_scalar(row.get("ess_min")),
            "out_path": str(row.get("out_path", "")),
        }
    summary = {
        "setting_id": str(setting_id),
        "replicate": int(replicate),
        "method": "GHS_plus_NUTS",
        "n_runs": int(len(results)),
        "n_converged": int(len(converged)),
        "scan_wall_seconds": float(time.perf_counter() - scan_t0),
        "best": best,
    }
    summary_path = ROOT / str(outdir) / f"{setting_id}__GHS_plus_NUTS__r{int(replicate)}__scan_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"scan_summary": summary, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {"scan_summary_path": str(summary_path), "scan_summary": summary, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe GHS+ full posterior with NumPyro NUTS.")
    parser.add_argument("--config", default="Simulation_highdimension/config/highdimension.yaml")
    parser.add_argument("--setting-id", default="hd_setting_1_classical_anchor")
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--outdir", default="tmp/highdim_ghs_plus_nuts_probe")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--max-tree-depth", type=int, default=10)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--dense-mass", action="store_true")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--save-artifacts", action="store_true")
    parser.add_argument(
        "--budget",
        action="append",
        default=None,
        help="Scan spec: warmup:draws[:max_tree_depth[:target_accept[:dense_mass]]]. Repeat for multiple budgets in one process.",
    )
    args = parser.parse_args()
    if args.budget:
        budgets = []
        for spec in args.budget:
            item = _parse_budget_spec(spec)
            if item["max_tree_depth"] is None:
                item["max_tree_depth"] = int(args.max_tree_depth)
            if item["target_accept"] is None:
                item["target_accept"] = float(args.target_accept)
            if item["dense_mass"] is None:
                item["dense_mass"] = bool(args.dense_mass)
            budgets.append(item)
        payload = run_scan(
            config=str(args.config),
            setting_id=str(args.setting_id),
            replicate=int(args.replicate),
            outdir=str(args.outdir),
            chains=int(args.chains),
            seed_offset=int(args.seed_offset),
            save_artifacts=bool(args.save_artifacts),
            budgets=budgets,
        )
    else:
        payload = run_case(
            config=str(args.config),
            setting_id=str(args.setting_id),
            replicate=int(args.replicate),
            outdir=str(args.outdir),
            warmup=int(args.warmup),
            draws=int(args.draws),
            chains=int(args.chains),
            max_tree_depth=int(args.max_tree_depth),
            target_accept=float(args.target_accept),
            dense_mass=bool(args.dense_mass),
            seed_offset=int(args.seed_offset),
            save_artifacts=bool(args.save_artifacts),
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
