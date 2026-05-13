from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation_project.src.experiments.evaluation import _evaluate_row
from simulation_project.src.utils import FitResult, save_fit_result_artifacts
from simulation_second.src.config import load_benchmark_config
from simulation_second.src.dataset import generate_grouped_dataset


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


def _json_scalar(value):
    if value is None:
        return None
    try:
        val = float(value)
    except Exception:
        return value
    return val if math.isfinite(val) else None


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
    save_artifacts: bool = False,
) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.diagnostics import summary as numpyro_summary
    from numpyro.infer import MCMC, NUTS

    numpyro.set_host_device_count(int(chains))
    try:
        jax.config.update("jax_enable_x64", True)
    except Exception:
        pass

    total_t0 = time.perf_counter()
    cfg = load_benchmark_config(config)
    setting = cfg.setting_map()[str(setting_id)]
    ds = generate_grouped_dataset(
        setting,
        replicate_id=int(replicate),
        master_seed=int(cfg.runner.seed),
        family_specs=cfg.families,
    )
    Xs, ys, x_mean, x_scale, y_mean = _standardize_train(ds.X_train, ds.y_train)
    group_id = _group_id(ds.groups, Xs.shape[1])
    X_j = jnp.asarray(Xs)
    y_j = jnp.asarray(ys)
    gid_j = jnp.asarray(group_id)
    n, p = Xs.shape
    G = int(max(group_id) + 1)

    def model(X, y, gid):
        p_use = X.shape[1]
        G_use = int(G)
        sigma = numpyro.sample("sigma", dist.HalfCauchy(1.0))
        tau = numpyro.sample("tau", dist.HalfCauchy(1.0))
        group_lambda = numpyro.sample("group_scale", dist.HalfCauchy(jnp.ones(G_use)).to_event(1))
        local_lambda = numpyro.sample("lambda", dist.HalfCauchy(jnp.ones(p_use)).to_event(1))
        beta_raw = numpyro.sample("beta_raw", dist.Normal(jnp.zeros(p_use), jnp.ones(p_use)).to_event(1))
        beta_std = numpyro.deterministic(
            "beta_std",
            beta_raw * sigma * tau * group_lambda[gid] * local_lambda,
        )
        numpyro.deterministic("beta", beta_std / jnp.asarray(x_scale))
        mean = X @ beta_std
        numpyro.sample("y", dist.Normal(mean, sigma), obs=y)

    kernel = NUTS(
        model,
        target_accept_prob=float(target_accept),
        max_tree_depth=int(max_tree_depth),
        dense_mass=bool(dense_mass),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=int(warmup),
        num_samples=int(draws),
        num_chains=int(chains),
        progress_bar=False,
    )
    fit_t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(int(cfg.runner.seed) + int(replicate)), X_j, y_j, gid_j)
    runtime_seconds = time.perf_counter() - fit_t0
    samples = mcmc.get_samples(group_by_chain=True)
    diag = numpyro_summary(samples, prob=0.9, group_by_chain=True)

    rhat_vals = []
    ess_vals = []
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
            "chains": int(chains),
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
        ds.beta,
        X_train=ds.X_train,
        y_train=ds.y_train,
        X_test=ds.X_test,
        y_test=ds.y_test,
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
    }
    outdir_path = ROOT / str(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    stem = f"{setting_id}__GHS_plus_NUTS__r{int(replicate)}_w{int(warmup)}_d{int(draws)}"
    out_path = outdir_path / f"{stem}.json"
    if bool(save_artifacts):
        artifacts = save_fit_result_artifacts(
            outdir_path / stem,
            result=result,
            run_context={
                "setting_id": str(setting_id),
                "method": "GHS_plus_NUTS",
                "replicate": int(replicate),
                "source_script": "run_highdim_ghs_plus_nuts_probe.py",
            },
            coefficient_truth=ds.beta,
            dataset_arrays={
                "X_train": ds.X_train,
                "y_train": ds.y_train,
                "X_test": ds.X_test,
                "y_test": ds.y_test,
                "beta": ds.beta,
            },
            dataset_metadata={"groups": [[int(i) for i in g] for g in ds.groups]},
            save_dataset_bundle=True,
        )
        payload["artifacts"] = artifacts
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"out_path": str(out_path), **payload}


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
    parser.add_argument("--save-artifacts", action="store_true")
    args = parser.parse_args()
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
        save_artifacts=bool(args.save_artifacts),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
