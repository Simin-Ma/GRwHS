from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation_project.src.core.diagnostics.convergence import summarize_convergence
from simulation_project.src.experiments.methods.fit_gr_rhs import fit_gr_rhs
from simulation_project.src.utils import FitResult, SamplerConfig, save_fit_result_artifacts
from simulation_second.src.config import load_benchmark_config
from simulation_second.src.dataset import generate_grouped_dataset


def _json_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Continuation workflow for GR-RHS high-dimensional chains.")
    parser.add_argument("--config", default="Simulation_highdimension/config/highdimension.yaml")
    parser.add_argument("--setting-id", default="hd_setting_1_classical_anchor")
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--outdir", default="tmp/grrhs_highdim_continuation")
    args = parser.parse_args()

    cfg = load_benchmark_config(args.config)
    setting = cfg.setting_map()[str(args.setting_id)]
    ds = generate_grouped_dataset(
        setting,
        replicate_id=int(args.replicate),
        master_seed=int(cfg.runner.seed),
        family_specs=cfg.families,
    )
    p0_groups = int(sum(any(abs(ds.beta[idx]) > 1e-12 for idx in g) for g in ds.groups))
    outdir = ROOT / str(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    payloads: list[dict[str, Any] | None] = [None] * int(args.chains)
    chain_beta_rounds: list[list[np.ndarray]] = [[] for _ in range(int(args.chains))]
    history: list[dict[str, Any]] = []

    total_start = time.perf_counter()
    for round_idx in range(1, int(args.rounds) + 1):
        round_rec: dict[str, Any] = {"round": int(round_idx), "chains": []}
        for chain_idx in range(int(args.chains)):
            sampler = SamplerConfig(
                chains=1,
                warmup=int(args.warmup),
                post_warmup_draws=int(args.draws),
                adapt_delta=0.9,
                max_treedepth=8,
                strict_adapt_delta=0.95,
                strict_max_treedepth=10,
                max_divergence_ratio=0.05,
                rhat_threshold=1.01,
                ess_threshold=20.0,
            )
            t0 = time.perf_counter()
            res = fit_gr_rhs(
                ds.X_train,
                ds.y_train,
                ds.groups,
                task="gaussian",
                seed=int(args.seed) + 1000 * int(round_idx) + chain_idx,
                p0=p0_groups,
                sampler=sampler,
                retry_resume_payload=payloads[chain_idx],
                retry_attempt=max(0, round_idx - 1),
                method_name=f"GR_RHS_chain{chain_idx+1}",
                tau_target="groups",
                sampler_backend="collapsed_profile",
                use_local_scale=False,
                progress_bar=False,
            )
            wall = time.perf_counter() - t0
            chain_draws = None if res.beta_draws is None else np.asarray(res.beta_draws, dtype=float)
            if chain_draws is not None:
                if chain_draws.ndim == 3 and chain_draws.shape[0] == 1:
                    chain_draws = chain_draws[0]
                chain_beta_rounds[chain_idx].append(np.asarray(chain_draws, dtype=float))
            diag = dict(res.diagnostics or {})
            payloads[chain_idx] = diag.get("retry_resume_payload") if isinstance(diag.get("retry_resume_payload"), dict) else None
            round_rec["chains"].append(
                {
                    "chain": int(chain_idx + 1),
                    "wall_seconds": float(wall),
                    "status": str(res.status),
                    "runtime_seconds": float(res.runtime_seconds),
                    "error": str(res.error or ""),
                    "has_beta_draws": bool(res.beta_draws is not None),
                    "resume_payload_present": bool(payloads[chain_idx] is not None),
                }
            )

        good = [idx for idx, chunks in enumerate(chain_beta_rounds) if chunks]
        if len(good) >= 2:
            aligned = [np.concatenate(chain_beta_rounds[idx], axis=0) for idx in good]
            min_draws = min(arr.shape[0] for arr in aligned)
            aligned = [arr[:min_draws] for arr in aligned]
            beta_chains = np.stack(aligned, axis=0)
            conv = summarize_convergence({"beta": beta_chains})
            beta_diag = dict(conv.get("beta", {}))
            round_rec["merged_beta_diag"] = beta_diag
            round_rec["merged_shape"] = [int(x) for x in beta_chains.shape]
        history.append(round_rec)
        _json_dump({"history": history}, outdir / "continuation_history.json")

    good = [idx for idx, chunks in enumerate(chain_beta_rounds) if chunks]
    if len(good) < 2:
        print(json.dumps({"error": "Need at least two successful chains to summarize convergence."}, indent=2))
        return 1

    aligned = [np.concatenate(chain_beta_rounds[idx], axis=0) for idx in good]
    min_draws = min(arr.shape[0] for arr in aligned)
    aligned = [arr[:min_draws] for arr in aligned]
    beta_chains = np.stack(aligned, axis=0)
    conv = summarize_convergence({"beta": beta_chains})
    beta_mean = beta_chains.reshape(-1, beta_chains.shape[-1]).mean(axis=0)
    final = FitResult(
        method="GR_RHS_continuation",
        status="ok",
        beta_mean=beta_mean,
        beta_draws=beta_chains,
        kappa_draws=None,
        group_scale_draws=None,
        tau_draws=None,
        runtime_seconds=float(time.perf_counter() - total_start),
        rhat_max=float(conv["beta"].get("rhat_max", float("nan"))),
        bulk_ess_min=float(conv["beta"].get("ess_min", float("nan"))),
        divergence_ratio=float("nan"),
        converged=False,
        diagnostics={
            "continuation_history": history,
            "convergence_detail": conv,
            "continuation_config": {
                "chains": int(args.chains),
                "rounds": int(args.rounds),
                "warmup_per_round": int(args.warmup),
                "draws_per_round": int(args.draws),
                "setting_id": str(args.setting_id),
            },
        },
    )
    written = save_fit_result_artifacts(outdir / "final", result=final, coefficient_truth=ds.beta)
    payload = {
        "artifact_dir": written.get("fit_dir", ""),
        "fit_summary": written.get("fit_summary", ""),
        "convergence_beta_summary": written.get("convergence_beta_summary", ""),
        "rhat_max": float(final.rhat_max),
        "ess_min": float(final.bulk_ess_min),
        "runtime_seconds": float(final.runtime_seconds),
        "merged_shape": [int(x) for x in beta_chains.shape],
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
