from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation_project.src.experiments.analysis.metrics import group_auroc, group_l2_score, mse_null_signal_overall
from simulation_project.src.experiments.fitting import _fit_with_convergence_retry
from simulation_project.src.experiments.methods.fit_gr_rhs import fit_gr_rhs
from simulation_project.src.experiments.runtime import (
    _EXP5_DEFAULT_MAX_CONV_RETRIES,
    _attempts_used,
    _sampler_for_exp5,
    _sampler_for_standard,
)
from simulation_project.src.utils import MASTER_SEED, canonical_groups, experiment_seed, sample_correlated_design


def generate_group_amplitude_dataset(
    *,
    n: int,
    group_sizes: list[int],
    rho_within: float,
    rho_between: float,
    sigma2: float,
    group_signal: list[float],
    group_amplitude: list[float],
    seed: int,
) -> dict[str, np.ndarray]:
    X, cov_x = sample_correlated_design(
        n=n,
        group_sizes=group_sizes,
        rho_within=rho_within,
        rho_between=rho_between,
        seed=seed,
    )
    groups = canonical_groups(group_sizes)
    beta = np.zeros(sum(group_sizes), dtype=float)
    for gid, (mu_g, a_g) in enumerate(zip(group_signal, group_amplitude)):
        idx = np.asarray(groups[gid], dtype=int)
        if float(mu_g) <= 0.0:
            continue
        beta[idx] = float(a_g) * math.sqrt((2.0 * float(sigma2) * float(mu_g)) / max(len(idx), 1))
    rng = np.random.default_rng(int(seed) + 23)
    y = X @ beta + rng.normal(loc=0.0, scale=math.sqrt(float(sigma2)), size=int(n))
    return {
        "X": X,
        "y": y,
        "beta0": beta,
        "sigma2": float(sigma2),
        "cov_x": cov_x,
        "groups": groups,
    }


def fit_variant(
    *,
    ds: dict[str, np.ndarray],
    p0_groups: int,
    seed: int,
    use_group_scale: bool,
) -> dict[str, object]:
    sampler = _sampler_for_exp5(_sampler_for_standard())
    res = _fit_with_convergence_retry(
        lambda st, att, _resume=None: fit_gr_rhs(
            ds["X"],
            ds["y"],
            ds["groups"],
            task="gaussian",
            seed=int(seed + 100 * att),
            p0=int(p0_groups),
            sampler=st,
            alpha_kappa=0.5,
            beta_kappa=1.0,
            use_group_scale=bool(use_group_scale),
            use_local_scale=True,
            shared_kappa=False,
            tau_target="groups",
            backend="nuts",
            progress_bar=False,
            retry_resume_payload=_resume,
        ),
        method="GR_RHS",
        sampler=sampler,
        bayes_min_chains=4,
        max_convergence_retries=int(_EXP5_DEFAULT_MAX_CONV_RETRIES),
        enforce_bayes_convergence=True,
        continue_on_retry=True,
    )
    out: dict[str, object] = {
        "status": str(res.status),
        "converged": bool(res.converged),
        "fit_attempts": int(_attempts_used(res)),
        "runtime_seconds": float(res.runtime_seconds),
        "rhat_max": float(res.rhat_max),
        "bulk_ess_min": float(res.bulk_ess_min),
        "divergence_ratio": float(res.divergence_ratio),
        "error": str(res.error),
    }
    if res.beta_mean is not None:
        metrics = mse_null_signal_overall(np.asarray(res.beta_mean, dtype=float), np.asarray(ds["beta0"], dtype=float))
        labels = (np.asarray([np.any(np.abs(ds["beta0"][np.asarray(g, dtype=int)]) > 1e-12) for g in ds["groups"]], dtype=int))
        scores = group_l2_score(np.asarray(res.beta_mean, dtype=float), ds["groups"])
        out.update(metrics)
        out["group_auroc"] = float(group_auroc(scores, labels))
    else:
        out["mse_null"] = float("nan")
        out["mse_signal"] = float("nan")
        out["mse_overall"] = float("nan")
        out["group_auroc"] = float("nan")
    return out


def main() -> None:
    out_dir = ROOT / "outputs" / "simulation_project" / "ag_vs_noag_fairtest"
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        {
            "scenario": "no_group_amplitude",
            "group_sizes": [10, 10, 10, 10, 10],
            "group_signal": [0.0, 0.0, 1.5, 4.0, 10.0],
            "group_amplitude": [1.0, 1.0, 1.0, 1.0, 1.0],
            "repeats": 8,
        },
        {
            "scenario": "moderate_group_amplitude",
            "group_sizes": [10, 10, 10, 10, 10],
            "group_signal": [0.0, 0.0, 1.5, 4.0, 10.0],
            "group_amplitude": [1.0, 1.0, 0.8, 1.1, 1.4],
            "repeats": 8,
        },
        {
            "scenario": "strong_group_amplitude",
            "group_sizes": [10, 10, 10, 10, 10],
            "group_signal": [0.0, 0.0, 1.5, 4.0, 10.0],
            "group_amplitude": [1.0, 1.0, 0.5, 1.0, 2.0],
            "repeats": 8,
        },
    ]

    rows: list[dict[str, object]] = []
    for scen_idx, spec in enumerate(scenarios, start=1):
        p0_groups = int(sum(v > 0 for v in spec["group_signal"]))
        for rep in range(1, int(spec["repeats"]) + 1):
            seed = experiment_seed(95, scen_idx, rep, master_seed=MASTER_SEED)
            ds = generate_group_amplitude_dataset(
                n=100,
                group_sizes=list(spec["group_sizes"]),
                rho_within=0.3,
                rho_between=0.05,
                sigma2=1.0,
                group_signal=list(spec["group_signal"]),
                group_amplitude=list(spec["group_amplitude"]),
                seed=seed,
            )
            for label, use_group_scale in [("with_ag", True), ("no_ag", False)]:
                row = {
                    "scenario": str(spec["scenario"]),
                    "replicate_id": int(rep),
                    "variant": str(label),
                    "group_sizes": json.dumps(spec["group_sizes"]),
                    "group_signal": json.dumps(spec["group_signal"]),
                    "group_amplitude": json.dumps(spec["group_amplitude"]),
                }
                row.update(
                    fit_variant(
                        ds=ds,
                        p0_groups=p0_groups,
                        seed=seed + (11 if use_group_scale else 29),
                        use_group_scale=use_group_scale,
                    )
                )
                rows.append(row)

    csv_path = out_dir / "raw_results.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows: list[dict[str, object]] = []
    for scenario in sorted({str(r["scenario"]) for r in rows}):
        for variant in ("with_ag", "no_ag"):
            sub = [r for r in rows if str(r["scenario"]) == scenario and str(r["variant"]) == variant]
            conv = [r for r in sub if bool(r["converged"])]
            def mean_col(name: str, source: list[dict[str, object]]) -> float:
                vals = [float(r[name]) for r in source if np.isfinite(float(r[name]))]
                return float(np.mean(vals)) if vals else float("nan")
            summary_rows.append(
                {
                    "scenario": scenario,
                    "variant": variant,
                    "n_runs": len(sub),
                    "n_converged": len(conv),
                    "convergence_rate": float(len(conv) / max(len(sub), 1)),
                    "mse_overall_mean_converged": mean_col("mse_overall", conv),
                    "mse_signal_mean_converged": mean_col("mse_signal", conv),
                    "mse_null_mean_converged": mean_col("mse_null", conv),
                    "group_auroc_mean_converged": mean_col("group_auroc", conv),
                    "runtime_mean_all": mean_col("runtime_seconds", sub),
                }
            )

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(summary_path)


if __name__ == "__main__":
    main()
