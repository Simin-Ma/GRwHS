from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from simulation_project.src.experiments.dgp.grouped_linear import generate_heterogeneity_dataset
from simulation_project.src.experiments.fitting import _fit_with_convergence_retry
from simulation_project.src.experiments.runtime import (
    _EXP5_DEFAULT_MAX_CONV_RETRIES,
    _attempts_used,
    _result_diag_fields,
    _sampler_for_exp5,
    _sampler_for_standard,
)
from simulation_project.src.experiments.methods.fit_gr_rhs import fit_gr_rhs
from simulation_project.src.utils import MASTER_SEED, experiment_seed


def _run_variant(
    *,
    setting_id: int,
    group_sizes: list[int],
    mu: list[float],
    alpha_kappa: float,
    beta_kappa: float,
    use_group_scale: bool,
    label: str,
) -> dict[str, object]:
    sampler = _sampler_for_exp5(_sampler_for_standard())
    p0_signal_groups = int(sum(v > 0.0 for v in mu))
    seed = experiment_seed(5, setting_id, 1, master_seed=MASTER_SEED)
    ds = generate_heterogeneity_dataset(
        n=100,
        group_sizes=group_sizes,
        rho_within=0.3,
        rho_between=0.05,
        sigma2=1.0,
        mu=mu,
        seed=seed,
    )
    res = _fit_with_convergence_retry(
        lambda st, att, _resume=None: fit_gr_rhs(
            ds["X"],
            ds["y"],
            ds["groups"],
            task="gaussian",
            seed=seed + 100 + 100 * att,
            p0=p0_signal_groups,
            sampler=st,
            alpha_kappa=float(alpha_kappa),
            beta_kappa=float(beta_kappa),
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
    row: dict[str, object] = {
        "setting_id": int(setting_id),
        "variant": str(label),
        "alpha_kappa": float(alpha_kappa),
        "beta_kappa": float(beta_kappa),
        "use_group_scale": bool(use_group_scale),
        "status": str(res.status),
        "converged": bool(res.converged),
        "fit_attempts": int(_attempts_used(res)),
        **_result_diag_fields(res),
    }
    if res.kappa_draws is not None:
        kd = np.asarray(res.kappa_draws, dtype=float)
        if kd.ndim > 2:
            kd = kd.reshape(-1, kd.shape[-1])
        row["kappa_group_means"] = json.dumps(np.mean(kd, axis=0).round(6).tolist())
    else:
        row["kappa_group_means"] = ""
    if res.group_scale_draws is not None:
        ad = np.asarray(res.group_scale_draws, dtype=float)
        if ad.ndim > 2:
            ad = ad.reshape(-1, ad.shape[-1])
        row["a_group_means"] = json.dumps(np.mean(ad, axis=0).round(6).tolist())
    else:
        row["a_group_means"] = ""
    return row


def main() -> None:
    out_dir = Path("outputs/simulation_project/exp5_structural_probe")
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        (1, [10, 10, 10, 10, 10], [0.0, 0.0, 1.5, 4.0, 10.0]),
        (2, [30, 10, 5, 3, 2], [0.0, 0.0, 1.5, 4.0, 10.0]),
    ]
    priors = [
        (0.5, 1.0),
        (1.0, 1.0),
        (2.0, 5.0),
    ]
    variants = [
        ("baseline", True),
        ("no_ag", False),
    ]

    rows: list[dict[str, object]] = []
    for setting_id, group_sizes, mu in scenarios:
        for alpha_kappa, beta_kappa in priors:
            for label, use_group_scale in variants:
                rows.append(
                    _run_variant(
                        setting_id=setting_id,
                        group_sizes=list(group_sizes),
                        mu=list(mu),
                        alpha_kappa=float(alpha_kappa),
                        beta_kappa=float(beta_kappa),
                        use_group_scale=bool(use_group_scale),
                        label=label,
                    )
                )

    csv_path = out_dir / "probe_results.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {
        "n_rows": len(rows),
        "csv": str(csv_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(csv_path)


if __name__ == "__main__":
    main()
