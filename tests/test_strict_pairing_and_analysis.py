from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from simulation_project.src.experiments.analysis.report import run_analysis
from simulation_project.src.experiments.evaluation import _evaluate_row
from simulation_project.src.experiments.reporting import _paired_converged_subset
from simulation_project.src.utils import FitResult


def test_paired_subset_requires_converged_and_status_ok() -> None:
    raw = pd.DataFrame(
        [
            {"setting_id": 1, "replicate_id": 1, "method": "GR_RHS", "converged": True, "status": "ok", "mse_null": 1.0, "mse_signal": 1.2, "mse_overall": 1.1, "lpd_test": -1.0},
            {"setting_id": 1, "replicate_id": 1, "method": "RHS", "converged": True, "status": "ok", "mse_null": 1.3, "mse_signal": 1.4, "mse_overall": 1.35, "lpd_test": -1.5},
            {"setting_id": 1, "replicate_id": 2, "method": "GR_RHS", "converged": True, "status": "ok", "mse_null": 0.9, "mse_signal": 1.1, "mse_overall": 1.0, "lpd_test": -0.9},
            {"setting_id": 1, "replicate_id": 2, "method": "RHS", "converged": True, "status": "error", "mse_null": 1.5, "mse_signal": 1.6, "mse_overall": 1.55, "lpd_test": -1.8},
        ]
    )

    paired, stats = _paired_converged_subset(
        raw,
        group_cols=["setting_id"],
        method_col="method",
        replicate_col="replicate_id",
        converged_col="converged",
        required_cols=["mse_null", "mse_signal", "mse_overall", "lpd_test"],
        method_levels=["GR_RHS", "RHS"],
    )

    assert sorted(set(paired["replicate_id"].tolist())) == [1]
    assert int(stats.iloc[0]["n_total_replicates"]) == 2
    assert int(stats.iloc[0]["n_common_replicates"]) == 1


def test_evaluate_row_exposes_lpd_test_ppd() -> None:
    res = FitResult(
        method="GR_RHS",
        status="ok",
        beta_mean=np.asarray([0.8], dtype=float),
        beta_draws=np.asarray([[0.7], [0.8], [0.9]], dtype=float),
        kappa_draws=None,
        group_scale_draws=None,
        runtime_seconds=1.0,
        rhat_max=1.0,
        bulk_ess_min=500.0,
        divergence_ratio=0.0,
        converged=True,
    )
    beta0 = np.asarray([1.0], dtype=float)
    X_train = np.asarray([[1.0], [2.0], [3.0]], dtype=float)
    y_train = np.asarray([1.0, 2.1, 3.0], dtype=float)
    X_test = np.asarray([[1.5], [2.5]], dtype=float)
    y_test = np.asarray([1.6, 2.6], dtype=float)

    out = _evaluate_row(res, beta0, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    assert "lpd_test_ppd" in out
    assert "lpd_test_plugin" in out
    assert np.isfinite(float(out["lpd_test_ppd"]))
    assert float(out["lpd_test"]) == float(out["lpd_test_ppd"])


def test_analysis_writes_diagnostics_table_and_gate(tmp_path: Path) -> None:
    base = tmp_path / "sim_outputs"
    res = base / "results"
    res.mkdir(parents=True, exist_ok=True)

    specs = [
        ("exp2_group_separation", "method", "GR_RHS"),
        ("exp3a_main_benchmark", "method", "GR_RHS"),
        ("exp3b_boundary_stress", "method", "GR_RHS"),
        ("exp3c_highdim_stress", "method", "GR_RHS"),
        ("exp4_variant_ablation", "method_type", "GR_RHS"),
        ("exp5_prior_sensitivity", "prior_id", "1"),
    ]
    fields = ["converged", "status", "runtime_seconds", "bulk_ess_min", "rhat_max", "divergence_ratio"]
    for subdir, col, value in specs:
        exp_dir = res / subdir
        exp_dir.mkdir(parents=True, exist_ok=True)
        with (exp_dir / "raw_results.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[col] + fields)
            writer.writeheader()
            writer.writerow(
                {
                    col: value,
                    "converged": "False",
                    "status": "error",
                    "runtime_seconds": "10.0",
                    "bulk_ess_min": "100.0",
                    "rhat_max": "1.2",
                    "divergence_ratio": "0.1",
                }
            )

    metrics = run_analysis(save_dir=str(base))
    gate = metrics.get("strict_convergence_gate", {})
    assert gate.get("overall_pass") is False

    diag_path = res / "diagnostics_runtime_table.csv"
    assert diag_path.exists()
    rows = list(csv.DictReader(diag_path.open("r", encoding="utf-8", newline="")))
    assert len(rows) > 0

    report_path = res / "analysis_report.txt"
    assert report_path.exists()
    txt = report_path.read_text(encoding="utf-8")
    assert "Strict Convergence Gate" in txt
