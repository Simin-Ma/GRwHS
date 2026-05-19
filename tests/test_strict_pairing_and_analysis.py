from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from simulation_second.src.bayes_kernel.experiments.evaluation import _evaluate_row
from simulation_second.src.bayes_kernel.experiments.reporting import _paired_converged_subset
from simulation_second.src.bayes_kernel.utils import FitResult
from simulation_second.src.reporting import build_summary


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


def test_current_reporting_builds_method_summary(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        [
            {
                "setting_id": "setting_a",
                "replicate_id": 1,
                "method": "GR_RHS",
                "status": "ok",
                "converged": True,
                "mse_null": 0.5,
                "mse_signal": 0.8,
                "mse_overall": 0.65,
                "lpd_test": -1.0,
                "runtime_seconds": 2.0,
            },
            {
                "setting_id": "setting_a",
                "replicate_id": 1,
                "method": "RHS",
                "status": "ok",
                "converged": True,
                "mse_null": 0.7,
                "mse_signal": 0.9,
                "mse_overall": 0.8,
                "lpd_test": -1.2,
                "runtime_seconds": 3.0,
            },
        ]
    )

    summary = build_summary(
        raw,
        group_cols=["setting_id"],
        method_order=["GR_RHS", "RHS"],
    )
    out_path = tmp_path / "method_summary.csv"
    summary.to_csv(out_path, index=False)

    assert out_path.exists()
    assert set(summary["method"].astype(str)) == {"GR_RHS", "RHS"}
    assert "rank_mse_overall" in summary.columns
