from __future__ import annotations

import json

from real_data_experiment.src.reporting import build_paired_summary, build_summary
from real_data_experiment.src.runner import finalize_real_data_results_dir
from simulation_project.src.utils import load_pandas


def _warning_raw_frame():
    pd = load_pandas()
    rows = []
    for method, status, converged, rmse in [
        ("GR_RHS_HighDim", "warning", False, 1.0),
        ("RHS_HighDim", "ok", True, 1.5),
    ]:
        rows.append(
            {
                "dataset_id": "toy",
                "dataset_label": "Toy",
                "target_label": "y",
                "covariate_mode": "none",
                "n_train": 8,
                "n_test": 4,
                "feature_count": 3,
                "group_count": 2,
                "notes": "warning rows retain draws",
                "replicate_id": 1,
                "method": method,
                "status": status,
                "converged": converged,
                "rmse_test": rmse,
                "mae_test": rmse,
                "lpd_test": -rmse,
                "r2_test": 0.1,
                "runtime_seconds": 2.0,
                "group_selected_json": "[true,false]",
                "group_scores_json": "[1.0,0.0]",
                "group_labels_json": "[\"g1\",\"g2\"]",
            }
        )
    return pd.DataFrame(rows)


def test_real_data_warning_rows_are_evaluable_but_not_converged() -> None:
    raw = _warning_raw_frame()
    group_cols = ["dataset_id", "dataset_label", "target_label", "covariate_mode", "n_train", "n_test", "feature_count", "group_count", "notes"]
    methods = ["GR_RHS_HighDim", "RHS_HighDim"]
    summary = build_summary(raw, group_cols=group_cols, method_order=methods)
    paired_raw, paired_stats, summary_paired = build_paired_summary(
        raw,
        group_cols=group_cols,
        method_levels=methods,
        required_metric_cols=("rmse_test", "mae_test", "lpd_test", "r2_test"),
        method_order=methods,
    )

    gr = summary.loc[summary["method"].eq("GR_RHS_HighDim")].iloc[0]
    assert int(gr["n_converged"]) == 0
    assert float(gr["rmse_test"]) == 1.0
    assert set(paired_raw["method"]) == set(methods)
    assert int(paired_stats["n_common_replicates"].iloc[0]) == 1
    assert set(summary_paired["summary_scope"]) == {"common_evaluable_paired"}


def test_finalize_results_dir_recovers_incremental_warning_rows(tmp_path) -> None:
    raw = _warning_raw_frame()
    path = tmp_path / "raw_results_incremental.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in raw.to_dict(orient="records")) + "\n", encoding="utf-8")
    (tmp_path / "artifact_catalog_incremental.jsonl").write_text("", encoding="utf-8")

    out = finalize_real_data_results_dir(
        tmp_path,
        method_order=["GR_RHS_HighDim", "RHS_HighDim"],
        baseline_method="RHS_HighDim",
        build_tables=False,
    )
    pd = load_pandas()
    summary = pd.read_csv(out["summary_paired"])
    assert set(summary["method"]) == {"GR_RHS_HighDim", "RHS_HighDim"}
    assert int(summary.loc[summary["method"].eq("GR_RHS_HighDim"), "n_converged"].iloc[0]) == 0
    assert float(summary.loc[summary["method"].eq("GR_RHS_HighDim"), "rmse_test"].iloc[0]) == 1.0
