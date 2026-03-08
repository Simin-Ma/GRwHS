from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload or {}


def _extract_param(cfg: Dict[str, Any]) -> Dict[str, Any]:
    model = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    return {
        "eta": model.get("eta"),
        "tau0": model.get("tau0"),
        "c": model.get("c"),
        "iters": model.get("iters"),
    }


def summarize_tuning(sweep_dir: Path, rhs_metrics_path: Path, out_csv: Path) -> pd.DataFrame:
    rhs_payload = json.loads(rhs_metrics_path.read_text(encoding="utf-8"))
    rhs_metrics = rhs_payload.get("metrics", rhs_payload)
    rhs_rmse = float(rhs_metrics["RMSE"])
    rhs_mlpd = float(rhs_metrics["MLPD"])

    records: List[Dict[str, Any]] = []
    for run_dir in sorted(path for path in sweep_dir.iterdir() if path.is_dir()):
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "resolved_config.yaml"
        if not metrics_path.exists() or not config_path.exists():
            continue
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = metrics_payload.get("metrics", metrics_payload)
        cfg = _load_yaml(config_path)
        params = _extract_param(cfg)
        records.append(
            {
                "run_name": run_dir.name,
                **params,
                "RMSE": metrics.get("RMSE"),
                "MLPD": metrics.get("MLPD"),
                "PredictiveLogLikelihood": metrics.get("PredictiveLogLikelihood"),
                "EffectiveDoF": metrics.get("EffectiveDoF"),
                "MeanEffectiveNonzeros": metrics.get("MeanEffectiveNonzeros"),
                "delta_rmse_vs_rhs": None if metrics.get("RMSE") is None else float(metrics["RMSE"]) - rhs_rmse,
                "delta_mlpd_vs_rhs": None if metrics.get("MLPD") is None else float(metrics["MLPD"]) - rhs_mlpd,
            }
        )

    frame = pd.DataFrame(records)
    if frame.empty:
        raise SystemExit(f"No completed runs found in {sweep_dir}")

    frame.sort_values(["RMSE", "MLPD"], ascending=[True, False], inplace=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize exploratory NHANES GR-RHS tuning results.")
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path("outputs/sweeps/real_nhanes_2003_2004_grrhs_tuning"),
        help="Directory containing tuning run subdirectories.",
    )
    parser.add_argument(
        "--rhs-metrics",
        type=Path,
        default=Path("outputs/sweeps/real_nhanes_2003_2004_ggt/nhanes_rhs-20260308-014419/metrics.json"),
        help="Reference RHS metrics.json",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/reports/nhanes_grrhs_tuning_summary.csv"),
        help="Destination CSV path.",
    )
    args = parser.parse_args()

    frame = summarize_tuning(args.sweep_dir, args.rhs_metrics, args.out_csv)
    print(frame.to_string(index=False))
    print(f"[ok] summary written to {args.out_csv}")


if __name__ == "__main__":
    main()
