
"""Aggregate results from repeated runs."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def aggregate_runs(run_dir: Path, *, write: bool = True) -> Dict[str, Any]:
    """Aggregate metrics across all run subdirectories.

    Args:
        run_dir: Directory containing per-run subdirectories (each with metrics.json).
        write:  Whether to persist the aggregate summary alongside the runs.

    Returns:
        Aggregated summary dictionary.
    """

    root = run_dir.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Run directory not found: {root}")

    subdirs = sorted([p for p in root.iterdir() if p.is_dir()])
    records: List[Dict[str, Any]] = []
    metric_values: Dict[str, List[float]] = {}
    status_counter: Counter[str] = Counter()

    for subdir in subdirs:
        metrics_path = subdir / "metrics.json"
        if not metrics_path.exists():
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if "metrics" in payload and isinstance(payload["metrics"], dict):
                metrics = payload["metrics"]
                status = str(payload.get("status", "UNKNOWN"))
            else:
                metrics = {k: v for k, v in payload.items() if _is_number(v)}
                status = str(payload.get("status", "OK" if metrics else "UNKNOWN"))
        else:
            metrics = {}
            status = "UNKNOWN"
        status_counter[status] += 1
        record = {
            "run_dir": str(subdir),
            "status": status,
            "metrics": metrics,
        }
        records.append(record)

        for key, value in metrics.items():
            if _is_number(value):
                metric_values.setdefault(key, []).append(float(value))

    aggregates: Dict[str, Dict[str, Any]] = {}
    for key, values in metric_values.items():
        if not values:
            continue
        aggregates[key] = {
            "count": len(values),
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }

    summary = {
        "run_dir": str(root),
        "total_runs": len(records),
        "status_counts": dict(status_counter),
        "metrics": aggregates,
        "runs": records,
    }

    if write and records:
        summary_path = root / "aggregate_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary

