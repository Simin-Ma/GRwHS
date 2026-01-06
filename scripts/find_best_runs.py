from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from grrhs.experiments.aggregator import aggregate_runs


def _is_nan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _load_model_name(run_dir: Path) -> Optional[str]:
    cfg_path = run_dir / "resolved_config.yaml"
    if not cfg_path.exists():
        return None
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(cfg, dict):
        return None
    model_cfg = cfg.get("model")
    if isinstance(model_cfg, dict):
        name = model_cfg.get("name")
        if isinstance(name, str):
            return name
    return None


def _collect_entries(
    runs_root: Path,
    metric_key: str,
    model_filter: Optional[str],
) -> List[Dict[str, Any]]:
    summary = aggregate_runs(runs_root, write=False)
    entries: List[Dict[str, Any]] = []
    for record in summary.get("runs", []):
        run_dir_raw = record.get("run_dir")
        if not run_dir_raw:
            continue
        run_dir = Path(run_dir_raw)
        metrics = record.get("metrics") or {}
        metric_val = metrics.get(metric_key)
        if metric_val is None:
            continue
        if _is_nan(metric_val):
            continue
        model_name = _load_model_name(run_dir)
        if model_filter and model_name != model_filter:
            continue
        try:
            metric_float = float(metric_val)
        except (TypeError, ValueError):
            continue
        entries.append(
            {
                "name": record.get("name"),
                "run_dir": str(run_dir),
                "metric": metric_float,
                "model": model_name,
                "status": record.get("status", "UNKNOWN"),
            }
        )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select top runs from a sweep directory by metric.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("runs_root", type=Path, help="Sweep directory containing run subfolders")
    parser.add_argument("--metric", default="RMSE", help="Metric key to score runs by (lower is better by default)")
    parser.add_argument("--top", type=int, default=10, help="Number of top runs to display")
    parser.add_argument("--model", default=None, help="Filter by model name (e.g., grrhs_gibbs)")
    parser.add_argument("--maximize", action="store_true", help="Treat higher metric values as better")
    parser.add_argument("--json", action="store_true", help="Emit results as JSON instead of text")
    args = parser.parse_args()

    runs_root = args.runs_root.expanduser().resolve()
    if not runs_root.exists():
        raise FileNotFoundError(f"Sweep directory not found: {runs_root}")

    entries = _collect_entries(runs_root, args.metric, args.model)
    if not entries:
        message = "No runs with metric '{metric}'".format(metric=args.metric)
        if args.model:
            message += f" for model '{args.model}'"
        print(message)
        return

    entries.sort(key=lambda item: item["metric"], reverse=args.maximize)
    top_entries = entries[: max(1, args.top)]

    if args.json:
        payload = {
            "runs_root": str(runs_root),
            "metric": args.metric,
            "maximize": bool(args.maximize),
            "model_filter": args.model,
            "top": top_entries,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    direction = "max" if args.maximize else "min"
    header = f"Top {len(top_entries)} runs in {runs_root} by {direction} {args.metric}"
    if args.model:
        header += f" (model={args.model})"
    print(header)
    print("-" * len(header))

    width = len(str(len(top_entries)))
    for idx, entry in enumerate(top_entries, 1):
        metric_val = entry["metric"]
        print(
            f"{idx:>{width}}. {entry['name']}  "
            f"{args.metric}={metric_val:.5f}  "
            f"model={entry['model'] or 'n/a'}  "
            f"status={entry['status']}  "
            f"run_dir={entry['run_dir']}"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
