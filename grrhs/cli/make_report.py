"""CLI entry point for report generation."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from grrhs.experiments.aggregator import aggregate_runs
from grrhs.cli.run_sweep import (
    _build_comparison_payload,
    _compute_metric_extrema,
    _format_metric,
    _resolve_comparison_metrics,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize GRRHS experiment runs")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("outputs/runs"),
        help="Directory containing run artifacts (used when --run is omitted)",
    )
    parser.add_argument(
        "--run",
        type=Path,
        action="append",
        help="Specific run directory to summarize (can be passed multiple times)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("outputs/reports"),
        help="Directory where summary files will be written",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Placeholder flag for plot generation (not implemented yet)",
    )
    parser.add_argument(
        "--comparison-mode",
        choices=("aggregate", "benchmark-safe", "both"),
        default="aggregate",
        help="How to summarize multiple runs for cross-run comparison outputs.",
    )
    return parser.parse_args()


def _select_runs(runs_arg: List[Path] | None, runs_dir: Path, *, all_runs: bool = False) -> Iterable[Path]:
    if runs_arg:
        for run in runs_arg:
            run_path = run if run.is_absolute() else Path.cwd() / run
            yield run_path.resolve()
        return

    if not runs_dir.exists():
        return

    run_dirs = sorted(
        [p for p in runs_dir.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not run_dirs:
        return

    if all_runs:
        yield from run_dirs
        return

    yield run_dirs[0]


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def _to_json(value):
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {k: _to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json(v) for v in value]
    if hasattr(value, "item"):
        return _to_json(value.item())
    return str(value)


def _format_timestamp(ts: float) -> str:
    """Convert epoch seconds into an ISO-8601 string with UTC timezone."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _summarize_outputs(run_dir: Path) -> List[dict]:
    outputs: List[dict] = []
    for entry in sorted(run_dir.iterdir()):
        stat = entry.stat()
        outputs.append(
            {
                "name": entry.name,
                "path": str(entry.relative_to(run_dir)),
                "type": "directory" if entry.is_dir() else "file",
                "size_bytes": stat.st_size,
                "modified": _format_timestamp(stat.st_mtime),
            }
        )
    return outputs


def _collect_fold_convergence(run_dir: Path) -> List[dict]:
    """Return per-fold convergence diagnostics with relative paths."""
    records: List[dict] = []
    repeat_dirs = sorted(p for p in run_dir.glob("repeat_*") if p.is_dir())
    for repeat_dir in repeat_dirs:
        fold_dirs = sorted(p for p in repeat_dir.glob("fold_*") if p.is_dir())
        for fold_dir in fold_dirs:
            diag_path = fold_dir / "convergence.json"
            if not diag_path.exists():
                continue
            rel_path = diag_path.relative_to(run_dir)
            records.append(
                {
                    "repeat": repeat_dir.name,
                    "fold": fold_dir.name,
                    "path": str(rel_path),
                    "diagnostics": _load_json(diag_path),
                }
            )
    return records


def _summarize_run(run_dir: Path) -> dict:
    metrics = _load_json(run_dir / "metrics.json")
    meta = _load_json(run_dir / "dataset_meta.json")
    cfg_path = run_dir / "resolved_config.yaml"
    convergence = _load_json(run_dir / "convergence.json")
    fold_convergence = _collect_fold_convergence(run_dir)
    resolved_config = _load_yaml(cfg_path)

    summary = {
        "run_dir": str(run_dir.resolve()),
        "run_name": run_dir.name,
        "metrics": _to_json(metrics),
        "dataset": {
            "n": meta.get("n"),
            "p": meta.get("p"),
            "model": meta.get("model"),
        },
        "artifacts": meta.get("dataset_path"),
        "posterior": _to_json(meta.get("posterior")),
        "seeds": _to_json(meta.get("seeds")),
        "dataset_meta": _to_json(meta) if meta else None,
        "resolved_config": _to_json(resolved_config) if resolved_config else None,
        "convergence": convergence or None,
        "fold_convergence": fold_convergence or None,
        "outputs": _summarize_outputs(run_dir),
        "has_resolved_config": cfg_path.exists(),
    }
    return summary


def _sanitize_label(label: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label.strip())
    return safe.strip("_") or "runs"


def _make_benchmark_safe_record(summary: Dict[str, Any]) -> Dict[str, Any]:
    metrics_payload = summary.get("metrics", {})
    resolved_config = summary.get("resolved_config", {})
    status = "UNKNOWN"
    model = None
    if isinstance(metrics_payload, dict):
        status = str(metrics_payload.get("status", "UNKNOWN"))
        payload_model = metrics_payload.get("model")
        if isinstance(payload_model, str):
            model = payload_model

    comparison_metrics: List[str] = []
    if isinstance(resolved_config, dict):
        comparison_metrics = _resolve_comparison_metrics(resolved_config)

    return {
        "name": summary.get("run_name"),
        "model": model,
        "status": status,
        "run_dir": summary.get("run_dir"),
        "comparison_metrics": comparison_metrics,
        "metrics": metrics_payload,
    }


def _write_benchmark_safe_artifacts(
    summaries: List[Dict[str, Any]],
    *,
    dest: Path,
    label: str,
) -> Dict[str, Any]:
    records = [_make_benchmark_safe_record(summary) for summary in summaries]
    rows, metric_keys, comparison_meta = _build_comparison_payload(records)
    safe_label = _sanitize_label(label)
    base_name = f"benchmark_safe_comparison_{safe_label}"

    csv_path = dest / f"{base_name}.csv"
    json_path = dest / f"{base_name}.json"
    md_path = dest / f"{base_name}.md"

    header = ["variation", "model", "status", "run_dir", *metric_keys]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for row in rows:
            formatted_metrics = [_format_metric(row["metrics"].get(metric)) for metric in metric_keys]
            writer.writerow([
                row["variation"],
                row["model"] or "",
                row["status"],
                row["run_dir"],
                *formatted_metrics,
            ])

    extrema = _compute_metric_extrema(rows, metric_keys)
    json_payload = {
        "label": label,
        "mode": "benchmark-safe",
        "metrics": metric_keys,
        "comparison": comparison_meta,
        "rows": rows,
        "metric_extrema": extrema,
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append(f"# {label} benchmark-safe comparison")
    lines.append("")
    if comparison_meta.get("comparison_basis") == "common_valid_folds":
        lines.append(
            f"Comparison basis: common valid outer folds across comparable runs "
            f"(`n={comparison_meta.get('common_valid_fold_count', 0)}`)."
        )
        lines.append("")
    header_cells = ["Variation", "Model", "Status", *metric_keys]
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_cells)) + " |")
    for row in rows:
        cells = [
            str(row["variation"]),
            row["model"] or "",
            str(row["status"]),
        ]
        for metric in metric_keys:
            cells.append(_format_metric(row["metrics"].get(metric)))
        lines.append("| " + " | ".join(cells) + " |")

    if extrema:
        lines.append("")
        lines.append("## Metric extrema")
        for metric in metric_keys:
            if metric not in extrema:
                continue
            info = extrema[metric]
            min_info = info["min"]
            max_info = info["max"]
            lines.append(
                f"- `{metric}` min: {min_info['value']:.6g} ({min_info['variation']}) | "
                f"max: {max_info['value']:.6g} ({max_info['variation']})"
            )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "label": label,
        "run_count": len(summaries),
        "comparison": comparison_meta,
        "artifacts": {
            "csv": str(csv_path),
            "json": str(json_path),
            "md": str(md_path),
        },
    }


def main() -> None:
    args = parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)

    select_all_runs = args.comparison_mode in {"benchmark-safe", "both"} and not args.run
    runs = list(_select_runs(args.run, args.runs_dir, all_runs=select_all_runs))
    if not runs:
        raise SystemExit("No runs found to summarize. Provide --run or populate outputs/runs.")

    if args.plots:
        print("[WARN] --plots flag acknowledged but plot generation is not implemented yet.")

    collected = []
    parents: Dict[Path, List[Dict[str, Any]]] = {}
    for run_dir in runs:
        summary = _summarize_run(run_dir)
        collected.append(summary)
        out_file = args.dest / f"{run_dir.name}_summary.json"
        out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[OK] Summary written to {out_file}")
        parent = run_dir.parent.resolve()
        parents.setdefault(parent, []).append(summary)

    aggregates = []
    if args.comparison_mode in {"aggregate", "both"}:
        for parent, run_summaries in parents.items():
            try:
                agg_summary = aggregate_runs(parent, write=True)
            except Exception as exc:  # pragma: no cover - aggregation best-effort
                print(f"[WARN] Failed to aggregate runs under {parent}: {exc}")
                continue
            aggregates.append(_to_json(agg_summary))

    benchmark_comparisons = []
    if args.comparison_mode in {"benchmark-safe", "both"}:
        for parent, run_summaries in parents.items():
            try:
                comparison = _write_benchmark_safe_artifacts(
                    run_summaries,
                    dest=args.dest,
                    label=parent.name,
                )
            except Exception as exc:  # pragma: no cover - comparison best-effort
                print(f"[WARN] Failed to build benchmark-safe comparison for {parent}: {exc}")
                continue
            benchmark_comparisons.append(_to_json(comparison))

    combined_payload = {
        "runs": collected,
        "aggregates": aggregates,
        "benchmark_comparisons": benchmark_comparisons,
    }
    combined_path = args.dest / "summary_index.json"
    combined_path.write_text(json.dumps(combined_payload, indent=2), encoding="utf-8")
    print(f"[OK] Aggregated index saved to {combined_path}")

    if aggregates:
        agg_path = args.dest / "aggregates_summary.json"
        agg_path.write_text(json.dumps(aggregates, indent=2), encoding="utf-8")
        print(f"[OK] Aggregate summaries saved to {agg_path}")

    if benchmark_comparisons:
        comparison_path = args.dest / "benchmark_comparisons_summary.json"
        comparison_path.write_text(json.dumps(benchmark_comparisons, indent=2), encoding="utf-8")
        print(f"[OK] Benchmark-safe comparisons saved to {comparison_path}")


if __name__ == "__main__":
    main()
