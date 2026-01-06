"""CLI entry point for report generation."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import yaml

from grrhs.experiments.aggregator import aggregate_runs

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
    return parser.parse_args()


def _select_runs(runs_arg: List[Path] | None, runs_dir: Path) -> Iterable[Path]:
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
    if run_dirs:
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


def main() -> None:
    args = parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)

    runs = list(_select_runs(args.run, args.runs_dir))
    if not runs:
        raise SystemExit("No runs found to summarize. Provide --run or populate outputs/runs.")

    if args.plots:
        print("[WARN] --plots flag acknowledged but plot generation is not implemented yet.")

    collected = []
    parents: Dict[Path, List[Path]] = {}
    for run_dir in runs:
        summary = _summarize_run(run_dir)
        collected.append(summary)
        out_file = args.dest / f"{run_dir.name}_summary.json"
        out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[OK] Summary written to {out_file}")
        parent = run_dir.parent.resolve()
        parents.setdefault(parent, []).append(run_dir)

    aggregates = []
    for parent, run_list in parents.items():
        try:
            agg_summary = aggregate_runs(parent, write=True)
        except Exception as exc:  # pragma: no cover - aggregation best-effort
            print(f"[WARN] Failed to aggregate runs under {parent}: {exc}")
            continue
        aggregates.append(_to_json(agg_summary))

    combined_payload = {
        "runs": collected,
        "aggregates": aggregates,
    }
    combined_path = args.dest / "summary_index.json"
    combined_path.write_text(json.dumps(combined_payload, indent=2), encoding="utf-8")
    print(f"[OK] Aggregated index saved to {combined_path}")

    if aggregates:
        agg_path = args.dest / "aggregates_summary.json"
        agg_path.write_text(json.dumps(aggregates, indent=2), encoding="utf-8")
        print(f"[OK] Aggregate summaries saved to {agg_path}")


if __name__ == "__main__":
    main()
