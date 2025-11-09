
"""CLI entry point for running configuration sweeps."""
from __future__ import annotations

import argparse
import csv
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from threading import Lock
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from grwhs.cli.run_experiment import (
    _deep_update,
    _derive_run_dir,
    _load_and_merge_configs,
    _maybe_call_runner,
    _parse_overrides,
    _save_resolved_config,
    _auto_inject_tau0,
)
from grwhs.experiments.sweeps import build_override_tree, deep_update
from grwhs.experiments.registry import get_model_name_from_config

_NUMPYRO_MODEL_KEYS = {
    "grwhs_gibbs",
    "grwhs_gibbs_logistic",
    "grwhs_svi",
    "grwhs_svi_numpyro",
    "horseshoe",
    "horseshoe_regression",
    "hs",
    "regularized_horseshoe",
    "regularised_horseshoe",
    "rhs",
    "group_horseshoe",
    "group_hs",
    "ghs",
}
_NUMPYRO_LOCK = Lock()


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping at top level.")
    return data


def _resolve_paths(paths: Iterable[str | Path], root: Path) -> List[Path]:
    resolved: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if not p.is_absolute():
            p = (root / p).resolve()
        resolved.append(p)
    return resolved


def _merge_config_files(base: Dict[str, Any], files: Iterable[Path]) -> Dict[str, Any]:
    cfg = deepcopy(base)
    for cfg_path in files:
        part = _load_and_merge_configs([cfg_path])
        deep_update(cfg, part)
    return cfg


def _coerce_seed(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _detect_data_seed(cfg: Dict[str, Any]) -> int | None:
    data_cfg = cfg.get("data")
    if isinstance(data_cfg, dict):
        seed = _coerce_seed(data_cfg.get("seed"))
        if seed is not None:
            return seed
    seeds_cfg = cfg.get("seeds")
    if isinstance(seeds_cfg, dict):
        seed = _coerce_seed(seeds_cfg.get("data_generation"))
        if seed is not None:
            return seed
    return None


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_numeric_metrics(payload: Any) -> Dict[str, float]:
    if not isinstance(payload, dict):
        return {}
    candidate = payload.get("metrics")
    source = candidate if isinstance(candidate, dict) else payload
    metrics: Dict[str, float] = {}
    for key, value in source.items():
        number = _safe_float(value)
        if number is not None:
            metrics[key] = number
    return metrics


def _determine_model_label(record: Dict[str, Any], payload: Any) -> str | None:
    model = record.get("model")
    if isinstance(model, str):
        return model
    if isinstance(payload, dict):
        payload_model = payload.get("model")
        if isinstance(payload_model, str):
            return payload_model
    return None


def _format_metric(value: float | None) -> str:
    if value is None:
        return "N/A"
    abs_val = abs(value)
    if abs_val and (abs_val >= 1e4 or abs_val < 1e-3):
        return f"{value:.3e}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _compute_metric_extrema(rows: List[Dict[str, Any]], metric_keys: List[str]) -> Dict[str, Dict[str, Any]]:
    extrema: Dict[str, Dict[str, Any]] = {}
    for metric in metric_keys:
        candidates = [
            (row["metrics"][metric], row["variation"])
            for row in rows
            if metric in row["metrics"]
        ]
        if not candidates:
            continue
        min_value, min_var = min(candidates, key=lambda item: item[0])
        max_value, max_var = max(candidates, key=lambda item: item[0])
        extrema[metric] = {
            "min": {"variation": min_var, "value": min_value},
            "max": {"variation": max_var, "value": max_value},
        }
    return extrema


def _build_comparison_rows(summary: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    metric_keys: set[str] = set()
    for record in summary:
        payload = record.get("metrics")
        metrics = _extract_numeric_metrics(payload)
        metric_keys.update(metrics.keys())
        rows.append({
            "variation": record.get("name"),
            "model": _determine_model_label(record, payload),
            "status": record.get("status") or "UNKNOWN",
            "run_dir": record.get("run_dir"),
            "metrics": metrics,
        })
    return rows, sorted(metric_keys)


def _write_comparison_artifacts(outdir_path: Path, sweep_name: str, timestamp: str, summary: List[Dict[str, Any]]) -> None:
    rows, metric_keys = _build_comparison_rows(summary)
    if not rows:
        return

    base_name = f"sweep_comparison_{timestamp}"
    csv_path = outdir_path / f"{base_name}.csv"
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
        "sweep_name": sweep_name,
        "generated_at": timestamp,
        "metrics": metric_keys,
        "rows": rows,
        "metric_extrema": extrema,
    }
    json_path = outdir_path / f"{base_name}.json"
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    md_path = outdir_path / f"{base_name}.md"
    lines: List[str] = []
    lines.append(f"# {sweep_name} sweep comparison ({timestamp})")
    lines.append("")
    if metric_keys:
        header_cells = ["Variation", "Model", "Status", *metric_keys]
    else:
        header_cells = ["Variation", "Model", "Status"]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a sweep of GRwHS experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-config", type=Path, required=True, help="Base YAML config applied to all runs")
    parser.add_argument("--sweep-config", type=Path, required=True, help="Sweep specification YAML")
    parser.add_argument("--outdir", type=Path, default=None, help="Override sweep output directory")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N variations")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without executing")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel workers")
    return parser.parse_args()


def _prepare_override_tree(spec: Dict[str, Any], extra_cli: List[str] | None = None) -> Dict[str, Any]:
    base_tree = build_override_tree(spec)
    if extra_cli:
        cli_tree = _parse_overrides(extra_cli)
        _deep_update(base_tree, cli_tree)
    return base_tree


def main() -> None:
    args = parse_args()

    base_config_path = args.base_config.expanduser().resolve()
    sweep_config_path = args.sweep_config.expanduser().resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")
    if not sweep_config_path.exists():
        raise FileNotFoundError(f"Sweep config not found: {sweep_config_path}")

    base_cfg = _load_and_merge_configs([base_config_path])
    sweep_root = sweep_config_path.parent
    sweep_spec = _load_yaml(sweep_config_path)

    variations = sweep_spec.get("variations", [])
    if not variations:
        raise ValueError("Sweep specification must provide a 'variations' list.")
    if not isinstance(variations, list):
        raise ValueError("'variations' must be a list of mappings.")

    outdir = args.outdir or sweep_spec.get("output_dir") or "outputs/sweeps"
    outdir_path = Path(outdir).expanduser().resolve()
    outdir_path.mkdir(parents=True, exist_ok=True)

    common_cfg_files = _resolve_paths(sweep_spec.get("common_config_files", []), sweep_root)
    common_overrides_spec = sweep_spec.get("common_overrides", {})
    common_override_cli = sweep_spec.get("common_override_cli", [])
    if not isinstance(common_overrides_spec, dict):
        raise ValueError("common_overrides must be a mapping if provided.")
    common_tree = _prepare_override_tree(common_overrides_spec, common_override_cli)

    base_with_common = _merge_config_files(base_cfg, common_cfg_files)

    shared_data_seed = _detect_data_seed(base_with_common)
    if shared_data_seed is None:
        shared_data_seed = random.randint(0, 2**32 - 1)
    base_with_common.setdefault("data", {})["seed"] = shared_data_seed
    base_with_common.setdefault("seeds", {})["data_generation"] = shared_data_seed

    sweep_name = sweep_spec.get("name") or sweep_config_path.stem
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary: List[Dict[str, Any]] = []

    jobs = max(1, int(args.jobs or 1))
    tasks: List[Dict[str, Any]] = []

    for idx, raw_var in enumerate(variations):
        if args.limit is not None and idx >= args.limit:
            break
        if not isinstance(raw_var, dict):
            raise ValueError("Each variation entry must be a mapping.")
        if not raw_var.get("enabled", True):
            continue

        variation = dict(raw_var)
        name = variation.get("name") or f"variation_{idx:03d}"
        var_cfg_files = _resolve_paths(variation.get("config_files", []), sweep_root)
        var_overrides_spec = variation.get("overrides", {})
        if not isinstance(var_overrides_spec, dict):
            raise ValueError(f"Variation '{name}' overrides must be a mapping.")
        var_override_cli = variation.get("override_cli", [])
        if var_override_cli and not isinstance(var_override_cli, list):
            raise ValueError(f"Variation '{name}' override_cli must be a list.")

        var_tree = _prepare_override_tree(var_overrides_spec, var_override_cli)

        resolved_cfg = _merge_config_files(base_with_common, var_cfg_files)
        deep_update(resolved_cfg, common_tree)
        deep_update(resolved_cfg, var_tree)
        resolved_cfg.setdefault("data", {})["seed"] = shared_data_seed
        resolved_cfg.setdefault("seeds", {})["data_generation"] = shared_data_seed

        resolved_cfg.setdefault("name", name)
        sweep_meta = resolved_cfg.setdefault("sweep", {})
        if isinstance(sweep_meta, dict):
            sweep_meta.update({
                "sweep_name": sweep_name,
                "variation": name,
                "index": idx,
            })

        run_dir = _derive_run_dir(outdir_path, name)
        tasks.append({
            "index": idx,
            "name": name,
            "run_dir": run_dir,
            "config": resolved_cfg,
            "dry_run": bool(args.dry_run),
        })

    def _execute(payload: Dict[str, Any]) -> Dict[str, Any]:
        name = payload["name"]
        run_dir: Path = payload["run_dir"]
        idx = payload["index"]
        resolved_cfg = payload["config"]
        dry_run = payload["dry_run"]

        try:
            resolved_model_name = get_model_name_from_config(resolved_cfg)
            model_name_key = resolved_model_name.lower()
        except Exception:
            resolved_model_name = None
            model_name_key = None
        if model_name_key is not None and model_name_key != "grwhs_gibbs":
            resolved_cfg.setdefault("experiments", {})["repeats"] = 1

        record: Dict[str, Any] = {
            "name": name,
            "run_dir": str(run_dir),
            "index": idx,
            "model": resolved_model_name,
        }

        if dry_run:
            print(f"[DRY-RUN] Would run variation '{name}' -> {run_dir}")
            record["status"] = "DRY_RUN"
            return record

        try:
            resolved_cfg = _auto_inject_tau0(resolved_cfg)
            _save_resolved_config(resolved_cfg, run_dir)
            needs_numpyro_lock = (
                jobs > 1 and model_name_key is not None and model_name_key in _NUMPYRO_MODEL_KEYS
            )
            lock_ctx = _NUMPYRO_LOCK if needs_numpyro_lock else nullcontext()
            with lock_ctx:
                metrics = _maybe_call_runner(resolved_cfg, run_dir)
            with (run_dir / "metrics.json").open("w", encoding="utf-8") as fh:
                json.dump(metrics, fh, indent=2, ensure_ascii=False)
            status = metrics.get("status", "OK") if isinstance(metrics, dict) else "OK"
            record.update({
                "status": status,
                "metrics": metrics,
                "config_path": str(run_dir / "resolved_config.yaml"),
            })
            print(f"[SWEEP] Completed '{name}' -> {run_dir}")
        except Exception as exc:  # pragma: no cover - safety net
            record.update({
                "status": "ERROR",
                "error": str(exc),
            })
            print(f"[SWEEP] ERROR while running '{name}': {exc}")
            traceback.print_exc()
        return record

    if jobs == 1 or not tasks:
        for payload in tasks:
            summary.append(_execute(payload))
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {executor.submit(_execute, payload): payload for payload in tasks}
            for future in as_completed(futures):
                summary.append(future.result())

    summary.sort(key=lambda r: r.get("index", 0))

    if not args.dry_run and summary:
        summary_path = outdir_path / f"sweep_summary_{timestamp}.json"
        payload = {
            "sweep": {
                "name": sweep_name,
                "base_config": str(base_config_path),
                "sweep_config": str(sweep_config_path),
                "generated_at": timestamp,
            },
            "runs": summary,
        }
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[SWEEP] Summary written to {summary_path}")
        _write_comparison_artifacts(outdir_path, sweep_name, timestamp, summary)



if __name__ == "__main__":  # pragma: no cover
    main()
