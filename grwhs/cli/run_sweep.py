
"""CLI entry point for running configuration sweeps."""
from __future__ import annotations

import argparse
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

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
            model_name = get_model_name_from_config(resolved_cfg).lower()
        except Exception:
            model_name = None
        if model_name is not None and model_name != "grwhs_gibbs":
            resolved_cfg.setdefault("experiments", {})["repeats"] = 1

        record: Dict[str, Any] = {
            "name": name,
            "run_dir": str(run_dir),
            "index": idx,
        }

        if dry_run:
            print(f"[DRY-RUN] Would run variation '{name}' -> {run_dir}")
            record["status"] = "DRY_RUN"
            return record

        try:
            resolved_cfg = _auto_inject_tau0(resolved_cfg)
            _save_resolved_config(resolved_cfg, run_dir)
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



if __name__ == "__main__":  # pragma: no cover
    main()
