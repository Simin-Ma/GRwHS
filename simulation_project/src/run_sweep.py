from __future__ import annotations

import argparse
import csv
import inspect
import itertools
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from .experiments import (
    run_exp1_kappa_profile_regimes,
    run_exp2_group_separation,
    run_exp3_linear_benchmark,
    run_exp3a_main_benchmark,
    run_exp3b_boundary_stress,
    run_exp3c_highdim_stress,
    run_exp3d_within_group_mixed,
    run_exp4_variant_ablation,
    run_exp5_prior_sensitivity,
)
from .experiments.orchestration import run_all_experiments
from .experiment_aliases import normalize_sweep_experiment
from .output_layout import resolve_explicit_save_dir, resolve_workspace_dir
from .utils import ensure_dir, save_json


_RUNNERS: dict[str, Callable[..., dict[str, Any]]] = {
    "all": run_all_experiments,
    "exp1": run_exp1_kappa_profile_regimes,
    "exp2": run_exp2_group_separation,
    "exp3": run_exp3_linear_benchmark,
    "exp3a": run_exp3a_main_benchmark,
    "exp3b": run_exp3b_boundary_stress,
    "exp3c": run_exp3c_highdim_stress,
    "exp3d": run_exp3d_within_group_mixed,
    "exp4": run_exp4_variant_ablation,
    "exp5": run_exp5_prior_sensitivity,
}

def _load_sweep_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"invalid sweep config at {path}: top-level must be a mapping")
    return dict(raw)


def _cartesian_grid(grid: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values: list[list[Any]] = []
    for k in keys:
        v = grid[k]
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError(f"grid field '{k}' must be a non-empty list")
        values.append(v)
    out: list[dict[str, Any]] = []
    for combo in itertools.product(*values):
        out.append({keys[i]: combo[i] for i in range(len(keys))})
    return out


def _validate_kwargs(fn: Callable[..., Any], params: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())
    unknown = sorted(set(params.keys()) - allowed)
    if unknown:
        raise ValueError(f"{label}: unsupported parameters for {fn.__name__}: {unknown}")
    return dict(params)


def _to_csv_value(v: Any) -> str | int | float | bool | None:
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    return json.dumps(v, ensure_ascii=False, sort_keys=True)


def _write_runs_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _to_csv_value(row.get(k)) for k in fields})


def _parse_set_items(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for raw in items:
        token = str(raw)
        if "=" not in token:
            raise ValueError(f"--set expects KEY=VALUE, got: {raw!r}")
        key, value_text = token.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--set key cannot be empty: {raw!r}")
        out[key] = yaml.safe_load(value_text)
    return out


def list_sweeps(*, config_path: str = "simulation_project/config/sweeps.yaml") -> list[dict[str, str]]:
    cfg = _load_sweep_config(Path(config_path))
    sweeps = cfg.get("sweeps", {})
    if not isinstance(sweeps, Mapping):
        raise ValueError("sweeps must be a mapping")
    out: list[dict[str, str]] = []
    for name, spec in sweeps.items():
        if not isinstance(spec, Mapping):
            continue
        out.append(
            {
                "name": str(name),
                "experiment": str(spec.get("experiment", "")),
                "description": str(spec.get("description", "")),
            }
        )
    return out


def run_sweep(
    *,
    sweep_name: str,
    config_path: str = "simulation_project/config/sweeps.yaml",
    save_dir: str | None = None,
    max_runs: int | None = None,
    dry_run: bool = False,
    fail_fast: bool = False,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = _load_sweep_config(Path(config_path))
    sweeps = cfg.get("sweeps", {})
    if not isinstance(sweeps, Mapping):
        raise ValueError("sweep config missing mapping: sweeps")
    if sweep_name not in sweeps:
        known = sorted(str(k) for k in sweeps.keys())
        raise ValueError(f"unknown sweep: {sweep_name!r}; available: {known}")

    spec = sweeps[sweep_name]
    if not isinstance(spec, Mapping):
        raise ValueError(f"sweep spec for {sweep_name!r} must be a mapping")

    exp_name = normalize_sweep_experiment(spec.get("experiment", ""))
    runner = _RUNNERS[exp_name]

    defaults = cfg.get("defaults", {})
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, Mapping):
        raise ValueError("defaults in sweep config must be a mapping")

    fixed = spec.get("fixed", {})
    if fixed is None:
        fixed = {}
    if not isinstance(fixed, Mapping):
        raise ValueError(f"sweep '{sweep_name}' field 'fixed' must be a mapping")

    grid = spec.get("grid", {})
    if grid is None:
        grid = {}
    if not isinstance(grid, Mapping):
        raise ValueError(f"sweep '{sweep_name}' field 'grid' must be a mapping")

    combos = _cartesian_grid(grid)
    if max_runs is not None:
        combos = combos[: max(0, int(max_runs))]

    common: dict[str, Any] = dict(defaults)
    common.update(dict(fixed))
    if overrides:
        common.update(dict(overrides))

    if save_dir is not None:
        base_save_dir = str(resolve_explicit_save_dir(str(save_dir), workspace="simulation_project", create=True))
    else:
        fallback = common.get("save_dir", None)
        if fallback is None or str(fallback).strip() == "":
            base_save_dir = str(resolve_workspace_dir("simulation_project", create=True))
        else:
            base_save_dir = str(resolve_explicit_save_dir(str(fallback), workspace="simulation_project", create=True))
    sweep_root = ensure_dir(Path(base_save_dir) / "sweeps" / str(sweep_name))
    session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    session_dir = ensure_dir(sweep_root / session_id)
    ensure_dir(session_dir / "runs")

    rows: list[dict[str, Any]] = []
    ok_count = 0
    err_count = 0
    t_all = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()

    for i, combo in enumerate(combos, start=1):
        run_name = f"run_{i:03d}"
        run_dir = ensure_dir(session_dir / "runs" / run_name)

        params = dict(common)
        params.update(combo)
        params["save_dir"] = str(run_dir)

        label = f"sweep={sweep_name} run={run_name}"
        kwargs = _validate_kwargs(runner, params, label=label)

        row: dict[str, Any] = {
            "run_name": run_name,
            "experiment": exp_name,
            "status": "dry_run" if dry_run else "pending",
            "duration_sec": 0.0,
            "params": kwargs,
            "outputs": {},
            "error": "",
        }

        t0 = time.perf_counter()
        if not dry_run:
            try:
                outputs = runner(**kwargs)
                row["status"] = "ok"
                row["outputs"] = outputs
                ok_count += 1
            except Exception as exc:
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["traceback"] = traceback.format_exc()
                err_count += 1
                if fail_fast:
                    row["duration_sec"] = float(time.perf_counter() - t0)
                    rows.append(row)
                    save_json(
                        {
                            "sweep_name": sweep_name,
                            "session_id": session_id,
                            "config_path": str(config_path),
                            "started_at": started_at,
                            "finished_at": datetime.now(timezone.utc).isoformat(),
                            "duration_sec": float(time.perf_counter() - t_all),
                            "experiment": exp_name,
                            "dry_run": bool(dry_run),
                            "fail_fast": bool(fail_fast),
                            "ok_count": int(ok_count),
                            "error_count": int(err_count),
                            "total_runs": int(len(combos)),
                            "rows": rows,
                        },
                        session_dir / "manifest.json",
                    )
                    _write_runs_csv(rows, session_dir / "runs.csv")
                    raise
        row["duration_sec"] = float(time.perf_counter() - t0)

        save_json(
            {
                "run_name": run_name,
                "experiment": exp_name,
                "status": row["status"],
                "duration_sec": row["duration_sec"],
                "params": row["params"],
                "outputs": row["outputs"],
                "error": row["error"],
                "traceback": row.get("traceback", ""),
            },
            run_dir / "run_meta.json",
        )
        rows.append(row)

    manifest = {
        "sweep_name": sweep_name,
        "session_id": session_id,
        "config_path": str(config_path),
        "experiment": exp_name,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_sec": float(time.perf_counter() - t_all),
        "dry_run": bool(dry_run),
        "fail_fast": bool(fail_fast),
        "ok_count": int(ok_count),
        "error_count": int(err_count),
        "total_runs": int(len(combos)),
        "rows": rows,
    }
    save_json(manifest, session_dir / "manifest.json")
    _write_runs_csv(rows, session_dir / "runs.csv")
    (sweep_root / "latest_session.txt").write_text(f"{session_id}\n", encoding="utf-8")
    return {
        "sweep_root": str(sweep_root),
        "session_dir": str(session_dir),
        "manifest": str(session_dir / "manifest.json"),
        "runs_csv": str(session_dir / "runs.csv"),
        "ok_count": int(ok_count),
        "error_count": int(err_count),
        "total_runs": int(len(combos)),
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run sweep configurations for simulation_project experiments.")
    parser.add_argument("--config", default="simulation_project/config/sweeps.yaml", help="Path to sweeps.yaml")
    parser.add_argument("--list", action="store_true", help="List available sweep names from config")
    parser.add_argument("--sweep", default="", help="Sweep name to run")
    parser.add_argument("--save-dir", default=None, help="Override base save_dir for this sweep session")
    parser.add_argument(
        "--workspace",
        default="simulation_project",
        help="Workspace root for organized outputs (default resolves to outputs/simulation_project).",
    )
    parser.add_argument("--max-runs", type=int, default=None, help="Limit number of expanded grid runs")
    parser.add_argument("--dry-run", action="store_true", help="Expand and validate sweep without executing runs")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at first failed run")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override fixed/default parameter; can be repeated, e.g. --set repeats=10 --set n_jobs=2",
    )
    args = parser.parse_args()

    if args.list:
        rows = list_sweeps(config_path=args.config)
        if not rows:
            print("No sweeps found.")
            return
        for row in rows:
            desc = row["description"].strip()
            print(f"- {row['name']} (experiment={row['experiment']})")
            if desc:
                print(f"  {desc}")
        return

    if not str(args.sweep).strip():
        raise SystemExit("Please provide --sweep <name> or use --list.")

    overrides = _parse_set_items(args.set)
    save_dir_use = args.save_dir
    if save_dir_use is None:
        save_dir_use = str(resolve_workspace_dir(args.workspace, create=True))
    out = run_sweep(
        sweep_name=str(args.sweep).strip(),
        config_path=str(args.config),
        save_dir=save_dir_use,
        max_runs=args.max_runs,
        dry_run=bool(args.dry_run),
        fail_fast=bool(args.fail_fast),
        overrides=overrides,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()


