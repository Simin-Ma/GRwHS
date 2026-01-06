"""Generate all plots for sweep runs matching a date token.

Example:
    python scripts/generate_plots_for_date.py --date 20260106
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RunKey:
    scenario: str
    snr_token: str
    timestamp: str


@dataclass(frozen=True)
class RunInfo:
    key: RunKey
    method: str
    path: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate all plots for sweep runs matching a date token.")
    parser.add_argument("--date", required=True, type=str, help="Date token embedded in run directory names (e.g., 20260106).")
    parser.add_argument(
        "--sweeps-root",
        type=Path,
        default=Path("outputs/sweeps"),
        help="Root directory containing sweep outputs (default: outputs/sweeps).",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=None,
        help="Root directory to write generated plots (default: outputs/figures/<date>).",
    )
    parser.add_argument(
        "--singleton-methods",
        nargs="+",
        default=["grrhs", "rhs"],
        help="Methods to run per-run plot scripts for (default: grrhs rhs).",
    )
    parser.add_argument(
        "--comparison-method-a",
        type=str,
        default="grrhs",
        help="Method A for pairwise comparisons (default: grrhs).",
    )
    parser.add_argument(
        "--comparison-method-b",
        type=str,
        default="rhs",
        help="Method B for pairwise comparisons (default: rhs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing them.",
    )
    return parser.parse_args()


def _iter_run_dirs(sweeps_root: Path, date_token: str) -> Iterable[Path]:
    if not sweeps_root.exists():
        return
    for path in sweeps_root.rglob(f"*{date_token}*"):
        if path.is_dir():
            yield path


def _parse_run_dir(path: Path) -> Optional[RunInfo]:
    scenario = path.parent.name
    name = path.name
    if "-" not in name:
        return None
    prefix, *rest = name.split("-")
    if not rest:
        return None
    timestamp = "-".join(rest)
    if "_" not in prefix:
        return None
    snr_token, method = prefix.split("_", 1)
    return RunInfo(key=RunKey(scenario=scenario, snr_token=snr_token, timestamp=timestamp), method=method, path=path)


def _run(cmd: Sequence[str], dry_run: bool) -> Tuple[bool, str]:
    cmd_str = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    if dry_run:
        return True, f"[DRY-RUN] {cmd_str}"
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, cmd_str
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        msg = stderr if stderr else stdout if stdout else repr(exc)
        return False, f"{cmd_str}\n{msg}"


def _find_sweep_csv_for_scenario(sweeps_root: Path, scenario: str, date_token: str) -> Optional[Path]:
    scenario_dir = sweeps_root / scenario
    if not scenario_dir.exists():
        return None
    candidates = sorted(scenario_dir.glob(f"sweep_comparison_{date_token}-*.csv"))
    return candidates[0] if candidates else None


def _ensure_dir(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Dict, dry_run: bool) -> None:
    if dry_run:
        return
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_run_like_dir(*, variation_dir: Path, dest_dir: Path, dry_run: bool) -> Path:
    """Create a minimal 'outputs/runs'-style directory for plotting scripts.

    Uses the first repeat/fold: repeat_001/fold_01.
    """
    import numpy as np
    import yaml

    from data.generators import generate_synthetic, synthetic_config_from_dict

    repeat_dir = sorted(variation_dir.glob("repeat_*"))
    if not repeat_dir:
        raise FileNotFoundError(f"No repeat_* directories found under {variation_dir}")
    repeat_dir = repeat_dir[0]
    fold_dir = sorted(repeat_dir.glob("fold_*"))
    if not fold_dir:
        raise FileNotFoundError(f"No fold_* directories found under {repeat_dir}")
    fold_dir = fold_dir[0]

    resolved_cfg_path = variation_dir / "resolved_config.yaml"
    if not resolved_cfg_path.exists():
        raise FileNotFoundError(f"resolved_config.yaml not found under {variation_dir}")
    resolved_cfg = yaml.safe_load(resolved_cfg_path.read_text(encoding="utf-8")) or {}

    arrays_path = fold_dir / "fold_arrays.npz"
    if not arrays_path.exists():
        raise FileNotFoundError(f"fold_arrays.npz not found under {fold_dir}")
    arrays = np.load(arrays_path)
    train_idx = np.asarray(arrays["train_idx"], dtype=int)
    test_idx = np.asarray(arrays["test_idx"], dtype=int)
    x_mean = np.asarray(arrays["x_mean"], dtype=float)
    x_scale = np.asarray(arrays["x_scale"], dtype=float)
    y_mean = float(np.asarray(arrays["y_mean"]).reshape(()).item())

    data_cfg = resolved_cfg.get("data")
    if data_cfg is None:
        raise ValueError(f"Resolved config missing 'data' block under {variation_dir}")
    seed = data_cfg.get("seed") or (resolved_cfg.get("seeds") or {}).get("data_generation")
    syn_cfg = synthetic_config_from_dict(
        data_cfg,
        seed=seed,
        name=resolved_cfg.get("name"),
        task=resolved_cfg.get("task"),
    )
    dataset = generate_synthetic(syn_cfg)
    X = np.asarray(dataset.X, dtype=float)
    y = np.asarray(dataset.y, dtype=float).reshape(-1)
    beta_raw = np.asarray(dataset.beta, dtype=float).reshape(-1)

    X_train_raw = X[train_idx]
    X_test_raw = X[test_idx]
    X_train = (X_train_raw - x_mean) / x_scale
    X_test = (X_test_raw - x_mean) / x_scale

    y_train = y[train_idx] - y_mean
    y_test = y[test_idx] - y_mean

    beta_true_std = beta_raw * x_scale

    if dry_run:
        return dest_dir

    dest_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest_dir / "dataset.npz",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        beta_true=beta_true_std,
    )

    posterior_path = fold_dir / "posterior_samples.npz"
    if posterior_path.exists():
        shutil.copy2(posterior_path, dest_dir / "posterior_samples.npz")

    metrics_path = fold_dir / "metrics.json"
    if metrics_path.exists():
        shutil.copy2(metrics_path, dest_dir / "metrics.json")

    dataset_meta_path = repeat_dir / "dataset_meta.json"
    if dataset_meta_path.exists():
        shutil.copy2(dataset_meta_path, dest_dir / "dataset_meta.json")
    else:
        _write_json(dest_dir / "dataset_meta.json", {}, dry_run=dry_run)

    shutil.copy2(resolved_cfg_path, dest_dir / "resolved_config.yaml")
    return dest_dir


def main() -> int:
    args = _parse_args()
    sweeps_root: Path = args.sweeps_root
    date_token: str = args.date
    dest_root: Path = args.dest_root or Path("outputs/figures") / date_token
    singleton_methods: List[str] = list(args.singleton_methods)
    method_a: str = args.comparison_method_a
    method_b: str = args.comparison_method_b
    dry_run: bool = bool(args.dry_run)

    run_infos: List[RunInfo] = []
    for d in _iter_run_dirs(sweeps_root, date_token):
        info = _parse_run_dir(d)
        if info is not None:
            run_infos.append(info)

    by_key: Dict[RunKey, Dict[str, RunInfo]] = {}
    for info in run_infos:
        by_key.setdefault(info.key, {})[info.method] = info

    ok: List[str] = []
    failed: List[str] = []

    # Per-run plots (singleton methods)
    for info in run_infos:
        if info.method not in singleton_methods:
            continue
        out_base = dest_root / info.key.scenario / info.key.snr_token / info.key.timestamp / info.method
        _ensure_dir(out_base, dry_run=dry_run)

        run_like = out_base / "run_like"
        check_dest = out_base / "check"
        diag_dest = out_base / "diagnostics"
        _ensure_dir(run_like, dry_run=dry_run)
        _ensure_dir(check_dest, dry_run=dry_run)
        _ensure_dir(diag_dest, dry_run=dry_run)

        try:
            _build_run_like_dir(variation_dir=info.path, dest_dir=run_like, dry_run=dry_run)
        except Exception as exc:
            failed.append(f"build_run_like: {info.path}\n{exc}")
            continue

        cmd_check = [sys.executable, "scripts/plot_check.py", str(run_like), "--out", str(check_dest)]
        success, msg = _run(cmd_check, dry_run=dry_run)
        (ok if success else failed).append(f"plot_check: {msg}")

        cmd_diag = [sys.executable, "scripts/plot_diagnostics.py", "--run-dir", str(run_like), "--dest", str(diag_dest)]
        success, msg = _run(cmd_diag, dry_run=dry_run)
        (ok if success else failed).append(f"plot_diagnostics: {msg}")

    # Pairwise comparison plots (method_a vs method_b)
    for key, methods in sorted(by_key.items(), key=lambda kv: (kv[0].scenario, kv[0].snr_token, kv[0].timestamp)):
        if method_a not in methods or method_b not in methods:
            continue
        a_dir = methods[method_a].path
        b_dir = methods[method_b].path
        out_pair = dest_root / key.scenario / key.snr_token / key.timestamp / f"{method_a}_vs_{method_b}"
        _ensure_dir(out_pair, dry_run=dry_run)

        sweep_csv = _find_sweep_csv_for_scenario(sweeps_root, key.scenario, date_token)

        group_out = out_pair / "group_level"
        _ensure_dir(group_out, dry_run=dry_run)
        cmd_group = [
            sys.executable,
            "scripts/plot_group_level_comparison.py",
            "--grrhs-dir",
            str(a_dir),
            "--rhs-dir",
            str(b_dir),
            "--output-dir",
            str(group_out),
            "--title",
            f"{key.scenario} ({key.snr_token})",
        ]
        if sweep_csv is not None:
            cmd_group += ["--sweep-csv", str(sweep_csv)]
        success, msg = _run(cmd_group, dry_run=dry_run)
        (ok if success else failed).append(f"plot_group_level_comparison: {msg}")

        coef_out = out_pair / "coef_recovery"
        _ensure_dir(coef_out, dry_run=dry_run)
        cmd_coef = [
            sys.executable,
            "scripts/plot_coefficient_recovery.py",
            "--grrhs-dir",
            str(a_dir),
            "--rhs-dir",
            str(b_dir),
            "--output-dir",
            str(coef_out),
            "--title",
            f"{key.scenario} ({key.snr_token})",
        ]
        success, msg = _run(cmd_coef, dry_run=dry_run)
        (ok if success else failed).append(f"plot_coefficient_recovery: {msg}")

        # Shrinkage structure plots require a dataset.npz; build run-like dirs and plot off those.
        shrink_out = out_pair / "shrinkage_structure"
        run_like_a = out_pair / f"run_like_{method_a}"
        run_like_b = out_pair / f"run_like_{method_b}"
        _ensure_dir(shrink_out, dry_run=dry_run)
        _ensure_dir(run_like_a, dry_run=dry_run)
        _ensure_dir(run_like_b, dry_run=dry_run)
        try:
            _build_run_like_dir(variation_dir=a_dir, dest_dir=run_like_a, dry_run=dry_run)
            _build_run_like_dir(variation_dir=b_dir, dest_dir=run_like_b, dry_run=dry_run)
        except Exception as exc:
            failed.append(f"build_run_like_pair: {a_dir} vs {b_dir}\n{exc}")
        else:
            cmd_shrink = [
                sys.executable,
                "scripts/plot_shrinkage_structure.py",
                "--run-dirs",
                str(run_like_a),
                str(run_like_b),
                "--dest",
                str(shrink_out),
                "--labels",
                method_a.upper(),
                method_b.upper(),
            ]
            success, msg = _run(cmd_shrink, dry_run=dry_run)
            (ok if success else failed).append(f"plot_shrinkage_structure: {msg}")

    # Scenario-level sweep CSV plots.
    for scenario_dir in sorted(sweeps_root.iterdir()) if sweeps_root.exists() else []:
        if not scenario_dir.is_dir():
            continue
        sweep_csv = _find_sweep_csv_for_scenario(sweeps_root, scenario_dir.name, date_token)
        if sweep_csv is None:
            continue
        out_path = dest_root / scenario_dir.name / f"mean_effective_nonzeros_{date_token}.png"
        _ensure_dir(out_path.parent, dry_run=dry_run)
        cmd_eff = [
            sys.executable,
            "scripts/plot_effective_nonzeros.py",
            "--sweep-csv",
            str(sweep_csv),
            "--output",
            str(out_path),
            "--title",
            f"{scenario_dir.name}: MeanEffectiveNonzeros vs SNR ({date_token})",
            "--labels",
            "grrhs",
            "rhs",
            "gigg",
            "lasso",
            "sgl",
            "ridge",
        ]
        success, msg = _run(cmd_eff, dry_run=dry_run)
        (ok if success else failed).append(f"plot_effective_nonzeros: {msg}")

    print(f"Found {len(run_infos)} run dirs matching date token {date_token}.")
    print(f"Wrote figures under: {dest_root}")
    print(f"Commands succeeded: {sum(1 for _ in ok)}")
    print(f"Commands failed: {sum(1 for _ in failed)}")
    if failed:
        print("\nFailures (first 20):")
        for item in failed[:20]:
            print("-", item.splitlines()[0])
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
