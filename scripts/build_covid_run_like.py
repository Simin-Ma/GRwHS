from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import yaml


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a plot_diagnostics-compatible run directory from a COVID fold.")
    parser.add_argument("--run-dir", required=True, type=Path, help="Variation directory such as trust_experts_grrhs-<timestamp>.")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat index (1-based).")
    parser.add_argument("--fold", type=int, default=1, help="Fold index (1-based).")
    parser.add_argument("--dest", required=True, type=Path, help="Destination directory for the run-like artifacts.")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    dest = args.dest.expanduser().resolve()
    repeat_dir = run_dir / f"repeat_{args.repeat:03d}"
    fold_dir = repeat_dir / f"fold_{args.fold:02d}"
    if not fold_dir.exists():
        raise FileNotFoundError(f"Fold directory not found: {fold_dir}")

    resolved_cfg = _load_yaml(run_dir / "resolved_config.yaml")
    loader_cfg = ((resolved_cfg.get("data") or {}).get("loader") or {})
    path_x = loader_cfg.get("path_X")
    path_y = loader_cfg.get("path_y")
    if not path_x or not path_y:
        raise ValueError("Resolved config does not expose loader path_X/path_y.")
    if loader_cfg.get("path_C"):
        raise ValueError("This helper is only for the COVID dataset path without covariates.")

    X = np.load(Path(path_x))
    y = np.load(Path(path_y))
    arrays = np.load(fold_dir / "fold_arrays.npz")
    train_idx = np.asarray(arrays["train_idx"], dtype=int)
    test_idx = np.asarray(arrays["test_idx"], dtype=int)
    x_mean = np.asarray(arrays["x_mean"], dtype=float)
    x_scale = np.asarray(arrays["x_scale"], dtype=float)
    y_mean_arr = np.asarray(arrays["y_mean"], dtype=float).reshape(-1)
    y_mean = float(y_mean_arr[0]) if y_mean_arr.size else 0.0

    X_train_raw = np.asarray(X[train_idx], dtype=float)
    X_test_raw = np.asarray(X[test_idx], dtype=float)
    X_train = (X_train_raw - x_mean) / x_scale
    X_test = (X_test_raw - x_mean) / x_scale
    y_train = np.asarray(y[train_idx], dtype=float) - y_mean
    y_test = np.asarray(y[test_idx], dtype=float) - y_mean

    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / "dataset.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    for src, dst_name in [
        (run_dir / "resolved_config.yaml", "resolved_config.yaml"),
        (repeat_dir / "dataset_meta.json", "dataset_meta.json"),
        (fold_dir / "metrics.json", "metrics.json"),
        (fold_dir / "fold_summary.json", "fold_summary.json"),
        (fold_dir / "posterior_samples.npz", "posterior_samples.npz"),
    ]:
        if src.exists():
            shutil.copy2(src, dest / dst_name)

    manifest = {
        "source_run_dir": str(run_dir),
        "source_repeat_dir": str(repeat_dir),
        "source_fold_dir": str(fold_dir),
        "dataset": "covid19_trust_experts",
        "repeat": int(args.repeat),
        "fold": int(args.fold),
    }
    (dest / "run_like_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[ok] run-like directory written to {dest}")


if __name__ == "__main__":
    main()
