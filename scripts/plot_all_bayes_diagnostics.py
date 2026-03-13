"""Batch-generate posterior diagnostics for all runs with posterior samples."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import subprocess


def _iter_run_dirs(outputs_root: Path) -> Iterable[Path]:
    for path in outputs_root.rglob("posterior_samples.npz"):
        yield path.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate diagnostics for all Bayesian runs.")
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs"),
        help="Root folder to scan for posterior_samples.npz (default: outputs).",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=Path("outputs") / "reports" / "bayes_diagnostics",
        help="Root folder to write diagnostics (default: outputs/reports/bayes_diagnostics).",
    )
    parser.add_argument("--max", type=int, default=None, help="Maximum number of runs to process.")
    parser.add_argument("--skip", type=int, default=0, help="Number of runs to skip from the start.")
    parser.add_argument("--dry-run", action="store_true", help="List runs without generating plots.")
    parser.add_argument("--dpi", type=int, default=180, help="DPI for saved figures.")
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Skip runs that already have posterior_density_strong.png in the destination.",
    )
    args = parser.parse_args()

    outputs_root = args.outputs_root.expanduser().resolve()
    dest_root = args.dest_root.expanduser().resolve()
    run_dirs = sorted(set(_iter_run_dirs(outputs_root)))

    if args.skip:
        run_dirs = run_dirs[args.skip :]
    if args.max is not None:
        run_dirs = run_dirs[: args.max]

    if not run_dirs:
        print("[WARN] No posterior_samples.npz found under outputs root.")
        return

    failures = []
    for idx, run_dir in enumerate(run_dirs, start=1):
        rel = run_dir.relative_to(outputs_root)
        dest = dest_root / rel
        if args.dry_run:
            print(f"[DRY] {run_dir} -> {dest}")
            continue
        if args.only_missing and (dest / "posterior_density_strong.png").exists():
            continue
        cmd = [
            sys.executable,
            "scripts/plot_diagnostics.py",
            "--run-dir",
            str(run_dir),
            "--dest",
            str(dest),
            "--dpi",
            str(args.dpi),
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            failures.append(f"{run_dir}: {exc}")

        if idx % 50 == 0:
            print(f"[INFO] Processed {idx}/{len(run_dirs)} runs...")

    if failures:
        print("[WARN] Some runs failed:")
        for item in failures:
            print(f"  - {item}")
        raise SystemExit(1)

    print(f"[OK] Diagnostics written under {dest_root}")


if __name__ == "__main__":
    main()
