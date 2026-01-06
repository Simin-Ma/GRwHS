"""Randomly sample mixed-signal sweep configurations and select the best RMSE."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class RunResult:
    run_dir: Path
    sweep_summary: Path
    rmse: float
    model: str
    variation: dict


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_yaml(data: dict, path: Path) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _collect_best_rmse(summary_path: Path, *, target_model: str) -> Optional[RunResult]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    runs = payload.get("runs", [])
    best: Optional[RunResult] = None
    for run in runs:
        metrics = run.get("metrics", {})
        if metrics.get("model") != target_model:
            continue
        rmse = metrics.get("metrics", {}).get("RMSE")
        if rmse is None:
            continue
        rmse = float(rmse)
        if best is None or rmse < best.rmse:
            best = RunResult(
                run_dir=Path(run.get("run_dir", "")),
                sweep_summary=summary_path,
                rmse=rmse,
                model=target_model,
                variation=run.get("variation", run.get("config", {})),
            )
    return best


def _discover_summary(outdir: Path, *, prev: set[Path]) -> Optional[Path]:
    candidates = sorted(outdir.glob("sweep_summary_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        if path not in prev:
            return path
    return None


def run_random_samples(
    *,
    base_config: Path,
    sweep_config: Path,
    output_root: Path,
    samples: int,
    subset_size: int,
    base_seed: int,
    target_model: str,
) -> List[RunResult]:
    rng = random.Random(base_seed)
    sweep_spec = _load_yaml(sweep_config)
    variations: List[dict] = list(sweep_spec.get("variations", []))
    if not variations:
        raise ValueError(f"No variations found in {sweep_config}")

    output_root.mkdir(parents=True, exist_ok=True)

    previous_summaries = set(output_root.glob("**/sweep_summary_*.json"))
    best_runs: List[RunResult] = []

    for idx in range(samples):
        if subset_size > len(variations):
            subset = rng.sample(variations, len(variations))
        else:
            subset = rng.sample(variations, subset_size)

        sample_name = f"{sweep_spec.get('name', 'random')}_sample_{idx:02d}"
        sample_dir = output_root / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        temp_spec = dict(sweep_spec)
        temp_spec["name"] = sample_name
        temp_spec["variations"] = subset
        temp_spec.pop("output_dir", None)

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as tmp:
            _write_yaml(temp_spec, Path(tmp.name))
            tmp_path = Path(tmp.name)

        try:
            cmd = [
                sys.executable,
                "-m",
                "grrhs.cli.run_sweep",
                "--base-config",
                str(base_config),
                "--sweep-config",
                str(tmp_path),
                "--outdir",
                str(sample_dir),
            ]
            subprocess.run(cmd, check=True)
        finally:
            tmp_path.unlink(missing_ok=True)

        summary_path = _discover_summary(sample_dir, prev=previous_summaries)
        if summary_path is None:
            print(f"[WARN] No sweep summary discovered under {sample_dir}")
            continue
        previous_summaries.add(summary_path)

        best = _collect_best_rmse(summary_path, target_model=target_model)
        if best:
            best_runs.append(best)

    return best_runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Randomly sample mixed-signal sweeps and pick best RMSE.")
    parser.add_argument("--base-config", type=Path, default=Path("configs/experiments/exp1_group_regression.yaml"), help="Base experiment config.")
    parser.add_argument("--sweep-config", type=Path, default=Path("configs/sweeps/exp1_methods.yaml"), help="Sweep specification to sample from.")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/sweeps/random_exp1"), help="Output root for sampled sweeps.")
    parser.add_argument("--samples", type=int, default=3, help="Number of random subsets to run.")
    parser.add_argument("--subset-size", type=int, default=4, help="Number of variations per random sweep.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--target-model", default="grrhs_gibbs", help="Model name whose RMSE is evaluated.")
    args = parser.parse_args()

    results = run_random_samples(
        base_config=args.base_config,
        sweep_config=args.sweep_config,
        output_root=args.outdir,
        samples=args.samples,
        subset_size=args.subset_size,
        base_seed=args.seed,
        target_model=args.target_model,
    )

    if not results:
        print("[WARN] No successful runs to evaluate.")
        return

    best = min(results, key=lambda r: r.rmse)
    print("\n=== Best RMSE across sampled sweeps ===")
    print(f"Model: {best.model}")
    print(f"RMSE: {best.rmse:.6f}")
    print(f"Run directory: {best.run_dir}")
    print(f"Summary file: {best.sweep_summary}")


if __name__ == "__main__":
    main()


