# grwhs/cli/run_experiment.py
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyYAML is required: pip install pyyaml") from e


def _setup_basic_logging(verbosity: int) -> None:
    import logging

    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_overrides(pairs: List[str]) -> Dict[str, Any]:
    """
    Parse CLI overrides like:
      ['train.seed=42', 'model.svi.lr=1e-3', 'data.n=1000', 'flags.use_gpu=true']

    Returns a nested dict merged later into config.
    """
    from functools import reduce

    def cast_val(v: str) -> Any:
        low = v.lower()
        if low in {"true", "false"}:
            return low == "true"
        try:
            if "." in v:
                return float(v)
            return int(v)
        except ValueError:
            return v

    root: Dict[str, Any] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: '{item}'")
        k, v = item.split("=", 1)
        v = cast_val(v)
        keys = k.split(".")
        d = reduce(lambda acc, kk: acc.setdefault(kk, {}), keys[:-1], root)
        if not isinstance(d, dict):
            raise ValueError(f"Key path conflict at '{k}'")
        d[keys[-1]] = v
    return root


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _load_and_merge_configs(paths: List[Path]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            part = yaml.safe_load(f) or {}
        if not isinstance(part, dict):
            raise ValueError(f"Config {p} must be a YAML mapping at top-level.")
        _deep_update(cfg, part)
    return cfg


def _save_resolved_config(cfg: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "resolved_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _derive_run_dir(base_out: Path, exp_name: str | None) -> Path:
    tag = exp_name if exp_name else "exp"
    return base_out / f"{tag}-{_timestamp()}"


def _maybe_call_runner(resolved_cfg: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    """
    Delegates the actual experiment logic to grwhs.experiments.runner.run_experiment.
    If not present yet, we provide a helpful error and a minimal no-op fallback.

    Returns:
        metrics dict (will be saved as metrics.json).
    """
    try:
        from grwhs.experiments.runner import run_experiment  # type: ignore
    except Exception as e:
        # Helpful message + fallback
        print(
            "[WARN] grwhs.experiments.runner.run_experiment not found yet. "
            "Create it to run full pipeline.\n"
            "Fallback: we will just echo the config and write a dummy metrics.json.",
            file=sys.stderr,
        )
        # dummy minimal "success"
        return {
            "status": "DRY_RUN",
            "message": "runner.py not implemented yet; wrote resolved_config.yaml only.",
        }

    # Call actual runner
    return run_experiment(resolved_cfg, run_dir)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a single GRwHS experiment from YAML config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-c",
        nargs="+",
        type=str,
        required=True,
        help="One or more YAML config files (merged from left to right).",
    )
    parser.add_argument(
        "--override",
        "-o",
        nargs="*",
        default=[],
        help="Override config keys: e.g., train.seed=42 model.svi.lr=1e-3",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/runs",
        help="Base output directory for this run.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name tag used in run directory naming.",
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        action="count",
        default=0,
        help="Increase logging verbosity (-v INFO, -vv DEBUG).",
    )

    args = parser.parse_args(argv)
    _setup_basic_logging(args.verbosity)

    try:
        cfg_paths = [Path(p).expanduser().resolve() for p in args.config]
        for p in cfg_paths:
            if not p.exists():
                raise FileNotFoundError(f"Config not found: {p}")

        base_cfg = _load_and_merge_configs(cfg_paths)
        overrides = _parse_overrides(args.override or [])
        resolved_cfg = _deep_update(base_cfg, overrides)

        # Derive run dir and embed it into cfg
        base_out = Path(args.outdir).expanduser().resolve()
        run_dir = _derive_run_dir(base_out, args.name)
        resolved_cfg.setdefault("io", {})
        resolved_cfg["io"]["run_dir"] = str(run_dir)

        # Make output dir and save final config snapshot
        _save_resolved_config(resolved_cfg, run_dir)

        # Try to run the experiment
        metrics = _maybe_call_runner(resolved_cfg, run_dir)

        # Persist metrics
        with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"[OK] Run finished. Artifacts in: {run_dir}")
        return 0
    except Exception as e:  # pragma: no cover
        print("[FATAL] Experiment failed:\n", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
