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
    """
    Load and recursively merge YAML configs. Honors a top-level `defaults`
    key by loading and merging parent configs (relative to the current file).
    """

    def _merge_into(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                _merge_into(dst[k], v)
            else:
                dst[k] = v
        return dst

    def _load_with_defaults(path: Path, seen: set[Path]) -> Dict[str, Any]:
        norm_path = path.resolve()
        if norm_path in seen:
            cycle = " -> ".join(str(p) for p in (*seen, norm_path))
            raise ValueError(f"Config defaults cycle detected: {cycle}")
        seen = set(seen)
        seen.add(norm_path)

        text = norm_path.read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config {norm_path} must be a YAML mapping at top-level.")

        defaults = data.pop("defaults", None)
        base: Dict[str, Any] = {}
        if defaults:
            if isinstance(defaults, (str, Path)):
                defaults = [defaults]
            if not isinstance(defaults, list):
                raise ValueError(f"'defaults' in {norm_path} must be string or list.")
            for item in defaults:
                ref = Path(item)
                if not ref.is_absolute():
                    candidates = [
                        (norm_path.parent / ref),
                        Path.cwd() / ref,
                    ]
                    resolved = None
                    for cand in candidates:
                        if cand.exists():
                            resolved = cand.resolve()
                            break
                    if resolved is None:
                        raise FileNotFoundError(f"Default config '{item}' referenced from {norm_path} not found.")
                    ref = resolved
                base = _merge_into(base, _load_with_defaults(ref, seen))
        return _merge_into(base, data)

    cfg: Dict[str, Any] = {}
    for p in paths:
        part = _load_with_defaults(p, set())
        _merge_into(cfg, part)
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



def _auto_inject_tau0(resolved_cfg: Dict[str, Any], *, n: int | None = None, p: int | None = None) -> Dict[str, Any]:
    """
    If model is GRwHS and tau0 not provided, set tau0 using the heuristic:
    unit_variance: tau0 = (s / (p - s)) / sqrt(n)
    unit_l2:       tau0 =  s / (p - s)
    """
    model = resolved_cfg.get("model", {})
    name = str(model.get("name", "")).lower()
    if not name.startswith("grwhs"):
        return resolved_cfg
    if "tau0" in model and model["tau0"] is not None:
        return resolved_cfg
    # fetch shapes if available
    data = resolved_cfg.get("data", {})
    if n is None:
        n = int(data.get("n", 0) or 0)
    if p is None:
        p = int(data.get("p", 0) or 0)
    s_guess = int(model.get("s_guess", max(1, p // 20))) if p else int(model.get("s_guess", 1))
    std = resolved_cfg.get("standardization", {})
    X_scaling = std.get("X", "unit_variance")
    try:
        from preprocess import tau0_heuristic  # local module
    except Exception:
        try:
            from data.preprocess import tau0_heuristic  # package layout
        except Exception:
            return resolved_cfg  # no heuristic available
    if p and n:
        tau0 = float(tau0_heuristic(n, p, s_guess, X_scaling))
        resolved_cfg.setdefault("model", {})["tau0"] = tau0
    return resolved_cfg

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
        resolved_cfg = _auto_inject_tau0(resolved_cfg)

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
