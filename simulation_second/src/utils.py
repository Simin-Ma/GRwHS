from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, List, Mapping, Sequence, Tuple

import numpy as np

MASTER_SEED = 20260425


def ensure_dir(path: Path | str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(payload: Any, path: Path | str) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def canonical_groups(group_sizes: Sequence[int]) -> List[List[int]]:
    groups: List[List[int]] = []
    start = 0
    for size in group_sizes:
        width = int(size)
        if width <= 0:
            raise ValueError("group sizes must be positive")
        groups.append(list(range(start, start + width)))
        start += width
    return groups


def standardize_columns(X: np.ndarray) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    arr = arr - arr.mean(axis=0, keepdims=True)
    scale = arr.std(axis=0, ddof=0, keepdims=True)
    scale = np.where(scale < 1e-10, 1.0, scale)
    return arr / scale


def nearest_positive_definite(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    sym = 0.5 * (mat + mat.T)
    vals, vecs = np.linalg.eigh(sym)
    vals = np.maximum(vals, float(eps))
    out = (vecs * vals) @ vecs.T
    diag = np.sqrt(np.clip(np.diag(out), eps, None))
    out = out / np.outer(diag, diag)
    out = 0.5 * (out + out.T)
    np.fill_diagonal(out, 1.0)
    return out


def block_correlation(
    group_sizes: Sequence[int],
    rho_within: float,
    rho_between: float,
) -> np.ndarray:
    groups = canonical_groups(group_sizes)
    total_p = int(sum(group_sizes))
    corr = np.full((total_p, total_p), float(rho_between), dtype=float)
    rho_w = float(rho_within)
    for group in groups:
        idx = np.asarray(group, dtype=int)
        corr[np.ix_(idx, idx)] = rho_w
    np.fill_diagonal(corr, 1.0)
    return nearest_positive_definite(corr)


def sample_correlated_design(
    n: int,
    group_sizes: Sequence[int],
    rho_within: float,
    rho_between: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    corr = block_correlation(group_sizes, rho_within=rho_within, rho_between=rho_between)
    X = rng.multivariate_normal(np.zeros(corr.shape[0], dtype=float), corr, size=int(n))
    return standardize_columns(X), corr


def stable_string_seed(text: str) -> int:
    return int(sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(text))))


def setting_replicate_seed(setting_id: str, replicate_id: int, master_seed: int = MASTER_SEED) -> int:
    return int(master_seed + 1000 * stable_string_seed(setting_id) + int(replicate_id))


def run_timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def snapshot_result_files(
    output_dir: Path | str,
    result_paths: Mapping[str, Any],
    *,
    timestamp: str | None = None,
) -> dict[str, Any]:
    root = ensure_dir(output_dir)
    ts = str(timestamp or run_timestamp_tag())
    run_dir = ensure_dir(root / "runs" / ts)

    archived_paths: dict[str, str] = {}
    root_resolved = root.resolve()
    for name, value in dict(result_paths).items():
        if not isinstance(value, str):
            continue
        src = Path(value)
        if not src.exists() or not src.is_file():
            continue
        src_resolved = src.resolve()
        try:
            rel = src_resolved.relative_to(root_resolved)
        except ValueError:
            rel = Path(src.name)
        dst = run_dir / rel
        ensure_dir(dst.parent)
        shutil.copy2(src_resolved, dst)
        archived_paths[str(name)] = str(dst)

    manifest = {
        "run_timestamp": ts,
        "output_dir": str(root),
        "run_dir": str(run_dir),
        "archived_paths": dict(archived_paths),
    }
    run_manifest_path = save_json(manifest, run_dir / "run_manifest.json")
    latest_path = save_json(manifest, root / "latest_run.json")
    return {
        "run_timestamp": ts,
        "run_dir": str(run_dir),
        "run_archive_manifest": str(run_manifest_path),
        "latest_run": str(latest_path),
        "archived_paths": dict(archived_paths),
    }
