from __future__ import annotations

import json
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


def prepare_history_run_dir(output_dir: Path | str, *, timestamp: str | None = None) -> tuple[Path, Path, str]:
    history_root = ensure_dir(output_dir)
    base_tag = str(timestamp or run_timestamp_tag())
    run_tag = base_tag
    run_dir = history_root / run_tag
    counter = 1
    while run_dir.exists():
        run_tag = f"{base_tag}_{counter:02d}"
        run_dir = history_root / run_tag
        counter += 1
    ensure_dir(run_dir)
    return history_root, run_dir, run_tag


def write_history_run_index(
    history_root: Path | str,
    *,
    run_timestamp: str,
    run_dir: Path | str,
    result_paths: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    root = ensure_dir(history_root)
    run_dir_path = Path(run_dir)
    entry = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_timestamp": str(run_timestamp),
        "run_dir": str(run_dir_path),
        "result_paths": dict(result_paths or {}),
    }
    latest_json = save_json(entry, root / "latest_run.json")
    latest_txt = root / "latest_run.txt"
    latest_txt.write_text(f"{run_dir_path}\n", encoding="utf-8")
    index_path = root / "session_index.jsonl"
    with index_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return {
        "history_root": str(root),
        "latest_run": str(latest_json),
        "latest_run_txt": str(latest_txt),
        "session_index": str(index_path),
    }


def _contains_required_files(base: Path, required_files: Sequence[str]) -> bool:
    if not required_files:
        return base.exists()
    return all((base / rel).exists() for rel in required_files)


def _latest_run_dir_from_root(root: Path) -> Path | None:
    latest_json = root / "latest_run.json"
    if latest_json.exists():
        try:
            payload = json.loads(latest_json.read_text(encoding="utf-8"))
            run_dir = payload.get("run_dir")
            if run_dir:
                candidate = Path(str(run_dir))
                if candidate.exists():
                    return candidate
        except (json.JSONDecodeError, OSError, TypeError):
            pass

    latest_txt = root / "latest_run.txt"
    if latest_txt.exists():
        try:
            text = latest_txt.read_text(encoding="utf-8").strip()
            if text:
                candidate = Path(text)
                if candidate.exists():
                    return candidate
        except OSError:
            pass
    return None


def resolve_history_results_dir(
    results_dir: Path | str,
    *,
    required_files: Sequence[str] = (),
) -> Path:
    root = Path(results_dir)
    if _contains_required_files(root, required_files):
        return root
    latest_run_dir = _latest_run_dir_from_root(root)
    if latest_run_dir is not None and _contains_required_files(latest_run_dir, required_files):
        return latest_run_dir
    return root
