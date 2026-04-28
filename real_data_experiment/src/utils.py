from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

MASTER_SEED = 20260426


def ensure_dir(path: Path | str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(payload: Any, path: Path | str) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


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
    with index_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
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


def standardize_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    center: bool = True,
    scale: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(X_train, dtype=float)
    test = np.asarray(X_test, dtype=float)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("X_train and X_test must be 2D arrays with the same column count.")
    mu = train.mean(axis=0, keepdims=True) if center else np.zeros((1, train.shape[1]), dtype=float)
    sd = train.std(axis=0, ddof=0, keepdims=True) if scale else np.ones((1, train.shape[1]), dtype=float)
    sd = np.where(sd < 1e-10, 1.0, sd)
    return (train - mu) / sd, (test - mu) / sd, mu.reshape(-1), sd.reshape(-1)


def center_train_test(
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    train = np.asarray(y_train, dtype=float).reshape(-1)
    test = np.asarray(y_test, dtype=float).reshape(-1)
    mu = float(np.mean(train))
    return train - mu, test - mu, mu


def center_scale_train_test(
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    train = np.asarray(y_train, dtype=float).reshape(-1)
    test = np.asarray(y_test, dtype=float).reshape(-1)
    mu = float(np.mean(train))
    sd = float(np.std(train, ddof=0))
    if sd < 1e-10:
        sd = 1.0
    return (train - mu) / sd, (test - mu) / sd, mu, sd


def stringify_groups(groups: Sequence[Sequence[int]]) -> str:
    return "[" + ",".join("[" + ",".join(str(int(i)) for i in group) + "]" for group in groups) + "]"


def group_sizes_from_groups(groups: Sequence[Sequence[int]]) -> list[int]:
    return [int(len(group)) for group in groups]
