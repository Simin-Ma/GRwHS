from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_WORKSPACE = Path("outputs") / "simulation_project"
_AUTO_MARKERS = {"", "auto", "default", "none"}


def _is_auto(value: str | None) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in _AUTO_MARKERS


def _slug(text: str) -> str:
    raw = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(text).strip())
    raw = raw.strip("_")
    return raw or "run"


def _session_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def resolve_workspace_dir(workspace: str | None, *, create: bool = True) -> Path:
    if workspace is None or str(workspace).strip() == "":
        out = DEFAULT_WORKSPACE
    else:
        p = Path(str(workspace))
        if p.is_absolute():
            out = p
        else:
            parts = [str(part).lower() for part in p.parts]
            out = p if (parts and parts[0] == "outputs") else (Path("outputs") / p)
    if create:
        out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_explicit_save_dir(save_dir: str, *, workspace: str | None, create: bool = True) -> Path:
    raw = Path(str(save_dir))
    if raw.is_absolute():
        out = raw
    else:
        parts = [str(part).lower() for part in raw.parts]
        if parts and parts[0] == "outputs":
            out = raw
        else:
            out = resolve_workspace_dir(workspace, create=create) / "adhoc" / raw
    if create:
        out.mkdir(parents=True, exist_ok=True)
    return out


def _update_workspace_index(*, workspace_dir: Path, save_dir: Path, run_label: str, mode: str) -> None:
    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_label": str(run_label),
        "mode": str(mode),
        "save_dir": str(save_dir),
    }
    latest = workspace_dir / "latest_session.json"
    latest.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
    (workspace_dir / "latest_session.txt").write_text(f"{save_dir}\n", encoding="utf-8")
    index_path = workspace_dir / "session_index.jsonl"
    with index_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def resolve_run_save_dir(
    save_dir: str | None,
    *,
    workspace: str | None,
    run_label: str,
) -> Path:
    workspace_dir = resolve_workspace_dir(workspace, create=True)
    if not _is_auto(save_dir):
        out = resolve_explicit_save_dir(str(save_dir), workspace=str(workspace_dir), create=True)
        _update_workspace_index(workspace_dir=workspace_dir, save_dir=out, run_label=run_label, mode="explicit")
        return out
    tag = _session_tag()
    out = workspace_dir / "sessions" / f"{tag}_{_slug(run_label)}"
    out.mkdir(parents=True, exist_ok=True)
    _update_workspace_index(workspace_dir=workspace_dir, save_dir=out, run_label=run_label, mode="auto_session")
    return out


def latest_session_dir(workspace: str | None) -> Path | None:
    workspace_dir = resolve_workspace_dir(workspace, create=False)
    latest_file = workspace_dir / "latest_session.txt"
    if not latest_file.exists():
        return None
    text = latest_file.read_text(encoding="utf-8").strip()
    if not text:
        return None
    p = Path(text)
    return p if p.exists() else None


def resolve_analysis_dir(save_dir: str | None, *, workspace: str | None) -> Path:
    if not _is_auto(save_dir):
        return resolve_explicit_save_dir(str(save_dir), workspace=workspace, create=False)
    latest = latest_session_dir(workspace)
    if latest is not None:
        return latest
    return resolve_workspace_dir(workspace, create=False)
