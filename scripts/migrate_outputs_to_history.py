from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


OUTPUTS_ROOT = Path("outputs")
HISTORY_ROOT = OUTPUTS_ROOT / "history"

SECOND_BENCHMARK_MARKERS = {
    "benchmark_spec.json",
    "paper_table_main.csv",
    "paper_table_main.md",
    "paper_table_appendix_full.csv",
    "paper_table_appendix_full.md",
}
MECHANISM_MARKERS = {
    "mechanism_spec.json",
    "paper_table_mechanism.csv",
    "paper_table_mechanism.md",
    "paper_table_mechanism_compact.csv",
    "paper_table_mechanism_compact.md",
}
SECOND_NAME_HINTS = (
    "simulation_second",
    "grrhs_sixway",
    "grrhs_vs_gigg",
    "weakid",
    "randomcoef",
    "classic_candidates",
    "stable_r",
    "advantage_region",
    "realistic_region",
    "paper_repro",
)
MECHANISM_NAME_HINTS = (
    "simulation_mechanism",
    "ga_v2",
    "mixed_boundary",
    "phase_map",
    "q_constraint",
)
MODULE_ROOT_NAMES = {"simulation_second", "simulation_mechanism"}
INDEXED_BUCKETS = {
    HISTORY_ROOT / "simulation_second" / "benchmark_main",
    HISTORY_ROOT / "simulation_second" / "samples",
    HISTORY_ROOT / "simulation_second" / "suite_samples",
    HISTORY_ROOT / "simulation_mechanism" / "mechanism_main",
    HISTORY_ROOT / "simulation_mechanism" / "samples",
    HISTORY_ROOT / "simulation_mechanism" / "suite_samples",
}


@dataclass(frozen=True)
class Decision:
    module: str
    bucket: str
    reason: str


@dataclass(frozen=True)
class SourceItem:
    path: Path
    preferred_module: str | None = None


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _iter_source_items(root: Path) -> Iterable[SourceItem]:
    for item in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if item.name.lower() == "history":
            continue
        if item.is_dir() and item.name in MODULE_ROOT_NAMES:
            preferred_module = item.name
            for child in sorted(item.iterdir(), key=lambda p: p.name.lower()):
                yield SourceItem(path=child, preferred_module=preferred_module)
            continue
        yield SourceItem(path=item)


def _read_csv_header(path: Path) -> set[str]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return {str(cell) for cell in next(csv.reader(handle))}
    except (OSError, StopIteration, UnicodeDecodeError, csv.Error):
        return set()


def _detect_module_from_headers(path: Path) -> Decision | None:
    for candidate in (path / "summary_paired.csv", path / "summary.csv", path / "raw_results.csv"):
        if not candidate.exists():
            continue
        header = _read_csv_header(candidate)
        if {"experiment_id", "primary_metric", "kappa_gap"} & header:
            return Decision("simulation_mechanism", "mechanism_main", f"header:{candidate.name}")
        if {"family", "target_r2", "setting_label"} & header:
            return Decision("simulation_second", "benchmark_main", f"header:{candidate.name}")
    return None


def _looks_like_dataset_sample_dir(path: Path) -> bool:
    files = [item for item in path.iterdir() if item.is_file()]
    if not files:
        return False
    suffixes = {item.suffix.lower() for item in files}
    return ".npz" in suffixes and ".json" in suffixes and ".csv" not in suffixes


def _classify_dir(path: Path, preferred_module: str | None = None) -> Decision:
    file_names = {item.name for item in path.rglob("*") if item.is_file()}
    low_name = path.name.lower()

    if file_names & SECOND_BENCHMARK_MARKERS:
        return Decision("simulation_second", "benchmark_main", "marker")
    if file_names & MECHANISM_MARKERS:
        return Decision("simulation_mechanism", "mechanism_main", "marker")
    if (path / "figures" / "figure1_mechanism_schematic.png").exists():
        return Decision("simulation_mechanism", "mechanism_main", "figure_marker")
    if (path / "paper_tables" / "figure_data").exists():
        return Decision("simulation_mechanism", "mechanism_main", "figure_data")
    if (path / "suite_manifest.json").exists():
        module = preferred_module or "unknown"
        if module in {"simulation_second", "simulation_mechanism"}:
            return Decision(module, "suite_samples", "suite_manifest")
    if _looks_like_dataset_sample_dir(path):
        module = preferred_module or "unknown"
        if module in {"simulation_second", "simulation_mechanism"}:
            return Decision(module, "samples", "dataset_sample")

    from_headers = _detect_module_from_headers(path)
    if from_headers is not None:
        return from_headers

    if any(hint in low_name for hint in SECOND_NAME_HINTS):
        return Decision("simulation_second", "benchmark_main", "name_hint")
    if any(hint in low_name for hint in MECHANISM_NAME_HINTS):
        return Decision("simulation_mechanism", "mechanism_main", "name_hint")
    if preferred_module in {"simulation_second", "simulation_mechanism"} and "sample" in low_name:
        return Decision(preferred_module, "samples", "preferred_module_sample")

    return Decision("unknown", "unclassified", "no_match")


def _classify_file(path: Path, preferred_module: str | None = None) -> Decision:
    low_name = path.name.lower()
    if path.suffix.lower() == ".csv":
        header = _read_csv_header(path)
        if {"experiment_id", "primary_metric", "kappa_gap"} & header:
            return Decision("simulation_mechanism", "imported_files", "header")
        if {"family", "target_r2", "setting_label"} & header:
            return Decision("simulation_second", "imported_files", "header")
    if any(hint in low_name for hint in SECOND_NAME_HINTS):
        return Decision("simulation_second", "imported_files", "name_hint")
    if any(hint in low_name for hint in MECHANISM_NAME_HINTS):
        return Decision("simulation_mechanism", "imported_files", "name_hint")
    if preferred_module in {"simulation_second", "simulation_mechanism"}:
        return Decision(preferred_module, "imported_files", "preferred_module")
    return Decision("unknown", "unclassified", "no_match")


def classify_item(source: SourceItem) -> Decision:
    if source.path.is_dir():
        return _classify_dir(source.path, preferred_module=source.preferred_module)
    return _classify_file(source.path, preferred_module=source.preferred_module)


def bucket_root(decision: Decision) -> Path:
    if decision.module == "simulation_second":
        if decision.bucket == "benchmark_main":
            return HISTORY_ROOT / "simulation_second" / "benchmark_main"
        if decision.bucket == "samples":
            return HISTORY_ROOT / "simulation_second" / "samples"
        if decision.bucket == "suite_samples":
            return HISTORY_ROOT / "simulation_second" / "suite_samples"
        if decision.bucket == "imported_files":
            return HISTORY_ROOT / "simulation_second" / "imported_files"
    if decision.module == "simulation_mechanism":
        if decision.bucket == "mechanism_main":
            return HISTORY_ROOT / "simulation_mechanism" / "mechanism_main"
        if decision.bucket == "samples":
            return HISTORY_ROOT / "simulation_mechanism" / "samples"
        if decision.bucket == "suite_samples":
            return HISTORY_ROOT / "simulation_mechanism" / "suite_samples"
        if decision.bucket == "imported_files":
            return HISTORY_ROOT / "simulation_mechanism" / "imported_files"
    return HISTORY_ROOT / "_unclassified"


def _write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def refresh_run_index(root: Path) -> None:
    if root not in INDEXED_BUCKETS:
        return
    run_dirs = [item for item in root.iterdir() if item.is_dir()]
    if not run_dirs:
        return
    run_dirs = sorted(run_dirs, key=lambda p: (p.stat().st_mtime, p.name))
    entries = []
    generated_at = datetime.now().isoformat(timespec="seconds")
    for run_dir in run_dirs:
        entries.append(
            {
                "generated_at": generated_at,
                "run_timestamp": run_dir.name,
                "run_dir": str(run_dir),
                "result_paths": {"migrated": True},
            }
        )
    latest = entries[-1]
    _write_json(root / "latest_run.json", latest)
    (root / "latest_run.txt").write_text(f"{latest['run_dir']}\n", encoding="utf-8")
    with (root / "session_index.jsonl").open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _delete_source_best_effort(path: Path) -> list[str]:
    failures: list[str] = []
    if not path.exists():
        return failures
    if path.is_file():
        try:
            path.unlink()
        except OSError as exc:
            failures.append(f"{path}: {type(exc).__name__}: {exc}")
        return failures

    for current_root, dir_names, file_names in os.walk(path, topdown=False):
        root_path = Path(current_root)
        for file_name in file_names:
            file_path = root_path / file_name
            try:
                file_path.unlink()
            except OSError as exc:
                failures.append(f"{file_path}: {type(exc).__name__}: {exc}")
        for dir_name in dir_names:
            dir_path = root_path / dir_name
            try:
                dir_path.rmdir()
            except OSError as exc:
                failures.append(f"{dir_path}: {type(exc).__name__}: {exc}")
    try:
        path.rmdir()
    except OSError as exc:
        failures.append(f"{path}: {type(exc).__name__}: {exc}")
    return failures


def _transfer_item(source: Path, target: Path) -> tuple[str, list[str]]:
    ensure_dir(target.parent)
    copied = False

    if source.is_dir():
        if not target.exists():
            try:
                shutil.move(str(source), str(target))
                return "moved", []
            except OSError:
                pass
        shutil.copytree(source, target, dirs_exist_ok=True)
        copied = True
    else:
        if not target.exists():
            try:
                shutil.move(str(source), str(target))
                return "moved", []
            except OSError:
                pass
        shutil.copy2(source, target)
        copied = True

    failures = _delete_source_best_effort(source)
    if failures:
        return ("copied_locked_source" if copied else "delete_partial"), failures
    return ("copied_then_removed" if copied else "removed"), []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate outputs/ artifacts into outputs/history/.")
    parser.add_argument("--dry-run", action="store_true", help="Show the migration plan without moving files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dir(HISTORY_ROOT / "simulation_second" / "benchmark_main")
    ensure_dir(HISTORY_ROOT / "simulation_second" / "samples")
    ensure_dir(HISTORY_ROOT / "simulation_second" / "suite_samples")
    ensure_dir(HISTORY_ROOT / "simulation_mechanism" / "mechanism_main")
    ensure_dir(HISTORY_ROOT / "simulation_mechanism" / "samples")
    ensure_dir(HISTORY_ROOT / "simulation_mechanism" / "suite_samples")
    ensure_dir(HISTORY_ROOT / "_unclassified")

    moves: list[dict[str, str]] = []
    counts: Counter[str] = Counter()

    for source in _iter_source_items(OUTPUTS_ROOT):
        decision = classify_item(source)
        destination_root = bucket_root(decision)
        target = destination_root / source.path.name
        counts[f"{decision.module}:{decision.bucket}"] += 1
        entry = {
            "source": str(source.path),
            "target": str(target),
            "module": decision.module,
            "bucket": decision.bucket,
            "reason": decision.reason,
            "item_type": "dir" if source.path.is_dir() else "file",
        }
        moves.append(entry)
        if args.dry_run:
            continue
        status, failures = _transfer_item(source.path, target)
        entry["status"] = status
        if failures:
            entry["failures"] = failures

    if not args.dry_run:
        for maybe_empty in [OUTPUTS_ROOT / "simulation_second", OUTPUTS_ROOT / "simulation_mechanism"]:
            if maybe_empty.exists() and maybe_empty.is_dir():
                try:
                    maybe_empty.rmdir()
                except OSError:
                    pass
        for root in INDEXED_BUCKETS:
            if root.exists():
                refresh_run_index(root)
        manifest = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "dry_run": False,
            "counts": dict(counts),
            "moves": moves,
        }
        _write_json(HISTORY_ROOT / "migration_manifest.json", manifest)
    else:
        manifest = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "dry_run": True,
            "counts": dict(counts),
            "moves": moves,
        }
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": False,
        "counts": dict(counts),
        "moved_items": len(moves),
        "history_root": str(HISTORY_ROOT),
        "migration_manifest": str(HISTORY_ROOT / "migration_manifest.json"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
