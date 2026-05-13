from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


METHODS = ["GR_RHS", "RHS", "GIGG_MMLE", "GHS_plus_NUTS"]


def _num(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _candidate_artifact_dirs(source: Path) -> list[Path]:
    stem = source.stem
    parent = source.parent
    candidates = [
        parent / stem / "fit",
        parent / stem / "final" / "fit",
        parent / "fit",
        parent.parent / stem / "fit",
        parent.parent / stem / "final" / "fit",
    ]
    payload = _read_json(source)
    for key in ["fit_dir", "artifact_dir", "fit_artifact_dir"]:
        value = payload.get(key)
        if isinstance(value, str) and value:
            candidates.append(Path(value))
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        for key in ["fit_dir", "fit_summary", "beta_mean", "posterior_draws"]:
            value = artifacts.get(key)
            if isinstance(value, str) and value:
                path = Path(value)
                candidates.append(path.parent if path.suffix else path)

    out: list[Path] = []
    seen: set[str] = set()
    for cand in candidates:
        norm = str(cand)
        if norm not in seen:
            seen.add(norm)
            out.append(cand)
    return out


def _load_beta_from_artifacts(source_file: str) -> tuple[np.ndarray | None, str, str]:
    source = Path(source_file)
    payload = _read_json(source)
    beta = payload.get("beta_mean")
    if isinstance(beta, list):
        arr = np.asarray(beta, dtype=float).reshape(-1)
        if arr.size:
            return arr, "json:beta_mean", ""

    for fit_dir in _candidate_artifact_dirs(source):
        beta_path = fit_dir / "beta_mean.npy"
        if beta_path.exists():
            try:
                arr = np.load(beta_path).astype(float).reshape(-1)
            except Exception as exc:
                return None, "", f"failed to load {beta_path}: {exc}"
            return arr, str(beta_path), ""

        coef_path = fit_dir / "coefficient_detail.csv"
        if coef_path.exists():
            try:
                with coef_path.open(newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                arr = np.asarray([_num(row.get("beta_estimate")) for row in rows], dtype=float)
            except Exception as exc:
                return None, "", f"failed to load {coef_path}: {exc}"
            if arr.size:
                return arr, str(coef_path), ""

        draws_path = fit_dir / "posterior_draws.npz"
        if draws_path.exists():
            try:
                with np.load(draws_path) as bundle:
                    if "beta_draws" not in bundle:
                        continue
                    draws = np.asarray(bundle["beta_draws"], dtype=float)
                arr = draws.reshape(-1, draws.shape[-1]).mean(axis=0)
            except Exception as exc:
                return None, "", f"failed to load {draws_path}: {exc}"
            if arr.size:
                return arr, str(draws_path), ""

    return None, "", "missing beta_mean.npy / coefficient_detail.csv / posterior_draws.npz"


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size or a.size < 2:
        return math.nan
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def _pair_metrics(
    a: np.ndarray,
    b: np.ndarray,
    beta_true: np.ndarray | None,
) -> dict[str, float]:
    p = min(int(a.size), int(b.size))
    aa = a[:p]
    bb = b[:p]
    diff = aa - bb
    out = {
        "p": float(p),
        "corr_all": _safe_corr(aa, bb),
        "rmse_all": float(np.sqrt(np.mean(diff ** 2))) if p else math.nan,
        "mean_abs_diff_all": float(np.mean(np.abs(diff))) if p else math.nan,
        "cosine_all": float(np.dot(aa, bb) / (np.linalg.norm(aa) * np.linalg.norm(bb)))
        if p and np.linalg.norm(aa) > 0 and np.linalg.norm(bb) > 0
        else math.nan,
    }
    if beta_true is not None:
        truth = np.asarray(beta_true, dtype=float).reshape(-1)[:p]
        signal = np.abs(truth) > 1e-12
        null = ~signal
        for label, mask in [("signal", signal), ("null", null)]:
            if int(np.sum(mask)) == 0:
                out[f"corr_{label}"] = math.nan
                out[f"rmse_{label}"] = math.nan
                out[f"mean_abs_diff_{label}"] = math.nan
            else:
                out[f"corr_{label}"] = _safe_corr(aa[mask], bb[mask])
                out[f"rmse_{label}"] = float(np.sqrt(np.mean((aa[mask] - bb[mask]) ** 2)))
                out[f"mean_abs_diff_{label}"] = float(np.mean(np.abs(aa[mask] - bb[mask])))
    return out


def _load_truth_from_source(source_file: str) -> np.ndarray | None:
    source = Path(source_file)
    payload = _read_json(source)
    for key in ["beta_true", "coefficient_truth"]:
        value = payload.get(key)
        if isinstance(value, list):
            arr = np.asarray(value, dtype=float).reshape(-1)
            if arr.size:
                return arr

    for fit_dir in _candidate_artifact_dirs(source):
        coef_path = fit_dir / "coefficient_detail.csv"
        if not coef_path.exists():
            continue
        try:
            with coef_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            arr = np.asarray([_num(row.get("beta_true")) for row in rows], dtype=float)
        except Exception:
            continue
        if arr.size and np.isfinite(arr).any():
            return arr
    return None


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit whether posterior beta summaries are available and compare methods when possible."
    )
    parser.add_argument(
        "--raw",
        default="tmp/highdim_convergence_evidence_with_fixes/final_quality_comparison/quality_raw_best_converged.csv",
    )
    parser.add_argument(
        "--outdir",
        default="tmp/highdim_convergence_evidence_with_fixes/posterior_similarity",
    )
    args = parser.parse_args()

    with Path(args.raw).open(newline="", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))

    availability: list[dict[str, Any]] = []
    beta_by_case: dict[tuple[str, int, str], np.ndarray] = {}
    truth_by_case: dict[tuple[str, int], np.ndarray] = {}
    for row in raw_rows:
        setting = str(row["setting_id"])
        method = str(row["method"])
        replicate = int(row["replicate"])
        source = str(row["source_file"])
        beta, artifact, note = _load_beta_from_artifacts(source)
        has_beta = beta is not None and beta.size > 0
        availability.append(
            {
                "setting_id": setting,
                "method": method,
                "replicate": replicate,
                "has_beta_summary": bool(has_beta),
                "beta_dim": int(beta.size) if has_beta else 0,
                "artifact_source": artifact,
                "note": note,
                "source_file": source,
            }
        )
        if has_beta:
            beta_by_case[(setting, replicate, method)] = beta
        truth = _load_truth_from_source(source)
        if truth is not None:
            truth_by_case[(setting, replicate)] = truth

    pair_rows: list[dict[str, Any]] = []
    for setting in sorted({str(row["setting_id"]) for row in raw_rows}):
        reps = sorted({int(row["replicate"]) for row in raw_rows if str(row["setting_id"]) == setting})
        for rep in reps:
            available_methods = [
                method for method in METHODS
                if (setting, rep, method) in beta_by_case
            ]
            truth = truth_by_case.get((setting, rep))
            for m1, m2 in combinations(available_methods, 2):
                metrics = _pair_metrics(
                    beta_by_case[(setting, rep, m1)],
                    beta_by_case[(setting, rep, m2)],
                    truth,
                )
                pair_rows.append(
                    {
                        "setting_id": setting,
                        "replicate": rep,
                        "method_a": m1,
                        "method_b": m2,
                        "has_truth_partition": bool(truth is not None),
                        **metrics,
                    }
                )

    outdir = Path(args.outdir)
    _write_csv(
        outdir / "posterior_similarity_availability.csv",
        availability,
        [
            "setting_id",
            "method",
            "replicate",
            "has_beta_summary",
            "beta_dim",
            "artifact_source",
            "note",
            "source_file",
        ],
    )
    pair_fields = [
        "setting_id",
        "replicate",
        "method_a",
        "method_b",
        "has_truth_partition",
        "p",
        "corr_all",
        "rmse_all",
        "mean_abs_diff_all",
        "cosine_all",
        "corr_signal",
        "rmse_signal",
        "mean_abs_diff_signal",
        "corr_null",
        "rmse_null",
        "mean_abs_diff_null",
    ]
    _write_csv(outdir / "posterior_similarity_pairwise.csv", pair_rows, pair_fields)

    total = len(availability)
    available = sum(1 for row in availability if row["has_beta_summary"])
    complete_cases = 0
    for setting in sorted({str(row["setting_id"]) for row in raw_rows}):
        reps = sorted({int(row["replicate"]) for row in raw_rows if str(row["setting_id"]) == setting})
        for rep in reps:
            if all((setting, rep, method) in beta_by_case for method in METHODS):
                complete_cases += 1

    report = {
        "raw": str(args.raw),
        "n_runs": int(total),
        "n_runs_with_beta_summary": int(available),
        "n_complete_setting_replicates_with_all_four_methods": int(complete_cases),
        "n_pairwise_comparisons": int(len(pair_rows)),
        "interpretation": (
            "Pairwise posterior similarity can only be computed for runs with beta_mean.npy, "
            "coefficient_detail.csv, posterior_draws.npz, or beta_mean embedded in JSON. "
            "Missing summaries mean the current final quality JSONs are sufficient for convergence/MSE "
            "comparison but not sufficient for posterior similarity claims."
        ),
    }
    (outdir / "posterior_similarity_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
