from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def _method_label(name: str) -> str:
    mapping = {
        "GR_RHS": "GR-RHS",
        "RHS": "RHS",
    }
    return mapping.get(str(name), str(name))


def analyze_ga_v2_group_separation(results_dir: Path) -> dict[str, Any]:
    summary_path = results_dir / "summary_paired.csv"
    if not summary_path.exists():
        summary_path = results_dir / "summary.csv"
    if not summary_path.exists():
        return {"metrics": {}, "findings": ["summary.csv not found, skipping."]}
    rows = _load_csv(summary_path)
    if not rows:
        return {"metrics": {}, "findings": ["summary.csv is empty, skipping."]}
    by_method: dict[str, dict[str, float]] = {}
    for r in rows:
        method = str(r.get("method", "")).strip()
        if not method:
            continue
        by_method[method] = {
            "group_auroc": _float(r.get("group_auroc", "nan")),
            "mse_overall": _float(r.get("mse_overall", "nan")),
            "kappa_gap": _float(r.get("kappa_gap", "nan")),
            "null_group_mse": _float(r.get("null_group_mse", "nan")),
            "signal_group_mse": _float(r.get("signal_group_mse", "nan")),
            "n_effective": _float(r.get("n_effective", "nan")),
        }
    findings = [
        "GA-V2-A focuses on group-level mechanism metrics.",
        "  " + "  ".join(
            f"{_method_label(m)}: auroc={v.get('group_auroc', float('nan')):.3f} mse={v.get('mse_overall', float('nan')):.5f}"
            for m, v in by_method.items()
        ),
    ]
    return {"metrics": {"by_method": by_method}, "findings": findings}


def analyze_ga_v2_complexity_mismatch(results_dir: Path) -> dict[str, Any]:
    summary_path = results_dir / "summary_paired.csv"
    if not summary_path.exists():
        summary_path = results_dir / "summary.csv"
    if not summary_path.exists():
        return {"metrics": {}, "findings": ["summary.csv not found, skipping."]}
    rows = _load_csv(summary_path)
    if not rows:
        return {"metrics": {}, "findings": ["summary.csv is empty, skipping."]}
    cells: dict[str, dict[str, dict[str, float]]] = {}
    for r in rows:
        cell = f"{r.get('complexity_pattern', '')}/{r.get('within_group_pattern', '')}"
        method = str(r.get("method", "")).strip()
        if not cell.strip("/") or not method:
            continue
        cells.setdefault(cell, {})[method] = {
            "group_auroc": _float(r.get("group_auroc", "nan")),
            "mse_overall": _float(r.get("mse_overall", "nan")),
            "kappa_gap": _float(r.get("kappa_gap", "nan")),
        }
    findings = ["GA-V2-B compares complexity layouts under fixed total activity."]
    for cell in sorted(cells):
        parts = [cell + ":"]
        for method, vals in cells[cell].items():
            parts.append(
                f"{_method_label(method)} auroc={vals['group_auroc']:.3f} mse={vals['mse_overall']:.5f} kappa_gap={vals['kappa_gap']:.3f}"
            )
        findings.append("  ".join(parts))
    return {"metrics": {"cells": cells}, "findings": findings}


def analyze_ga_v2_correlation_stress(results_dir: Path) -> dict[str, Any]:
    summary_path = results_dir / "summary_paired.csv"
    if not summary_path.exists():
        summary_path = results_dir / "summary.csv"
    if not summary_path.exists():
        return {"metrics": {}, "findings": ["summary.csv not found, skipping."]}
    rows = _load_csv(summary_path)
    if not rows:
        return {"metrics": {}, "findings": ["summary.csv is empty, skipping."]}
    trend: dict[str, list[dict[str, float | str]]] = {}
    for r in rows:
        pat = str(r.get("within_group_pattern", "")).strip()
        method = str(r.get("method", "")).strip()
        if not pat or not method:
            continue
        trend.setdefault(pat, []).append(
            {
                "rho_within": _float(r.get("rho_within", "nan")),
                "method": method,
                "group_auroc": _float(r.get("group_auroc", "nan")),
                "mse_overall": _float(r.get("mse_overall", "nan")),
                "kappa_gap": _float(r.get("kappa_gap", "nan")),
            }
        )
    findings = ["GA-V2-C varies within-group correlation while holding sparsity structure fixed."]
    for pat in sorted(trend):
        findings.append(f"pattern={pat}")
        for rec in sorted(trend[pat], key=lambda d: (float(d["rho_within"]), str(d["method"]))):
            findings.append(
                "  "
                + f"rho={float(rec['rho_within']):.2f} {_method_label(str(rec['method']))} "
                + f"auroc={float(rec['group_auroc']):.3f} mse={float(rec['mse_overall']):.5f} "
                + f"kappa_gap={float(rec['kappa_gap']):.3f}"
            )
    return {"metrics": {"by_pattern": trend}, "findings": findings}


def analyze_group_aware_v2_suite(results_dir: Path) -> dict[str, Any]:
    suite_specs = [
        ("ga_v2a", "GA-V2-A", "ga_v2_group_separation", analyze_ga_v2_group_separation),
        ("ga_v2b", "GA-V2-B", "ga_v2_complexity_mismatch", analyze_ga_v2_complexity_mismatch),
        ("ga_v2c", "GA-V2-C", "ga_v2_correlation_stress", analyze_ga_v2_correlation_stress),
    ]
    findings: list[str] = []
    metrics: dict[str, Any] = {}
    for key, label, subdir, fn in suite_specs:
        exp_dir = results_dir / subdir
        if not exp_dir.exists():
            findings.append(f"{label}: missing results directory")
            metrics[key] = {}
            continue
        result = fn(exp_dir)
        metrics[key] = result.get("metrics", {})
        findings.append(label)
        findings.extend("  " + str(x) for x in result.get("findings", []))
    return {"metrics": metrics, "findings": findings}


def run_analysis(save_dir: str = "outputs/simulation_project") -> dict[str, Any]:
    base = Path(save_dir)
    res = base / "results"
    ga_v2_root = res / "group_aware_v2"
    report_lines = ["=" * 68, "SIMULATION RESULTS ANALYSIS -- GA-V2 suite", "=" * 68]
    all_metrics: dict[str, Any] = {}
    if ga_v2_root.exists():
        suite_result = analyze_group_aware_v2_suite(ga_v2_root)
        all_metrics["group_aware_v2_suite"] = suite_result.get("metrics", {})
        report_lines.extend(suite_result.get("findings", []))
    else:
        report_lines.append(f"[results directory not found: {ga_v2_root}]")
        all_metrics["group_aware_v2_suite"] = {}
    report_text = "\n".join(report_lines)
    _safe_print(report_text)
    res.mkdir(parents=True, exist_ok=True)
    (res / "analysis_report.txt").write_text(report_text, encoding="utf-8")
    with open(res / "analysis_report.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    with open(res / "diagnostics_runtime_table.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "method", "method_label", "n_total", "n_converged_ok", "convergence_rate"])
        writer.writeheader()
    return all_metrics
