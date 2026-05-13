from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


SETTINGS = [
    "hd_setting_1_classical_anchor",
    "hd_setting_2_single_mode",
    "hd_setting_3_multimode_showcase",
]
METHODS = ["GR_RHS", "RHS", "GIGG_MMLE", "GHS_plus_NUTS"]
PASS_JUDGEMENTS = {"PASS_STRONG", "PASS_MARGINAL"}
METRICS = [
    "mse_overall",
    "mse_signal",
    "mse_null",
    "coverage_95",
    "lpd_test",
    "runtime_seconds",
    "wall_seconds",
    "rhat_max",
    "ess_min",
]


def _num(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def _load_rows(evidence_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with evidence_path.open(newline="", encoding="utf-8") as f:
        for audit_row in csv.DictReader(f):
            setting = str(audit_row.get("setting_id", ""))
            method = str(audit_row.get("method", ""))
            judgement = str(audit_row.get("judgement", ""))
            if setting not in SETTINGS or method not in METHODS or judgement not in PASS_JUDGEMENTS:
                continue

            source = Path(str(audit_row["source_file"]))
            payload = json.loads(source.read_text(encoding="utf-8-sig"))
            row: dict[str, object] = {
                "setting_id": setting,
                "method": method,
                "replicate": int(audit_row["replicate"]),
                "converged": True,
                "judgement": judgement,
                "recorded_params": audit_row.get("recorded_params", ""),
                "max_param_rhat": _num(audit_row.get("max_param_rhat")),
                "min_param_ess": _num(audit_row.get("min_param_ess")),
                "source_file": str(source),
            }
            for metric in METRICS:
                row[metric] = _num(payload.get(metric))

            # Prefer the audited full-parameter diagnostics over model-level shortcuts.
            max_param_rhat = _num(row["max_param_rhat"])
            min_param_ess = _num(row["min_param_ess"])
            if math.isfinite(max_param_rhat):
                row["rhat_max"] = max_param_rhat
            if math.isfinite(min_param_ess):
                row["ess_min"] = min_param_ess
            rows.append(row)
    return rows


def _assert_complete(rows: list[dict[str, object]]) -> None:
    keys = {
        (str(row["setting_id"]), str(row["method"]), int(row["replicate"]))
        for row in rows
    }
    missing = [
        (setting, method, rep)
        for setting in SETTINGS
        for method in METHODS
        for rep in range(1, 6)
        if (setting, method, rep) not in keys
    ]
    if missing:
        raise RuntimeError(f"Missing passing evidence rows: {missing}")


def _summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for setting in SETTINGS:
        for method in METHODS:
            subset = [
                row for row in rows
                if row["setting_id"] == setting and row["method"] == method
            ]
            record: dict[str, object] = {
                "setting_id": setting,
                "method": method,
                "n": len(subset),
                "n_pass_strong": sum(row["judgement"] == "PASS_STRONG" for row in subset),
                "n_pass_marginal": sum(row["judgement"] == "PASS_MARGINAL" for row in subset),
            }
            for metric in METRICS:
                vals = [_num(row.get(metric)) for row in subset]
                vals = [val for val in vals if math.isfinite(val)]
                record[f"{metric}_mean"] = mean(vals) if vals else math.nan
                record[f"{metric}_sd"] = stdev(vals) if len(vals) > 1 else math.nan
            record["max_param_rhat_worst"] = max(
                _num(row.get("rhat_max")) for row in subset if math.isfinite(_num(row.get("rhat_max")))
            )
            record["min_param_ess_worst"] = min(
                _num(row.get("ess_min")) for row in subset if math.isfinite(_num(row.get("ess_min")))
            )
            summary.append(record)

    for setting in SETTINGS:
        subset = [row for row in summary if row["setting_id"] == setting]
        subset.sort(key=lambda row: (_num(row["mse_overall_mean"]), str(row["method"])))
        for rank, row in enumerate(subset, start=1):
            row["rank_mse_overall"] = rank
    summary.sort(key=lambda row: (str(row["setting_id"]), int(row["rank_mse_overall"])))
    return summary


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, summary: list[dict[str, object]]) -> None:
    lines = [
        "# Final high-dimensional Bayesian quality comparison",
        "",
        "Rule: same simulated datasets; each method enters only after all 5 replicates pass full posterior diagnostics for that method/model.",
        "",
        "| Setting | Rank | Method | MSE overall mean | MSE signal mean | MSE null mean | Coverage | worst Rhat | worst ESS | Wall sec mean | Pass |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['setting_id']} | {row['rank_mse_overall']} | {row['method']} | "
            f"{_num(row['mse_overall_mean']):.6g} | {_num(row['mse_signal_mean']):.6g} | "
            f"{_num(row['mse_null_mean']):.6g} | {_num(row['coverage_95_mean']):.4f} | "
            f"{_num(row['max_param_rhat_worst']):.6g} | {_num(row['min_param_ess_worst']):.1f} | "
            f"{_num(row['wall_seconds_mean']):.1f} | "
            f"{row['n_pass_strong']} strong / {row['n_pass_marginal']} marginal |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build quality comparison tables from audited converged posterior evidence."
    )
    parser.add_argument(
        "--evidence",
        default="tmp/highdim_convergence_evidence_with_fixes/posterior_convergence_evidence_best.csv",
    )
    parser.add_argument(
        "--outdir",
        default="tmp/highdim_convergence_evidence_with_fixes/final_quality_comparison",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(Path(args.evidence))
    _assert_complete(rows)

    raw_fields = list(rows[0].keys())
    _write_csv(outdir / "quality_raw_best_converged.csv", rows, raw_fields)

    summary = _summarize(rows)
    summary_fields = ["setting_id", "rank_mse_overall", "method", "n", "n_pass_strong", "n_pass_marginal"]
    for metric in METRICS:
        summary_fields.extend([f"{metric}_mean", f"{metric}_sd"])
    summary_fields.extend(["max_param_rhat_worst", "min_param_ess_worst"])
    _write_csv(outdir / "quality_summary_best_converged.csv", summary, summary_fields)
    _write_report(outdir / "quality_report_best_converged.md", summary)

    print(f"Wrote {outdir}")
    for row in summary:
        print(
            f"{row['setting_id']} | #{row['rank_mse_overall']} {row['method']} | "
            f"mse={_num(row['mse_overall_mean']):.6g} | "
            f"worst_rhat={_num(row['max_param_rhat_worst']):.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
