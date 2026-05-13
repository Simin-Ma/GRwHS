from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_SETTINGS = [
    "hd_setting_1_classical_anchor",
    "hd_setting_2_single_mode",
    "hd_setting_3_multimode_showcase",
]

DEFAULT_CANDIDATES = [
    {
        "label": "baseline",
        "description": "current config: tau_target=groups, local_scale=false, kappa~Beta(0.5,1.0)",
    },
    {
        "label": "kappa_b2",
        "description": "stronger null shrinkage: kappa~Beta(0.5,2.0)",
        "alpha_kappa": 0.5,
        "beta_kappa": 2.0,
    },
    {
        "label": "kappa_b3",
        "description": "intermediate null shrinkage: kappa~Beta(0.5,3.0)",
        "alpha_kappa": 0.5,
        "beta_kappa": 3.0,
    },
    {
        "label": "kappa_b4",
        "description": "stronger null shrinkage: kappa~Beta(0.5,4.0)",
        "alpha_kappa": 0.5,
        "beta_kappa": 4.0,
    },
    {
        "label": "kappa_b6",
        "description": "very strong null shrinkage: kappa~Beta(0.5,6.0)",
        "alpha_kappa": 0.5,
        "beta_kappa": 6.0,
    },
    {
        "label": "kappa_b8",
        "description": "boundary test for very strong null shrinkage: kappa~Beta(0.5,8.0)",
        "alpha_kappa": 0.5,
        "beta_kappa": 8.0,
    },
    {
        "label": "kappa_b10",
        "description": "stronger boundary test: kappa~Beta(0.5,10.0)",
        "alpha_kappa": 0.5,
        "beta_kappa": 10.0,
    },
    {
        "label": "kappa_b12",
        "description": "stronger boundary test: kappa~Beta(0.5,12.0)",
        "alpha_kappa": 0.5,
        "beta_kappa": 12.0,
    },
    {
        "label": "kappa_b8_tau05",
        "description": "kappa~Beta(0.5,8.0) with tau0 halved from the high-dimensional calibrated value",
        "alpha_kappa": 0.5,
        "beta_kappa": 8.0,
        "tau0": 0.00392837100659193,
    },
    {
        "label": "kappa_b10_tau05",
        "description": "kappa~Beta(0.5,10.0) with tau0 halved from the high-dimensional calibrated value",
        "alpha_kappa": 0.5,
        "beta_kappa": 10.0,
        "tau0": 0.00392837100659193,
    },
    {
        "label": "kappa_a025_b2",
        "description": "more spike near zero: kappa~Beta(0.25,2.0)",
        "alpha_kappa": 0.25,
        "beta_kappa": 2.0,
    },
    {
        "label": "local_kappa_b2",
        "description": "restore coefficient local scale with stronger kappa shrinkage",
        "alpha_kappa": 0.5,
        "beta_kappa": 2.0,
        "use_local_scale": True,
    },
    {
        "label": "shared_kappa_b2",
        "description": "one shared group kappa, stronger shrinkage",
        "alpha_kappa": 0.5,
        "beta_kappa": 2.0,
        "shared_kappa": True,
    },
]


def _json_scalar(value):
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return value
    return value if math.isfinite(value) else None


def _safe_float(value):
    try:
        value = float(value)
    except Exception:
        return math.nan
    return value if math.isfinite(value) else math.nan


def _run_candidate(
    *,
    python_exe: str,
    config: str,
    setting_id: str,
    replicate: int,
    outdir: Path,
    warmup: int,
    draws: int,
    adapt_delta: float,
    max_treedepth: int,
    seed_offset: int,
    candidate: dict[str, object],
    force: bool,
) -> dict[str, object]:
    label = str(candidate["label"])
    expected = (
        outdir
        / f"{setting_id}__GR_RHS__{label}__r{int(replicate)}__w{int(warmup)}_d{int(draws)}_s{int(seed_offset)}.json"
    )
    if expected.exists() and not force:
        return json.loads(expected.read_text(encoding="utf-8"))

    cmd = [
        python_exe,
        str(ROOT / "scripts" / "run_highdim_budget_probe_case.py"),
        "--config",
        config,
        "--setting-id",
        setting_id,
        "--method",
        "GR_RHS",
        "--replicate",
        str(int(replicate)),
        "--outdir",
        str(outdir.relative_to(ROOT) if outdir.is_relative_to(ROOT) else outdir),
        "--warmup",
        str(int(warmup)),
        "--draws",
        str(int(draws)),
        "--adapt-delta",
        str(float(adapt_delta)),
        "--max-treedepth",
        str(int(max_treedepth)),
        "--seed-offset",
        str(int(seed_offset)),
        "--label",
        label,
    ]
    if candidate.get("alpha_kappa") is not None:
        cmd += ["--grrhs-alpha-kappa", str(float(candidate["alpha_kappa"]))]
    if candidate.get("beta_kappa") is not None:
        cmd += ["--grrhs-beta-kappa", str(float(candidate["beta_kappa"]))]
    if candidate.get("tau0") is not None:
        cmd += ["--grrhs-tau0", str(float(candidate["tau0"]))]
    if candidate.get("use_local_scale") is not None:
        cmd.append("--grrhs-use-local-scale" if bool(candidate["use_local_scale"]) else "--grrhs-no-local-scale")
    if candidate.get("shared_kappa") is not None:
        cmd.append("--grrhs-shared-kappa" if bool(candidate["shared_kappa"]) else "--grrhs-no-shared-kappa")
    if candidate.get("tau_target") is not None:
        cmd += ["--grrhs-tau-target", str(candidate["tau_target"])]

    subprocess.run(cmd, cwd=ROOT, check=True)
    return json.loads(expected.read_text(encoding="utf-8"))


def _summarize(rows: list[dict[str, object]], outdir: Path) -> None:
    fields = [
        "candidate",
        "setting_id",
        "replicate",
        "converged",
        "rhat_max",
        "ess_min",
        "divergence_ratio",
        "mse_overall",
        "mse_signal",
        "mse_null",
        "wall_seconds",
        "runtime_seconds",
        "alpha_kappa",
        "beta_kappa",
        "use_local_scale",
        "shared_kappa",
        "tau_target",
        "description",
        "out_path",
    ]
    csv_path = outdir / "grrhs_shrinkage_scan_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})

    by_candidate: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_candidate.setdefault(str(row["candidate"]), []).append(row)

    baseline_rows = by_candidate.get("baseline", [])
    baseline_by_case = {
        (str(r["setting_id"]), int(r["replicate"])): r
        for r in baseline_rows
        if bool(r.get("converged"))
    }
    summary_rows: list[dict[str, object]] = []
    for candidate, cand_rows in by_candidate.items():
        conv_rows = [r for r in cand_rows if bool(r.get("converged"))]
        def mean_metric(key: str) -> float:
            vals = [_safe_float(r.get(key)) for r in conv_rows]
            vals = [v for v in vals if math.isfinite(v)]
            return float(sum(vals) / len(vals)) if vals else math.nan

        deltas = {"overall": [], "signal": [], "null": []}
        for r in conv_rows:
            b = baseline_by_case.get((str(r["setting_id"]), int(r["replicate"])))
            if not b:
                continue
            for metric, key in [
                ("overall", "mse_overall"),
                ("signal", "mse_signal"),
                ("null", "mse_null"),
            ]:
                rv = _safe_float(r.get(key))
                bv = _safe_float(b.get(key))
                if math.isfinite(rv) and math.isfinite(bv):
                    deltas[metric].append(rv - bv)
        summary_rows.append(
            {
                "candidate": candidate,
                "n": len(cand_rows),
                "n_converged": len(conv_rows),
                "mse_overall_mean": mean_metric("mse_overall"),
                "mse_signal_mean": mean_metric("mse_signal"),
                "mse_null_mean": mean_metric("mse_null"),
                "delta_overall_vs_baseline": (
                    sum(deltas["overall"]) / len(deltas["overall"]) if deltas["overall"] else math.nan
                ),
                "delta_signal_vs_baseline": (
                    sum(deltas["signal"]) / len(deltas["signal"]) if deltas["signal"] else math.nan
                ),
                "delta_null_vs_baseline": (
                    sum(deltas["null"]) / len(deltas["null"]) if deltas["null"] else math.nan
                ),
                "wall_seconds_mean": mean_metric("wall_seconds"),
                "description": str(conv_rows[0].get("description") if conv_rows else cand_rows[0].get("description")),
            }
        )
    summary_rows.sort(
        key=lambda r: (
            -int(r["n_converged"]),
            _safe_float(r["mse_overall_mean"]),
            _safe_float(r["mse_signal_mean"]),
        )
    )
    summary_path = outdir / "grrhs_shrinkage_scan_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    md = [
        "# GR-RHS high-dimensional shrinkage scan",
        "",
        "Only converged runs are used for the mean metrics. Deltas are candidate minus baseline on matched setting/replicate cases, so negative delta is better.",
        "",
        "| candidate | converged/n | mse_overall | delta overall | mse_signal | delta signal | mse_null | delta null | mean seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        md.append(
            "| {candidate} | {n_converged}/{n} | {mse_overall_mean:.6g} | {delta_overall_vs_baseline:.6g} | "
            "{mse_signal_mean:.6g} | {delta_signal_vs_baseline:.6g} | {mse_null_mean:.6g} | "
            "{delta_null_vs_baseline:.6g} | {wall_seconds_mean:.1f} |".format(**r)
        )
    md += ["", "## Candidate meanings", ""]
    for c in DEFAULT_CANDIDATES:
        md.append(f"- `{c['label']}`: {c['description']}")
    (outdir / "grrhs_shrinkage_scan_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def _rows_from_existing_json(outdir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(outdir.glob("hd_setting_*__GR_RHS__*__r*__w*_d*_s*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        name_parts = path.stem.split("__")
        if len(name_parts) < 5:
            continue
        candidate = name_parts[2]
        params = payload.get("grrhs_parameters", {}) if isinstance(payload.get("grrhs_parameters"), dict) else {}
        rows.append(
            {
                "candidate": str(candidate),
                "setting_id": str(payload.get("setting_id", name_parts[0])),
                "replicate": int(payload.get("replicate", 0)),
                "converged": bool(payload.get("converged")),
                "rhat_max": _json_scalar(payload.get("rhat_max")),
                "ess_min": _json_scalar(payload.get("ess_min")),
                "divergence_ratio": _json_scalar(payload.get("divergence_ratio")),
                "mse_overall": _json_scalar(payload.get("mse_overall")),
                "mse_signal": _json_scalar(payload.get("mse_signal")),
                "mse_null": _json_scalar(payload.get("mse_null")),
                "wall_seconds": _json_scalar(payload.get("wall_seconds")),
                "runtime_seconds": _json_scalar(payload.get("runtime_seconds")),
                "alpha_kappa": params.get("alpha_kappa"),
                "beta_kappa": params.get("beta_kappa"),
                "use_local_scale": params.get("use_local_scale"),
                "shared_kappa": params.get("shared_kappa"),
                "tau_target": params.get("tau_target"),
                "description": next(
                    (
                        str(c.get("description", ""))
                        for c in DEFAULT_CANDIDATES
                        if str(c.get("label")) == str(candidate)
                    ),
                    "",
                ),
                "out_path": str(path),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan GR-RHS shrinkage hyperparameters on high-dimensional settings.")
    parser.add_argument("--config", default="Simulation_highdimension/config/highdimension.yaml")
    parser.add_argument("--outdir", default="tmp/highdim_grrhs_shrinkage_scan")
    parser.add_argument("--settings", nargs="+", default=DEFAULT_SETTINGS)
    parser.add_argument("--replicates", nargs="+", type=int, default=[1])
    parser.add_argument("--candidates", nargs="+", default=[c["label"] for c in DEFAULT_CANDIDATES])
    parser.add_argument("--warmup", type=int, default=240)
    parser.add_argument("--draws", type=int, default=720)
    parser.add_argument("--adapt-delta", type=float, default=0.92)
    parser.add_argument("--max-treedepth", type=int, default=13)
    parser.add_argument("--seed-offset", type=int, default=161)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize-existing", action="store_true")
    args = parser.parse_args()

    outdir = ROOT / str(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if bool(args.summarize_existing):
        _summarize(_rows_from_existing_json(outdir), outdir)
        return 0
    candidates_by_label = {str(c["label"]): c for c in DEFAULT_CANDIDATES}
    rows: list[dict[str, object]] = []
    for setting_id in args.settings:
        for replicate in args.replicates:
            for label in args.candidates:
                candidate = candidates_by_label[str(label)]
                payload = _run_candidate(
                    python_exe=str(args.python),
                    config=str(args.config),
                    setting_id=str(setting_id),
                    replicate=int(replicate),
                    outdir=outdir,
                    warmup=int(args.warmup),
                    draws=int(args.draws),
                    adapt_delta=float(args.adapt_delta),
                    max_treedepth=int(args.max_treedepth),
                    seed_offset=int(args.seed_offset),
                    candidate=candidate,
                    force=bool(args.force),
                )
                params = payload.get("grrhs_parameters", {}) if isinstance(payload.get("grrhs_parameters"), dict) else {}
                rows.append(
                    {
                        "candidate": str(label),
                        "setting_id": str(setting_id),
                        "replicate": int(replicate),
                        "converged": bool(payload.get("converged")),
                        "rhat_max": _json_scalar(payload.get("rhat_max")),
                        "ess_min": _json_scalar(payload.get("ess_min")),
                        "divergence_ratio": _json_scalar(payload.get("divergence_ratio")),
                        "mse_overall": _json_scalar(payload.get("mse_overall")),
                        "mse_signal": _json_scalar(payload.get("mse_signal")),
                        "mse_null": _json_scalar(payload.get("mse_null")),
                        "wall_seconds": _json_scalar(payload.get("wall_seconds")),
                        "runtime_seconds": _json_scalar(payload.get("runtime_seconds")),
                        "alpha_kappa": params.get("alpha_kappa"),
                        "beta_kappa": params.get("beta_kappa"),
                        "use_local_scale": params.get("use_local_scale"),
                        "shared_kappa": params.get("shared_kappa"),
                        "tau_target": params.get("tau_target"),
                        "description": str(candidate.get("description", "")),
                        "out_path": payload.get("out_path"),
                    }
                )
                _summarize(rows, outdir)
                print(
                    json.dumps(
                        {
                            "candidate": label,
                            "setting_id": setting_id,
                            "replicate": int(replicate),
                            "converged": bool(payload.get("converged")),
                            "mse_overall": _json_scalar(payload.get("mse_overall")),
                            "mse_signal": _json_scalar(payload.get("mse_signal")),
                            "mse_null": _json_scalar(payload.get("mse_null")),
                        },
                        ensure_ascii=False,
                    )
                )
    _summarize(rows, outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
