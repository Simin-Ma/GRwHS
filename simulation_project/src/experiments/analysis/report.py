"""
Post-run analysis for Exp1-5 (including Exp3c/Exp3d).

Each analyze_expN() reads the saved CSVs/JSONs for that experiment and
returns a dict of key metrics plus a list of human-readable finding strings.

run_analysis() calls all configured experiments, prints a formatted report to stdout, and
saves results/analysis_report.json + results/analysis_report.txt.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_csv(path: Path) -> list[dict]:
    import csv
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def _pass_flag(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _method_label(name: str) -> str:
    mapping = {
        "RHS": "RHS [stan_rstanarm_hs]",
        "RHS_oracle": "RHS_oracle [stan_rstanarm_hs]",
    }
    return mapping.get(name, name)


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def _quantile(v: list[float], q: float) -> float:
    arr = np.asarray([float(x) for x in v if np.isfinite(float(x))], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, float(q)))


def _median(v: list[float]) -> float:
    arr = np.asarray([float(x) for x in v if np.isfinite(float(x))], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def _compute_diag_rows(rows: list[dict], *, exp_key: str, method_col: str, bayes_methods: set[str] | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not rows:
        return out
    methods = sorted({str(r.get(method_col, "")).strip() for r in rows if str(r.get(method_col, "")).strip()})
    for m in methods:
        if bayes_methods is not None and m not in bayes_methods:
            continue
        sub = [r for r in rows if str(r.get(method_col, "")).strip() == m]
        n_total = len(sub)
        conv_ok = [
            r for r in sub
            if _bool(r.get("converged", False)) and str(r.get("status", "")).strip().lower() == "ok"
        ]
        rt = [_float(r.get("runtime_seconds", "nan")) for r in sub]
        ess = [_float(r.get("bulk_ess_min", "nan")) for r in sub]
        rh = [_float(r.get("rhat_max", "nan")) for r in sub]
        div = [_float(r.get("divergence_ratio", "nan")) for r in sub]
        out.append(
            {
                "experiment": str(exp_key),
                "method": str(m),
                "method_label": _method_label(str(m)),
                "n_total": int(n_total),
                "n_converged_ok": int(len(conv_ok)),
                "convergence_rate": float(len(conv_ok) / max(n_total, 1)),
                "runtime_seconds_median": _median(rt),
                "runtime_seconds_p95": _quantile(rt, 0.95),
                "bulk_ess_min_median": _median(ess),
                "rhat_max_p95": _quantile(rh, 0.95),
                "divergence_ratio_mean": float(np.nanmean(np.asarray(div, dtype=float))) if len(div) else float("nan"),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Exp1 -- kappa_g profile regimes
# ---------------------------------------------------------------------------

def analyze_exp1(results_dir: Path) -> dict[str, Any]:
    findings: list[str] = []
    metrics: dict[str, Any] = {}

    # Panel A
    slope_path = results_dir / "null_slope_check.json"
    if slope_path.exists():
        sc = _load_json(slope_path)
        slope = _float(sc.get("slope", "nan"))
        ci = sc.get("slope_ci", [float("nan"), float("nan")])
        ci_lo, ci_hi = _float(ci[0]), _float(ci[1])
        ci_contains = sc.get("ci_contains_theory", ci_lo < -0.5 < ci_hi)
        passed = bool(sc.get("pass", False))
        fit_range = sc.get("fit_range_pg", "all")
        fit_str = f"{fit_range[0]}-{fit_range[1]}" if isinstance(fit_range, list) else str(fit_range)
        metrics["panel_A"] = {"slope": slope, "ci": [ci_lo, ci_hi], "ci_contains_theory": ci_contains, "pass": passed}
        findings.append(
            f"  Panel A -- Null Contraction (Thm 3.22):\n"
            f"    Fit range: p_g = {fit_str}\n"
            f"    Slope = {slope:.3f}  (theory = -0.500)  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]\n"
            f"    CI contains theory (-0.5): {'YES' if ci_contains else 'NO'}  -->  {_pass_flag(passed)}"
        )
    else:
        findings.append("  Panel A -- null_slope_check.json not found, skipping.")

    # Panel B
    phase_path = results_dir / "summary_phase.csv"
    if phase_path.exists():
        rows = _load_csv(phase_path)
        tau_vals = sorted({_float(r["tau"]) for r in rows})
        pg_vals  = sorted({int(_float(r["p_g"])) for r in rows})
        xi_vals  = sorted({_float(r["xi_ratio"]) for r in rows})

        # Monotonicity check across (tau, p_g) curves
        monotone_count = 0
        total_curves = 0
        for tau in tau_vals:
            for pg in pg_vals:
                sub = sorted(
                    [r for r in rows if abs(_float(r["tau"]) - tau) < 1e-6 and int(_float(r["p_g"])) == pg],
                    key=lambda r: _float(r["xi_ratio"])
                )
                if len(sub) < 2:
                    continue
                probs = [_float(r["mean_prob_kappa_gt_u0"]) for r in sub]
                total_curves += 1
                if all(probs[i] <= probs[i + 1] + 0.02 for i in range(len(probs) - 1)):
                    monotone_count += 1

        # Separation: mean P below vs above xi_crit
        below = [_float(r["mean_prob_kappa_gt_u0"]) for r in rows if _float(r["xi_ratio"]) < 1.0]
        above = [_float(r["mean_prob_kappa_gt_u0"]) for r in rows if _float(r["xi_ratio"]) > 1.0]
        mean_below = float(np.mean(below)) if below else float("nan")
        mean_above = float(np.mean(above)) if above else float("nan")
        separation = mean_above - mean_below

        # Sharpness for largest p_g
        largest_pg = max(pg_vals)
        hi_xi = max(xi_vals)
        lo_xi = min(xi_vals)
        hi_probs = [_float(r["mean_prob_kappa_gt_u0"]) for r in rows if int(_float(r["p_g"])) == largest_pg and abs(_float(r["xi_ratio"]) - hi_xi) < 1e-6]
        lo_probs = [_float(r["mean_prob_kappa_gt_u0"]) for r in rows if int(_float(r["p_g"])) == largest_pg and abs(_float(r["xi_ratio"]) - lo_xi) < 1e-6]
        sharpness_lo = float(np.mean(lo_probs)) if lo_probs else float("nan")
        sharpness_hi = float(np.mean(hi_probs)) if hi_probs else float("nan")

        metrics["panel_B"] = {
            "tau_vals": tau_vals, "pg_vals": pg_vals,
            "monotone_curves_pct": monotone_count / max(total_curves, 1),
            "mean_prob_below_xi_crit": mean_below,
            "mean_prob_above_xi_crit": mean_above,
            "separation": separation,
            "sharpness_largest_pg_lo": sharpness_lo,
            "sharpness_largest_pg_hi": sharpness_hi,
        }
        sharpness_line = ""
        if not math.isnan(sharpness_lo):
            sharpness_line = f"\n    Sharpness (p_g={largest_pg}): P(kappa>u0) rises from {sharpness_lo:.3f} to {sharpness_hi:.3f}"
        findings.append(
            f"  Panel B -- Phase Diagram (Cor. 3.33):\n"
            f"    tau values: {[round(t, 2) for t in tau_vals]}   p_g: {pg_vals}\n"
            f"    Monotone-increasing curves: {monotone_count}/{total_curves} "
            f"({100 * monotone_count / max(total_curves, 1):.0f}%)\n"
            f"    Mean P(kappa>u0) below xi_crit: {mean_below:.3f}   above xi_crit: {mean_above:.3f}\n"
            f"    Separation (above - below): {separation:+.3f}"
            + sharpness_line
        )
    else:
        findings.append("  Panel B -- summary_phase.csv not found, skipping.")

    return {"metrics": metrics, "findings": findings}


# ---------------------------------------------------------------------------
# Exp2 -- group separation
# ---------------------------------------------------------------------------

def analyze_exp2(results_dir: Path) -> dict[str, Any]:
    findings: list[str] = []

    summary_path = results_dir / "summary_paired.csv"
    if not summary_path.exists():
        summary_path = results_dir / "summary.csv"
    if not summary_path.exists():
        return {"metrics": {}, "findings": ["  summary.csv not found, skipping."]}

    rows = _load_csv(summary_path)
    lpd_metric = "lpd_test_ppd" if rows and ("lpd_test_ppd" in rows[0]) else "lpd_test"

    def _agg(metric: str, agg_fn=np.mean) -> dict[str, float]:
        out: dict[str, list] = {}
        for r in rows:
            m = r.get("method", "")
            v = _float(r.get(metric, "nan"))
            if not math.isnan(v):
                out.setdefault(m, []).append(v)
        return {m: float(agg_fn(vs)) for m, vs in out.items()}

    mse_by_m   = _agg("mse_overall")
    auroc_by_m = _agg("group_auroc")
    lpd_by_m   = _agg(lpd_metric)

    mse_rank   = sorted(mse_by_m.items(), key=lambda t: t[1])
    auroc_rank = sorted(auroc_by_m.items(), key=lambda t: t[1], reverse=True)
    lpd_rank   = sorted(lpd_by_m.items(), key=lambda t: t[1], reverse=True)

    def _rank_of(method: str, ranked_list: list) -> int | None:
        return next((i + 1 for i, (m, _) in enumerate(ranked_list) if m == method), None)

    gr_mse_rank   = _rank_of("GR_RHS", mse_rank)
    gr_auroc_rank = _rank_of("GR_RHS", auroc_rank)
    gr_lpd_rank   = _rank_of("GR_RHS", lpd_rank)
    gr_mse   = mse_by_m.get("GR_RHS", float("nan"))
    gr_auroc = auroc_by_m.get("GR_RHS", float("nan"))
    gr_lpd   = lpd_by_m.get("GR_RHS", float("nan"))

    n_methods = len(mse_rank)
    metrics = {
        "mse_by_method": mse_by_m,
        "auroc_by_method": auroc_by_m,
        "lpd_by_method": lpd_by_m,
        "gr_rhs_mse_rank": gr_mse_rank,
        "gr_rhs_auroc_rank": gr_auroc_rank,
        "gr_rhs_lpd_rank": gr_lpd_rank,
    }

    mse_table  = "    " + "  ".join(f"{_method_label(m)}: {v:.4f}" for m, v in mse_rank)
    auroc_table = "    " + "  ".join(f"{_method_label(m)}: {v:.3f}" for m, v in auroc_rank)
    gr_mse_str   = f"{gr_mse:.4f} (rank {gr_mse_rank}/{n_methods})" if gr_mse_rank else "not found in results"
    gr_auroc_str = f"{gr_auroc:.3f} (rank {gr_auroc_rank}/{n_methods})" if gr_auroc_rank else "not found"
    gr_lpd_str   = f"{gr_lpd:.3f} (rank {gr_lpd_rank}/{n_methods})" if gr_lpd_rank else "not found"

    findings.append(
        f"  MSE ranking (lower=better):\n{mse_table}\n"
        f"  AUROC ranking (higher=better):\n{auroc_table}\n"
        f"  GR_RHS -- MSE: {gr_mse_str}   AUROC: {gr_auroc_str}   {lpd_metric}: {gr_lpd_str}\n"
        "  RHS in this report denotes the unified Stan/HMC rstanarm-style baseline."
    )
    return {"metrics": metrics, "findings": findings}


# ---------------------------------------------------------------------------
# Exp3 -- linear benchmark
# ---------------------------------------------------------------------------

def analyze_exp3(results_dir: Path) -> dict[str, Any]:
    findings: list[str] = []
    metrics: dict[str, Any] = {}

    summary_path = results_dir / "summary_paired.csv"
    if not summary_path.exists():
        summary_path = results_dir / "summary.csv"
    if not summary_path.exists():
        return {"metrics": {}, "findings": ["  summary.csv not found, skipping."]}

    rows = _load_csv(summary_path)
    lpd_metric = "lpd_test_ppd" if rows and ("lpd_test_ppd" in rows[0]) else "lpd_test"
    signals   = sorted({r.get("signal", "") for r in rows if r.get("signal")})
    all_methods = sorted({r.get("method", "") for r in rows if r.get("method")})

    per_signal: dict[str, Any] = {}
    for sig in signals:
        sig_rows = [r for r in rows if r.get("signal") == sig]
        mse_by_m: dict[str, list] = {}
        lpd_by_m: dict[str, list] = {}
        for r in sig_rows:
            m = r.get("method", "")
            v_mse = _float(r.get("mse_overall", "nan"))
            v_lpd = _float(r.get(lpd_metric, "nan"))
            if not math.isnan(v_mse):
                mse_by_m.setdefault(m, []).append(v_mse)
            if not math.isnan(v_lpd):
                lpd_by_m.setdefault(m, []).append(v_lpd)
        mse_mean = {m: float(np.mean(vs)) for m, vs in mse_by_m.items()}
        lpd_mean = {m: float(np.mean(vs)) for m, vs in lpd_by_m.items()}
        mse_rank = sorted(mse_mean.items(), key=lambda t: t[1])
        per_signal[sig] = {"mse_rank": mse_rank, "lpd_mean": lpd_mean, "methods_present": list(mse_mean.keys())}

    metrics["per_signal"] = per_signal
    metrics["all_methods_in_summary"] = all_methods

    lines = [f"  Methods present in summary: {[_method_label(m) for m in all_methods]}"]
    missing = [m for m in ["GR_RHS", "RHS"] if m not in all_methods]
    if missing:
        lines.append(
            f"  NOTE: {[_method_label(m) for m in missing]} absent from summary (likely did not converge -- check logs)"
        )

    for sig, dat in per_signal.items():
        mse_rank = dat["mse_rank"]
        gr_rank  = next((i + 1 for i, (m, _) in enumerate(mse_rank) if m == "GR_RHS"), None)
        gr_mse   = next((v for m, v in mse_rank if m == "GR_RHS"), float("nan"))
        best_m, best_v = mse_rank[0] if mse_rank else ("?", float("nan"))
        n_m = len(mse_rank)
        detail = "    ".join(f"{_method_label(m)}: {v:.5f}" for m, v in mse_rank)

        if gr_rank is not None:
            rel = (gr_mse / best_v - 1) * 100 if best_v > 1e-12 else float("nan")
            gr_str = f"GR_RHS MSE={gr_mse:.5f} (rank {gr_rank}/{n_m}"
            if not math.isnan(rel):
                gr_str += f", {rel:+.1f}% vs best {best_m})"
            else:
                gr_str += ")"
        else:
            gr_str = "GR_RHS not in results (did not converge)"

        lines.append(f"  signal={sig}: {gr_str}")
        lines.append(f"    All: {detail}")

    findings.append("\n".join(lines))
    return {"metrics": metrics, "findings": findings}


# ---------------------------------------------------------------------------
# Exp4 -- variant ablation
# ---------------------------------------------------------------------------

def analyze_exp4(results_dir: Path) -> dict[str, Any]:
    findings: list[str] = []
    metrics: dict[str, Any] = {}

    summary_path = results_dir / "summary_paired.csv"
    if not summary_path.exists():
        summary_path = results_dir / "summary.csv"
    if not summary_path.exists():
        return {"metrics": {}, "findings": ["  summary.csv not found, skipping."]}

    rows = _load_csv(summary_path)
    p0_vals = sorted({int(_float(r.get("p0_true", 0))) for r in rows})

    per_p0: dict[int, dict] = {}
    for p0 in p0_vals:
        sub = [r for r in rows if int(_float(r.get("p0_true", -1))) == p0]
        mse_mean: dict[str, float] = {}
        mse_sem: dict[str, float] = {}
        mse_rel_rhs: dict[str, float] = {}
        mse_delta_rhs_pct: dict[str, float] = {}
        kappa_gap: dict[str, float] = {}
        tau_ratio: dict[str, float] = {}
        n_eff: dict[str, int] = {}

        for r in sub:
            v = r.get("variant", "")
            vm = _float(r.get("mse_overall", "nan"))
            vse = _float(r.get("mse_overall_sem", "nan"))
            vr = _float(r.get("mse_rel_rhs_oracle", "nan"))
            vd = _float(r.get("mse_delta_rhs_oracle_pct", "nan"))
            vg = _float(r.get("kappa_gap", "nan"))
            vt = _float(r.get("tau_ratio_to_oracle", "nan"))
            vn = int(_float(r.get("n_effective", 0)))
            if not math.isnan(vm):
                mse_mean[v] = float(vm)
            if not math.isnan(vse):
                mse_sem[v] = float(vse)
            if not math.isnan(vr):
                mse_rel_rhs[v] = float(vr)
            if not math.isnan(vd):
                mse_delta_rhs_pct[v] = float(vd)
            if not math.isnan(vg):
                kappa_gap[v] = float(vg)
            if not math.isnan(vt):
                tau_ratio[v] = float(vt)
            n_eff[v] = int(vn)

        per_p0[p0] = {
            "mse_mean": mse_mean,
            "mse_sem": mse_sem,
            "mse_rel_rhs_oracle": mse_rel_rhs,
            "mse_delta_rhs_oracle_pct": mse_delta_rhs_pct,
            "kappa_gap_mean": kappa_gap,
            "tau_ratio_mean": tau_ratio,
            "n_effective": n_eff,
        }

    metrics["per_p0"] = {str(k): v for k, v in per_p0.items()}

    lines = [
        "  p0 = true number of active coefficients (not active groups)",
        "  RHS_oracle [stan_rstanarm_hs] denotes the unified Stan/HMC rstanarm-style RHS baseline with oracle p0.",
    ]
    for p0, dat in per_p0.items():
        calib_mse = dat["mse_mean"].get("calibrated", float("nan"))
        calib_sem = dat["mse_sem"].get("calibrated", float("nan"))
        fix_mse = dat["mse_mean"].get("fixed_10x", float("nan"))
        rhs_mse = dat["mse_mean"].get("RHS_oracle", float("nan"))
        rel_rhs = dat["mse_rel_rhs_oracle"].get("calibrated", float("nan"))
        rel_fix = dat["mse_mean"].get("calibrated", float("nan")) / max(fix_mse, 1e-12) if not math.isnan(fix_mse) and fix_mse > 0 else float("nan")
        kappa_gap = dat["kappa_gap_mean"].get("calibrated", float("nan"))
        tau_r = dat["tau_ratio_mean"].get("calibrated", float("nan"))
        n_eff = dat["n_effective"].get("calibrated", 0)
        rel_rhs_str = f"{(rel_rhs - 1.0) * 100:+.1f}%" if not math.isnan(rel_rhs) else "nan"
        rel_fix_str = f"{(rel_fix - 1.0) * 100:+.1f}%" if not math.isnan(rel_fix) else "nan"
        sem_str = f"±{calib_sem:.5f}" if not math.isnan(calib_sem) else "±nan"
        kappa_str = f"{kappa_gap:.3f}" if not math.isnan(kappa_gap) else "nan"
        tau_str = f"{tau_r:.3f}" if not math.isnan(tau_r) else "nan"
        lines.append(
            f"  p0={p0}: calibrated MSE={calib_mse:.5f} {sem_str} (n={n_eff})"
            f"  vs RHS_oracle={rhs_mse:.5f} ({rel_rhs_str})"
            f"  vs fixed_10x={fix_mse:.5f} ({rel_fix_str})"
            f"  kappa_gap={kappa_str}  tau_ratio(diagnostic)={tau_str}"
        )

    findings.append("\n".join(lines))
    return {"metrics": metrics, "findings": findings}


# ---------------------------------------------------------------------------
# Exp5 -- prior sensitivity
# ---------------------------------------------------------------------------

def analyze_exp5(results_dir: Path) -> dict[str, Any]:
    findings: list[str] = []
    metrics: dict[str, Any] = {}

    summary_path = results_dir / "summary.csv"
    if not summary_path.exists():
        return {"metrics": {}, "findings": ["  summary.csv not found, skipping."]}

    rows = _load_csv(summary_path)
    settings = sorted({int(_float(r.get("setting_id", 0))) for r in rows})

    per_setting: dict[int, dict] = {}
    for sid in settings:
        sub = [r for r in rows if int(_float(r.get("setting_id", -1))) == sid]
        mse_null   = [_float(r.get("mse_null",   "nan")) for r in sub]
        mse_signal = [_float(r.get("mse_signal", "nan")) for r in sub]
        auroc      = [_float(r.get("group_auroc","nan")) for r in sub]
        kappa_null = [_float(r.get("kappa_null_mean","nan")) for r in sub]
        prior_pairs = [(round(_float(r.get("alpha_kappa","nan")), 2), round(_float(r.get("beta_kappa","nan")), 2)) for r in sub]
        per_setting[sid] = {
            "n_priors":         len(sub),
            "prior_pairs":      prior_pairs,
            "mse_null_range":   [float(np.nanmin(mse_null)),   float(np.nanmax(mse_null))],
            "mse_signal_range": [float(np.nanmin(mse_signal)), float(np.nanmax(mse_signal))],
            "auroc_range":      [float(np.nanmin(auroc)),      float(np.nanmax(auroc))],
            "mse_null_cv":      float(np.nanstd(mse_null)   / max(np.nanmean(mse_null),   1e-12)),
            "mse_signal_cv":    float(np.nanstd(mse_signal) / max(np.nanmean(mse_signal), 1e-12)),
            "kappa_null_range": [float(np.nanmin(kappa_null)), float(np.nanmax(kappa_null))],
        }

    metrics["per_setting"] = {str(k): v for k, v in per_setting.items()}

    lines = ["  Prior grid: (alpha_kappa, beta_kappa)"]
    for sid, dat in per_setting.items():
        auroc_lo, auroc_hi = dat["auroc_range"]
        mse_s_lo, mse_s_hi = dat["mse_signal_range"]
        cv_s = dat["mse_signal_cv"]
        stable_auroc = (auroc_hi - auroc_lo) < 0.05
        stable_mse   = cv_s < 0.20
        lines.append(
            f"  Setting {sid} ({dat['n_priors']} prior configs):\n"
            f"    AUROC range:      [{auroc_lo:.3f}, {auroc_hi:.3f}]  "
            f"{'(stable)' if stable_auroc else '(varies -- prior-sensitive!)'}\n"
            f"    MSE_signal range: [{mse_s_lo:.5f}, {mse_s_hi:.5f}]  "
            f"CV={cv_s:.2f}  {'(robust)' if stable_mse else '(prior-sensitive!)'}"
        )
    findings.append("\n".join(lines))
    return {"metrics": metrics, "findings": findings}


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------

def run_analysis(save_dir: str = "outputs/simulation_project") -> dict[str, Any]:
    base = Path(save_dir)
    res  = base / "results"

    sep   = "=" * 68
    sep2  = "-" * 60
    report_lines: list[str] = [sep, "SIMULATION RESULTS ANALYSIS -- Exp1-5(+optional Exp3c/Exp3d)", sep]

    all_metrics: dict[str, Any] = {}
    exp_configs = [
        ("exp1", "Exp1: kappa_g Profile Regimes (Thm 3.22, Cor. 3.33)",
         analyze_exp1, res / "exp1_kappa_profile_regimes"),
        ("exp2", "Exp2: Group Separation (Thm 3.34)",
         analyze_exp2, res / "exp2_group_separation"),
        ("exp3a", "Exp3a: Main Benchmark",
         analyze_exp3, res / "exp3a_main_benchmark"),
        ("exp3b", "Exp3b: Boundary Stress",
         analyze_exp3, res / "exp3b_boundary_stress"),
        ("exp3c", "Exp3c: Highdim Stress",
         analyze_exp3, res / "exp3c_highdim_stress"),
        ("exp3d", "Exp3d: Within-Group Mixed Stress",
         analyze_exp3, res / "exp3d_within_group_mixed"),
        ("exp4", "Exp4: Variant Ablation",
         analyze_exp4, res / "exp4_variant_ablation"),
        ("exp5", "Exp5: Prior Sensitivity",
         analyze_exp5, res / "exp5_prior_sensitivity"),
    ]

    for key, label, fn, exp_dir in exp_configs:
        report_lines.append(f"\n{label}")
        report_lines.append(sep2)
        if not exp_dir.exists():
            report_lines.append(f"  [results directory not found: {exp_dir}]")
            all_metrics[key] = {}
            continue
        result = fn(exp_dir)
        all_metrics[key] = result.get("metrics", {})
        for finding in result.get("findings", []):
            report_lines.append(finding)

    # Strict Bayesian convergence gate + diagnostics table.
    bayes_method_set = {"GR_RHS", "RHS", "GIGG_MMLE", "GIGG_b_small", "GIGG_GHS", "GIGG_b_large", "GHS_plus"}
    gate_specs = [
        ("exp2", res / "exp2_group_separation" / "raw_results.csv", "method", {"GR_RHS", "RHS"}, True),
        ("exp3a", res / "exp3a_main_benchmark" / "raw_results.csv", "method", bayes_method_set, True),
        ("exp3b", res / "exp3b_boundary_stress" / "raw_results.csv", "method", bayes_method_set, True),
        ("exp3c", res / "exp3c_highdim_stress" / "raw_results.csv", "method", bayes_method_set, False),
        ("exp3d", res / "exp3d_within_group_mixed" / "raw_results.csv", "method", bayes_method_set, False),
        ("exp4", res / "exp4_variant_ablation" / "raw_results.csv", "method_type", {"GR_RHS", "RHS"}, True),
        ("exp5", res / "exp5_prior_sensitivity" / "raw_results.csv", "", None, True),
    ]
    gate_ok = True
    gate_lines: list[str] = []
    diag_rows: list[dict[str, Any]] = []
    for exp_name, path, method_col, include_methods, required in gate_specs:
        if not path.exists():
            if required:
                gate_ok = False
                gate_lines.append(f"  {exp_name}: missing raw_results.csv")
            else:
                gate_lines.append(f"  {exp_name}: missing raw_results.csv  [SKIP optional]")
            continue
        rows = _load_csv(path)
        if not rows:
            if required:
                gate_ok = False
                gate_lines.append(f"  {exp_name}: empty raw_results.csv")
            else:
                gate_lines.append(f"  {exp_name}: empty raw_results.csv  [SKIP optional]")
            continue
        if method_col:
            target = [r for r in rows if str(r.get(method_col, "")).strip() in set(include_methods or set())]
            diag_rows.extend(_compute_diag_rows(rows, exp_key=exp_name, method_col=method_col, bayes_methods=set(include_methods or set())))
        else:
            target = list(rows)
            diag_rows.extend(_compute_diag_rows(rows, exp_key=exp_name, method_col="prior_id", bayes_methods=None))
        n_total = len(target)
        n_ok = sum(
            1 for r in target
            if _bool(r.get("converged", False)) and str(r.get("status", "")).strip().lower() == "ok"
        )
        pass_flag = n_ok == n_total and n_total > 0
        if required:
            gate_ok = gate_ok and pass_flag
            gate_lines.append(f"  {exp_name}: {n_ok}/{n_total} converged&ok  [{'PASS' if pass_flag else 'FAIL'}]")
        else:
            gate_lines.append(f"  {exp_name}: {n_ok}/{n_total} converged&ok  [{'PASS' if pass_flag else 'FAIL'} optional]")

    report_lines.append("\nStrict Convergence Gate")
    report_lines.append(sep2)
    report_lines.append(f"  Overall: {'PASS' if gate_ok else 'FAIL'}")
    report_lines.extend(gate_lines)
    all_metrics["strict_convergence_gate"] = {
        "overall_pass": bool(gate_ok),
        "details": gate_lines,
    }

    report_lines.append(f"\n{sep}")
    report_lines.append("END OF ANALYSIS")
    report_lines.append(sep)

    report_text = "\n".join(report_lines)
    _safe_print(report_text)

    out_dir = res
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis_report.txt").write_text(report_text, encoding="utf-8")
    # Diagnostics side-table for rebuttal support.
    import csv
    diag_fields = [
        "experiment",
        "method",
        "method_label",
        "n_total",
        "n_converged_ok",
        "convergence_rate",
        "runtime_seconds_median",
        "runtime_seconds_p95",
        "bulk_ess_min_median",
        "rhat_max_p95",
        "divergence_ratio_mean",
    ]
    with open(out_dir / "diagnostics_runtime_table.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=diag_fields)
        writer.writeheader()
        for row in diag_rows:
            writer.writerow({k: row.get(k) for k in diag_fields})
    all_metrics["diagnostics_runtime_table"] = diag_rows
    with open(out_dir / "analysis_report.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    return all_metrics
