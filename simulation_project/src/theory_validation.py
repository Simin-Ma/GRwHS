from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd


@dataclass
class TheoryCheckResult:
    pass_all: bool
    checks: Dict[str, bool]
    metrics: Dict[str, float]
    notes: Dict[str, str]


def _linreg_slope_ci(x: np.ndarray, y: np.ndarray) -> tuple[float, tuple[float, float]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    x0 = x - x.mean()
    beta1 = float(np.sum(x0 * (y - y.mean())) / np.sum(x0 * x0))
    beta0 = float(y.mean() - beta1 * x.mean())
    resid = y - (beta0 + beta1 * x)
    s2 = float(np.sum(resid * resid) / max(n - 2, 1))
    se = math.sqrt(s2 / max(np.sum(x0 * x0), 1e-12))
    ci = (beta1 - 1.96 * se, beta1 + 1.96 * se)
    return beta1, ci


def _is_monotone_increasing(vals: Sequence[float], tol: float = 1e-6) -> bool:
    arr = np.asarray(vals, dtype=float)
    return bool(np.all(np.diff(arr) >= -float(tol)))


def _is_monotone_decreasing(vals: Sequence[float], tol: float = 1e-6) -> bool:
    arr = np.asarray(vals, dtype=float)
    return bool(np.all(np.diff(arr) <= float(tol)))


def _safe_min(arr: np.ndarray) -> float:
    return float(np.nanmin(arr)) if arr.size else float("nan")


def _safe_max(arr: np.ndarray) -> float:
    return float(np.nanmax(arr)) if arr.size else float("nan")


def validate_theory_results(save_dir: str = "simulation_project") -> TheoryCheckResult:
    base = Path(save_dir) / "results"
    exp1 = pd.read_csv(base / "exp1_null_contraction" / "summary.csv")
    exp2 = pd.read_csv(base / "exp2_adaptive_localization" / "summary.csv")
    exp3 = pd.read_csv(base / "exp3_phase_diagram" / "summary.csv")
    meta3 = json.loads((base / "exp3_phase_diagram" / "phase_threshold_meta.json").read_text(encoding="utf-8"))

    exp1_sorted = exp1.sort_values("p_g")
    x1 = np.log(exp1_sorted["p_g"].to_numpy(dtype=float))
    y1 = np.log(exp1_sorted["median_post_mean_kappa"].to_numpy(dtype=float))
    slope1, ci1 = _linreg_slope_ci(x1, y1)
    tail_cols = sorted([c for c in exp1_sorted.columns if str(c).startswith("mean_tail_prob_eps_")])
    if tail_cols:
        tail_col_use = tail_cols[0]
        tail_vals = exp1_sorted[tail_col_use].to_numpy(dtype=float)
    else:
        tail_col_use = "mean_tail_prob"
        tail_vals = exp1_sorted["mean_tail_prob"].to_numpy(dtype=float)
    pg1 = exp1_sorted["p_g"].to_numpy(dtype=float)
    tail_last = float(tail_vals[-1])

    tau_curves = []
    if "tau_eval" in exp2.columns:
        for tau in sorted(exp2["tau_eval"].dropna().unique().tolist()):
            sub = exp2.loc[np.isclose(exp2["tau_eval"], tau)].sort_values("p_g")
            tau_curves.append((float(tau), sub))
    else:
        tau_curves.append((float("nan"), exp2.sort_values("p_g")))
    ratio_last_err = []
    window_last = []
    ratio_monotone_ok = True
    window_monotone_ok = True
    for _, sub in tau_curves:
        r_vals = sub["median_ratio_R"].to_numpy(dtype=float)
        w_vals = sub["mean_window_prob"].to_numpy(dtype=float)
        if r_vals.size == 0:
            continue
        ratio_last_err.append(abs(float(r_vals[-1]) - 1.0))
        window_last.append(float(w_vals[-1]))
        # Ratio should stabilize and not diverge as p_g increases.
        ratio_monotone_ok = ratio_monotone_ok and bool(np.nanmax(np.abs(np.diff(r_vals))) < 0.7)
        # The fixed-width localization window should gain mass with p_g.
        window_monotone_ok = window_monotone_ok and _is_monotone_increasing(w_vals, tol=0.03)

    xi_crit = float(meta3["xi_crit"])
    rho = float(meta3["rho"])
    u0 = float(meta3["u0"])
    theta_expected = (u0 * (rho**2)) / max((u0 + (1.0 - u0) * (rho**2)), 1e-12)
    xi_expected = 0.5 * theta_expected
    xi_err = abs(xi_crit - xi_expected)
    exp3_eval = exp3.copy()
    if "xi_ratio" not in exp3_eval.columns:
        exp3_eval["xi_ratio"] = exp3_eval["xi"] / max(xi_crit, 1e-12)
    gt = exp3_eval.loc[exp3_eval["xi_ratio"] > 1.0].copy()
    lt = exp3_eval.loc[exp3_eval["xi_ratio"] < 1.0].copy()
    gt_line_pass = True
    gt_best_last = 0.0
    if not gt.empty:
        for xratio in sorted(gt["xi_ratio"].unique()):
            sub = gt.loc[np.isclose(gt["xi_ratio"], xratio)].sort_values("p_g")
            vals = sub["mean_prob_gt_u0"].to_numpy(dtype=float)
            gt_best_last = max(gt_best_last, float(vals[-1]))
            if len(vals) >= 3 and not _is_monotone_increasing(vals, tol=0.03):
                gt_line_pass = False
    lt_not_to_one = True
    lt_max_last = 0.0
    if not lt.empty:
        for xratio in sorted(lt["xi_ratio"].unique()):
            sub = lt.loc[np.isclose(lt["xi_ratio"], xratio)].sort_values("p_g")
            vals = sub["mean_prob_gt_u0"].to_numpy(dtype=float)
            lt_max_last = max(lt_max_last, float(vals[-1]))
            if vals[-1] > 0.70:
                lt_not_to_one = False

    checks = {
        "c1_exp1_slope_near_minus_half": (-0.65 < slope1 < -0.35),
        "c2_exp1_slope_ci_covers_minus_half": (ci1[0] <= -0.5 <= ci1[1]),
        "c3_exp1_fixed_tail_decreases": _is_monotone_decreasing(tail_vals, tol=0.02) and (tail_last < 0.20),
        "c4_exp2_ratio_converges_to_one": (len(ratio_last_err) > 0) and (max(ratio_last_err) < 0.40) and ratio_monotone_ok,
        "c5_exp2_window_mass_increases": (len(window_last) > 0) and (min(window_last) > 0.85) and window_monotone_ok,
        "c6_exp3_threshold_correct": xi_err < 1e-8,
        "c7_exp3_above_threshold_rises": gt_line_pass and (gt_best_last > 0.85),
        "c8_exp3_below_threshold_not_one": lt_not_to_one and (lt_max_last < 0.70),
    }
    pass_all = bool(all(checks.values()))
    metrics = {
        "exp1_slope": float(slope1),
        "exp1_slope_ci_low": float(ci1[0]),
        "exp1_slope_ci_high": float(ci1[1]),
        "exp1_tail_col_used": float(len(tail_cols)),
        "exp1_tail_last": tail_last,
        "exp1_tail_max": _safe_max(tail_vals),
        "exp1_p_max": float(pg1[-1]),
        "exp2_ratio_last_err_max": float(max(ratio_last_err)) if ratio_last_err else float("nan"),
        "exp2_window_last_min": float(min(window_last)) if window_last else float("nan"),
        "exp2_window_last_max": float(max(window_last)) if window_last else float("nan"),
        "exp3_xi_crit": xi_crit,
        "exp3_xi_expected": xi_expected,
        "exp3_gt_best_last": float(gt_best_last),
        "exp3_lt_max_last": float(lt_max_last),
    }
    notes = {
        "exp1_tail_rule": f"tail uses fixed epsilon via column {tail_col_use}; should decrease with p_g and approach 0.",
        "exp2_scale_rule": "R=E[kappa]/s_g should approach 1 and fixed-ratio window mass should increase toward 1 for each tau.",
        "exp3_phase_rule": "x-axis is xi/xi_crit; above-threshold curves should rise with p_g and below-threshold should stay away from 1.",
    }
    return TheoryCheckResult(pass_all=pass_all, checks=checks, metrics=metrics, notes=notes)


def write_theory_report(result: TheoryCheckResult, save_dir: str = "simulation_project") -> Dict[str, str]:
    out = Path(save_dir) / "results" / "theory_validation"
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "theory_check_report.json"
    md_path = out / "theory_check_report.md"
    payload = asdict(result)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Theory Validation Report",
        "",
        f"- overall: {'PASS' if result.pass_all else 'FAIL'}",
        "",
        "## Checks",
    ]
    for k, v in result.checks.items():
        lines.append(f"- {k}: {'PASS' if v else 'FAIL'}")
    lines.append("")
    lines.append("## Metrics")
    for k, v in result.metrics.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Notes")
    for k, v in result.notes.items():
        lines.append(f"- {k}: {v}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path)}
