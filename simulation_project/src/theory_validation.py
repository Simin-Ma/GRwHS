from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Sequence

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


def validate_theory_results(save_dir: str = "simulation_project") -> TheoryCheckResult:
    base = Path(save_dir) / "results"
    exp1 = pd.read_csv(base / "exp1_null_contraction" / "summary.csv")
    exp2 = pd.read_csv(base / "exp2_adaptive_localization" / "summary.csv")
    exp3 = pd.read_csv(base / "exp3_phase_diagram" / "summary.csv")
    meta3 = json.loads((base / "exp3_phase_diagram" / "phase_threshold_meta.json").read_text(encoding="utf-8"))

    x1 = np.log(exp1["p_g"].values)
    y1 = np.log(exp1["median_post_mean_kappa"].values)
    slope1, ci1 = _linreg_slope_ci(x1, y1)
    exp1_sorted = exp1.sort_values("p_g")
    tail_cols = [c for c in ("mean_tail_prob_M2", "mean_tail_prob_M3", "mean_tail_prob_M5") if c in exp1_sorted.columns]
    if tail_cols:
        tail_matrix = np.column_stack([exp1_sorted[c].to_numpy(dtype=float) for c in tail_cols])
        tail_vals = np.nanmean(tail_matrix, axis=1)
    else:
        tail_vals = exp1_sorted["mean_tail_prob"].to_numpy(dtype=float)
    scaled_kappa = exp1_sorted["sqrt_pg_times_mean_kappa"].to_numpy(dtype=float) if "sqrt_pg_times_mean_kappa" in exp1_sorted.columns else np.array([])
    pg1 = exp1.sort_values("p_g")["p_g"].to_numpy(dtype=float)
    tail_upper_half = tail_vals[len(tail_vals) // 2 :]
    tail_decline_upper = float(tail_upper_half[0] - tail_upper_half[-1]) if tail_upper_half.size >= 2 else 0.0
    tail_peak = float(np.max(tail_vals)) if tail_vals.size else float("nan")

    exp2_sorted = exp2.sort_values("p_g")
    r_vals = exp2_sorted["median_ratio_R"].to_numpy(dtype=float)
    win_vals = exp2_sorted["mean_window_prob"].to_numpy(dtype=float)
    r_lower = float(meta3.get("adaptive_x_lower", 0.3))
    r_upper = float(meta3.get("adaptive_x_upper", 3.0))

    xi_crit = float(meta3["xi_crit"])
    rho = float(meta3["rho"])
    u0 = float(meta3["u0"])
    theta_expected = (u0 * (rho**2)) / max((u0 + (1.0 - u0) * (rho**2)), 1e-12)
    xi_expected = 0.5 * theta_expected
    xi_err = abs(xi_crit - xi_expected)

    gt = exp3.loc[exp3["xi"] > xi_crit].copy()
    lt = exp3.loc[exp3["xi"] < xi_crit].copy()
    gt_line_pass = True
    gt_best_last = 0.0
    if not gt.empty:
        for xi in sorted(gt["xi"].unique()):
            sub = gt.loc[np.isclose(gt["xi"], xi)].sort_values("p_g")
            vals = sub["mean_prob_gt_u0"].to_numpy(dtype=float)
            gt_best_last = max(gt_best_last, float(vals[-1]))
            if len(vals) >= 3 and not _is_monotone_increasing(vals, tol=0.03):
                gt_line_pass = False
    lt_not_to_one = True
    lt_max_last = 0.0
    if not lt.empty:
        for xi in sorted(lt["xi"].unique()):
            sub = lt.loc[np.isclose(lt["xi"], xi)].sort_values("p_g")
            vals = sub["mean_prob_gt_u0"].to_numpy(dtype=float)
            lt_max_last = max(lt_max_last, float(vals[-1]))
            if vals[-1] > 0.75:
                lt_not_to_one = False

    checks = {
        "c1_exp1_slope_negative": slope1 < -0.45,
        "c2_exp1_slope_ci_excludes_zero": ci1[1] < 0.0,
        "c3_exp1_tail_declines": (tail_decline_upper > 0.03) and (tail_peak - tail_vals[-1] > 0.10) and (tail_vals[-1] < tail_peak),
        "c3b_exp1_scaled_kappa_bounded": bool(scaled_kappa.size == 0 or np.nanmax(scaled_kappa) < 2.5),
        "c4_exp2_R_nonzero": float(np.min(r_vals)) > r_lower,
        "c5_exp2_R_nondivergent": float(np.max(r_vals)) < r_upper,
        "c6_exp2_window_mass_high": float(win_vals[-1]) > 0.90,
        "c7_exp3_threshold_correct": xi_err < 1e-8,
        "c8_exp3_above_threshold_rises": gt_line_pass and (gt_best_last > 0.85),
        "c9_exp3_below_threshold_not_one": lt_not_to_one and (lt_max_last < 0.70),
    }
    pass_all = bool(all(checks.values()))
    metrics = {
        "exp1_slope": float(slope1),
        "exp1_slope_ci_low": float(ci1[0]),
        "exp1_slope_ci_high": float(ci1[1]),
        "exp1_tail_last": float(tail_vals[-1]),
        "exp1_tail_peak": tail_peak,
        "exp1_tail_decline_upper": float(tail_decline_upper),
        "exp1_scaled_kappa_max": float(np.nanmax(scaled_kappa)) if scaled_kappa.size else float("nan"),
        "exp2_R_min": float(np.min(r_vals)),
        "exp2_R_max": float(np.max(r_vals)),
        "exp2_R_range": float(np.max(r_vals) - np.min(r_vals)),
        "exp2_R_theory_lower": r_lower,
        "exp2_R_theory_upper": r_upper,
        "exp2_window_last": float(win_vals[-1]),
        "exp3_xi_crit": xi_crit,
        "exp3_xi_expected": xi_expected,
        "exp3_gt_best_last": float(gt_best_last),
        "exp3_lt_max_last": float(lt_max_last),
        "exp1_p_max": float(pg1[-1]),
    }
    notes = {
        "exp1_tail_rule": "tail uses M/sqrt(p_g), requires clear decline in larger p_g and terminal level clearly below peak.",
        "exp2_scale_rule": "R should remain finite and non-degenerate; scaled-window mass should be high.",
        "exp3_phase_rule": "xi_crit computed from theorem formula; above-threshold curves should rise, below-threshold should not approach 1.",
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
