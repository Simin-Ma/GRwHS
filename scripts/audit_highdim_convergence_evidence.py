from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_PARAMS = {
    "GR_RHS": ["beta", "tau", "kappa"],
    "RHS": ["beta", "tau", "lambda", "c"],
    "GIGG_MMLE": ["beta", "gamma2", "lambda", "tau", "sigma"],
    "GHS_plus_NUTS": ["beta", "tau", "group_scale", "lambda", "sigma"],
}

JUDGEMENT_ORDER = {
    "PASS_STRONG": 5,
    "PASS_MARGINAL": 4,
    "PARTIAL": 3,
    "FAIL": 2,
    "INSUFFICIENT": 1,
}

REQUIRED_PARAMS = {
    "GR_RHS": ["beta", "tau", "kappa"],
    "RHS": ["beta", "tau", "lambda"],
    "GIGG_MMLE": ["beta", "gamma2"],
    "GHS_plus_NUTS": ["beta", "tau", "group_scale", "lambda", "sigma"],
}


def _num(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def _dig(mapping: dict[str, Any], *keys: str) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _param_table(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    diag = payload.get("diagnostics", {})
    out: dict[str, dict[str, float]] = {}
    candidates = [
        diag.get("convergence_detail"),
        diag.get("param_diagnostics"),
    ]
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        for name, item in cand.items():
            if not isinstance(item, dict):
                continue
            rhat = _num(item.get("rhat_max"))
            ess = _num(item.get("ess_min"))
            if math.isfinite(rhat) or math.isfinite(ess):
                out[str(name)] = {"rhat_max": rhat, "ess_min": ess}
    return out


def _hmc_table(payload: dict[str, Any]) -> dict[str, Any]:
    diag = payload.get("diagnostics", {})
    hmc = _dig(diag, "sampler_diagnostics", "hmc")
    if not isinstance(hmc, dict):
        return {}
    return {
        "divergences": hmc.get("divergences"),
        "treedepth_hits": hmc.get("treedepth_hits"),
        "ebfmi_min": hmc.get("ebfmi_min"),
    }


def _judge(method: str, params: dict[str, dict[str, float]], payload: dict[str, Any], *, rhat_thr: float, ess_thr: float) -> tuple[str, str]:
    expected = EXPECTED_PARAMS.get(method, ["beta"])
    required = REQUIRED_PARAMS.get(method, ["beta"])
    missing_required = [p for p in required if p not in params]
    missing_expected = [p for p in expected if p not in params]
    observed = list(params)
    failing = []
    marginal = []
    for name, vals in params.items():
        rhat = vals.get("rhat_max", math.nan)
        ess = vals.get("ess_min", math.nan)
        if math.isfinite(rhat) and rhat > rhat_thr:
            failing.append(f"{name}:rhat={rhat:.6g}")
        if math.isfinite(ess) and ess < ess_thr:
            failing.append(f"{name}:ess={ess:.6g}")
        if math.isfinite(rhat) and rhat > (rhat_thr - 0.002):
            marginal.append(f"{name}:rhat={rhat:.6g}")
        if math.isfinite(ess) and ess < max(ess_thr * 1.25, ess_thr + 100):
            marginal.append(f"{name}:ess={ess:.6g}")

    reported_ess = _num(payload.get("ess_min"))
    reported_rhat = _num(payload.get("rhat_max"))
    if math.isfinite(reported_rhat) and reported_rhat > rhat_thr:
        failing.append(f"reported:rhat={reported_rhat:.6g}")
    if math.isfinite(reported_ess) and reported_ess < ess_thr:
        failing.append(f"reported:ess={reported_ess:.6g}")

    hmc = _hmc_table(payload)
    div = _num(hmc.get("divergences")) if hmc else math.nan
    td = _num(hmc.get("treedepth_hits")) if hmc else math.nan
    ebfmi = _num(hmc.get("ebfmi_min")) if hmc else math.nan
    if math.isfinite(div) and div > 0:
        failing.append(f"divergences={div:.0f}")
    if math.isfinite(td) and td > 0:
        failing.append(f"treedepth_hits={td:.0f}")
    if math.isfinite(ebfmi) and ebfmi < 0.3:
        failing.append(f"ebfmi_min={ebfmi:.3g}")

    if missing_required:
        return "INSUFFICIENT", f"missing required posterior diagnostics: {','.join(missing_required)}; observed={','.join(observed)}"
    if failing:
        return "FAIL", "; ".join(failing)
    if missing_expected:
        return "PARTIAL", f"required diagnostics pass, but missing expected nuisance diagnostics: {','.join(missing_expected)}"
    if marginal:
        return "PASS_MARGINAL", "; ".join(marginal)
    return "PASS_STRONG", "all recorded posterior diagnostics pass"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit high-dimensional posterior convergence evidence.")
    parser.add_argument("--raw", default="tmp/highdim_bayes_rerun_20260512_full/rerun_raw.csv")
    parser.add_argument("--outdir", default="tmp/highdim_bayes_rerun_20260512_full")
    parser.add_argument("--extra-json-dir", action="append", default=[])
    parser.add_argument("--rhat-threshold", type=float, default=1.01)
    parser.add_argument("--ess-threshold", type=float, default=400.0)
    args = parser.parse_args()

    raw = pd.read_csv(args.raw)
    items: list[dict[str, Any]] = raw.to_dict(orient="records")
    for folder in args.extra_json_dir or []:
        for path in sorted(Path(folder).glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
            except Exception:
                continue
            if not isinstance(payload, dict) or "method" not in payload:
                continue
            items.append(
                {
                    "setting_id": payload.get("setting_id", ""),
                    "method": payload.get("method", ""),
                    "replicate": payload.get("replicate", 0),
                    "converged": payload.get("converged", False),
                    "source_file": str(path),
                }
            )

    rows: list[dict[str, Any]] = []
    for item in items:
        path = Path(str(item["source_file"]))
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        method = str(item["method"])
        params = _param_table(payload)
        judgement, reason = _judge(
            method,
            params,
            payload,
            rhat_thr=float(args.rhat_threshold),
            ess_thr=float(args.ess_threshold),
        )
        hmc = _hmc_table(payload)
        rows.append(
            {
                "setting_id": str(item["setting_id"]),
                "method": method,
                "replicate": int(item["replicate"]),
                "json_converged_flag": bool(item["converged"]),
                "judgement": judgement,
                "reason": reason,
                "recorded_params": ",".join(sorted(params)),
                "missing_expected_params": ",".join([p for p in EXPECTED_PARAMS.get(method, []) if p not in params]),
                "max_param_rhat": max([v["rhat_max"] for v in params.values() if math.isfinite(v["rhat_max"])], default=math.nan),
                "min_param_ess": min([v["ess_min"] for v in params.values() if math.isfinite(v["ess_min"])], default=math.nan),
                "reported_rhat": _num(payload.get("rhat_max")),
                "reported_ess": _num(payload.get("ess_min")),
                "divergences": hmc.get("divergences"),
                "treedepth_hits": hmc.get("treedepth_hits"),
                "ebfmi_min": hmc.get("ebfmi_min"),
                "source_file": str(path),
            }
        )

    out = pd.DataFrame(rows).sort_values(["setting_id", "method", "replicate"])
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir / "posterior_convergence_evidence.csv", index=False)

    best = out.copy()
    best["_score"] = best["judgement"].map(JUDGEMENT_ORDER).fillna(0).astype(int)
    best = best.sort_values(
        ["setting_id", "method", "replicate", "_score", "min_param_ess"],
        ascending=[True, True, True, False, False],
    )
    best = best.groupby(["setting_id", "method", "replicate"], as_index=False, sort=False).head(1)
    best = best.drop(columns=["_score"])
    best.to_csv(outdir / "posterior_convergence_evidence_best.csv", index=False)

    summary = (
        best.groupby(["setting_id", "method", "judgement"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["setting_id", "method", "judgement"])
    )
    summary.to_csv(outdir / "posterior_convergence_evidence_summary.csv", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
