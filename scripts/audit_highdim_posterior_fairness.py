from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


POSTERIOR_FAMILY = {
    "GR_RHS": "gr_rhs_regularized_grouped_horseshoe",
    "RHS": "regularized_horseshoe",
    "GIGG_MMLE": "gigg_empirical_bayes_mmle",
    "GHS_plus_NUTS": "grouped_horseshoe_plus",
}

POSTERIOR_DIAGNOSTIC_SCOPE = {
    "GR_RHS": ["beta", "tau", "kappa"],
    "RHS": ["beta", "tau", "lambda", "c"],
    "GIGG_MMLE": ["beta", "gamma2", "lambda", "tau", "sigma"],
    "GHS_plus_NUTS": ["beta", "tau", "group_scale", "lambda", "sigma"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit whether high-dimensional Bayesian model comparisons are posterior-valid.")
    parser.add_argument("--raw", default="tmp/highdim_bayes_rerun_20260512_full/rerun_raw.csv")
    parser.add_argument("--out", default="tmp/highdim_bayes_rerun_20260512_full/posterior_fairness_audit.json")
    args = parser.parse_args()

    raw_path = Path(args.raw)
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    df = pd.read_csv(raw_path)
    if "method" not in df.columns or "setting_id" not in df.columns:
        raise ValueError("raw table must contain setting_id and method columns")
    df["posterior_family"] = df["method"].map(POSTERIOR_FAMILY).fillna("unknown")
    if "converged" in df.columns:
        converged = df["converged"].astype(str).str.lower().isin(["true", "1", "yes"])
    elif "judgement" in df.columns:
        converged = df["judgement"].astype(str).isin(["PASS_STRONG", "PASS_MARGINAL"])
    else:
        converged = pd.Series(False, index=df.index)
    df["_posterior_converged"] = converged

    groups = []
    for setting, g in df.groupby("setting_id"):
        families = {
            str(fam): sorted(g.loc[g["posterior_family"] == fam, "method"].astype(str).unique().tolist())
            for fam in sorted(g["posterior_family"].astype(str).unique())
        }
        groups.append(
            {
                "setting_id": str(setting),
                "posterior_families": families,
                "n_methods": int(g["method"].nunique()),
                "n_runs": int(g.shape[0]),
                "n_converged_runs": int(g["_posterior_converged"].sum()),
                "all_runs_converged": bool(g["_posterior_converged"].all()),
            }
        )

    audit = {
        "verdict": "valid_model_comparison_if_each_model_posterior_converged",
        "reason": (
            "The methods target different posterior families by design. This is valid as a model-level "
            "comparison on the same simulated datasets, provided each method samples its own model posterior "
            "to convergence before metrics are ranked."
        ),
        "posterior_family_by_method": POSTERIOR_FAMILY,
        "posterior_diagnostic_scope_by_method": POSTERIOR_DIAGNOSTIC_SCOPE,
        "settings": groups,
        "allowed_model_comparison_rule": (
            "Rank methods within each setting only after every method has 5/5 converged runs under its own "
            "posterior diagnostics. Report posterior_family so the ranking is not mistaken for a same-posterior "
            "sampler comparison."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
