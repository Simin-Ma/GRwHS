from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation_project.src.experiments.evaluation import _evaluate_row
from simulation_project.src.experiments.fitting import _fit_all_methods
from simulation_project.src.utils import SamplerConfig, canonical_groups, sample_correlated_design


OUT_DIR = ROOT / "outputs" / "grrhs_classic_candidates_paper"
PAPER_DIR = OUT_DIR / "paper_tables"
METHODS = ["GR_RHS", "RHS", "GHS_plus", "LASSO_CV", "GIGG_MMLE", "OLS"]
METHOD_LABEL = {
    "GR_RHS": "GR-RHS",
    "RHS": "RHS",
    "GHS_plus": "GHS+",
    "LASSO_CV": "Lasso-CV",
    "GIGG_MMLE": "GIGG-MMLE",
    "OLS": "OLS",
}


def capped_sparse_dense(size: int, head: float, plateau: float, tail: float, n_head: int, n_plateau: int) -> np.ndarray:
    vals = np.full(size, tail, dtype=float)
    vals[:n_plateau] = plateau
    vals[:n_head] = head
    return vals


def smooth_cap(size: int, low: float, high: float, power: float = 1.25) -> np.ndarray:
    x = np.linspace(0.0, 1.0, size)
    return low + (high - low) * (1.0 - x**power)


def alternating_tier(size: int, a: float, b: float, c: float) -> np.ndarray:
    seq = np.array([a, b, c, -c, -b], dtype=float)
    vals = np.resize(seq, size)
    vals *= np.linspace(1.0, 0.75, size)
    return vals


def beta_from_group_patterns(group_sizes: list[int], patterns: dict[int, np.ndarray]) -> np.ndarray:
    beta = np.zeros(sum(group_sizes), dtype=float)
    start = 0
    for gid, size in enumerate(group_sizes):
        vals = np.zeros(size, dtype=float)
        raw = np.asarray(patterns.get(gid, []), dtype=float)
        if raw.size:
            vals[: min(size, raw.size)] = raw[: min(size, raw.size)]
        beta[start : start + size] = vals
        start += size
    return beta


def simulate_dataset(
    *,
    n: int,
    group_sizes: list[int],
    beta: np.ndarray,
    rho_within: float,
    rho_between: float,
    seed: int,
    target_r2: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, Sigma = sample_correlated_design(
        n=n,
        group_sizes=group_sizes,
        rho_within=rho_within,
        rho_between=rho_between,
        seed=seed,
    )
    signal_var = float(beta @ Sigma @ beta)
    sigma2 = signal_var * (1.0 - float(target_r2)) / max(float(target_r2), 1e-12)
    rng = np.random.default_rng(seed + 17)
    y_train = X_train @ beta + rng.normal(0.0, np.sqrt(sigma2), size=n)

    X_test, _ = sample_correlated_design(
        n=n,
        group_sizes=group_sizes,
        rho_within=rho_within,
        rho_between=rho_between,
        seed=seed + 77777,
    )
    rng_test = np.random.default_rng(seed + 88888)
    y_test = X_test @ beta + rng_test.normal(0.0, np.sqrt(sigma2), size=n)
    return X_train, y_train, X_test, y_test, Sigma, np.asarray([sigma2], dtype=float)


def setting_label(row: pd.Series) -> str:
    return (
        f"{row['candidate_label']} | "
        f"n={int(row['n_train'])}, "
        f"rw={float(row['rho_within']):.1f}, "
        f"rb={float(row['rho_between']):.1f}, "
        f"R2={float(row['target_r2']):.1f}"
    )


def fmt(x: float, nd: int = 3) -> str:
    if pd.isna(x):
        return "--"
    return f"{float(x):.{nd}f}"


def build_main_table(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["candidate_id", "candidate_label", "n_train", "rho_within", "rho_between", "target_r2"]
    rows: list[dict[str, Any]] = []
    for _, sub in df.groupby(keys, sort=False):
        sub = sub.sort_values(["rank_mse_overall", "rank_mse_signal", "runtime_mean"], kind="stable")
        winner = sub.iloc[0]
        runner_up = sub.iloc[1]
        gigg = sub.loc[sub["method"] == "GIGG_MMLE"].iloc[0]
        rows.append(
            {
                "setting": setting_label(winner),
                "winner": METHOD_LABEL[str(winner["method"])],
                "winner_mse_overall": float(winner["mse_overall"]),
                "winner_mse_signal": float(winner["mse_signal"]),
                "winner_coverage_95": float(winner["coverage_95"]),
                "winner_runtime_s": float(winner["runtime_mean"]),
                "runner_up": METHOD_LABEL[str(runner_up["method"])],
                "runner_up_mse_overall": float(runner_up["mse_overall"]),
                "runner_up_mse_signal": float(runner_up["mse_signal"]),
                "delta_overall_vs_runner_up": float(runner_up["mse_overall"]) - float(winner["mse_overall"]),
                "delta_signal_vs_runner_up": float(runner_up["mse_signal"]) - float(winner["mse_signal"]),
                "gigg_mse_overall": float(gigg["mse_overall"]),
                "gigg_mse_signal": float(gigg["mse_signal"]),
                "gigg_coverage_95": float(gigg["coverage_95"]),
                "gigg_runtime_s": float(gigg["runtime_mean"]),
                "gigg_over_gr": float(gigg["mse_overall"]) / max(float(sub.loc[sub["method"] == "GR_RHS", "mse_overall"].iloc[0]), 1e-12),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["winner_mse_overall", "delta_overall_vs_runner_up"], ascending=[True, False]).reset_index(drop=True)


def build_appendix_table(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["setting"] = work.apply(setting_label, axis=1)
    work["method_label"] = work["method"].map(METHOD_LABEL)
    bayes = {"GR_RHS", "RHS", "GHS_plus", "GIGG_MMLE"}
    work["bayes_conv"] = work.apply(
        lambda r: "yes" if str(r["method"]) in bayes and int(r["n_converged"]) == int(r["n_ok"]) == int(r["n_runs"]) else ("n/a" if str(r["method"]) in {"OLS", "LASSO_CV"} else "no"),
        axis=1,
    )
    cols = [
        "setting",
        "method_label",
        "n_runs",
        "n_ok",
        "n_converged",
        "rank_mse_overall",
        "rank_mse_signal",
        "mse_overall",
        "mse_signal",
        "mse_null",
        "coverage_95",
        "runtime_mean",
        "bayes_conv",
    ]
    return work.loc[:, cols].sort_values(["setting", "rank_mse_overall", "runtime_mean"], kind="stable").reset_index(drop=True)


def write_markdown_main(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "| Setting | Winner | MSE Overall | MSE Signal | Coverage | Runtime (s) | Runner-up | Delta Overall | Delta Signal | GIGG Overall | GIGG Signal | GIGG/GR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["setting"]),
                    str(r["winner"]),
                    fmt(r["winner_mse_overall"]),
                    fmt(r["winner_mse_signal"]),
                    fmt(r["winner_coverage_95"], 2),
                    fmt(r["winner_runtime_s"], 2),
                    str(r["runner_up"]),
                    fmt(r["delta_overall_vs_runner_up"]),
                    fmt(r["delta_signal_vs_runner_up"]),
                    fmt(r["gigg_mse_overall"]),
                    fmt(r["gigg_mse_signal"]),
                    fmt(r["gigg_over_gr"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_appendix(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "| Setting | Method | Runs | Ok | Converged | Rank Overall | Rank Signal | MSE Overall | MSE Signal | MSE Null | Coverage | Runtime (s) | Bayesian Converged |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["setting"]),
                    str(r["method_label"]),
                    str(int(r["n_runs"])),
                    str(int(r["n_ok"])),
                    str(int(r["n_converged"])),
                    str(int(r["rank_mse_overall"])),
                    str(int(r["rank_mse_signal"])),
                    fmt(r["mse_overall"]),
                    fmt(r["mse_signal"]),
                    fmt(r["mse_null"]),
                    fmt(r["coverage_95"], 2),
                    fmt(r["runtime_mean"], 2),
                    str(r["bayes_conv"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_main(df: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{lccccccccccc}",
        r"\toprule",
        r"Setting & Winner & Overall & Signal & Cov. & Time & Runner-up & $\Delta$ Overall & $\Delta$ Signal & GIGG Overall & GIGG Signal & GIGG/GR \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['setting']} & {r['winner']} & {fmt(r['winner_mse_overall'])} & {fmt(r['winner_mse_signal'])} & "
            f"{fmt(r['winner_coverage_95'], 2)} & {fmt(r['winner_runtime_s'], 2)} & {r['runner_up']} & "
            f"{fmt(r['delta_overall_vs_runner_up'])} & {fmt(r['delta_signal_vs_runner_up'])} & "
            f"{fmt(r['gigg_mse_overall'])} & {fmt(r['gigg_mse_signal'])} & {fmt(r['gigg_over_gr'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_appendix(df: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{longtable}{llrrrrrcccccl}",
        r"\toprule",
        r"Setting & Method & Runs & Ok & Conv. & Rank O. & Rank S. & Overall & Signal & Null & Cov. & Time & Conv. \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Setting & Method & Runs & Ok & Conv. & Rank O. & Rank S. & Overall & Signal & Null & Cov. & Time & Conv. \\",
        r"\midrule",
        r"\endhead",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['setting']} & {r['method_label']} & {int(r['n_runs'])} & {int(r['n_ok'])} & {int(r['n_converged'])} & "
            f"{int(r['rank_mse_overall'])} & {int(r['rank_mse_signal'])} & {fmt(r['mse_overall'])} & "
            f"{fmt(r['mse_signal'])} & {fmt(r['mse_null'])} & {fmt(r['coverage_95'], 2)} & {fmt(r['runtime_mean'], 2)} & {r['bayes_conv']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    sampler = SamplerConfig(
        chains=2,
        warmup=250,
        post_warmup_draws=250,
        adapt_delta=0.9,
        max_treedepth=12,
        strict_adapt_delta=0.95,
        strict_max_treedepth=14,
        max_divergence_ratio=0.01,
        rhat_threshold=1.01,
        ess_threshold=200.0,
    )

    candidates = [
        {
            "candidate_id": "classic_multimode_2act_n500_rw08_rb02",
            "candidate_label": "Classic Multimode 2 Active Groups",
            "group_sizes": [10, 10, 10, 10, 10],
            "n_train": 500,
            "n_test": 500,
            "rho_within": 0.8,
            "rho_between": 0.2,
            "target_r2": 0.7,
            "patterns": {
                0: capped_sparse_dense(10, 0.42, 0.24, 0.12, 1, 4),
                1: alternating_tier(10, 0.20, 0.12, 0.06),
            },
        },
        {
            "candidate_id": "classic_multimode_3act_n500_rw06_rb02",
            "candidate_label": "Classic Multimode 3 Active Groups",
            "group_sizes": [10, 10, 10, 10, 10],
            "n_train": 500,
            "n_test": 500,
            "rho_within": 0.6,
            "rho_between": 0.2,
            "target_r2": 0.7,
            "patterns": {
                0: capped_sparse_dense(10, 0.40, 0.24, 0.12, 1, 4),
                1: smooth_cap(10, 0.10, 0.28),
                2: alternating_tier(10, 0.18, 0.12, 0.06),
            },
        },
    ]

    raw_rows: list[dict[str, Any]] = []
    for c_idx, cfg in enumerate(candidates, start=1):
        groups = canonical_groups(cfg["group_sizes"])
        beta = beta_from_group_patterns(cfg["group_sizes"], cfg["patterns"])
        p0 = int(np.sum(np.abs(beta) > 1e-12))
        for rep in range(4):
            seed = 20260425 + c_idx * 100 + rep
            X_train, y_train, X_test, y_test, _, sigma2_arr = simulate_dataset(
                n=int(cfg["n_train"]),
                group_sizes=list(cfg["group_sizes"]),
                beta=beta,
                rho_within=float(cfg["rho_within"]),
                rho_between=float(cfg["rho_between"]),
                seed=seed,
                target_r2=float(cfg["target_r2"]),
            )
            fits = _fit_all_methods(
                X_train,
                y_train,
                groups,
                task="gaussian",
                seed=seed,
                p0=p0,
                sampler=sampler,
                methods=METHODS,
                grrhs_kwargs={"sampler_backend": "gibbs_staged", "tau_target": "groups", "progress_bar": False},
                gigg_config={"allow_budget_retry": True, "extra_retry": 0, "no_retry": True},
                bayes_min_chains=2,
                enforce_bayes_convergence=True,
                max_convergence_retries=-1,
                method_jobs=1,
            )
            for method, res in fits.items():
                metrics = _evaluate_row(
                    res,
                    beta,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                )
                raw_rows.append(
                    {
                        "candidate_id": cfg["candidate_id"],
                        "candidate_label": cfg["candidate_label"],
                        "replicate_id": int(rep),
                        "method": method,
                        "status": str(res.status),
                        "converged": bool(res.converged) if method in {"GR_RHS", "RHS", "GHS_plus", "GIGG_MMLE"} else True,
                        "runtime_seconds": float(res.runtime_seconds),
                        "n_train": int(cfg["n_train"]),
                        "n_test": int(cfg["n_test"]),
                        "rho_within": float(cfg["rho_within"]),
                        "rho_between": float(cfg["rho_between"]),
                        "target_r2": float(cfg["target_r2"]),
                        "target_snr": float(cfg["target_r2"] / max(1.0 - float(cfg["target_r2"]), 1e-12)),
                        "sigma2": float(sigma2_arr[0]),
                        **metrics,
                    }
                )

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(OUT_DIR / "raw_results.csv", index=False)

    summary = (
        raw_df.groupby(
            ["candidate_id", "candidate_label", "n_train", "rho_within", "rho_between", "target_r2", "target_snr", "method"],
            as_index=False,
        )
        .agg(
            n_runs=("method", "size"),
            n_ok=("status", lambda s: int(np.sum(pd.Series(s).astype(str) == "ok"))),
            n_converged=("converged", lambda s: int(np.sum(pd.Series(s).astype(bool)))),
            mse_overall=("mse_overall", "mean"),
            mse_signal=("mse_signal", "mean"),
            mse_null=("mse_null", "mean"),
            coverage_95=("coverage_95", "mean"),
            avg_ci_length=("avg_ci_length", "mean"),
            lpd_test=("lpd_test", "mean"),
            runtime_mean=("runtime_seconds", "mean"),
        )
    )
    summary["method"] = pd.Categorical(summary["method"], categories=METHODS, ordered=True)
    summary = summary.sort_values(["candidate_id", "method"], kind="stable").reset_index(drop=True)
    summary["rank_mse_overall"] = summary.groupby("candidate_id")["mse_overall"].rank(method="first", ascending=True)
    summary["rank_mse_signal"] = summary.groupby("candidate_id")["mse_signal"].rank(method="first", ascending=True)
    summary.to_csv(OUT_DIR / "summary_all_methods.csv", index=False)

    main_df = build_main_table(summary)
    appendix_df = build_appendix_table(summary)
    main_df.to_csv(PAPER_DIR / "paper_table_main.csv", index=False)
    appendix_df.to_csv(PAPER_DIR / "paper_table_appendix_full.csv", index=False)
    write_markdown_main(main_df, PAPER_DIR / "paper_table_main.md")
    write_markdown_appendix(appendix_df, PAPER_DIR / "paper_table_appendix_full.md")
    write_latex_main(main_df, PAPER_DIR / "paper_table_main.tex")
    write_latex_appendix(appendix_df, PAPER_DIR / "paper_table_appendix_full.tex")

    manifest_candidates: list[dict[str, Any]] = []
    for cfg in candidates:
        cfg_out = dict(cfg)
        cfg_out["patterns"] = {
            str(k): np.asarray(v, dtype=float).round(6).tolist()
            for k, v in dict(cfg["patterns"]).items()
        }
        manifest_candidates.append(cfg_out)

    manifest = {
        "candidates": manifest_candidates,
        "methods": METHODS,
        "bayes_convergence_required": True,
        "max_convergence_retries": -1,
        "n_repeats": 4,
        "outputs": {
            "raw_results": str(OUT_DIR / "raw_results.csv"),
            "summary_all_methods": str(OUT_DIR / "summary_all_methods.csv"),
            "paper_main_md": str(PAPER_DIR / "paper_table_main.md"),
            "paper_appendix_md": str(PAPER_DIR / "paper_table_appendix_full.md"),
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
