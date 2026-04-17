from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _records(df: Any) -> list[dict[str, Any]]:
    if isinstance(df, list):
        return [dict(r) for r in df]
    if hasattr(df, "to_dict"):
        try:
            return list(df.to_dict(orient="records"))
        except TypeError:
            pass
    return [dict(r) for r in df]


def _as_frame(df: Any):
    import pandas as pd

    if hasattr(df, "groupby"):
        return df
    return pd.DataFrame(_records(df))


def plot_exp1(df: Any, slope: float, slope_ci: tuple[float, float], out_path: Path) -> None:
    rows = sorted(_records(df), key=lambda r: float(r["p_g"]))
    if not rows:
        return
    p_g = np.asarray([float(r["p_g"]) for r in rows], dtype=float)
    med = np.asarray([float(r["median_post_mean_kappa"]) for r in rows], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.log(p_g)
    y = np.log(np.maximum(med, 1e-12))
    axes[0].plot(x, y, "o-")
    coef = np.polyfit(x, y, deg=1)
    axes[0].plot(x, coef[0] * x + coef[1], "--", color="black")
    axes[0].set_xlabel("log p_g")
    axes[0].set_ylabel("log median E[kappa|Y]")
    axes[0].set_title(f"Slope={slope:.3f} [{slope_ci[0]:.3f},{slope_ci[1]:.3f}]")

    # Right panel: IQR band using q25/q75 if available, else tail probability
    tail_cols = sorted([c for c in rows[0].keys() if str(c).startswith("mean_tail_prob_eps_")])
    q25_col = "q25_post_mean_kappa" if "q25_post_mean_kappa" in rows[0] else None
    if q25_col is not None:
        q25 = np.asarray([float(r["q25_post_mean_kappa"]) for r in rows], dtype=float)
        q75 = np.asarray([float(r["q75_post_mean_kappa"]) for r in rows], dtype=float)
        axes[1].plot(p_g, med, "o-", label="median")
        axes[1].fill_between(p_g, q25, q75, alpha=0.25, label="IQR")
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("p_g")
        axes[1].set_ylabel("E[kappa_g | Y]")
        axes[1].legend(fontsize=8)
    elif tail_cols:
        for c in tail_cols:
            label = str(c).replace("mean_tail_prob_eps_", "eps=").replace("_", ".")
            tail = np.asarray([float(r[c]) for r in rows], dtype=float)
            axes[1].plot(p_g, tail, "o-", label=label)
        axes[1].legend(fontsize=8)
        axes[1].set_xlabel("p_g")
        axes[1].set_ylabel("Mean P(kappa>eps|Y)")
    else:
        tail_key = "mean_tail_prob" if "mean_tail_prob" in rows[0] else list(rows[0].keys())[-1]
        tail = np.asarray([float(r.get(tail_key, float("nan"))) for r in rows], dtype=float)
        axes[1].plot(p_g, tail, "o-")
        axes[1].set_xlabel("p_g")
        axes[1].set_ylabel(tail_key)
    _save(fig, out_path)


def plot_exp1_phase(df: Any, out_path: Path) -> None:
    rows = _records(df)
    if not rows:
        return
    tau_vals = sorted({float(r["tau"]) for r in rows})
    pg_vals = sorted({int(r["p_g"]) for r in rows})
    ncols = min(len(tau_vals), 4)
    nrows = int(np.ceil(len(tau_vals) / max(ncols, 1)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.5 * nrows), squeeze=False)
    cmap = plt.cm.get_cmap("tab10", len(pg_vals))
    for i, tau in enumerate(tau_vals):
        ax = axes[i // ncols][i % ncols]
        sub_tau = [r for r in rows if float(r["tau"]) == tau]
        for j, pg in enumerate(pg_vals):
            sub = sorted((r for r in sub_tau if int(r["p_g"]) == pg), key=lambda r: float(r["xi_ratio"]))
            if not sub:
                continue
            xs = [float(r["xi_ratio"]) for r in sub]
            ys = [float(r["mean_prob_kappa_gt_u0"]) for r in sub]
            ax.plot(xs, ys, "o-", color=cmap(j), label=f"p_g={pg}")
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0, label="xi_crit")
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(f"tau={tau:g}")
        ax.set_xlabel("xi / xi_crit")
        ax.set_ylabel("P(kappa_g > u0 | Y)")
        ax.legend(fontsize=6, ncol=2)
    for j in range(len(tau_vals), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    _save(fig, out_path)


def plot_exp2(df: Any, out_path: Path) -> None:
    rows = _records(df)
    if not rows:
        return
    tau_vals = sorted({float(r["tau_eval"]) for r in rows if "tau_eval" in r})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    if not tau_vals:
        tau_vals = [float("nan")]
    for tau in tau_vals:
        sub = sorted(
            (r for r in rows if ("tau_eval" not in r) or (float(r["tau_eval"]) == tau)),
            key=lambda r: float(r["p_g"]),
        )
        if not sub:
            continue
        x = np.asarray([float(r["p_g"]) for r in sub], dtype=float)
        m = np.asarray([float(r["median_ratio_R"]) for r in sub], dtype=float)
        lo = np.asarray([float(r["iqr_ratio_R_low"]) for r in sub], dtype=float)
        hi = np.asarray([float(r["iqr_ratio_R_high"]) for r in sub], dtype=float)
        label = f"tau={tau:g}" if np.isfinite(tau) else "combined"
        axes[0].plot(x, m, "o-", label=label)
        axes[0].fill_between(x, lo, hi, alpha=0.15)
        axes[1].plot(x, np.asarray([float(r["mean_window_prob"]) for r in sub], dtype=float), "o-", label=label)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("p_g")
    axes[0].set_ylabel("Median R = E[kappa]/s_g")
    axes[0].legend(fontsize=8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("p_g")
    axes[1].set_ylabel("Mean P(kappa in [x_lo s_g, x_hi s_g])")
    axes[1].legend(fontsize=8)
    _save(fig, out_path)


def plot_exp3_heatmap(df: Any, out_path: Path) -> None:
    rows = _records(df)
    if not rows:
        return
    x_key = "xi_ratio" if "xi_ratio" in rows[0] else "xi"
    p_vals = sorted({float(r["p_g"]) for r in rows})
    x_vals = sorted({float(r[x_key]) for r in rows})
    mat = np.full((len(p_vals), len(x_vals)), np.nan, dtype=float)
    p_idx = {v: i for i, v in enumerate(p_vals)}
    x_idx = {v: i for i, v in enumerate(x_vals)}
    for r in rows:
        mat[p_idx[float(r["p_g"])], x_idx[float(r[x_key])]] = float(r["mean_prob_gt_u0"])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(np.arange(len(x_vals)), labels=[f"{v:.2g}" for v in x_vals])
    ax.set_yticks(np.arange(len(p_vals)), labels=[str(int(v)) for v in p_vals])
    ax.set_xlabel("xi/xi_crit" if x_key == "xi_ratio" else "xi")
    ax.set_ylabel("p_g")
    fig.colorbar(im, ax=ax, label="Mean P(kappa>u0|Y)")
    _save(fig, out_path)


def plot_exp3_curves(df: Any, xi_crit: float, out_path: Path) -> None:
    rows = _records(df)
    if not rows:
        return
    x_key = "xi_ratio" if "xi_ratio" in rows[0] else "xi"
    p_vals = sorted({float(r["p_g"]) for r in rows})
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for pg in p_vals:
        sub = sorted((r for r in rows if float(r["p_g"]) == pg), key=lambda r: float(r[x_key]))
        ax.plot([float(r[x_key]) for r in sub], [float(r["mean_prob_gt_u0"]) for r in sub], marker="o", label=f"p_g={int(pg)}")
    ax.axvline(float(xi_crit), color="black", linestyle="--", label="threshold")
    if x_key != "xi_ratio":
        ax.set_xscale("log")
    ax.set_xlabel("xi/xi_crit" if x_key == "xi_ratio" else "xi")
    ax.set_ylabel("Mean P(kappa>u0|Y)")
    ax.legend(fontsize=8)
    _save(fig, out_path)


def plot_exp3_tau_sweep(df: Any, out_path: Path) -> None:
    rows = _records(df)
    if not rows:
        return
    tau_vals = sorted({float(r["tau"]) for r in rows})
    x_key = "xi_ratio" if "xi_ratio" in rows[0] else "xi"
    n = len(tau_vals)
    ncols = min(3, n)
    nrows = int(np.ceil(n / max(ncols, 1)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.3 * nrows), squeeze=False)
    for i, tau in enumerate(tau_vals):
        ax = axes[i // ncols][i % ncols]
        sub_tau = [r for r in rows if float(r["tau"]) == tau]
        for pg in sorted({int(r["p_g"]) for r in sub_tau}):
            sub = sorted((r for r in sub_tau if int(r["p_g"]) == pg), key=lambda r: float(r[x_key]))
            ax.plot([float(r[x_key]) for r in sub], [float(r["mean_prob_gt_u0"]) for r in sub], marker="o", label=f"p_g={pg}")
        if x_key == "xi_ratio":
            ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0)
        if x_key != "xi_ratio":
            ax.set_xscale("log")
        ax.set_title(f"tau={tau:g}")
        ax.set_xlabel("xi/xi_crit" if x_key == "xi_ratio" else "xi")
        ax.set_ylabel("P(kappa>u0|Y)")
        ax.legend(fontsize=7)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    _save(fig, out_path)


def plot_exp4_overall_mse(df: Any, out_path: Path) -> None:
    frame = _as_frame(df)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for method, sub in frame.groupby("method"):
        s = sub.sort_values("setting")
        ax.plot(s["setting"], s["mse_overall"], marker="o", label=method)
    ax.set_xlabel("Setting")
    ax.set_ylabel("Overall MSE")
    ax.legend()
    _save(fig, out_path)


def plot_exp4_mse_partition(df: Any, out_path: Path) -> None:
    frame = _as_frame(df)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    metrics = [("mse_null", "Null MSE"), ("mse_signal", "Signal MSE"), ("mse_overall", "Overall MSE")]
    for ax, (key, label) in zip(axes, metrics):
        for method, sub in frame.groupby("method"):
            s = sub.sort_values("setting")
            ax.plot(s["setting"], s[key], marker="o", label=method)
        ax.set_xlabel("Setting")
        ax.set_ylabel(label)
    axes[-1].legend(fontsize=8)
    _save(fig, out_path)


def plot_exp5_kappa_stratification(df: Any, out_path: Path) -> None:
    frame = _as_frame(df)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    groups = sorted(frame["group_id"].unique())
    data = [frame.loc[frame["group_id"] == g, "post_mean_kappa_g"].dropna().to_numpy() for g in groups]
    ax.boxplot(data, positions=np.arange(1, len(groups) + 1))
    ax.set_xticks(np.arange(1, len(groups) + 1), labels=[str(int(g) + 1) for g in groups])
    ax.set_xlabel("Group")
    ax.set_ylabel("Posterior mean kappa_g")
    _save(fig, out_path)


def plot_exp5_null_signal_mse(df: Any, out_path: Path) -> None:
    frame = _as_frame(df)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    xpos = np.arange(len(frame))
    w = 0.35
    ax.bar(xpos - w / 2, frame["avg_null_group_mse"], width=w, label="null")
    ax.bar(xpos + w / 2, frame["avg_signal_group_mse"], width=w, label="signal")
    ax.set_xticks(xpos, labels=frame["method"].tolist())
    ax.set_ylabel("Group MSE")
    ax.legend()
    _save(fig, out_path)


def plot_exp5_group_ranking(df_kappa: Any, df_auroc: Any, out_path: Path) -> None:
    frame_k = _as_frame(df_kappa)
    frame_a = _as_frame(df_auroc)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    agg = frame_k.groupby("mu_g", as_index=False)["post_mean_kappa_g"].mean()
    axes[0].plot(agg["mu_g"], agg["post_mean_kappa_g"], "o-")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("mu_g")
    axes[0].set_ylabel("Avg E[kappa_g|Y]")
    axes[1].bar(frame_a["method"], frame_a["group_auroc"])
    axes[1].set_ylabel("Group AUROC")
    _save(fig, out_path)


def plot_exp6_coefficients(df: Any, out_path: Path) -> None:
    frame = _as_frame(df)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for method, sub in frame.groupby("method"):
        ax.scatter(sub["beta11_post_mean"], sub["beta12_post_mean"], alpha=0.6, label=method)
    ax.set_xlabel("beta11 posterior mean")
    ax.set_ylabel("beta12 posterior mean")
    ax.legend()
    _save(fig, out_path)


def plot_exp6_null_group(df: Any, out_path: Path) -> None:
    frame = _as_frame(df)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    vals = [sub["beta_group2_l2_norm"].dropna().to_numpy() for _, sub in frame.groupby("method")]
    labels = [m for m, _ in frame.groupby("method")]
    ax.boxplot(vals, labels=labels)
    ax.set_ylabel("||beta_group2||_2")
    _save(fig, out_path)


def plot_exp6_diagnostics(df: Any, out_path: Path) -> None:
    frame = _as_frame(df)
    agg = frame.groupby("method", as_index=False)[["divergence_ratio", "overall_runtime"]].mean(numeric_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(agg["method"], agg["divergence_ratio"])
    axes[0].set_ylabel("Divergence ratio")
    axes[1].bar(agg["method"], agg["overall_runtime"])
    axes[1].set_ylabel("Runtime (s)")
    _save(fig, out_path)


def plot_exp6_kappa(df: Any, out_path: Path) -> None:
    frame = _as_frame(df)
    gr = frame.loc[frame["method"] == "GR_RHS"].copy()
    if gr.empty:
        return
    mean_cols = ["post_mean_kappa_group1", "post_mean_kappa_group2", "post_mean_kappa_group3"]
    prob_cols = ["post_prob_kappa_group1_gt_0_5", "post_prob_kappa_group2_gt_0_5", "post_prob_kappa_group3_gt_0_5"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    vals = [gr[c].dropna().to_numpy() for c in mean_cols if c in gr.columns]
    labels = [f"group{i+1}" for i in range(len(vals))]
    if vals:
        axes[0].boxplot(vals, labels=labels)
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Posterior mean kappa")
    if all(c in gr.columns for c in prob_cols):
        probs = [float(gr[c].mean()) for c in prob_cols]
        axes[1].bar(["group1", "group2", "group3"], probs)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("P(kappa>0.5)")
    _save(fig, out_path)


def plot_exp7_ablation_bars(df: Any, out_path: Path) -> None:
    frame = _as_frame(df)
    from .utils import method_display_name

    metrics = [
        ("null_group_mse_avg", "Null MSE"),
        ("signal_group_mse_avg", "Signal MSE"),
        ("overall_mse", "Overall MSE"),
        ("group_auroc", "Group AUROC"),
    ]
    dgp_types = sorted(frame["dgp_type"].dropna().unique().tolist())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (metric, label) in zip(axes.flatten(), metrics):
        for dgp in dgp_types:
            sub = frame.loc[frame["dgp_type"] == dgp].sort_values("variant")
            x = np.arange(len(sub)) + (0.18 if dgp == dgp_types[-1] else -0.18)
            ax.bar(x, sub[metric], width=0.35, alpha=0.8, label=dgp)
            ax.set_xticks(np.arange(len(sub)), labels=[method_display_name(v) for v in sub["variant"]], rotation=25, ha="right")
        ax.set_ylabel(label)
    axes[0, 0].legend(fontsize=8)
    _save(fig, out_path)


def plot_exp8_tau(df: Any, out_path: Path) -> None:
    frame = _as_frame(df)
    if {"tau_post_mean", "tau_target", "tau_mode"}.issubset(frame.columns):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for mode, sub in frame.groupby("tau_mode"):
            axes[0].scatter(sub["tau_target"], sub["tau_post_mean"], alpha=0.5, s=16, label=mode)
        lo = float(min(frame["tau_target"].min(), frame["tau_post_mean"].min()))
        hi = float(max(frame["tau_target"].max(), frame["tau_post_mean"].max()))
        axes[0].plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.0)
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("tau target")
        axes[0].set_ylabel("posterior mean tau")
        axes[0].legend(fontsize=8)

        if "tau_rel_error" in frame.columns:
            for mode, sub in frame.groupby("tau_mode"):
                axes[1].scatter(sub["tau_target"], sub["tau_rel_error"], alpha=0.5, s=16, label=mode)
            axes[1].axhline(0.2, color="black", linestyle="--", linewidth=1.0)
            axes[1].set_xscale("log")
            axes[1].set_xlabel("tau target")
            axes[1].set_ylabel("relative error |tau_post-tau*|/tau*")
        else:
            for mode, sub in frame.groupby("tau_mode"):
                vals = (sub["tau_post_mean"] - sub["tau_target"]).to_numpy()
                axes[1].hist(vals, bins=30, alpha=0.35, label=mode)
            axes[1].set_xlabel("tau posterior error")
            axes[1].set_ylabel("Count")
        axes[1].legend(fontsize=8)
        _save(fig, out_path)
        return

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for name, sub in frame.groupby("tau_prior"):
        vals = sub["m_eff"].to_numpy()
        ax.hist(vals, bins=40, density=True, alpha=0.35, label=name)
    ax.set_xlabel("m_eff")
    ax.set_ylabel("Density")
    ax.legend()
    _save(fig, out_path)


def plot_exp2_separation(df_summary: Any, df_kappa: Any, out_dir: Path) -> None:
    summary = _as_frame(df_summary)
    kappa = _as_frame(df_kappa)
    out_dir = Path(out_dir)

    # Figure A: null vs signal MSE and AUROC per method
    if not summary.empty and "method" in summary.columns:
        methods = list(summary["method"])
        x = np.arange(len(methods))
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        w = 0.35
        if "null_group_mse" in summary.columns and "signal_group_mse" in summary.columns:
            axes[0].bar(x - w / 2, summary["null_group_mse"], width=w, label="null", color="steelblue")
            axes[0].bar(x + w / 2, summary["signal_group_mse"], width=w, label="signal", color="salmon")
            axes[0].set_xticks(x, labels=methods, rotation=30, ha="right")
            axes[0].set_ylabel("Group MSE")
            axes[0].legend()
        if "group_auroc" in summary.columns:
            axes[1].bar(x, summary["group_auroc"], color="mediumseagreen")
            axes[1].set_xticks(x, labels=methods, rotation=30, ha="right")
            axes[1].set_ylim(0, 1)
            axes[1].set_ylabel("Group AUROC")
        if "lpd_test" in summary.columns:
            axes[2].bar(x, summary["lpd_test"], color="mediumpurple")
            axes[2].set_xticks(x, labels=methods, rotation=30, ha="right")
            axes[2].set_ylabel("MLPD (test)")
        _save(fig, out_dir / "fig2a_method_comparison.png")

    # Figure B: GR-RHS kappa by group
    if not kappa.empty and "group_id" in kappa.columns and "mean_kappa" in kappa.columns:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        groups = sorted(kappa["group_id"].unique())
        kappas = [kappa.loc[kappa["group_id"] == g, "mean_kappa"].to_numpy() for g in groups]
        labels_g = []
        for g in groups:
            sl = kappa.loc[kappa["group_id"] == g, "signal_label"].iloc[0] if not kappa.loc[kappa["group_id"] == g].empty else ""
            labels_g.append(f"g{int(g)+1}\n({sl})")
        ax.bar(np.arange(len(groups)), [np.nanmean(k) if len(k) > 0 else float("nan") for k in kappas], color="steelblue")
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0, label="u0=0.5")
        ax.set_xticks(np.arange(len(groups)), labels=labels_g)
        ax.set_ylabel("Mean kappa_g (GR-RHS)")
        ax.legend()
        _save(fig, out_dir / "fig2b_kappa_by_group.png")


def plot_exp3_benchmark(df: Any, out_dir: Path) -> None:
    frame = _as_frame(df)
    out_dir = Path(out_dir)
    if frame.empty or "signal" not in frame.columns:
        return

    signal_types = sorted(frame["signal"].unique())
    from .utils import method_display_name

    # Figure A: mse_overall by method per signal type (averaged over rho/snr)
    fig, axes = plt.subplots(1, len(signal_types), figsize=(5 * len(signal_types), 4.5), squeeze=False)
    for col_idx, sig in enumerate(signal_types):
        ax = axes[0][col_idx]
        sub = frame.loc[frame["signal"] == sig].groupby("method", as_index=False)["mse_overall"].mean()
        methods = [method_display_name(m) for m in sub["method"]]
        ax.bar(np.arange(len(methods)), sub["mse_overall"])
        ax.set_xticks(np.arange(len(methods)), labels=methods, rotation=35, ha="right")
        ax.set_title(f"Signal: {sig}")
        ax.set_ylabel("Mean MSE overall")
    _save(fig, out_dir / "fig3a_mse_by_signal.png")

    # Figure B: MLPD by method per signal type
    if "lpd_test" in frame.columns:
        fig, axes = plt.subplots(1, len(signal_types), figsize=(5 * len(signal_types), 4.5), squeeze=False)
        for col_idx, sig in enumerate(signal_types):
            ax = axes[0][col_idx]
            sub = frame.loc[frame["signal"] == sig].groupby("method", as_index=False)["lpd_test"].mean()
            methods = [method_display_name(m) for m in sub["method"]]
            ax.bar(np.arange(len(methods)), sub["lpd_test"])
            ax.set_xticks(np.arange(len(methods)), labels=methods, rotation=35, ha="right")
            ax.set_title(f"Signal: {sig}")
            ax.set_ylabel("Mean MLPD (test)")
        _save(fig, out_dir / "fig3b_lpd_by_signal.png")

    # Figure C: null MSE vs signal MSE scatter per method (boundary setting only)
    import pandas as pd
    boundary = frame.loc[frame["signal"] == "boundary"] if "boundary" in signal_types else pd.DataFrame()
    if not boundary.empty:
        agg = boundary.groupby("method", as_index=False).agg(mse_null=("mse_null", "mean"), mse_signal=("mse_signal", "mean"))
        fig, ax = plt.subplots(figsize=(6, 5))
        for _, row in agg.iterrows():
            ax.scatter(row["mse_null"], row["mse_signal"], s=80)
            ax.annotate(method_display_name(str(row["method"])), (row["mse_null"], row["mse_signal"]), fontsize=7)
        ax.set_xlabel("Null MSE (boundary signal)")
        ax.set_ylabel("Signal MSE (boundary signal)")
        _save(fig, out_dir / "fig3c_boundary_null_signal.png")


def plot_exp4_ablation(df: Any, out_dir: Path) -> None:
    frame = _as_frame(df)
    out_dir = Path(out_dir)
    if frame.empty or "variant" not in frame.columns:
        return

    variants = list(frame["variant"].unique()) if "variant" in frame.columns else []
    p0_vals = sorted(frame["p0_true"].unique()) if "p0_true" in frame.columns else []

    # Figure A: mse_overall by variant per p0_true
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    cmap = plt.cm.get_cmap("tab10", max(len(p0_vals), 1))
    for k, p0 in enumerate(p0_vals):
        sub = frame.loc[frame["p0_true"] == p0].set_index("variant") if len(p0_vals) > 1 else frame.set_index("variant")
        x = np.arange(len(variants))
        vals_mse = [float(sub.loc[v, "mse_overall"]) if v in sub.index else float("nan") for v in variants]
        vals_null = [float(sub.loc[v, "mse_null"]) if v in sub.index and "mse_null" in sub.columns else float("nan") for v in variants]
        vals_sig = [float(sub.loc[v, "mse_signal"]) if v in sub.index and "mse_signal" in sub.columns else float("nan") for v in variants]
        axes[0].plot(x, vals_mse, "o-", color=cmap(k), label=f"p0={p0}")
        axes[1].plot(x, vals_null, "o-", color=cmap(k))
        axes[2].plot(x, vals_sig, "o-", color=cmap(k))
    for ax, title in zip(axes, ["MSE Overall", "MSE Null", "MSE Signal"]):
        ax.set_xticks(np.arange(len(variants)), labels=variants, rotation=30, ha="right")
        ax.set_ylabel(title)
    axes[0].legend(fontsize=8)
    _save(fig, out_dir / "fig4a_mse_by_variant.png")

    # Figure B: tau calibration (ratio to oracle tau)
    if "tau_ratio_to_oracle" in frame.columns:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for k, p0 in enumerate(p0_vals):
            sub = frame.loc[frame["p0_true"] == p0]
            ax.plot(sub["variant"], sub["tau_ratio_to_oracle"], "o-", color=cmap(k), label=f"p0={p0}")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="oracle")
        ax.set_ylabel("tau_post_mean / tau_oracle")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", labelrotation=30)
        _save(fig, out_dir / "fig4b_tau_ratio.png")


def plot_exp5_prior_sensitivity(df: Any, out_dir: Path) -> None:
    frame = _as_frame(df)
    out_dir = Path(out_dir)
    if frame.empty:
        return

    frame = frame.copy()
    frame["prior_label"] = frame["alpha_kappa"].astype(str) + "," + frame["beta_kappa"].astype(str)
    setting_ids = sorted(frame["setting_id"].unique()) if "setting_id" in frame.columns else [None]

    metrics = [
        ("group_auroc", "Group AUROC"),
        ("mse_null", "Null MSE"),
        ("mse_signal", "Signal MSE"),
        ("kappa_null_mean", "Null kappa mean"),
        ("kappa_signal_mean", "Signal kappa mean"),
    ]
    valid_metrics = [(c, l) for c, l in metrics if c in frame.columns]
    ncols = len(valid_metrics)
    nrows = max(len(setting_ids), 1)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

    for row_idx, sid in enumerate(setting_ids):
        sub = frame.loc[frame["setting_id"] == sid] if sid is not None else frame
        priors = list(sub["prior_label"]) if not sub.empty else []
        x = np.arange(len(priors))
        for col_idx, (col, label) in enumerate(valid_metrics):
            ax = axes[row_idx][col_idx]
            if not sub.empty and col in sub.columns:
                ax.bar(x, sub[col].to_numpy(), color="steelblue")
                ax.set_xticks(x, labels=priors, rotation=30, ha="right", fontsize=7)
                ax.set_ylabel(label)
            if row_idx == 0:
                ax.set_title(label)
        axes[row_idx][0].set_ylabel(f"Scenario {sid}\n" + valid_metrics[0][1] if sid is not None else valid_metrics[0][1])

    _save(fig, out_dir / "fig5_prior_sensitivity.png")


def plot_exp9_prior_sensitivity(df_summary: Any, df_curve: Any, out_path: Path) -> None:
    summary = _as_frame(df_summary)
    curve = _as_frame(df_curve)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    baseline = summary.loc[summary["scenario_base"] == "baseline"].copy() if "scenario_base" in summary.columns else summary.copy()
    if not baseline.empty:
        baseline["prior"] = baseline["alpha_kappa"].astype(str) + "," + baseline["beta_kappa"].astype(str)
        agg_auc = baseline.groupby("prior", as_index=False)["group_auroc"].mean()
        axes[0].bar(agg_auc["prior"], agg_auc["group_auroc"])
        axes[0].tick_params(axis="x", labelrotation=25)
    axes[0].set_ylabel("AUROC (baseline avg)")
    axes[0].set_xlabel("(alpha_kappa, beta_kappa)")

    curve_base = curve.loc[curve["scenario_base"] == "baseline"].copy() if "scenario_base" in curve.columns else curve.copy()
    if not curve_base.empty:
        for (a, b), sub in curve_base.groupby(["alpha_kappa", "beta_kappa"]):
            s = sub.sort_values("p_g")
            axes[1].plot(s["p_g"], s["null_group_kappa_mean"], marker="o", label=f"({a:g},{b:g})")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("p_g")
    axes[1].set_ylabel("Null-group mean kappa")
    axes[1].legend(fontsize=7, ncol=2)
    _save(fig, out_path)
