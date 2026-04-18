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


def plot_exp2_separation(df_summary: Any, df_kappa: Any, out_dir: Path) -> None:
    summary = _as_frame(df_summary)
    kappa = _as_frame(df_kappa)
    out_dir = Path(out_dir)

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

    if not kappa.empty and "group_id" in kappa.columns and "mean_kappa" in kappa.columns:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        groups = sorted(kappa["group_id"].unique())
        kappas = [kappa.loc[kappa["group_id"] == g, "mean_kappa"].to_numpy() for g in groups]
        labels_g = []
        for g in groups:
            subset = kappa.loc[kappa["group_id"] == g]
            signal_label = subset["signal_label"].iloc[0] if ("signal_label" in subset.columns and not subset.empty) else ""
            labels_g.append(f"g{int(g)+1}\\n({signal_label})")
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


def plot_exp4_ablation(df: Any, out_dir: Path) -> None:
    frame = _as_frame(df)
    out_dir = Path(out_dir)
    if frame.empty or "variant" not in frame.columns:
        return

    variants = list(frame["variant"].unique()) if "variant" in frame.columns else []
    p0_vals = sorted(frame["p0_true"].unique()) if "p0_true" in frame.columns else []

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    cmap = plt.cm.get_cmap("tab10", max(len(p0_vals), 1))
    for k, p0 in enumerate(p0_vals):
        sub = frame.loc[frame["p0_true"] == p0].set_index("variant") if len(p0_vals) > 1 else frame.set_index("variant")
        x = np.arange(len(variants))
        vals_mse = [float(sub.loc[v, "mse_overall"]) if v in sub.index else float("nan") for v in variants]
        vals_null = [float(sub.loc[v, "mse_null"]) if (v in sub.index and "mse_null" in sub.columns) else float("nan") for v in variants]
        vals_sig = [float(sub.loc[v, "mse_signal"]) if (v in sub.index and "mse_signal" in sub.columns) else float("nan") for v in variants]
        axes[0].plot(x, vals_mse, "o-", color=cmap(k), label=f"p0={p0}")
        axes[1].plot(x, vals_null, "o-", color=cmap(k))
        axes[2].plot(x, vals_sig, "o-", color=cmap(k))
    for ax, title in zip(axes, ["MSE Overall", "MSE Null", "MSE Signal"]):
        ax.set_xticks(np.arange(len(variants)), labels=variants, rotation=30, ha="right")
        ax.set_ylabel(title)
    axes[0].legend(fontsize=8)
    _save(fig, out_dir / "fig4a_mse_by_variant.png")

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
        axes[row_idx][0].set_ylabel(f"Scenario {sid}\\n" + valid_metrics[0][1] if sid is not None else valid_metrics[0][1])

    _save(fig, out_dir / "fig5_prior_sensitivity.png")
