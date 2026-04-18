from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Unified method color scheme (consistent across all exp1-5 figures)
# ---------------------------------------------------------------------------
_METHOD_COLORS: dict[str, str] = {
    "GR_RHS":       "#1f77b4",   # blue   — the proposed method
    "RHS":          "#ff7f0e",   # orange — individual-level baseline
    "GIGG_MMLE":    "#2ca02c",   # green
    "GIGG_b_small": "#98df8a",   # light green
    "GIGG_GHS":     "#17becf",   # cyan
    "GIGG_b_large": "#aec7e8",   # light blue
    "GHS_plus":     "#9467bd",   # purple
    "OLS":          "#8c564b",   # brown
    "LASSO_CV":     "#d62728",   # red
}
_METHOD_ORDER = ["GR_RHS", "RHS", "GIGG_MMLE", "GIGG_b_small", "GIGG_GHS", "GIGG_b_large", "GHS_plus", "OLS", "LASSO_CV"]

# Descriptive labels for (alpha_kappa, beta_kappa) prior configurations
_PRIOR_LABELS: dict[tuple[float, float], str] = {
    (0.5, 1.0): "default\n(0.5,1)",
    (1.0, 1.0): "uniform\n(1,1)",
    (0.5, 0.5): "U-shape\n(0.5,0.5)",
    (2.0, 5.0): "null-pref\n(2,5)",
    (1.0, 3.0): "moderate\n(1,3)",
}


def _method_color(name: str) -> str:
    return _METHOD_COLORS.get(str(name), "#7f7f7f")


def _sort_methods(methods) -> list[str]:
    methods_set = set(str(m) for m in methods)
    ordered = [m for m in _METHOD_ORDER if m in methods_set]
    extra = [m for m in methods if str(m) not in set(_METHOD_ORDER)]
    return ordered + extra


def _prior_label(alpha: float, beta: float) -> str:
    return _PRIOR_LABELS.get((float(alpha), float(beta)), f"({alpha:g},{beta:g})")


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
    """
    Exp1 Panel A — null contraction.

    Left:  log-log scatter of median E[kappa|Y_null] vs p_g with fitted slope and
           reference line at slope=-0.5 (Theorem 3.22).
    Right: P(kappa_g > eps | Y_null) vs p_g (log scale) — shows the WHOLE
           distribution shrinks, not just the mean.
    """
    rows = sorted(_records(df), key=lambda r: float(r["p_g"]))
    if not rows:
        return
    p_g = np.asarray([float(r["p_g"]) for r in rows], dtype=float)
    med = np.asarray([float(r["median_post_mean_kappa"]) for r in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # --- Left: log-log slope validation ---
    ax = axes[0]
    lx = np.log(p_g)
    ly = np.log(np.maximum(med, 1e-12))
    # Distinguish fit range (p_g 20-500) from out-of-range points visually
    fit_mask = (p_g >= 20) & (p_g <= 500)
    ax.plot(lx[fit_mask], ly[fit_mask], "o", color=_METHOD_COLORS["GR_RHS"], ms=7, zorder=3, label="fit range (p_g 20–500)")
    ax.plot(lx[~fit_mask], ly[~fit_mask], "o", color=_METHOD_COLORS["GR_RHS"], ms=7, zorder=3, alpha=0.35, markerfacecolor="none")
    ax.plot(lx, ly, "-", color=_METHOD_COLORS["GR_RHS"], alpha=0.4)
    # Draw fitted line through full x-range but anchored to fit-range regression
    fit_lx = lx[fit_mask]
    coef = np.polyfit(fit_lx, ly[fit_mask], deg=1)
    ax.plot(lx, coef[0] * lx + coef[1], "--", color="black", lw=1.5, label=f"fitted slope={slope:.3f}")
    ref = np.log(med[fit_mask][0]) - (-0.5) * fit_lx[0]
    ax.plot(lx, -0.5 * lx + ref, ":", color="gray", lw=1.2, label="theory slope=−0.5")
    ax.set_xlabel("log p_g", fontsize=10)
    ax.set_ylabel("log E[κ_g | Y_null]  (median)", fontsize=10)
    ci_str = f"[{slope_ci[0]:.3f}, {slope_ci[1]:.3f}]"
    ax.set_title(f"Null contraction (Thm 3.22)\nslope={slope:.3f}  95% CI {ci_str}", fontsize=9)
    ax.legend(fontsize=8)

    # --- Right: tail probability P(kappa > eps) vs p_g ---
    ax = axes[1]
    tail = np.asarray([float(r.get("mean_tail_prob_kappa_gt_eps", float("nan"))) for r in rows], dtype=float)
    if np.any(np.isfinite(tail)):
        ax.plot(p_g, tail, "o-", color=_METHOD_COLORS["GR_RHS"], ms=6, label="P(κ > ε | Y_null)")
        ax.set_xscale("log")
        ax.set_xlabel("p_g  (log scale)", fontsize=10)
        ax.set_ylabel("Mean P(κ_g > ε | Y_null)", fontsize=10)
        ax.set_title("Tail suppression as p_g grows\n(ε from tail_eps setting)", fontsize=9)
        ax.axhline(0.0, color="gray", lw=0.8, ls=":")
        ax.set_ylim(-0.02, max(0.5, float(np.nanmax(tail)) * 1.15))
        ax.legend(fontsize=8)
    else:
        # Fallback: IQR band (old behavior)
        q25 = np.asarray([float(r.get("q25_post_mean_kappa", float("nan"))) for r in rows], dtype=float)
        q75 = np.asarray([float(r.get("q75_post_mean_kappa", float("nan"))) for r in rows], dtype=float)
        ax.plot(p_g, med, "o-", color=_METHOD_COLORS["GR_RHS"], label="median")
        if np.any(np.isfinite(q25)):
            ax.fill_between(p_g, q25, q75, alpha=0.22, color=_METHOD_COLORS["GR_RHS"], label="IQR")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("p_g", fontsize=10)
        ax.set_ylabel("E[κ_g | Y_null]", fontsize=10)
        ax.legend(fontsize=8)

    _save(fig, out_path)


def plot_exp1_phase(df: Any, out_path: Path) -> None:
    """
    Exp1 Panel B — phase diagram (Corollary 3.33).

    Single panel: x = ξ/ξ_crit (dimensionless, tau-normalized), y = P(κ > u0 | Y).
    Lines are colored by p_g; tau is averaged over since the x-axis already
    normalizes it out — so curves for different tau should overlap, confirming
    that xi/xi_crit is the right scale.

    A shaded band shows the tau-variability. The vertical line at ξ/ξ_crit=1
    marks the theoretical transition point.
    """
    rows = _records(df)
    if not rows:
        return

    import pandas as pd
    frame = pd.DataFrame(rows)
    pg_vals = sorted(frame["p_g"].unique())
    xi_vals = sorted(frame["xi_ratio"].unique())
    cmap = plt.cm.get_cmap("plasma", len(pg_vals) + 1)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for j, pg in enumerate(pg_vals):
        sub = frame[frame["p_g"] == pg]
        # Average P(kappa>u0) over tau values at each xi_ratio
        agg = sub.groupby("xi_ratio")["mean_prob_kappa_gt_u0"].agg(["mean", "min", "max"]).reset_index()
        agg = agg.sort_values("xi_ratio")
        color = cmap(j)
        ax.plot(agg["xi_ratio"], agg["mean"], "o-", color=color, lw=1.8, ms=5, label=f"p_g={int(pg)}", zorder=3)
        # Shaded band shows tau-variability (how much curves differ across tau)
        if not (agg["min"] == agg["max"]).all():
            ax.fill_between(agg["xi_ratio"], agg["min"], agg["max"], alpha=0.12, color=color)

    ax.axvline(1.0, color="black", linestyle="--", lw=1.5, label="ξ = ξ_crit  (theory threshold)")
    ax.axhline(0.5, color="gray", linestyle=":", lw=0.9)
    ax.set_xlabel("ξ / ξ_crit  (signal strength / critical threshold)", fontsize=10)
    ax.set_ylabel("P(κ_g > u₀ | Y)", fontsize=10)
    ax.set_title("Phase diagram: signal retention  (Cor. 3.33)\nShaded band = variation across τ values; curves should overlap when normalized", fontsize=9)
    ax.set_ylim(-0.04, 1.08)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    _save(fig, out_path)


def plot_exp2_separation(df_summary: Any, df_kappa_raw: Any, out_dir: Path) -> None:
    """
    Exp2 — group separation (Theorem 3.34).

    Fig A: Method comparison — null/signal group MSE with error bars (SEM across
           replicates), AUROC, and MLPD. Methods sorted by group AUROC.
    Fig B: GR-RHS κ_g profile across groups — boxplot across replicates ordered
           by μ_g (gradient from null to strong signal), with null/signal regions
           shaded. Shows the step-up of κ_g at the signal boundary.

    df_kappa_raw should be the raw per-replicate kappa DataFrame (kappa_df),
    with columns: replicate_id, group_id, mu_g, signal_label, post_mean_kappa_g.
    """
    summary = _as_frame(df_summary)
    kappa = _as_frame(df_kappa_raw)
    out_dir = Path(out_dir)
    from .utils import method_display_name

    # --- Figure A: method comparison with error bars ---
    if not summary.empty and "method" in summary.columns:
        # Sort by group_auroc descending; GR_RHS first if tied
        if "group_auroc" in summary.columns:
            summary = summary.copy()
            summary["_rank"] = summary["group_auroc"].rank(ascending=False, method="first")
            gr_mask = summary["method"] == "GR_RHS"
            if gr_mask.any():
                summary.loc[gr_mask, "_rank"] = 0
            summary = summary.sort_values("_rank").drop(columns=["_rank"])

        methods = [method_display_name(m) for m in summary["method"]]
        raw_methods = list(summary["method"])
        x = np.arange(len(methods))
        colors = [_method_color(m) for m in raw_methods]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

        # Null vs signal group MSE
        ax = axes[0]
        w = 0.38
        null_vals = summary["null_group_mse"].to_numpy() if "null_group_mse" in summary else np.full(len(methods), np.nan)
        sig_vals = summary["signal_group_mse"].to_numpy() if "signal_group_mse" in summary else np.full(len(methods), np.nan)
        null_sem = summary["null_group_mse_std"].to_numpy() / np.sqrt(np.maximum(summary["n_effective"].to_numpy(), 1)) if "null_group_mse_std" in summary.columns else None
        sig_sem = summary["signal_group_mse_std"].to_numpy() / np.sqrt(np.maximum(summary["n_effective"].to_numpy(), 1)) if "signal_group_mse_std" in summary.columns else None
        bars_null = ax.bar(x - w / 2, null_vals, width=w, color=colors, alpha=0.7, label="null groups")
        bars_sig = ax.bar(x + w / 2, sig_vals, width=w, color=colors, alpha=1.0, label="signal groups", edgecolor="white", linewidth=0.5)
        if null_sem is not None:
            ax.errorbar(x - w / 2, null_vals, yerr=null_sem, fmt="none", color="black", capsize=3, lw=1)
        if sig_sem is not None:
            ax.errorbar(x + w / 2, sig_vals, yerr=sig_sem, fmt="none", color="black", capsize=3, lw=1)
        ax.set_xticks(x, labels=methods, rotation=32, ha="right", fontsize=8)
        ax.set_ylabel("Group-level L2 Error", fontsize=9)
        ax.set_title("Null (light) vs Signal (dark) MSE\nError bars = SEM across replicates", fontsize=8)
        ax.legend(fontsize=7)

        # Group AUROC
        ax = axes[1]
        auroc_vals = summary["group_auroc"].to_numpy() if "group_auroc" in summary.columns else np.full(len(methods), np.nan)
        auroc_sem = summary["group_auroc_std"].to_numpy() / np.sqrt(np.maximum(summary["n_effective"].to_numpy(), 1)) if "group_auroc_std" in summary.columns else None
        ax.bar(x, auroc_vals, color=colors, alpha=0.9)
        if auroc_sem is not None:
            ax.errorbar(x, auroc_vals, yerr=auroc_sem, fmt="none", color="black", capsize=3, lw=1)
        ax.set_xticks(x, labels=methods, rotation=32, ha="right", fontsize=8)
        ax.set_ylim(0, 1.08)
        ax.axhline(0.5, color="gray", ls=":", lw=0.8)
        ax.set_ylabel("Group AUROC", fontsize=9)
        ax.set_title("Group-separation AUROC\n(random = 0.5)", fontsize=8)

        # MLPD (test)
        ax = axes[2]
        if "lpd_test" in summary.columns:
            lpd_vals = summary["lpd_test"].to_numpy()
            lpd_sem = summary["lpd_test_std"].to_numpy() / np.sqrt(np.maximum(summary["n_effective"].to_numpy(), 1)) if "lpd_test_std" in summary.columns else None
            ax.bar(x, lpd_vals, color=colors, alpha=0.9)
            if lpd_sem is not None:
                ax.errorbar(x, lpd_vals, yerr=lpd_sem, fmt="none", color="black", capsize=3, lw=1)
            ax.set_xticks(x, labels=methods, rotation=32, ha="right", fontsize=8)
            ax.set_ylabel("MLPD (test set)", fontsize=9)
            ax.set_title("Marginal log predictive density\n(higher = better)", fontsize=8)

        # n_effective annotation
        for ax_i in axes:
            ax_i.annotate("", xy=(0, 0), xytext=(0, 0))
        for mi, (ax_idx, n_eff) in enumerate(zip(range(3), [])):
            pass

        _save(fig, out_dir / "fig2a_method_comparison.png")

    # --- Figure B: GR-RHS κ_g distribution by group (boxplot across replicates) ---
    # kappa can be either the raw kappa_df (post_mean_kappa_g column) or kappa_summary (mean_kappa)
    if not kappa.empty:
        # Detect which data format we have
        kappa_col = "post_mean_kappa_g" if "post_mean_kappa_g" in kappa.columns else "mean_kappa"
        if kappa_col not in kappa.columns or "group_id" not in kappa.columns:
            return

        groups = sorted(kappa["group_id"].unique())
        # Build tick labels with signal info
        labels_g = []
        colors_g = []
        for g in groups:
            sub = kappa[kappa["group_id"] == g]
            if sub.empty:
                labels_g.append(f"g{int(g)+1}")
                colors_g.append("#cccccc")
                continue
            sig = int(sub["signal_label"].iloc[0]) if "signal_label" in sub.columns else 0
            mu_val = float(sub["mu_g"].iloc[0]) if "mu_g" in sub.columns else 0.0
            xi_r = float(sub["xi_ratio"].iloc[0]) if "xi_ratio" in sub.columns else float("nan")
            if not math.isnan(xi_r):
                regime = "null" if xi_r == 0 else f"xi/xi_c={xi_r:.1f}"
                labels_g.append(f"G{int(g)}\n{regime}")
            else:
                labels_g.append(f"G{int(g)}\n{'null' if mu_val == 0 else f'mu={mu_val:.2g}'}")
            colors_g.append("#ff7f7f" if sig else "#7fb3d3")

        fig, ax = plt.subplots(figsize=(8, 4.5))
        data_per_group = [kappa.loc[kappa["group_id"] == g, kappa_col].dropna().to_numpy() for g in groups]

        bp = ax.boxplot(
            data_per_group,
            positions=np.arange(len(groups)),
            patch_artist=True,
            medianprops=dict(color="black", lw=2),
            widths=0.55,
            showfliers=len(data_per_group[0]) > 3 if data_per_group else True,
        )
        for patch, color in zip(bp["boxes"], colors_g):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Shade null vs signal regions
        n_null = sum(1 for g in groups if kappa.loc[kappa["group_id"] == g, "signal_label"].iloc[0] == 0
                     if not kappa.loc[kappa["group_id"] == g].empty and "signal_label" in kappa.columns)
        if n_null > 0:
            ax.axvspan(-0.5, n_null - 0.5, alpha=0.06, color="blue", label="null groups")
            ax.axvspan(n_null - 0.5, len(groups) - 0.5, alpha=0.06, color="red", label="signal groups")

        ax.axhline(0.5, color="black", ls="--", lw=1.2, label="u₀ = 0.5")
        ax.set_xticks(np.arange(len(groups)), labels=labels_g, fontsize=8)
        ax.set_ylabel("Posterior mean κ_g  (GR-RHS)", fontsize=10)
        ax.set_title(
            "GR-RHS κ_g profile by group  (Theorem 3.34)\n"
            "Blue = null groups, Red = signal groups  |  Box = IQR across replicates",
            fontsize=9,
        )
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=8, loc="upper left")
        _save(fig, out_dir / "fig2b_kappa_by_group.png")


def plot_exp3_benchmark(df: Any, out_dir: Path) -> None:
    """
    Exp3 — factorial benchmark (signal_type × rho_within × snr).

    Fig A (primary): One panel per signal type; x = rho_within, y = MSE overall;
      one line per method with consistent colors. snr is shown via subplots or
      averaged if only one snr value is present. Preserves the rho interaction
      that bar charts collapse away.

    Fig B (secondary, if lpd_test present): Same layout for MLPD.

    Fig C (scatter, if boundary columns present): null vs signal group MSE scatter
      per method, colored by method, to show joint null+signal trade-off.
    """
    frame = _as_frame(df)
    out_dir = Path(out_dir)
    if frame.empty:
        return

    from .utils import method_display_name

    signal_col = "signal" if "signal" in frame.columns else ("signal_type" if "signal_type" in frame.columns else None)
    rho_col = "rho_within" if "rho_within" in frame.columns else None

    if signal_col is None or rho_col is None:
        # Fallback: bar chart per method, single panel
        if "method" in frame.columns and "mse_overall" in frame.columns:
            methods_raw = _sort_methods(frame["method"].unique())
            fig, ax = plt.subplots(figsize=(8, 4.5))
            x = np.arange(len(methods_raw))
            vals = [float(frame.loc[frame["method"] == m, "mse_overall"].mean()) for m in methods_raw]
            colors = [_method_color(m) for m in methods_raw]
            labels = [method_display_name(m) for m in methods_raw]
            ax.bar(x, vals, color=colors)
            ax.set_xticks(x, labels=labels, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel("Mean MSE overall")
            ax.set_title("Method comparison  (Exp3)")
            _save(fig, out_dir / "fig3a_mse_by_signal.png")
        return

    signal_types = sorted(frame[signal_col].unique())
    snr_col = "snr" if "snr" in frame.columns else None
    snr_vals = sorted(frame[snr_col].unique()) if snr_col else [None]
    methods_raw = _sort_methods(frame["method"].unique())

    # --- Fig A: MSE line plots (rho × method, panel per signal type) ---
    n_snr = len(snr_vals)
    n_sig = len(signal_types)
    fig_a, axes_a = plt.subplots(n_snr, n_sig, figsize=(5 * n_sig, 4.2 * n_snr), squeeze=False)

    for row_i, snr in enumerate(snr_vals):
        for col_i, sig in enumerate(signal_types):
            ax = axes_a[row_i][col_i]
            sub = frame[frame[signal_col] == sig]
            if snr is not None:
                sub = sub[sub[snr_col] == snr]
            rho_vals = sorted(sub[rho_col].unique())
            for m in methods_raw:
                msub = sub[sub["method"] == m]
                if msub.empty:
                    continue
                agg = msub.groupby(rho_col)["mse_overall"].mean()
                xs = [r for r in rho_vals if r in agg.index]
                ys = [float(agg.loc[r]) for r in xs]
                ax.plot(xs, ys, "o-", color=_method_color(m), label=method_display_name(m), lw=1.8, ms=5)
            snr_str = f"  SNR={snr}" if snr is not None else ""
            ax.set_title(f"Signal: {sig}{snr_str}", fontsize=9)
            ax.set_xlabel("ρ_within", fontsize=9)
            ax.set_ylabel("MSE overall", fontsize=9)
            if col_i == n_sig - 1 and row_i == 0:
                ax.legend(fontsize=7, loc="upper left", ncol=1)

    fig_a.suptitle("Exp3: MSE vs correlation (per signal type/SNR)\nPreserves rho×method interaction", fontsize=10, y=1.01)
    _save(fig_a, out_dir / "fig3a_mse_by_signal.png")

    # --- Fig B: MLPD line plots (same layout) ---
    if "lpd_test" in frame.columns:
        fig_b, axes_b = plt.subplots(n_snr, n_sig, figsize=(5 * n_sig, 4.2 * n_snr), squeeze=False)
        for row_i, snr in enumerate(snr_vals):
            for col_i, sig in enumerate(signal_types):
                ax = axes_b[row_i][col_i]
                sub = frame[frame[signal_col] == sig]
                if snr is not None:
                    sub = sub[sub[snr_col] == snr]
                rho_vals = sorted(sub[rho_col].unique())
                for m in methods_raw:
                    msub = sub[sub["method"] == m]
                    if msub.empty:
                        continue
                    agg = msub.groupby(rho_col)["lpd_test"].mean()
                    xs = [r for r in rho_vals if r in agg.index]
                    ys = [float(agg.loc[r]) for r in xs]
                    ax.plot(xs, ys, "o-", color=_method_color(m), label=method_display_name(m), lw=1.8, ms=5)
                snr_str = f"  SNR={snr}" if snr is not None else ""
                ax.set_title(f"Signal: {sig}{snr_str}", fontsize=9)
                ax.set_xlabel("ρ_within", fontsize=9)
                ax.set_ylabel("MLPD (test)", fontsize=9)
                if col_i == n_sig - 1 and row_i == 0:
                    ax.legend(fontsize=7, loc="lower left", ncol=1)
        fig_b.suptitle("Exp3: MLPD vs correlation (per signal type/SNR)", fontsize=10, y=1.01)
        _save(fig_b, out_dir / "fig3b_lpd_by_signal.png")

    # --- Fig C: null vs signal group MSE scatter (joint trade-off, one point per method) ---
    if "null_group_mse" in frame.columns and "signal_group_mse" in frame.columns:
        fig_c, ax_c = plt.subplots(figsize=(6.5, 5.5))
        for m in methods_raw:
            msub = frame[frame["method"] == m]
            if msub.empty:
                continue
            xv = float(msub["null_group_mse"].mean())
            yv = float(msub["signal_group_mse"].mean())
            ax_c.scatter(xv, yv, color=_method_color(m), s=80, zorder=3, label=method_display_name(m))
            ax_c.annotate(method_display_name(m), (xv, yv), textcoords="offset points", xytext=(5, 3), fontsize=7)
        ax_c.set_xlabel("Null group MSE (lower = better shrinkage)", fontsize=9)
        ax_c.set_ylabel("Signal group MSE (lower = better recovery)", fontsize=9)
        ax_c.set_title("Null vs signal MSE trade-off  (averaged over rho, snr, signal)\nBottom-left = best on both axes", fontsize=9)
        ax_c.legend(fontsize=7, loc="upper right")
        _save(fig_c, out_dir / "fig3c_null_signal_scatter.png")


def plot_exp4_ablation(df: Any, out_dir: Path) -> None:
    """
    Exp4 — ablation / τ calibration.

    Fig A (PRIMARY): τ_post_mean vs τ_oracle scatter.
      - One point per (variant × p0_true) combination.
      - Color = p0_true, marker shape = variant.
      - Identity line (y=x) is the oracle target.
      - This is the clearest diagnostic for τ calibration.

    Fig B (secondary): Normalized MSE per variant (MSE_variant / MSE_oracle_p0).
      - One group of bars per p0, bars colored by variant.
      - Normalizing to oracle removes the trivial p0 scaling, so all p0 values
        are on the same plot without dominating each other.

    The old `set_index("variant")` pattern crashed when multiple p0 values caused
    duplicate index entries — replaced with pivot_table throughout.
    """
    frame = _as_frame(df)
    out_dir = Path(out_dir)
    if frame.empty or "variant" not in frame.columns:
        return

    import pandas as pd

    variants = _sort_methods(frame["variant"].unique())  # use consistent ordering helper
    p0_vals = sorted(frame["p0_true"].unique()) if "p0_true" in frame.columns else [None]
    cmap_p0 = plt.cm.get_cmap("tab10", max(len(p0_vals), 1))

    # Marker cycle for variants
    _MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]
    var_marker = {v: _MARKERS[i % len(_MARKERS)] for i, v in enumerate(variants)}

    # --- Fig A: τ_post_mean vs τ_oracle scatter ---
    tau_cols_present = "tau_post_mean" in frame.columns and "tau0_oracle" in frame.columns
    if tau_cols_present:
        fig_a, ax_a = plt.subplots(figsize=(6.5, 5.5))
        handles_p0 = []
        for k, p0 in enumerate(p0_vals):
            sub = frame[frame["p0_true"] == p0] if p0 is not None else frame
            for v in variants:
                vsub = sub[sub["variant"] == v]
                if vsub.empty:
                    continue
                xv = float(vsub["tau0_oracle"].mean())
                yv = float(vsub["tau_post_mean"].mean())
                sc = ax_a.scatter(
                    xv, yv,
                    color=cmap_p0(k),
                    marker=var_marker[v],
                    s=90, zorder=3,
                    label=f"p0={p0}" if v == variants[0] else None,
                )
            if p0 is not None:
                handles_p0.append(plt.Line2D([0], [0], marker="o", color="w",
                                              markerfacecolor=cmap_p0(k), markersize=8,
                                              label=f"p0={p0}"))

        # Marker legend for variants
        var_handles = [plt.Line2D([0], [0], marker=var_marker[v], color="gray",
                                   linestyle="None", markersize=8, label=v)
                       for v in variants]

        # Identity line
        all_x = frame["tau0_oracle"].dropna().to_numpy()
        all_y = frame["tau_post_mean"].dropna().to_numpy()
        lims = [min(all_x.min(), all_y.min()) * 0.85, max(all_x.max(), all_y.max()) * 1.15]
        ax_a.plot(lims, lims, "--", color="black", lw=1.3, label="identity (oracle)")
        ax_a.set_xlim(lims); ax_a.set_ylim(lims)
        ax_a.set_xlabel("τ oracle  (p0/(p-p0)/√n)", fontsize=10)
        ax_a.set_ylabel("τ posterior mean", fontsize=10)
        ax_a.set_title("τ calibration scatter  (Exp4)\nPoints on identity line = perfect calibration", fontsize=9)
        leg1 = ax_a.legend(handles=handles_p0 + var_handles, fontsize=8, loc="upper left",
                            ncol=2, title="color=p0, marker=variant")
        ax_a.add_artist(leg1)
        _save(fig_a, out_dir / "fig4a_tau_scatter.png")

    # --- Fig B: normalized MSE bar chart (variant / oracle_p0) ---
    if "mse_overall" in frame.columns:
        # Compute oracle MSE per p0 (variant=="GR_RHS_full" or first variant as reference)
        ref_variant = "GR_RHS_full" if "GR_RHS_full" in variants else variants[0]
        ref_df = frame[frame["variant"] == ref_variant]
        oracle_mse: dict = {}
        for p0 in p0_vals:
            sub = ref_df[ref_df["p0_true"] == p0] if p0 is not None else ref_df
            oracle_mse[p0] = float(sub["mse_overall"].mean()) if not sub.empty else float("nan")

        # pivot: rows=p0, cols=variant → normalized MSE
        pivot = pd.DataFrame(index=[str(p) for p in p0_vals], columns=variants, dtype=float)
        for p0 in p0_vals:
            for v in variants:
                sub = frame[(frame["p0_true"] == p0) & (frame["variant"] == v)] if p0 is not None else frame[frame["variant"] == v]
                raw = float(sub["mse_overall"].mean()) if not sub.empty else float("nan")
                denom = oracle_mse.get(p0, float("nan"))
                pivot.loc[str(p0), v] = raw / denom if np.isfinite(denom) and denom > 0 else float("nan")

        fig_b, ax_b = plt.subplots(figsize=(max(7, 2 * len(variants)), 4.5))
        n_p0 = len(p0_vals)
        n_var = len(variants)
        width = 0.8 / max(n_p0, 1)
        x = np.arange(n_var)
        for k, p0 in enumerate(p0_vals):
            ys = [float(pivot.loc[str(p0), v]) if not np.isnan(float(pivot.loc[str(p0), v])) else 0.0
                  for v in variants]
            offset = (k - n_p0 / 2.0 + 0.5) * width
            ax_b.bar(x + offset, ys, width=width, color=cmap_p0(k), alpha=0.85, label=f"p0={p0}")
        ax_b.axhline(1.0, color="black", ls="--", lw=1.2, label=f"reference ({ref_variant})")
        ax_b.set_xticks(x, labels=variants, rotation=30, ha="right", fontsize=8)
        ax_b.set_ylabel(f"MSE / MSE({ref_variant})  (normalized)", fontsize=9)
        ax_b.set_title("Exp4: Normalized MSE per ablation variant\n<1 = better than reference; >1 = worse", fontsize=9)
        ax_b.legend(fontsize=8)
        _save(fig_b, out_dir / "fig4b_mse_normalized.png")


def plot_exp5_prior_sensitivity(df: Any, out_dir: Path) -> None:
    """
    Exp5 — prior sensitivity for (alpha_kappa, beta_kappa).

    Design goal: demonstrate ROBUSTNESS — show that different (alpha, beta) priors
    give nearly identical results, so the default (0.5, 1.0) is safe.

    Fig A (primary): One panel per metric; x-axis = prior configuration (with
      descriptive labels); one colored line per scenario; default prior (0.5, 1.0)
      highlighted with a vertical dashed line. Flat lines = robust.

    Fig B (secondary, if kappa columns present): κ_null vs κ_signal scatter across
      priors, one point per prior × scenario; shows separation is stable.

    Fixes the old bugs:
      - `\\n` → `\n` in ylabel (was literal backslash-n)
      - Each setting/scenario is now a LINE (not a row of subplots), making the
        robustness story immediately visible.
    """
    frame = _as_frame(df)
    out_dir = Path(out_dir)
    if frame.empty:
        return

    frame = frame.copy()

    # Assign descriptive labels to each (alpha, beta) pair
    def _pl(row) -> str:
        return _prior_label(float(row["alpha_kappa"]), float(row["beta_kappa"]))

    frame["prior_label"] = frame.apply(_pl, axis=1)

    # Determine canonical prior order for x-axis (default prior first / highlighted)
    _DEFAULT_PRIOR = (0.5, 1.0)
    prior_pairs = sorted(
        frame[["alpha_kappa", "beta_kappa"]].drop_duplicates().itertuples(index=False),
        key=lambda r: (0 if (float(r.alpha_kappa), float(r.beta_kappa)) == _DEFAULT_PRIOR else 1,
                       float(r.alpha_kappa), float(r.beta_kappa)),
    )
    prior_order = [_prior_label(r.alpha_kappa, r.beta_kappa) for r in prior_pairs]
    default_label = _prior_label(*_DEFAULT_PRIOR)
    default_x = prior_order.index(default_label) if default_label in prior_order else None

    setting_ids = sorted(frame["setting_id"].unique()) if "setting_id" in frame.columns else [None]
    scenario_cmap = plt.cm.get_cmap("tab10", max(len(setting_ids), 1))

    metrics = [
        ("group_auroc",       "Group AUROC"),
        ("mse_null",          "Null MSE"),
        ("mse_signal",        "Signal MSE"),
        ("kappa_null_mean",   "κ_null mean"),
        ("kappa_signal_mean", "κ_signal mean"),
    ]
    valid_metrics = [(c, lbl) for c, lbl in metrics if c in frame.columns]
    if not valid_metrics:
        return

    # --- Fig A: robustness line plots ---
    ncols = min(len(valid_metrics), 3)
    nrows = (len(valid_metrics) + ncols - 1) // ncols
    fig_a, axes_a = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.0 * nrows), squeeze=False)

    x_pos = np.arange(len(prior_order))
    scenario_handles = []

    for panel_idx, (col, lbl) in enumerate(valid_metrics):
        ri, ci = divmod(panel_idx, ncols)
        ax = axes_a[ri][ci]

        # Vertical band highlighting the default prior
        if default_x is not None:
            ax.axvspan(default_x - 0.4, default_x + 0.4, alpha=0.10, color="gold", zorder=0)
            if panel_idx == 0:
                ax.axvline(default_x, color="goldenrod", ls="--", lw=1.4, label="default prior")

        for k, sid in enumerate(setting_ids):
            sub = frame[frame["setting_id"] == sid] if sid is not None else frame
            ys = []
            for pl in prior_order:
                psub = sub[sub["prior_label"] == pl]
                ys.append(float(psub[col].mean()) if not psub.empty and col in psub.columns else float("nan"))
            color = scenario_cmap(k)
            line, = ax.plot(x_pos, ys, "o-", color=color, lw=1.8, ms=5,
                            label=f"Scenario {sid}" if sid is not None else "all")
            if panel_idx == 0:
                scenario_handles.append(line)

        ax.set_xticks(x_pos, labels=prior_order, rotation=28, ha="right", fontsize=8)
        ax.set_ylabel(lbl, fontsize=9)
        ax.set_title(lbl, fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    # Hide unused panels
    for idx in range(len(valid_metrics), nrows * ncols):
        ri, ci = divmod(idx, ncols)
        axes_a[ri][ci].set_visible(False)

    # Shared legend
    all_handles = scenario_handles
    if default_x is not None:
        all_handles = [plt.Line2D([0], [0], color="goldenrod", ls="--", lw=1.4, label="default prior")] + scenario_handles
    fig_a.legend(handles=all_handles, fontsize=8, loc="lower center",
                 ncol=min(len(all_handles), 6), bbox_to_anchor=(0.5, -0.04))
    fig_a.suptitle(
        "Exp5: Prior sensitivity for (α_κ, β_κ)  — flat lines = robust\n"
        "Gold band = default prior (0.5, 1.0)  |  One line per scenario",
        fontsize=10, y=1.01,
    )
    _save(fig_a, out_dir / "fig5_prior_sensitivity.png")

    # --- Fig B: κ separation scatter across priors (if both kappa columns present) ---
    if "kappa_null_mean" in frame.columns and "kappa_signal_mean" in frame.columns:
        fig_b, ax_b = plt.subplots(figsize=(6, 5.5))
        for k, pl in enumerate(prior_order):
            psub = frame[frame["prior_label"] == pl]
            if psub.empty:
                continue
            xv = float(psub["kappa_null_mean"].mean())
            yv = float(psub["kappa_signal_mean"].mean())
            is_default = (pl == default_label)
            ax_b.scatter(xv, yv,
                         color="goldenrod" if is_default else "steelblue",
                         s=110 if is_default else 70,
                         zorder=3 if is_default else 2,
                         edgecolors="black" if is_default else "none",
                         linewidth=1.2 if is_default else 0)
            ax_b.annotate(pl, (xv, yv), textcoords="offset points", xytext=(5, 3), fontsize=7)
        ax_b.set_xlabel("κ_null mean  (target: low)", fontsize=9)
        ax_b.set_ylabel("κ_signal mean  (target: high)", fontsize=9)
        ax_b.set_title("Exp5: κ separation across priors\nGold = default prior; ideal = bottom-right", fontsize=9)
        _save(fig_b, out_dir / "fig5b_kappa_separation.png")
