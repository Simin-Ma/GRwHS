from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..runtime import kappa_star_xi_ratio_u0_rho

# ---------------------------------------------------------------------------
# Unified method color scheme (consistent across all exp1-5 figures)
# ---------------------------------------------------------------------------
_METHOD_COLORS: dict[str, str] = {
    "GR_RHS":       "#1f77b4",   # blue   �� the proposed method
    "RHS":          "#ff7f0e",   # orange �� individual-level baseline
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

# Common math-label snippets for Exp1 figures.
_LBL_PG = r"$p_g$"
_LBL_KAPPA = r"$\kappa_g$"
_LBL_XI_RATIO = r"$\xi/\xi_{\mathrm{crit}}$"
_LBL_E_KAPPA = rf"$\mathbb{{E}}[{_LBL_KAPPA[1:-1]}\mid Y]$"
_LBL_E_KAPPA_NULL = rf"$\mathbb{{E}}[{_LBL_KAPPA[1:-1]}\mid Y_{{\mathrm{{null}}}}]$"
_LBL_P_KEEP = rf"$\mathbb{{P}}({_LBL_KAPPA[1:-1]}>u_0\mid Y)$"


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
    suptitle = getattr(fig, "_suptitle", None)
    has_suptitle = bool(suptitle is not None and str(suptitle.get_text()).strip())
    if has_suptitle:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    else:
        fig.tight_layout()
    save_kws = dict(dpi=240, bbox_inches="tight", pad_inches=0.10)
    fig.savefig(path, **save_kws)
    # Keep an immutable timestamped snapshot for each generated figure.
    history_dir = path.parent / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fig.savefig(history_dir / f"{path.stem}_{ts}{path.suffix}", **save_kws)
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
    from ...utils import load_pandas
    pd = load_pandas()

    if hasattr(df, "groupby"):
        return df
    return pd.DataFrame(_records(df))


def plot_exp1(
    df: Any,
    slope: float,
    slope_ci: tuple[float, float],
    out_path: Path,
    *,
    full_df: Any | None = None,
    full_slope: float | None = None,
    full_slope_ci: tuple[float, float] | None = None,
) -> None:
    """Exp1 Panel A: profile null-contraction with optional full-model overlay."""
    rows = sorted(_records(df), key=lambda r: float(r["p_g"]))
    if not rows:
        return
    p_g = np.asarray([float(r["p_g"]) for r in rows], dtype=float)
    med = np.asarray([float(r["median_post_mean_kappa"]) for r in rows], dtype=float)

    rows_full = sorted(_records(full_df), key=lambda r: float(r["p_g"])) if full_df is not None else []
    p_full = np.asarray([float(r["p_g"]) for r in rows_full], dtype=float) if rows_full else np.asarray([], dtype=float)
    med_full = (
        np.asarray([float(r["median_post_mean_kappa"]) for r in rows_full], dtype=float)
        if rows_full
        else np.asarray([], dtype=float)
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # --- Left: log-log slope validation ---
    ax = axes[0]
    lx = np.log(p_g)
    ly = np.log(np.maximum(med, 1e-12))
    fit_mask = (p_g >= 20) & (p_g <= 500)
    ax.plot(
        lx[fit_mask],
        ly[fit_mask],
        "o",
        color=_METHOD_COLORS["GR_RHS"],
        ms=7,
        zorder=3,
        label=r"Profile fit range: $20\leq p_g\leq 500$",
    )
    ax.plot(lx[~fit_mask], ly[~fit_mask], "o", color=_METHOD_COLORS["GR_RHS"], ms=7, zorder=3, alpha=0.35, markerfacecolor="none")
    ax.plot(lx, ly, "-", color=_METHOD_COLORS["GR_RHS"], alpha=0.4)

    fit_lx = lx[fit_mask]
    coef = np.polyfit(fit_lx, ly[fit_mask], deg=1)
    ax.plot(lx, coef[0] * lx + coef[1], "--", color="black", lw=1.5, label=rf"Profile fit: $\hat s={slope:.3f}$")
    ref = np.log(med[fit_mask][0]) - (-0.5) * fit_lx[0]
    ax.plot(lx, -0.5 * lx + ref, ":", color="gray", lw=1.2, label=r"Theory: $s=-\frac{1}{2}$")

    if rows_full:
        ok_full = np.isfinite(p_full) & np.isfinite(med_full) & (med_full > 0)
        if np.any(ok_full):
            lx_full = np.log(p_full[ok_full])
            ly_full = np.log(np.maximum(med_full[ok_full], 1e-12))
            fit_mask_full = (p_full[ok_full] >= 20) & (p_full[ok_full] <= 500)
            ax.plot(
                lx_full[fit_mask_full],
                ly_full[fit_mask_full],
                "s",
                color=_METHOD_COLORS["RHS"],
                ms=6,
                zorder=3,
                label=r"Full fit range: $20\leq p_g\leq 500$",
            )
            ax.plot(lx_full[~fit_mask_full], ly_full[~fit_mask_full], "s", color=_METHOD_COLORS["RHS"], ms=6, zorder=3, alpha=0.35, markerfacecolor="none")
            ax.plot(lx_full, ly_full, "-", color=_METHOD_COLORS["RHS"], alpha=0.45)
            if int(np.sum(fit_mask_full)) >= 2:
                coef_full = np.polyfit(lx_full[fit_mask_full], ly_full[fit_mask_full], deg=1)
                slope_full_show = float(full_slope) if full_slope is not None else float(coef_full[0])
                ax.plot(
                    lx_full,
                    coef_full[0] * lx_full + coef_full[1],
                    "--",
                    color=_METHOD_COLORS["RHS"],
                    lw=1.5,
                    label=rf"Full fit: $\hat s={slope_full_show:.3f}$",
                )

    ax.set_xlabel(r"$\log p_g$", fontsize=10)
    ax.set_ylabel(r"$\log\,\operatorname{median}\!\left(\mathbb{E}[\kappa_g\mid Y_{\mathrm{null}}]\right)$", fontsize=10)
    ci_str = f"[{slope_ci[0]:.3f}, {slope_ci[1]:.3f}]"
    title = rf"Null contraction (Thm 3.22)" + "\n" + rf"Profile $\hat s={slope:.3f}$, 95% CI {ci_str}"
    if full_slope is not None and full_slope_ci is not None:
        title += (
            "\n"
            + rf"Full $\hat s={float(full_slope):.3f}$, 95% CI "
            + rf"[{float(full_slope_ci[0]):.3f}, {float(full_slope_ci[1]):.3f}]"
        )
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)

    # --- Right: tail probability P(kappa > eps) vs p_g ---
    ax = axes[1]
    tail = np.asarray([float(r.get("mean_tail_prob_kappa_gt_eps", float("nan"))) for r in rows], dtype=float)
    if np.any(np.isfinite(tail)):
        ax.plot(p_g, tail, "o-", color=_METHOD_COLORS["GR_RHS"], ms=6, label=r"Profile $\mathbb{P}(\kappa_g>\varepsilon)$")
        if rows_full:
            tail_full = np.asarray([float(r.get("mean_tail_prob_kappa_gt_eps", float("nan"))) for r in rows_full], dtype=float)
            ok_tail = np.isfinite(p_full) & np.isfinite(tail_full)
            if np.any(ok_tail):
                ax.plot(
                    p_full[ok_tail],
                    tail_full[ok_tail],
                    "s--",
                    color=_METHOD_COLORS["RHS"],
                    ms=5,
                    lw=1.2,
                    label=r"Full $\mathbb{P}(\kappa_g>\varepsilon)$",
                )
        ax.set_xscale("log")
        ax.set_xlabel(r"$p_g$ (log scale)", fontsize=10)
        ax.set_ylabel(r"$\mathbb{E}\!\left[\mathbb{P}(\kappa_g>\varepsilon\mid Y_{\mathrm{null}})\right]$", fontsize=10)
        ax.set_title(r"Tail suppression as $p_g$ grows", fontsize=9)
        ax.axhline(0.0, color="gray", lw=0.8, ls=":")
        ax.set_ylim(-0.02, max(0.5, float(np.nanmax(tail)) * 1.15))
        ax.legend(fontsize=8)
    else:
        q25 = np.asarray([float(r.get("q25_post_mean_kappa", float("nan"))) for r in rows], dtype=float)
        q75 = np.asarray([float(r.get("q75_post_mean_kappa", float("nan"))) for r in rows], dtype=float)
        ax.plot(p_g, med, "o-", color=_METHOD_COLORS["GR_RHS"], label=r"$\mathrm{median}$")
        if np.any(np.isfinite(q25)):
            ax.fill_between(p_g, q25, q75, alpha=0.22, color=_METHOD_COLORS["GR_RHS"], label=r"$\mathrm{IQR}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(_LBL_PG, fontsize=10)
        ax.set_ylabel(_LBL_E_KAPPA_NULL, fontsize=10)
        ax.legend(fontsize=8)

    _save(fig, out_path)


def plot_exp1_phase(df: Any, out_path: Path) -> None:
    """
    Exp1 Panel B phase diagram (Corollary 3.33).

    Use xi/xi_crit on x-axis and P(kappa_g > u0 | Y) on y-axis.
    To avoid floating-point split artifacts (e.g., "1.5" appearing as
    multiple near-identical x values), x is rounded to a fixed display grid.

    Per p_g curve:
      line  = mean across tau
      band  = interquartile range across tau (q25-q75)
    """
    rows = _records(df)
    if not rows:
        return

    from ...utils import load_pandas
    pd = load_pandas()
    frame = pd.DataFrame(rows).copy()
    frame["xi_plot"] = frame["xi_ratio"].astype(float).round(6)
    pg_vals = sorted(frame["p_g"].unique())
    cmap = plt.cm.get_cmap("plasma", len(pg_vals) + 1)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for j, pg in enumerate(pg_vals):
        sub = frame[frame["p_g"] == pg].copy()
        agg = (
            sub.groupby("xi_plot", as_index=False)["mean_prob_kappa_gt_u0"]
            .agg(
                mean="mean",
                q25=lambda s: float(np.quantile(np.asarray(s, dtype=float), 0.25)),
                q75=lambda s: float(np.quantile(np.asarray(s, dtype=float), 0.75)),
            )
            .sort_values("xi_plot")
        )
        color = cmap(j)
        ax.plot(agg["xi_plot"], agg["mean"], "o-", color=color, lw=1.8, ms=5, label=rf"$p_g={int(pg)}$", zorder=3)
        if not (agg["q25"] == agg["q75"]).all():
            ax.fill_between(agg["xi_plot"], agg["q25"], agg["q75"], alpha=0.12, color=color)

    ax.axvline(1.0, color="black", linestyle="--", lw=1.5, label=r"$\xi=\xi_{\mathrm{crit}}$ (theory threshold)")
    ax.axhline(0.5, color="gray", linestyle=":", lw=0.9)
    ax.set_xlabel(_LBL_XI_RATIO, fontsize=10)
    ax.set_ylabel(_LBL_P_KEEP, fontsize=10)
    ax.set_title(
        r"Phase diagram: signal retention (Cor. 3.33)" + "\n" + r"Band = IQR across $\tau$; curves align under $\xi/\xi_{\mathrm{crit}}$ normalization",
        fontsize=9,
    )
    ax.set_ylim(-0.04, 1.08)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    _save(fig, out_path)


def plot_exp1_phase_kappa_overlay(df: Any, out_path: Path) -> None:
    """
    Exp1 Panel C: E[kappa_g | Y] vs xi/xi_crit with Cor 3.18 kappa*(xi) overlay.
    """
    rows = _records(df)
    if not rows:
        return

    from ...utils import load_pandas
    pd = load_pandas()
    frame = pd.DataFrame(rows).copy()
    req = {"p_g", "xi_ratio", "mean_post_mean_kappa", "mean_kappa_star_theory"}
    if not req.issubset(set(frame.columns)):
        return
    frame["xi_plot"] = frame["xi_ratio"].astype(float).round(6)

    pg_vals = sorted(frame["p_g"].unique())
    cmap = plt.cm.get_cmap("viridis", len(pg_vals) + 1)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for j, pg in enumerate(pg_vals):
        sub = frame[frame["p_g"] == pg].copy()
        agg = (
            sub.groupby("xi_plot", as_index=False)["mean_post_mean_kappa"]
            .agg(
                mean="mean",
                q25=lambda s: float(np.quantile(np.asarray(s, dtype=float), 0.25)),
                q75=lambda s: float(np.quantile(np.asarray(s, dtype=float), 0.75)),
            )
            .sort_values("xi_plot")
        )
        color = cmap(j)
        ax.plot(
            agg["xi_plot"],
            agg["mean"],
            "o-",
            color=color,
            lw=1.8,
            ms=5,
            label=rf"Empirical $\mathbb{{E}}[\kappa_g\mid Y]$, $p_g={int(pg)}$",
            zorder=3,
        )
        if not (agg["q25"] == agg["q75"]).all():
            ax.fill_between(agg["xi_plot"], agg["q25"], agg["q75"], alpha=0.10, color=color)

    # Use median across tau to reduce sensitivity to one extreme tau branch.
    theo = frame.groupby("xi_plot")["mean_kappa_star_theory"].median().reset_index().sort_values("xi_plot")
    theo_vals = np.clip(theo["mean_kappa_star_theory"].to_numpy(dtype=float), 0.0, 1.0)
    ax.plot(
        theo["xi_plot"],
        theo_vals,
        "--",
        color="black",
        lw=2.0,
        label=r"Theory $\kappa^\star(\xi)$ (Cor 3.18)",
        zorder=4,
    )
    ax.axvline(1.0, color="gray", linestyle=":", lw=1.2)
    ax.set_xlabel(_LBL_XI_RATIO, fontsize=10)
    ax.set_ylabel(_LBL_E_KAPPA, fontsize=10)
    ax.set_title(r"Exp1 phase overlay: empirical $\mathbb{E}[\kappa_g\mid Y]$ vs theory $\kappa^\star(\xi)$", fontsize=9)
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    _save(fig, out_path)


def _exp1_representative_pg(pg_vals: list[int]) -> list[int]:
    vals = sorted(int(v) for v in pg_vals)
    if len(vals) <= 3:
        return vals
    return [vals[0], vals[len(vals) // 2], vals[-1]]


def plot_exp1_phase_readable(
    df: Any,
    out_path: Path,
    *,
    u0: float = 0.5,
    representative_pg: list[int] | None = None,
) -> None:
    """
    Exp1 readable phase plot:
      - show only representative p_g curves (small/medium/large)
      - keep physical x-axis (xi/xi_crit)
      - use IQR band across tau
    """
    rows = _records(df)
    if not rows:
        return

    from ...utils import load_pandas
    pd = load_pandas()
    frame = pd.DataFrame(rows).copy()
    req = {"p_g", "xi_ratio", "mean_prob_kappa_gt_u0"}
    if not req.issubset(set(frame.columns)):
        return

    frame["xi_plot"] = frame["xi_ratio"].astype(float).round(6)
    all_pg = sorted(int(v) for v in frame["p_g"].unique())
    reps = sorted(int(v) for v in (representative_pg or _exp1_representative_pg(all_pg)))

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    xi_vals = sorted(float(v) for v in frame["xi_plot"].unique())
    if not xi_vals:
        return
    xmin, xmax = float(min(xi_vals)), float(max(xi_vals))
    ax.axvspan(xmin, 1.0, alpha=0.06, color="#b0bec5")
    ax.axvspan(1.0, xmax, alpha=0.06, color="#c8e6c9")
    ax.axvline(1.0, color="black", linestyle="--", lw=1.4, label=r"$\xi=\xi_{\mathrm{crit}}$")
    ax.axhline(float(u0), color="gray", linestyle=":", lw=1.0, label=rf"$u_0={float(u0):.2f}$")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, pg in enumerate(reps):
        sub = frame[frame["p_g"].astype(int) == int(pg)].copy()
        if sub.empty:
            continue
        agg = (
            sub.groupby("xi_plot", as_index=False)["mean_prob_kappa_gt_u0"]
            .agg(
                mean="mean",
                q25=lambda s: float(np.quantile(np.asarray(s, dtype=float), 0.25)),
                q75=lambda s: float(np.quantile(np.asarray(s, dtype=float), 0.75)),
            )
            .sort_values("xi_plot")
        )
        c = colors[i % len(colors)]
        ax.plot(
            agg["xi_plot"],
            agg["mean"],
            "o-",
            color=c,
            lw=2.0,
            ms=6,
            label=rf"$p_g={int(pg)}$",
            zorder=3,
        )
        ax.fill_between(agg["xi_plot"], agg["q25"], agg["q75"], color=c, alpha=0.14)

    ax.set_xlabel(_LBL_XI_RATIO)
    ax.set_ylabel(_LBL_P_KEEP)
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(r"Exp1 readable phase view: representative $p_g$ + uncertainty")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(fontsize=8, loc="upper left")
    _save(fig, out_path)


def plot_exp1_kappa_residual_readable(
    df: Any,
    out_path: Path,
    *,
    representative_pg: list[int] | None = None,
) -> None:
    """
    Residual view for Exp1:
      residual = empirical E[kappa_g|Y] - theory kappa*(xi)
    This is usually easier to read than direct overlays when lines are close.
    """
    rows = _records(df)
    if not rows:
        return

    from ...utils import load_pandas
    pd = load_pandas()
    frame = pd.DataFrame(rows).copy()
    req = {"p_g", "xi_ratio", "mean_post_mean_kappa", "mean_kappa_star_theory"}
    if not req.issubset(set(frame.columns)):
        return

    frame["xi_plot"] = frame["xi_ratio"].astype(float).round(6)
    theo = (
        frame.groupby("xi_plot", as_index=False)["mean_kappa_star_theory"]
        .median()
        .rename(columns={"mean_kappa_star_theory": "kappa_theory"})
    )
    theo["kappa_theory"] = np.clip(theo["kappa_theory"].to_numpy(dtype=float), 0.0, 1.0)

    all_pg = sorted(int(v) for v in frame["p_g"].unique())
    reps = sorted(int(v) for v in (representative_pg or _exp1_representative_pg(all_pg)))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.axhline(0.0, color="black", lw=1.2, ls="--")
    ax.axhspan(-0.05, 0.05, color="#eeeeee", alpha=0.7, zorder=0, label=r"near-theory band ($\pm 0.05$)")
    ax.axvline(1.0, color="gray", linestyle=":", lw=1.2)

    for i, pg in enumerate(reps):
        sub = frame[frame["p_g"].astype(int) == int(pg)].copy()
        if sub.empty:
            continue
        agg = (
            sub.groupby("xi_plot", as_index=False)["mean_post_mean_kappa"]
            .agg(
                mean="mean",
                q25=lambda s: float(np.quantile(np.asarray(s, dtype=float), 0.25)),
                q75=lambda s: float(np.quantile(np.asarray(s, dtype=float), 0.75)),
            )
            .sort_values("xi_plot")
        )
        merged = agg.merge(theo, on="xi_plot", how="left")
        resid = merged["mean"].to_numpy(dtype=float) - merged["kappa_theory"].to_numpy(dtype=float)
        resid_lo = merged["q25"].to_numpy(dtype=float) - merged["kappa_theory"].to_numpy(dtype=float)
        resid_hi = merged["q75"].to_numpy(dtype=float) - merged["kappa_theory"].to_numpy(dtype=float)
        c = colors[i % len(colors)]
        ax.plot(merged["xi_plot"], resid, "o-", color=c, lw=2.0, ms=6, label=rf"$p_g={int(pg)}$")
        ax.fill_between(merged["xi_plot"], resid_lo, resid_hi, color=c, alpha=0.14)

    ax.set_xlabel(_LBL_XI_RATIO)
    ax.set_ylabel(r"$\mathbb{E}[\kappa_g\mid Y]-\kappa^\star(\xi)$")
    ax.set_title(r"Exp1 residual view: empirical minus theory")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(fontsize=8, loc="upper left")
    _save(fig, out_path)


def plot_exp1_phase_heatmap_readable(df: Any, out_path: Path) -> None:
    """
    Heatmap supplement for Exp1:
      x = xi/xi_crit bins (display grid)
      y = p_g
      color = mean P(kappa_g > u0 | Y), averaged over tau.
    """
    rows = _records(df)
    if not rows:
        return

    from ...utils import load_pandas
    pd = load_pandas()
    frame = pd.DataFrame(rows).copy()
    req = {"p_g", "xi_ratio", "mean_prob_kappa_gt_u0"}
    if not req.issubset(set(frame.columns)):
        return

    frame["xi_plot"] = frame["xi_ratio"].astype(float).round(6)
    work = (
        frame.groupby(["p_g", "xi_plot"], as_index=False)["mean_prob_kappa_gt_u0"]
        .mean()
        .rename(columns={"mean_prob_kappa_gt_u0": "prob"})
    )
    piv = work.pivot(index="p_g", columns="xi_plot", values="prob").sort_index().sort_index(axis=1)
    if piv.empty:
        return

    z = piv.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    im = ax.imshow(z, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0, origin="lower")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\mathbb{E}_{\tau}[\mathbb{P}(\kappa_g>u_0\mid Y)]$")

    xi_cols = [float(v) for v in piv.columns.tolist()]
    pg_rows = [int(v) for v in piv.index.tolist()]
    ax.set_xticks(np.arange(len(xi_cols)))
    ax.set_xticklabels([f"{x:.2f}".rstrip("0").rstrip(".") for x in xi_cols], rotation=0, fontsize=8)
    ax.set_yticks(np.arange(len(pg_rows)))
    ax.set_yticklabels([str(v) for v in pg_rows], fontsize=9)
    ax.set_xlabel(_LBL_XI_RATIO + " (display bins)")
    ax.set_ylabel(_LBL_PG)
    ax.set_title(r"Exp1 heatmap supplement: retention across $(\xi/\xi_{\mathrm{crit}}, p_g)$")

    if xi_cols:
        xi_arr = np.asarray(xi_cols, dtype=float)
        left = np.where(xi_arr <= 1.0)[0]
        right = np.where(xi_arr > 1.0)[0]
        if left.size and right.size:
            x_thr = 0.5 * (float(left.max()) + float(right.min()))
        else:
            x_thr = float(np.argmin(np.abs(xi_arr - 1.0)))
        ax.axvline(x_thr, color="white", lw=1.5, ls="--")

    _save(fig, out_path)


def plot_exp1_phase_by_tau_readable(
    df: Any,
    out_path: Path,
    *,
    u0: float = 0.5,
    representative_pg: list[int] | None = None,
) -> None:
    """
    Faceted phase plot by tau:
      - each panel is one tau
      - curves are representative p_g values
      - avoids cross-tau averaging that can smooth the transition near xi/xi_crit=1
    """
    rows = _records(df)
    if not rows:
        return

    from ...utils import load_pandas
    pd = load_pandas()
    frame = pd.DataFrame(rows).copy()
    req = {"tau", "p_g", "xi_ratio", "mean_prob_kappa_gt_u0"}
    if not req.issubset(set(frame.columns)):
        return

    frame["xi_plot"] = frame["xi_ratio"].astype(float).round(6)
    tau_vals = sorted(float(v) for v in frame["tau"].unique())
    if not tau_vals:
        return
    all_pg = sorted(int(v) for v in frame["p_g"].unique())
    reps = sorted(int(v) for v in (representative_pg or _exp1_representative_pg(all_pg)))

    # Glance style: keep only smallest/largest p_g to make trend obvious at first sight.
    if representative_pg is None:
        reps = [all_pg[0], all_pg[-1]] if len(all_pg) >= 2 else all_pg
    else:
        reps = sorted(int(v) for v in representative_pg)

    n = len(tau_vals)
    ncols = 2 if n > 1 else 1
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.6 * ncols, 4.9 * nrows), squeeze=False)
    colors = ["#1f77b4", "#d62728"]

    legend_added = False
    for i, tau in enumerate(tau_vals):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        sub_tau = frame[np.isclose(frame["tau"].astype(float), float(tau))].copy()
        xi_vals = sorted(float(v) for v in sub_tau["xi_plot"].unique())
        if xi_vals:
            ax.axvspan(min(xi_vals), 1.0, alpha=0.08, color="#eceff1")
            ax.axvspan(1.0, max(xi_vals), alpha=0.08, color="#e8f5e9")
        ax.axvline(1.0, color="black", lw=2.0, ls="--")
        ax.axhline(float(u0), color="#666666", lw=1.2, ls=":")

        anno_lines: list[str] = []
        for j, pg in enumerate(reps):
            sub = sub_tau[sub_tau["p_g"].astype(int) == int(pg)].copy()
            if sub.empty:
                continue
            agg = (
                sub.groupby("xi_plot", as_index=False)["mean_prob_kappa_gt_u0"]
                .mean()
                .sort_values("xi_plot")
            )
            ax.plot(
                agg["xi_plot"],
                agg["mean_prob_kappa_gt_u0"],
                "o-",
                color=colors[j % len(colors)],
                lw=3.0,
                ms=7.2,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=rf"$p_g={int(pg)}$",
                zorder=3,
            )
            try:
                p_lo = float(agg.loc[np.isclose(agg["xi_plot"], 0.85), "mean_prob_kappa_gt_u0"].iloc[0])
                p_hi = float(agg.loc[np.isclose(agg["xi_plot"], 1.15), "mean_prob_kappa_gt_u0"].iloc[0])
                anno_lines.append(rf"$p_g={int(pg)}$: $\Delta_{{\mathrm{{wide}}}}={p_hi - p_lo:+.2f}$")
            except Exception:
                pass

        ax.set_ylim(-0.03, 1.05)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticks([0.5, 0.85, 1.0, 1.15, 1.5, 2.0])
        ax.set_title(rf"$\tau={tau:.2g}$", fontsize=12, fontweight="semibold")
        ax.set_xlabel(_LBL_XI_RATIO, fontsize=11)
        ax.set_ylabel(_LBL_P_KEEP, fontsize=11)
        ax.tick_params(labelsize=10)
        ax.grid(axis="y", alpha=0.18)
        if anno_lines:
            ax.text(
                0.03,
                0.06,
                "\n".join(anno_lines),
                transform=ax.transAxes,
                fontsize=9,
                color="#222222",
                bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#cccccc", alpha=0.92),
            )
        if not legend_added:
            ax.legend(fontsize=9, title=r"Only min/max $p_g$", title_fontsize=9, loc="upper left", framealpha=0.96)
            legend_added = True

    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(r"Exp1 Main Phase (Glance View): fixed $\tau$, compare smallest vs largest $p_g$", fontsize=15, fontweight="bold")
    _save(fig, out_path)


def plot_exp1_phase_zoom_by_tau_readable(
    df: Any,
    out_path: Path,
    *,
    u0: float = 0.5,
    representative_pg: list[int] | None = None,
    x_min: float = 0.85,
    x_max: float = 1.15,
) -> None:
    """
    Faceted phase plot by tau, zoomed around the threshold x=1.
    """
    rows = _records(df)
    if not rows:
        return

    from ...utils import load_pandas
    pd = load_pandas()
    frame = pd.DataFrame(rows).copy()
    req = {"tau", "p_g", "xi_ratio", "mean_prob_kappa_gt_u0"}
    if not req.issubset(set(frame.columns)):
        return

    xmin = float(x_min)
    xmax = float(x_max)
    if not (xmin < 1.0 < xmax):
        return

    frame["xi_plot"] = frame["xi_ratio"].astype(float).round(6)
    frame = frame[(frame["xi_plot"] >= xmin) & (frame["xi_plot"] <= xmax)].copy()
    if frame.empty:
        return

    tau_vals = sorted(float(v) for v in frame["tau"].unique())
    if not tau_vals:
        return
    all_pg = sorted(int(v) for v in frame["p_g"].unique())
    reps = sorted(int(v) for v in (representative_pg or _exp1_representative_pg(all_pg)))

    if representative_pg is None:
        reps = [all_pg[0], all_pg[-1]] if len(all_pg) >= 2 else all_pg
    else:
        reps = sorted(int(v) for v in representative_pg)

    n = len(tau_vals)
    ncols = 2 if n > 1 else 1
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.6 * ncols, 4.9 * nrows), squeeze=False)
    colors = ["#1f77b4", "#d62728"]

    legend_added = False
    zoom_ticks = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
    zoom_ticks = [x for x in zoom_ticks if xmin <= x <= xmax]

    for i, tau in enumerate(tau_vals):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        sub_tau = frame[np.isclose(frame["tau"].astype(float), float(tau))].copy()
        ax.axvspan(xmin, 1.0, alpha=0.09, color="#eceff1")
        ax.axvspan(1.0, xmax, alpha=0.09, color="#e8f5e9")
        ax.axvline(1.0, color="black", lw=2.0, ls="--")
        ax.axhline(float(u0), color="#666666", lw=1.2, ls=":")

        anno_lines: list[str] = []
        for j, pg in enumerate(reps):
            sub = sub_tau[sub_tau["p_g"].astype(int) == int(pg)].copy()
            if sub.empty:
                continue
            agg = (
                sub.groupby("xi_plot", as_index=False)["mean_prob_kappa_gt_u0"]
                .mean()
                .sort_values("xi_plot")
            )
            ax.plot(
                agg["xi_plot"],
                agg["mean_prob_kappa_gt_u0"],
                "o-",
                color=colors[j % len(colors)],
                lw=3.0,
                ms=7.2,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=rf"$p_g={int(pg)}$",
                zorder=3,
            )
            try:
                p_l = float(agg.loc[np.isclose(agg["xi_plot"], 0.95), "mean_prob_kappa_gt_u0"].iloc[0])
                p_r = float(agg.loc[np.isclose(agg["xi_plot"], 1.05), "mean_prob_kappa_gt_u0"].iloc[0])
                anno_lines.append(rf"$p_g={int(pg)}$: $\Delta_{{\mathrm{{local}}}}={p_r - p_l:+.2f}$")
            except Exception:
                pass

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-0.03, 1.05)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticks(zoom_ticks)
        ax.set_title(rf"$\tau={tau:.2g}$", fontsize=12, fontweight="semibold")
        ax.set_xlabel(_LBL_XI_RATIO, fontsize=11)
        ax.set_ylabel(_LBL_P_KEEP, fontsize=11)
        ax.tick_params(labelsize=10)
        ax.grid(axis="y", alpha=0.18)
        if anno_lines:
            ax.text(
                0.03,
                0.06,
                "\n".join(anno_lines),
                transform=ax.transAxes,
                fontsize=9,
                color="#222222",
                bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#cccccc", alpha=0.92),
            )
        if not legend_added:
            ax.legend(fontsize=9, title=r"Only min/max $p_g$", title_fontsize=9, loc="upper left", framealpha=0.96)
            legend_added = True

    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(r"Exp1 Threshold Zoom (Glance View): $x\in[0.85,1.15]$ with $\Delta_{\mathrm{local}}$ labels", fontsize=15, fontweight="bold")
    _save(fig, out_path)


def plot_exp1_threshold_sharpness_readable(
    df: Any,
    out_path: Path,
    *,
    local_left: float = 0.95,
    local_right: float = 1.05,
    wide_left: float = 0.85,
    wide_right: float = 1.15,
) -> None:
    """
    Threshold sharpness diagnostics with both local and wide deltas:
      Delta_local = P(x=local_right) - P(x=local_left)
      Delta_wide  = P(x=wide_right)  - P(x=wide_left)
    """
    rows = _records(df)
    if not rows:
        return

    from ...utils import load_pandas
    pd = load_pandas()
    frame = pd.DataFrame(rows).copy()
    req = {"tau", "p_g", "xi_ratio", "mean_prob_kappa_gt_u0"}
    if not req.issubset(set(frame.columns)):
        return

    frame["xi_plot"] = frame["xi_ratio"].astype(float).round(6)
    work = (
        frame.groupby(["tau", "p_g", "xi_plot"], as_index=False)["mean_prob_kappa_gt_u0"]
        .mean()
        .rename(columns={"mean_prob_kappa_gt_u0": "prob"})
    )
    wide = work.pivot_table(
        index=["tau", "p_g"],
        columns="xi_plot",
        values="prob",
        aggfunc="mean",
    )
    if wide.empty:
        return

    ll = float(local_left)
    lr = float(local_right)
    wl = float(wide_left)
    wr = float(wide_right)

    rows_delta: list[dict[str, Any]] = []
    for (tau, pg), row in wide.iterrows():
        if ll in wide.columns and lr in wide.columns:
            dl = float(row[lr] - row[ll])
            if np.isfinite(dl):
                rows_delta.append({"tau": float(tau), "p_g": int(pg), "kind": "local", "delta": dl})
        if wl in wide.columns and wr in wide.columns:
            dw = float(row[wr] - row[wl])
            if np.isfinite(dw):
                rows_delta.append({"tau": float(tau), "p_g": int(pg), "kind": "wide", "delta": dw})

    if not rows_delta:
        return
    delta = pd.DataFrame(rows_delta)

    fig, ax = plt.subplots(figsize=(8.8, 5.3))

    # Glance style: one panel, tau-averaged local/wide deltas.
    palette = {"local": "#1f77b4", "wide": "#ff7f0e"}
    labels = {
        "local": rf"$\Delta_{{\mathrm{{local}}}}=\mathbb{{P}}(x={lr:.2f})-\mathbb{{P}}(x={ll:.2f})$",
        "wide": rf"$\Delta_{{\mathrm{{wide}}}}=\mathbb{{P}}(x={wr:.2f})-\mathbb{{P}}(x={wl:.2f})$",
    }
    end_texts: list[str] = []
    for kind in ["local", "wide"]:
        sub = delta[delta["kind"] == kind].copy()
        if sub.empty:
            continue
        agg = (
            sub.groupby("p_g", as_index=False)["delta"]
            .agg(mean="mean", min="min", max="max")
            .sort_values("p_g")
        )
        x = agg["p_g"].to_numpy(dtype=float)
        y = agg["mean"].to_numpy(dtype=float)
        ylo = agg["min"].to_numpy(dtype=float)
        yhi = agg["max"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            "o-",
            color=palette[kind],
            lw=2.6,
            ms=6.8,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=labels[kind],
        )
        ax.fill_between(x, ylo, yhi, color=palette[kind], alpha=0.14)
        end_texts.append(rf"{kind}: {y[0]:.2f} \rightarrow {y[-1]:.2f}")
    ax.axhline(0.0, color="black", lw=1.2, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"$p_g$ (log scale)", fontsize=11)
    ax.set_ylabel(r"$\Delta\mathbb{P}$", fontsize=11)
    ax.set_title("Threshold Sharpness Summary", fontsize=13, fontweight="semibold")
    ax.tick_params(labelsize=10)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.95)
    if end_texts:
        ax.text(
            0.03,
            0.06,
            " | ".join(end_texts),
            transform=ax.transAxes,
            fontsize=9,
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#cccccc", alpha=0.92),
        )
    ax.annotate(
        r"Larger $\Delta\mathbb{P}$ = sharper transition",
        xy=(0.87, 0.82),
        xytext=(0.55, 0.67),
        textcoords="axes fraction",
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=1.4, color="#444444"),
        fontsize=10,
        color="#333333",
    )

    fig.suptitle(r"Exp1 Sharpness (Glance View): $\Delta_{\mathrm{local}}$ and $\Delta_{\mathrm{wide}}$ rise with $p_g$", fontsize=15, fontweight="bold")
    _save(fig, out_path)


def plot_exp1_threshold_jump_readable(
    df: Any,
    out_path: Path,
    *,
    xi_left: float = 0.95,
    xi_right: float = 1.05,
) -> None:
    """
    Local threshold-jump diagnostic:
      delta = P(kappa_g > u0 | x=xi_right) - P(kappa_g > u0 | x=xi_left)
    reported by p_g and by tau.
    """
    rows = _records(df)
    if not rows:
        return

    from ...utils import load_pandas
    pd = load_pandas()
    frame = pd.DataFrame(rows).copy()
    req = {"tau", "p_g", "xi_ratio", "mean_prob_kappa_gt_u0"}
    if not req.issubset(set(frame.columns)):
        return

    xl = float(xi_left)
    xr = float(xi_right)
    frame["xi_plot"] = frame["xi_ratio"].astype(float).round(6)

    work = (
        frame.groupby(["tau", "p_g", "xi_plot"], as_index=False)["mean_prob_kappa_gt_u0"]
        .mean()
        .rename(columns={"mean_prob_kappa_gt_u0": "prob"})
    )
    wide = work.pivot_table(
        index=["tau", "p_g"],
        columns="xi_plot",
        values="prob",
        aggfunc="mean",
    )
    if xl not in wide.columns or xr not in wide.columns:
        return
    delta = (wide[xr] - wide[xl]).reset_index(name="delta")
    if delta.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.3))

    # Left panel: tau-averaged jump with min-max band across tau.
    ax = axes[0]
    agg = (
        delta.groupby("p_g", as_index=False)["delta"]
        .agg(mean="mean", min="min", max="max")
        .sort_values("p_g")
    )
    x = agg["p_g"].to_numpy(dtype=float)
    y = agg["mean"].to_numpy(dtype=float)
    ylo = agg["min"].to_numpy(dtype=float)
    yhi = agg["max"].to_numpy(dtype=float)
    ax.plot(x, y, "o-", color="#1f77b4", lw=2.2, ms=6, label=r"$\mathrm{mean}_{\tau}$")
    ax.fill_between(x, ylo, yhi, color="#1f77b4", alpha=0.16, label=r"$[\min_{\tau},\max_{\tau}]$")
    ax.axhline(0.0, color="black", lw=1.1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"$p_g$ (log scale)")
    ax.set_ylabel(rf"$\Delta\mathbb{{P}}=\mathbb{{P}}(x={xr:.2f})-\mathbb{{P}}(x={xl:.2f})$")
    ax.set_title(r"Local jump around threshold ($\tau$-averaged)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8, loc="upper left")

    # Right panel: per-tau lines.
    ax = axes[1]
    tau_vals = sorted(float(v) for v in delta["tau"].unique())
    cmap = plt.cm.get_cmap("tab10", len(tau_vals) + 1)
    for i, tau in enumerate(tau_vals):
        sub = delta[np.isclose(delta["tau"].astype(float), float(tau))].sort_values("p_g")
        ax.plot(
            sub["p_g"].to_numpy(dtype=float),
            sub["delta"].to_numpy(dtype=float),
            "o-",
            color=cmap(i),
            lw=1.9,
            ms=5,
            label=rf"$\tau={tau:.2g}$",
        )
    ax.axhline(0.0, color="black", lw=1.1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"$p_g$ (log scale)")
    ax.set_ylabel(rf"$\Delta\mathbb{{P}}=\mathbb{{P}}(x={xr:.2f})-\mathbb{{P}}(x={xl:.2f})$")
    ax.set_title(r"Local jump around threshold (by $\tau$)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        rf"Exp1 threshold diagnostic: local jump from $x={xl:.2f}$ to $x={xr:.2f}$",
        fontsize=11,
        y=1.02,
    )
    _save(fig, out_path)


def _exp1_interp_x_at_prob(x: np.ndarray, p: np.ndarray, target: float) -> float:
    """
    Invert a monotone response curve p(x) by linear interpolation.
    Returns NaN if target is outside the observed probability range.
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(x) & np.isfinite(p)
    if int(np.sum(m)) < 2:
        return float("nan")

    x = x[m]
    p = p[m]
    order = np.argsort(x)
    x = x[order]
    p = np.clip(p[order], 0.0, 1.0)

    # Enforce the expected monotone shape to stabilize finite-sample wiggles.
    p = np.maximum.accumulate(p)

    t = float(target)
    if t < float(p[0]) or t > float(p[-1]):
        return float("nan")

    idx = int(np.searchsorted(p, t, side="left"))
    if idx <= 0:
        return float(x[0])
    if idx >= len(p):
        return float(x[-1])

    x0, x1 = float(x[idx - 1]), float(x[idx])
    p0, p1 = float(p[idx - 1]), float(p[idx])
    if abs(p1 - p0) < 1e-12:
        return float(x1)
    w = (t - p0) / (p1 - p0)
    return float(x0 + w * (x1 - x0))


def plot_exp1_posterior_density_main(
    df: Any,
    out_path: Path,
    *,
    xi_ratio_order: Sequence[float] | None = None,
    y_mode: str = "density",
    log_floor: float = 1e-4,
) -> None:
    """
    Main-text Exp1 figure:
      x = kappa_g, y = p(kappa_g | Y)
      Facet by xi/xi_crit in {0.5, 1.0, 1.5},
      with black-and-white line-style encoding for p_g values.
      Style target: classic paper figure (no bright colors).
    """
    rows = _records(df)
    if not rows:
        return

    from ...utils import load_pandas
    pd = load_pandas()
    frame = pd.DataFrame(rows).copy()
    req = {"p_g", "xi_ratio", "kappa", "density"}
    if not req.issubset(set(frame.columns)):
        return
    mode = str(y_mode).strip().lower()
    if mode not in {"density", "log_density", "relative_density"}:
        mode = "density"

    frame["p_g"] = frame["p_g"].astype(int)
    frame["xi_plot"] = frame["xi_ratio"].astype(float).round(6)
    frame["kappa"] = frame["kappa"].astype(float)
    frame["density"] = frame["density"].astype(float)
    frame = frame[np.isfinite(frame["kappa"]) & np.isfinite(frame["density"])]
    if frame.empty:
        return

    all_pg = sorted(int(v) for v in frame["p_g"].unique())
    if not all_pg:
        return

    ratio_targets = [float(v) for v in (xi_ratio_order or [0.5, 1.0, 1.5])]
    ncols = len(ratio_targets)
    fig, axes = plt.subplots(1, ncols, figsize=(5.6 * ncols, 4.9), sharey=False)
    if ncols == 1:
        axes = [axes]

    line_styles = [
        "-",
        "--",
        "-.",
        ":",
        (0, (5, 1)),
        (0, (3, 1, 1, 1)),
        (0, (1, 1)),
    ]
    pg_styles = {
        int(pg): {
            "color": "black",
            "ls": line_styles[idx % len(line_styles)],
            "lw": 2.1 if idx == 0 else 1.9,
            "label": rf"$p_g={int(pg)}$",
        }
        for idx, pg in enumerate(all_pg)
    }

    avail_global = sorted(float(v) for v in frame["xi_plot"].unique())
    if not avail_global:
        return
    avail_arr_global = np.asarray(avail_global, dtype=float)

    with plt.rc_context({"font.family": "serif"}):
        for panel_idx, (ax, target) in enumerate(zip(axes, ratio_targets), start=1):
            idx = int(np.argmin(np.abs(avail_arr_global - float(target))))
            chosen = float(avail_arr_global[idx])
            if abs(chosen - float(target)) > 0.03:
                continue

            any_curve = False
            local_peak = 0.0
            local_min = float("inf")
            local_max = float("-inf")
            for pg in all_pg:
                sub_pg = frame[(frame["p_g"] == int(pg)) & np.isclose(frame["xi_plot"], chosen, atol=1e-9)].copy()
                if sub_pg.empty:
                    continue

                agg = (
                    sub_pg.groupby("kappa", as_index=False)["density"]
                    .mean()
                    .sort_values("kappa")
                )
                x = agg["kappa"].to_numpy(dtype=float)
                y = agg["density"].to_numpy(dtype=float)
                if x.size < 2:
                    continue
                area = float(np.trapezoid(y, x))
                if area > 0:
                    y = y / area
                y_plot = y.copy()
                if mode == "relative_density":
                    peak = float(np.nanmax(y_plot))
                    if peak > 0:
                        y_plot = y_plot / peak
                elif mode == "log_density":
                    y_plot = np.log10(np.maximum(y_plot, float(log_floor)))

                style = pg_styles[int(pg)]
                ax.plot(
                    x,
                    y_plot,
                    linestyle=style["ls"],
                    lw=float(style["lw"]),
                    color=str(style["color"]),
                    label=str(style["label"]),
                )
                if mode == "log_density":
                    local_min = min(local_min, float(np.nanmin(y_plot)))
                    local_max = max(local_max, float(np.nanmax(y_plot)))
                else:
                    local_peak = max(local_peak, float(np.nanmax(y_plot)))
                any_curve = True

            if not any_curve:
                continue

            panel_tag = chr(ord("a") + panel_idx - 1)
            status = (
                r"$\xi/\xi_{\mathrm{crit}}<1$"
                if float(target) < 1.0
                else (r"$\xi/\xi_{\mathrm{crit}}>1$" if float(target) > 1.0 else r"$\xi/\xi_{\mathrm{crit}}=1$")
            )
            ax.set_xlim(0.0, 1.0)
            if mode == "log_density":
                span = max(local_max - local_min, 0.5)
                ax.set_ylim(local_min - 0.06 * span, local_max + 0.10 * span)
            else:
                ax.set_ylim(0.0, local_peak * 1.08 if local_peak > 0 else 1.0)
            ax.set_xlabel(_LBL_KAPPA, fontsize=12)
            ax.set_title(rf"({panel_tag})  {status}", fontsize=12)
            ax.tick_params(axis="both", labelsize=10, length=5)
            ax.legend(
                fontsize=8.5,
                frameon=False,
                loc="upper right",
                ncol=2,
                handlelength=1.8,
                columnspacing=0.8,
            )

        if mode == "density":
            y_label = r"$p(\kappa_g\mid Y)$"
        elif mode == "relative_density":
            y_label = r"$p(\kappa_g\mid Y)\,/\,\max p$"
        else:
            y_label = r"$\log_{10} p(\kappa_g\mid Y)$"
        axes[0].set_ylabel(y_label, fontsize=12)

    tau_note = ""
    if "tau" in frame.columns:
        tau_vals = sorted(float(v) for v in frame["tau"].dropna().unique())
        if len(tau_vals) == 1:
            tau_note = rf" ($\tau={tau_vals[0]:.2f}$)"
    pg_note = rf", $p_g\in[{int(all_pg[0])},{int(all_pg[-1])}]$"
    mode_note = {
        "density": r"density scale",
        "relative_density": r"peak-normalized scale",
        "log_density": r"$\log_{10}$ density scale",
    }[mode]
    fig.suptitle(
        r"Exp1: posterior density $p(\kappa_g\mid Y)$ in $\kappa_g$ space with varying $p_g$"
        + tau_note
        + pg_note
        + f" [{mode_note}]",
        fontsize=13,
    )
    _save(fig, out_path)


def _posterior_quantile_from_density(x: np.ndarray, d: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    d = np.asarray(d, dtype=float)
    m = np.isfinite(x) & np.isfinite(d)
    if int(np.sum(m)) < 2:
        return float("nan")
    x = x[m]
    d = d[m]
    order = np.argsort(x)
    x = x[order]
    d = np.maximum(d[order], 0.0)
    area = float(np.trapezoid(d, x))
    if not np.isfinite(area) or area <= 0:
        return float("nan")
    d = d / area
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(x))])
    cdf = cdf / max(float(cdf[-1]), 1e-12)
    return float(np.interp(float(q), cdf, x))


def plot_exp1_posterior_density_heatmap(
    df: Any,
    out_path: Path,
    *,
    xi_ratio_order: Sequence[float] | None = None,
    log_floor: float = 1e-5,
) -> None:
    """
    Non-overlapping Exp1 density view:
      x = kappa_g, y = p_g, color = log10 p(kappa_g | Y), faceted by xi/xi_crit.
    Includes posterior median and IQR ribbons as white guide lines.
    """
    rows = _records(df)
    if not rows:
        return

    from ...utils import load_pandas
    pd = load_pandas()
    frame = pd.DataFrame(rows).copy()
    req = {"p_g", "xi_ratio", "kappa", "density"}
    if not req.issubset(set(frame.columns)):
        return

    frame["p_g"] = frame["p_g"].astype(int)
    frame["xi_plot"] = frame["xi_ratio"].astype(float).round(6)
    frame["kappa"] = frame["kappa"].astype(float)
    frame["density"] = frame["density"].astype(float)
    frame = frame[np.isfinite(frame["kappa"]) & np.isfinite(frame["density"])]
    if frame.empty:
        return

    all_pg = sorted(int(v) for v in frame["p_g"].unique())
    if not all_pg:
        return
    all_kappa = np.asarray(sorted(float(v) for v in frame["kappa"].unique()), dtype=float)
    if all_kappa.size < 2:
        return

    ratio_targets = [float(v) for v in (xi_ratio_order or [0.5, 1.0, 1.5])]
    ncols = len(ratio_targets)
    fig, axes = plt.subplots(1, ncols, figsize=(5.6 * ncols, 4.9), sharey=True)
    if ncols == 1:
        axes = [axes]

    avail_global = np.asarray(sorted(float(v) for v in frame["xi_plot"].unique()), dtype=float)
    if avail_global.size == 0:
        return

    image_handle = None
    with plt.rc_context({"font.family": "serif"}):
        for panel_idx, (ax, target) in enumerate(zip(axes, ratio_targets), start=1):
            chosen = float(avail_global[int(np.argmin(np.abs(avail_global - float(target))))])
            if abs(chosen - float(target)) > 0.03:
                continue
            sub_ratio = frame[np.isclose(frame["xi_plot"], chosen, atol=1e-9)].copy()
            if sub_ratio.empty:
                continue

            z = np.full((len(all_pg), all_kappa.size), np.nan, dtype=float)
            med = np.full(len(all_pg), np.nan, dtype=float)
            q25 = np.full(len(all_pg), np.nan, dtype=float)
            q75 = np.full(len(all_pg), np.nan, dtype=float)

            for row_idx, pg in enumerate(all_pg):
                sub = (
                    sub_ratio[sub_ratio["p_g"] == int(pg)]
                    .groupby("kappa", as_index=False)["density"]
                    .mean()
                    .sort_values("kappa")
                )
                if sub.empty:
                    continue
                x = sub["kappa"].to_numpy(dtype=float)
                y = sub["density"].to_numpy(dtype=float)
                if x.size < 2:
                    continue
                y_interp = np.interp(all_kappa, x, y, left=0.0, right=0.0)
                area = float(np.trapezoid(y_interp, all_kappa))
                if area > 0:
                    y_interp = y_interp / area
                z[row_idx, :] = y_interp
                med[row_idx] = _posterior_quantile_from_density(all_kappa, y_interp, 0.50)
                q25[row_idx] = _posterior_quantile_from_density(all_kappa, y_interp, 0.25)
                q75[row_idx] = _posterior_quantile_from_density(all_kappa, y_interp, 0.75)

            z_plot = np.log10(np.maximum(z, float(log_floor)))
            image_handle = ax.imshow(
                z_plot,
                origin="lower",
                aspect="auto",
                cmap="viridis",
                interpolation="nearest",
            )

            y_rows = np.arange(len(all_pg), dtype=float)
            if np.any(np.isfinite(med)):
                ax.plot(med, y_rows, color="white", lw=1.8, label=r"$\mathrm{median}$")
            if np.any(np.isfinite(q25)) and np.any(np.isfinite(q75)):
                ax.plot(q25, y_rows, color="white", lw=1.0, ls="--", alpha=0.85, label=r"$\mathrm{IQR}$")
                ax.plot(q75, y_rows, color="white", lw=1.0, ls="--", alpha=0.85)

            xt_vals = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
            xt_pos = np.interp(xt_vals, all_kappa, np.arange(all_kappa.size, dtype=float))
            ax.set_xticks(xt_pos)
            ax.set_xticklabels([f"{v:.2g}" for v in xt_vals], fontsize=10)

            ax.set_yticks(np.arange(len(all_pg), dtype=float))
            ax.set_yticklabels([str(v) for v in all_pg], fontsize=10)
            ax.set_xlabel(_LBL_KAPPA, fontsize=12)
            panel_tag = chr(ord("a") + panel_idx - 1)
            status = (
                r"$\xi/\xi_{\mathrm{crit}}<1$"
                if float(target) < 1.0
                else (r"$\xi/\xi_{\mathrm{crit}}>1$" if float(target) > 1.0 else r"$\xi/\xi_{\mathrm{crit}}=1$")
            )
            ax.set_title(rf"({panel_tag})  {status}", fontsize=12)
            ax.tick_params(axis="both", length=4)
            if panel_idx == ncols:
                ax.legend(fontsize=8.5, frameon=False, loc="upper right")

        axes[0].set_ylabel(_LBL_PG, fontsize=12)

    if image_handle is not None:
        cbar = fig.colorbar(image_handle, ax=axes, fraction=0.022, pad=0.015)
        cbar.set_label(r"$\log_{10} p(\kappa_g\mid Y)$", fontsize=11)
        cbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        r"Exp1 Density Heatmap View: non-overlapping posterior density across $p_g$",
        fontsize=13,
    )
    _save(fig, out_path)


def plot_exp1_posterior_density_ridgeline(
    df: Any,
    out_path: Path,
    *,
    xi_ratio_order: Sequence[float] | None = None,
) -> None:
    """
    Ridgeline (joyplot-style) density view for Exp1.

    Facets by xi/xi_crit and stacks one density ridge per p_g.
    Each ridge uses:
      - shared x support kappa in [0, 1]
      - within-ridge peak normalization for height (0~1 scale)

    Notes:
      - y tick labels indicate ridge baselines (p_g values), not raw density.
      - ridge height scale is shown in-panel to avoid ambiguity.
    """
    rows = _records(df)
    if not rows:
        return

    from ...utils import load_pandas
    pd = load_pandas()
    frame = pd.DataFrame(rows).copy()
    req = {"p_g", "xi_ratio", "kappa", "density"}
    if not req.issubset(set(frame.columns)):
        return

    frame["p_g"] = frame["p_g"].astype(int)
    frame["xi_plot"] = frame["xi_ratio"].astype(float).round(6)
    frame["kappa"] = frame["kappa"].astype(float)
    frame["density"] = frame["density"].astype(float)
    frame = frame[np.isfinite(frame["kappa"]) & np.isfinite(frame["density"])]
    if frame.empty:
        return

    all_pg = sorted(int(v) for v in frame["p_g"].unique())
    if not all_pg:
        return

    ratio_targets = [float(v) for v in (xi_ratio_order or [0.5, 1.0, 1.5])]
    ncols = len(ratio_targets)
    fig, axes = plt.subplots(1, ncols, figsize=(5.9 * ncols, 6.3), sharex=True, sharey=True)
    if ncols == 1:
        axes = [axes]

    avail_global = np.asarray(sorted(float(v) for v in frame["xi_plot"].unique()), dtype=float)
    if avail_global.size == 0:
        return

    step = 1.05
    ridge_height = 0.80
    n_pg = max(len(all_pg), 1)
    x_left, x_right = 0.0, 1.0
    x_margin = 0.02

    with plt.rc_context({"font.family": "serif", "mathtext.fontset": "stix", "axes.unicode_minus": False}):
        for panel_idx, (ax, target) in enumerate(zip(axes, ratio_targets), start=1):
            chosen = float(avail_global[int(np.argmin(np.abs(avail_global - float(target))))])
            if abs(chosen - float(target)) > 0.03:
                continue
            sub_ratio = frame[np.isclose(frame["xi_plot"], chosen, atol=1e-9)].copy()
            if sub_ratio.empty:
                continue

            any_ridge = False
            for i, pg in enumerate(all_pg):
                sub = (
                    sub_ratio[sub_ratio["p_g"] == int(pg)]
                    .groupby("kappa", as_index=False)["density"]
                    .mean()
                    .sort_values("kappa")
                )
                if sub.empty:
                    continue
                x = sub["kappa"].to_numpy(dtype=float)
                y = sub["density"].to_numpy(dtype=float)
                if x.size < 2:
                    continue
                area = float(np.trapezoid(y, x))
                if area > 0:
                    y = y / area

                x_plot = np.clip(x, x_left, x_right)

                peak = float(np.nanmax(y))
                if not np.isfinite(peak) or peak <= 0:
                    continue
                y_shape = y / peak
                base = float(i) * step
                y_top = base + ridge_height * y_shape

                # Monochrome-friendly gradient for print-style figures.
                g = 0.88 - 0.55 * (i / max(n_pg - 1, 1))
                color = (g, g, g)
                ax.fill_between(x_plot, base, y_top, color=color, alpha=0.95, linewidth=0.0)
                ax.plot(x_plot, y_top, color="black", lw=1.0, alpha=0.95)
                ax.plot([x_left, x_right], [base, base], color="black", lw=0.55, alpha=0.55)

                med = _posterior_quantile_from_density(x, y, 0.5)
                if np.isfinite(med):
                    med_plot = float(np.clip(float(med), x_left, x_right))
                    ax.plot([med_plot, med_plot], [base + 0.05, base + ridge_height * 0.95], color="black", lw=1.0, ls="--", alpha=0.95)
                any_ridge = True

            if not any_ridge:
                continue

            y_ticks = [i * step + ridge_height * 0.45 for i in range(len(all_pg))]
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([str(v) for v in all_pg], fontsize=10)
            ax.set_xlim(x_left - x_margin, x_right + x_margin)
            ax.set_xticks(np.linspace(0.0, 1.0, 6))
            ax.set_ylim(-0.2, (len(all_pg) - 1) * step + ridge_height + 0.28)
            ax.set_xlabel(r"$\kappa_g$", fontsize=13)
            ax.axvline(x_left, color="black", lw=0.7, ls=":", alpha=0.55)
            ax.axvline(x_right, color="black", lw=0.7, ls=":", alpha=0.55)

            panel_tag = chr(ord("a") + panel_idx - 1)
            ax.text(
                0.5,
                -0.16,
                f"({panel_tag})",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=13,
                clip_on=False,
            )
            ax.text(
                0.985,
                0.015,
                r"support: $0\leq \kappa_g \leq 1$",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8.2,
                color="#333333",
            )
            if panel_idx == 1:
                # Compact in-panel scale cue for ridge-height meaning.
                bar_x = 0.05
                bar_y0, bar_y1 = 0.06, 0.19
                ax.plot([bar_x, bar_x], [bar_y0, bar_y1], transform=ax.transAxes, color="black", lw=0.9, clip_on=False)
                ax.plot([bar_x - 0.008, bar_x + 0.008], [bar_y0, bar_y0], transform=ax.transAxes, color="black", lw=0.9, clip_on=False)
                ax.plot([bar_x - 0.008, bar_x + 0.008], [bar_y1, bar_y1], transform=ax.transAxes, color="black", lw=0.9, clip_on=False)
                ax.text(bar_x - 0.013, bar_y0, "0", transform=ax.transAxes, ha="right", va="center", fontsize=8)
                ax.text(bar_x - 0.013, bar_y1, "1", transform=ax.transAxes, ha="right", va="center", fontsize=8)
                ax.text(
                    bar_x + 0.013,
                    0.5 * (bar_y0 + bar_y1),
                    "ridge height\n(normalized density)",
                    transform=ax.transAxes,
                    ha="left",
                    va="center",
                    fontsize=8,
                )

            ax.grid(axis="x", visible=False)
            ax.grid(axis="y", visible=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", labelsize=11, length=4)

        axes[0].set_ylabel(r"$p_g$ (ridge baseline)", fontsize=13)

    # No figure-level title; keep only panel tags (a), (b), (c).
    _save(fig, out_path)


def plot_exp1_posterior_density_small_multiples(
    df: Any,
    out_path: Path,
    *,
    xi_ratio_order: Sequence[float] | None = None,
    normalize_peak: bool = False,
    fill_area: bool = False,
    log_y: bool = False,
    log_floor: float = 1e-4,
) -> None:
    """
    Clean small-multiples layout:
      columns = xi/xi_crit values
      rows    = p_g values
      each panel contains exactly one density curve

    This avoids overlap entirely and is usually the easiest to read.
    """
    rows = _records(df)
    if not rows:
        return

    from ...utils import load_pandas
    pd = load_pandas()
    frame = pd.DataFrame(rows).copy()
    req = {"p_g", "xi_ratio", "kappa", "density"}
    if not req.issubset(set(frame.columns)):
        return

    frame["p_g"] = frame["p_g"].astype(int)
    frame["xi_plot"] = frame["xi_ratio"].astype(float).round(6)
    frame["kappa"] = frame["kappa"].astype(float)
    frame["density"] = frame["density"].astype(float)
    frame = frame[np.isfinite(frame["kappa"]) & np.isfinite(frame["density"])]
    if frame.empty:
        return

    all_pg = sorted(int(v) for v in frame["p_g"].unique())
    if not all_pg:
        return
    ratio_targets = [float(v) for v in (xi_ratio_order or [0.5, 1.0, 1.5])]
    avail = np.asarray(sorted(float(v) for v in frame["xi_plot"].unique()), dtype=float)
    if avail.size == 0:
        return
    chosen_ratios = []
    for t in ratio_targets:
        c = float(avail[int(np.argmin(np.abs(avail - float(t))))])
        if abs(c - float(t)) <= 0.03:
            chosen_ratios.append(c)
        else:
            chosen_ratios.append(float("nan"))

    nrows = len(all_pg)
    ncols = len(ratio_targets)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.5 * ncols, 1.35 * nrows + 1.1),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    # Use one y-range per p_g row so each row has a clear, comparable density axis.
    row_ymax: dict[int, float] = {}
    row_ymin: dict[int, float] = {}
    for pg in all_pg:
        ymax = 0.0
        ymin = float("inf")
        for chosen in chosen_ratios:
            if not np.isfinite(chosen):
                continue
            sub = (
                frame[
                    (frame["p_g"] == int(pg))
                    & np.isclose(frame["xi_plot"], float(chosen), atol=1e-9)
                ]
                .groupby("kappa", as_index=False)["density"]
                .mean()
                .sort_values("kappa")
            )
            if sub.empty:
                continue
            x = sub["kappa"].to_numpy(dtype=float)
            y = sub["density"].to_numpy(dtype=float)
            if x.size < 2:
                continue
            area = float(np.trapezoid(y, x))
            if area > 0:
                y = y / area
            if bool(normalize_peak):
                peak = float(np.nanmax(y))
                if peak > 0:
                    y = y / peak
            if bool(log_y):
                y_eval = np.maximum(y, float(log_floor))
                ymax = max(ymax, float(np.nanmax(y_eval)))
                ymin = min(ymin, float(np.nanmin(y_eval)))
            else:
                ymax = max(ymax, float(np.nanmax(y)))
        if bool(log_y):
            row_ymin[int(pg)] = max(min(ymin, ymax), float(log_floor))
        row_ymax[int(pg)] = max(ymax, 1e-8)

    with plt.rc_context({"font.family": "serif"}):
        for r_idx, pg in enumerate(all_pg):
            for c_idx, target in enumerate(ratio_targets):
                ax = axes[r_idx][c_idx]
                chosen = chosen_ratios[c_idx]
                if not np.isfinite(chosen):
                    ax.axis("off")
                    continue

                sub = (
                    frame[
                        (frame["p_g"] == int(pg))
                        & np.isclose(frame["xi_plot"], float(chosen), atol=1e-9)
                    ]
                    .groupby("kappa", as_index=False)["density"]
                    .mean()
                    .sort_values("kappa")
                )
                if sub.empty:
                    ax.axis("off")
                    continue

                x = sub["kappa"].to_numpy(dtype=float)
                y = sub["density"].to_numpy(dtype=float)
                if x.size < 2:
                    ax.axis("off")
                    continue
                area = float(np.trapezoid(y, x))
                if area > 0:
                    y = y / area
                if bool(normalize_peak):
                    peak = float(np.nanmax(y))
                    if peak > 0:
                        y = y / peak

                if bool(log_y):
                    y = np.maximum(y, float(log_floor))

                if bool(fill_area) and (not bool(log_y)):
                    ax.fill_between(x, 0.0, y, color="#d9d9d9", alpha=0.55, linewidth=0)
                ax.plot(x, y, color="black", lw=1.15)
                ax.set_xlim(-0.01, 1.01)
                if bool(log_y):
                    y_min = float(row_ymin.get(int(pg), float(log_floor)))
                    y_max = float(row_ymax.get(int(pg), max(y_min * 10.0, float(log_floor))))
                    ax.set_yscale("log")
                    ax.set_ylim(
                        max(y_min * 0.95, float(log_floor)),
                        max(y_max * 1.05, max(y_min * 1.2, float(log_floor) * 10.0)),
                    )
                else:
                    ylim_top = 1.08 if bool(normalize_peak) else float(row_ymax.get(int(pg), 1e-8)) * 1.08
                    ax.set_ylim(0.0, max(ylim_top, 1e-8))
                ax.set_xticks(np.linspace(0.0, 1.0, 6))
                ax.tick_params(axis="both", labelsize=8, length=3, pad=1)
                ax.grid(axis="x", color="#ececec", lw=0.5, alpha=0.8)
                ax.grid(axis="y", color="#f2f2f2", lw=0.5, alpha=0.8)
                ax.axvline(0.0, color="#888888", lw=0.6, ls=":")
                ax.axvline(1.0, color="#888888", lw=0.6, ls=":")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                if c_idx == 0:
                    if bool(log_y):
                        ylab = r"$p(\kappa_g\mid Y)$ (log y)"
                    else:
                        ylab = r"$p(\kappa_g\mid Y)$" if not bool(normalize_peak) else r"$p(\kappa_g\mid Y)/\max p$"
                    ax.set_ylabel(ylab, fontsize=8.5)
                    ax.text(
                        -0.32,
                        0.5,
                        rf"$p_g={int(pg)}$",
                        transform=ax.transAxes,
                        ha="right",
                        va="center",
                        fontsize=8.5,
                    )
                else:
                    ax.set_ylabel("")
                if r_idx == nrows - 1:
                    ax.set_xlabel(_LBL_KAPPA, fontsize=9)
                else:
                    ax.set_xlabel("")

        for c_idx, target in enumerate(ratio_targets):
            status = (
                r"$\xi/\xi_{\mathrm{crit}}<1$"
                if float(target) < 1.0
                else (r"$\xi/\xi_{\mathrm{crit}}>1$" if float(target) > 1.0 else r"$\xi/\xi_{\mathrm{crit}}=1$")
            )
            axes[0][c_idx].set_title(status, fontsize=10.5, pad=6)

    mode_note = "peak-normalized density" if bool(normalize_peak) else "raw density"
    if bool(log_y):
        mode_note += ", log-y"
    fig.suptitle(
        rf"Exp1 Small Multiples: one curve per panel ({mode_note})",
        fontsize=12.5,
    )
    _save(fig, out_path)


def plot_exp1_single_story_readable(
    df: Any,
    out_path: Path,
    *,
    u0: float = 0.5,
    slope: float | None = None,
    slope_ci: tuple[float, float] | None = None,
) -> None:
    """
    One-figure storytelling view for Exp1:
      Panel A: threshold zoom, smallest vs largest p_g
      Panel B: Delta_local and Delta_wide vs p_g (should increase)
      Panel C: W50 vs p_g (should decrease)
    """
    rows = _records(df)
    if not rows:
        return

    from ...utils import load_pandas
    pd = load_pandas()
    frame = pd.DataFrame(rows).copy()
    req = {"tau", "p_g", "xi_ratio", "mean_prob_kappa_gt_u0"}
    if not req.issubset(set(frame.columns)):
        return

    frame["xi_plot"] = frame["xi_ratio"].astype(float).round(6)
    work = (
        frame.groupby(["tau", "p_g", "xi_plot"], as_index=False)["mean_prob_kappa_gt_u0"]
        .mean()
        .rename(columns={"mean_prob_kappa_gt_u0": "prob"})
    )
    if work.empty:
        return

    # Tau-averaged curve by (p_g, xi).
    curve = (
        work.groupby(["p_g", "xi_plot"], as_index=False)["prob"]
        .mean()
        .sort_values(["p_g", "xi_plot"])
    )
    pg_vals = sorted(int(v) for v in curve["p_g"].unique())
    if not pg_vals:
        return
    pg_min, pg_max = int(pg_vals[0]), int(pg_vals[-1])

    # Metrics per p_g.
    metric_rows: list[dict[str, Any]] = []
    for pg in pg_vals:
        sub = curve[curve["p_g"].astype(int) == int(pg)].sort_values("xi_plot")
        x = sub["xi_plot"].to_numpy(dtype=float)
        p = sub["prob"].to_numpy(dtype=float)
        if x.size < 2:
            continue
        p95 = np.nan
        p105 = np.nan
        p85 = np.nan
        p115 = np.nan
        try:
            p95 = float(sub.loc[np.isclose(sub["xi_plot"], 0.95), "prob"].iloc[0])
            p105 = float(sub.loc[np.isclose(sub["xi_plot"], 1.05), "prob"].iloc[0])
        except Exception:
            pass
        try:
            p85 = float(sub.loc[np.isclose(sub["xi_plot"], 0.85), "prob"].iloc[0])
            p115 = float(sub.loc[np.isclose(sub["xi_plot"], 1.15), "prob"].iloc[0])
        except Exception:
            pass
        x25 = _exp1_interp_x_at_prob(x, p, 0.25)
        x75 = _exp1_interp_x_at_prob(x, p, 0.75)
        metric_rows.append(
            {
                "p_g": int(pg),
                "delta_local": float(p105 - p95) if np.isfinite(p95) and np.isfinite(p105) else float("nan"),
                "delta_wide": float(p115 - p85) if np.isfinite(p85) and np.isfinite(p115) else float("nan"),
                "w50": float(x75 - x25) if np.isfinite(x25) and np.isfinite(x75) and x75 >= x25 else float("nan"),
            }
        )
    metrics = pd.DataFrame(metric_rows).sort_values("p_g")
    if metrics.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.5))

    # Panel A: threshold zoom with min/max p_g.
    ax = axes[0]
    for color, pg in [("#1f77b4", pg_min), ("#d62728", pg_max)]:
        sub = curve[curve["p_g"].astype(int) == int(pg)].sort_values("xi_plot")
        sub = sub[(sub["xi_plot"] >= 0.85) & (sub["xi_plot"] <= 1.15)].copy()
        if sub.empty:
            continue
        ax.plot(
            sub["xi_plot"].to_numpy(dtype=float),
            sub["prob"].to_numpy(dtype=float),
            "o-",
            color=color,
            lw=3.0,
            ms=7.2,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=rf"$p_g={int(pg)}$",
        )
    ax.axvspan(0.85, 1.0, alpha=0.08, color="#eceff1")
    ax.axvspan(1.0, 1.15, alpha=0.08, color="#e8f5e9")
    ax.axvline(1.0, color="black", lw=2.0, ls="--")
    ax.axhline(float(u0), color="#666666", lw=1.2, ls=":")
    ax.set_xlim(0.85, 1.15)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title(r"A. Threshold Region (small vs large $p_g$)", fontsize=12, fontweight="semibold")
    ax.set_xlabel(_LBL_XI_RATIO, fontsize=11)
    ax.set_ylabel(_LBL_P_KEEP, fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.96)

    # Panel B: Delta curves.
    ax = axes[1]
    mm = metrics[np.isfinite(metrics["delta_local"]) | np.isfinite(metrics["delta_wide"])].copy()
    if not mm.empty:
        if np.any(np.isfinite(mm["delta_local"])):
            ax.plot(
                mm["p_g"].to_numpy(dtype=float),
                mm["delta_local"].to_numpy(dtype=float),
                "o-",
                color="#1f77b4",
                lw=2.8,
                ms=6.8,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=r"$\Delta_{\mathrm{local}}=\mathbb{P}(1.05)-\mathbb{P}(0.95)$",
            )
        if np.any(np.isfinite(mm["delta_wide"])):
            ax.plot(
                mm["p_g"].to_numpy(dtype=float),
                mm["delta_wide"].to_numpy(dtype=float),
                "s-",
                color="#ff7f0e",
                lw=2.8,
                ms=6.2,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=r"$\Delta_{\mathrm{wide}}=\mathbb{P}(1.15)-\mathbb{P}(0.85)$",
            )
        ax.set_xscale("log")
        ax.axhline(0.0, color="black", lw=1.1, ls="--")
        ax.set_title("B. Phase Sharpness Metrics", fontsize=12, fontweight="semibold")
        ax.set_xlabel(r"$p_g$ (log scale)", fontsize=11)
        ax.set_ylabel(r"$\Delta\mathbb{P}$ (higher is sharper)", fontsize=11)
        ax.tick_params(labelsize=10)
        ax.grid(axis="y", alpha=0.18)
        ax.legend(fontsize=8.7, loc="upper left", framealpha=0.96)

    # Panel C: transition width W50.
    ax = axes[2]
    mw = metrics[np.isfinite(metrics["w50"])].copy()
    if not mw.empty:
        ax.plot(
            mw["p_g"].to_numpy(dtype=float),
            mw["w50"].to_numpy(dtype=float),
            "o-",
            color="#2ca02c",
            lw=2.8,
            ms=6.8,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )
        ax.set_xscale("log")
        ax.set_title(r"C. Transition Width $W_{50}$", fontsize=12, fontweight="semibold")
        ax.set_xlabel(r"$p_g$ (log scale)", fontsize=11)
        ax.set_ylabel(r"$W_{50}=x@P_{0.75}-x@P_{0.25}$ (lower is sharper)", fontsize=11)
        ax.tick_params(labelsize=10)
        ax.grid(axis="y", alpha=0.18)
        ax.annotate(
            "Expected trend: downward",
            xy=(0.86, 0.24),
            xytext=(0.49, 0.52),
            textcoords="axes fraction",
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", lw=1.3, color="#444444"),
            fontsize=9.5,
            color="#333333",
        )

    title = r"Exp1 Single-Figure Evidence: finite-sample smooth transition + asymptotic threshold sharpening"
    if slope is not None and np.isfinite(float(slope)):
        if slope_ci is not None and np.isfinite(float(slope_ci[0])) and np.isfinite(float(slope_ci[1])):
            title += rf" | null $\hat s={float(slope):.3f}$ (95% CI [{float(slope_ci[0]):.3f}, {float(slope_ci[1]):.3f}])"
        else:
            title += rf" | null $\hat s={float(slope):.3f}$"
    fig.suptitle(title, fontsize=14, fontweight="bold")
    _save(fig, out_path)


def plot_exp1_transition_width_readable(
    df: Any,
    out_path: Path,
    *,
    p_low: float = 0.25,
    p_high: float = 0.75,
) -> None:
    """
    Transition-width diagnostic for Exp1:
      W = x@P(p_high) - x@P(p_low), where x = xi/xi_crit.
    Smaller W means a sharper threshold transition.
    """
    rows = _records(df)
    if not rows:
        return

    from ...utils import load_pandas
    pd = load_pandas()
    frame = pd.DataFrame(rows).copy()
    req = {"tau", "p_g", "xi_ratio", "mean_prob_kappa_gt_u0"}
    if not req.issubset(set(frame.columns)):
        return

    lo = float(p_low)
    hi = float(p_high)
    if not (0.0 < lo < hi < 1.0):
        return

    frame["xi_plot"] = frame["xi_ratio"].astype(float).round(6)
    work = (
        frame.groupby(["tau", "p_g", "xi_plot"], as_index=False)["mean_prob_kappa_gt_u0"]
        .mean()
        .rename(columns={"mean_prob_kappa_gt_u0": "prob"})
    )
    if work.empty:
        return

    rows_width: list[dict[str, Any]] = []
    for (tau, pg), sub in work.groupby(["tau", "p_g"]):
        sub = sub.sort_values("xi_plot")
        x = sub["xi_plot"].to_numpy(dtype=float)
        p = sub["prob"].to_numpy(dtype=float)
        x_lo = _exp1_interp_x_at_prob(x, p, lo)
        x_hi = _exp1_interp_x_at_prob(x, p, hi)
        if np.isfinite(x_lo) and np.isfinite(x_hi) and (x_hi >= x_lo):
            rows_width.append(
                {
                    "tau": float(tau),
                    "p_g": int(pg),
                    "x_at_low": float(x_lo),
                    "x_at_high": float(x_hi),
                    "width": float(x_hi - x_lo),
                }
            )

    if not rows_width:
        return

    width_df = pd.DataFrame(rows_width)
    fig, ax = plt.subplots(figsize=(8.8, 5.3))

    # Glance style: one panel with tau-averaged width.
    agg = (
        width_df.groupby("p_g", as_index=False)["width"]
        .agg(mean="mean", min="min", max="max")
        .sort_values("p_g")
    )
    x = agg["p_g"].to_numpy(dtype=float)
    y = agg["mean"].to_numpy(dtype=float)
    ylo = agg["min"].to_numpy(dtype=float)
    yhi = agg["max"].to_numpy(dtype=float)
    ax.plot(
        x,
        y,
        "o-",
        color="#1f77b4",
        lw=2.6,
        ms=6.8,
        markeredgecolor="white",
        markeredgewidth=0.8,
        label=r"$\mathrm{mean}_{\tau}$",
    )
    ax.fill_between(x, ylo, yhi, color="#1f77b4", alpha=0.16, label=r"$[\min_{\tau},\max_{\tau}]$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$p_g$ (log scale)", fontsize=11)
    ax.set_ylabel(rf"$W_{{50}}=x@P_{{{hi:.2f}}}-x@P_{{{lo:.2f}}}$", fontsize=11)
    ax.set_title(r"$\tau$-averaged Transition Width", fontsize=12, fontweight="semibold")
    ax.tick_params(labelsize=10)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.95)
    ax.text(
        0.03,
        0.92,
        r"Lower $W_{50}$ = sharper threshold",
        transform=ax.transAxes,
        fontsize=9,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )
    ax.annotate(
        r"Downward trend supports" + "\n" + r"asymptotic threshold sharpening",
        xy=(0.80, 0.25),
        xytext=(0.52, 0.50),
        textcoords="axes fraction",
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=1.4, color="#444444"),
        fontsize=10,
        color="#333333",
    )

    fig.suptitle(
        r"Exp1 Width (Glance View): $W_{50}$ shrinks as $p_g$ grows",
        fontsize=15,
        fontweight="bold",
    )
    _save(fig, out_path)


def plot_exp2_separation(df_summary: Any, df_kappa_raw: Any, out_dir: Path) -> None:
    """
    Exp2 -- group separation (Theorem 3.34).

    Fig A: Method comparison of null/signal group MSE with error bars (SEM across
           replicates), AUROC, and MLPD. Methods sorted by group AUROC.
    Fig B: GR-RHS kappa_g profile across groups -- boxplot across replicates
           ordered by kappa_g (gradient from null to strong signal), with
           null/signal regions shaded. Shows the step-up of kappa_g at the
           signal boundary.

    df_kappa_raw should be the raw per-replicate kappa DataFrame (kappa_df),
    with columns: replicate_id, group_id, mu_g, signal_label, post_mean_kappa_g.
    """
    summary = _as_frame(df_summary)
    kappa = _as_frame(df_kappa_raw)
    out_dir = Path(out_dir)
    from ...utils import method_display_name

    # --- Figure A: MSE-only comparison (detailed) ---
    if not summary.empty and "method" in summary.columns:
        # Sort by MSE (lower is better); GR_RHS first if tied.
        summary = summary.copy()
        sort_col = "mse_overall" if "mse_overall" in summary.columns else "null_group_mse"
        if sort_col in summary.columns:
            summary["_rank"] = summary[sort_col].rank(ascending=True, method="first")
            gr_mask = summary["method"] == "GR_RHS"
            if gr_mask.any():
                summary.loc[gr_mask, "_rank"] = np.minimum(summary.loc[gr_mask, "_rank"], 0.0)
            summary = summary.sort_values("_rank").drop(columns=["_rank"])

        methods = [method_display_name(m) for m in summary["method"]]
        raw_methods = list(summary["method"])
        x = np.arange(len(methods), dtype=float)
        colors = [_method_color(m) for m in raw_methods]

        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))

        metric_specs = [
            ("null_group_mse", "null_group_mse_std", "Null-group MSE"),
            ("signal_group_mse", "signal_group_mse_std", "Signal-group MSE"),
            ("mse_overall", "mse_overall_std", "Overall MSE"),
        ]
        n_eff = (
            summary["n_effective"].to_numpy(dtype=float)
            if "n_effective" in summary.columns
            else np.ones(len(methods), dtype=float)
        )

        for ax, (metric, std_metric, title_txt) in zip(axes, metric_specs):
            vals = (
                summary[metric].to_numpy(dtype=float)
                if metric in summary.columns
                else np.full(len(methods), np.nan, dtype=float)
            )
            ax.bar(x, vals, color=colors, alpha=0.9, edgecolor="white", linewidth=0.6)
            if std_metric in summary.columns:
                sem = summary[std_metric].to_numpy(dtype=float) / np.sqrt(np.maximum(n_eff, 1.0))
                ax.errorbar(x, vals, yerr=sem, fmt="none", color="black", capsize=3, lw=1.0)

            ax.set_xticks(x, labels=methods, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("MSE", fontsize=9)
            ax.set_title(f"{title_txt}\n(lower is better)", fontsize=8)
            ax.grid(axis="y", alpha=0.18)

            finite_idx = np.where(np.isfinite(vals))[0]
            if finite_idx.size:
                ymax = float(np.nanmax(vals[finite_idx]))
                offset = 0.02 * max(ymax, 1e-6)
                for i in finite_idx.tolist():
                    ax.text(
                        float(x[i]),
                        float(vals[i]) + offset,
                        f"{vals[i]:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="#333333",
                    )
                ax.set_ylim(0.0, ymax * 1.18 + 1e-6)

        title = "Exp2 MSE-only comparison"
        if {"GR_RHS", "RHS"}.issubset(set(summary["method"].astype(str).tolist())):
            rel_parts = []
            for metric, label in (
                ("null_group_mse", "null"),
                ("signal_group_mse", "signal"),
                ("mse_overall", "overall"),
            ):
                if metric not in summary.columns:
                    continue
                gr_v = float(summary.loc[summary["method"] == "GR_RHS", metric].iloc[0])
                rhs_v = float(summary.loc[summary["method"] == "RHS", metric].iloc[0])
                if np.isfinite(gr_v) and np.isfinite(rhs_v) and abs(rhs_v) > 1e-12:
                    rel_parts.append(f"{label}: {(rhs_v - gr_v) / rhs_v * 100.0:+.1f}%")
            if rel_parts:
                title += " | GR_RHS reduction vs RHS: " + " | ".join(rel_parts)
        fig.suptitle(title, fontsize=10, y=1.02)

        if "n_effective" in summary.columns:
            n_txt = " | ".join(
                f"{method_display_name(m)}: n={int(n)}"
                for m, n in zip(summary["method"], summary["n_effective"])
            )
            fig.text(0.5, 0.02, f"Paired-converged replicates: {n_txt}", ha="center", va="bottom", fontsize=8)

        _save(fig, out_dir / "fig2a_method_comparison.png")

    # --- Figure B: GR-RHS kappa_g distribution by group (boxplot across replicates) ---
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
                regime = "null" if xi_r == 0 else rf"$\xi/\xi_{{\mathrm{{crit}}}}={xi_r:.1f}$"
                labels_g.append(f"G{int(g)}\n{regime}")
            else:
                labels_g.append(f"G{int(g)}\n{'null' if mu_val == 0 else rf'$\\mu={mu_val:.2g}$'}")
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

        ax.axhline(0.5, color="black", ls="--", lw=1.2, label=r"$u_0 = 0.5$")
        ax.set_xticks(np.arange(len(groups)), labels=labels_g, fontsize=8)
        ax.set_ylabel(r"Posterior mean $\kappa_g$ (GR-RHS)", fontsize=10)
        ax.set_title(
            r"GR-RHS $\kappa_g$ profile by group (Theorem 3.34)" "\n"
            "Blue = null groups, Red = signal groups  |  Box = IQR across replicates",
            fontsize=9,
        )
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=8, loc="upper left")
        _save(fig, out_dir / "fig2b_kappa_by_group.png")


def plot_exp3_benchmark(df: Any, out_dir: Path) -> None:
    """
    Exp3 �� factorial benchmark (signal_type �� rho_within �� snr).

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

    from ...utils import method_display_name

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

    # --- Fig A: MSE line plots (rho �� method, panel per signal type) ---
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
            ax.set_xlabel("��_within", fontsize=9)
            ax.set_ylabel("MSE overall", fontsize=9)
            if col_i == n_sig - 1 and row_i == 0:
                ax.legend(fontsize=7, loc="upper left", ncol=1)

    fig_a.suptitle("Exp3: MSE vs correlation (per signal type/SNR)\nPreserves rho��method interaction", fontsize=10, y=1.01)
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
                ax.set_xlabel("��_within", fontsize=9)
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


def plot_exp3_boundary_phase_transition(df: Any, out_path: Path, *, u0: float = 0.5) -> None:
    """
    Exp3b phase-scan plot:
      y = P(kappa_g > u0 | Y) over xi/xi_crit, with Cor 3.18 kappa*(xi) overlay.
    """
    frame = _as_frame(df)
    if frame.empty:
        return
    req = {"signal", "method", "boundary_xi_ratio", "boundary_rho_profile", "kappa_signal_prob_gt_u0"}
    if not req.issubset(set(frame.columns)):
        return

    from ...utils import method_display_name

    work = frame.copy()
    work = work.loc[work["signal"].astype(str) == "boundary"].copy()
    work = work.loc[np.isfinite(work["boundary_xi_ratio"]) & np.isfinite(work["boundary_rho_profile"])].copy()
    work = work.loc[np.isfinite(work["kappa_signal_prob_gt_u0"])].copy()
    if work.empty:
        return

    rho_vals = sorted(float(v) for v in work["boundary_rho_profile"].unique())
    n_panels = max(1, len(rho_vals))
    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, 4.8), squeeze=False)
    methods = _sort_methods(work["method"].unique())

    for i, rho in enumerate(rho_vals):
        ax = axes[0][i]
        sub = work.loc[np.isclose(work["boundary_rho_profile"].astype(float), float(rho))]
        if sub.empty:
            ax.set_visible(False)
            continue

        xi_vals = sorted(float(v) for v in sub["boundary_xi_ratio"].unique())
        for m in methods:
            msub = sub.loc[sub["method"].astype(str) == str(m)]
            if msub.empty:
                continue
            agg = msub.groupby("boundary_xi_ratio")["kappa_signal_prob_gt_u0"].agg(["mean", "min", "max"]).reset_index()
            agg = agg.sort_values("boundary_xi_ratio")
            xs = agg["boundary_xi_ratio"].to_numpy(dtype=float)
            ys = agg["mean"].to_numpy(dtype=float)
            ax.plot(xs, ys, "o-", color=_method_color(str(m)), lw=1.8, ms=5, label=method_display_name(str(m)), zorder=3)
            if not (agg["min"] == agg["max"]).all():
                ax.fill_between(xs, agg["min"], agg["max"], alpha=0.10, color=_method_color(str(m)))

        xs_theory = np.asarray(xi_vals, dtype=float)
        ys_theory = np.asarray(
            [kappa_star_xi_ratio_u0_rho(xi_ratio=float(x), u0=float(u0), rho=float(rho)) for x in xs_theory],
            dtype=float,
        )
        ys_theory = np.clip(ys_theory, 0.0, 1.0)
        ax.plot(xs_theory, ys_theory, "--", color="black", lw=2.0, label="Theory kappa*(xi)", zorder=4)
        ax.axvline(1.0, color="gray", lw=1.2, ls=":")
        ax.set_xlabel("xi / xi_crit", fontsize=10)
        ax.set_ylabel(f"P(kappa_g > {float(u0):.2f} | Y)", fontsize=10)
        ax.set_title(f"Boundary phase scan (rho={float(rho):.2f})", fontsize=9)
        ax.set_ylim(-0.03, 1.05)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Exp3b: phase transition across xi/xi_crit", fontsize=11, y=1.01)
    _save(fig, Path(out_path))


def plot_exp4_ablation(df: Any, out_dir: Path) -> None:
    """
    Exp4 �� ablation / tau calibration.

    Fig A (diagnostic): tau_post_mean vs tau_oracle scatter.
      - One point per (variant �� p0_true) combination.
      - Color = p0_true, marker shape = variant.
      - Identity line (y=x) is the oracle target.
      - This is a diagnostic view of posterior scale relative to tau0.

    Fig B (primary): Normalized MSE per variant (MSE_variant / MSE_RHS_oracle_p0).
      - One group of bars per p0, bars colored by variant.
      - Normalizing to RHS_oracle removes the trivial p0 scaling, so all p0 values
        are on the same plot without dominating each other.

    The old `set_index("variant")` pattern crashed when multiple p0 values caused
    duplicate index entries �� replaced with pivot_table throughout.
    """
    frame = _as_frame(df)
    out_dir = Path(out_dir)
    if frame.empty or "variant" not in frame.columns:
        return

    from ...utils import load_pandas
    pd = load_pandas()

    variants = _sort_methods(frame["variant"].unique())  # use consistent ordering helper
    p0_vals = sorted(frame["p0_true"].unique()) if "p0_true" in frame.columns else [None]
    cmap_p0 = plt.cm.get_cmap("tab10", max(len(p0_vals), 1))

    # Marker cycle for variants
    _MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]
    var_marker = {v: _MARKERS[i % len(_MARKERS)] for i, v in enumerate(variants)}

    # --- Fig A: ��_post_mean vs ��_oracle scatter ---
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
                ax_a.scatter(
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
        ax_a.set_xlim(lims)
        ax_a.set_ylim(lims)
        ax_a.set_xlabel("tau oracle  (p0/(p-p0)/sqrt(n))", fontsize=10)
        ax_a.set_ylabel("tau posterior mean", fontsize=10)
        ax_a.set_title("tau diagnostic scatter (Exp4)\nIdentity line is a prior-scale reference, not a strict target", fontsize=9)
        leg1 = ax_a.legend(handles=handles_p0 + var_handles, fontsize=8, loc="upper left",
                            ncol=2, title="color=p0, marker=variant")
        ax_a.add_artist(leg1)
        _save(fig_a, out_dir / "fig4a_tau_scatter.png")

    # --- Fig B: normalized MSE bar chart (variant / RHS_oracle_p0) ---
    if "mse_overall" in frame.columns:
        # Use RHS_oracle as canonical reference when available.
        ref_variant = "RHS_oracle" if "RHS_oracle" in variants else variants[0]
        ref_df = frame[frame["variant"] == ref_variant]
        oracle_mse: dict = {}
        for p0 in p0_vals:
            sub = ref_df[ref_df["p0_true"] == p0] if p0 is not None else ref_df
            oracle_mse[p0] = float(sub["mse_overall"].mean()) if not sub.empty else float("nan")

        # pivot: rows=p0, cols=variant �� normalized MSE
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
        ax_b.set_title("Exp4: Normalized MSE per ablation variant\n<1 = better than RHS_oracle; >1 = worse", fontsize=9)
        ax_b.legend(fontsize=8)
        _save(fig_b, out_dir / "fig4b_mse_normalized.png")


def plot_exp5_prior_sensitivity(df: Any, out_dir: Path) -> None:
    """
    Exp5 �� prior sensitivity for (alpha_kappa, beta_kappa).

    Design goal: demonstrate ROBUSTNESS �� show that different (alpha, beta) priors
    give nearly identical results, so the default (0.5, 1.0) is safe.

    Fig A (primary): One panel per metric; x-axis = prior configuration (with
      descriptive labels); one colored line per scenario; default prior (0.5, 1.0)
      highlighted with a vertical dashed line. Flat lines = robust.

    Fig B (secondary, if kappa columns present): ��_null vs ��_signal scatter across
      priors, one point per prior �� scenario; shows separation is stable.

    Fixes the old bugs:
      - `\\n` �� `\n` in ylabel (was literal backslash-n)
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
        ("kappa_null_mean",   "��_null mean"),
        ("kappa_signal_mean", "��_signal mean"),
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
        "Exp5: Prior sensitivity for (��_��, ��_��)  �� flat lines = robust\n"
        "Gold band = default prior (0.5, 1.0)  |  One line per scenario",
        fontsize=10, y=1.01,
    )
    _save(fig_a, out_dir / "fig5_prior_sensitivity.png")

    # --- Fig B: �� separation scatter across priors (if both kappa columns present) ---
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
        ax_b.set_xlabel("��_null mean  (target: low)", fontsize=9)
        ax_b.set_ylabel("��_signal mean  (target: high)", fontsize=9)
        ax_b.set_title("Exp5: �� separation across priors\nGold = default prior; ideal = bottom-right", fontsize=9)
        _save(fig_b, out_dir / "fig5b_kappa_separation.png")

