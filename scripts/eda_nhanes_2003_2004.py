from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prepare_nhanes_2003_2004 import (
    BASE_COVARIATES,
    BMI_VAR,
    EXPOSURE_GROUPS,
    OUTCOME_VAR,
    REQUIRED_FILES,
    UCR_VAR,
    VARIABLE_LABELS,
)

PAPER_LABELS: Dict[str, str] = {
    "LBXSGTSI": "GGT",
    "LBXBPB": "Lead",
    "LBXBCD": "Cadmium",
    "LBXTHG": "Mercury",
    "URXMEP": "Mono-ethyl phthalate",
    "URXMBP": "Mono-n-butyl phthalate",
    "URXMIB": "Mono-isobutyl phthalate",
    "URXMHP": "Mono-(2-ethylhexyl) phthalate",
    "URXMHH": "Mono-(2-ethyl-5-hydroxyhexyl) phthalate",
    "URXMOH": "Mono-(2-ethyl-5-oxohexyl) phthalate",
    "URXMZP": "Mono-benzyl phthalate",
    "LBXPDE": "p,p'-DDE",
    "LBXPDT": "p,p'-DDT",
    "LBXBHC": "Beta-hexachlorocyclohexane",
    "LBXOXY": "Oxychlordane",
    "LBXTNA": "Trans-nonachlor",
    "LBXHPE": "Heptachlor epoxide",
    "LBXMIR": "Mirex",
    "LBXHCB": "Hexachlorobenzene",
    "LBXBR2": "BDE-28",
    "LBXBR3": "BDE-47",
    "LBXBR5": "BDE-99",
    "LBXBR6": "BDE-100",
    "LBXBR7": "BDE-153",
    "LBXBR8": "BDE-154",
    "LBXBR9": "BDE-183",
    "URXP01": "1-hydroxynaphthalene",
    "URXP02": "2-hydroxynaphthalene",
    "URXP03": "2-hydroxyfluorene",
    "URXP04": "3-hydroxyfluorene",
    "URXP05": "1-hydroxyphenanthrene",
    "URXP06": "2-hydroxyphenanthrene",
    "URXP07": "3-hydroxyphenanthrene",
    "URXP10": "9-hydroxyfluorene",
    "URXP17": "1-hydroxypyrene",
    "URXP19": "4-phenanthrene",
}

GROUP_DISPLAY_NAMES: Dict[str, str] = {
    "metals": "Metals",
    "phthalates": "Phthalates",
    "organochlorines": "OCPs",
    "pbdes": "PBDEs",
    "pahs": "PAHs",
}

HEATMAP_SHORT_LABELS: Dict[str, str] = {
    "LBXBPB": "Lead",
    "LBXBCD": "Cd",
    "LBXTHG": "Hg",
    "URXMEP": "MEP",
    "URXMBP": "MBP",
    "URXMIB": "MiBP",
    "URXMHP": "MEHP",
    "URXMHH": "MEHHP",
    "URXMOH": "MEOHP",
    "URXMZP": "MBzP",
    "LBXPDE": "p,p'-DDE",
    "LBXPDT": "p,p'-DDT",
    "LBXBHC": "beta-HCH",
    "LBXOXY": "Oxy",
    "LBXTNA": "TNA",
    "LBXHPE": "HeptE",
    "LBXMIR": "Mirex",
    "LBXHCB": "HCB",
    "LBXBR2": "BDE-28",
    "LBXBR3": "BDE-47",
    "LBXBR5": "BDE-99",
    "LBXBR6": "BDE-100",
    "LBXBR7": "BDE-153",
    "LBXBR8": "BDE-154",
    "LBXBR9": "BDE-183",
    "URXP01": "1-NAP",
    "URXP02": "2-NAP",
    "URXP03": "2-FLU",
    "URXP04": "3-FLU",
    "URXP05": "1-PHE",
    "URXP06": "2-PHE",
    "URXP07": "3-PHE",
    "URXP10": "9-FLU",
    "URXP17": "1-PYR",
    "URXP19": "4-PHE",
}


def _read_xpt(path: Path) -> pd.DataFrame:
    return pd.read_sas(path, format="xport", encoding="utf-8")


def _safe_log_series(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    if (values <= 0).any():
        raise ValueError(f"Encountered non-positive values in {series.name}; log transform is undefined.")
    return np.log(values)


def _flatten_exposures(groups: Dict[str, List[str]]) -> List[str]:
    return [column for block in groups.values() for column in block]


def _pretty_label(column: str) -> str:
    raw = PAPER_LABELS.get(column, VARIABLE_LABELS.get(column, column))
    return raw.replace("_", " ")


def _heatmap_label(column: str, *, short: bool) -> str:
    if short:
        return HEATMAP_SHORT_LABELS.get(column, _pretty_label(column))
    return _pretty_label(column)


def _adult_analysis_frame(raw_dir: Path) -> pd.DataFrame:
    paths = {key: raw_dir / value for key, value in REQUIRED_FILES.items()}
    demo = _read_xpt(paths["demo"])[BASE_COVARIATES].copy()
    frame = demo.merge(_read_xpt(paths["biochem"])[["SEQN", OUTCOME_VAR]], on="SEQN", how="left")
    frame = frame.merge(_read_xpt(paths["creatinine"])[["SEQN", UCR_VAR]], on="SEQN", how="left")
    frame = frame.merge(_read_xpt(paths["bmx"])[["SEQN", BMI_VAR]], on="SEQN", how="left")
    frame = frame.merge(_read_xpt(paths["metals"])[["SEQN", *EXPOSURE_GROUPS["metals"]]], on="SEQN", how="left")
    frame = frame.merge(_read_xpt(paths["phthalates"])[["SEQN", *EXPOSURE_GROUPS["phthalates"]]], on="SEQN", how="left")
    frame = frame.merge(_read_xpt(paths["ocp"])[["SEQN", *EXPOSURE_GROUPS["organochlorines"]]], on="SEQN", how="left")
    frame = frame.merge(_read_xpt(paths["pbde"])[["SEQN", *EXPOSURE_GROUPS["pbdes"]]], on="SEQN", how="left")
    frame = frame.merge(_read_xpt(paths["pah"])[["SEQN", *EXPOSURE_GROUPS["pahs"]]], on="SEQN", how="left")
    return frame[frame["RIDAGEYR"] >= 18].copy()


def _complete_case_frame(adult: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    exposure_columns = _flatten_exposures(EXPOSURE_GROUPS)
    required = [
        OUTCOME_VAR,
        BMI_VAR,
        UCR_VAR,
        "RIDAGEYR",
        "RIAGENDR",
        "RIDRETH1",
        "INDFMPIR",
        *exposure_columns,
    ]
    complete = adult.dropna(subset=required).copy()
    complete.sort_values("SEQN", inplace=True)
    complete.reset_index(drop=True, inplace=True)
    return complete, exposure_columns


def _missing_rate_table(adult: pd.DataFrame, complete: pd.DataFrame, exposure_columns: Sequence[str]) -> pd.DataFrame:
    tracked = [OUTCOME_VAR, BMI_VAR, UCR_VAR, "RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDFMPIR", *exposure_columns]
    rows = []
    for column in tracked:
        non_missing_adult = adult[column].notna().sum()
        non_missing_complete = complete[column].notna().sum()
        rows.append(
            {
                "variable": column,
                "label": _pretty_label(column),
                "missing_rate_adults": float(adult[column].isna().mean()),
                "non_missing_adults": int(non_missing_adult),
                "non_missing_complete_cases": int(non_missing_complete),
            }
        )
    table = pd.DataFrame(rows)
    table.sort_values(["missing_rate_adults", "variable"], ascending=[False, True], inplace=True)
    return table


def _plot_distribution_grid(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    transform: str,
    title: str,
    out_path: Path,
    bins: int = 30,
) -> None:
    if not columns:
        return
    n_cols = 4
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 2.9 * n_rows))
    axes_arr = np.atleast_1d(axes).reshape(n_rows, n_cols)
    for ax in axes_arr.ravel():
        ax.set_visible(False)

    for ax, column in zip(axes_arr.ravel(), columns):
        series = frame[column].dropna().astype(float)
        if transform == "log":
            series = _safe_log_series(series.rename(column))
            xlabel = f"log({_pretty_label(column)})"
        else:
            xlabel = _pretty_label(column)
        ax.set_visible(True)
        ax.hist(series.to_numpy(), bins=bins, color="#2E5EAA", alpha=0.85, edgecolor="white", linewidth=0.4)
        ax.set_title(_pretty_label(column), fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("count", fontsize=8)
        ax.tick_params(labelsize=8)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_group_heatmap(
    frame: pd.DataFrame,
    columns: Sequence[str],
    title: str,
    out_path: Path,
    *,
    groups: Dict[str, List[str]] | None = None,
    tick_fontsize: int = 8,
    group_label_fontsize: int = 9,
    title_fontsize: int = 14,
    fig_scale: float = 0.78,
    short_labels: bool = False,
    cell_size_in: float | None = None,
) -> None:
    corr = frame.loc[:, columns].corr()
    if cell_size_in is not None:
        matrix_size = max(5.0, cell_size_in * len(columns))
        fig_width = matrix_size + 4.8
        fig_height = matrix_size + 4.0
    else:
        fig_size = max(5.6, fig_scale * len(columns))
        fig_width = fig_size
        fig_height = fig_size
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(columns)))
    ax.set_yticks(np.arange(len(columns)))
    labels = [_heatmap_label(column, short=short_labels) for column in columns]
    ax.set_xticklabels(labels, rotation=90, fontsize=tick_fontsize)
    ax.set_yticklabels(labels, fontsize=tick_fontsize)
    ax.set_title(title, fontsize=title_fontsize, pad=20, fontweight="bold")
    if groups:
        boundaries: List[int] = []
        centers: List[Tuple[float, str]] = []
        cursor = 0
        for group_name, group_columns in groups.items():
            size = len(group_columns)
            if size <= 0:
                continue
            centers.append((cursor + (size - 1) / 2, GROUP_DISPLAY_NAMES.get(group_name, group_name)))
            cursor += size
            boundaries.append(cursor)
        for boundary in boundaries[:-1]:
            ax.axhline(boundary - 0.5, color="white", linewidth=1.2)
            ax.axvline(boundary - 0.5, color="white", linewidth=1.2)
        for center, group_label in centers:
            ax.text(
                center,
                -5.4,
                group_label,
                ha="center",
                va="bottom",
                fontsize=group_label_fontsize,
                fontweight="bold",
                clip_on=False,
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Pearson r", rotation=90, fontsize=max(10, tick_fontsize))
    cbar.ax.tick_params(labelsize=max(9, tick_fontsize - 1))
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _write_summary(
    out_dir: Path,
    *,
    adult_n: int,
    complete_case_n: int,
    missing_table: pd.DataFrame,
    exposure_columns: Sequence[str],
) -> None:
    by_group = {
        group: {
            "count": len(columns),
            "mean_missing_rate_adults": float(missing_table.loc[missing_table["variable"].isin(columns), "missing_rate_adults"].mean()),
        }
        for group, columns in EXPOSURE_GROUPS.items()
    }
    summary = {
        "adult_n": int(adult_n),
        "complete_case_n": int(complete_case_n),
        "complete_case_fraction": float(complete_case_n / adult_n) if adult_n else 0.0,
        "exposure_count": int(len(exposure_columns)),
        "top_missing_variables": missing_table.head(10).to_dict(orient="records"),
        "group_missing_summary": by_group,
    }
    (out_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_eda(raw_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    adult = _adult_analysis_frame(raw_dir)
    complete, exposure_columns = _complete_case_frame(adult)

    missing_table = _missing_rate_table(adult, complete, exposure_columns)
    missing_table.to_csv(out_dir / "missing_rate_table.csv", index=False)

    _plot_distribution_grid(
        complete,
        [OUTCOME_VAR],
        transform="raw",
        title="GGT distribution before log transform",
        out_path=out_dir / "ggt_distribution_raw.png",
    )
    _plot_distribution_grid(
        complete,
        [OUTCOME_VAR],
        transform="log",
        title="GGT distribution after log transform",
        out_path=out_dir / "ggt_distribution_log.png",
    )

    for group_name, columns in EXPOSURE_GROUPS.items():
        _plot_distribution_grid(
            complete,
            columns,
            transform="raw",
            title=f"{group_name} exposures before log transform",
            out_path=out_dir / f"{group_name}_distributions_raw.png",
        )
        _plot_distribution_grid(
            complete,
            columns,
            transform="log",
            title=f"{group_name} exposures after log transform",
            out_path=out_dir / f"{group_name}_distributions_log.png",
        )
        log_frame = complete.loc[:, columns].apply(_safe_log_series, axis=0)
        _plot_group_heatmap(
            log_frame,
            columns,
            title=f"{group_name} within-group correlations (log scale)",
            out_path=out_dir / f"{group_name}_correlation_heatmap.png",
        )

    all_log = complete.loc[:, exposure_columns].apply(_safe_log_series, axis=0)
    _plot_group_heatmap(
        all_log,
        exposure_columns,
        title="All exposure correlations (log scale)",
        out_path=out_dir / "all_exposures_correlation_heatmap.png",
        groups=EXPOSURE_GROUPS,
        tick_fontsize=18,
        group_label_fontsize=15,
        title_fontsize=22,
        fig_scale=0.64,
        short_labels=True,
        cell_size_in=0.28,
    )

    _write_summary(
        out_dir,
        adult_n=len(adult),
        complete_case_n=len(complete),
        missing_table=missing_table,
        exposure_columns=exposure_columns,
    )
    print(f"[ok] EDA artifacts written to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EDA artifacts for NHANES 2003-2004 GGT exposure analysis.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/real/nhanes_2003_2004/raw"),
        help="Directory containing downloaded NHANES XPT files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/reports/nhanes_2003_2004_eda"),
        help="Destination directory for EDA tables and figures.",
    )
    args = parser.parse_args()
    run_eda(args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
