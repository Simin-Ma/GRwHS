from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml


PAPER_LABELS: Dict[str, str] = {
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
    "organochlorines": "Organochlorine pesticides",
    "pbdes": "PBDEs",
    "pahs": "PAHs",
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload or {}


def _discover_runs(run_dirs: Optional[Sequence[Path]], sweep_dir: Optional[Path]) -> List[Path]:
    if run_dirs:
        return [path.expanduser().resolve() for path in run_dirs]
    if sweep_dir is None:
        raise SystemExit("Provide either --run-dir or --sweep-dir.")
    root = sweep_dir.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Sweep directory not found: {root}")
    return sorted([path for path in root.iterdir() if path.is_dir()])


def _model_label(run_dir: Path) -> str:
    cfg_path = run_dir / "resolved_config.yaml"
    if not cfg_path.exists():
        return run_dir.name
    cfg = _load_yaml(cfg_path)
    model_cfg = cfg.get("model", {})
    name = str(model_cfg.get("name", run_dir.name))
    aliases = {
        "grrhs_nuts": "GR-RHS",
        "gigg": "GIGG",
        "ridge": "Ridge",
        "lasso": "Lasso",
        "sparse_group_lasso": "Sparse Group Lasso",
        "logistic_regression": "Logistic regression",
    }
    if name == "grrhs_nuts" and model_cfg.get("use_groups") is False:
        return "RHS"
    return aliases.get(name, name)


def _group_mapping(groups: Sequence[Sequence[int]]) -> Dict[int, str]:
    ordered_names = list(GROUP_DISPLAY_NAMES.keys())
    mapping: Dict[int, str] = {}
    for gid, members in enumerate(groups):
        label = ordered_names[gid] if gid < len(ordered_names) else f"group_{gid}"
        for idx in members:
            mapping[int(idx)] = label
    return mapping


def _fold_effect_rows(run_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for repeat_dir in sorted(path for path in run_dir.glob("repeat_*") if path.is_dir()):
        meta_path = repeat_dir / "dataset_meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        feature_names = meta.get("feature_names") or []
        groups = meta.get("groups") or []
        group_name_by_index = _group_mapping(groups)

        for fold_dir in sorted(path for path in repeat_dir.glob("fold_*") if path.is_dir()):
            posterior_path = fold_dir / "posterior_samples.npz"
            fold_arrays_path = fold_dir / "fold_arrays.npz"
            if not posterior_path.exists() or not fold_arrays_path.exists():
                continue
            posterior = np.load(posterior_path)
            if "beta" not in posterior:
                continue
            beta_draws = np.asarray(posterior["beta"], dtype=float)
            if beta_draws.ndim == 1:
                beta_draws = beta_draws.reshape(-1, 1)
            x_scale = np.load(fold_arrays_path)["x_scale"].astype(float)
            if x_scale.size != beta_draws.shape[1]:
                continue
            delta = np.log(2.0) / np.maximum(x_scale, 1e-8)
            percent_change_draws = 100.0 * (np.exp(beta_draws * delta[np.newaxis, :]) - 1.0)
            for idx, feature_name in enumerate(feature_names):
                draws = percent_change_draws[:, idx]
                rows.append(
                    {
                        "run_dir": str(run_dir),
                        "run_name": run_dir.name,
                        "model": _model_label(run_dir),
                        "repeat": repeat_dir.name,
                        "fold": fold_dir.name,
                        "feature": feature_name,
                        "label": PAPER_LABELS.get(feature_name, feature_name),
                        "group": GROUP_DISPLAY_NAMES.get(group_name_by_index.get(idx, ""), group_name_by_index.get(idx, "unknown")),
                        "mean_percent_change": float(np.mean(draws)),
                        "median_percent_change": float(np.quantile(draws, 0.5)),
                        "ci95_low": float(np.quantile(draws, 0.025)),
                        "ci95_high": float(np.quantile(draws, 0.975)),
                        "ci95_length": float(np.quantile(draws, 0.975) - np.quantile(draws, 0.025)),
                        "draw_count": int(draws.size),
                    }
                )
    return rows


def _aggregate_effects(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return (
        frame.groupby(["run_dir", "run_name", "model", "feature", "label", "group"], as_index=False)
        .agg(
            mean_percent_change=("mean_percent_change", "mean"),
            median_percent_change=("median_percent_change", "mean"),
            ci95_low=("ci95_low", "mean"),
            ci95_high=("ci95_high", "mean"),
            ci95_length=("ci95_length", "mean"),
            folds=("fold", "nunique"),
            total_draws=("draw_count", "sum"),
        )
        .sort_values(["model", "group", "feature"])
    )


def _group_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return (
        frame.groupby(["run_dir", "run_name", "model", "group"], as_index=False)
        .agg(
            mean_abs_percent_change=("median_percent_change", lambda s: float(np.mean(np.abs(s)))),
            mean_ci95_length=("ci95_length", "mean"),
            median_ci95_length=("ci95_length", "median"),
            exposure_count=("feature", "count"),
        )
        .sort_values(["model", "group"])
    )


def _ci_reduction(group_frame: pd.DataFrame, reference_model: str) -> pd.DataFrame:
    if group_frame.empty:
        return group_frame
    ref = group_frame[group_frame["model"].str.lower() == reference_model.lower()]
    if ref.empty:
        return pd.DataFrame()
    merged = group_frame.merge(
        ref[["group", "mean_ci95_length"]].rename(columns={"mean_ci95_length": "reference_mean_ci95_length"}),
        on="group",
        how="left",
    )
    merged = merged[merged["reference_mean_ci95_length"].notna()].copy()
    merged["ci95_length_reduction_pct"] = 100.0 * (
        1.0 - merged["mean_ci95_length"] / np.maximum(merged["reference_mean_ci95_length"], 1e-8)
    )
    return merged.sort_values(["group", "model"])


def _write_markdown(
    out_path: Path,
    exposure_frame: pd.DataFrame,
    group_frame: pd.DataFrame,
    reduction_frame: pd.DataFrame,
    reference_model: Optional[str],
) -> None:
    lines: List[str] = ["# NHANES effect summary", ""]
    if exposure_frame.empty:
        lines.append("No posterior effect summaries were found.")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append("## Group-level CI summary")
    lines.append("")
    lines.append("| Model | Group | Mean |% change| | Mean 95% CI length |")
    lines.append("| --- | --- | ---: | ---: |")
    for row in group_frame.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.group} | {row.mean_abs_percent_change:.2f} | {row.mean_ci95_length:.2f} |"
        )

    if not reduction_frame.empty and reference_model:
        lines.append("")
        lines.append(f"## CI length reduction vs {reference_model}")
        lines.append("")
        lines.append("| Model | Group | Reduction (%) |")
        lines.append("| --- | --- | ---: |")
        for row in reduction_frame.itertuples(index=False):
            lines.append(f"| {row.model} | {row.group} | {row.ci95_length_reduction_pct:.2f} |")

    lines.append("")
    lines.append("## Exposure-level summary")
    lines.append("")
    lines.append("| Model | Exposure | Group | Median % change | 95% CI | CI length |")
    lines.append("| --- | --- | --- | ---: | --- | ---: |")
    for row in exposure_frame.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.label} | {row.group} | {row.median_percent_change:.2f} | "
            f"[{row.ci95_low:.2f}, {row.ci95_high:.2f}] | {row.ci95_length:.2f} |"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize NHANES exposure effects as percent change in GGT for 2x exposure.")
    parser.add_argument("--run-dir", type=Path, action="append", help="Specific run directory. Can be passed multiple times.")
    parser.add_argument("--sweep-dir", type=Path, help="Sweep directory containing multiple run directories.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/reports/nhanes_effects"), help="Destination for CSV/JSON/Markdown outputs.")
    parser.add_argument("--reference-model", type=str, default=None, help="Optional model label for CI length reduction, e.g. RHS or GIGG.")
    args = parser.parse_args()

    run_dirs = _discover_runs(args.run_dir, args.sweep_dir)
    rows: List[Dict[str, Any]] = []
    skipped_runs: List[str] = []
    for run_dir in run_dirs:
        effect_rows = _fold_effect_rows(run_dir)
        if effect_rows:
            rows.extend(effect_rows)
        else:
            skipped_runs.append(str(run_dir))

    exposure_frame = _aggregate_effects(rows)
    group_frame = _group_summary(exposure_frame)
    reduction_frame = _ci_reduction(group_frame, args.reference_model) if args.reference_model else pd.DataFrame()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    exposure_csv = args.out_dir / "nhanes_exposure_effects.csv"
    group_csv = args.out_dir / "nhanes_group_ci_summary.csv"
    json_path = args.out_dir / "nhanes_effect_summary.json"
    md_path = args.out_dir / "nhanes_effect_summary.md"

    exposure_frame.to_csv(exposure_csv, index=False)
    group_frame.to_csv(group_csv, index=False)
    payload = {
        "runs_analyzed": sorted(exposure_frame["run_dir"].unique().tolist()) if not exposure_frame.empty else [],
        "skipped_runs": skipped_runs,
        "reference_model": args.reference_model,
        "exposure_effects": exposure_frame.to_dict(orient="records"),
        "group_summary": group_frame.to_dict(orient="records"),
        "ci_reduction": reduction_frame.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(md_path, exposure_frame, group_frame, reduction_frame, args.reference_model)

    print(f"[ok] exposure-level summary -> {exposure_csv}")
    print(f"[ok] group-level summary -> {group_csv}")
    print(f"[ok] json summary -> {json_path}")
    print(f"[ok] markdown summary -> {md_path}")
    if skipped_runs:
        print(f"[warn] skipped {len(skipped_runs)} run(s) without posterior fold outputs")


if __name__ == "__main__":
    main()

