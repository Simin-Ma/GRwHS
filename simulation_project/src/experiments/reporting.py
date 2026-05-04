from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from ..utils import method_result_label

from .schemas import RunManifest
from ..utils import ensure_dir, load_pandas

def _timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _stable_name_seed(name: str, *, mod: int = 1000) -> int:
    code = 0
    for ch in str(name):
        code = (code * 131 + ord(ch)) % int(mod)
    return int(code)


def _record_produced_paths(store: set[Path], *paths: Path) -> None:
    for p in paths:
        try:
            if Path(p).exists() and Path(p).is_file():
                store.add(Path(p).resolve())
        except Exception:
            continue


def _collect_existing_paths(obj: Any) -> set[Path]:
    out: set[Path] = set()

    def _walk(v: Any) -> None:
        if isinstance(v, dict):
            for vv in v.values():
                _walk(vv)
            return
        if isinstance(v, (list, tuple)):
            for vv in v:
                _walk(vv)
            return
        if isinstance(v, str):
            p = Path(v)
            if p.exists() and p.is_file():
                out.add(p)

    _walk(obj)
    return out


def _analyze_single_experiment(exp_key: str, results_dir: Path) -> dict[str, Any]:
    try:
        from .analysis.report import (
            analyze_exp1,
            analyze_exp2,
            analyze_exp3,
            analyze_exp4,
            analyze_exp5,
            analyze_ga_v2_complexity_mismatch,
            analyze_ga_v2_correlation_stress,
            analyze_ga_v2_group_separation,
        )
        analyzers = {
            "exp1": analyze_exp1,
            "exp2": analyze_exp2,
            "exp3": analyze_exp3,
            "exp3a": analyze_exp3,
            "exp3b": analyze_exp3,
            "exp3c": analyze_exp3,
            "exp3d": analyze_exp3,
            "exp4": analyze_exp4,
            "exp5": analyze_exp5,
            "ga_v2_group_separation": analyze_ga_v2_group_separation,
            "ga_v2_complexity_mismatch": analyze_ga_v2_complexity_mismatch,
            "ga_v2_correlation_stress": analyze_ga_v2_correlation_stress,
        }
        fn = analyzers.get(str(exp_key).strip().lower())
        if fn is None:
            return {"metrics": {}, "findings": [f"No analyzer registered for {exp_key}."]}
        return fn(results_dir)
    except Exception as exc:
        return {"metrics": {}, "findings": [f"Analyzer failed for {exp_key}: {type(exc).__name__}: {exc}"]}


def _build_run_summary_table(exp_key: str, results_dir: Path):
    pd = load_pandas()

    exp_norm = str(exp_key).strip().lower()
    parse_warnings: list[str] = []

    def _warn(msg: str) -> None:
        parse_warnings.append(str(msg))

    def _warning_table():
        rows: list[dict[str, Any]] = [{"metric": "parse_warning_count", "value": int(len(parse_warnings))}]
        for idx, msg in enumerate(parse_warnings, start=1):
            rows.append({"metric": f"parse_warning_{idx}", "value": str(msg)})
        return pd.DataFrame(rows)

    if exp_norm == "exp1":
        rows: list[dict[str, Any]] = []
        slope_path = results_dir / "null_slope_check.json"
        if slope_path.exists():
            try:
                slope_obj = json.loads(slope_path.read_text(encoding="utf-8"))
                rows.append({
                    "metric": "panel_A_slope",
                    "value": float(slope_obj.get("slope", float("nan"))),
                })
                ci = slope_obj.get("slope_ci", [float("nan"), float("nan")])
                rows.append({"metric": "panel_A_slope_ci_lo", "value": float(ci[0])})
                rows.append({"metric": "panel_A_slope_ci_hi", "value": float(ci[1])})
                rows.append({"metric": "panel_A_pass", "value": int(bool(slope_obj.get("pass", False)))})
            except Exception as exc:
                _warn(f"null_slope_check parse failed: {type(exc).__name__}: {exc}")

        phase_path = results_dir / "summary_phase.csv"
        if phase_path.exists():
            try:
                phase_df = pd.read_csv(phase_path)
                if {"xi_ratio", "mean_prob_kappa_gt_u0"}.issubset(set(phase_df.columns)):
                    below = phase_df.loc[phase_df["xi_ratio"] < 1.0, "mean_prob_kappa_gt_u0"]
                    above = phase_df.loc[phase_df["xi_ratio"] > 1.0, "mean_prob_kappa_gt_u0"]
                    rows.append({"metric": "panel_B_prob_below_xi_crit", "value": float(below.mean()) if len(below) else float("nan")})
                    rows.append({"metric": "panel_B_prob_above_xi_crit", "value": float(above.mean()) if len(above) else float("nan")})
                    if len(below) and len(above):
                        rows.append({"metric": "panel_B_separation", "value": float(above.mean() - below.mean())})
            except Exception as exc:
                _warn(f"summary_phase parse failed: {type(exc).__name__}: {exc}")
        if parse_warnings:
            rows.extend(_warning_table().to_dict(orient="records"))
        return pd.DataFrame(rows)

    summary_path = results_dir / "summary.csv"
    raw_path = results_dir / "raw_results.csv"
    if not summary_path.exists():
        _warn(f"summary.csv missing: {summary_path}")
        return _warning_table()

    try:
        summary_df = pd.read_csv(summary_path)
    except Exception as exc:
        _warn(f"summary.csv parse failed: {type(exc).__name__}: {exc}")
        return _warning_table()

    if summary_df.empty:
        return summary_df

    preferred_group_keys = ["method", "variant", "alpha_kappa", "beta_kappa", "setting_id", "p0_true"]
    group_keys = [c for c in preferred_group_keys if c in set(summary_df.columns)]
    if "method" in set(summary_df.columns):
        group_keys = ["method"]
    elif "variant" in set(summary_df.columns):
        group_keys = ["variant"]
    elif {"alpha_kappa", "beta_kappa"}.issubset(set(summary_df.columns)):
        group_keys = ["alpha_kappa", "beta_kappa"]
    elif "setting_id" in set(summary_df.columns):
        group_keys = ["setting_id"]
    elif "p0_true" in set(summary_df.columns):
        group_keys = ["p0_true"]
    else:
        group_keys = []

    numeric_cols = [c for c in summary_df.columns if c not in set(group_keys) and pd.api.types.is_numeric_dtype(summary_df[c])]
    if group_keys:
        if numeric_cols:
            compact = summary_df.groupby(group_keys, as_index=False)[numeric_cols].mean()
        else:
            compact = summary_df[group_keys].copy()
    else:
        compact = summary_df.copy()

    if raw_path.exists() and ("method" in set(group_keys)):
        try:
            raw_df = pd.read_csv(raw_path)
            if {"method", "converged"}.issubset(set(raw_df.columns)):
                cdf = raw_df.groupby("method", as_index=False).agg(
                    n_rows=("method", "count"),
                    n_converged=("converged", lambda s: int(s.fillna(False).astype(bool).sum())),
                )
                cdf["converged_rate"] = cdf["n_converged"] / cdf["n_rows"].clip(lower=1)
                compact = cdf.merge(compact, on="method", how="left")
                compact["method_label"] = compact["method"].map(method_result_label)
        except Exception as exc:
            compact["parse_warning_raw_results"] = f"{type(exc).__name__}: {exc}"
    elif "method" in set(compact.columns):
        compact["method_label"] = compact["method"].map(method_result_label)
    elif "variant" in set(compact.columns):
        compact["variant_label"] = compact["variant"].map(method_result_label)

    if "mse_overall" in set(compact.columns):
        compact = compact.sort_values(["mse_overall"], ascending=True, kind="stable")
    return compact.reset_index(drop=True)


def _markdown_table(df, max_rows: int = 30) -> str:
    if df is None or getattr(df, "empty", True):
        return "_No rows._"
    rows = df.head(int(max_rows)).copy()
    cols = [str(c) for c in rows.columns.tolist()]
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    header = "| " + " | ".join(cols) + " |"

    def _fmt(v: Any) -> str:
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return "nan"
            return f"{float(v):.6g}"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return str(v)

    body = []
    for _, r in rows.iterrows():
        body.append("| " + " | ".join(_fmt(r[c]) for c in rows.columns) + " |")
    if len(df) > len(rows):
        body.append(f"\n_Only first {len(rows)} rows shown; total rows: {len(df)}._")
    return "\n".join([header, sep] + body)


def _write_markdown_run_summary(
    *,
    exp_key: str,
    timestamp: str,
    run_dir: Path,
    result_paths: dict[str, Any],
    analysis_result: dict[str, Any],
    summary_table,
) -> Path:
    lines: list[str] = []
    lines.append(f"# {str(exp_key).upper()} Run Summary")
    lines.append("")
    lines.append(f"- Timestamp: `{timestamp}`")
    lines.append(f"- Run directory: `{run_dir}`")
    lines.append("")
    lines.append("## Output Files")
    for k, v in sorted(result_paths.items(), key=lambda kv: kv[0]):
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Compact Summary Table")
    lines.append(_markdown_table(summary_table, max_rows=30))
    lines.append("")
    lines.append("## Analyzer Findings")
    findings = list((analysis_result or {}).get("findings", []))
    if findings:
        for i, item in enumerate(findings, start=1):
            lines.append(f"### Finding {i}")
            lines.append("```text")
            lines.append(str(item))
            lines.append("```")
    else:
        lines.append("_No findings generated._")
    lines.append("")

    out_path = run_dir / "run_summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _archive_experiment_outputs(
    *,
    save_root: Path,
    run_dir: Path,
    produced_paths: set[Path],
    result_paths: dict[str, Any],
) -> list[str]:
    artifacts_dir = ensure_dir(run_dir / "artifacts")
    to_copy: set[Path] = set(produced_paths)
    to_copy |= _collect_existing_paths(result_paths)

    copied: list[str] = []
    for src in sorted(to_copy):
        try:
            rel = src.relative_to(save_root)
        except Exception:
            rel = Path(src.name)
        dst = artifacts_dir / rel
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)
        copied.append(str(dst))
    return copied


def _finalize_experiment_run(
    *,
    exp_key: str,
    save_dir: str,
    results_dir: Path,
    produced_paths: set[Path] | None,
    result_paths: dict[str, Any],
    skip_run_analysis: bool = False,
    archive_artifacts: bool = True,
) -> dict[str, Any]:
    pd = load_pandas()

    ts = _timestamp_tag()
    run_dir = ensure_dir(results_dir / "runs" / ts)
    save_root = Path(save_dir)

    summary_table = _build_run_summary_table(exp_key=exp_key, results_dir=results_dir)
    summary_table_path = run_dir / "run_summary_table.csv"
    if summary_table is not None:
        if getattr(summary_table, "empty", False):
            pd.DataFrame().to_csv(summary_table_path, index=False)
        else:
            summary_table.to_csv(summary_table_path, index=False)

    if bool(skip_run_analysis):
        analysis_result = {"metrics": {}, "findings": ["Run-level analysis skipped by configuration."]}
    else:
        analysis_result = _analyze_single_experiment(exp_key=exp_key, results_dir=results_dir)
    analysis_json_path = run_dir / "run_analysis.json"
    analysis_json_path.write_text(json.dumps(analysis_result, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_md_path = _write_markdown_run_summary(
        exp_key=exp_key,
        timestamp=ts,
        run_dir=run_dir,
        result_paths=result_paths,
        analysis_result=analysis_result,
        summary_table=summary_table,
    )

    copied_artifacts: list[str] = []
    if bool(archive_artifacts):
        copied_artifacts = _archive_experiment_outputs(
            save_root=save_root,
            run_dir=run_dir,
            produced_paths=set(produced_paths or set()),
            result_paths=result_paths,
        )

    manifest_obj = RunManifest(
        exp_key=str(exp_key),
        timestamp=ts,
        run_dir=str(run_dir),
        result_paths=dict(result_paths),
        run_summary_table=str(summary_table_path),
        run_summary_md=str(summary_md_path),
        run_analysis_json=str(analysis_json_path),
        archived_artifacts=list(copied_artifacts),
    )
    manifest = manifest_obj.to_dict()
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (results_dir / "latest_run.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    out = dict(result_paths)
    out.update({
        "run_timestamp": ts,
        "run_dir": str(run_dir),
        "run_summary_table": str(summary_table_path),
        "run_summary_md": str(summary_md_path),
        "run_analysis_json": str(analysis_json_path),
        "run_manifest": str(run_dir / "run_manifest.json"),
        "latest_run": str(results_dir / "latest_run.json"),
    })
    return out

# ---------------------------------------------------------------------------
# Paired-converged subset helper
# ---------------------------------------------------------------------------

def _paired_converged_subset(
    raw,
    *,
    group_cols: Sequence[str],
    method_col: str,
    replicate_col: str,
    converged_col: str,
    required_cols: Sequence[str],
    method_levels: Sequence[str] | None = None,
    status_col: str | None = "status",
    status_ok_values: Sequence[str] | None = ("ok",),
    require_converged: bool = True,
):
    pd = load_pandas()

    group_cols_use = [str(c) for c in group_cols]
    if raw.empty:
        return raw.copy(), pd.DataFrame(
            columns=list(group_cols_use) + ["n_total_replicates", "n_common_replicates", "common_rate", "methods_required", "methods_list"]
        )
    work = raw.copy()
    work[method_col] = work[method_col].astype(str)
    methods_present = sorted(set(work[method_col].tolist()))
    methods_target = [str(m) for m in (method_levels or methods_present) if str(m) in set(methods_present)]
    if not methods_target:
        return work.iloc[0:0].copy(), pd.DataFrame(
            columns=list(group_cols_use) + ["n_total_replicates", "n_common_replicates", "common_rate", "methods_required", "methods_list"]
        )
    work = work.loc[work[method_col].isin(methods_target)].copy()
    if bool(require_converged):
        valid = work[converged_col].fillna(False).astype(bool)
    else:
        valid = pd.Series(True, index=work.index)
    if status_col is not None and str(status_col) in work.columns and status_ok_values is not None:
        allowed = {str(v).strip().lower() for v in status_ok_values}
        valid &= work[str(status_col)].astype(str).str.strip().str.lower().isin(allowed)
    for c in required_cols:
        valid &= work[c].notna()
    work["_pair_valid"] = valid
    key_cols = list(group_cols_use) + [str(replicate_col)]
    pivot = work.pivot_table(index=key_cols, columns=method_col, values="_pair_valid", aggfunc="max")
    for m in methods_target:
        if m not in pivot.columns:
            pivot[m] = False
    common_idx = pivot[methods_target].fillna(False).all(axis=1)
    common_keys = pivot.loc[common_idx].reset_index()[key_cols]
    paired = work.merge(common_keys, on=key_cols, how="inner").drop(columns=["_pair_valid"], errors="ignore")
    if group_cols_use:
        total = work.groupby(group_cols_use, as_index=False).agg(n_total_replicates=(replicate_col, "nunique"))
        if common_keys.empty:
            common = total[group_cols_use].copy()
            common["n_common_replicates"] = 0
        else:
            common = common_keys.groupby(group_cols_use, as_index=False).agg(n_common_replicates=(replicate_col, "nunique"))
        stats = total.merge(common, on=group_cols_use, how="left")
    else:
        stats = pd.DataFrame([{
            "n_total_replicates": int(work[replicate_col].nunique()),
            "n_common_replicates": int(common_keys[replicate_col].nunique()) if not common_keys.empty else 0,
        }])
    stats["n_common_replicates"] = stats["n_common_replicates"].fillna(0).astype(int)
    stats["n_total_replicates"] = stats["n_total_replicates"].fillna(0).astype(int)
    stats["common_rate"] = stats["n_common_replicates"] / stats["n_total_replicates"].clip(lower=1)
    stats["methods_required"] = int(len(methods_target))
    stats["methods_list"] = "|".join(methods_target)
    return paired, stats




