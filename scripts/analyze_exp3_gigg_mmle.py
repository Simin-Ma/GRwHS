from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation_project.src.utils import ensure_dir, load_pandas, save_json


def _resolve_paths(session_dir: Path) -> tuple[Path, Path, Path]:
    raw_path = session_dir / "results" / "exp3a_main_benchmark" / "raw_results.csv"
    log_path = session_dir / "logs" / "exp3a_main_benchmark.log"
    out_dir = ensure_dir(session_dir / "results" / "exp3a_main_benchmark" / "gigg_mmle_diagnostics")
    return raw_path, log_path, out_dir


def _bool_col(series):
    return series.fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])


def _parse_json_list(text: Any) -> list[float]:
    raw = str(text).strip()
    if not raw:
        return []
    try:
        obj = json.loads(raw)
    except Exception:
        return []
    if not isinstance(obj, list):
        return []
    out: list[float] = []
    for item in obj:
        try:
            val = float(item)
        except Exception:
            continue
        if np.isfinite(val):
            out.append(val)
    return out


def _write_partial_log_summary(log_path: Path, out_dir: Path) -> dict[str, Any]:
    pd = load_pandas()
    text = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    pat = re.compile(
        r"setting_id=(?P<setting_id>\d+).*?"
        r"replicate_id=(?P<replicate_id>\d+).*?"
        r"method=(?P<method>\S+).*?"
        r"signal=(?P<signal>\S+).*?"
        r"group_config=(?P<group_config>\S+).*?"
        r"env_id=(?P<env_id>\S+).*?"
        r"status=(?P<status>\S+).*?"
        r"converged=(?P<converged>\S+).*?"
        r"fit_attempts=(?P<fit_attempts>\d+).*?"
        r"runtime_seconds=(?P<runtime_seconds>[-+0-9.eE]+).*?"
        r"mse_overall=(?P<mse_overall>[-+0-9.eE]+|nan).*?"
        r"mse_null=(?P<mse_null>[-+0-9.eE]+|nan).*?"
        r"mse_signal=(?P<mse_signal>[-+0-9.eE]+|nan).*?"
        r"lpd_test=(?P<lpd_test>[-+0-9.eE]+|nan)"
    )
    rows: list[dict[str, Any]] = []
    for line in text:
        m = pat.search(line)
        if not m:
            continue
        row = m.groupdict()
        row["setting_id"] = int(row["setting_id"])
        row["replicate_id"] = int(row["replicate_id"])
        row["fit_attempts"] = int(row["fit_attempts"])
        for key in ["runtime_seconds", "mse_overall", "mse_null", "mse_signal", "lpd_test"]:
            try:
                row[key] = float(row[key])
            except Exception:
                row[key] = float("nan")
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        save_json({"status": "no_rows", "source": str(log_path)}, out_dir / "partial_log_summary.json")
        return {"status": "no_rows", "source": str(log_path)}
    ok_mask = df["status"].astype(str).str.lower().eq("ok") & df["converged"].astype(str).str.lower().eq("true")
    df_ok = df.loc[ok_mask].copy()
    summary = (
        df_ok.groupby(["setting_id", "signal", "group_config", "env_id", "method"], as_index=False)
        .agg(
            n_ok=("replicate_id", "count"),
            mse_overall_mean=("mse_overall", "mean"),
            mse_null_mean=("mse_null", "mean"),
            mse_signal_mean=("mse_signal", "mean"),
            lpd_test_mean=("lpd_test", "mean"),
        )
    )
    summary.to_csv(out_dir / "partial_log_method_summary.csv", index=False)
    save_json(
        {
            "status": "partial_log_only",
            "source": str(log_path),
            "n_rows": int(df.shape[0]),
            "n_ok_rows": int(df_ok.shape[0]),
            "available_settings": sorted(int(v) for v in df["setting_id"].dropna().unique().tolist()),
        },
        out_dir / "partial_log_summary.json",
    )
    return {
        "status": "partial_log_only",
        "source": str(log_path),
        "n_rows": int(df.shape[0]),
        "n_ok_rows": int(df_ok.shape[0]),
    }


def _analyze_raw(raw_path: Path, out_dir: Path) -> dict[str, Any]:
    pd = load_pandas()
    raw = pd.read_csv(raw_path)
    ok_mask = raw["status"].astype(str).str.lower().eq("ok") & _bool_col(raw["converged"])
    raw_ok = raw.loc[ok_mask].copy()

    group_keys = ["setting_id", "signal", "group_config", "env_id"]
    method_summary = (
        raw_ok.groupby(group_keys + ["method"], as_index=False)
        .agg(
            n_ok=("replicate_id", "count"),
            mse_overall_mean=("mse_overall", "mean"),
            mse_null_mean=("mse_null", "mean"),
            mse_signal_mean=("mse_signal", "mean"),
            lpd_test_mean=("lpd_test", "mean"),
            runtime_mean=("runtime_seconds", "mean"),
        )
    )
    method_summary.to_csv(out_dir / "gigg_mmle_method_summary_by_setting.csv", index=False)

    gigg = method_summary.loc[method_summary["method"] == "GIGG_MMLE"].copy()
    others = method_summary.loc[method_summary["method"] != "GIGG_MMLE"].copy()
    best_other = (
        others.groupby(group_keys, as_index=False)
        .agg(
            best_other_mse=("mse_overall_mean", "min"),
            best_other_lpd=("lpd_test_mean", "max"),
        )
    )
    by_setting = gigg.merge(best_other, on=group_keys, how="left")
    by_setting["mse_ratio_to_best_other"] = by_setting["mse_overall_mean"] / by_setting["best_other_mse"]
    by_setting["lpd_gap_to_best_other"] = by_setting["lpd_test_mean"] - by_setting["best_other_lpd"]
    by_setting["gigg_is_worse_mse"] = by_setting["mse_ratio_to_best_other"] > 1.05
    by_setting["gigg_is_much_worse_mse"] = by_setting["mse_ratio_to_best_other"] > 1.5
    by_setting.to_csv(out_dir / "gigg_mmle_vs_best_other_by_setting.csv", index=False)

    env_summary = (
        by_setting.groupby(["signal", "env_id"], as_index=False)
        .agg(
            n_settings=("setting_id", "count"),
            gigg_mse_mean=("mse_overall_mean", "mean"),
            best_other_mse_mean=("best_other_mse", "mean"),
            mse_ratio_mean=("mse_ratio_to_best_other", "mean"),
            mse_ratio_median=("mse_ratio_to_best_other", "median"),
            pct_worse_mse=("gigg_is_worse_mse", "mean"),
            pct_much_worse_mse=("gigg_is_much_worse_mse", "mean"),
            lpd_gap_mean=("lpd_gap_to_best_other", "mean"),
        )
    )
    env_summary.to_csv(out_dir / "gigg_mmle_env_summary.csv", index=False)

    mmle_rows = raw.loc[raw["method"] == "GIGG_MMLE"].copy()
    mmle_group_rows: list[dict[str, Any]] = []
    has_mmle_diag = "mmle_q_estimate_json" in mmle_rows.columns
    if has_mmle_diag:
        for _, row in mmle_rows.iterrows():
            q_vals = _parse_json_list(row.get("mmle_q_estimate_json", ""))
            a_vals = _parse_json_list(row.get("mmle_a_estimate_json", ""))
            for gid, q_val in enumerate(q_vals):
                a_val = float(a_vals[gid]) if gid < len(a_vals) else float("nan")
                mmle_group_rows.append(
                    {
                        "setting_id": int(row["setting_id"]),
                        "replicate_id": int(row["replicate_id"]),
                        "signal": str(row["signal"]),
                        "group_config": str(row["group_config"]),
                        "env_id": str(row["env_id"]),
                        "status": str(row["status"]),
                        "converged": bool(str(row["converged"]).lower() == "true"),
                        "group_id": int(gid),
                        "q_estimate": float(q_val),
                        "b_estimate": float(q_val),
                        "a_estimate": float(a_val),
                    }
                )
    mmle_group_df = pd.DataFrame(mmle_group_rows)
    if not mmle_group_df.empty:
        mmle_group_df.to_csv(out_dir / "gigg_mmle_qb_by_group.csv", index=False)
        mmle_group_summary = (
            mmle_group_df.groupby(["setting_id", "signal", "group_config", "env_id", "group_id"], as_index=False)
            .agg(
                q_mean=("q_estimate", "mean"),
                q_std=("q_estimate", "std"),
                q_min=("q_estimate", "min"),
                q_max=("q_estimate", "max"),
                a_mean=("a_estimate", "mean"),
                n_rows=("q_estimate", "count"),
            )
        )
        mmle_group_summary.to_csv(out_dir / "gigg_mmle_qb_group_summary.csv", index=False)

    summary_obj = {
        "status": "ok",
        "source": str(raw_path),
        "n_raw_rows": int(raw.shape[0]),
        "n_ok_rows": int(raw_ok.shape[0]),
        "n_settings_with_gigg": int(gigg.shape[0]),
        "has_mmle_diagnostics": bool(has_mmle_diag),
        "mse_ratio_mean": float(by_setting["mse_ratio_to_best_other"].mean()) if not by_setting.empty else float("nan"),
        "mse_ratio_median": float(by_setting["mse_ratio_to_best_other"].median()) if not by_setting.empty else float("nan"),
        "pct_settings_gigg_worse": float(by_setting["gigg_is_worse_mse"].mean()) if not by_setting.empty else float("nan"),
        "pct_settings_gigg_much_worse": float(by_setting["gigg_is_much_worse_mse"].mean()) if not by_setting.empty else float("nan"),
    }
    save_json(summary_obj, out_dir / "gigg_mmle_summary.json")
    return summary_obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze GIGG_MMLE performance and MMLE estimates for Exp3a.")
    parser.add_argument("--session-dir", required=True, help="Session directory, e.g. outputs/.../20260424_..._cli_exp3a")
    args = parser.parse_args()

    session_dir = Path(args.session_dir).resolve()
    raw_path, log_path, out_dir = _resolve_paths(session_dir)

    if raw_path.exists():
        result = _analyze_raw(raw_path, out_dir)
    elif log_path.exists():
        result = _write_partial_log_summary(log_path, out_dir)
    else:
        result = {"status": "missing_inputs", "session_dir": str(session_dir)}
        save_json(result, out_dir / "gigg_mmle_summary.json")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
