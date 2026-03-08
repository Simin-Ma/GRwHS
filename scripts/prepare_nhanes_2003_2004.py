from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd


REQUIRED_FILES = {
    "demo": "DEMO_C.xpt",
    "biochem": "L40_C.xpt",
    "creatinine": "L16_C.xpt",
    "bmx": "BMX_C.xpt",
    "metals": "L06BMT_C.xpt",
    "phthalates": "L24PH_C.xpt",
    "ocp": "L28OCP_C.xpt",
    "pbde": "L28PBE_C.xpt",
    "pah": "L31PAH_C.xpt",
}

OUTCOME_VAR = "LBXSGTSI"
BASE_COVARIATES = ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDFMPIR"]
BMI_VAR = "BMXBMI"
UCR_VAR = "URXUCR"

EXPOSURE_GROUPS: Dict[str, List[str]] = {
    "metals": ["LBXBPB", "LBXBCD", "LBXTHG"],
    "phthalates": ["URXMEP", "URXMBP", "URXMIB", "URXMHP", "URXMHH", "URXMOH", "URXMZP"],
    "organochlorines": ["LBXPDE", "LBXPDT", "LBXBHC", "LBXOXY", "LBXTNA", "LBXHPE", "LBXMIR", "LBXHCB"],
    "pbdes": ["LBXBR2", "LBXBR3", "LBXBR5", "LBXBR6", "LBXBR7", "LBXBR8", "LBXBR9"],
    "pahs": ["URXP01", "URXP02", "URXP03", "URXP04", "URXP05", "URXP06", "URXP07", "URXP10", "URXP17", "URXP19"],
}

VARIABLE_LABELS: Dict[str, str] = {
    "LBXSGTSI": "gamma_glutamyl_transferase",
    "RIDAGEYR": "age_years",
    "RIAGENDR": "sex",
    "RIDRETH1": "race_ethnicity",
    "INDFMPIR": "poverty_income_ratio",
    "BMXBMI": "body_mass_index",
    "URXUCR": "urinary_creatinine",
    "LBXBPB": "blood_lead",
    "LBXBCD": "blood_cadmium",
    "LBXTHG": "blood_total_mercury",
    "URXMEP": "monoethyl_phthalate",
    "URXMBP": "mono_n_butyl_phthalate",
    "URXMIB": "mono_isobutyl_phthalate",
    "URXMHP": "mono_2_ethylhexyl_phthalate",
    "URXMHH": "mono_2_ethyl_5_hydroxyhexyl_phthalate",
    "URXMOH": "mono_2_ethyl_5_oxohexyl_phthalate",
    "URXMZP": "monobenzyl_phthalate",
    "LBXPDE": "p_p_dde",
    "LBXPDT": "p_p_ddt",
    "LBXBHC": "beta_hexachlorocyclohexane",
    "LBXOXY": "oxychlordane",
    "LBXTNA": "trans_nonachlor",
    "LBXHPE": "heptachlor_epoxide",
    "LBXMIR": "mirex",
    "LBXHCB": "hexachlorobenzene",
    "LBXBR2": "pbde_28",
    "LBXBR3": "pbde_47",
    "LBXBR5": "pbde_99",
    "LBXBR6": "pbde_100",
    "LBXBR7": "pbde_153",
    "LBXBR8": "pbde_154",
    "LBXBR9": "pbde_183",
    "URXP01": "hydroxynaphthalene_1",
    "URXP02": "hydroxynaphthalene_2",
    "URXP03": "hydroxyfluorene_2",
    "URXP04": "hydroxyfluorene_3",
    "URXP05": "hydroxyphenanthrene_1",
    "URXP06": "hydroxyphenanthrene_2",
    "URXP07": "hydroxyphenanthrene_3",
    "URXP10": "hydroxyfluorene_9",
    "URXP17": "hydroxypyrene_1",
    "URXP19": "hydroxyphenanthrene_total",
}


def _read_xpt(path: Path) -> pd.DataFrame:
    return pd.read_sas(path, format="xport", encoding="utf-8")


def _zscore(frame: pd.DataFrame) -> pd.DataFrame:
    centered = frame - frame.mean(axis=0)
    scales = frame.std(axis=0, ddof=0).replace(0.0, 1.0)
    return centered / scales


def _safe_log(frame: pd.DataFrame) -> pd.DataFrame:
    arr = frame.astype(float)
    if (arr <= 0).any().any():
        raise ValueError("Encountered non-positive values in variables expected to be strictly positive.")
    return np.log(arr)


def _missing_rates(frame: pd.DataFrame, variables: Iterable[str]) -> Dict[str, float]:
    return {
        col: float(frame[col].isna().mean())
        for col in variables
    }


def _group_index_from_groups(groups: Mapping[str, List[str]]) -> np.ndarray:
    index: List[int] = []
    for gid, columns in enumerate(groups.values()):
        index.extend([gid] * len(columns))
    return np.asarray(index, dtype=np.int32)


def _write_lines(path: Path, values: Iterable[str]) -> None:
    path.write_text("\n".join(map(str, values)) + "\n", encoding="utf-8")


def build_dataset(raw_dir: Path, out_dir: Path) -> Dict[str, object]:
    paths = {key: raw_dir / value for key, value in REQUIRED_FILES.items()}
    missing_files = [str(path) for path in paths.values() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing NHANES files: {missing_files}")

    demo = _read_xpt(paths["demo"])[BASE_COVARIATES].copy()
    data = demo.merge(_read_xpt(paths["biochem"])[["SEQN", OUTCOME_VAR]], on="SEQN", how="left")
    data = data.merge(_read_xpt(paths["creatinine"])[["SEQN", UCR_VAR]], on="SEQN", how="left")
    data = data.merge(_read_xpt(paths["bmx"])[["SEQN", BMI_VAR]], on="SEQN", how="left")
    data = data.merge(_read_xpt(paths["metals"])[["SEQN", *EXPOSURE_GROUPS["metals"]]], on="SEQN", how="left")
    data = data.merge(_read_xpt(paths["phthalates"])[["SEQN", *EXPOSURE_GROUPS["phthalates"]]], on="SEQN", how="left")
    data = data.merge(_read_xpt(paths["ocp"])[["SEQN", *EXPOSURE_GROUPS["organochlorines"]]], on="SEQN", how="left")
    data = data.merge(_read_xpt(paths["pbde"])[["SEQN", *EXPOSURE_GROUPS["pbdes"]]], on="SEQN", how="left")
    data = data.merge(_read_xpt(paths["pah"])[["SEQN", *EXPOSURE_GROUPS["pahs"]]], on="SEQN", how="left")

    analysis_vars = [
        OUTCOME_VAR,
        BMI_VAR,
        UCR_VAR,
        "RIDAGEYR",
        "RIAGENDR",
        "RIDRETH1",
        "INDFMPIR",
        *[feature for columns in EXPOSURE_GROUPS.values() for feature in columns],
    ]
    adult = data[data["RIDAGEYR"] >= 18].copy()
    complete = adult.dropna(subset=analysis_vars).copy()
    complete.sort_values("SEQN", inplace=True)
    complete.reset_index(drop=True, inplace=True)

    exposure_columns = [feature for columns in EXPOSURE_GROUPS.values() for feature in columns]
    exposure_log = _safe_log(complete[exposure_columns])
    exposure_scaled = _zscore(exposure_log)

    y_log = np.log(complete[OUTCOME_VAR].astype(float).to_numpy())
    y_mean = float(y_log.mean())
    y_std = float(y_log.std(ddof=0)) if len(y_log) else 1.0
    if y_std == 0.0:
        y_std = 1.0
    y_centered = y_log - y_mean
    y_z = y_centered / y_std

    covariate_continuous = pd.DataFrame(
        {
            "RIDAGEYR": complete["RIDAGEYR"].astype(float),
            "INDFMPIR": complete["INDFMPIR"].astype(float),
            "BMXBMI": complete[BMI_VAR].astype(float),
            "log_URXUCR": np.log(complete[UCR_VAR].astype(float)),
        }
    )
    covariate_continuous_scaled = _zscore(covariate_continuous)
    covariate_categorical = pd.get_dummies(
        complete[["RIAGENDR", "RIDRETH1"]].astype("Int64"),
        columns=["RIAGENDR", "RIDRETH1"],
        prefix=["RIAGENDR", "RIDRETH1"],
        drop_first=True,
        dtype=float,
    )
    covariates_raw = pd.concat([covariate_continuous, covariate_categorical], axis=1)
    covariates_model = pd.concat([covariate_continuous_scaled, covariate_categorical], axis=1)

    design_raw = pd.concat([exposure_log, covariates_raw], axis=1)
    design_for_runner = exposure_log.copy()

    group_map_runner: Dict[str, int] = {}
    feature_names_runner = list(design_for_runner.columns)
    gid = 0
    for group_name, columns in EXPOSURE_GROUPS.items():
        for column in columns:
            group_map_runner[column] = gid
        gid += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    runner_dir = out_dir / "runner_ready"
    analysis_dir = out_dir / "analysis_bundle"
    runner_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    np.save(runner_dir / "X.npy", design_for_runner.to_numpy(dtype=np.float32))
    np.save(runner_dir / "C.npy", covariates_raw.to_numpy(dtype=np.float32))
    np.save(runner_dir / "y.npy", y_log.astype(np.float32))
    np.save(analysis_dir / "seqn.npy", complete["SEQN"].to_numpy(dtype=np.int64))

    np.savez_compressed(
        analysis_dir / "nhanes_2003_2004_ggt_analysis.npz",
        seqn=complete["SEQN"].to_numpy(dtype=np.int64),
        y_log=y_log.astype(np.float32),
        y_centered=y_centered.astype(np.float32),
        y_zscore=y_z.astype(np.float32),
        X_exposure_log=exposure_log.to_numpy(dtype=np.float32),
        X_exposure_scaled=exposure_scaled.to_numpy(dtype=np.float32),
        C_raw=covariates_raw.to_numpy(dtype=np.float32),
        C_model=covariates_model.to_numpy(dtype=np.float32),
        X_design_raw=design_raw.to_numpy(dtype=np.float32),
        X_runner=design_for_runner.to_numpy(dtype=np.float32),
        group_index_exposure=_group_index_from_groups(EXPOSURE_GROUPS),
        exposure_feature_names=np.asarray(exposure_columns, dtype=object),
        covariate_feature_names=np.asarray(list(covariates_model.columns), dtype=object),
        runner_feature_names=np.asarray(feature_names_runner, dtype=object),
    )

    complete.to_csv(analysis_dir / "nhanes_2003_2004_ggt_complete_cases.csv", index=False)
    _write_lines(runner_dir / "feature_names.txt", feature_names_runner)
    _write_lines(runner_dir / "covariate_feature_names.txt", covariates_raw.columns)
    _write_lines(analysis_dir / "exposure_feature_names.txt", exposure_columns)
    _write_lines(analysis_dir / "covariate_feature_names.txt", covariates_model.columns)
    (runner_dir / "group_map.json").write_text(json.dumps(group_map_runner, indent=2), encoding="utf-8")

    metadata = {
        "source_cycle": "NHANES 2003-2004",
        "population": "adults >= 18 years",
        "outcome": OUTCOME_VAR,
        "outcome_label": VARIABLE_LABELS[OUTCOME_VAR],
        "complete_case": True,
        "adult_n": int(len(adult)),
        "complete_case_n": int(len(complete)),
        "exposure_count": int(len(exposure_columns)),
        "covariate_count_model": int(covariates_model.shape[1]),
        "covariate_count_runner": int(covariates_raw.shape[1]),
        "runner_feature_count": int(design_for_runner.shape[1]),
        "selected_exposure_groups": EXPOSURE_GROUPS,
        "variable_labels": {
            column: VARIABLE_LABELS.get(column, column)
            for column in [OUTCOME_VAR, *exposure_columns]
        },
        "missing_rate_before_complete_case": _missing_rates(adult, analysis_vars),
        "runner_note": "Runner-ready exports now separate exposure matrix X from covariate matrix C; shrinkage groups apply only to the 35 exposures.",
        "files": {
            "runner_X": str((runner_dir / "X.npy").resolve()),
            "runner_C": str((runner_dir / "C.npy").resolve()),
            "runner_y": str((runner_dir / "y.npy").resolve()),
            "runner_feature_names": str((runner_dir / "feature_names.txt").resolve()),
            "runner_covariate_feature_names": str((runner_dir / "covariate_feature_names.txt").resolve()),
            "runner_group_map": str((runner_dir / "group_map.json").resolve()),
            "analysis_npz": str((analysis_dir / "nhanes_2003_2004_ggt_analysis.npz").resolve()),
            "complete_cases_csv": str((analysis_dir / "nhanes_2003_2004_ggt_complete_cases.csv").resolve()),
        },
    }

    metadata["covariate_feature_names_model"] = list(covariates_model.columns)

    metadata_path = out_dir / "dataset_summary.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NHANES 2003-2004 GGT/exposure dataset for GRRHS experiments.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/real/nhanes_2003_2004/raw"),
        help="Directory containing downloaded NHANES XPT files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/real/nhanes_2003_2004/processed"),
        help="Directory where processed arrays and metadata will be written.",
    )
    args = parser.parse_args()

    metadata = build_dataset(args.raw_dir, args.out_dir)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
