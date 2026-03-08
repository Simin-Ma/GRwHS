from __future__ import annotations

import argparse
import io
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer

try:
    import pyreadr
except ImportError as exc:  # pragma: no cover - runtime dependency for one-off dataset prep
    raise RuntimeError(
        "pyreadr is required to prepare the trust_experts dataset. "
        "Install it with `python -m pip install pyreadr`."
    ) from exc


SPARSEGL_VERSION = "1.1.1"
SPARSEGL_TARBALL_URL = f"https://cran.r-project.org/src/contrib/sparsegl_{SPARSEGL_VERSION}.tar.gz"
RDATA_MEMBER = "sparsegl/data/trust_experts.rda"

CATEGORICAL_COLUMNS = ["period", "region", "age", "gender", "raceethnicity"]
SPLINE_COLUMNS = ["cli", "hh_cmnty_cli"]
TARGET_COLUMN = "trust_experts"

GROUP_ORDER = [
    "period",
    "region",
    "age",
    "gender",
    "raceethnicity",
    "cli_spline",
    "hh_cmnty_cli_spline",
]


def _write_lines(path: Path, values: Iterable[str]) -> None:
    path.write_text("\n".join(map(str, values)) + "\n", encoding="utf-8")


def _download_sparsegl_rda(*, url: str = SPARSEGL_TARBALL_URL) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        raw = response.read()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        member = archive.getmember(RDATA_MEMBER)
        extracted = archive.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(f"Failed to extract {RDATA_MEMBER} from {url}")
        return extracted.read()


def _load_source_frame(*, url: str = SPARSEGL_TARBALL_URL) -> pd.DataFrame:
    raw_rda = _download_sparsegl_rda(url=url)
    tmp_path = Path("data/real/covid19_trust_experts/processed/_trust_experts_tmp.rda")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_bytes(raw_rda)
    try:
        result = pyreadr.read_r(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)
    if "trust_experts" not in result:
        raise KeyError("R data file does not contain a `trust_experts` object.")
    frame = result["trust_experts"].copy()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected trust_experts to decode into a pandas DataFrame.")
    return frame


def _build_categorical_block(frame: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]], Dict[str, int]]:
    blocks: List[pd.DataFrame] = []
    group_columns: Dict[str, List[str]] = {}
    level_counts: Dict[str, int] = {}

    for column in CATEGORICAL_COLUMNS:
        dummies = pd.get_dummies(frame[column], prefix=column, prefix_sep="=", dtype=np.float32)
        blocks.append(dummies)
        group_columns[column] = list(dummies.columns)
        level_counts[column] = int(dummies.shape[1])

    return pd.concat(blocks, axis=1), group_columns, level_counts


def _build_spline_block(frame: pd.DataFrame, *, degree: int = 3, n_basis: int = 10) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    blocks: List[pd.DataFrame] = []
    group_columns: Dict[str, List[str]] = {}
    n_knots = n_basis - degree + 1
    if n_knots < 2:
        raise ValueError("Requested spline basis is too small for the selected degree.")

    for column in SPLINE_COLUMNS:
        transformer = SplineTransformer(
            n_knots=n_knots,
            degree=degree,
            include_bias=True,
        )
        values = frame[[column]].to_numpy(dtype=np.float32)
        transformed = transformer.fit_transform(values).astype(np.float32)
        names = [f"{column}_bs_{idx:02d}" for idx in range(1, transformed.shape[1] + 1)]
        block = pd.DataFrame(transformed, columns=names, index=frame.index)
        blocks.append(block)
        group_columns[f"{column}_spline"] = names

    return pd.concat(blocks, axis=1), group_columns


def build_dataset(*, url: str = SPARSEGL_TARBALL_URL, out_dir: Path) -> Dict[str, object]:
    frame = _load_source_frame(url=url)
    frame = frame.dropna(subset=[TARGET_COLUMN, *CATEGORICAL_COLUMNS, *SPLINE_COLUMNS]).copy()
    frame.reset_index(drop=True, inplace=True)

    y = frame[TARGET_COLUMN].to_numpy(dtype=np.float32)

    categorical_block, categorical_groups, level_counts = _build_categorical_block(frame)
    spline_block, spline_groups = _build_spline_block(frame)
    X_frame = pd.concat([categorical_block, spline_block], axis=1)

    feature_names = list(X_frame.columns)
    group_columns = {**categorical_groups, **spline_groups}
    group_map: Dict[str, int] = {}
    group_sizes: Dict[str, int] = {}
    for gid, group_name in enumerate(GROUP_ORDER):
        cols = group_columns[group_name]
        group_sizes[group_name] = len(cols)
        for column in cols:
            group_map[column] = gid

    out_dir.mkdir(parents=True, exist_ok=True)
    runner_dir = out_dir / "runner_ready"
    analysis_dir = out_dir / "analysis_bundle"
    runner_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    X = X_frame.to_numpy(dtype=np.float32)
    np.save(runner_dir / "X.npy", X)
    np.save(runner_dir / "y.npy", y)
    _write_lines(runner_dir / "feature_names.txt", feature_names)
    (runner_dir / "group_map.json").write_text(json.dumps(group_map, indent=2), encoding="utf-8")

    frame.to_csv(analysis_dir / "trust_experts_raw.csv", index=False)
    np.savez_compressed(
        analysis_dir / "trust_experts_analysis.npz",
        X=X,
        y=y,
        feature_names=np.asarray(feature_names, dtype=object),
        group_order=np.asarray(GROUP_ORDER, dtype=object),
    )

    metadata = {
        "source": "CRAN sparsegl::trust_experts",
        "sparsegl_version": SPARSEGL_VERSION,
        "source_tarball": url,
        "n_rows": int(frame.shape[0]),
        "n_features": int(X.shape[1]),
        "target": TARGET_COLUMN,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "spline_columns": SPLINE_COLUMNS,
        "group_order": GROUP_ORDER,
        "group_sizes": group_sizes,
        "categorical_level_counts": level_counts,
        "target_summary": {
            "mean": float(np.mean(y)),
            "std": float(np.std(y, ddof=0)),
            "min": float(np.min(y)),
            "max": float(np.max(y)),
        },
        "files": {
            "runner_X": str((runner_dir / "X.npy").resolve()),
            "runner_y": str((runner_dir / "y.npy").resolve()),
            "runner_feature_names": str((runner_dir / "feature_names.txt").resolve()),
            "runner_group_map": str((runner_dir / "group_map.json").resolve()),
            "analysis_csv": str((analysis_dir / "trust_experts_raw.csv").resolve()),
            "analysis_npz": str((analysis_dir / "trust_experts_analysis.npz").resolve()),
        },
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the sparsegl trust_experts COVID-19 dataset for GRRHS experiments."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/real/covid19_trust_experts/processed"),
        help="Directory where runner-ready arrays and metadata will be written.",
    )
    parser.add_argument(
        "--source-url",
        type=str,
        default=SPARSEGL_TARBALL_URL,
        help="Tarball URL for the sparsegl source package containing trust_experts.rda.",
    )
    args = parser.parse_args()

    metadata = build_dataset(url=args.source_url, out_dir=args.out_dir)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
