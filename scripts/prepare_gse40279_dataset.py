from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
import shutil
import ssl
import subprocess
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "real" / "gse40279_methylation_age"
RAW_DIR = DATASET_ROOT / "raw"
PROCESSED_DIR = DATASET_ROOT / "processed"
ANALYSIS_BUNDLE_DIR = PROCESSED_DIR / "analysis_bundle"

BETA_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE40nnn/GSE40279/suppl/GSE40279_average_beta.txt.gz"
SAMPLE_KEY_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE40nnn/GSE40279/suppl/GSE40279_sample_key.txt.gz"
SERIES_MATRIX_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE40nnn/GSE40279/matrix/GSE40279_series_matrix.txt.gz"
ANNOTATION_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL13nnn/GPL13534/suppl/"
    "GPL13534_HumanMethylation450_15017482_v.1.1.csv.gz"
)

AGE_RE = re.compile(r"age \(y\):\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


@dataclass(frozen=True)
class Paths:
    beta: Path
    sample_key: Path
    annotation: Path
    series_header_excerpt: Path
    download_manifest: Path


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _maybe_existing(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _curl_path() -> str | None:
    for name in ("curl.exe", "curl"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _download_to_path(url: str, destination: Path) -> None:
    if _maybe_existing(destination):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    curl_bin = _curl_path()
    if curl_bin:
        cmd = [
            curl_bin,
            "-k",
            "-L",
            "--fail",
            "--silent",
            "--show-error",
            "-C",
            "-",
            "-o",
            str(destination),
            url,
        ]
        subprocess.run(cmd, check=True)
        return

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    context = ssl._create_unverified_context()
    with urllib.request.urlopen(request, context=context, timeout=120) as response, destination.open("wb") as fh:
        shutil.copyfileobj(response, fh)


def _fetch_bytes(url: str, *, byte_range: str | None = None) -> bytes:
    curl_bin = _curl_path()
    if curl_bin:
        cmd = [curl_bin, "-k", "-L", "--fail", "--silent", "--show-error"]
        if byte_range:
            cmd.extend(["-r", byte_range])
        cmd.append(url)
        result = subprocess.run(cmd, capture_output=True, check=True)
        return result.stdout

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    if byte_range:
        request.add_header("Range", f"bytes={byte_range}")
    context = ssl._create_unverified_context()
    with urllib.request.urlopen(request, context=context, timeout=120) as response:
        return response.read()


def _parse_tsv_line(line: str) -> list[str]:
    reader = csv.reader([line.rstrip("\n")], delimiter="\t")
    return [cell.strip().strip('"') for cell in next(reader)]


def _decompress_partial_gzip_lines(raw_bytes: bytes) -> list[str]:
    lines: list[str] = []
    with gzip.GzipFile(fileobj=io.BytesIO(raw_bytes), mode="rb") as gz:
        wrapper = io.TextIOWrapper(gz, encoding="utf-8", errors="replace")
        try:
            for line in wrapper:
                lines.append(line)
        except EOFError:
            pass
    return lines


def _fetch_series_metadata(excerpt_path: Path) -> pd.DataFrame:
    lines = _decompress_partial_gzip_lines(_fetch_bytes(SERIES_MATRIX_URL, byte_range="0-1048575"))
    excerpt_lines: list[str] = []
    source_names: list[str] | None = None
    geo_accessions: list[str] | None = None
    titles: list[str] | None = None
    age_values: list[float] | None = None
    source_sites: list[str] | None = None
    plates: list[str] | None = None
    genders: list[str] | None = None
    ethnicities: list[str] | None = None
    tissues: list[str] | None = None

    for line in lines:
        if line.startswith("!series_matrix_table_begin"):
            excerpt_lines.append(line)
            break
        excerpt_lines.append(line)
        if line.startswith("!Sample_source_name_ch1"):
            source_names = _parse_tsv_line(line)[1:]
        elif line.startswith("!Sample_geo_accession"):
            geo_accessions = _parse_tsv_line(line)[1:]
        elif line.startswith("!Sample_title"):
            titles = _parse_tsv_line(line)[1:]
        elif line.startswith("!Sample_characteristics_ch1"):
            values = _parse_tsv_line(line)[1:]
            if not values:
                continue
            lower_first = values[0].lower()
            if lower_first.startswith("age (y):"):
                age_values = []
                for item in values:
                    match = AGE_RE.search(item)
                    if not match:
                        raise ValueError(f"Failed to parse age from series-matrix header entry: {item!r}")
                    age_values.append(float(match.group(1)))
            elif lower_first.startswith("source:"):
                source_sites = [item.split(":", 1)[1].strip() if ":" in item else item.strip() for item in values]
            elif lower_first.startswith("plate:"):
                plates = [item.split(":", 1)[1].strip() if ":" in item else item.strip() for item in values]
            elif lower_first.startswith("gender:"):
                genders = [item.split(":", 1)[1].strip() if ":" in item else item.strip() for item in values]
            elif lower_first.startswith("ethnicity:"):
                ethnicities = [item.split(":", 1)[1].strip() if ":" in item else item.strip() for item in values]
            elif lower_first.startswith("tissue:"):
                tissues = [item.split(":", 1)[1].strip() if ":" in item else item.strip() for item in values]

    if source_names is None or geo_accessions is None or age_values is None:
        raise RuntimeError("Series matrix header did not expose sample names, GEO accessions, and ages as expected.")

    n_samples = len(source_names)
    if len(geo_accessions) != n_samples or len(age_values) != n_samples:
        raise RuntimeError("Parsed series-matrix metadata has inconsistent sample counts.")

    excerpt_path.parent.mkdir(parents=True, exist_ok=True)
    excerpt_path.write_text("".join(excerpt_lines), encoding="utf-8")

    frame = pd.DataFrame(
        {
            "sample_name": source_names,
            "geo_accession": geo_accessions,
            "age_years": age_values,
            "title": titles if titles and len(titles) == n_samples else [""] * n_samples,
            "source_site": source_sites if source_sites and len(source_sites) == n_samples else [""] * n_samples,
            "plate": plates if plates and len(plates) == n_samples else [""] * n_samples,
            "gender": genders if genders and len(genders) == n_samples else [""] * n_samples,
            "ethnicity": ethnicities if ethnicities and len(ethnicities) == n_samples else [""] * n_samples,
            "tissue": tissues if tissues and len(tissues) == n_samples else [""] * n_samples,
        }
    )
    return frame


def _download_required_files() -> Paths:
    _ensure_dir(RAW_DIR)
    beta_path = RAW_DIR / "GSE40279_average_beta.txt.gz"
    sample_key_path = RAW_DIR / "GSE40279_sample_key.txt.gz"
    annotation_path = RAW_DIR / "GPL13534_HumanMethylation450_15017482_v.1.1.csv.gz"
    series_header_excerpt = RAW_DIR / "GSE40279_series_matrix_header_excerpt.txt"
    download_manifest = RAW_DIR / "download_manifest.json"

    _download_to_path(BETA_URL, beta_path)
    _download_to_path(SAMPLE_KEY_URL, sample_key_path)
    _download_to_path(ANNOTATION_URL, annotation_path)

    manifest = {
        "generated_at": _now_iso(),
        "files": {
            "beta_matrix": {"url": BETA_URL, "path": str(beta_path.resolve())},
            "sample_key": {"url": SAMPLE_KEY_URL, "path": str(sample_key_path.resolve())},
            "annotation": {"url": ANNOTATION_URL, "path": str(annotation_path.resolve())},
            "series_matrix_age_header": {"url": SERIES_MATRIX_URL, "path": str(series_header_excerpt.resolve())},
        },
    }
    download_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return Paths(
        beta=beta_path,
        sample_key=sample_key_path,
        annotation=annotation_path,
        series_header_excerpt=series_header_excerpt,
        download_manifest=download_manifest,
    )


def _first_gene_label(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    for token in text.split(";"):
        cleaned = token.strip()
        if cleaned:
            return cleaned
    return ""


def _load_annotation(annotation_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(annotation_path, compression="gzip", skiprows=7, low_memory=False)
    expected = {"Name", "CHR", "UCSC_RefGene_Name"}
    missing = expected.difference(frame.columns)
    if missing:
        raise RuntimeError(f"Annotation file is missing expected columns: {sorted(missing)}")

    subset = frame.loc[:, ["Name", "CHR", "UCSC_RefGene_Name"]].copy()
    subset.rename(columns={"Name": "probe_id", "CHR": "chromosome"}, inplace=True)
    subset["probe_id"] = subset["probe_id"].astype(str).str.strip()
    subset["chromosome"] = subset["chromosome"].astype(str).str.strip()
    subset["group_label"] = subset["UCSC_RefGene_Name"].map(_first_gene_label)

    valid_mask = (
        subset["probe_id"].str.startswith("cg")
        & subset["group_label"].astype(str).str.len().gt(0)
        & ~subset["chromosome"].isin({"", "nan", "NA", "0", "X", "Y", "chrX", "chrY"})
    )
    filtered = subset.loc[valid_mask, ["probe_id", "chromosome", "group_label"]].drop_duplicates("probe_id")
    filtered = filtered.reset_index(drop=True)
    return filtered


def _read_beta_header(beta_path: Path) -> list[str]:
    with gzip.open(beta_path, mode="rt", encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\n")
    cols = header.split("\t")
    if not cols or cols[0] != "ID_REF":
        raise RuntimeError(f"Unexpected beta-matrix header in {beta_path}: {header[:120]!r}")
    return cols[1:]


def _compute_probe_variances(
    beta_path: Path,
    *,
    annotation: pd.DataFrame,
    sample_columns: list[str],
    chunk_size: int,
) -> pd.DataFrame:
    annotation_indexed = annotation.set_index("probe_id")
    records: list[pd.DataFrame] = []

    reader = pd.read_csv(
        beta_path,
        sep="\t",
        compression="gzip",
        chunksize=chunk_size,
        low_memory=False,
    )
    for chunk_no, chunk in enumerate(reader, start=1):
        if "ID_REF" not in chunk.columns:
            raise RuntimeError("Beta-matrix chunk did not contain an ID_REF column.")
        probes = chunk["ID_REF"].astype(str).str.strip()
        annot = annotation_indexed.reindex(probes)
        valid = annot["group_label"].notna().to_numpy()
        if not bool(valid.any()):
            continue

        chunk_valid = chunk.loc[valid, ["ID_REF", *sample_columns]].copy()
        annot_valid = annot.loc[valid].reset_index(drop=True)
        values = chunk_valid.loc[:, sample_columns].to_numpy(dtype=np.float32, copy=False)
        finite_mask = np.isfinite(values).all(axis=1)
        if not bool(finite_mask.any()):
            continue

        values = values[finite_mask]
        variances = values.var(axis=1, dtype=np.float64)
        positive_mask = variances > 0.0
        if not bool(positive_mask.any()):
            continue

        kept_chunk = chunk_valid.loc[finite_mask].reset_index(drop=True).loc[positive_mask, ["ID_REF"]]
        kept_annot = annot_valid.loc[finite_mask].reset_index(drop=True).loc[positive_mask, ["chromosome", "group_label"]]
        records.append(
            pd.DataFrame(
                {
                    "probe_id": kept_chunk["ID_REF"].astype(str).tolist(),
                    "variance": variances[positive_mask],
                    "chromosome": kept_annot["chromosome"].astype(str).tolist(),
                    "group_label": kept_annot["group_label"].astype(str).tolist(),
                }
            )
        )
        if chunk_no % 25 == 0:
            print(f"[gse40279] processed {chunk_no} beta chunks for variance ranking...")

    if not records:
        raise RuntimeError("No eligible probes survived annotation and variance filtering.")

    ranked = pd.concat(records, ignore_index=True)
    ranked = ranked.sort_values(["variance", "probe_id"], ascending=[False, True], kind="stable").reset_index(drop=True)
    return ranked


def _select_features(
    ranked: pd.DataFrame,
    *,
    top_k: int,
    min_group_size: int,
    max_group_size: int,
) -> pd.DataFrame:
    capped = ranked.groupby("group_label", sort=False, group_keys=False).head(max_group_size).copy()
    eligible_counts = capped["group_label"].value_counts()
    eligible_groups = set(eligible_counts[eligible_counts >= min_group_size].index.tolist())
    capped = capped.loc[capped["group_label"].isin(eligible_groups)].reset_index(drop=True)
    if capped.empty:
        raise RuntimeError("No groups survived the min-group-size constraint after group-size truncation.")

    grouped_frames: list[tuple[str, float, pd.DataFrame]] = []
    for group_label, sub in capped.groupby("group_label", sort=False):
        sub_reset = sub.reset_index(drop=True)
        group_score = float(sub_reset["variance"].head(min(10, sub_reset.shape[0])).sum())
        grouped_frames.append((str(group_label), group_score, sub_reset))
    grouped_frames.sort(key=lambda item: (-item[1], item[0]))

    selected_group_frames: list[pd.DataFrame] = []
    selected_group_labels: list[str] = []
    available_features = 0
    for group_label, _, frame in grouped_frames:
        if available_features >= top_k:
            break
        selected_group_frames.append(frame)
        selected_group_labels.append(group_label)
        available_features += int(frame.shape[0])

    if not selected_group_frames:
        raise RuntimeError("Unable to seed any groups under the current top_k and min_group_size settings.")

    seeded_groups = [frame.iloc[:min_group_size].copy() for frame in selected_group_frames]
    selected = pd.concat(seeded_groups, ignore_index=True)
    remaining_slots = top_k - int(selected.shape[0])

    extras_parts = []
    for group_label, _, frame in grouped_frames:
        if group_label in selected_group_labels and frame.shape[0] > min_group_size:
            extras_parts.append(frame.iloc[min_group_size:].copy())
    if extras_parts and remaining_slots > 0:
        extras = pd.concat(extras_parts, ignore_index=True)
        extras = extras.sort_values(["variance", "probe_id"], ascending=[False, True], kind="stable").head(remaining_slots)
        selected = pd.concat([selected, extras], ignore_index=True)

    if selected.shape[0] != top_k:
        raise RuntimeError(
            f"Selection produced {selected.shape[0]} features, expected exactly {top_k}. "
            "The current group constraints are too strict for the requested top_k."
        )

    selected = selected.sort_values(["variance", "probe_id"], ascending=[False, True], kind="stable").reset_index(drop=True)
    group_order: list[str] = []
    seen: set[str] = set()
    for label in selected["group_label"].astype(str).tolist():
        if label not in seen:
            seen.add(label)
            group_order.append(label)
    group_to_id = {label: idx for idx, label in enumerate(group_order)}
    selected["group_id"] = selected["group_label"].map(group_to_id).astype(int)
    return selected.loc[:, ["probe_id", "variance", "chromosome", "group_label", "group_id"]]


def _extract_selected_matrix(
    beta_path: Path,
    *,
    selected_probe_ids: list[str],
    sample_columns: list[str],
    chunk_size: int,
) -> np.ndarray:
    wanted = set(selected_probe_ids)
    selected_rows: dict[str, np.ndarray] = {}

    reader = pd.read_csv(
        beta_path,
        sep="\t",
        compression="gzip",
        chunksize=chunk_size,
        low_memory=False,
    )
    for chunk_no, chunk in enumerate(reader, start=1):
        mask = chunk["ID_REF"].astype(str).isin(wanted)
        if bool(mask.any()):
            subset = chunk.loc[mask, ["ID_REF", *sample_columns]].copy()
            for row in subset.itertuples(index=False):
                probe_id = str(row[0])
                selected_rows[probe_id] = np.asarray(row[1:], dtype=np.float32)
        if chunk_no % 25 == 0:
            print(f"[gse40279] processed {chunk_no} beta chunks for matrix extraction...")

    missing = [probe_id for probe_id in selected_probe_ids if probe_id not in selected_rows]
    if missing:
        raise RuntimeError(f"Failed to recover {len(missing)} selected probes from beta matrix; first missing={missing[0]!r}")

    matrix = np.column_stack([selected_rows[probe_id] for probe_id in selected_probe_ids]).astype(np.float32, copy=False)
    return matrix


def _age_summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def _load_existing_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _prepare_dataset(
    *,
    variant: str,
    top_k: int,
    min_group_size: int,
    max_group_size: int,
    chunk_size: int,
) -> dict[str, Any]:
    paths = _download_required_files()
    _ensure_dir(PROCESSED_DIR)
    _ensure_dir(ANALYSIS_BUNDLE_DIR)

    sample_metadata = _fetch_series_metadata(paths.series_header_excerpt)
    sample_metadata_path = ANALYSIS_BUNDLE_DIR / "sample_metadata.tsv"
    sample_metadata.to_csv(sample_metadata_path, sep="\t", index=False)

    sample_key_path = ANALYSIS_BUNDLE_DIR / "sample_key.tsv"
    sample_key = pd.read_csv(paths.sample_key, sep="\t", header=None, names=["row_id", "sample_numeric_id", "sentrix_position"])
    sample_key.to_csv(sample_key_path, sep="\t", index=False)

    annotation = _load_annotation(paths.annotation)
    annotation_path = ANALYSIS_BUNDLE_DIR / "annotation_eligible.tsv.gz"
    annotation.to_csv(annotation_path, sep="\t", index=False, compression="gzip")

    sample_columns = _read_beta_header(paths.beta)
    if list(sample_metadata["sample_name"]) != sample_columns:
        raise RuntimeError("Sample order from series-matrix metadata does not match beta-matrix columns.")

    ranked = _compute_probe_variances(paths.beta, annotation=annotation, sample_columns=sample_columns, chunk_size=chunk_size)
    ranked_path = ANALYSIS_BUNDLE_DIR / f"{variant}_ranked_candidates.tsv.gz"
    ranked.to_csv(ranked_path, sep="\t", index=False, compression="gzip")

    selected = _select_features(
        ranked,
        top_k=top_k,
        min_group_size=min_group_size,
        max_group_size=max_group_size,
    )
    selected_path = ANALYSIS_BUNDLE_DIR / f"{variant}_selected_features.tsv"
    selected.to_csv(selected_path, sep="\t", index=False)

    selected_probe_ids = selected["probe_id"].astype(str).tolist()
    X = _extract_selected_matrix(
        paths.beta,
        selected_probe_ids=selected_probe_ids,
        sample_columns=sample_columns,
        chunk_size=chunk_size,
    )
    y = sample_metadata["age_years"].to_numpy(dtype=np.float32)

    runner_dir = PROCESSED_DIR / f"runner_ready_{variant}"
    _ensure_dir(runner_dir)

    np.save(runner_dir / "X.npy", X)
    np.save(runner_dir / "y.npy", y)
    _write_lines(runner_dir / "feature_names.txt", selected_probe_ids)

    group_order = (
        selected.loc[:, ["group_id", "group_label"]]
        .drop_duplicates()
        .sort_values("group_id", kind="stable")
        ["group_label"]
        .astype(str)
        .tolist()
    )
    _write_lines(runner_dir / "group_labels.txt", group_order)

    group_map = {probe_id: int(group_id) for probe_id, group_id in selected[["probe_id", "group_id"]].itertuples(index=False)}
    (runner_dir / "group_map.json").write_text(
        json.dumps(group_map, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    feature_group_sizes = selected["group_label"].value_counts()
    group_catalog = (
        selected.groupby(["group_id", "group_label"], as_index=False)
        .agg(feature_count=("probe_id", "count"), max_variance=("variance", "max"))
        .sort_values("group_id", kind="stable")
    )
    group_catalog_path = ANALYSIS_BUNDLE_DIR / f"{variant}_group_catalog.tsv"
    group_catalog.to_csv(group_catalog_path, sep="\t", index=False)

    summary_path = PROCESSED_DIR / "dataset_summary.json"
    summary = _load_existing_summary(summary_path)
    summary.setdefault("dataset_id", "gse40279_methylation_age")
    summary.setdefault("label", "GSE40279 human methylation age")
    summary["generated_at"] = _now_iso()
    summary["response"] = {
        "target_label": "chronological_age",
        "task": "gaussian",
        "unit": "years",
    }
    summary["source"] = {
        "geo_accession": "GSE40279",
        "beta_matrix_url": BETA_URL,
        "sample_key_url": SAMPLE_KEY_URL,
        "series_matrix_url": SERIES_MATRIX_URL,
        "annotation_url": ANNOTATION_URL,
    }
    summary["preprocessing"] = {
        "annotation_rule": (
            "Use the first non-empty gene symbol in UCSC_RefGene_Name as a single-gene proxy for "
            "nearest-gene grouping so each probe satisfies the loader single-group-id constraint."
        ),
        "removed_probes": [
            "missing gene annotation",
            "non-cg probes",
            "chrX probes",
            "chrY probes",
            "non-finite beta rows",
            "zero-variance rows",
        ],
        "screening_rule": "Global unsupervised variance ranking on filtered probes; y is never used for screening.",
        "group_constraints": {
            "min_group_size": int(min_group_size),
            "max_group_size": int(max_group_size),
            "selection_scheme": (
                "rank groups by the sum of their top 10 probe variances after filtering and per-group truncation, "
                "select enough groups to cover top_k, seed each selected group with its top 2 probes, then fill "
                "remaining slots by variance within those selected groups"
            ),
        },
    }
    summary.setdefault("variants", {})
    summary["variants"][variant] = {
        "top_k_target": int(top_k),
        "sample_count": int(X.shape[0]),
        "feature_count": int(X.shape[1]),
        "group_count": int(len(group_order)),
        "group_size_summary": {
            "min": int(feature_group_sizes.min()),
            "median": float(feature_group_sizes.median()),
            "max": int(feature_group_sizes.max()),
        },
        "age_summary": _age_summary(y),
        "input_files": {
            "beta_matrix": str(paths.beta.resolve()),
            "sample_key": str(paths.sample_key.resolve()),
            "series_matrix_header_excerpt": str(paths.series_header_excerpt.resolve()),
            "annotation": str(paths.annotation.resolve()),
        },
        "files": {
            "runner_X": str((runner_dir / "X.npy").resolve()),
            "runner_y": str((runner_dir / "y.npy").resolve()),
            "runner_feature_names": str((runner_dir / "feature_names.txt").resolve()),
            "runner_group_map": str((runner_dir / "group_map.json").resolve()),
            "runner_group_labels": str((runner_dir / "group_labels.txt").resolve()),
            "sample_metadata": str(sample_metadata_path.resolve()),
            "sample_key_tsv": str(sample_key_path.resolve()),
            "annotation_eligible": str(annotation_path.resolve()),
            "ranked_candidates": str(ranked_path.resolve()),
            "selected_features": str(selected_path.resolve()),
            "group_catalog": str(group_catalog_path.resolve()),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "variant": variant,
        "runner_dir": str(runner_dir.resolve()),
        "sample_count": int(X.shape[0]),
        "feature_count": int(X.shape[1]),
        "group_count": int(len(group_order)),
        "group_sizes_head": [int(v) for v in feature_group_sizes.sort_values(ascending=False).head(10).tolist()],
        "dataset_summary": str(summary_path.resolve()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare GSE40279 methylation-age data for real_data_experiment.")
    parser.add_argument("--variant", default="smoke", help="Output variant name, e.g. smoke or main.")
    parser.add_argument("--top-k", type=int, default=2000, help="Number of features to keep after filtering.")
    parser.add_argument("--min-group-size", type=int, default=2, help="Drop groups smaller than this size.")
    parser.add_argument("--max-group-size", type=int, default=30, help="Cap each group at this many features.")
    parser.add_argument("--chunk-size", type=int, default=4000, help="Chunk size for beta-matrix streaming.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = _prepare_dataset(
        variant=str(args.variant),
        top_k=int(args.top_k),
        min_group_size=int(args.min_group_size),
        max_group_size=int(args.max_group_size),
        chunk_size=int(args.chunk_size),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
