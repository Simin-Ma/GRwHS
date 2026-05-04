from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

XENA_URLS = {
    "clinical": "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/BRCA_clinicalMatrix",
    "mrna": "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.gz",
    "mirna": "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/miRNA_HiSeq_gene.gz",
    "cnv": "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/Gistic2_CopyNumber_Gistic2_all_data_by_genes.gz",
}
ANNOTATION_URLS = {
    "hgnc": "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt",
    "mirbase_hsa_gff3": "https://www.mirbase.org/download/hsa.gff3",
}

BLOCK_ORDER = ("mrna", "mirna", "cnv")
BLOCK_LABELS = {
    "mrna": "mRNA expression",
    "mirna": "miRNA expression",
    "cnv": "Gene-level copy number",
}
DEFAULT_FEATURES_PER_BLOCK = {
    "mrna": 150,
    "mirna": 120,
    "cnv": 150,
}
DEFAULT_LITE_MAX_FEATURES_PER_REFINED_GROUP = 2
CHROM_ORDER = [str(i) for i in range(1, 23)] + ["X", "Y"]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = Request(url, headers={"User-Agent": "GR-RHS data preparation"})
    with urlopen(req, timeout=120) as response, tmp.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    tmp.replace(dest)


def _parse_chromosome(location: object) -> str | None:
    loc = "" if pd.isna(location) else str(location).strip()
    if not loc:
        return None
    if loc.lower().startswith("chr"):
        loc = loc[3:]
    chrom = []
    for char in loc:
        if char.isdigit() or char.upper() in {"X", "Y"}:
            chrom.append(char.upper())
            continue
        break
    value = "".join(chrom)
    if value in CHROM_ORDER:
        return value
    return None


def _add_symbol_mapping(mapping: dict[str, str], key: object, chrom: str | None, *, overwrite: bool) -> None:
    if chrom is None or pd.isna(key):
        return
    for item in str(key).replace('"', "").split("|"):
        symbol = item.strip()
        if not symbol:
            continue
        old = mapping.get(symbol)
        if old is None:
            mapping[symbol] = chrom
        elif old != chrom and overwrite:
            mapping[symbol] = ""


def _load_hgnc_chromosome_map(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    mapping: dict[str, str] = {}
    for _, row in frame.iterrows():
        chrom = _parse_chromosome(row.get("location"))
        _add_symbol_mapping(mapping, row.get("symbol"), chrom, overwrite=True)
    for _, row in frame.iterrows():
        chrom = _parse_chromosome(row.get("location"))
        _add_symbol_mapping(mapping, row.get("alias_symbol"), chrom, overwrite=False)
        _add_symbol_mapping(mapping, row.get("prev_symbol"), chrom, overwrite=False)
    return {key: value for key, value in mapping.items() if value}


def _parse_gff3_attributes(attr: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in str(attr).split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        out[key] = value
    return out


def _load_mirbase_mimat_chromosome_map(path: Path) -> dict[str, str]:
    candidates: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "miRNA":
                continue
            chrom = _parse_chromosome(fields[0])
            attrs = _parse_gff3_attributes(fields[8])
            mimat = attrs.get("Alias") or attrs.get("ID")
            if chrom is None or not mimat:
                continue
            candidates.setdefault(str(mimat), set()).add(chrom)
    return {key: next(iter(vals)) for key, vals in candidates.items() if len(vals) == 1}


def _read_xena_matrix(path: Path) -> pd.DataFrame:
    compression = "gzip" if path.suffix == ".gz" else None
    frame = pd.read_csv(path, sep="\t", compression=compression, index_col=0, low_memory=False)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    return frame


def _tumor_patient_columns(columns: list[str]) -> dict[str, str]:
    patient_to_col: dict[str, str] = {}
    for col in columns:
        parts = str(col).split("-")
        if len(parts) < 4:
            continue
        sample_type = parts[3][:2]
        if sample_type != "01":
            continue
        patient = "-".join(parts[:3])
        patient_to_col.setdefault(patient, str(col))
    return patient_to_col


def _dedupe_feature_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        key = str(name).strip() or "unnamed"
        count = seen.get(key, 0)
        seen[key] = count + 1
        out.append(key if count == 0 else f"{key}#{count + 1}")
    return out


def _clean_numeric_matrix(frame: pd.DataFrame, patients: list[str], *, block: str) -> pd.DataFrame:
    col_map = _tumor_patient_columns(list(frame.columns))
    missing = [pid for pid in patients if pid not in col_map]
    if missing:
        patients = [pid for pid in patients if pid in col_map]
    cols = [col_map[pid] for pid in patients]
    out = frame.loc[:, cols].T
    out.index = patients
    out.columns = [f"{block}::{name}" for name in _dedupe_feature_names([str(c) for c in out.columns])]
    out = out.apply(pd.to_numeric, errors="coerce")
    return out


def _select_top_variance(
    matrix: pd.DataFrame,
    *,
    n_features: int,
    max_missing_fraction: float,
) -> pd.DataFrame:
    miss = matrix.isna().mean(axis=0)
    kept = matrix.loc[:, miss <= float(max_missing_fraction)]
    if kept.empty:
        raise ValueError("No features remain after missingness filtering.")
    med = kept.median(axis=0, skipna=True)
    kept = kept.fillna(med)
    variances = kept.var(axis=0, ddof=0).sort_values(ascending=False)
    selected = variances.head(int(min(n_features, variances.shape[0]))).index.tolist()
    return kept.loc[:, selected]


def _load_clinical(path: Path) -> pd.DataFrame:
    clinical = pd.read_csv(path, sep="\t", dtype=str)
    clinical["patient_id"] = clinical["sampleID"].astype(str).str[:12]
    clinical = clinical.drop_duplicates("patient_id", keep="first").set_index("patient_id")
    return clinical


def _age_target(clinical: pd.DataFrame) -> pd.Series:
    col = "Age_at_Initial_Pathologic_Diagnosis_nature2012"
    age = pd.to_numeric(clinical[col], errors="coerce")
    return age.rename("age_at_initial_pathologic_diagnosis")


def _pam50_labels(clinical: pd.DataFrame) -> pd.Series:
    labels = clinical.get("PAM50Call_RNAseq")
    if labels is None:
        return pd.Series(dtype=str, name="pam50")
    labels = labels.astype(str).replace({"nan": np.nan, "NA": np.nan, "": np.nan})
    return labels.rename("pam50")


def _write_runner_ready(
    *,
    out_dir: Path,
    X: pd.DataFrame,
    y: pd.Series,
    block_sizes: dict[str, int],
    sample_ids: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", X.to_numpy(dtype=np.float32))
    np.save(out_dir / "y.npy", y.to_numpy(dtype=np.float32))
    (out_dir / "feature_names.txt").write_text("\n".join(X.columns) + "\n", encoding="utf-8")
    (out_dir / "sample_ids.txt").write_text("\n".join(sample_ids) + "\n", encoding="utf-8")

    group_map: dict[str, int] = {}
    labels: list[str] = []
    offset = 0
    for gid, block in enumerate(BLOCK_ORDER):
        size = int(block_sizes.get(block, 0))
        if size <= 0:
            continue
        labels.append(BLOCK_LABELS[block])
        for name in X.columns[offset: offset + size]:
            group_map[str(name)] = int(gid)
        offset += size
    (out_dir / "group_map.json").write_text(json.dumps(group_map, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "group_labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")


def _write_runner_ready_with_explicit_groups(
    *,
    out_dir: Path,
    X: pd.DataFrame,
    y: pd.Series,
    sample_ids: list[str],
    feature_to_group_label: dict[str, str],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = sorted(
        set(feature_to_group_label.values()),
        key=lambda item: (
            BLOCK_ORDER.index(item.split("::", 1)[0]) if item.split("::", 1)[0] in BLOCK_ORDER else 99,
            CHROM_ORDER.index(item.rsplit("chr", 1)[-1]) if item.rsplit("chr", 1)[-1] in CHROM_ORDER else 99,
            item,
        ),
    )
    label_to_gid = {label: gid for gid, label in enumerate(labels)}
    group_map = {name: label_to_gid[feature_to_group_label[name]] for name in X.columns}

    np.save(out_dir / "X.npy", X.to_numpy(dtype=np.float32))
    np.save(out_dir / "y.npy", y.to_numpy(dtype=np.float32))
    (out_dir / "feature_names.txt").write_text("\n".join(X.columns) + "\n", encoding="utf-8")
    (out_dir / "sample_ids.txt").write_text("\n".join(sample_ids) + "\n", encoding="utf-8")
    (out_dir / "group_map.json").write_text(json.dumps(group_map, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "group_labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")

    sizes = {label: int(sum(1 for value in feature_to_group_label.values() if value == label)) for label in labels}
    return {
        "n_groups": int(len(labels)),
        "group_sizes": sizes,
        "group_labels": labels,
    }


def _build_omics_chromosome_annotation(
    features: list[str],
    *,
    hgnc_map: dict[str, str],
    mirbase_map: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in features:
        block, raw_name = str(feature).split("::", 1)
        if block == "mirna":
            chrom = mirbase_map.get(raw_name)
            source = "miRBase v22 hsa.gff3"
        else:
            lookup_name = raw_name.split("|", 1)[0]
            chrom = hgnc_map.get(lookup_name)
            source = "HGNC complete set"
        group_label = f"{block}::chr{chrom}" if chrom else ""
        rows.append(
            {
                "feature_name": feature,
                "omics_block": block,
                "feature_symbol": raw_name,
                "chromosome": "" if chrom is None else chrom,
                "group_label": group_label,
                "annotation_source": source,
                "mapped": bool(chrom),
            }
        )
    return pd.DataFrame(rows)


def prepare_dataset(
    *,
    root: Path,
    features_per_block: dict[str, int],
    max_missing_fraction: float,
    lite_max_features_per_group: int,
    force_download: bool,
) -> dict[str, Any]:
    base = root / "data" / "real" / "tcga_brca_xena_multiomics"
    raw_dir = base / "raw"
    processed_dir = base / "processed"
    raw_paths = {
        "clinical": raw_dir / "BRCA_clinicalMatrix.tsv",
        "mrna": raw_dir / "HiSeqV2.gz",
        "mirna": raw_dir / "miRNA_HiSeq_gene.gz",
        "cnv": raw_dir / "Gistic2_CopyNumber_Gistic2_all_data_by_genes.gz",
    }
    annotation_paths = {
        "hgnc": raw_dir / "hgnc_complete_set.txt",
        "mirbase_hsa_gff3": raw_dir / "mirbase_hsa_v22.gff3",
    }

    if force_download:
        for path in raw_paths.values():
            if path.exists():
                path.unlink()
    for key, url in XENA_URLS.items():
        _download(url, raw_paths[key])
    for key, url in ANNOTATION_URLS.items():
        _download(url, annotation_paths[key])

    clinical = _load_clinical(raw_paths["clinical"])
    age = _age_target(clinical)
    pam50 = _pam50_labels(clinical)
    eligible = clinical.index[age.notna()].tolist()

    matrices: dict[str, pd.DataFrame] = {}
    selected_shapes: dict[str, list[int]] = {}
    for block in BLOCK_ORDER:
        frame = _read_xena_matrix(raw_paths[block])
        block_matrix = _clean_numeric_matrix(frame, eligible, block=block)
        selected = _select_top_variance(
            block_matrix,
            n_features=int(features_per_block[block]),
            max_missing_fraction=float(max_missing_fraction),
        )
        matrices[block] = selected
        selected_shapes[block] = [int(selected.shape[0]), int(selected.shape[1])]

    common = set(eligible)
    for matrix in matrices.values():
        common &= set(matrix.index)
    common_ids = sorted(common)
    if not common_ids:
        raise ValueError("No common tumor patients remain after aligning omics blocks.")

    X_blocks = []
    block_sizes: dict[str, int] = {}
    for block in BLOCK_ORDER:
        mat = matrices[block].loc[common_ids]
        X_blocks.append(mat)
        block_sizes[block] = int(mat.shape[1])
    X = pd.concat(X_blocks, axis=1)
    y_age = age.loc[common_ids].astype(float)
    pam50_aligned = pam50.reindex(common_ids)

    runner_dir = processed_dir / "runner_ready_age_blocks"
    _write_runner_ready(out_dir=runner_dir, X=X, y=y_age, block_sizes=block_sizes, sample_ids=common_ids)

    hgnc_map = _load_hgnc_chromosome_map(annotation_paths["hgnc"])
    mirbase_map = _load_mirbase_mimat_chromosome_map(annotation_paths["mirbase_hsa_gff3"])
    feature_annotation = _build_omics_chromosome_annotation(
        list(X.columns),
        hgnc_map=hgnc_map,
        mirbase_map=mirbase_map,
    )
    mapped_features = feature_annotation.loc[feature_annotation["mapped"], "feature_name"].astype(str).tolist()
    X_refined = X.loc[:, mapped_features]
    refined_group_labels = dict(
        zip(
            feature_annotation.loc[feature_annotation["mapped"], "feature_name"].astype(str),
            feature_annotation.loc[feature_annotation["mapped"], "group_label"].astype(str),
        )
    )
    refined_runner_dir = processed_dir / "runner_ready_age_omics_chromosome"
    refined_group_summary = _write_runner_ready_with_explicit_groups(
        out_dir=refined_runner_dir,
        X=X_refined,
        y=y_age,
        sample_ids=common_ids,
        feature_to_group_label=refined_group_labels,
    )

    lite_features: list[str] = []
    lite_group_labels: dict[str, str] = {}
    max_per_group = max(1, int(lite_max_features_per_group))
    for group_label in refined_group_summary["group_labels"]:
        members = [name for name in X_refined.columns if refined_group_labels.get(name) == group_label]
        chosen = members[:max_per_group]
        lite_features.extend(chosen)
        for name in chosen:
            lite_group_labels[name] = group_label
    X_lite = X_refined.loc[:, lite_features]
    lite_runner_dir = processed_dir / "runner_ready_age_omics_chromosome_lite"
    lite_group_summary = _write_runner_ready_with_explicit_groups(
        out_dir=lite_runner_dir,
        X=X_lite,
        y=y_age,
        sample_ids=common_ids,
        feature_to_group_label=lite_group_labels,
    )

    phenotype = pd.DataFrame(
        {
            "patient_id": common_ids,
            "age_at_initial_pathologic_diagnosis": y_age.to_numpy(dtype=float),
            "pam50": pam50_aligned.astype("string").fillna("").to_numpy(),
            "pam50_basal_vs_other": (pam50_aligned.astype(str).str.lower() == "basal").astype(int).to_numpy(),
        }
    )
    analysis_dir = processed_dir / "analysis_bundle"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    phenotype.to_csv(analysis_dir / "phenotype_aligned.tsv", sep="\t", index=False)
    pd.DataFrame({"feature_name": X.columns}).to_csv(analysis_dir / "selected_features.tsv", sep="\t", index=False)
    feature_annotation.to_csv(analysis_dir / "selected_feature_chromosome_annotation.tsv", sep="\t", index=False)

    manifest = {
        "dataset_id": "tcga_brca_xena_multiomics_age_blocks",
        "source": "UCSC Xena TCGA.BRCA.sampleMap",
        "download_urls": XENA_URLS,
        "target": "Age_at_Initial_Pathologic_Diagnosis_nature2012",
        "omics_blocks": {block: BLOCK_LABELS[block] for block in BLOCK_ORDER},
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "block_sizes": block_sizes,
        "features_per_block_requested": {k: int(v) for k, v in features_per_block.items()},
        "max_missing_fraction": float(max_missing_fraction),
        "sample_filter": "Primary tumor samples only (TCGA sample type 01), aligned to patients with non-missing age.",
        "feature_filter": "Within each omics block, features with missing fraction above threshold are removed; remaining missing values are median-imputed; top-variance features are selected without using the outcome.",
        "runner_ready": {
            "path_X": str(runner_dir.relative_to(root) / "X.npy"),
            "path_y": str(runner_dir.relative_to(root) / "y.npy"),
            "path_feature_names": str(runner_dir.relative_to(root) / "feature_names.txt"),
            "path_group_map": str(runner_dir.relative_to(root) / "group_map.json"),
            "path_group_labels": str(runner_dir.relative_to(root) / "group_labels.txt"),
        },
        "runner_ready_omics_chromosome": {
            "path_X": str(refined_runner_dir.relative_to(root) / "X.npy"),
            "path_y": str(refined_runner_dir.relative_to(root) / "y.npy"),
            "path_feature_names": str(refined_runner_dir.relative_to(root) / "feature_names.txt"),
            "path_group_map": str(refined_runner_dir.relative_to(root) / "group_map.json"),
            "path_group_labels": str(refined_runner_dir.relative_to(root) / "group_labels.txt"),
            "n_samples": int(X_refined.shape[0]),
            "n_features": int(X_refined.shape[1]),
            "n_groups": int(refined_group_summary["n_groups"]),
            "group_sizes": refined_group_summary["group_sizes"],
            "unmapped_or_ambiguous_features_dropped": int(X.shape[1] - X_refined.shape[1]),
            "grouping_rule": "Disjoint omics-by-chromosome refinement using HGNC chromosome locations for mRNA/CNV gene symbols and miRBase mature miRNA coordinates for MIMAT accessions.",
        },
        "runner_ready_omics_chromosome_lite": {
            "path_X": str(lite_runner_dir.relative_to(root) / "X.npy"),
            "path_y": str(lite_runner_dir.relative_to(root) / "y.npy"),
            "path_feature_names": str(lite_runner_dir.relative_to(root) / "feature_names.txt"),
            "path_group_map": str(lite_runner_dir.relative_to(root) / "group_map.json"),
            "path_group_labels": str(lite_runner_dir.relative_to(root) / "group_labels.txt"),
            "n_samples": int(X_lite.shape[0]),
            "n_features": int(X_lite.shape[1]),
            "n_groups": int(lite_group_summary["n_groups"]),
            "group_sizes": lite_group_summary["group_sizes"],
            "max_features_per_refined_group": int(max_per_group),
            "feature_selection_rule": "Within each refined omics-by-chromosome group, retain the first features inherited from the outcome-free top-variance omics-block ranking.",
        },
        "raw_file_sizes_bytes": {k: int(path.stat().st_size) for k, path in raw_paths.items()},
        "annotation_file_sizes_bytes": {k: int(path.stat().st_size) for k, path in annotation_paths.items()},
        "selected_matrix_shapes_by_block_before_alignment": selected_shapes,
        "notes": [
            "This asset is intentionally Gaussian-runner-ready for the existing GR-RHS real-data pipeline.",
            "PAM50 labels are saved in analysis_bundle/phenotype_aligned.tsv for future subtype/logistic experiments.",
        ],
    }
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "dataset_summary.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (raw_dir / "download_manifest.json").write_text(json.dumps(XENA_URLS, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TCGA-BRCA UCSC Xena multi-omics runner-ready data.")
    parser.add_argument("--features-per-mrna", type=int, default=DEFAULT_FEATURES_PER_BLOCK["mrna"])
    parser.add_argument("--features-per-mirna", type=int, default=DEFAULT_FEATURES_PER_BLOCK["mirna"])
    parser.add_argument("--features-per-cnv", type=int, default=DEFAULT_FEATURES_PER_BLOCK["cnv"])
    parser.add_argument("--max-missing-fraction", type=float, default=0.2)
    parser.add_argument("--lite-max-features-per-group", type=int, default=DEFAULT_LITE_MAX_FEATURES_PER_REFINED_GROUP)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    manifest = prepare_dataset(
        root=PROJECT_ROOT,
        features_per_block={
            "mrna": int(args.features_per_mrna),
            "mirna": int(args.features_per_mirna),
            "cnv": int(args.features_per_cnv),
        },
        max_missing_fraction=float(args.max_missing_fraction),
        lite_max_features_per_group=int(args.lite_max_features_per_group),
        force_download=bool(args.force_download),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
