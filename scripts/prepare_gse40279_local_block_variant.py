from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prepare_gse40279_dataset as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "real" / "gse40279_methylation_age"
PROCESSED_DIR = DATASET_ROOT / "processed"
ANALYSIS_BUNDLE_DIR = PROCESSED_DIR / "analysis_bundle"


@dataclass(frozen=True)
class VariantDefaults:
    top_k: int
    min_group_size: int
    max_group_size: int
    block_gap_bp: int
    chunk_size: int


VARIANT_DEFAULTS: dict[str, VariantDefaults] = {
    "micro_local_block_gap1000": VariantDefaults(
        top_k=200,
        min_group_size=3,
        max_group_size=0,
        block_gap_bp=1000,
        chunk_size=4000,
    ),
    "smoke_local_block_gap1000": VariantDefaults(
        top_k=2000,
        min_group_size=3,
        max_group_size=0,
        block_gap_bp=1000,
        chunk_size=4000,
    ),
}


def _resolve_settings(
    *,
    variant: str,
    top_k: int | None,
    min_group_size: int | None,
    max_group_size: int | None,
    block_gap_bp: int | None,
    chunk_size: int | None,
) -> VariantDefaults:
    defaults = VARIANT_DEFAULTS.get(str(variant), VARIANT_DEFAULTS["micro_local_block_gap1000"])
    return VariantDefaults(
        top_k=int(defaults.top_k if top_k is None else top_k),
        min_group_size=int(defaults.min_group_size if min_group_size is None else min_group_size),
        max_group_size=int(defaults.max_group_size if max_group_size is None else max_group_size),
        block_gap_bp=int(defaults.block_gap_bp if block_gap_bp is None else block_gap_bp),
        chunk_size=int(defaults.chunk_size if chunk_size is None else chunk_size),
    )


def _block_anchor_gene(labels: pd.Series) -> str:
    cleaned = [str(v).strip() for v in labels.tolist() if str(v).strip()]
    if not cleaned:
        return ""
    counts = Counter(cleaned)
    best_count = max(counts.values())
    best_labels = sorted(label for label, count in counts.items() if count == best_count)
    return best_labels[0]


def _load_local_block_annotation(annotation_path: Path, *, block_gap_bp: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(
        annotation_path,
        compression="gzip",
        skiprows=7,
        usecols=["Name", "CHR", "MAPINFO", "UCSC_RefGene_Name"],
        low_memory=False,
    )
    subset = frame.loc[:, ["Name", "CHR", "MAPINFO", "UCSC_RefGene_Name"]].copy()
    subset.rename(columns={"Name": "probe_id", "CHR": "chromosome", "MAPINFO": "position"}, inplace=True)
    subset["probe_id"] = subset["probe_id"].astype(str).str.strip()
    subset["chromosome"] = subset["chromosome"].astype(str).str.strip()
    subset["position"] = pd.to_numeric(subset["position"], errors="coerce")
    subset["anchor_gene"] = subset["UCSC_RefGene_Name"].map(base._first_gene_label)

    valid_mask = (
        subset["probe_id"].str.startswith("cg")
        & subset["position"].notna()
        & ~subset["chromosome"].isin({"", "nan", "NA", "0", "X", "Y", "chrX", "chrY"})
    )
    eligible = subset.loc[valid_mask, ["probe_id", "chromosome", "position", "anchor_gene"]].copy()
    eligible["position"] = eligible["position"].astype(int)
    eligible = eligible.sort_values(["chromosome", "position", "probe_id"], kind="stable").reset_index(drop=True)

    block_ids: list[int] = []
    block_id = -1
    prev_chrom: str | None = None
    prev_pos: int | None = None
    for row in eligible.itertuples(index=False):
        chrom = str(row.chromosome)
        pos = int(row.position)
        if prev_chrom is None or chrom != prev_chrom or prev_pos is None or pos - prev_pos > int(block_gap_bp):
            block_id += 1
        block_ids.append(block_id)
        prev_chrom = chrom
        prev_pos = pos
    eligible["block_id"] = np.asarray(block_ids, dtype=np.int32)

    block_catalog = (
        eligible.groupby("block_id", as_index=False)
        .agg(
            chromosome=("chromosome", "first"),
            block_start=("position", "min"),
            block_end=("position", "max"),
            block_probe_count=("probe_id", "count"),
            anchor_gene=("anchor_gene", _block_anchor_gene),
        )
        .sort_values("block_id", kind="stable")
        .reset_index(drop=True)
    )
    block_catalog["group_label"] = [
        (
            f"{anchor_gene}@chr{chrom}:{int(start)}-{int(end)}"
            if str(anchor_gene).strip()
            else f"chr{chrom}:{int(start)}-{int(end)}"
        )
        for chrom, start, end, anchor_gene in block_catalog[
            ["chromosome", "block_start", "block_end", "anchor_gene"]
        ].itertuples(index=False, name=None)
    ]

    annotation = eligible.merge(
        block_catalog.loc[:, ["block_id", "block_start", "block_end", "block_probe_count", "group_label"]],
        on="block_id",
        how="left",
        sort=False,
    )
    return annotation, block_catalog


def _select_features(
    ranked: pd.DataFrame,
    *,
    top_k: int,
    min_group_size: int,
    max_group_size: int,
) -> pd.DataFrame:
    if int(max_group_size) > 0:
        capped = ranked.groupby("group_label", sort=False, group_keys=False).head(int(max_group_size)).copy()
    else:
        capped = ranked.copy()

    eligible_counts = capped["group_label"].value_counts()
    eligible_groups = set(eligible_counts[eligible_counts >= int(min_group_size)].index.tolist())
    capped = capped.loc[capped["group_label"].isin(eligible_groups)].reset_index(drop=True)
    if capped.empty:
        raise RuntimeError("No local-block groups survived the min_group_size constraint.")

    grouped_frames: list[tuple[str, float, pd.DataFrame]] = []
    for group_label, sub in capped.groupby("group_label", sort=False):
        sub_reset = sub.reset_index(drop=True)
        group_score = float(sub_reset["variance"].head(min(10, sub_reset.shape[0])).sum())
        grouped_frames.append((str(group_label), group_score, sub_reset))
    grouped_frames.sort(key=lambda item: (-item[1], item[0]))

    selected_group_frames: list[pd.DataFrame] = []
    selected_group_labels: list[str] = []
    available_features = 0
    for group_label, _score, frame in grouped_frames:
        if available_features >= int(top_k):
            break
        selected_group_frames.append(frame)
        selected_group_labels.append(group_label)
        available_features += int(frame.shape[0])

    if not selected_group_frames:
        raise RuntimeError("Unable to seed any local-block groups under the current settings.")

    seeded_groups = [frame.iloc[: int(min_group_size)].copy() for frame in selected_group_frames]
    selected = pd.concat(seeded_groups, ignore_index=True)
    remaining_slots = int(top_k) - int(selected.shape[0])

    extras_parts: list[pd.DataFrame] = []
    for group_label, _score, frame in grouped_frames:
        if group_label in selected_group_labels and frame.shape[0] > int(min_group_size):
            extras_parts.append(frame.iloc[int(min_group_size) :].copy())
    if extras_parts and remaining_slots > 0:
        extras = pd.concat(extras_parts, ignore_index=True)
        extras = extras.sort_values(["variance", "probe_id"], ascending=[False, True], kind="stable").head(remaining_slots)
        selected = pd.concat([selected, extras], ignore_index=True)

    if int(selected.shape[0]) != int(top_k):
        raise RuntimeError(
            f"Local-block selection produced {selected.shape[0]} features, expected exactly {top_k}. "
            "The current constraints are too strict for the requested top_k."
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
    return selected


def _correlation_summary(X: np.ndarray, group_ids: np.ndarray) -> dict[str, dict[str, float]]:
    corr = np.corrcoef(np.asarray(X, dtype=float), rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    tri_i, tri_j = np.triu_indices(corr.shape[0], k=1)
    pair_corr = corr[tri_i, tri_j]
    within_mask = group_ids[tri_i] == group_ids[tri_j]
    between_mask = ~within_mask

    def summarize(values: np.ndarray) -> dict[str, float]:
        arr = np.asarray(values, dtype=float)
        q05, q25, q50, q75, q95 = np.quantile(arr, [0.05, 0.25, 0.50, 0.75, 0.95])
        return {
            "n_pairs": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "min": float(arr.min()),
            "q05": float(q05),
            "q25": float(q25),
            "median": float(q50),
            "q75": float(q75),
            "q95": float(q95),
            "max": float(arr.max()),
        }

    within = pair_corr[within_mask]
    between = pair_corr[between_mask]
    return {
        "within_group_corr": summarize(within),
        "between_group_corr": summarize(between),
        "within_group_abs_corr": summarize(np.abs(within)),
        "between_group_abs_corr": summarize(np.abs(between)),
    }


def _prepare_variant(
    *,
    variant: str,
    top_k: int | None,
    min_group_size: int | None,
    max_group_size: int | None,
    block_gap_bp: int | None,
    chunk_size: int | None,
) -> dict[str, Any]:
    settings = _resolve_settings(
        variant=variant,
        top_k=top_k,
        min_group_size=min_group_size,
        max_group_size=max_group_size,
        block_gap_bp=block_gap_bp,
        chunk_size=chunk_size,
    )
    paths = base._download_required_files()
    base._ensure_dir(PROCESSED_DIR)
    base._ensure_dir(ANALYSIS_BUNDLE_DIR)

    annotation, block_catalog = _load_local_block_annotation(paths.annotation, block_gap_bp=int(settings.block_gap_bp))
    block_catalog_path = ANALYSIS_BUNDLE_DIR / f"{variant}_block_catalog.tsv.gz"
    block_catalog.to_csv(block_catalog_path, sep="\t", index=False, compression="gzip")

    sample_metadata = base._fetch_series_metadata(paths.series_header_excerpt)
    sample_metadata_path = ANALYSIS_BUNDLE_DIR / "sample_metadata.tsv"
    sample_metadata.to_csv(sample_metadata_path, sep="\t", index=False)

    sample_columns = base._read_beta_header(paths.beta)
    if list(sample_metadata["sample_name"]) != sample_columns:
        raise RuntimeError("Sample order from series-matrix metadata does not match beta-matrix columns.")

    ranked_base = base._compute_probe_variances(
        paths.beta,
        annotation=annotation.loc[:, ["probe_id", "chromosome", "group_label"]],
        sample_columns=sample_columns,
        chunk_size=int(settings.chunk_size),
    )
    ranked = ranked_base.merge(
        annotation.loc[
            :,
            ["probe_id", "position", "anchor_gene", "block_id", "block_start", "block_end", "block_probe_count", "group_label"],
        ].drop_duplicates("probe_id"),
        on=["probe_id", "group_label"],
        how="left",
        sort=False,
    )
    ranked_path = ANALYSIS_BUNDLE_DIR / f"{variant}_ranked_candidates.tsv.gz"
    ranked.to_csv(ranked_path, sep="\t", index=False, compression="gzip")

    selected = _select_features(
        ranked,
        top_k=int(settings.top_k),
        min_group_size=int(settings.min_group_size),
        max_group_size=int(settings.max_group_size),
    )
    selected_path = ANALYSIS_BUNDLE_DIR / f"{variant}_selected_features.tsv"
    selected.to_csv(selected_path, sep="\t", index=False)

    selected_probe_ids = selected["probe_id"].astype(str).tolist()
    X = base._extract_selected_matrix(
        paths.beta,
        selected_probe_ids=selected_probe_ids,
        sample_columns=sample_columns,
        chunk_size=int(settings.chunk_size),
    )
    y = sample_metadata["age_years"].to_numpy(dtype=np.float32)

    runner_dir = PROCESSED_DIR / f"runner_ready_{variant}"
    base._ensure_dir(runner_dir)
    np.save(runner_dir / "X.npy", X)
    np.save(runner_dir / "y.npy", y)
    base._write_lines(runner_dir / "feature_names.txt", selected_probe_ids)

    group_order = (
        selected.loc[:, ["group_id", "group_label"]]
        .drop_duplicates()
        .sort_values("group_id", kind="stable")["group_label"]
        .astype(str)
        .tolist()
    )
    base._write_lines(runner_dir / "group_labels.txt", group_order)

    group_map = {probe_id: int(group_id) for probe_id, group_id in selected[["probe_id", "group_id"]].itertuples(index=False)}
    (runner_dir / "group_map.json").write_text(json.dumps(group_map, indent=2, ensure_ascii=False), encoding="utf-8")

    feature_group_sizes = selected["group_label"].value_counts()
    group_catalog = (
        selected.groupby(["group_id", "group_label"], as_index=False)
        .agg(
            feature_count=("probe_id", "count"),
            block_probe_count=("block_probe_count", "first"),
            chromosome=("chromosome", "first"),
            block_start=("block_start", "first"),
            block_end=("block_end", "first"),
            max_variance=("variance", "max"),
        )
        .sort_values("group_id", kind="stable")
    )
    group_catalog_path = ANALYSIS_BUNDLE_DIR / f"{variant}_group_catalog.tsv"
    group_catalog.to_csv(group_catalog_path, sep="\t", index=False)

    corr_summary = _correlation_summary(X, selected["group_id"].to_numpy(dtype=int))
    corr_summary_path = ANALYSIS_BUNDLE_DIR / f"{variant}_correlation_summary.json"
    corr_summary_path.write_text(json.dumps(corr_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_payload = {
        "dataset_id": "gse40279_methylation_age",
        "variant": variant,
        "generated_at": base._now_iso(),
        "grouping_mode": "local_methylation_block",
        "grouping_rule": {
            "block_gap_bp": int(settings.block_gap_bp),
            "min_group_size": int(settings.min_group_size),
            "max_group_size": None if int(settings.max_group_size) <= 0 else int(settings.max_group_size),
            "label_rule": "Use autosomal CpGs sorted by genomic coordinate; start a new block when the next probe is farther than block_gap_bp; label each block as anchor_gene@chr:start-end when a gene label exists, otherwise chr:start-end.",
        },
        "selection_rule": "Global unsupervised variance ranking on autosomal probes with genomic coordinates; rank blocks by the sum of their top 10 probe variances, seed each selected block with min_group_size probes, then fill remaining slots by variance within those selected blocks.",
        "sample_count": int(X.shape[0]),
        "feature_count": int(X.shape[1]),
        "group_count": int(len(group_order)),
        "group_size_summary": {
            "min": int(feature_group_sizes.min()),
            "median": float(feature_group_sizes.median()),
            "max": int(feature_group_sizes.max()),
        },
        "age_summary": base._age_summary(y),
        "correlation_summary": corr_summary,
        "files": {
            "runner_dir": str(runner_dir.resolve()),
            "block_catalog": str(block_catalog_path.resolve()),
            "ranked_candidates": str(ranked_path.resolve()),
            "selected_features": str(selected_path.resolve()),
            "group_catalog": str(group_catalog_path.resolve()),
            "correlation_summary": str(corr_summary_path.resolve()),
        },
    }
    summary_path = ANALYSIS_BUNDLE_DIR / f"{variant}_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "variant": variant,
        "runner_dir": str(runner_dir.resolve()),
        "sample_count": int(X.shape[0]),
        "feature_count": int(X.shape[1]),
        "group_count": int(len(group_order)),
        "group_sizes_head": [int(v) for v in feature_group_sizes.sort_values(ascending=False).head(10).tolist()],
        "correlation_summary": corr_summary,
        "summary_path": str(summary_path.resolve()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare an exploratory GSE40279 local-methylation-block grouped variant.")
    parser.add_argument("--variant", default="micro_local_block_gap1000", help="Output variant name.")
    parser.add_argument("--top-k", type=int, default=None, help="Number of features to keep.")
    parser.add_argument("--min-group-size", type=int, default=None, help="Drop blocks smaller than this size.")
    parser.add_argument("--max-group-size", type=int, default=None, help="Optional cap per block; use 0 or negative for no cap.")
    parser.add_argument("--block-gap-bp", type=int, default=None, help="Maximum genomic gap between adjacent probes in the same block.")
    parser.add_argument("--chunk-size", type=int, default=None, help="Chunk size for beta-matrix streaming.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = _prepare_variant(
        variant=str(args.variant),
        top_k=None if args.top_k is None else int(args.top_k),
        min_group_size=None if args.min_group_size is None else int(args.min_group_size),
        max_group_size=None if args.max_group_size is None else int(args.max_group_size),
        block_gap_bp=None if args.block_gap_bp is None else int(args.block_gap_bp),
        chunk_size=None if args.chunk_size is None else int(args.chunk_size),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
