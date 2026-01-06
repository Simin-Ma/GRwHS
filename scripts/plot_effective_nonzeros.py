"""Plot MeanEffectiveNonzeros vs SNR for selected variations."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MeanEffectiveNonzeros across SNR levels.")
    parser.add_argument(
        "--sweep-csv",
        required=True,
        type=Path,
        help="Path to sweep comparison CSV (e.g., outputs/sweeps/sim_s3/sweep_comparison_*.csv).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination PNG (directories will be created automatically).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Mean Effective Nonzeros vs SNR",
        help="Figure title.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["grrhs", "rhs"],
        help="Variation suffixes to plot (matched against variation names).",
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Optional y-axis limits.",
    )
    return parser.parse_args()


def _variation_label(var: str) -> tuple[float | None, str | None]:
    """
    Parse entries like 'snr0p5_grrhs' -> (0.5, 'grrhs').
    Returns (None, None) if parsing fails.
    """
    parts = var.split("_", 1)
    if len(parts) != 2 or not parts[0].startswith("snr"):
        return None, None
    snr_token = parts[0][3:]
    snr_str = snr_token.replace("p", ".")
    try:
        snr = float(snr_str)
    except ValueError:
        return None, None
    return snr, parts[1]


def _load_data(csv_path: Path, labels: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    requested = {label.lower() for label in labels}
    records = []
    for _, row in df.iterrows():
        variation = str(row["variation"])
        snr, label = _variation_label(variation)
        if snr is None or label is None or label.lower() not in requested:
            continue
        records.append(
            {
                "snr": snr,
                "label": label.lower(),
                "mean_effective_nonzeros": row["MeanEffectiveNonzeros"],
            }
        )
    if not records:
        raise RuntimeError(f"No matching variations found in {csv_path}")
    return pd.DataFrame.from_records(records)


def _format_label(label: str) -> str:
    mapping: Dict[str, str] = {
        "grrhs": "GRRHS",
        "rhs": "RHS",
    }
    return mapping.get(label.lower(), label)


def _plot(df: pd.DataFrame, output: Path, title: str, ylim: tuple[float, float] | None) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    styles = {
        "grrhs": {"color": "#1f77b4", "marker": "o"},
        "rhs": {"color": "#7f7f7f", "marker": "^"},
    }
    for label, group in df.groupby("label"):
        style = styles.get(label, {})
        group_sorted = group.sort_values("snr")
        ax.plot(
            group_sorted["snr"],
            group_sorted["mean_effective_nonzeros"],
            label=_format_label(label),
            marker=style.get("marker", "o"),
            color=style.get("color", None),
            linewidth=2,
        )

    ax.set_xlabel("SNR")
    ax.set_ylabel("MeanEffectiveNonzeros")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
    ax.legend()
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    df = _load_data(args.sweep_csv, args.labels)
    ylim = tuple(args.ylim) if args.ylim else None
    _plot(df, args.output, args.title, ylim)
    print(f"[OK] Figure written to {args.output}")


if __name__ == "__main__":
    main()
