from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_exp3_benchmark(df: Any, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        "Legacy benchmark plotting has been retired.\nUse current GA-V2 summaries instead.",
        ha="center",
        va="center",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "legacy_benchmark_retired.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
