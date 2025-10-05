"""Table rendering utilities."""
from __future__ import annotations

from typing import Iterable


def to_latex_table(rows: Iterable[Iterable[str]]) -> str:
    """Render rows into a simple LaTeX tabular environment."""
    lines = ["\\begin{tabular}{l}"]
    for row in rows:
        lines.append(" \\ ".join(row) + " \\\")
    lines.append("\\end{tabular}")
    return "\n".join(lines)
