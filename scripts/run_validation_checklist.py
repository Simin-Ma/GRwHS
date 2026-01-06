"""Execute the GRRHS validation checklist."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from grrhs.diagnostics.validation import ValidationChecklistRunner


def _to_serializable(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GRRHS validation checklist scenarios.")
    parser.add_argument("--minimum", action="store_true", help="Run only the minimum publishable subset.")
    parser.add_argument("--fast", action="store_true", help="Use faster (fewer-iteration) settings.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write a JSON summary. When omitted, only prints to stdout.",
    )
    args = parser.parse_args()

    runner = ValidationChecklistRunner(fast=args.fast)
    outcomes = runner.run_all(minimum_only=args.minimum)

    summary = {
        "minimum_only": bool(args.minimum),
        "fast_mode": bool(args.fast),
        "scenarios": [
            {
                "key": oc.key,
                "label": oc.label,
                "status": oc.status,
                "metrics": oc.metrics,
                "expectations": oc.expectations,
                "notes": oc.notes,
            }
            for oc in outcomes
        ],
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, default=_to_serializable), encoding="utf-8")

    for oc in outcomes:
        print(f"[{oc.status.upper():5}] {oc.key:6} - {oc.label}")
        if oc.notes:
            print(f"    notes: {', '.join(oc.notes)}")


if __name__ == "__main__":
    main()
