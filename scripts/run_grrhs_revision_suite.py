from __future__ import annotations

import argparse
import json
from pathlib import Path

from grrhs.simulations.revision_suite import DEFAULT_METHODS, run_revision_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GR-RHS revision simulation suite.")
    parser.add_argument("--output-dir", type=str, default="outputs/revision_suite_quick")
    parser.add_argument("--sections", nargs="*", default=["theory", "benchmark", "heterogeneity", "inferential", "logistic", "ablation", "hyper"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=20260416)
    parser.add_argument("--methods", nargs="*", default=list(DEFAULT_METHODS))
    args = parser.parse_args()

    summary = run_revision_suite(
        output_dir=Path(args.output_dir),
        sections=args.sections,
        quick=bool(args.quick),
        seed=int(args.seed),
        methods=args.methods,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
