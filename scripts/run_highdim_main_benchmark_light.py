from __future__ import annotations

import sys

from Simulation_highdimension.src.run_highdimension import _cli


def main() -> int:
    sys.argv = [
        sys.argv[0],
        "run-benchmark",
        "--repeats",
        "1",
        "--n-jobs",
        "1",
        "--method-jobs",
        "1",
        "--methods",
        "GR_RHS",
        "RHS",
        "GIGG_MMLE",
        "LASSO_CV",
    ]
    return _cli()


if __name__ == "__main__":
    raise SystemExit(main())
