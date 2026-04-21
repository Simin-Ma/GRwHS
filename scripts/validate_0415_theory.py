from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation_project.src.theory_0415_validation import (  # noqa: E402
    ValidationConfig,
    run_0415_theory_validation,
    validation_summary,
)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate key 0415 GR-RHS theory statements by numerical checks.")
    p.add_argument("--mode", choices=["quick", "full"], default="quick", help="Validation intensity.")
    p.add_argument("--seed", type=int, default=20260415, help="Random seed.")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "theory_0415_validation_report.json",
        help="Output JSON report path.",
    )
    return p


def main() -> int:
    args = _parser().parse_args()
    cfg = ValidationConfig.for_mode(args.mode, seed=int(args.seed))
    results = run_0415_theory_validation(cfg)
    summary = validation_summary(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"0415 validation mode={cfg.mode} seed={cfg.seed}")
    print(f"Passed {summary['passed']}/{summary['total']} checks")
    for item in summary["results"]:
        mark = "PASS" if item["passed"] else "FAIL"
        print(f"[{mark}] {item['name']} :: {item['details']}")
    print(f"Report: {args.output}")
    return 0 if bool(summary["all_passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

