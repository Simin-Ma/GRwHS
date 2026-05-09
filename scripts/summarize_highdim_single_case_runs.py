from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize single-case high-dimensional benchmark run json files.")
    parser.add_argument("--indir", default="tmp/highdim_single_case_runs")
    parser.add_argument("--outfile", default="tmp/highdim_single_case_runs_summary.json")
    args = parser.parse_args()

    indir = Path(args.indir)
    rows = []
    for path in sorted(indir.glob("*.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue

    summary = {
        "indir": str(indir),
        "n_rows": len(rows),
        "all_converged": all(bool(r.get("converged")) for r in rows) if rows else False,
        "max_runtime_seconds": max(float(r.get("runtime_seconds") or 0.0) for r in rows) if rows else None,
        "max_wall_seconds": max(float(r.get("wall_seconds") or 0.0) for r in rows) if rows else None,
        "rows": rows,
    }
    out = Path(args.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
