from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


def timed_cmd(args: list[str], timeout: float = 20.0) -> dict[str, object]:
    t0 = time.perf_counter()
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        dt = time.perf_counter() - t0
        return {
            "ok": p.returncode == 0,
            "returncode": p.returncode,
            "seconds": dt,
            "stdout": p.stdout.strip(),
            "stderr": p.stderr.strip(),
            "timeout": False,
        }
    except subprocess.TimeoutExpired as e:
        dt = time.perf_counter() - t0
        return {
            "ok": False,
            "returncode": None,
            "seconds": dt,
            "stdout": (e.stdout or "").strip() if isinstance(e.stdout, str) else "",
            "stderr": (e.stderr or "").strip() if isinstance(e.stderr, str) else "",
            "timeout": True,
        }


def main() -> None:
    py = sys.executable
    checks = {
        "python_version": timed_cmd([py, "-c", "import sys; print(sys.version)"], timeout=10),
        "import_numpy": timed_cmd([py, "-c", "import numpy as np; print(np.__version__)"], timeout=20),
        "import_pandas_libs": timed_cmd([py, "-c", "import pandas._libs as p; print('ok')"], timeout=40),
        "import_pandas": timed_cmd([py, "-c", "import pandas as pd; print(pd.__version__)"], timeout=60),
    }
    result = {
        "python": py,
        "checks": checks,
        "summary": {
            "numpy_ok": checks["import_numpy"]["ok"],
            "pandas_libs_ok": checks["import_pandas_libs"]["ok"],
            "pandas_ok": checks["import_pandas"]["ok"],
            "pandas_seconds": checks["import_pandas"]["seconds"],
        },
    }
    out = Path("simulation_project") / "logs" / "pandas_import_diagnosis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

