from __future__ import annotations

import os
import sys
from pathlib import Path


def can_use_process_pool() -> tuple[bool, str]:
    """Best-effort guard against Windows spawn failures in interactive entrypoints."""
    if os.name != "nt":
        return True, ""
    allow_windows_process_pool = (
        str(os.environ.get("SIM_ALLOW_WINDOWS_PROCESS_POOL", "")).strip().lower()
        in {"1", "true", "yes", "on"}
    )
    if not allow_windows_process_pool:
        return False, "disabled by default on Windows; set SIM_ALLOW_WINDOWS_PROCESS_POOL=1 to enable"
    main_mod = sys.modules.get("__main__")
    main_file = str(getattr(main_mod, "__file__", "") or "").strip()
    argv0 = str(sys.argv[0]).strip() if sys.argv else ""
    if argv0 in {"-", "-c"}:
        return False, f"sys.argv[0]={argv0!r} is not spawn-safe on Windows"
    if not main_file:
        return False, "missing __main__.__file__ in interactive context"
    main_file_l = main_file.lower()
    if "<stdin>" in main_file_l or main_file_l == "-c":
        return False, f"__main__.__file__={main_file!r} is not spawn-safe on Windows"
    if not Path(main_file).exists():
        return False, f"__main__.__file__={main_file!r} is not a real file path"
    return True, ""
