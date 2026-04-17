# Root conftest.py — loaded by pytest before any test module is imported.
#
# Problem: platform._wmi_query() hangs indefinitely on this machine because the
# Windows WMI service (winmgmt) does not respond to COM queries.  pandas 3.x
# calls platform.machine() at module-level (pandas/compat/_constants.py) which
# triggers _wmi_query on Python 3.13 Windows.  The call never raises an error
# and never returns — so pandas import never completes.
#
# Fix: set platform._wmi = None *before* pandas is imported.  With _wmi == None,
# _wmi_query() immediately raises OSError, _get_machine_win32() falls through to
# the PROCESSOR_ARCHITECTURE environment variable, and pandas imports normally.

import os
import platform as _platform

# Disable the hanging WMI module for the duration of this process.
_platform._wmi = None  # type: ignore[attr-defined]

# Ensure the fallback env-var path returns a sensible value.
os.environ.setdefault("PROCESSOR_ARCHITECTURE", "AMD64")
os.environ.setdefault("PROCESSOR_ARCHITEW6432", "AMD64")
