#!/usr/bin/env python3
"""Retired one-off patch helper.

The Strix Halo q4 guard is now implemented directly in ds4_hip.cpp:

* HIP runtime/build version reporting for the ROCm 7.2.3 baseline.
* Runtime SOMA capability detection with a synchronous allocation fallback.
* RDNA3.5 TTM/GTT shared-memory limit reporting.
* Safe model mmap registration via hipHostRegisterMapped.

This file is kept only so older notes that mention q4_guard.py do not fail with
an import or execution error. It intentionally does not rewrite source files.
"""

print("q4_guard.py is retired; Strix Halo guards live in ds4_hip.cpp.")
