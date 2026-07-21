#!/bin/sh
# rocm-doctor.sh: one-screen triage for Strix Halo ROCm setups.
#
# Runs in order: device permissions, gfx1151 capability, TTM/GTT limit,
# amd-ttm availability, optional BIOS VRAM warning, the runtime profile
# (via the smoke test), and the quick kernel bench. Exits non-zero on any
# HARD failure (missing device, wrong arch, smoke or bench failure).
#
# Soft notes (low TTM limit, missing amd-smi) are printed but do not fail.

set -e

cd "$(dirname "$0")/.."

fail=0

note() { printf "doctor: %s\n" "$1"; }
warn() { printf "doctor: WARN: %s\n" "$1"; }
hard() { printf "doctor: FAIL: %s\n" "$1"; fail=1; }

note "=== 0. OS requirement (Ubuntu 24.04 HWE) ==="
if [ -r /etc/os-release ]; then
    . /etc/os-release
    if [ "${ID:-}" = "ubuntu" ] && [ "${VERSION_ID:-}" = "24.04" ]; then
        note "Ubuntu 24.04 LTS"
    else
        hard "expected Ubuntu 24.04; detected ${PRETTY_NAME:-$VERSION_ID}. The ds4 Strix Halo backend assumes Ubuntu 24.04 with HWE kernel (6.18.4+ with KFD fixes)."
    fi
else
    warn "/etc/os-release not readable; skipping OS check"
fi

note "=== 1. device permissions ==="
if [ -e /dev/kfd ]; then
    ls -l /dev/kfd 2>/dev/null | head -1
else
    hard "/dev/kfd not present; KFD driver not loaded?"
fi
if ls /dev/dri/render* >/dev/null 2>&1; then
    ls -l /dev/dri/render* | head -3
else
    hard "no /dev/dri/render* nodes found"
fi
case " $(groups 2>/dev/null) " in
    *" render "*) note "user is in the 'render' group";;
    *) warn "user is NOT in the 'render' group (sudo usermod -aG render \$USER)";;
esac

note "=== 2. gfx1151 capability ==="
if command -v rocminfo >/dev/null 2>&1; then
    if rocminfo 2>/dev/null | grep -A1 "Name:.*gfx1151" >/dev/null; then
        note "rocminfo reports gfx1151"
    else
        hard "rocminfo does not report gfx1151 (kernel/ROCm mismatch?)"
    fi
else
    warn "rocminfo not installed; skipping gfx1151 check"
fi

note "=== 3. TTM/GTT limit ==="
if [ -r /sys/module/ttm/parameters/pages_limit ]; then
    pages=$(cat /sys/module/ttm/parameters/pages_limit)
    gib=$(( pages * 4096 / 1073741824 ))
    note "live pages_limit=$pages ($gib GiB)"
else
    warn "could not read ttm pages_limit (module not loaded?)"
fi
if [ -r /proc/cmdline ]; then
    if grep -q "amdgpu.gttsize=" /proc/cmdline 2>/dev/null; then
        note "amdgpu.gttsize set: $(grep -oE 'amdgpu.gttsize=[0-9]+' /proc/cmdline | head -1)"
    else
        warn "amdgpu.gttsize is NOT set in /proc/cmdline (driver default)"
    fi
fi

note "=== 4. amd-ttm availability + power/tuning ==="
if [ -x /usr/bin/amd-ttm ]; then
    note "/usr/bin/amd-ttm is present"
else
    warn "/usr/bin/amd-ttm not found; DS4_ROCM_TTM_AUTORAISE will be a no-op"
fi
case " ${DS4_ROCM_TTM_AUTORAISE:-} " in
    *1*|*true*|*yes*) note "DS4_ROCM_TTM_AUTORAISE is set (auto-raise on OOM)";;
    *) note "DS4_ROCM_TTM_AUTORAISE not set (manual TTM tuning only)";;
esac
if [ -n "${DS4_ROCM_TTM_PAGES:-}" ]; then
    gib=$(( DS4_ROCM_TTM_PAGES * 4096 / 1073741824 ))
    note "DS4_ROCM_TTM_PAGES=$DS4_ROCM_TTM_PAGES ($gib GiB override)"
fi
if command -v tuned-adm >/dev/null 2>&1; then
    if tuned-adm active 2>/dev/null | grep -q "accelerator-performance"; then
        note "tuned: accelerator-performance active"
    else
        warn "tuned profile is not accelerator-performance (run: sudo tuned-adm profile accelerator-performance)"
    fi
else
    warn "tuned not installed; install it for the accelerator-performance power profile"
fi
if [ -f /etc/udev/rules.d/99-amd-kfd.rules ]; then
    note "udev rules 99-amd-kfd.rules present"
else
    warn "udev rules 99-amd-kfd.rules missing (copy misc/99-amd-kfd.rules); GPU access may need root"
fi
case " $(groups 2>/dev/null) " in
    *" render "*) note "user is in the 'render' group";;
    *) warn "user is NOT in the 'render' group";;
esac
if grep -q "amdgpu.cwsr_enable=0" /proc/cmdline 2>/dev/null; then
    note "amdgpu.cwsr_enable=0 set (CWSR disabled)"
else
    warn "amdgpu.cwsr_enable=0 not set; add it for the Strix Halo tuned profile"
fi

note "=== 5. BIOS VRAM hint ==="
if dmesg 2>/dev/null | grep -iE "VRAM|gttsize" | head -3; then
    note "review above lines; BIOS dedicated VRAM > 512 MB will starve the OS"
else
    note "no recent dmesg VRAM lines (boot may be stale); dmesg | grep -i vram"
fi

note "=== 6. runtime profile (smoke test) ==="
if [ -x ./tests/rocm_smoke ]; then
    if ! ./tests/rocm_smoke; then hard "rocm-smoke FAILED"; fi
else
    warn "./tests/rocm_smoke not built yet (run: make rocm-smoke)"
fi

note "=== 7. kernel execution bench ==="
if [ -x ./tests/rocm_bench_quick ]; then
    if ! ./tests/rocm_bench_quick; then hard "rocm-bench-quick FAILED"; fi
else
    warn "./tests/rocm_bench_quick not built yet (run: make rocm-bench-quick)"
fi

note "=== 8. model fit (optional) ==="
if [ -n "${DS4_TEST_MODEL:-}" ]; then
    if [ -x ./tests/rocm_model_fit ]; then
        if ! ./tests/rocm_model_fit; then hard "DS4_TEST_MODEL does not fit TTM/GTT limit"; fi
    else
        warn "./tests/rocm_model_fit not built yet (run: make rocm-model-fit)"
    fi
else
    note "DS4_TEST_MODEL not set; skipping fit gate (set it to validate a model)"
fi

echo
if [ "$fail" -eq 0 ]; then
    note "doctor: OK"
else
    note "doctor: FAILED (see above)"
fi
exit "$fail"
