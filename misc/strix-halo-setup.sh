#!/bin/bash
# ds4 Strix Halo (gfx1151) setup for Ubuntu 24.04.
#
# Applies the persistent system configuration the ds4 ROCm backend needs:
#   1. GRUB kernel parameters (RAM-sized GTT aperture + CWSR off)
#   2. modprobe.d amdgpu/ttm tuning
#   3. udev GPU access rules (see misc/99-amd-kfd.rules)
#   4. render/video group membership
#   5. tuned 'accelerator-performance' power profile
#
# It does NOT change BIOS settings or install ROCm; do those separately.
# After boot-parameter changes, reboot before running ds4.
#
# Usage:
#   sudo -u "$USER" bash misc/strix-halo-setup.sh   # must NOT run as root
#   DS4_SETUP_OS=24.04 bash misc/strix-halo-setup.sh # explicit (still non-root)

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log()  { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[!!]${NC} $1"; }
err()  { echo -e "${RED}[ERR]${NC} $1"; }
info() { echo -e "${BLUE}[..]${NC} $1"; }

# --- Ubuntu 24.04 requirement -------------------------------------------------
# The ds4 Strix Halo backend assumes the kernel/KFD fixes shipped in kernel
# 6.18.4+. Ubuntu 24.04 with HWE kernel (currently 7.x) provides these.
# Everything in this script targets that release.
DETECTED_OS="$(. /etc/os-release 2>/dev/null; echo "${VERSION_ID:-unknown}")"
REQUIRED_OS="${DS4_SETUP_OS:-24.04}"
if [ "$DETECTED_OS" != "$REQUIRED_OS" ]; then
    err "This setup requires Ubuntu ${REQUIRED_OS} with HWE kernel (6.18.4+ with KFD fixes). Detected: ${DETECTED_OS}."
    err "On other distros use Linux 6.18.4+ with the KFD fixes; this script only supports Ubuntu ${REQUIRED_OS}."
    if [ "${DS4_SETUP_ALLOW_OS:-0}" = "1" ]; then
        warn "DS4_SETUP_ALLOW_OS=1 set; continuing anyway (unsupported)."
    else
        exit 1
    fi
fi

# --- preflight ----------------------------------------------------------------
if [ "$(id -u)" -eq 0 ]; then
    err "Do not run this script as root. Run as your normal user; it uses sudo when needed."
    exit 1
fi
if ! grep -qi "amd" /proc/cpuinfo 2>/dev/null; then
    warn "This does not appear to be an AMD Strix Halo system. Continuing anyway..."
fi
TOTAL_RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
if [ "$TOTAL_RAM_GB" -lt 16 ]; then
    err "Only ${TOTAL_RAM_GB}GB RAM visible. Set the BIOS UMA Frame Buffer to 512MB first."
    exit 1
fi

REBOOT_REQUIRED=0
SESSION_REFRESH_REQUIRED=0

# --- 1. GRUB kernel parameters (RAM-sized) ------------------------------------
# Size the GTT aperture to ~90% of visible RAM so large models fit. For a 31 GiB
# box this is ~28 GiB; for a 124 GiB box ~111 GiB. The doc's reference values
# (gttsize=126976, pages_limit=32505856) are the 128 GB equivalent.
RAM_MIB=$(( TOTAL_RAM_GB * 1024 ))
GTTSIZE_MIB=$(( RAM_MIB * 90 / 100 ))            # amdgpu.gttsize is in MiB
PAGES_LIMIT=$(( GTTSIZE_MIB * 1024 * 1024 / 4096 ))  # ttm.pages_limit is in 4 KiB pages
PAGE_POOL=$PAGES_LIMIT

GRUB_FILE="/etc/default/grub"
NEEDED_PARAMS="amdgpu.gttsize=${GTTSIZE_MIB} ttm.pages_limit=${PAGES_LIMIT} ttm.page_pool_size=${PAGE_POOL} amdgpu.cwsr_enable=0"
MISSING_PARAMS=""
CURRENT_CMDLINE=$(grep "^GRUB_CMDLINE_LINUX_DEFAULT" "$GRUB_FILE" 2>/dev/null || echo "")
for param in $NEEDED_PARAMS; do
    key=$(echo "$param" | cut -d= -f1)
    if ! echo "$CURRENT_CMDLINE" | grep -q "$key"; then
        MISSING_PARAMS="$MISSING_PARAMS $param"
    fi
done
if [ -n "$MISSING_PARAMS" ]; then
    info "Adding kernel parameters:$MISSING_PARAMS"
    CURRENT_VALUE=$(echo "$CURRENT_CMDLINE" | sed 's/GRUB_CMDLINE_LINUX_DEFAULT="//' | sed 's/"$//')
    NEW_VALUE="$CURRENT_VALUE$MISSING_PARAMS"
    sudo sed -i "s|^GRUB_CMDLINE_LINUX_DEFAULT=.*|GRUB_CMDLINE_LINUX_DEFAULT=\"$NEW_VALUE\"|" "$GRUB_FILE"
    sudo update-grub
    log "GRUB updated. Reboot for the GTT aperture to take effect."
    REBOOT_REQUIRED=1
else
    log "Kernel parameters already configured."
fi

# --- 2. modprobe.d amdgpu/ttm tuning ------------------------------------------
MODPROBE_FILE="/etc/modprobe.d/amdgpu_strix_halo.conf"
if [ ! -f "$MODPROBE_FILE" ]; then
    info "Creating modprobe configuration..."
    sudo tee "$MODPROBE_FILE" > /dev/null << MODPROBE
options amdgpu gttsize=${GTTSIZE_MIB}
options ttm pages_limit=${PAGES_LIMIT}
options ttm page_pool_size=${PAGE_POOL}
MODPROBE
    sudo update-initramfs -u -k all 2>/dev/null || true
    log "Modprobe configuration created."
    REBOOT_REQUIRED=1
else
    log "Modprobe configuration already exists ($MODPROBE_FILE)."
fi

# --- 3. udev GPU access rules -------------------------------------------------
UDEV_FILE="/etc/udev/rules.d/99-amd-kfd.rules"
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_RULES="$SELF_DIR/99-amd-kfd.rules"
if [ ! -f "$UDEV_FILE" ]; then
    info "Installing udev rules..."
    if [ -f "$SRC_RULES" ]; then
        sudo cp "$SRC_RULES" "$UDEV_FILE"
    else
        sudo tee "$UDEV_FILE" > /dev/null << 'UDEV'
SUBSYSTEM=="kfd", GROUP="render", MODE="0666"
SUBSYSTEM=="drm", KERNEL=="card[0-9]*", GROUP="render", MODE="0666"
SUBSYSTEM=="drm", KERNEL=="renderD[0-9]*", GROUP="render", MODE="0666"
UDEV
    fi
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    log "udev rules installed."
else
    log "udev rules already present ($UDEV_FILE)."
fi

# --- 4. GPU groups ------------------------------------------------------------
if ! groups | grep -q render; then
    sudo usermod -aG render "$USER"
    sudo usermod -aG video "$USER"
    log "Added $USER to render and video groups."
    SESSION_REFRESH_REQUIRED=1
else
    log "User already in GPU groups."
fi

# --- 5. tuned accelerator-performance -----------------------------------------
if ! command -v tuned-adm &>/dev/null; then
    info "Installing tuned..."
    sudo apt install -y tuned
fi
sudo systemctl enable --now tuned 2>/dev/null || true
sudo tuned-adm profile accelerator-performance 2>/dev/null || true
if tuned-adm active 2>/dev/null | grep -q "accelerator-performance"; then
    log "tuned: accelerator-performance active."
else
    warn "tuned profile may not be set; run: sudo tuned-adm profile accelerator-performance"
fi

# --- amd-ttm (preferred runtime TTM control) ----------------------------------
if [ -x /usr/bin/amd-ttm ]; then
    log "amd-ttm present."
else
    info "amd-ttm not found. Installing amd-ttm for runtime TTM/GTT control..."
    # amd-ttm may be available as a package or standalone binary. Try apt first.
    if sudo apt-cache show amd-ttm >/dev/null 2>&1; then
        sudo apt-get install -y -qq amd-ttm 2>/dev/null && log "amd-ttm installed." || \
            warn "amd-ttm package install failed. Install manually for runtime TTM control."
    else
        # Try downloading the standalone binary from the Strix Halo toolboxes repo.
        AMD_TTM_URL="https://raw.githubusercontent.com/kyuz0/amd-strix-halo-toolboxes/main/amd-ttm"
        if curl -fsSL "$AMD_TTM_URL" -o /tmp/amd-ttm 2>/dev/null && [ -s /tmp/amd-ttm ]; then
            sudo install -m 755 /tmp/amd-ttm /usr/bin/amd-ttm && \
                log "amd-ttm installed from GitHub." || \
                warn "amd-ttm install failed. Install manually for runtime TTM control."
            rm -f /tmp/amd-ttm
        else
            warn "amd-ttm not available. GRUB kernel params will be the only TTM control."
        fi
    fi
fi

# --- ROCm toolchain presence --------------------------------------------------
KVER=$(uname -r)
KMAJOR=$(echo "$KVER" | cut -d. -f1)
KMINOR=$(echo "$KVER" | cut -d. -f2)
if [ "$KMAJOR" -lt 6 ] || { [ "$KMAJOR" -eq 6 ] && [ "$KMINOR" -lt 18 ]; }; then
    warn "Kernel $KVER is too old. The Strix Halo KFD fixes require kernel 6.18.4+."
    warn "Install Ubuntu 24.04 HWE kernel or use mainline kernel PPA."
fi
if ! command -v hipcc &>/dev/null; then
    info "hipcc not found. Installing ROCm 7.2.x toolchain..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
      hipcc rocminfo rocm-smi libamdhip64-dev \
      libhipblas-dev libhipblaslt-dev librocblas-dev \
      librocwmma-dev libhipcub-dev 2>/dev/null || \
      warn "ROCm install failed. Install manually per STRIXHALO.md."
    if ! command -v hipcc &>/dev/null; then
        warn "hipcc still not found after install attempt. See STRIXHALO.md."
    else
        log "ROCm toolchain installed: $(hipcc --version 2>&1 | head -1)"
    fi
else
    log "ROCm/hipcc present: $(hipcc --version 2>&1 | head -1)"
fi
# Install rocWMMA internal headers if missing
if [ ! -d /usr/local/include/rocwmma/internal ]; then
    info "Installing rocWMMA internal headers..."
    git clone --depth 1 --branch rocm-7.2.3 https://github.com/ROCm/rocWMMA.git /tmp/rocWMMA 2>/dev/null && \
        sudo cp -a /tmp/rocWMMA/library/include/rocwmma /usr/local/include/ && \
        log "rocWMMA internal headers installed." || \
        warn "rocWMMA header install failed. Build may fail without internal headers."
    rm -rf /tmp/rocWMMA 2>/dev/null
fi

echo ""
echo "============================================="
echo "  ds4 Strix Halo setup complete (Ubuntu ${REQUIRED_OS})"
echo "============================================="
echo ""
echo "  RAM:        ${TOTAL_RAM_GB} GiB"
echo "  gttsize:    ${GTTSIZE_MIB} MiB (~90% of RAM)"
echo "  tuned:      $(tuned-adm active 2>/dev/null | grep -o 'accelerator-performance' || echo 'not active')"
echo "  ROCm/hipcc: $(command -v hipcc >/dev/null && echo present || echo MISSING)"
echo ""
echo "  Next: build and verify"
echo "    make strix-halo -j\"\$(nproc)\""
echo "    make rocm-doctor"
echo ""
if [ "$REBOOT_REQUIRED" -eq 1 ]; then
    warn "REBOOT REQUIRED for boot-time changes (GTT aperture) to take effect."
    warn "Then: sudo reboot"
elif [ "$SESSION_REFRESH_REQUIRED" -eq 1 ]; then
    warn "Log out and back in so GPU group membership applies to your shell."
else
    log "No reboot needed."
fi
